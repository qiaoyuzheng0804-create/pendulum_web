"""
单摆 (Simple Pendulum) 处理器
Adapted from danbai_ultimate_version.py — GUI calibration replaced with programmatic API.
"""

import os as _os
_os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import os
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import cv2
from scipy.signal import savgol_filter, find_peaks
from ultralytics import YOLO


@dataclass
class DanbaiConfig:
    conf_thres: float = 0.25
    iou_thres: float = 0.45
    imgsz: int = 1280
    max_det: int = 5
    device: str = "cpu"
    use_tracking_like_selection: bool = True
    skip_initial_seconds: float = 3.0
    savgol_window: int = 21
    savgol_polyorder: int = 3
    max_jump_after_filter_deg: float = 0.45
    zero_cross_abs_angle_gate_deg: float = 1.0
    zero_cross_min_index: int = 40
    target_cycles: int = 400
    min_cycles: int = 398
    fallback_to_available_cycles: bool = True
    enable_early_stop_by_cycles: bool = True
    online_smooth_window: int = 7
    online_zero_cross_abs_gate_deg: float = 1.2
    early_stop_extra_frames: int = 8
    angle_step_deg: float = 0.15
    min_frame_gap_between_samples: int = 1
    resample_use_filtered_angle: bool = True
    enable_global_zero_offset_correction: bool = True
    enable_local_zero_offset_correction: bool = True
    zero_offset_percentile_low: float = 5.0
    zero_offset_percentile_high: float = 95.0
    force_zero_point_at_start: bool = True
    enable_envelope_extraction: bool = True
    peak_min_distance: int = 8
    peak_prominence: float = 0.08
    envelope_savgol_window: int = 11
    envelope_savgol_polyorder: int = 2
    csv_float_format: str = "%.8f"
    save_angle_time_plot: bool = True
    plot_dpi: int = 150
    plot_figsize: Tuple[float, float] = (10, 6)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < eps:
        return v.copy()
    return v / n


def make_odd(n: int) -> int:
    return n if n % 2 == 1 else n + 1


def valid_savgol_window(n: int, desired_window: int, polyorder: int) -> int:
    if n <= polyorder + 1:
        return 0
    w = min(desired_window, n if n % 2 == 1 else n - 1)
    if w <= polyorder:
        w = make_odd(polyorder + 2)
    if w > n:
        w = n if n % 2 == 1 else n - 1
    if w <= polyorder:
        return 0
    return w


def moving_linear_fill_nan(arr: np.ndarray) -> np.ndarray:
    x = np.arange(len(arr))
    mask = np.isfinite(arr)
    if mask.sum() == 0:
        return arr.copy()
    out = arr.copy()
    out[~mask] = np.interp(x[~mask], x[mask], out[mask])
    return out


def moving_average_tail(arr: List[float], window: int) -> float:
    if len(arr) == 0:
        return np.nan
    w = max(1, min(window, len(arr)))
    vals = np.array(arr[-w:], dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return np.nan
    return float(np.mean(vals))


def remove_spikes_deg(angle_deg: np.ndarray, max_jump_deg: float) -> np.ndarray:
    a = angle_deg.copy()
    n = len(a)
    if n < 3:
        return a
    for i in range(1, n - 1):
        if not np.isfinite(a[i - 1]) or not np.isfinite(a[i]) or not np.isfinite(a[i + 1]):
            continue
        if abs(a[i] - a[i - 1]) > max_jump_deg and abs(a[i] - a[i + 1]) > max_jump_deg:
            a[i] = 0.5 * (a[i - 1] + a[i + 1])
    return a


def percentile_zero_offset(arr: np.ndarray, low: float, high: float) -> float:
    upper = np.nanpercentile(arr, high)
    lower = np.nanpercentile(arr, low)
    return 0.5 * (upper + lower)


def interp_zero_crossing_time(t1: float, a1: float, t2: float, a2: float) -> float:
    if abs(a2 - a1) < 1e-12:
        return t1
    return t1 - a1 * (t2 - t1) / (a2 - a1)


def is_valid_zero_cross(a1: float, a2: float, gate_deg: float) -> bool:
    crossed = (a1 == 0.0) or (a2 == 0.0) or (a1 < 0 < a2) or (a1 > 0 > a2)
    if not crossed:
        return False
    return (min(abs(a1), abs(a2)) <= gate_deg) or ((abs(a1) + abs(a2)) <= 2 * gate_deg)


def signed_angle_deg_right_positive(vertical_unit: np.ndarray, radius_vec: np.ndarray) -> float:
    r = normalize(radius_vec.astype(np.float64))
    v = normalize(vertical_unit.astype(np.float64))
    cross_z = v[0] * r[1] - v[1] * r[0]
    dot_vr = np.clip(v[0] * r[0] + v[1] * r[1], -1.0, 1.0)
    ang_std = math.degrees(math.atan2(cross_z, dot_vr))
    return -ang_std


def centers_to_angles(centers: np.ndarray, origin: np.ndarray, vertical_unit: np.ndarray) -> np.ndarray:
    angles = []
    for c in centers:
        if np.any(~np.isfinite(c)):
            angles.append(np.nan)
            continue
        radius_vec = c - origin
        if np.linalg.norm(radius_vec) < 1e-12:
            angles.append(np.nan)
            continue
        angles.append(signed_angle_deg_right_positive(vertical_unit, radius_vec))
    return np.array(angles, dtype=np.float64)


def detect_first_bottom_crossing(times, angle_deg, abs_gate_deg, min_index):
    n = len(angle_deg)
    if n < 2:
        return None
    for i in range(max(0, min_index), n - 1):
        a1, a2 = angle_deg[i], angle_deg[i + 1]
        if not np.isfinite(a1) or not np.isfinite(a2):
            continue
        crossed = (a1 == 0.0) or (a2 == 0.0) or (a1 < 0 < a2) or (a1 > 0 > a2)
        if not crossed:
            continue
        if min(abs(a1), abs(a2)) <= abs_gate_deg or abs(a1) + abs(a2) <= 2 * abs_gate_deg:
            t0 = interp_zero_crossing_time(times[i], a1, times[i + 1], a2)
            return i, t0
    idx = int(np.nanargmin(np.abs(angle_deg[min_index:]))) + min_index if n > min_index else int(np.nanargmin(np.abs(angle_deg)))
    return max(0, idx - 1), float(times[idx])


def detect_bottom_crossing_times(times, angle_deg, abs_gate_deg):
    zt = []
    n = len(angle_deg)
    for i in range(n - 1):
        a1, a2 = angle_deg[i], angle_deg[i + 1]
        if not np.isfinite(a1) or not np.isfinite(a2):
            continue
        if is_valid_zero_cross(a1, a2, abs_gate_deg):
            zt.append(interp_zero_crossing_time(times[i], a1, times[i + 1], a2))
    return zt


def choose_end_time_by_cycles(zero_times, t0, target_cycles, min_cycles, fallback_to_available):
    future_zeros = [z for z in zero_times if z > t0 + 1e-9]
    need_target = 2 * target_cycles
    need_min = 2 * min_cycles
    if len(future_zeros) >= need_target:
        return future_zeros[need_target - 1]
    if len(future_zeros) >= need_min:
        return future_zeros[need_min - 1]
    if fallback_to_available and len(future_zeros) > 0:
        return future_zeros[-1]
    return None


def angle_uniform_resample(times, angles_deg, angle_step_deg, min_frame_gap=1):
    if len(times) == 0:
        return times, angles_deg, np.array([], dtype=int)
    keep_idx = [0]
    last_angle = angles_deg[0]
    last_idx = 0
    for i in range(1, len(angles_deg)):
        if i - last_idx < min_frame_gap:
            continue
        if abs(angles_deg[i] - last_angle) >= angle_step_deg:
            keep_idx.append(i)
            last_angle = angles_deg[i]
            last_idx = i
    if keep_idx[-1] != len(angles_deg) - 1:
        keep_idx.append(len(angles_deg) - 1)
    keep_idx = np.array(keep_idx, dtype=int)
    return times[keep_idx], angles_deg[keep_idx], keep_idx


def extract_peak_envelope(times, angle_deg, peak_min_distance, peak_prominence,
                          env_savgol_window, env_savgol_polyorder):
    n = len(angle_deg)
    empty_result = {
        "peak_idx": np.array([], dtype=int),
        "peak_time": np.array([], dtype=float),
        "peak_angle": np.array([], dtype=float),
        "peak_abs_angle": np.array([], dtype=float),
        "peak_type": np.array([], dtype=object),
        "envelope": np.abs(angle_deg.copy())
    }
    if n < 5:
        return empty_result

    pos_idx, _ = find_peaks(angle_deg, distance=max(1, peak_min_distance), prominence=peak_prominence)
    neg_idx, _ = find_peaks(-angle_deg, distance=max(1, peak_min_distance), prominence=peak_prominence)
    all_idx = np.sort(np.unique(np.concatenate([pos_idx, neg_idx])))
    if len(all_idx) == 0:
        return empty_result

    peak_time = times[all_idx]
    peak_angle = angle_deg[all_idx]
    peak_abs_angle = np.abs(peak_angle)
    pos_set = set(pos_idx.tolist())
    peak_type = np.array(["pos" if idx in pos_set else "neg" for idx in all_idx], dtype=object)

    envelope = np.interp(times, peak_time, peak_abs_angle,
                         left=peak_abs_angle[0], right=peak_abs_angle[-1])
    win = valid_savgol_window(len(envelope), env_savgol_window, env_savgol_polyorder)
    if win > 0:
        envelope = savgol_filter(envelope, window_length=win, polyorder=env_savgol_polyorder, mode="interp")
    envelope = np.maximum(envelope, 0.0)

    return {
        "peak_idx": all_idx,
        "peak_time": peak_time,
        "peak_angle": peak_angle,
        "peak_abs_angle": peak_abs_angle,
        "peak_type": peak_type,
        "envelope": envelope
    }


def process_angle_series(times, raw_angles_deg, cfg):
    angles = moving_linear_fill_nan(raw_angles_deg)
    angles = remove_spikes_deg(angles, max_jump_deg=cfg.max_jump_after_filter_deg)

    win = valid_savgol_window(len(angles), cfg.savgol_window, cfg.savgol_polyorder)
    if win > 0:
        filt_angles = savgol_filter(angles, window_length=win, polyorder=cfg.savgol_polyorder, mode="interp")
    else:
        filt_angles = angles.copy()

    if cfg.enable_global_zero_offset_correction and len(filt_angles) > 0:
        zero_bias_global = percentile_zero_offset(
            filt_angles, cfg.zero_offset_percentile_low, cfg.zero_offset_percentile_high)
        filt_angles = filt_angles - zero_bias_global
        angles = angles - zero_bias_global

    zc = detect_first_bottom_crossing(
        times=times, angle_deg=filt_angles,
        abs_gate_deg=cfg.zero_cross_abs_angle_gate_deg,
        min_index=cfg.zero_cross_min_index)
    if zc is None:
        raise RuntimeError("未能检测到第一次通过最低点（angle=0）的时刻。")
    _, t0 = zc

    zero_times = detect_bottom_crossing_times(
        times=times, angle_deg=filt_angles,
        abs_gate_deg=cfg.zero_cross_abs_angle_gate_deg)
    t_end = choose_end_time_by_cycles(
        zero_times=zero_times, t0=t0,
        target_cycles=cfg.target_cycles, min_cycles=cfg.min_cycles,
        fallback_to_available=cfg.fallback_to_available_cycles)
    if t_end is None or t_end <= t0:
        t_end = times[-1]

    mask = (times >= t0) & (times <= t_end)
    times_cut = times[mask]
    raw_cut = angles[mask]
    filt_cut = filt_angles[mask]

    times_seg = times_cut - t0
    raw_seg = raw_cut.copy()
    filt_seg = filt_cut.copy()

    if cfg.force_zero_point_at_start:
        if len(times_seg) == 0 or abs(times_seg[0]) > 1e-12:
            times_seg = np.insert(times_seg, 0, 0.0)
            raw_seg = np.insert(raw_seg, 0, 0.0)
            filt_seg = np.insert(filt_seg, 0, 0.0)
        else:
            times_seg[0] = 0.0
            raw_seg[0] = 0.0
            filt_seg[0] = 0.0

    if cfg.enable_local_zero_offset_correction and len(filt_seg) > 1:
        zero_bias_local = percentile_zero_offset(
            filt_seg[1:], cfg.zero_offset_percentile_low, cfg.zero_offset_percentile_high)
        filt_seg[1:] = filt_seg[1:] - zero_bias_local
        raw_seg[1:] = raw_seg[1:] - zero_bias_local
        if cfg.force_zero_point_at_start:
            times_seg[0] = 0.0
            raw_seg[0] = 0.0
            filt_seg[0] = 0.0

    envelope = np.abs(filt_seg.copy())
    peak_idx = np.array([], dtype=int)
    peak_time = np.array([], dtype=float)
    peak_angle = np.array([], dtype=float)
    peak_abs_angle = np.array([], dtype=float)
    peak_type = np.array([], dtype=object)

    if cfg.enable_envelope_extraction and len(filt_seg) > 5:
        env_info = extract_peak_envelope(
            times=times_seg, angle_deg=filt_seg,
            peak_min_distance=cfg.peak_min_distance,
            peak_prominence=cfg.peak_prominence,
            env_savgol_window=cfg.envelope_savgol_window,
            env_savgol_polyorder=cfg.envelope_savgol_polyorder)
        envelope = env_info["envelope"]
        peak_idx = env_info["peak_idx"]
        peak_time = env_info["peak_time"]
        peak_angle = env_info["peak_angle"]
        peak_abs_angle = env_info["peak_abs_angle"]
        peak_type = env_info["peak_type"]

    resample_angles = filt_seg if cfg.resample_use_filtered_angle else raw_seg
    sample_t, _, keep_idx = angle_uniform_resample(
        times=times_seg, angles_deg=resample_angles,
        angle_step_deg=cfg.angle_step_deg,
        min_frame_gap=cfg.min_frame_gap_between_samples)

    if len(sample_t) == 0:
        raise RuntimeError("均匀采样后无有效数据。")

    sample_raw = raw_seg[keep_idx]
    sample_filt = filt_seg[keep_idx]
    sample_env = envelope[keep_idx]

    if cfg.force_zero_point_at_start:
        if abs(sample_t[0]) > 1e-12:
            sample_t = np.insert(sample_t, 0, 0.0)
            sample_raw = np.insert(sample_raw, 0, 0.0)
            sample_filt = np.insert(sample_filt, 0, 0.0)
            sample_env = np.insert(sample_env, 0, 0.0)
        else:
            sample_t[0] = 0.0
            sample_raw[0] = 0.0
            sample_filt[0] = 0.0
            sample_env[0] = 0.0

    df = pd.DataFrame({
        "time_s": sample_t,
        "angle_deg": sample_filt,
        "raw_angle_deg": sample_raw,
        "filtered_angle_deg": sample_filt,
        "peak_envelope_deg": sample_env
    })

    peaks_df = pd.DataFrame({
        "peak_time_s": peak_time,
        "peak_angle_deg": peak_angle,
        "peak_abs_angle_deg": peak_abs_angle,
        "peak_type": peak_type
    })

    return {
        "times_seg": times_seg, "raw_seg": raw_seg, "filt_seg": filt_seg,
        "envelope_seg": envelope, "peak_idx": peak_idx, "peak_time": peak_time,
        "peak_angle": peak_angle, "peak_abs_angle": peak_abs_angle,
        "peak_type": peak_type, "sample_t": sample_t, "sample_raw": sample_raw,
        "sample_filt": sample_filt, "sample_env": sample_env,
        "df": df, "peaks_df": peaks_df
    }


def plot_angle_time(result_dict, save_path, cfg, title):
    plt.figure(figsize=cfg.plot_figsize)
    plt.plot(result_dict["times_seg"], result_dict["raw_seg"], label="Raw Angle", linewidth=0.9, alpha=0.5)
    plt.plot(result_dict["times_seg"], result_dict["filt_seg"], label="Filtered Angle", linewidth=1.5)
    plt.plot(result_dict["times_seg"], result_dict["envelope_seg"], label="Peak Envelope", linewidth=1.3, linestyle="--")
    plt.plot(result_dict["times_seg"], -result_dict["envelope_seg"], linewidth=1.3, linestyle="--")
    if len(result_dict["peak_time"]) > 0:
        plt.scatter(result_dict["peak_time"], result_dict["peak_angle"], s=18, label="Detected Peaks")
    plt.scatter(result_dict["sample_t"], result_dict["sample_filt"], s=10, label="Sampled Points")
    plt.axhline(0.0, linestyle="--", linewidth=1.0)
    plt.xlabel("Time (s)")
    plt.ylabel("Angle (deg)")
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=cfg.plot_dpi, bbox_inches="tight")
    plt.close()


def process_danbai(video_path, output_dir, model_path, calibration,
                   skip_initial_seconds=3.0, slow_motion_factor=1.0,
                   target_cycles=400, model=None, progress_callback=None):
    cfg = DanbaiConfig()
    cfg.skip_initial_seconds = skip_initial_seconds
    cfg.target_cycles = target_cycles
    cfg.min_cycles = max(1, target_cycles - 2)

    try:
        import torch
        cfg.device = "0" if torch.cuda.is_available() else "cpu"
    except Exception:
        cfg.device = "cpu"

    ensure_dir(output_dir)

    center_origin = np.array(calibration["center"], dtype=np.float64)
    vertical_start = np.array(calibration["vertical_start"], dtype=np.float64)
    vertical_end = np.array(calibration["vertical_end"], dtype=np.float64)
    vertical_vec = vertical_end - vertical_start
    vertical_unit = normalize(vertical_vec)
    if np.linalg.norm(vertical_unit) < 1e-12:
        raise RuntimeError("Vertical direction vector is zero.")

    if model is None:
        model = YOLO(model_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1e-6:
        fps = 30.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    basename = os.path.splitext(os.path.basename(video_path))[0]

    times, centers, confs = [], [], []
    prev_center = None
    frame_idx = 0
    skip_frames = int(round(cfg.skip_initial_seconds / slow_motion_factor * fps))

    online_raw_angles = []
    online_smooth_angles = []
    first_cross_found = False
    post_start_cross_count = 0
    extra_frames_left = -1

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        t = frame_idx / fps * slow_motion_factor

        results = model.predict(
            source=frame, conf=cfg.conf_thres, iou=cfg.iou_thres,
            imgsz=cfg.imgsz, max_det=cfg.max_det, device=cfg.device, verbose=False)

        det_centers, det_scores = [], []
        if len(results) > 0 and results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes_xyxy = results[0].boxes.xyxy.cpu().numpy()
            scores = results[0].boxes.conf.cpu().numpy()
            for box, sc in zip(boxes_xyxy, scores):
                x1, y1, x2, y2 = box[:4]
                cx = 0.5 * (x1 + x2)
                cy = 0.5 * (y1 + y2)
                det_centers.append(np.array([cx, cy], dtype=np.float64))
                det_scores.append(float(sc))

        chosen_center = np.array([np.nan, np.nan], dtype=np.float64)
        if len(det_centers) > 0:
            if cfg.use_tracking_like_selection and prev_center is not None and np.all(np.isfinite(prev_center)):
                dists = [np.linalg.norm(c - prev_center) for c in det_centers]
                best_idx = int(np.argmin(dists))
            else:
                best_idx = int(np.argmax(det_scores))
            chosen_center = det_centers[best_idx]
            prev_center = chosen_center.copy()

        if frame_idx < skip_frames:
            frame_idx += 1
            continue

        times.append(t)
        centers.append(chosen_center)

        if np.all(np.isfinite(chosen_center)):
            raw_ang = signed_angle_deg_right_positive(vertical_unit, chosen_center - center_origin)
        else:
            raw_ang = np.nan

        online_raw_angles.append(raw_ang)
        smooth_ang = moving_average_tail(online_raw_angles, cfg.online_smooth_window)
        online_smooth_angles.append(smooth_ang)

        if cfg.enable_early_stop_by_cycles and len(online_smooth_angles) >= max(3, cfg.zero_cross_min_index + 2):
            a1 = online_smooth_angles[-2]
            a2 = online_smooth_angles[-1]
            if is_valid_zero_cross(a1, a2, cfg.online_zero_cross_abs_gate_deg):
                if not first_cross_found:
                    first_cross_found = True
                else:
                    post_start_cross_count += 1
                    if post_start_cross_count >= 2 * cfg.target_cycles and extra_frames_left < 0:
                        extra_frames_left = cfg.early_stop_extra_frames

        if extra_frames_left >= 0:
            extra_frames_left -= 1
            if extra_frames_left < 0:
                frame_idx += 1
                break

        frame_idx += 1
        if progress_callback and frame_idx % 50 == 0:
            progress_callback(frame_idx, total_frames)

    cap.release()
    if progress_callback:
        progress_callback(total_frames, total_frames)

    times = np.array(times, dtype=np.float64)
    centers = np.array(centers, dtype=np.float64)

    raw_angles_deg = centers_to_angles(centers, center_origin, vertical_unit)
    result = process_angle_series(times, raw_angles_deg, cfg)

    out_csv = os.path.join(output_dir, f"{basename}_angle_time.csv")
    result["df"].to_csv(out_csv, index=False, float_format=cfg.csv_float_format, encoding="utf-8-sig")

    peaks_csv = os.path.join(output_dir, f"{basename}_peaks.csv")
    result["peaks_df"].to_csv(peaks_csv, index=False, float_format=cfg.csv_float_format, encoding="utf-8-sig")

    plot_path = os.path.join(output_dir, f"{basename}_angle_time.png")
    plot_angle_time(result_dict=result, save_path=plot_path, cfg=cfg,
                    title=f"{basename} Angle-Time Curve")

    return {
        "csv_path": out_csv,
        "peaks_csv": peaks_csv,
        "plot_path": plot_path,
        "num_points": len(result["df"])
    }
