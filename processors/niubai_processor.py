"""
扭摆 (Torsional Pendulum) 处理器
Adapted from analyze_pendulum_fixed.py — GUI calibration replaced with programmatic API.
"""

import os
import math
import csv
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import YOLO
from filterpy.kalman import KalmanFilter
from scipy.signal import find_peaks

# 共享工具（KMP 环境变量在 common 导入时设置，须先于 torch/ultralytics 导入）
from .common import best_detection_center


# ---------- Kalman filter (RTS smoother) ----------

def build_kalman(dt, q, r):
    kf = KalmanFilter(dim_x=2, dim_z=1)
    kf.F = np.array([[1, dt], [0, 1]])
    kf.H = np.array([[1, 0]])
    kf.R = np.array([[r]])
    kf.Q = np.eye(2) * q
    kf.P = np.eye(2) * 10
    kf.x = np.zeros((2, 1))
    return kf


def kalman_smooth(times, angles, q, r):
    n = len(angles)
    if n < 2:
        return list(angles)

    dts = np.diff(times, prepend=times[0])
    dts[0] = dts[1] if n > 1 else 1 / 30

    # Forward pass
    kf = build_kalman(dts[0], q, r)
    kf.x[0] = angles[0]
    if n >= 2 and dts[1] > 1e-9:
        kf.x[1] = (angles[1] - angles[0]) / dts[1]

    x_pred_list, P_pred_list = [], []
    x_upd_list, P_upd_list = [], []

    for i, ang in enumerate(angles):
        if i > 0:
            kf.F[0, 1] = dts[i]
        kf.predict()
        x_pred_list.append(kf.x.copy())
        P_pred_list.append(kf.P.copy())
        kf.update(np.array([[ang]]))
        x_upd_list.append(kf.x.copy())
        P_upd_list.append(kf.P.copy())

    # RTS backward pass
    x_smooth = [None] * n
    x_smooth[-1] = x_upd_list[-1].copy()

    for k in range(n - 2, -1, -1):
        F_k = np.array([[1, dts[k + 1]], [0, 1]])
        P_pred_inv = np.linalg.inv(P_pred_list[k + 1])
        G_k = P_upd_list[k] @ F_k.T @ P_pred_inv
        diff = x_smooth[k + 1] - x_pred_list[k + 1]
        x_smooth[k] = x_upd_list[k] + G_k @ diff

    return [float(x[0]) for x in x_smooth]


# ---------- Coordinate transforms ----------

def pixel_to_physical(px_point, calib):
    vec = np.array(px_point, dtype=float) - calib["origin"]
    x_px = np.dot(vec, calib["x_unit"])
    y_px = np.dot(vec, calib["y_unit"])
    return x_px / calib["px_per_m"], y_px / calib["px_per_m"]


def compute_angle(x_m, y_m, prev=None):
    ang = math.degrees(math.atan2(x_m, y_m))
    if prev is not None:
        diff = ang - prev
        while diff > 180:
            diff -= 360
        while diff < -180:
            diff += 360
        ang = prev + diff
    return ang


# ---------- Cycle extraction ----------

def extract_cycles(times, angles, n_cycles, spc):
    arr_a = np.array(angles)
    arr_t = np.array(times)

    peaks, _ = find_peaks(arr_a, prominence=0.3)
    if len(peaks) > 0 and arr_a[0] > arr_a[peaks[0]]:
        peaks = np.concatenate(([0], peaks))

    if len(peaks) < 2:
        return arr_t.tolist(), arr_a.tolist()

    n_avail = len(peaks) - 1
    if n_avail < n_cycles:
        n_cycles = n_avail

    i0 = peaks[0]
    i1 = peaks[n_cycles]
    t_win = arr_t[i0:i1 + 1]
    a_win = arr_a[i0:i1 + 1]

    total = n_cycles * spc
    if len(t_win) <= total:
        return t_win.tolist(), a_win.tolist()

    idx = np.round(np.linspace(0, len(t_win) - 1, total)).astype(int)
    return t_win[idx].tolist(), a_win[idx].tolist()


# ---------- Main processing function ----------

def process_niubai(video_path, output_dir, model_path, calibration,
                   slow_motion_factor=1,
                   skip_start_sec=1.0,
                   skip_end_sec=0.0,
                   num_cycles=80,
                   samples_per_cycle=25,
                   known_physical_dist_m=0.064,
                   yolo_conf=0.40,
                   yolo_iou=0.45,
                   target_class=0,
                   kalman_process_noise=1.0,
                   kalman_meas_noise=1e-1,
                   model=None,
                   progress_callback=None):
    """
    Process a torsional pendulum video.

    Parameters
    ----------
    video_path : str
    output_dir : str
    model_path : str
    calibration : dict
        {
            "origin": [x, y],
            "x_axis_end": [x, y],
            "y_axis_end": [x, y],
            "scale_p1": [x, y],
            "scale_p2": [x, y]
        }
        In original image coordinates.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Parse calibration
    calib = {}
    calib["origin"] = np.array(calibration["origin"], dtype=float)
    calib["x_axis_end"] = np.array(calibration["x_axis_end"], dtype=float)
    calib["y_axis_end"] = np.array(calibration["y_axis_end"], dtype=float)
    calib["scale_p1"] = np.array(calibration["scale_p1"], dtype=float)
    calib["scale_p2"] = np.array(calibration["scale_p2"], dtype=float)

    dx = calib["x_axis_end"] - calib["origin"]
    dy = calib["y_axis_end"] - calib["origin"]
    calib["x_unit"] = dx / np.linalg.norm(dx)
    calib["y_unit"] = dy / np.linalg.norm(dy)

    px_dist = np.linalg.norm(calib["scale_p2"] - calib["scale_p1"])
    if not np.isfinite(px_dist) or px_dist <= 1e-6:
        raise RuntimeError("标尺两个标定点重合或无效，无法计算比例尺，请重新标定。")
    if known_physical_dist_m <= 0:
        raise RuntimeError("标尺物理距离必须为正数（单位：米）。")
    calib["px_per_m"] = px_dist / known_physical_dist_m

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    slow = max(1.0, float(slow_motion_factor))  # 0/负数会触发除零

    f_start = int(skip_start_sec / slow * fps)
    f_end = total_frames - int(skip_end_sec / slow * fps)
    f_end = max(f_start + 1, f_end)

    # Load YOLO
    if model is None:
        model = YOLO(model_path)

    cap.set(cv2.CAP_PROP_POS_FRAMES, f_start)

    raw_times, raw_angles = [], []
    prev_angle = None

    # Display-only Kalman
    real_dt = 1.0 / fps * slow
    kf_disp = build_kalman(real_dt, kalman_process_noise, kalman_meas_noise)
    kf_disp_ready = False

    # Early-stop buffers
    peak_buf_angles = []
    peak_buf_times = []
    peak_idx_list = []      # 已确认的峰值绝对索引（增量统计）
    _last_peak_idx = -1
    stopped_early = False

    for fidx in range(f_start, f_end):
        ret, frame = cap.read()
        if not ret:
            break

        real_t = (fidx - f_start) / fps * slow

        results = model(frame, conf=yolo_conf, iou=yolo_iou, verbose=False)[0]

        # 置信度最高且类别匹配的检测（与原内联循环逐字等价：严格大于挑选、同遍历顺序）
        det = best_detection_center([results], {target_class})
        if det is not None:
            cx, cy, _best_conf = det

            xm, ym = pixel_to_physical((cx, cy), calib)
            angle = compute_angle(xm, ym, prev_angle)
            prev_angle = angle

            raw_times.append(real_t)
            raw_angles.append(angle)

            # Display-only Kalman update
            if not kf_disp_ready:
                kf_disp.x[0] = angle
                if len(raw_angles) >= 2:
                    dt_actual = raw_times[-1] - raw_times[-2]
                    if dt_actual > 0:
                        kf_disp.F[0, 1] = dt_actual
                        kf_disp.x[1] = (raw_angles[-1] - raw_angles[-2]) / dt_actual
                kf_disp_ready = True

            # Early stopping — check every frame once buffer is large enough
            peak_buf_angles.append(angle)
            peak_buf_times.append(real_t)

            if len(peak_buf_angles) > 20:
                # 只扫尾部窗口增量确认峰值（窗口滑动+去重），避免每帧全量 find_peaks 的 O(n²)
                W = 40
                lo = max(0, len(peak_buf_angles) - W)
                tail = np.asarray(peak_buf_angles[lo:])
                peaks_tail, _ = find_peaks(tail, prominence=0.3)
                for p_rel in peaks_tail:
                    p_abs = lo + int(p_rel)
                    if p_abs >= len(peak_buf_angles) - 2 or p_abs <= _last_peak_idx:
                        continue
                    _last_peak_idx = p_abs
                    peak_idx_list.append(p_abs)
                if len(peak_idx_list) >= num_cycles + 1:
                    last_peak_idx = peak_idx_list[num_cycles]
                    raw_times = peak_buf_times[:last_peak_idx + 1]
                    raw_angles = peak_buf_angles[:last_peak_idx + 1]
                    stopped_early = True

        if progress_callback and (fidx - f_start) % 50 == 0:
            progress_callback(fidx - f_start, f_end - f_start)

        if stopped_early:
            break

    cap.release()
    if progress_callback:
        progress_callback(f_end - f_start, f_end - f_start)

    if not raw_times or len(raw_times) < 2:
        raise RuntimeError("Too few detections. Check YOLO model and thresholds.")

    # RTS smoother on raw data
    smoothed = kalman_smooth(raw_times, raw_angles, kalman_process_noise, kalman_meas_noise)

    # Zero baseline at median
    med = float(np.median(smoothed))
    smoothed = [a - med for a in smoothed]

    # Cycle extraction and downsampling
    t_cyc, a_cyc = extract_cycles(raw_times, smoothed, num_cycles, samples_per_cycle)

    basename = os.path.splitext(os.path.basename(video_path))[0]

    # CSV export
    csv_path = out_dir / f"{basename}_angle_time.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_s", "angle_deg"])
        for i in range(len(t_cyc)):
            w.writerow(["%.6f" % t_cyc[i], "%.4f" % a_cyc[i]])

    # Plot
    plot_path = out_dir / f"{basename}_angle_time.png"
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle("Torsional Pendulum - Angle vs Time", fontsize=14, fontweight="bold")

    axes[0].plot(raw_times, raw_angles, "o", markersize=2, alpha=0.35, color="#4fc3f7",
                 label=f"Raw detections ({len(raw_times)})")
    axes[0].plot(raw_times, smoothed, "-", linewidth=1.5, color="#ef5350",
                 label="RTS smoother")
    axes[0].set_ylabel("Angle (deg)")
    axes[0].set_xlabel("Real Time (s)")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t_cyc, a_cyc, "s-", markersize=5, linewidth=1.5, color="#66bb6a",
                 label=f"Extracted cycles (n={num_cycles})")
    axes[1].axhline(0, color="gray", linestyle="--", linewidth=0.8)
    axes[1].set_ylabel("Angle (deg)")
    axes[1].set_xlabel("Real Time (s)")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(plot_path), dpi=150, bbox_inches="tight")
    plt.close()

    return {
        "csv_path": str(csv_path),
        "plot_path": str(plot_path),
        "num_points": len(t_cyc)
    }
