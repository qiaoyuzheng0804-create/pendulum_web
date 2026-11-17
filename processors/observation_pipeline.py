# -*- coding: utf-8 -*-
"""Preserve frame observations and timing; smoothing is diagnostic only."""
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shutil

import cv2
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from .common import H264VideoWriter, video_readable


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sanitize_json(obj):
    if isinstance(obj, dict):
        return {k: _sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_json(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, np.ndarray):
        return _sanitize_json(obj.tolist())
    return obj


def write_json(path, data):
    clean = _sanitize_json(data)
    Path(path).write_text(
        json.dumps(clean, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def observation_times(frame_indices, decoder_seconds, fps, cfg):
    """N-times slow playback means physical elapsed time = playback time/N."""
    factor = float(cfg.slow_motion_factor)
    if not np.isfinite(factor) or factor <= 0:
        raise ValueError("slow_motion_factor must be positive; 1 means original speed")
    if cfg.timestamp_mode == "decoder_pts":
        playback = np.asarray(decoder_seconds, dtype=float)
    elif cfg.timestamp_mode == "confirmed_cfr":
        if not cfg.cfr_confirmed:
            raise ValueError("CFR fallback requires cfr_confirmed=true from capture/export records")
        rate = cfg.confirmed_fps if cfg.confirmed_fps is not None else fps
        if not np.isfinite(rate) or rate <= 0:
            raise ValueError("No valid confirmed frame rate; no assumed 30 FPS fallback")
        playback = np.asarray(frame_indices, dtype=float) / float(rate)
    else:
        raise ValueError("timestamp_mode must be decoder_pts or confirmed_cfr")
    if len(playback) < 2 or not np.isfinite(playback).all() or np.any(np.diff(playback) <= 0):
        raise ValueError("Video timestamps are missing/nonmonotonic; retain audit and confirm the recording clock before inversion")
    physical = (playback - playback[0]) / factor
    return playback, physical


def resolve_observation_times(frame_indices, decoder_seconds, fps, cfg):
    """Call observation_times; fall back to container FPS ONLY when PTS fails and fps > 0."""
    try:
        playback, physical = observation_times(frame_indices, decoder_seconds, fps, cfg)
        return playback, physical, cfg.timestamp_mode
    except ValueError as exc:
        if (
            cfg.timestamp_mode == "decoder_pts"
            and bool(getattr(cfg, "allow_fps_fallback", True))
            and fps is not None
            and np.isfinite(fps)
            and float(fps) > 0
            and len(frame_indices) >= 2
        ):
            factor = float(cfg.slow_motion_factor)
            if not np.isfinite(factor) or factor <= 0:
                raise
            print(
                f"[observation_pipeline] decoder_pts invalid ({exc}); falling back to container FPS={float(fps):.4f}",
                flush=True,
            )
            playback = np.asarray(frame_indices, dtype=float) / float(fps)
            physical = (playback - playback[0]) / factor
            return playback, physical, "cfr_fps_fallback"
        raise


def project_points(points, homography):
    points = np.asarray(points, dtype=float)
    if homography is None:
        return points
    matrix = np.asarray(homography, dtype=float)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all() or abs(np.linalg.det(matrix)) < 1e-12:
        raise ValueError("plane_homography must be a finite nonsingular calibrated 3x3 matrix")
    mapped = np.c_[points, np.ones(len(points))] @ matrix.T
    if np.any(np.abs(mapped[:, 2]) < 1e-12):
        raise ValueError("Calibration maps a point to infinity")
    return mapped[:, :2] / mapped[:, 2:]


def measured_angle(center, calibration, flip_angle=False):
    points = [calibration["center"], calibration["vertical_start"],
              calibration["vertical_end"], center]
    pivot, start, end, bob = project_points(points, calibration.get("plane_homography"))
    vertical, radius = end - start, bob - pivot
    if min(np.linalg.norm(vertical), np.linalg.norm(radius)) < 1e-12:
        raise ValueError("Zero vertical vector or bob coincident with pivot")
    cross = vertical[0] * radius[1] - vertical[1] * radius[0]
    deg = float(-np.degrees(np.arctan2(cross, np.dot(vertical, radius))))
    return -deg if flip_angle else deg


def diagnostic_series(times, angles, cfg):
    """Filter contiguous observed runs; raw values, times and gaps never change."""
    times, angles = np.asarray(times, dtype=float), np.asarray(angles, dtype=float)
    if len(times) < 2 or len(times) != len(angles) or np.any(np.diff(times) <= 0):
        raise ValueError("Require a strictly increasing observation grid")
    raw = angles.copy()
    filt = np.full(len(raw), np.nan)
    nominal = float(np.median(np.diff(times)))
    finite = np.isfinite(raw)
    starts = np.flatnonzero(finite & ~np.r_[False, finite[:-1]])
    ends = np.flatnonzero(finite & ~np.r_[finite[1:], False]) + 1
    for start, end in zip(starts, ends):
        # Split detected runs at timestamp gaps as well as missing detections.
        cuts = np.r_[start, start + 1 + np.flatnonzero(np.diff(times[start:end]) > 1.8 * nominal), end]
        for left, right in zip(cuts[:-1], cuts[1:]):
            segment = raw[left:right]
            window = min(int(cfg.savgol_window), len(segment))
            if window % 2 == 0:
                window -= 1
            if window > int(cfg.savgol_polyorder):
                # Reconstruct ONLY the display signal on a uniform time grid.
                uniform = np.linspace(times[left], times[right - 1], len(segment))
                smoothed = savgol_filter(np.interp(uniform, times[left:right], segment),
                                         window, int(cfg.savgol_polyorder))
                filt[left:right] = np.interp(times[left:right], uniform, smoothed)
            else:
                filt[left:right] = segment
    valid = finite
    df = pd.DataFrame(dict(time_s=times[valid], angle_deg=raw[valid],
                           raw_angle_deg=raw[valid], filtered_angle_deg=filt[valid]))
    return dict(
        times_seg=times.copy(), raw_seg=raw, filt_seg=filt,
        envelope_seg=np.full(len(raw), np.nan), peak_idx=np.array([], dtype=int),
        peak_time=np.array([]), peak_angle=np.array([]), peak_abs_angle=np.array([]),
        peak_type=np.array([]), sample_t=times[valid], sample_raw=raw[valid],
        sample_filt=filt[valid], sample_env=np.full(valid.sum(), np.nan), df=df,
        peaks_df=pd.DataFrame(columns=["peak_time_s", "peak_angle_deg",
                                       "peak_abs_angle_deg", "peak_type"]),
    )


def _to_numpy(arr):
    if hasattr(arr, "cpu"):
        return arr.cpu().numpy()
    return np.asarray(arr)


def select_detection(boxes, previous, cfg):
    """Return actual chosen box, retaining missing/rejection reasons for audit."""
    if boxes is None or len(boxes) == 0:
        return None, 0, "no_detection"
    xyxy = _to_numpy(boxes.xyxy)
    scores = _to_numpy(boxes.conf)
    classes = _to_numpy(boxes.cls).astype(int)
    eligible = np.arange(len(scores))
    if cfg.bob_class_id is not None:
        eligible = eligible[classes == int(cfg.bob_class_id)]
    if not len(eligible):
        return None, len(scores), "class_mismatch"
    centers = (xyxy[:, :2] + xyxy[:, 2:]) / 2
    if cfg.use_tracking_like_selection and previous is not None:
        index = int(eligible[np.argmin(np.linalg.norm(centers[eligible] - previous, axis=1))])
    else:
        index = int(eligible[np.argmax(scores[eligible])])
    jump = float(np.linalg.norm(centers[index] - previous)) if previous is not None else None
    result = dict(
        center=centers[index], confidence=float(scores[index]),
        class_id=int(classes[index]), box=xyxy[index].tolist(), center_jump_px=jump,
    )
    if cfg.max_center_jump_px is not None and jump is not None and jump > cfg.max_center_jump_px:
        return result, len(scores), "jump_gate_rejected"
    return result, len(scores), "observed"


def load_local_detector(weights):
    """Use local weights and a project-owned Ultralytics settings directory."""
    if not Path(weights).is_file():
        raise FileNotFoundError(weights)
    settings = Path(__file__).resolve().parents[1] / "config/yolo_runtime"
    settings.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(settings))
    from ultralytics import YOLO
    return YOLO(str(weights))


def selected_frame_bounds(cfg):
    """Exact inclusive frame bounds take precedence over legacy second bounds.

    Supports partial lower bound (start_frame_index without end_frame_index)
    for web interactive start-frame picking.
    """
    start = getattr(cfg, "start_frame_index", None)
    end = getattr(cfg, "end_frame_index", None)
    if start is None and end is None:
        return None
    for v in (start, end):
        if v is not None and (isinstance(v, bool) or not isinstance(v, (int, np.integer)) or int(v) < 0):
            raise ValueError("Selected frame bounds must be nonnegative integers")
    s_val = int(start) if start is not None else 0
    e_val = int(end) if end is not None else None
    if e_val is not None and e_val < s_val + 2:
        raise ValueError("Selected frame interval requires at least three frames (end >= start + 2)")
    return s_val, e_val


def extract(
    video_path,
    output_dir,
    model_path,
    calibration,
    cfg,
    model=None,
    progress_callback=None,
    plotter=None,
    frame_annotator=None,
    write_video=False,
    video_filename="annotated_video.mp4",
    csv_filename=None,
    plot_filename=None,
    csv_prefix=None,
    audit_suffix="_frame_audit.csv",
    extra_metadata=None,
):
    frame_bounds = selected_frame_bounds(cfg)
    out = Path(output_dir).resolve()
    allow_nonempty = bool(getattr(cfg, "allow_nonempty_output", True))
    if not allow_nonempty and out.exists() and any(out.iterdir()):
        raise FileExistsError("Output directory is not empty; choose a new run directory")
    out.mkdir(parents=True, exist_ok=True)

    video = Path(video_path).resolve()
    weights = Path(model_path).resolve() if model_path else Path("")
    if not video.is_file():
        raise FileNotFoundError(f"Video file not found: {video}")
    if model is None and not weights.is_file():
        raise FileNotFoundError(f"Provide existing local model weights: {weights}")

    flip_angle = bool(getattr(cfg, "flip_angle", False))
    # Validate geometric controls before loading a model or processing a video.
    measured_angle(
        np.asarray(calibration["center"], dtype=float)
        + np.asarray(calibration["vertical_end"], dtype=float)
        - np.asarray(calibration["vertical_start"], dtype=float),
        calibration,
        flip_angle=flip_angle,
    )
    if frame_bounds is None:
        if not np.isfinite(cfg.skip_initial_seconds) or cfg.skip_initial_seconds < 0:
            raise ValueError("skip_initial_seconds must be nonnegative physical seconds")
        if cfg.release_end_s is not None and cfg.release_end_s <= cfg.skip_initial_seconds:
            raise ValueError("release_end_s must exceed skip_initial_seconds")
    if cfg.max_center_jump_px is not None and cfg.max_center_jump_px <= 0:
        raise ValueError("max_center_jump_px must be positive when enabled")

    input_hashes = {str(video): sha(video)}
    if weights.is_file():
        input_hashes[str(weights)] = sha(weights)
    source_hashes = {p.name: sha(p) for p in Path(__file__).parent.glob("*.py")}

    if model is None:
        model = load_local_detector(weights)

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError("Cannot open video: " + str(video))

    if getattr(cfg, "video_orientation_auto", None) is not None:
        cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, float(bool(cfg.video_orientation_auto)))

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    advertised = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    backend = cap.getBackendName()

    start_idx = frame_bounds[0] if frame_bounds is not None else None
    end_idx = frame_bounds[1] if frame_bounds is not None else None

    video_writer = None
    out_video_path = str(out / video_filename)
    out_w, out_h, out_scale = orig_w, orig_h, 1.0
    max_side = int(getattr(cfg, "output_video_max_side", 1280) or 0)
    if max_side > 0 and max(orig_w, orig_h) > max_side:
        out_scale = float(max_side) / float(max(orig_w, orig_h))
        out_w = max(2, int(round(orig_w * out_scale)) // 2 * 2)
        out_h = max(2, int(round(orig_h * out_scale)) // 2 * 2)

    if write_video and frame_annotator is not None and orig_w > 0 and orig_h > 0:
        writer_fps = float(fps) if (np.isfinite(fps) and fps > 0) else 30.0
        video_writer = H264VideoWriter(out_video_path, writer_fps, out_w, out_h)
        if not video_writer.isOpened():
            video_writer = None

    rows, previous, missed = [], None, 0
    decode_error = False
    disp_t0 = None
    slow_val = float(cfg.slow_motion_factor)
    try:
        while True:
            i = len(rows)
            if end_idx is not None and i > end_idx:
                break
            ok, frame = cap.read()
            if not ok:
                decode_error = advertised > 0 and len(rows) < advertised
                break
            decoder = float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
            in_infer_window = (start_idx is None or i >= start_idx) and (end_idx is None or i <= end_idx)

            row = dict(
                frame_index=i,
                decoder_time_s=decoder,
                detection_count=0,
                observed=False,
                detection_status="before_start_frame" if not in_infer_window else "no_detection",
                gap_filled=False,
                center_x_px=np.nan,
                center_y_px=np.nan,
                confidence=np.nan,
                class_id=np.nan,
                raw_angle_deg=np.nan,
                radius_px=np.nan,
                center_jump_px=np.nan,
                box_x1_px=np.nan,
                box_y1_px=np.nan,
                box_x2_px=np.nan,
                box_y2_px=np.nan,
            )

            chosen = None
            if in_infer_window:
                results = model.predict(
                    source=frame,
                    conf=cfg.conf_thres,
                    iou=cfg.iou_thres,
                    imgsz=cfg.imgsz,
                    max_det=cfg.max_det,
                    device=cfg.device,
                    verbose=False,
                )
                chosen, count, reason = select_detection(results[0].boxes if results else None, previous, cfg)
                row["detection_count"] = count
                row["observed"] = (reason == "observed")
                row["detection_status"] = reason
                if chosen is not None:
                    row.update(
                        center_x_px=float(chosen["center"][0]),
                        center_y_px=float(chosen["center"][1]),
                        confidence=chosen["confidence"],
                        class_id=chosen["class_id"],
                        radius_px=float(np.linalg.norm(chosen["center"] - np.asarray(calibration["center"], dtype=float))),
                        center_jump_px=chosen["center_jump_px"],
                    )
                    row.update(dict(zip(["box_x1_px", "box_y1_px", "box_x2_px", "box_y2_px"], chosen["box"])))
                if row["observed"]:
                    try:
                        row["raw_angle_deg"] = measured_angle(chosen["center"], calibration, flip_angle=flip_angle)
                        previous, missed = chosen["center"].copy(), 0
                    except ValueError:
                        row.update(observed=False, detection_status="invalid_geometry")
                if not row["observed"]:
                    missed += 1
                    if missed >= cfg.tracking_reset_after_misses:
                        previous = None

                if video_writer is not None and frame_annotator is not None:
                    if disp_t0 is None:
                        disp_t0 = decoder if (np.isfinite(decoder) and (i == 0 or decoder > 0)) else (
                            i / fps if (np.isfinite(fps) and fps > 0) else 0.0
                        )
                    cur_t = decoder if (np.isfinite(decoder) and decoder >= disp_t0) else (
                        i / fps if (np.isfinite(fps) and fps > 0) else disp_t0
                    )
                    t_disp = max(0.0, (cur_t - disp_t0) / slow_val)
                    disp = frame.copy() if out_scale >= 1.0 else cv2.resize(
                        frame, (out_w, out_h), interpolation=cv2.INTER_AREA
                    )
                    ann = frame_annotator(disp, row, chosen, t_disp, calibration, out_scale, i, advertised)
                    if not video_writer.write(ann):
                        video_writer.close()
                        video_writer = None

            rows.append(row)
            if progress_callback and (len(rows) % 25 == 0 or (advertised > 0 and len(rows) >= advertised)):
                progress_callback(len(rows), max(advertised, len(rows)))
    finally:
        cap.release()

    video_codec = ""
    video_playable = False
    video_error = None
    saved_video_path = None
    if video_writer is not None:
        close_err = video_writer.close()
        video_codec = video_writer.codec
        if close_err:
            video_error = close_err
        elif not os.path.isfile(out_video_path) or os.path.getsize(out_video_path) < 256 or not video_readable(out_video_path):
            video_error = f"标注视频写出异常（编码器 {video_writer.backend}，已写 {video_writer.written} 帧）"
        else:
            saved_video_path = out_video_path
            video_playable = (video_codec == "h264")
            if not video_playable:
                video_error = f"缺少 libx264/H.264 编码器（当前 {video_codec or video_writer.backend}），浏览器无法直接内联播放。"
        if video_error:
            print(f"[视频 WARN] {video_error}", flush=True)

    audit = pd.DataFrame(rows)
    if len(audit) < 3:
        raise ValueError(f"Decoded only {len(audit)} frames; require at least 3 frames")

    prefix = csv_prefix or (Path(csv_filename).stem if csv_filename else video.stem)
    audit_path = out / (prefix + audit_suffix)
    audit.to_csv(audit_path, index=False, encoding="utf-8-sig", float_format="%.12g")

    cfg_dict = asdict(cfg) if hasattr(cfg, "__dataclass_fields__") else dict(vars(cfg))
    metadata = dict(
        schema_version=2,
        created=datetime.now().isoformat(timespec="seconds"),
        video_path=str(video),
        video_sha256=input_hashes[str(video)],
        model_path=str(weights) if weights.is_file() else "preloaded_model",
        model_sha256=input_hashes.get(str(weights), "preloaded_model"),
        source_sha256=source_hashes,
        configuration=cfg_dict,
        calibration=calibration,
        decoder_backend=backend,
        reported_fps=fps if math.isfinite(fps) else None,
        advertised_frames=advertised,
        decoded_frames=len(rows),
        decode_ended_before_advertised_count=decode_error,
        camera_original_dropped_frames="unknown; encoded timestamps do not establish camera completeness",
        time_calibration_confirmed=cfg.time_calibration_confirmed,
        release_id=cfg.release_id,
        configuration_id=cfg.configuration_id,
        projection_interpretation="rectified plane angle" if calibration.get("plane_homography") else "image-plane projected angle",
        physical_angle_calibration_confirmed=calibration.get("physical_angle_calibration_confirmed", False),
        raw_angle_processing="atan2 of actual detected center and saved calibration only",
        synthetic_training_points=0,
        raw_gap_interpolation=False,
        angle_thinning=False,
        automatic_zero_offset=False,
        forced_initial_zero=False,
        requested_cycle_count_not_used_for_capture=cfg.target_cycles,
    )
    if extra_metadata:
        metadata.update(extra_metadata)

    if any(sha(path) != digest for path, digest in input_hashes.items()):
        metadata.update(status="source_changed_during_extraction")
        write_json(out / "提取清单.json", metadata)
        raise RuntimeError("Video or weights changed during extraction; use a fresh run")
    write_json(out / "标定.json", calibration)

    try:
        playback, physical, mode_used = resolve_observation_times(audit.frame_index, audit.decoder_time_s, fps, cfg)
    except ValueError as exc:
        metadata.update(status="timing_failed", reason=str(exc))
        write_json(out / "提取清单.json", metadata)
        raise

    audit["video_time_s"], audit["physical_elapsed_s"] = playback, physical
    if frame_bounds is not None:
        start_b, end_b = frame_bounds
        if start_b >= len(audit):
            metadata.update(status="selected_frames_not_decoded")
            write_json(out / "提取清单.json", metadata)
            raise ValueError("Selected start frame was not decoded; check video completeness")
        if end_b is not None and end_b >= len(audit):
            metadata.update(status="selected_frames_not_decoded")
            write_json(out / "提取清单.json", metadata)
            raise ValueError("Selected end frame was not decoded; check video completeness")
        eff_end = end_b if end_b is not None else (len(audit) - 1)
        if eff_end < start_b + 2:
            metadata.update(status="insufficient_observations")
            write_json(out / "提取清单.json", metadata)
            raise ValueError("Selected frame interval contains fewer than 3 frames")
        keep = (audit.frame_index.to_numpy() >= start_b) & (audit.frame_index.to_numpy() <= eff_end)
    else:
        keep = physical >= cfg.skip_initial_seconds
        if cfg.release_end_s is not None:
            keep &= physical <= cfg.release_end_s

    audit["in_selected_interval"] = keep
    audit["include_for_inverse"] = keep & audit.observed.to_numpy(bool)
    if keep.sum() < 3 or audit.include_for_inverse.sum() < 3:
        metadata.update(status="insufficient_observations")
        write_json(out / "提取清单.json", metadata)
        raise ValueError("Insufficient observed frames in the requested physical time interval")

    # Origin is the actual timestamp of the first selected decoded frame.
    origin = float(physical[np.flatnonzero(keep)[0]])
    audit["time_s"] = physical - origin
    series = diagnostic_series(
        audit.loc[keep, "time_s"].to_numpy(),
        audit.loc[keep, "raw_angle_deg"].to_numpy(),
        cfg,
    )
    audit["filtered_angle_deg"] = np.nan
    audit.loc[keep, "filtered_angle_deg"] = series["filt_seg"]
    audit.to_csv(audit_path, index=False, encoding="utf-8-sig", float_format="%.12g")

    columns = [
        "time_s", "raw_angle_deg", "filtered_angle_deg", "frame_index",
        "video_time_s", "physical_elapsed_s", "center_x_px", "center_y_px",
        "confidence", "class_id", "radius_px", "observed", "gap_filled",
    ]
    csv_path = out / (csv_filename if csv_filename else (prefix + "_angle_time.csv"))
    audit.loc[audit.include_for_inverse, columns].to_csv(
        csv_path, index=False, encoding="utf-8-sig", float_format="%.12g"
    )

    plot_path = None
    if cfg.save_angle_time_plot and plotter:
        plot_path = out / (plot_filename if plot_filename else (prefix + "_angle_time.png"))
        plotter(series, str(plot_path), cfg, prefix + " observed frame angles")

    dt = np.diff(physical)
    median = float(np.median(dt))
    obs_times = audit.loc[audit.include_for_inverse, "time_s"]
    metadata.update(
        status="complete",
        observed_frames=int(audit.observed.sum()),
        interval_selection="inclusive_frame_indices" if frame_bounds is not None else "physical_seconds",
        selected_start_frame_index=int(audit.loc[keep, "frame_index"].iloc[0]),
        selected_end_frame_index=int(audit.loc[keep, "frame_index"].iloc[-1]),
        missed_or_rejected_frames=int((~audit.observed).sum()),
        inverse_observations=int(audit.include_for_inverse.sum()),
        time_origin_physical_s=origin,
        timestamp_mode=mode_used,
        playback_to_physical_time="(timestamp-timestamp_first)/slow_motion_factor",
        physical_step_min_s=float(dt.min()),
        physical_step_median_s=median,
        physical_step_max_s=float(dt.max()),
        timestamp_gap_frame_indices=audit.frame_index.iloc[1:][dt > 1.8 * median].tolist(),
        detection_status_counts={str(k): int(v) for k, v in audit.detection_status.value_counts().items()},
        training_columns={"time": "time_s", "signal": "raw_angle_deg"},
        diagnostic_filter_never_used_to_replace_raw=True,
        video_codec=video_codec,
        video_playable=video_playable,
        video_error=video_error,
        output_sha256={p.name: sha(p) for p in out.iterdir() if p.is_file()},
    )
    manifest_path = out / "提取清单.json"
    write_json(manifest_path, metadata)
    return dict(
        csv_path=str(csv_path),
        frame_audit_csv=str(audit_path),
        manifest=str(manifest_path),
        meta_path=str(manifest_path),
        peaks_csv=None,
        plot_path=str(plot_path) if plot_path else None,
        video_path=saved_video_path,
        video_codec=video_codec,
        video_playable=video_playable,
        video_error=video_error,
        num_points=int(audit.include_for_inverse.sum()),
        start_frame=int(audit.loc[keep, "frame_index"].iloc[0]),
        end_frame=int(audit.loc[keep, "frame_index"].iloc[-1]),
        real_fps=(1.0 / median) if median > 0 else float(fps) * slow_val,
        slow_motion_factor=slow_val,
        time_range_s=[float(obs_times.iloc[0]), float(obs_times.iloc[-1])],
    )
