"""
磁阻尼摆 YOLO 数据提取 — 对齐 cizuni3.py（含中文字体 & 双面板图）
"""
import os, math, cv2, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy.ndimage import gaussian_filter1d
from ultralytics import YOLO

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# ---------- 配置（与 cizuni3.py 完全一致）----------
SKIP_START_SEC  = 0.15
SKIP_END_SEC    = 0
MIN_DATA_POINTS = 180
CONF_THRESHOLD  = 0.25
IOU_THRESHOLD   = 0.45
GAUSSIAN_SIGMA  = 0.8
TARGET_CLASS_IDS = None


def _setup_matplotlib_chinese():
    """Try to enable Chinese font in matplotlib (fallback gracefully)."""
    chinese_fonts = [
        "SimHei", "Microsoft YaHei", "WenQuanYi Micro Hei",
        "Noto Sans CJK SC", "Source Han Sans CN", "PingFang SC",
        "STHeiti", "Arial Unicode MS",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for font in chinese_fonts:
        if font in available:
            plt.rcParams["font.family"] = font
            plt.rcParams["axes.unicode_minus"] = False
            return True
    plt.rcParams["axes.unicode_minus"] = False
    return False


def compute_angle(px, py, ox, oy, vertical_angle_rad, flip=False):
    dx, dy = px - ox, py - oy
    ca, sa = math.cos(-vertical_angle_rad), math.sin(-vertical_angle_rad)
    angle = math.degrees(math.atan2(dx*ca - dy*sa, dx*sa + dy*ca))
    return -angle if flip else angle

def get_best_detection(results, target_class_ids):
    if not results or len(results)==0: return None
    r = results[0]
    if r.boxes is None or len(r.boxes)==0: return None
    bx = r.boxes.xywh.cpu().numpy(); bc = r.boxes.conf.cpu().numpy(); bcl = r.boxes.cls.cpu().numpy()
    best = -1.0; bx_c, by_c = None, None
    for box, conf, cls in zip(bx, bc, bcl):
        if target_class_ids is not None and int(cls) not in target_class_ids: continue
        if conf > best: best=float(conf); bx_c=float(box[0]); by_c=float(box[1])
    return (bx_c, by_c, best) if bx_c is not None else None

def auto_sample_step(total_frames, fps, ss, se, min_pts):
    sf = int(ss*fps); ef = int(se*fps); eff = total_frames - sf - ef
    return max(1, eff // min_pts) if eff > 0 else 1

def process_cizuni(video_path, output_dir, model_path, calibration,
                   skip_start_sec=SKIP_START_SEC, skip_end_sec=SKIP_END_SEC,
                   min_data_points=MIN_DATA_POINTS, conf_threshold=CONF_THRESHOLD,
                   iou_threshold=IOU_THRESHOLD, gaussian_sigma=GAUSSIAN_SIGMA,
                   target_class_ids=TARGET_CLASS_IDS, slow_motion_factor=1.0,
                   flip_angle=False, model=None, progress_callback=None):
    os.makedirs(output_dir, exist_ok=True)

    # 标定点（来自网页点击，已转为原始坐标）
    v_pt1 = (int(calibration["vertical_ref_1"][0]), int(calibration["vertical_ref_1"][1]))
    v_pt2 = (int(calibration["vertical_ref_2"][0]), int(calibration["vertical_ref_2"][1]))
    origin_px = (int(calibration["pivot"][0]), int(calibration["pivot"][1]))
    dx, dy = v_pt2[0] - v_pt1[0], v_pt2[1] - v_pt1[1]
    vertical_angle_rad = math.atan2(dx, dy)

    # 打开视频（完全照搬 cizuni3.py process_video）
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0: raise RuntimeError("Cannot read FPS")

    # Convert skip times (physical) to frame indices (video time).
    # sample_step is computed in video-time frames.
    sample_step = auto_sample_step(total_frames, fps, skip_start_sec / slow_motion_factor, skip_end_sec / slow_motion_factor, min_data_points)

    # Always create a fresh model to avoid CUDA context issues with pre-loaded models
    model = YOLO(model_path)

    sf = int(skip_start_sec / slow_motion_factor * fps)
    ef = int(skip_end_sec / slow_motion_factor * fps)
    end_idx = total_frames - ef

    times_raw, angles_raw = [], []
    last_valid_angle = 0.0
    prev_det, frame_idx, processed = None, 0, 0

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    while True:
        ret, frame = cap.read()
        if not ret: break
        in_range = (sf <= frame_idx < end_idx)
        is_sample = in_range and (frame_idx % sample_step == 0)
        frame_time = frame_idx / fps * slow_motion_factor
        det, results_obj = None, None

        if in_range and is_sample:
            results_obj = model.predict(frame, conf=conf_threshold, iou=iou_threshold, verbose=False)
            det = get_best_detection(results_obj, target_class_ids)
            if det is not None:
                angle = compute_angle(det[0], det[1], origin_px[0], origin_px[1], vertical_angle_rad, flip=flip_angle)
                times_raw.append(frame_time - skip_start_sec)
                angles_raw.append(angle)
                prev_det = det; last_valid_angle = angle
                processed += 1
            else:
                det = prev_det
        frame_idx += 1
        if progress_callback and frame_idx % 50 == 0:
            progress_callback(frame_idx, total_frames)

    cap.release()
    if progress_callback:
        progress_callback(total_frames, total_frames)

    if len(times_raw) < 2:
        raise RuntimeError(f"YOLO detected 0 objects in {frame_idx} frames (conf={conf_threshold}).")
    times_raw = np.array(times_raw); angles_raw = np.array(angles_raw)

    # Gaussian filter（照搬 cizuni3.py）
    angles_filtered = gaussian_filter1d(angles_raw, sigma=gaussian_sigma)

    # 输出 CSV（列名照搬原版）
    basename = os.path.splitext(os.path.basename(video_path))[0]
    csv_path = os.path.join(output_dir, f"{basename}_angle_time.csv")
    df = pd.DataFrame({"time_s": np.round(times_raw,6), "angle_deg": np.round(angles_filtered,6)})
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    # ---------- 画图（对齐 cizuni3.py 双面板 + 中文字体）----------
    has_cn = _setup_matplotlib_chinese()
    if has_cn:
        title_main  = "磁阻尼复摆 — 摆角时间图 (YOLOv8)"
        label_raw   = "原始摆角"
        label_filt  = f"高斯滤波 (σ={gaussian_sigma})"
        label_filt2 = "滤波后摆角"
        xlabel      = "时间 t (s)"
        ylabel      = "摆角 θ (°)"
    else:
        title_main  = "Magnetic Damping Pendulum — Angle vs Time (YOLOv8)"
        label_raw   = "Raw angle"
        label_filt  = f"Gaussian filtered (sigma={gaussian_sigma})"
        label_filt2 = "Filtered angle"
        xlabel      = "Time t (s)"
        ylabel      = "Angle θ (deg)"

    plot_path = os.path.join(output_dir, f"{basename}_angle_time.png")
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(title_main, fontsize=14, fontweight="bold")

    axes[0].plot(times_raw, angles_raw,
                 color="#5b8dd9", lw=0.8, alpha=0.6, label=label_raw)
    axes[0].plot(times_raw, angles_filtered,
                 color="#e05c5c", lw=1.8, label=label_filt)
    axes[0].set_ylabel(ylabel)
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[0].axhline(0, color="gray", lw=0.8, ls="--")

    axes[1].plot(times_raw, angles_filtered,
                 color="#e05c5c", lw=1.8, label=label_filt2)
    axes[1].fill_between(times_raw, angles_filtered, 0,
                         alpha=0.15, color="#e05c5c")
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel(ylabel)
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    axes[1].axhline(0, color="gray", lw=0.8, ls="--")

    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    return {"csv_path": csv_path, "plot_path": plot_path, "num_points": len(df)}
