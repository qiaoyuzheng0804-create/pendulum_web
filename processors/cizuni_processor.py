# -*- coding: utf-8 -*-
"""
磁阻尼摆 (Magnetic Damping Pendulum) 处理器
==========================================
完全对齐底层 cizuni3.py：

  · 标定 3 点：竖直参考点 1 / 竖直参考点 2 / 转轴原点（网页点击传入，原图坐标）
  · 角度变换 compute_angle 与底层逐字一致，并用 verify_compute_angle 做自检
    （⚠️ 早期网页版用的是 sin(-a)/cos(-a)，会给每个角度叠加恒定 2a 偏差）
  · 时间基准：real_time = video_time / slow_motion_factor，real_fps = fps * slow；
    slow 支持 "auto"（读 MP4 的 com.android.capture.fps 标签）与显式数值
  · 采样步长 auto_sample_step（预算 max_model_frames），短片段逐帧
  · 高斯滤波 gaussian_filter1d（与底层一致，不做额外重采样）
  · 输出 pendulum_angle.csv / angle_time_plot.png / annotated_video.mp4
"""

import math
import os
import struct

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from ultralytics import YOLO

# 共享工具（KMP 环境变量在 common 导入时设置，须先于 torch/ultralytics 导入）
from .common import setup_matplotlib_chinese

# ---------- 配置（与 cizuni3.py 对齐）----------
CONF_THRESHOLD   = 0.25
IOU_THRESHOLD    = 0.45
TARGET_CLASS_IDS = None
GAUSSIAN_SIGMA   = 0.8
SAMPLE_STEP      = None       # None = 自动
MAX_MODEL_FRAMES = 3000
SLOW_MOTION_FACTOR = "auto"

CSV_FILENAME   = "pendulum_angle.csv"
PLOT_FILENAME  = "angle_time_plot.png"
VIDEO_FILENAME = "annotated_video.mp4"

DRAW_TRAJECTORY = True
TRAJECTORY_LEN  = 60
CENTER_COLOR    = (0, 128, 255)
ORIGIN_COLOR    = (255, 80, 80)
AXIS_COLOR      = (255, 220, 0)
TEXT_COLOR      = (255, 255, 255)


# ==============================================================================
#  绘制 / UI 小工具
# ==============================================================================

def _put_ascii_text(img, text, pos, scale=0.6, color=(255, 255, 255),
                    thickness=1, bg=False):
    """ASCII 文本（避开 OpenCV 中文字体问题），可选黑底描边。"""
    if bg:
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        x, y = int(pos[0]), int(pos[1])
        cv2.rectangle(img, (x - 4, y - th - 4), (x + tw + 4, y + baseline + 2),
                      (0, 0, 0), -1)
    cv2.putText(img, text, (int(pos[0]), int(pos[1])), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)


# ==============================================================================
#  时间基准：慢放倍数（可从 MP4 标签自动读取）
# ==============================================================================

def _parse_meta_capture_fps(meta):
    """从 movie-level 'meta' box 解析 com.android.capture.fps（32 位浮点，类型 23）。"""
    try:
        k = meta.find(b"keys")
        if k < 0:
            return None
        off = k + 8
        n = struct.unpack(">I", meta[off:off + 4])[0]
        off += 4
        names = []
        for _ in range(n):
            esize = struct.unpack(">I", meta[off:off + 4])[0]
            if esize < 8 or off + esize > len(meta):
                return None
            names.append(meta[off + 8:off + esize].decode("utf-8", "replace"))
            off += esize
        if "com.android.capture.fps" not in names:
            return None
        want = names.index("com.android.capture.fps") + 1

        ip = meta.find(b"ilst")
        if ip < 0:
            return None
        off = ip + 4
        while off + 8 <= len(meta):
            sz = struct.unpack(">I", meta[off:off + 4])[0]
            idx = struct.unpack(">I", meta[off + 4:off + 8])[0]
            if sz < 8 or off + sz > len(meta):
                break
            if idx == want:
                body = meta[off + 8:off + sz]
                dp = body.find(b"data")
                if dp >= 0:
                    code = struct.unpack(">I", body[dp + 4:dp + 8])[0]
                    payload = body[dp + 12:]
                    if code == 23 and len(payload) >= 4:
                        return float(struct.unpack(">f", payload[:4])[0])
                    if code == 1:
                        try:
                            return float(payload.decode("utf-8", "replace").strip())
                        except ValueError:
                            return None
                return None
            off += sz
    except Exception:
        return None
    return None


def read_mp4_capture_fps(path):
    """相机 HAL 记录的原始拍摄帧率，读不到返回 None；绝不抛异常。"""
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            fsize = f.tell()
            f.seek(0)
            while f.tell() + 8 <= fsize:
                pos = f.tell()
                head = f.read(8)
                if len(head) < 8:
                    return None
                size, typ = struct.unpack(">I4s", head)
                if size == 1:
                    size = struct.unpack(">Q", f.read(8))[0]
                elif size == 0:
                    size = fsize - pos
                if size < 8:
                    return None
                if typ == b"moov":
                    f.seek(pos + 8)
                    vend = pos + size
                    while f.tell() + 8 <= vend:
                        p2 = f.tell()
                        s2, t2 = struct.unpack(">I4s", f.read(8))
                        if s2 == 1:
                            s2 = struct.unpack(">Q", f.read(8))[0]
                        elif s2 == 0:
                            s2 = vend - p2
                        if s2 < 8:
                            return None
                        if t2 == b"meta":
                            f.seek(p2)
                            return _parse_meta_capture_fps(f.read(s2))
                        f.seek(p2 + s2)
                    return None
                f.seek(pos + size)
    except Exception:
        return None
    return None


def resolve_slow_motion_factor(video_path, container_fps, cfg_value):
    """
    video time -> real time：real_time = video_time / factor。
    返回 (factor, source)，source ∈ {'CONFIG', 'mp4-tag', 'assumed'}。
    """
    if isinstance(cfg_value, bool):
        cfg_value = None
    if isinstance(cfg_value, (int, float)):
        f = float(cfg_value)
        return (f if f > 0 else 1.0), "CONFIG"
    if isinstance(cfg_value, str) and cfg_value.strip().lower() not in ("", "auto"):
        try:
            f = float(cfg_value)
            return (f if f > 0 else 1.0), "CONFIG"
        except ValueError:
            pass
    cap = read_mp4_capture_fps(video_path)
    if cap and cap > 0 and container_fps > 0:
        f = cap / container_fps
        if 0.995 <= f <= 1.005:
            return 1.0, "mp4-tag: capture %.1f fps == container" % cap
        return f, "mp4-tag: capture %.1f fps" % cap
    return 1.0, "assumed"


# ==============================================================================
#  角度
# ==============================================================================

def compute_angle(px, py, origin_px, origin_py, vertical_angle_rad, flip=False):
    """
    摆角（度）。原点在转轴，+Y 轴为竖直向下，顺时针为正。

    ⚠️ 与底层逐字一致：用 +sin/+cos 旋转，才能把"点在标定的竖直参考线上"
    映射为 0°。写成 sin(-a)/cos(-a) 会额外叠加恒定的 2a 偏差（自检会抓到）。
    """
    dx = px - origin_px
    dy = py - origin_py
    cos_a = math.cos(vertical_angle_rad)
    sin_a = math.sin(vertical_angle_rad)
    dx_rot = dx * cos_a - dy * sin_a
    dy_rot = dx * sin_a + dy * cos_a
    angle = math.degrees(math.atan2(dx_rot, dy_rot))
    return -angle if flip else angle


def verify_compute_angle(vertical_angle_rad, flip=False, tol=1e-6):
    """
    自检：标定参考线上的点必须读 0°；偏离 d 度的点必须读 ±d。
    返回 True 表示通过。
    """
    a = vertical_angle_rad
    cases = [("on reference line", 0.0, 0.0)]
    for d in (-90.0, -30.0, 30.0, 90.0, 135.0):
        cases.append(("displaced %+.0f deg" % d, d, -d if flip else d))

    ok = True
    for name, true_deg, expected in cases:
        phi = a + math.radians(true_deg)
        got = compute_angle(math.sin(phi), math.cos(phi), 0.0, 0.0, a, flip=flip)
        if abs(got - expected) > tol:
            ok = False
            print("[Self-Check] FAIL %s: got %+.4f, expected %+.4f" % (name, got, expected), flush=True)
    if not ok:
        print("[Self-Check] compute_angle 自检失败：标定旋转不一致，"
              "所有导出角度都会有系统性偏差。", flush=True)
    return ok


def get_best_detection(results, target_class_ids):
    """取置信度最高且类别匹配的检测框中心。"""
    if not results or len(results) == 0:
        return None
    r = results[0]
    if r.boxes is None or len(r.boxes) == 0:
        return None
    boxes_xywh = r.boxes.xywh.cpu().numpy()
    boxes_conf = r.boxes.conf.cpu().numpy()
    boxes_cls = r.boxes.cls.cpu().numpy()

    best_conf, best_cx, best_cy = -1.0, None, None
    for box, conf, cls in zip(boxes_xywh, boxes_conf, boxes_cls):
        if target_class_ids is not None and int(cls) not in target_class_ids:
            continue
        if conf > best_conf:
            best_conf = float(conf)
            best_cx = float(box[0])
            best_cy = float(box[1])
    return (best_cx, best_cy, best_conf) if best_cx is not None else None


# ==============================================================================
#  标注
# ==============================================================================

def draw_annotation(frame, det, origin_px, vertical_angle_rad,
                    angle_deg, frame_time, traj_history,
                    results=None, det_lost=False):
    """画检测框、轨迹、角度文字与坐标轴。"""
    ann = frame.copy()

    if results is not None and len(results) > 0 and hasattr(results[0], "plot"):
        ann = results[0].plot(img=ann, conf=True, labels=True, line_width=2, font_size=0.5)

    ox, oy = origin_px
    axis_len = int(min(frame.shape[:2]) * 0.15)
    cos_a = math.cos(vertical_angle_rad)
    sin_a = math.sin(vertical_angle_rad)
    ax_end = (int(ox + sin_a * axis_len), int(oy + cos_a * axis_len))
    cv2.arrowedLine(ann, origin_px, ax_end, AXIS_COLOR, 2, tipLength=0.2)
    _put_ascii_text(ann, "Y(down)", (ax_end[0] + 5, ax_end[1]), scale=0.5,
                    color=AXIS_COLOR, thickness=1)

    cv2.circle(ann, origin_px, 7, ORIGIN_COLOR, -1)
    cv2.circle(ann, origin_px, 11, ORIGIN_COLOR, 2)

    if det is not None:
        cx, cy = int(det[0]), int(det[1])
        if DRAW_TRAJECTORY:
            traj_history.append((cx, cy))
            if len(traj_history) > TRAJECTORY_LEN:
                traj_history.pop(0)
            for i in range(1, len(traj_history)):
                alpha = i / len(traj_history)
                c = (int(CENTER_COLOR[0] * alpha), int(CENTER_COLOR[1] * alpha),
                     int(CENTER_COLOR[2] * alpha))
                cv2.line(ann, traj_history[i - 1], traj_history[i], c, 2)
        cv2.line(ann, origin_px, (cx, cy), (200, 200, 200), 1, cv2.LINE_AA)
        cv2.circle(ann, (cx, cy), 5, CENTER_COLOR, -1)

    h = frame.shape[0]
    if det_lost:
        _put_ascii_text(ann, "DETECTION LOST", (10, h - 48), scale=0.8,
                        color=(0, 0, 255), thickness=2)
    else:
        _put_ascii_text(ann, "theta = %.2f deg" % angle_deg, (10, h - 48),
                        scale=0.8, color=TEXT_COLOR, thickness=2)
    _put_ascii_text(ann, "t = %.3f s" % frame_time, (10, h - 20),
                    scale=0.8, color=TEXT_COLOR, thickness=2)
    return ann


def auto_sample_step(n_frames, real_fps, max_model_frames):
    """自动采样步长：只在模型调用数超过预算时才降采样，短片段逐帧。"""
    if n_frames <= 0:
        return 1
    max_model_frames = max(1, int(max_model_frames))
    step = max(1, int(math.ceil(n_frames / float(max_model_frames))))
    n_calls = (n_frames + step - 1) // step
    print("[Sampling] 分析范围 %d 帧 | 真实帧率 %.2f | 步长 %d | 模型调用 ~%d"
          % (n_frames, real_fps, step, n_calls), flush=True)
    return step


def _setup_matplotlib_chinese():
    return setup_matplotlib_chinese()


# ==============================================================================
#  主流程（去掉交互：起始帧来自网页，标定来自网页点击）
# ==============================================================================

def process_cizuni(video_path, output_dir, model_path, calibration,
                   skip_start_sec=0.0, skip_end_sec=0.0,
                   start_frame=None,
                   sample_step=SAMPLE_STEP, max_model_frames=MAX_MODEL_FRAMES,
                   conf_threshold=CONF_THRESHOLD, iou_threshold=IOU_THRESHOLD,
                   gaussian_sigma=GAUSSIAN_SIGMA, target_class_ids=TARGET_CLASS_IDS,
                   slow_motion_factor=SLOW_MOTION_FACTOR,
                   flip_angle=False, gen_video=True, output_video_max_side=0,
                   model=None, progress_callback=None):
    """
    calibration: {
        "vertical_ref_1": [x, y],   # 竖直参考点 1（上）
        "vertical_ref_2": [x, y],   # 竖直参考点 2（下）
        "pivot":          [x, y],   # 转轴原点
    }  均为原图坐标。
    start_frame: 网页"计时第一帧"，该帧即 t=0；为空则回退 skip_start_sec。
    """
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # ---- 标定 ----
    v_pt1 = (int(calibration["vertical_ref_1"][0]), int(calibration["vertical_ref_1"][1]))
    v_pt2 = (int(calibration["vertical_ref_2"][0]), int(calibration["vertical_ref_2"][1]))
    origin_px = (int(calibration["pivot"][0]), int(calibration["pivot"][1]))
    dx, dy = v_pt2[0] - v_pt1[0], v_pt2[1] - v_pt1[1]
    if dx == 0 and dy == 0:
        raise RuntimeError("竖直参考两点重合，请重新标定。")
    vertical_angle_rad = math.atan2(dx, dy)
    ox, oy = origin_px
    print("[cizuni] 标定: ref1=%s ref2=%s origin=%s" % (v_pt1, v_pt2, origin_px), flush=True)

    # ---- 打开视频 ----
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("无法打开视频: %s" % video_path)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0:
        raise RuntimeError("无法读取视频帧率（fps=%s）。" % fps)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if total_frames < 2:
        raise RuntimeError("视频只有 %d 帧。" % total_frames)

    # ---- 时间基准 ----
    slow, slow_src = resolve_slow_motion_factor(video_path, fps, slow_motion_factor)
    real_fps = fps * slow
    print("[cizuni] 时间基准: %gx (%s)  真实采样率 %.2f Hz"
          % (slow, slow_src, real_fps), flush=True)

    if not verify_compute_angle(vertical_angle_rad, flip=flip_angle):
        raise RuntimeError("角度变换自检失败，已中止（标定旋转不一致）。")

    # ---- 起始帧 / 采样步长 ----
    if start_frame is not None:
        f_start = max(0, min(int(start_frame), max(0, total_frames - 1)))
    else:
        f_start = int(skip_start_sec * real_fps)
        f_start = max(0, min(f_start, max(0, total_frames - 1)))
    end_frame = total_frames - int(skip_end_sec * real_fps)
    end_frame = max(f_start + 1, min(end_frame, total_frames))
    n_analyze = end_frame - f_start
    if sample_step is None:
        sample_step = auto_sample_step(n_analyze, real_fps, max_model_frames)
    sample_step = max(1, int(sample_step))
    print("[cizuni] 起始帧=%d（t=0）  分析 %d 帧，步长 %d" % (f_start, n_analyze, sample_step), flush=True)

    if model is None:
        model = YOLO(model_path)

    # ---- 输出视频（与底层一致：原尺寸，探测可用编码）----
    out_video_path = os.path.join(output_dir, VIDEO_FILENAME)
    out_w, out_h = orig_w, orig_h
    out_scale = 1.0
    if output_video_max_side and max(orig_w, orig_h) > int(output_video_max_side):
        out_scale = float(output_video_max_side) / max(orig_w, orig_h)
        out_w = int(round(orig_w * out_scale))
        out_h = int(round(orig_h * out_scale))
    video_out = None
    if gen_video:
        for codec_str in ["avc1", "H264", "X264", "mp4v"]:
            fourcc = cv2.VideoWriter_fourcc(*codec_str)
            test_writer = cv2.VideoWriter(out_video_path, fourcc, fps, (out_w, out_h))
            if test_writer.isOpened():
                video_out = test_writer
                print("[cizuni] 标注视频编码: %s" % codec_str, flush=True)
                break
            test_writer.release()
        if video_out is None:
            print("[WARN] 找不到可用编码，标注视频将不会生成。", flush=True)

    # ---- 逐帧 ----
    cap.set(cv2.CAP_PROP_POS_FRAMES, f_start)
    times_raw, angles_raw = [], []
    traj_history = []
    last_valid_angle = 0.0
    prev_det = None
    det_lost = False
    processed = 0
    n_sampled = 0
    frame_idx = f_start

    while True:
        ret, frame = cap.read()
        if not ret or frame_idx >= end_frame:
            break

        rel_idx = frame_idx - f_start
        is_sample = (rel_idx % sample_step == 0)
        frame_time = rel_idx / real_fps          # 真实秒，起点帧为 0
        det, results_obj = None, None

        if is_sample:
            n_sampled += 1
            results_obj = model.predict(frame, conf=conf_threshold, iou=iou_threshold, verbose=False)
            det = get_best_detection(results_obj, target_class_ids)
            if det is not None:
                angle = compute_angle(det[0], det[1], ox, oy, vertical_angle_rad, flip=flip_angle)
                times_raw.append(frame_time)
                angles_raw.append(angle)
                prev_det = det
                last_valid_angle = angle
                det_lost = False
                processed += 1
            else:
                det = prev_det
                det_lost = True
        else:
            det = prev_det

        if video_out is not None:
            if out_scale < 1.0:
                small = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
                ann = draw_annotation(small, det, (int(ox * out_scale), int(oy * out_scale)),
                                      vertical_angle_rad, last_valid_angle, frame_time,
                                      [], results_obj if is_sample else None, det_lost=det_lost)
            else:
                ann = draw_annotation(frame, det, origin_px, vertical_angle_rad,
                                      last_valid_angle, frame_time, traj_history,
                                      results_obj if is_sample else None, det_lost=det_lost)
            video_out.write(ann)

        frame_idx += 1
        if progress_callback and rel_idx % 50 == 0:
            progress_callback(rel_idx, n_analyze)

    cap.release()
    if video_out is not None:
        video_out.release()
    if progress_callback:
        progress_callback(n_analyze, n_analyze)

    if len(times_raw) < 2:
        raise RuntimeError(
            "有效检测帧不足（仅 %d 帧，conf=%.2f）。请检查摆球是否完整在画面内、光照是否充足。"
            % (len(times_raw), conf_threshold))

    times_raw = np.array(times_raw)
    angles_raw = np.array(angles_raw)
    if n_sampled:
        print("[cizuni] 采样帧 %d | 检出 %d (%.1f%%) | 丢失 %d"
              % (n_sampled, len(times_raw), 100.0 * len(times_raw) / n_sampled,
                 n_sampled - len(times_raw)), flush=True)

    # ---- 高斯滤波（与底层一致：直接对原始序列做）----
    angles_filtered = gaussian_filter1d(angles_raw, sigma=gaussian_sigma)

    # ---- CSV（列名/文件名与底层一致）----
    csv_path = os.path.join(output_dir, CSV_FILENAME)
    df = pd.DataFrame({
        "time(s)":        np.round(times_raw, 6),
        "angle(deg)":     np.round(angles_filtered, 6),
        "angle_raw(deg)": np.round(angles_raw, 6),
    })
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    # ---- 图（双面板 + 中文字体）----
    has_cn = _setup_matplotlib_chinese()
    if has_cn:
        title_main = "磁阻尼复摆 — 摆角时间图 (YOLOv8)"
        label_raw = "原始摆角"
        label_filt = "高斯滤波 (σ=%s)" % gaussian_sigma
        label_filt2 = "滤波后摆角"
        xlabel, ylabel = "时间 t (s)", "摆角 θ (°)"
    else:
        title_main = "Magnetic Damping Pendulum - Angle vs Time (YOLOv8)"
        label_raw = "Raw angle"
        label_filt = "Gaussian filtered (sigma=%s)" % gaussian_sigma
        label_filt2 = "Filtered angle"
        xlabel, ylabel = "Time t (s)", "Angle theta (deg)"

    plot_path = os.path.join(output_dir, PLOT_FILENAME)
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(title_main, fontsize=14, fontweight="bold")

    axes[0].plot(times_raw, angles_raw, color="#5b8dd9", lw=0.8, alpha=0.6, label=label_raw)
    axes[0].plot(times_raw, angles_filtered, color="#e05c5c", lw=1.8, label=label_filt)
    axes[0].set_ylabel(ylabel)
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[0].axhline(0, color="gray", lw=0.8, ls="--")

    axes[1].plot(times_raw, angles_filtered, color="#e05c5c", lw=1.8, label=label_filt2)
    axes[1].fill_between(times_raw, angles_filtered, 0, alpha=0.15, color="#e05c5c")
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel(ylabel)
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    axes[1].axhline(0, color="gray", lw=0.8, ls="--")

    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    print("[cizuni] 完成: %d 点 | 时间 %.3f~%.3f s | 角度 %.2f~%.2f deg"
          % (len(df), times_raw[0], times_raw[-1],
             angles_filtered.min(), angles_filtered.max()), flush=True)

    return {
        "csv_path": csv_path,
        "plot_path": plot_path,
        "video_path": out_video_path if video_out is not None else None,
        "num_points": len(df),
        "slow_motion_factor": slow,
        "slow_motion_source": slow_src,
        "real_fps": real_fps,
        "start_frame": int(f_start),
        "sample_step": int(sample_step),
        "time_range_s": [float(times_raw[0]), float(times_raw[-1])],
        "angle_range_deg": [float(angles_filtered.min()), float(angles_filtered.max())],
    }
