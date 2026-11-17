# -*- coding: utf-8 -*-
"""
磁阻尼复摆 (Magnetic Damping Pendulum) 处理器
==============================================
对齐 yolov8/src/磁阻尼摆/extract_observations.py + observation_pipeline.py：

  · 兼容两种 3 点标定键名：
      网页前端: vertical_ref_1, vertical_ref_2, pivot
      管线内部: vertical_start, vertical_end, center
  · 保留慢放倍数 "auto" 读取 MP4 `com.android.capture.fps` 与 compute_angle/verify_compute_angle 自检
  · 基于网页选定起始帧 start_frame（t=0）逐帧提取真实角度（不采样跳帧、不平滑覆盖 raw、不插值补帧）
  · 输出 pendulum_angle.csv（列名 time_s, raw_angle_deg, filtered_angle_deg, ...，仅含有效观测帧，
    供 symbolic_regression.py 直接读取）、pendulum_angle_frame_audit.csv、提取清单.json、
    angle_time_plot.png 与 H.264 annotated_video.mp4
"""

import math
import os
import struct

import cv2
import numpy as np
import pandas as pd

from .danbai_processor import (
    DanbaiConfig,
    make_pendulum_frame_annotator,
    plot_angle_time,
    resolve_device,
)
from .observation_pipeline import extract


CSV_FILENAME = "pendulum_angle.csv"
PLOT_FILENAME = "angle_time_plot.png"
VIDEO_FILENAME = "annotated_video.mp4"


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
#  角度与自检
# ==============================================================================

def compute_angle(px, py, origin_px, origin_py, vertical_angle_rad, flip=False):
    """
    摆角（度）。原点在转轴，+Y 轴为竖直向下，顺时针为正。

    用 +sin/+cos 旋转，才能把"点在标定的竖直参考线上"映射为 0°。
    写成 sin(-a)/cos(-a) 会额外叠加恒定的 2a 偏差（自检会抓到）。
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
    """自检：标定参考线上的点必须读 0°；偏离 d 度的点必须读 ±d。"""
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
        print("[Self-Check] compute_angle 自检失败：标定旋转不一致。", flush=True)
    return ok


def _normalize_cizuni_calibration(calibration: dict) -> dict:
    cal = dict(calibration)
    if "vertical_start" not in cal and "vertical_ref_1" in cal:
        cal["vertical_start"] = cal["vertical_ref_1"]
    if "vertical_end" not in cal and "vertical_ref_2" in cal:
        cal["vertical_end"] = cal["vertical_ref_2"]
    if "center" not in cal:
        for alt in ("pivot", "origin"):
            if alt in cal:
                cal["center"] = cal[alt]
                break
    return cal


# ==============================================================================
#  主流程
# ==============================================================================

def process_cizuni(
    video_path,
    output_dir,
    model_path,
    calibration,
    start_frame=None,
    end_frame=None,
    skip_start_sec=0.0,
    skip_end_sec=0.0,
    slow_motion_factor="auto",
    flip_angle=False,
    conf_threshold=0.25,
    iou_threshold=0.45,
    imgsz=1280,
    max_det=5,
    target_class_ids=None,
    gen_video=True,
    model=None,
    progress_callback=None,
    **kwargs,
):
    """
    磁阻尼摆主处理流程：
      start_frame: 计时第一帧（帧号，t=0）；为 None 时若 skip_start_sec>0 则兼容旧行为，否则从第 0 帧开始。
    """
    cal = _normalize_cizuni_calibration(calibration)
    dx = float(cal["vertical_end"][0]) - float(cal["vertical_start"][0])
    dy = float(cal["vertical_end"][1]) - float(cal["vertical_start"][1])
    if dx == 0.0 and dy == 0.0:
        raise RuntimeError("竖直参考两点重合，请重新标定。")
    vertical_angle_rad = math.atan2(dx, dy)
    if not verify_compute_angle(vertical_angle_rad, flip=bool(flip_angle)):
        raise RuntimeError("角度变换自检失败，已中止（标定旋转不一致）。")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("无法打开视频: %s" % video_path)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    cap.release()
    if fps <= 0:
        raise RuntimeError("无法读取视频帧率（fps=%s）。" % fps)

    slow, slow_src = resolve_slow_motion_factor(video_path, fps, slow_motion_factor)
    print("[cizuni] 时间基准: %gx (%s)  真实采样率 %.2f Hz" % (slow, slow_src, fps * slow), flush=True)

    cfg = DanbaiConfig()
    cfg.video_path = str(video_path)
    cfg.output_dir = str(output_dir)
    cfg.model_path = str(model_path or "")
    cfg.conf_thres = float(conf_threshold)
    cfg.iou_thres = float(iou_threshold)
    cfg.imgsz = int(imgsz)
    cfg.max_det = int(max_det)
    cfg.slow_motion_factor = float(slow)
    cfg.flip_angle = bool(flip_angle)
    if target_class_ids is not None and len(target_class_ids) == 1:
        cfg.bob_class_id = int(list(target_class_ids)[0])

    if start_frame is not None:
        cfg.start_frame_index = max(0, int(start_frame))
        cfg.end_frame_index = max(cfg.start_frame_index + 2, int(end_frame)) if end_frame is not None else None
        cfg.skip_initial_seconds = 0.0
    elif float(skip_start_sec or 0.0) > 0.0:
        cfg.start_frame_index = None
        cfg.end_frame_index = int(end_frame) if end_frame is not None else None
        cfg.skip_initial_seconds = float(skip_start_sec)
    else:
        cfg.start_frame_index = 0
        cfg.end_frame_index = int(end_frame) if end_frame is not None else None
        cfg.skip_initial_seconds = 0.0

    resolve_device(cfg)
    annotator = make_pendulum_frame_annotator() if gen_video else None

    res = extract(
        video_path=video_path,
        output_dir=output_dir,
        model_path=model_path,
        calibration=cal,
        cfg=cfg,
        model=model,
        progress_callback=progress_callback,
        plotter=plot_angle_time,
        frame_annotator=annotator,
        write_video=bool(gen_video),
        video_filename=VIDEO_FILENAME,
        csv_filename=CSV_FILENAME,
        plot_filename=PLOT_FILENAME,
        csv_prefix="pendulum_angle",
        audit_suffix="_frame_audit.csv",
        extra_metadata={
            "experiment": "magnetic_pendulum",
            "slow_motion_source": slow_src,
        },
    )

    df = pd.read_csv(res["csv_path"])
    raw_vals = pd.to_numeric(df["raw_angle_deg"], errors="coerce").dropna()
    res["slow_motion_source"] = slow_src
    res["sample_step"] = 1
    res["angle_range_deg"] = [float(raw_vals.min()), float(raw_vals.max())] if len(raw_vals) else [0.0, 0.0]
    return res
