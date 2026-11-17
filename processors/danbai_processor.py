# -*- coding: utf-8 -*-
"""
单摆 (Simple Pendulum) 处理器
==============================
对齐 yolov8/src/单摆/danbai_ultimate_version.py + observation_pipeline.py：

  · 标定 3 点：center（悬点）、vertical_start（竖直起点）、vertical_end（竖直终点）
  · 网页选定起始帧 start_frame（t=0，不再按固定秒数跳过开头、不再按目标周期早停截断）
  · 按真实解码 PTS 时间轴 (decoder_time - t0) / slow_motion_factor 还原物理时间
  · 保留真实逐帧观测角度（不自动归零、不对 raw 做平滑、不填补漏检）
  · 输出 <basename>_angle_time.csv / <basename>_frame_audit.csv / 提取清单.json /
         <basename>_angle_time.png / annotated_video.mp4
"""

from dataclasses import dataclass
import math
import os
from pathlib import Path
from typing import Optional, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .common import normalize, setup_matplotlib_chinese
from .observation_pipeline import extract


@dataclass
class DanbaiConfig:
    model_path: str = ""
    video_path: str = ""
    output_dir: str = ""
    calibration_path: str = ""
    timestamp_mode: str = "decoder_pts"
    confirmed_fps: Optional[float] = None
    cfr_confirmed: bool = False
    allow_fps_fallback: bool = True
    allow_nonempty_output: bool = True
    time_calibration_confirmed: bool = False
    release_end_s: Optional[float] = None
    release_id: Optional[str] = None
    configuration_id: Optional[str] = None
    bob_class_id: Optional[int] = None
    max_center_jump_px: Optional[float] = None
    tracking_reset_after_misses: int = 5

    # YOLO 推理配置
    conf_thres: float = 0.25
    iou_thres: float = 0.45
    imgsz: int = 1280
    max_det: int = 5
    device: str = "auto"
    use_tracking_like_selection: bool = True

    # 时间 / 帧选择
    start_frame_index: Optional[int] = None
    end_frame_index: Optional[int] = None
    skip_initial_seconds: float = 0.0
    slow_motion_factor: float = 1.0
    flip_angle: bool = False
    target_cycles: Optional[int] = None  # 仅兼容旧参数，不再用于早停截断

    # 诊断平滑与绘图（仅用于展示/预览，绝不覆盖原始 raw_angle_deg）
    savgol_window: int = 15
    savgol_polyorder: int = 3
    save_angle_time_plot: bool = True
    plot_figsize: Tuple[float, float] = (12.0, 5.0)
    plot_dpi: int = 180
    output_video_max_side: int = 1280


def resolve_device(cfg: DanbaiConfig):
    if cfg.device == "auto":
        try:
            import torch
            cfg.device = "0" if torch.cuda.is_available() else "cpu"
        except Exception:
            cfg.device = "cpu"
    return cfg.device


def _normalize_danbai_calibration(calibration: dict) -> dict:
    cal = dict(calibration)
    if "center" not in cal:
        for alt in ("origin", "pivot"):
            if alt in cal:
                cal["center"] = cal[alt]
                break
    if "vertical_start" not in cal and "vertical_ref_1" in cal:
        cal["vertical_start"] = cal["vertical_ref_1"]
    if "vertical_end" not in cal and "vertical_ref_2" in cal:
        cal["vertical_end"] = cal["vertical_ref_2"]
    return cal


def make_pendulum_frame_annotator(vunit_override=None):
    """生成单摆/磁阻尼摆共用的标注视频帧回调（坐标系 + 检测框 + θ/t 读数）。"""

    def _annotate(disp, row, chosen, t_disp, calibration, scale, frame_idx, total_frames):
        ox = int(round(float(calibration["center"][0]) * scale))
        oy = int(round(float(calibration["center"][1]) * scale))
        if vunit_override is not None:
            vx, vy = float(vunit_override[0]), float(vunit_override[1])
        else:
            vx, vy = normalize(
                np.asarray(calibration["vertical_end"], dtype=float)
                - np.asarray(calibration["vertical_start"], dtype=float)
            )

        h, w = disp.shape[:2]
        axis_len = int(max(60, min(w, h) * 0.35))
        ax_end = (int(round(ox + vx * axis_len)), int(round(oy + vy * axis_len)))
        cv2.arrowedLine(disp, (ox, oy), ax_end, (255, 229, 0), 2, cv2.LINE_AA, tipLength=0.14)
        cv2.circle(disp, (ox, oy), 6, (59, 235, 255), -1, cv2.LINE_AA)
        cv2.circle(disp, (ox, oy), 9, (255, 255, 255), 1, cv2.LINE_AA)

        if row.get("observed") and chosen is not None and math.isfinite(row.get("raw_angle_deg", float("nan"))):
            x1, y1, x2, y2 = [int(round(v * scale)) for v in chosen["box"]]
            cx = int(round(float(chosen["center"][0]) * scale))
            cy = int(round(float(chosen["center"][1]) * scale))
            cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 196, 255), 2, cv2.LINE_AA)
            cv2.line(disp, (ox, oy), (cx, cy), (235, 235, 235), 2, cv2.LINE_AA)
            cv2.circle(disp, (cx, cy), 5, (0, 196, 255), -1, cv2.LINE_AA)
            ang = float(row["raw_angle_deg"])
            msg = f"theta = {ang:+.2f} deg"
            col = (3, 255, 118)
        else:
            msg = "NO DETECTION"
            col = (60, 60, 255)

        t_msg = f"t = {t_disp:.4f} s   frame {frame_idx}"
        for text, org, c in ((msg, (14, h - 42), col), (t_msg, (14, h - 16), (245, 245, 245))):
            cv2.putText(disp, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(disp, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.72, c, 2, cv2.LINE_AA)
        return disp

    return _annotate


def plot_angle_time(result_dict, save_path, cfg, title):
    setup_matplotlib_chinese()
    plt.figure(figsize=cfg.plot_figsize)
    plt.plot(result_dict["times_seg"], result_dict["raw_seg"], label="Raw Angle", linewidth=0.9, alpha=0.55)
    plt.plot(result_dict["times_seg"], result_dict["filt_seg"], label="Diagnostic Filtered", linewidth=1.6)
    plt.scatter(result_dict["sample_t"], result_dict["sample_raw"], s=8, alpha=0.35, label="Observed Frames")
    plt.axhline(0.0, linestyle="--", linewidth=1.0, color="gray")
    plt.xlabel("Time (s)")
    plt.ylabel("Angle (deg)")
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=cfg.plot_dpi, bbox_inches="tight")
    plt.close()


def process_danbai(
    video_path,
    output_dir,
    model_path,
    calibration,
    start_frame=None,
    end_frame=None,
    skip_initial_seconds=0.0,
    slow_motion_factor=1.0,
    target_cycles=None,
    flip_angle=False,
    gen_video=True,
    model=None,
    progress_callback=None,
    cfg=None,
):
    """单摆处理入口：基于选定起始帧逐帧提取真实角度与时间戳。"""
    base_cfg = DanbaiConfig()
    if cfg is not None:
        for k, v in vars(cfg).items():
            if hasattr(base_cfg, k):
                setattr(base_cfg, k, v)
    cfg = base_cfg

    cfg.video_path = str(video_path)
    cfg.output_dir = str(output_dir)
    cfg.model_path = str(model_path or "")
    cfg.flip_angle = bool(flip_angle)
    cfg.target_cycles = target_cycles
    cfg.slow_motion_factor = float(slow_motion_factor) if slow_motion_factor is not None else 1.0

    if start_frame is not None:
        cfg.start_frame_index = max(0, int(start_frame))
        cfg.end_frame_index = max(cfg.start_frame_index + 2, int(end_frame)) if end_frame is not None else None
        cfg.skip_initial_seconds = 0.0
    elif float(skip_initial_seconds or 0.0) > 0.0:
        cfg.start_frame_index = None
        cfg.end_frame_index = int(end_frame) if end_frame is not None else None
        cfg.skip_initial_seconds = float(skip_initial_seconds)
    else:
        cfg.start_frame_index = 0
        cfg.end_frame_index = int(end_frame) if end_frame is not None else None
        cfg.skip_initial_seconds = 0.0

    resolve_device(cfg)
    cal = _normalize_danbai_calibration(calibration)
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
        video_filename="annotated_video.mp4",
        extra_metadata={"experiment": "simple_pendulum"},
    )
    return res
