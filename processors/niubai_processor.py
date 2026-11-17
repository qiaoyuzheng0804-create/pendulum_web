# -*- coding: utf-8 -*-
"""
扭摆 (Torsional Pendulum) 处理器
================================
完全对齐底层 analyze_pendulum_fixed.py：

  · 标定 3 点：① 原点  ② 竖直方向起点  ③ 竖直方向终点（网页点击传入，原图坐标）
  · 摆角 = (框心-原点) 与 (②->③) 的有符号夹角（signed_angle_deg）
  · 逐帧：解缠绕 -> 离群点剔除 -> 补缝 -> Savitzky-Golay 自适应降噪（含防过平滑自检）
  · 包络寻峰 find_cycle_peaks -> 分段阻尼 decay_report -> FFT 周期 estimate_period_fft
  · 输出 angle_time_data.csv / angle_time_plot.png / annotated_video.mp4 / run_meta.json

与底层脚本的唯一差别是"去掉交互"：
  1. 手工标记窗口 -> 网页点击的 3 个标定点；
  2. 起始帧 -> 网页"计时第一帧"传入的帧号 start_frame（该帧即 t=0）。

时间基准与底层一致（注意是【除以】慢放倍数）：
    real_fps = video_fps * slow_motion_factor
    t = (帧号 - start_frame) / real_fps
底层脚本特别注释过：写成 fps/slow 或 fps*slow 会让时间轴整体差该倍数。
"""

import csv
import json
import math
import os
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, butter, filtfilt
from scipy.ndimage import median_filter
from ultralytics import YOLO

from .common import H264VideoWriter as _AnnotatedVideoWriter, setup_matplotlib_chinese, video_readable
from . import torsion_signal

setup_matplotlib_chinese()


# ---------- 默认配置（与底层 CONFIG 对齐） ----------
DEFAULT_CONFIG = {
    "imgsz": 672,
    "yolo_conf": 0.40,
    "yolo_iou": 0.45,
    "target_class": 0,
    "video_orientation_auto": True,
    "denoise_method": "savgol",          # "savgol" | "butter" | "median" | "none"
    "savgol_window_ratio": 0.20,
    "savgol_polyorder": 3,
    "default_savgol_window": 9,
    "savgol_max_resid_signal_frac": 0.10,
    "savgol_max_span_frac": 0.05,
    "butter_cutoff_ratio": 3.0,
    "butter_order": 4,
    "median_window": 5,
    "max_gap_frames": 30,
    "max_jump_deg": 60.0,
    "output_video_max_side": 1280,
}


def hr(title=""):
    print("-" * 68, flush=True)
    if title:
        print(title, flush=True)


def fmt(v, nd=4, nan="--"):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return nan
    if not np.isfinite(f):
        return nan
    return ("%%.%df" % nd) % f


def draw_text(img, txt, org, scale=0.6, color=(0, 255, 255), thick=1):
    """带黑色描边的 putText。"""
    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


# ================================================================
#  标注视频上的坐标系叠加
# ================================================================
# 这套配色/几何与网页标定画布（templates/index.html 的 drawNiubaiFrame）刻意保持一致：
#   Y 轴 = 标定 ②->③ 定义的"竖直方向"，是摆角 θ=0 的基准轴（主参考，亮青）
#   X 轴 = 把 Y 轴在图像坐标里转 +90°，即 θ=+90° 的方向（亮橙）
#   原点 = 标定 ① 的转轴点（亮黄）
# 图像坐标 x 右 / y 下；每个元素先描黑边再上亮色，任何背景上都醒目。
COL_Y = (255, 229, 0)        # #00E5FF 亮青 —— Y 轴（θ=0 基准）
COL_X = (0, 140, 255)        # #FF8C00 亮橙 —— X 轴（θ=+90°）
COL_ORIGIN = (59, 235, 255)  # #FFEB3B 亮黄 —— 原点
COL_ANGLE = (3, 255, 118)    # #76FF03 亮绿 —— 角度弧与 θ 读数
COL_BOX = (0, 196, 255)      # #FFC400 亮黄橙 —— 检测框
COL_RADIUS = (235, 235, 235)  # 原点->框心 的半径连线
SHADOW = (0, 0, 0)


def _shade(color, k):
    """同色暗一档（反方向虚线段用）。"""
    return tuple(int(max(0, min(255, round(c * k)))) for c in color)


def _axis_head_len(img):
    """箭头长度：跟着画面尺寸走，绝不用 OpenCV arrowedLine 的"按线长比例"（长轴会画出巨大的箭头）。"""
    return int(max(9, min(18, round(min(img.shape[:2]) * 0.022))))


def _axis_line(img, p1, p2, color, thickness=3, head=None):
    """轴线 = 细杆 + 紧凑实心箭头（先描黑边再上亮色）。"""
    if head is None:
        head = _axis_head_len(img)
    cv2.line(img, p1, p2, SHADOW, thickness + 3, cv2.LINE_AA)
    cv2.line(img, p1, p2, color, thickness, cv2.LINE_AA)
    dx, dy = float(p2[0] - p1[0]), float(p2[1] - p1[1])
    length = math.hypot(dx, dy)
    if length <= head * 0.5:
        return
    ux, uy = dx / length, dy / length
    hw = head * 0.42                          # 箭头半宽
    bx, by = p2[0] - ux * head, p2[1] - uy * head
    outline = np.array([[p2[0], p2[1]],
                        [bx + (-uy) * (hw + 2.5), by + ux * (hw + 2.5)],
                        [bx - (-uy) * (hw + 2.5), by - ux * (hw + 2.5)]], np.int32)
    tri = np.array([[p2[0], p2[1]],
                    [bx + (-uy) * hw, by + ux * hw],
                    [bx - (-uy) * hw, by - ux * hw]], np.int32)
    cv2.fillConvexPoly(img, outline, SHADOW, cv2.LINE_AA)
    cv2.fillConvexPoly(img, tri, color, cv2.LINE_AA)


def _dashed_line(img, p1, p2, color, thickness=2, dash=22, gap=14):
    x1, y1 = p1
    x2, y2 = p2
    length = math.hypot(x2 - x1, y2 - y1)
    if length < 1:
        return
    ux, uy = (x2 - x1) / length, (y2 - y1) / length
    s = 0.0
    while s < length:
        e = min(s + dash, length)
        a = (int(round(x1 + ux * s)), int(round(y1 + uy * s)))
        b = (int(round(x1 + ux * e)), int(round(y1 + uy * e)))
        cv2.line(img, a, b, SHADOW, thickness + 2, cv2.LINE_AA)
        cv2.line(img, a, b, color, thickness, cv2.LINE_AA)
        s = e + gap


def _angle_arc(img, center, from_deg, to_deg, radius, color, thickness=3):
    """圆心到两个方向的短弧（angles 用图像坐标角度，x 右 y 下）。"""
    a0 = from_deg % 360.0
    delta = ((to_deg - from_deg + 180.0) % 360.0) - 180.0
    if abs(delta) < 0.5:
        return
    s, e = (a0, a0 + delta) if delta > 0 else (a0 + delta, a0)
    axes = (int(radius), int(radius))
    cv2.ellipse(img, center, axes, 0.0, s, e, SHADOW, thickness + 3, cv2.LINE_AA)
    cv2.ellipse(img, center, axes, 0.0, s, e, color, thickness, cv2.LINE_AA)


def _label_at(img, txt, pos, color, scale=0.6, thick=1, dx=10, dy=-10):
    """把标签贴在 pos 附近，自动收进画面内。"""
    h, w = img.shape[:2]
    (tw, th), _base = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    x = int(min(max(6, pos[0] + dx), max(6, w - tw - 8)))
    y = int(min(max(th + 6, pos[1] + dy), h - 8))
    draw_text(img, txt, (x, y), scale, color, thick)


def _axis_label(img, txt, tip, u, other_u, color, scale, thick=2, head=0):
    """轴端标签：贴着箭头外侧放（沿轴再外推一个箭头长度），并向"背离另一条轴"的一侧让开，
    保证文字既不压轴线也不压箭头。"""
    px, py = -u[1], u[0]                      # 轴线法线
    if px * other_u[0] + py * other_u[1] > 0:  # 法线朝着另一条轴 -> 取反向
        px, py = -px, -py
    d = 15.0 * scale + 8.0
    out = head + 7.0 * scale
    pos = (tip[0] + u[0] * out + px * d, tip[1] + u[1] * out + py * d)
    _label_at(img, txt, pos, color, scale, thick, dx=-6, dy=int(6 * scale))


def _fit_axis_len(origin, u, w, h, want, margin=34.0):
    """从原点沿 u 能画多长还留在画面内（原点贴边时别让箭头和轴名跑到画外）。"""
    t = float(want)
    for comp, oo, size in ((u[0], origin[0], w), (u[1], origin[1], h)):
        if comp > 1e-9:
            t = min(t, (size - margin - oo) / comp)
        elif comp < -1e-9:
            t = min(t, (margin - oo) / comp)
    return max(24.0, t)


def _hud_panel(img, x, y, w, h, alpha=0.5):
    """半透明黑底：读数压在亮背景（白墙/不锈钢）上也读得清。"""
    x1, y1 = max(0, int(x)), max(0, int(y))
    x2, y2 = min(img.shape[1], int(x + w)), min(img.shape[0], int(y + h))
    if x2 <= x1 or y2 <= y1:
        return
    roi = img[y1:y2, x1:x2]
    cv2.addWeighted(np.full_like(roi, 16), alpha, roi, 1.0 - alpha, 0, roi)


def _corner_box(img, box, color, thickness=2):
    """检测框：细整框 + 四角加粗括号（远看也认得出是框，不被摆杆干扰）。"""
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    cv2.rectangle(img, (x1, y1), (x2, y2), SHADOW, thickness + 3, cv2.LINE_AA)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
    arm = max(10, min(x2 - x1, y2 - y1) // 4)
    for cx, cy, sx, sy in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
        for p1, p2 in (((cx, cy), (cx + sx * arm, cy)), ((cx, cy), (cx, cy + sy * arm))):
            cv2.line(img, p1, p2, SHADOW, thickness + 4, cv2.LINE_AA)
            cv2.line(img, p1, p2, color, thickness + 1, cv2.LINE_AA)


def draw_overlay(frame, origin_px, vunit, angle_deg, t_real, det=None, conf=None,
                 frame_idx=None, frame_total=None, det_lost=False):
    """在画面上叠加"我们建立的坐标系"、检测框与读数，返回画好的帧（就地修改）。

    坐标一律用输出画面像素：origin_px 为原点，vunit 为 ②->③ 单位方向，
    det = (x1, y1, x2, y2, cx, cy) 为检测框；det_lost 表示本帧没检出（沿用上一次的角度）。
    """
    h, w = frame.shape[:2]
    ox, oy = int(round(origin_px[0])), int(round(origin_px[1]))
    vx, vy = float(vunit[0]), float(vunit[1])
    xvx, xvy = -vy, vx                       # θ=+90° 方向（跨过 Y 轴即变号）
    fs = max(0.5, min(1.1, w / 1900.0))      # 字号跟着画面宽走
    Lw = max(90.0, min(w, h) * 0.52)         # 轴线长度（超出画面时按实际能画的距离收短）
    Ly = _fit_axis_len((ox, oy), (vx, vy), w, h, Lw)
    Lx = _fit_axis_len((ox, oy), (xvx, xvy), w, h, Lw)

    # ---- 坐标轴：反方向虚线（暗一档）+ 半径连线在最底层，正方向实线 + 实心小箭头压在上面 ----
    o = (ox, oy)
    _dashed_line(frame, o, (ox - vx * Ly, oy - vy * Ly), _shade(COL_Y, 0.45), 2)
    _dashed_line(frame, o, (ox - xvx * Lx, oy - xvy * Lx), _shade(COL_X, 0.45), 2)
    y_end = (int(round(ox + vx * Ly)), int(round(oy + vy * Ly)))
    x_end = (int(round(ox + xvx * Lx)), int(round(oy + xvy * Lx)))
    if det is not None:
        _dashed_line(frame, o, (int(round(det[4])), int(round(det[5]))), COL_RADIUS, 2, dash=16, gap=12)
    head = _axis_head_len(frame)
    _axis_line(frame, o, y_end, COL_Y, 3, head)
    _axis_line(frame, o, x_end, COL_X, 3, head)
    _axis_label(frame, "Y  theta=0", y_end, (vx, vy), (xvx, xvy), COL_Y, fs * 0.95, 2, head)
    _axis_label(frame, "X  theta=+90", x_end, (xvx, xvy), (vx, vy), COL_X, fs * 0.95, 2, head)

    # ---- 原点：十字 + 黄心白环 ----
    for p1, p2 in (((ox - 16, oy), (ox + 16, oy)), ((ox, oy - 16), (ox, oy + 16))):
        cv2.line(frame, p1, p2, SHADOW, 5, cv2.LINE_AA)
        cv2.line(frame, p1, p2, COL_ORIGIN, 2, cv2.LINE_AA)
    cv2.circle(frame, o, 9, SHADOW, -1, cv2.LINE_AA)
    cv2.circle(frame, o, 7, COL_ORIGIN, -1, cv2.LINE_AA)
    cv2.circle(frame, o, 7, (255, 255, 255), 2, cv2.LINE_AA)

    # ---- 检测框 + 角度弧 ----
    if det is not None:
        x1, y1, x2, y2, cx, cy = [float(v) for v in det]
        _corner_box(frame, (x1, y1, x2, y2), COL_BOX, 2)
        c = (int(round(cx)), int(round(cy)))
        cv2.circle(frame, c, 8, SHADOW, -1, cv2.LINE_AA)
        cv2.circle(frame, c, 5, COL_BOX, -1, cv2.LINE_AA)
        a_y = math.degrees(math.atan2(vy, vx))
        a_r = math.degrees(math.atan2(cy - oy, cx - ox))
        _angle_arc(frame, o, a_y, a_r, int(max(36.0, min(Ly, Lx) * 0.30)), COL_ANGLE, 3)
        # θ 读数跟着框心走（弧附近容易被摆杆压住），HUD 里另有一份大号读数
        _label_at(frame, "%+.2f deg" % angle_deg, c, COL_ANGLE, fs, 2,
                  dx=int(14 * fs), dy=int(30 * fs))

    # ---- 左下角读数面板 ----
    lines = []
    if det_lost:
        lines.append(("NO DETECTION (hold last)", (60, 60, 255), fs * 1.05, 2))
    else:
        lines.append(("theta = %+.2f deg" % angle_deg, COL_ANGLE, fs * 1.15, 2))
    lines.append(("t = %.4f s" % t_real, (245, 245, 245), fs * 0.85, 1))
    if frame_idx is not None and frame_total:
        lines.append(("frame %d / %d    conf %s"
                      % (frame_idx, frame_total - 1,
                         "%.2f" % conf if conf else "--"), (200, 200, 200), fs * 0.75, 1))
    pad = int(12 * fs)
    tw = max(cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, sc, th)[0][0] for t, _c, sc, th in lines)
    th_all = sum(int(34 * fs) + cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, sc, th)[0][1] // 4
                 for t, _c, sc, th in lines)
    panel_h = th_all + pad
    _hud_panel(frame, pad, h - panel_h - pad, tw + pad * 3, panel_h + pad // 2)
    y = h - panel_h - pad + int(26 * fs)
    for txt, col, sc, th in lines:
        draw_text(frame, txt, (pad * 2, y), sc, col, th)
        y += int(34 * fs)
    return frame


# ================================================================
#  角度计算
# ================================================================

def signed_angle_deg(vertical_unit, radius_vec):
    """
    竖直方向 vertical_unit 与半径向量 radius_vec 的有符号夹角（度）。

        theta = atan2( cross(v, r), dot(v, r) )

    图像坐标（x 右、y 下）下：②->③ 自上而下时，偏"右"为负、偏"左"为正。
    （把 ②③ 对调即可翻符号；不影响幅度/周期。）
    """
    vx, vy = float(vertical_unit[0]), float(vertical_unit[1])
    rx, ry = float(radius_vec[0]), float(radius_vec[1])
    return math.degrees(math.atan2(vx * ry - vy * rx, vx * rx + vy * ry))


def unwrap_deg(angles):
    """
    解缠绕：把 atan2 的 ±180° 分支跳变接回连续曲线。

    每步把增量折到 (-180°, 180°] 内再累加；遇到检测缝隙时【重新起算】
    而不是把某个 360° 倍数强加到后面。

    返回 (解缠绕后的序列, 每个连续段的起始索引)。
    """
    a = np.array(angles, dtype=float)
    out = a.copy()
    good = np.isfinite(a)
    idx = np.nonzero(good)[0]
    if len(idx) == 0:
        return out, []

    seg_starts = []
    last = None
    for k in idx:
        if last is None or k != last + 1:
            seg_starts.append(int(k))
            last = k
            continue
        d = out[k] - out[last]
        d = (d + 180.0) % 360.0 - 180.0
        out[k] = out[last] + d
        last = k
    return out, seg_starts


def remove_spikes(angles, max_jump_deg):
    """剔除孤立跳框：某点相对左右邻点都偏出 max_jump_deg 以上就置 NaN。"""
    a = np.array(angles, dtype=float)
    n = len(a)
    if n < 3:
        return a
    killed = 0
    for i in range(1, n - 1):
        if not (np.isfinite(a[i - 1]) and np.isfinite(a[i]) and np.isfinite(a[i + 1])):
            continue
        if abs(a[i] - a[i - 1]) > max_jump_deg and abs(a[i] - a[i + 1]) > max_jump_deg:
            a[i] = np.nan
            killed += 1
    if killed:
        print("[Denoise] 离群点剔除: %d 个（单帧跳变 > %.0f deg）" % (killed, max_jump_deg), flush=True)
    return a


def fill_gaps(angles, max_gap):
    """线性插值补缝；长度 > max_gap 的缝隙留 NaN。返回 (序列, gap_filled 标记, 统计)。"""
    a = np.array(angles, dtype=float)
    n = len(a)
    flag = np.zeros(n, dtype=int)
    x = np.arange(n)
    good = np.isfinite(a)
    stats = {"n_nan_in": int((~good).sum()), "n_filled": 0, "n_left": 0, "max_gap": 0}

    if not good.any():
        stats["n_left"] = n
        return a, flag, stats

    i = 0
    while i < n:
        if good[i]:
            i += 1
            continue
        j = i
        while j < n and not good[j]:
            j += 1
        run = j - i
        stats["max_gap"] = max(stats["max_gap"], run)
        if run <= max_gap and i > 0 and j < n:
            a[i:j] = np.interp(x[i:j], x[good], a[good])
            flag[i:j] = 1
            stats["n_filled"] += run
        else:
            stats["n_left"] += run
        i = j
    return a, flag, stats


def find_cycle_peaks(t, angles, rel_prom=0.25, min_prom_deg=0.03):
    """找每个摆动周期的极大点；支持负值局部极大，不假定平衡角为 0。"""
    return torsion_signal.find_cycle_peaks(t, angles, rel_prom, min_prom_deg)


def _noise_floor_deg(a_raw, a_dn):
    """降噪残差一阶差分的 std，作为逐帧噪声量级估计（度）。"""
    d = np.asarray(a_raw, float) - np.asarray(a_dn, float)
    if not np.isfinite(d).any() or len(d) < 3:
        return 0.0
    v = float(np.nanstd(np.diff(d)))
    return v if np.isfinite(v) else 0.0


def cycle_amplitudes(t, angles, pk):
    """逐周期半峰谷差（上包络插值与谷值对比）。"""
    return torsion_signal.cycle_amplitudes(t, angles, pk)


def decay_report(t, angles, pk, n_blocks=4, min_pts=6):
    """把周期序列分段拟合 ln(A) ~ -beta t，保留负斜率。"""
    return torsion_signal.decay_report(t, angles, pk, n_blocks, min_pts)


def _decay_rate(t, angles, pk):
    """全局指数衰减率 beta（1/s），仅作量级参考。"""
    val = torsion_signal.decay_rate(t, angles, pk)
    return 0.0 if val is None else val


def estimate_period_fft(t, angles, min_cycles=3):
    """FFT 找主频，返回 (周期 s, 主频 Hz, 观测到的周期个数)。"""
    return torsion_signal.estimate_period_fft(t, angles, min_cycles)


def choose_savgol_window(n, t, angles, cfg):
    """Savitzky-Golay 窗口首猜值：周期判据与长度判据取小者。"""
    ratio = cfg["savgol_window_ratio"]
    n = int(n)
    cap = max(5, int(n * float(cfg.get("savgol_max_span_frac", 0.05))))

    if isinstance(ratio, (int, np.integer)) or (isinstance(ratio, float) and ratio >= 3):
        return max(5, min(int(ratio), cap))

    T, _f, ncyc = estimate_period_fft(t, angles)
    if T is not None and ncyc >= 3:
        dt = float(np.median(np.diff(t)))
        w = int(round(float(ratio) * T / max(dt, 1e-12)))
        return max(5, min(w, cap))
    return max(5, min(int(cfg.get("default_savgol_window", 9)), cap))


def _savgol_apply(x, w, po):
    """做一次 savgol；窗口取奇数时往下取（w -= 1），避免超过调用方给的上限。"""
    n = len(x)
    w = int(w)
    if n < po + 3 or w < po + 2:
        return x, 0, False
    if w > n:
        w = n
    if w % 2 == 0:
        w -= 1
    if w <= po:
        return x, 0, False
    return savgol_filter(x, window_length=w, polyorder=po, mode="interp"), w, True


def _resid_signal_frac(x, sm):
    """残差里残留的信号比例：res = x - sm 在 sm 上的归一化投影系数。"""
    res = x - sm
    s = sm - sm.mean()
    ss = float(np.dot(s, s))
    if ss <= 1e-12:
        return 0.0
    return float(np.dot(s, res - res.mean()) / ss)


def _savgol_autotune(x, w0, po, cfg, log=True):
    """防过平滑兜底：窗口只可能变小，从不放大。返回 (平滑结果, 实际窗口, 残差信号比)。"""
    if len(x) < po + 3:
        return x, 0, 0.0
    max_frac = float(cfg.get("savgol_max_resid_signal_frac", 0.10))

    cap = max(5, int(len(x) * float(cfg.get("savgol_max_span_frac", 0.05))))
    w = int(w0) if w0 and w0 >= 5 else max(5, len(x) // 20)
    w = min(w, cap)

    sm, wu, ok = _savgol_apply(x, w, po)
    if not ok:
        return x, 0, 0.0
    frac = _resid_signal_frac(x, sm)

    tried = [(wu, frac)]
    while abs(frac) > max_frac and len(tried) < 6 and wu > 5:
        w = max(5, wu // 2)
        sm2, wu2, ok2 = _savgol_apply(x, w, po)
        if not ok2 or wu2 >= wu:
            break
        sm, wu = sm2, wu2
        frac = _resid_signal_frac(x, sm)
        tried.append((wu, frac))
    if log and len(tried) > 1:
        print("[Denoise] savgol 窗口自检（防过平滑）: "
              + "  ".join("win=%d 残差含信号%+.1f%%" % (a, 100 * b) for a, b in tried), flush=True)
    return sm, wu, frac


def denoise(t, angles, method, cfg, log=True):
    """降噪主函数，与底层 denoise 一致。返回 (denoised, info)。"""
    a = np.array(angles, dtype=float)
    info = {"method": method, "steps": []}

    mw = int(cfg["median_window"])
    if mw >= 3 and method != "none":
        before = int(np.isfinite(a).sum())
        a = remove_spikes(a, cfg["max_jump_deg"])
        info["steps"].append("spike_removed=%d" % (before - int(np.isfinite(a).sum())))

    good = np.isfinite(a)
    if method == "none" or good.sum() < 5:
        filled, flag, gs = fill_gaps(a, cfg["max_gap_frames"])
        info["steps"].append("no_smoothing")
        info["gap"] = gs
        info["gap_filled"] = flag
        return filled, info

    tmp = np.array(a, dtype=float)
    x = np.arange(len(tmp))
    tmp[~good] = np.interp(x[~good], x[good], tmp[good])

    if method == "median":
        w = mw if mw >= 3 else 5
        sm = median_filter(tmp, size=w, mode="nearest")
        info["steps"].append("median win=%d" % w)
    elif method == "savgol":
        w0 = choose_savgol_window(len(tmp), t, tmp, cfg)
        po = int(cfg["savgol_polyorder"])
        sm, w, frac = _savgol_autotune(tmp, w0, po, cfg, log=log)
        if w:
            info["steps"].append("savgol win=%d poly=%d (残差含信号 %+.1f%%)" % (w, po, 100 * frac))
            info["resid_signal_frac"] = frac
        else:
            info["steps"].append("savgol skipped (series too short)")
        info["savgol_window"] = w
    elif method == "butter":
        T, f0, ncyc = estimate_period_fft(t, tmp)
        dt = float(np.median(np.diff(t)))
        fs = 1.0 / max(dt, 1e-12)
        if f0 is None or ncyc < 1:
            fc = 0.25 * fs
            info["steps"].append("butter 无可靠主频，退回 fc=0.25*fs")
        else:
            fc = min(0.45 * fs, float(cfg["butter_cutoff_ratio"]) * f0)
        wn = max(1e-6, min(0.99, fc / (0.5 * fs)))
        order = int(cfg["butter_order"])
        if len(tmp) < max(9, 3 * order):
            info["steps"].append("butter skipped (series too short)")
            sm = tmp
        else:
            b, aa = butter(order, wn, btype="low")
            sm = filtfilt(b, aa, tmp)
            info["steps"].append("butter order=%d fc=%.3f Hz (f0=%s)"
                                 % (order, fc, "%.3f" % f0 if f0 else "--"))
        info["butter_cutoff_hz"] = fc
    else:
        raise ValueError("未知 denoise_method: %r" % method)

    sm[~good] = np.nan
    filled, flag, gs = fill_gaps(sm, cfg["max_gap_frames"])
    info["gap"] = gs
    info["gap_filled"] = flag
    return filled, info


def _save_calibration_preview(frame, origin, va, vb, path):
    vis = frame.copy()
    o = (int(round(origin[0])), int(round(origin[1])))
    pa = (int(round(va[0])), int(round(va[1])))
    pb = (int(round(vb[0])), int(round(vb[1])))
    cv2.line(vis, pa, pb, COL_Y, 2, cv2.LINE_AA)
    cv2.circle(vis, o, 7, COL_ORIGIN, -1, cv2.LINE_AA)
    cv2.circle(vis, pa, 6, (0, 200, 255), -1, cv2.LINE_AA)
    cv2.circle(vis, pb, 6, (0, 200, 255), -1, cv2.LINE_AA)
    draw_text(vis, "1:origin", (o[0] + 8, o[1] - 8), 0.55, COL_ORIGIN, 1)
    draw_text(vis, "2", (pa[0] + 8, pa[1] - 6), 0.55, (0, 200, 255), 1)
    draw_text(vis, "3", (pb[0] + 8, pb[1] + 18), 0.55, (0, 200, 255), 1)
    ok, enc = cv2.imencode(".png", vis)
    if ok:
        enc.tofile(path)


def center_diagnostic(t, envelope):
    """仅在受支持的相邻周期之间插值估计中心角，不在长缺口上跨接。"""
    t = np.asarray(t, float)
    center = np.full(len(t), np.nan)
    for i in range(len(envelope["amp_t"]) - 1):
        left, right = envelope["amp_t"][i:i + 2]
        period = max(envelope["period_s"][i:i + 2])
        if right - left > 1.8 * period:
            continue
        mask = (t >= left) & (t <= right)
        center[mask] = np.interp(t[mask], [left, right], envelope["center"][i:i + 2])
    return center


def write_raw_csv(path, t, raw, slow, frame_indices=None, centers=None, confidences=None):
    """主 CSV 仅保留真实逐帧测角；angle_deg 为 angle_deg_raw 的别名，gap_filled 恒为 0。"""
    t, raw = np.asarray(t, float), np.asarray(raw, float)
    if t.shape != raw.shape:
        raise ValueError("时间与原始角度长度不一致。")
    indices = np.arange(len(t)) if frame_indices is None else np.asarray(frame_indices)
    xy = np.full((len(t), 2), np.nan) if centers is None else np.asarray(centers, float)
    confidence = np.full(len(t), np.nan) if confidences is None else np.asarray(confidences, float)

    def number(value):
        return format(float(value), ".15g") if np.isfinite(value) else ""

    with open(path, "w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "time_s", "angle_deg_raw", "angle_deg", "gap_filled",
            "frame_index", "video_elapsed_s", "center_x_px", "center_y_px",
            "confidence", "observed",
        ])
        for i in range(len(t)):
            writer.writerow([
                number(t[i]),
                number(raw[i]),
                number(raw[i]),
                0,
                int(indices[i]),
                number(t[i] * slow),
                number(xy[i, 0]),
                number(xy[i, 1]),
                number(confidence[i]),
                int(np.isfinite(raw[i])),
            ])


def make_plot(t, a_raw, a_dn, pk, path, slow, cfg, dec=None):
    """角度-时间图（单面板摆角—时间曲线）。"""
    t, a_raw, a_dn = np.asarray(t), np.asarray(a_raw), np.asarray(a_dn)
    dec = decay_report(t, a_dn, pk) if dec is None else dec
    setup_matplotlib_chinese()
    fig, ax = plt.subplots(figsize=(14, 5.4))
    title = "摆角—时间曲线"
    if abs(slow - 1.0) > 1e-9:
        title += "   [慢放 x%g]" % slow
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.plot(t, a_raw, lw=0.65, color="#78909c", alpha=0.7, label="原始角度（CSV 保留此列）")
    ax.plot(t, a_dn, lw=1.0, color="#1565c0", label="降噪诊断 (%s)" % cfg["denoise_method"])
    if len(pk):
        ax.plot(t[pk], a_dn[pk], ".", ms=4, color="#c2185b", label="极大点 %d" % len(pk))
    if len(dec.get("amp_t", [])):
        at = dec["amp_t"]
        ax.plot(at, dec["upper"], color="#e67e22", lw=1.2, label="上/下包络：中心 ± 幅度")
        ax.plot(at, dec["lower"], color="#e67e22", lw=1.2)
        ax.plot(at, dec["center"], "--", color="#212121", lw=1.2, label="逐周期估计中心（未扣除）")
    ax.axhline(0, color="gray", ls=":", lw=0.8, label="所选竖直参考方向的零度")
    ax.set_ylabel("测量角度 (deg)")
    ax.set_xlabel("真实时间 (s)")
    ax.legend(fontsize=8.5, loc="upper right")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def make_detail_plot(t, raw, filtered, peaks, path, slow, dec):
    setup_matplotlib_chinese()
    period, _, _ = estimate_period_fft(t, filtered)
    span = min(float(t[-1] - t[0]), 5 * period if period is not None else 2.0)
    span = max(span, 1e-3)
    windows = [
        (t[0], t[0] + span),
        ((t[0] + t[-1] - span) / 2.0, (t[0] + t[-1] + span) / 2.0),
        (max(t[0], t[-1] - span), t[-1]),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(12, 9))
    center = center_diagnostic(t, dec)
    for ax, title, (left, right) in zip(axes, ["起始段", "中段", "尾段"], windows):
        mask = (t >= left) & (t <= right)
        ax.plot(t[mask], raw[mask], color="#78909c", lw=0.8, label="原始角度")
        ax.plot(t[mask], filtered[mask], color="#1565c0", lw=1.0, label="降噪诊断")
        if len(peaks):
            selected = peaks[(t[peaks] >= left) & (t[peaks] <= right)]
            ax.plot(t[selected], filtered[selected], "v", ms=4, color="#c2185b", label="极大点")
        ax.plot(t[mask], center[mask], "--", color="#6d4c41", label="估计振动中心")
        ax.axhline(0, color="gray", ls=":", lw=0.8)
        ax.set_title(title)
        ax.set_ylabel("测量角度 (deg)")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, ncol=4)
    axes[-1].set_xlabel("按慢放 x%g 换算的时间 (s)" % slow)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ================================================================
#  主流程（去掉交互：标定来自网页，起始帧来自 start_frame）
# ================================================================

def process_niubai(video_path, output_dir, model_path, calibration,
                   slow_motion_factor=1,
                   start_frame=None,
                   skip_start_sec=0.0,
                   skip_end_sec=0.0,
                   denoise_method="savgol",
                   imgsz=672,
                   yolo_conf=0.40,
                   yolo_iou=0.45,
                   target_class=0,
                   gen_video=True,
                   output_video_max_side=1280,
                   model=None,
                   progress_callback=None):
    """
    calibration: {
        "origin":         [x, y],   # ① 原点（转轴）
        "vertical_start": [x, y],   # ② 竖直方向起点
        "vertical_end":   [x, y],   # ③ 竖直方向终点（②->③ 定义竖直方向）
    }  均为原图坐标。
    start_frame: 网页"计时第一帧"，该帧即 t=0；未传时默认第 0 帧。
    """
    out_dir = os.path.abspath(output_dir)
    os.makedirs(out_dir, exist_ok=True)

    cfg = dict(DEFAULT_CONFIG)
    cfg["denoise_method"] = denoise_method or "savgol"
    cfg["imgsz"] = int(imgsz)
    cfg["yolo_conf"] = float(yolo_conf)
    cfg["yolo_iou"] = float(yolo_iou)
    cfg["target_class"] = int(target_class)
    cfg["output_video_max_side"] = int(output_video_max_side)

    # ---- 标定：3 点 ----
    origin = np.asarray(calibration["origin"], dtype=float)
    va = np.asarray(calibration["vertical_start"], dtype=float)
    vb = np.asarray(calibration["vertical_end"], dtype=float)
    v = vb - va
    nv = float(np.linalg.norm(v))
    if nv < 1e-6:
        raise RuntimeError("竖直方向两点重合，无法确定竖直方向，请重新标定。")
    vunit = v / nv
    print("[扭摆] 标定: 原点=(%.1f, %.1f)  竖直方向=(%.4f, %.4f) 屏幕角=%.2f°"
          % (origin[0], origin[1], vunit[0], vunit[1],
             math.degrees(math.atan2(vunit[1], vunit[0]))), flush=True)

    # ---- 视频 / 时间基准 ----
    src = cv2.VideoCapture(video_path)
    if not src.isOpened():
        raise RuntimeError("无法打开视频: %s" % video_path)
    src.set(cv2.CAP_PROP_ORIENTATION_AUTO, float(cfg.get("video_orientation_auto", True)))
    fps = float(src.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 1e-6:
        fps = 30.0
    total_frames = int(src.get(cv2.CAP_PROP_FRAME_COUNT))
    vw = int(src.get(cv2.CAP_PROP_FRAME_WIDTH))
    vh = int(src.get(cv2.CAP_PROP_FRAME_HEIGHT))

    slow = float(slow_motion_factor)
    if slow <= 0:
        src.release()
        raise RuntimeError("慢放倍数必须 > 0（N 倍慢动作就填 N）。")
    real_fps = fps * slow

    if start_frame is not None:
        f_start = max(0, min(int(start_frame), max(0, total_frames - 1)))
    else:
        f_start = 0
    f_end = total_frames - int(skip_end_sec * real_fps)
    f_end = max(f_start + 1, min(f_end, total_frames))
    print("[扭摆] 起始帧=%d（t=0）  真实采样率=%.3f Hz  分析 %d..%d 帧"
          % (f_start, real_fps, f_start, f_end - 1), flush=True)

    if model is None:
        model = YOLO(model_path)

    # ---- 输出视频（用源帧率播放；优先 H.264，结果页要内联预览）----
    out_scale = min(1.0, float(cfg["output_video_max_side"]) / max(1, max(vw, vh)))
    out_w = max(2, int(round(vw * out_scale)))
    out_h = max(2, int(round(vh * out_scale)))
    sx = out_w / float(vw)
    sy = out_h / float(vh)
    out_fps = max(1.0, min(240.0, fps))
    vid_path = os.path.join(out_dir, "annotated_video.mp4")
    writer = _AnnotatedVideoWriter(vid_path, out_fps, out_w, out_h) if gen_video else None

    # ---- 逐帧检测 + 算角 ----
    src.set(cv2.CAP_PROP_POS_FRAMES, f_start)
    times, ang_raw, cx_list, cy_list, conf_list = [], [], [], [], []
    n_det = 0
    prev_angle = None
    last_disp_angle = 0.0
    calib_preview_path = os.path.join(out_dir, "calibration_preview.png")
    saved_calib_preview = False
    total = max(1, f_end - f_start)

    for fidx in range(f_start, f_end):
        ok, frame = src.read()
        if not ok:
            break
        if not saved_calib_preview:
            _save_calibration_preview(frame, origin, va, vb, calib_preview_path)
            saved_calib_preview = True
        t_real = (fidx - f_start) / max(real_fps, 1e-9)

        _infer = model.predict if hasattr(model, "predict") else model
        res = _infer(frame, conf=cfg["yolo_conf"], iou=cfg["yolo_iou"],
                     imgsz=cfg["imgsz"], verbose=False)[0]
        best_box, best_conf = None, 0.0
        for box in res.boxes:
            if int(box.cls[0]) == cfg["target_class"]:
                c = float(box.conf[0])
                if c > best_conf:
                    best_conf, best_box = c, box

        ang = np.nan
        cx_val = cy_val = np.nan
        x1 = y1 = x2 = y2 = cx = cy = None
        if best_box is not None:
            x1, y1, x2, y2 = [float(vv) for vv in best_box.xyxy[0]]
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            cx_val, cy_val = cx, cy
            a_now = signed_angle_deg(vunit, (cx - origin[0], cy - origin[1]))
            if prev_angle is not None:
                d = (a_now - prev_angle + 180.0) % 360.0 - 180.0
                a_now = prev_angle + d
            ang = a_now
            prev_angle = a_now
            last_disp_angle = a_now
            n_det += 1
        else:
            # 漏检期间转过的圈数未知，不把前一段的展开角强加到下一段。
            prev_angle = None

        times.append(t_real)
        ang_raw.append(ang)
        cx_list.append(cx_val)
        cy_list.append(cy_val)
        conf_list.append(best_conf if best_box is not None else np.nan)

        if writer is not None:
            disp = frame.copy() if out_scale >= 1.0 else cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
            det_disp = None
            if best_box is not None:
                det_disp = (x1 * sx, y1 * sy, x2 * sx, y2 * sy, cx * sx, cy * sy)
            draw_overlay(
                disp,
                (origin[0] * sx, origin[1] * sy),
                vunit,
                ang if np.isfinite(ang) else last_disp_angle,
                t_real,
                det=det_disp,
                conf=best_conf if best_box is not None else None,
                frame_idx=fidx,
                frame_total=total_frames,
                det_lost=best_box is None,
            )
            writer.write(disp)

        if progress_callback and (fidx - f_start) % 50 == 0:
            progress_callback(fidx - f_start, total)

    src.release()
    video_error = writer.close() if writer is not None else None
    video_codec = getattr(writer, "codec", "") if writer is not None else ""
    video_playable = False
    if writer is not None:
        if video_error:
            print("[扭摆] 标注视频写盘告警: %s" % video_error, flush=True)
        if writer.written and os.path.exists(vid_path) and video_readable(vid_path):
            video_playable = video_codec == "h264"
            print("[扭摆] 标注视频: %s（%s，%d 帧%s）"
                  % (vid_path, writer.backend or "?", writer.written,
                     "，浏览器可内联播放" if video_playable else "，mp4v 浏览器播不了（仅落盘）"),
                  flush=True)
        else:
            print("[扭摆] 标注视频写盘失败，已删除：%s" % (video_error or video_codec or "未知原因"),
                  flush=True)
            try:
                if os.path.exists(vid_path):
                    os.remove(vid_path)
            except Exception:
                pass
            if not video_error:
                video_error = "标注视频写入失败"
    if progress_callback:
        progress_callback(total, total)

    n_frames = len(times)
    if n_frames == 0:
        raise RuntimeError("一帧都没读到，请检查视频。")
    if n_det == 0:
        raise RuntimeError("零检出：检查模型 / imgsz / target_class 是否匹配。")
    rate = 100.0 * n_det / n_frames
    print("[扭摆] 处理完成: %d 帧，检出 %d 帧（%.1f%%）" % (n_frames, n_det, rate), flush=True)

    t = np.array(times, dtype=float)
    a_raw = unwrap_deg(np.array(ang_raw, dtype=float))[0]

    # ---- 降噪（仅作诊断，不改主 CSV 的 raw 观测列）----
    method = cfg["denoise_method"]
    a_dn, info = denoise(t, a_raw, method, cfg)
    print("[扭摆] 降噪: %s   %s" % (method, "; ".join(info.get("steps", []))), flush=True)

    amp_raw = float(np.nanmax(a_raw) - np.nanmin(a_raw)) / 2.0
    amp_dn = float(np.nanmax(a_dn) - np.nanmin(a_dn)) / 2.0
    keep = 100.0 * amp_dn / max(amp_raw, 1e-9)

    resid = float("nan")
    if len(a_dn) > 10:
        diff = a_raw - a_dn
        if np.isfinite(diff).any():
            resid = float(np.nanstd(np.diff(diff)))

    # ---- 周期 / 极大点 / 阻尼 ----
    T, f0, ncyc = estimate_period_fft(t, a_dn)
    pk, _pk_prom = find_cycle_peaks(t, a_dn)
    dec = decay_report(t, a_dn, pk)
    gs = info.get("gap", {})

    period_peaks = float(np.median(np.diff(t[pk]))) if len(pk) >= 3 else None
    print("[扭摆] FFT周期=%s s (%s Hz, %s 个周期)  峰值周期=%s s  极大点=%d"
          % (fmt(T, 5), fmt(f0, 4), fmt(ncyc, 1), fmt(period_peaks, 5), len(pk)), flush=True)

    # ---- 导出 CSV（主 CSV 仅保留真实观测 + gap_filled=0；诊断与周期指标分文件保存）----
    csv_path = os.path.join(out_dir, "angle_time_data.csv")
    write_raw_csv(
        csv_path,
        t,
        a_raw,
        slow,
        frame_indices=np.arange(f_start, f_start + n_frames),
        centers=np.column_stack([cx_list, cy_list]),
        confidences=np.asarray(conf_list, dtype=float),
    )

    estimated_center = center_diagnostic(t, dec)
    flags = info.get("gap_filled", np.zeros(n_frames, dtype=int))
    diag_csv_path = os.path.join(out_dir, "angle_time_diagnostics.csv")
    with open(diag_csv_path, "w", newline="", encoding="utf-8-sig") as stream:
        writer_csv = csv.writer(stream)
        writer_csv.writerow([
            "time_s", "filtered_angle_deg", "diagnostic_gap_filled",
            "estimated_center_deg", "raw_minus_estimated_center_deg",
        ])
        for values in zip(t, a_dn, flags, estimated_center, a_raw - estimated_center):
            writer_csv.writerow([format(float(v), ".15g") if np.isfinite(v) else "" for v in values])

    cycle_csv_path = os.path.join(out_dir, "cycle_metrics.csv")
    with open(cycle_csv_path, "w", newline="", encoding="utf-8-sig") as stream:
        writer_cyc = csv.writer(stream)
        names = [
            "amp_t", "amp", "center", "upper", "lower", "period_s",
            "peak_left_index", "trough_index", "peak_right_index",
        ]
        writer_cyc.writerow([
            "time_s", "amplitude_deg", "estimated_center_deg", "upper_deg",
            "lower_deg", "period_s", *names[6:],
        ])
        if len(dec.get("amp_t", [])):
            writer_cyc.writerows(zip(*(dec[name] for name in names)))

    plot_path = os.path.join(out_dir, "angle_time_plot.png")
    make_plot(t, a_raw, a_dn, pk, plot_path, slow, cfg, dec=dec)
    detail_plot_path = os.path.join(out_dir, "angle_time_detail.png")
    make_detail_plot(t, a_raw, a_dn, pk, detail_plot_path, slow, dec=dec)

    meta = {
        "video": video_path,
        "model": model_path,
        "video_fps": fps,
        "frames_total": total_frames,
        "start_frame": int(f_start),
        "frames_analyzed": n_frames,
        "detections": n_det,
        "detection_rate_percent": rate,
        "slow_motion_factor": slow,
        "video_orientation_auto": bool(cfg.get("video_orientation_auto", True)),
        "decoded_frame_size_px": [vw, vh],
        "real_fps": real_fps,
        "origin_px": [float(origin[0]), float(origin[1])],
        "vertical_unit": [float(vunit[0]), float(vunit[1])],
        "calibration_preview": calib_preview_path if saved_calib_preview else None,
        "denoise_method": method,
        "denoise_steps": info.get("steps", []),
        "savgol_window": info.get("savgol_window"),
        "butter_cutoff_hz": info.get("butter_cutoff_hz"),
        "resid_signal_frac": info.get("resid_signal_frac"),
        "noise_resid_deg": resid if np.isfinite(resid) else None,
        "amplitude_deg_raw": amp_raw,
        "amplitude_deg_denoised": amp_dn,
        "amplitude_keep_percent": keep,
        "period_fft_s": T,
        "freq_fft_hz": f0,
        "fft_period_cycles_in_record": ncyc,
        "fft_period_trusted": bool(T is not None and ncyc >= 3),
        "period_peaks_s": period_peaks,
        "peaks_found": int(len(pk)),
        "last_peak_time_s": float(t[pk[-1]]) if len(pk) else None,
        "cycle_center_median_deg": float(np.median(dec["center"])) if len(dec.get("center", [])) else None,
        "raw_csv_columns": {"angle_deg_raw": "unfiltered measured reference angle",
                            "angle_deg": "alias of angle_deg_raw"},
        "raw_csv_smoothing": False,
        "raw_csv_gap_interpolation": False,
        "raw_csv_automatic_zero_correction": False,
        "decay_blocks": dec.get("segments", []),
        "decay_global_beta_per_s": _decay_rate(t, a_dn, pk),
        "decay_note": "仅为对数振幅的经验斜率，不据此确定阻尼机制或物理阻尼系数。",
        "gap_filled_frames": gs.get("n_filled", 0),
        "gap_left_frames": gs.get("n_left", 0),
    }
    meta_path = os.path.join(out_dir, "run_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return {
        "csv_path": csv_path,
        "diagnostics_csv": diag_csv_path,
        "cycle_metrics_csv": cycle_csv_path,
        "plot_path": plot_path,
        "detail_plot_path": detail_plot_path,
        "calibration_preview_path": calib_preview_path if saved_calib_preview else None,
        "video_path": vid_path if (writer is not None and writer.written and os.path.exists(vid_path)) else None,
        "video_codec": video_codec,
        "video_playable": video_playable,
        "video_error": video_error,
        "video_frames": writer.written if writer is not None else 0,
        "meta_path": meta_path,
        "num_points": n_frames,
        "start_frame": int(f_start),
        "real_fps": real_fps,
        "period_s": T,
        "period_peaks_s": period_peaks,
        "decay_blocks": dec.get("segments", []),
        "amp_raw_deg": amp_raw,
        "amp_denoised_deg": amp_dn,
    }
