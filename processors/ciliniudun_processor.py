# -*- coding: utf-8 -*-
"""
磁力牛顿摆 (Magnetic Newton's Cradle) 处理器
================================================
移植自 yolov8/src/磁力牛顿摆/magnetic_newton_cradle.py，供 Web 后端调用。

与原始脚本的差异：
  1. 去掉 OpenCV 手动标记窗口（ManualMarker），标定点改由网页点击传入。
  2. 去掉实时预览，增加 progress_callback 上报处理进度。
  3. 输出文件统一加上视频 basename 前缀，避免同一输出目录内多次运行互相覆盖。
  4. 增加 gen_video 参数，可控制是否输出带检测框/摆编号/实时角度的标注视频。

流程（与原脚本一致）：
  1. 由网页分阶段标定：依次点击每个摆球的悬挂点（num_pivots 个），
     再标记竖直方向两点、水平方向两点（所有摆共用）。
  2. YOLOv8 检测 + ByteTrack 跟踪：逐帧得到每个目标的轨迹。
  3. 轨迹绑定：每个跟踪轨迹的质心与哪个悬挂点（摆动圆心）最近，就归哪个摆。
  4. 水平几何约束修正：匈牙利算法全局匹配，解决摆之间混淆。
  5. 角度计算：检测框中心与悬挂点连线相对竖直方向的夹角（最低点为 0°，向右为正）。
  6. 降噪：统一时间网格 + 短缺失线性插值 + Savitzky-Golay 滤波。
  7. 输出：
       {basename}_angle_1.csv ... angle_N.csv   每个摆的时间-角度
       {basename}_all_pendulums.csv             所有摆汇总
       {basename}_tracked_video.mp4             标注视频（可选）
       {basename}_all_pendulums_plot.png        所有摆的时空数据图
"""

import os
import sys
import math
import time
import tempfile
import traceback

import cv2
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.signal import savgol_filter
from scipy.ndimage import median_filter
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

# ============================================================================
# 默认配置（与原脚本 CONFIG 对齐，Web 参数可覆盖）
# ============================================================================
DEFAULT_CFG = {
    "skip_start_s": 2.0,       # 跳过视频开头的秒数
    "skip_end_s": 0.0,         # 跳过视频结尾的秒数
    "target_fps": 30,          # 目标采样帧率
    "target_class_id": None,   # 只检测某个类别 ID，None=全部类别
    "conf": 0.18,              # 置信度阈值
    "iou": 0.5,                # NMS IoU 阈值
    "imgsz": 1536,             # 推理分辨率
    "tracker": "bytetrack.yaml",
    "angle_unit": "deg",       # "deg" 或 "rad"
    "right_positive": True,    # True: 摆向右偏为正
    "sg_window": 11,           # Savitzky-Golay 窗口
    "sg_polyorder": 3,         # Savitzky-Golay 多项式阶数
    "median_k": 0,             # 可选的中值滤波核大小（0=关闭）
    "max_nan_fill": 8,         # 连续缺失帧数 ≤ 该值时线性插值
    "video_output_scale": 1.0, # 标注视频输出缩放
    "max_frames": None,        # 最多处理多少帧（调试用）
}

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".MP4", ".AVI", ".MOV")


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------
def normalize(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < 1e-12:
        raise ValueError("两点重合，无法确定方向。")
    return v / n


def compute_angle(center, origin, v_unit, right_positive=True, unit="deg"):
    """检测框中心与圆心连线相对竖直方向的夹角。
    最低点（连线与竖直同向）为 0°，向右偏为正（right_positive=True）。
    """
    r = np.asarray(center, dtype=float) - np.asarray(origin, dtype=float)
    h_unit = np.array([v_unit[1], -v_unit[0]])  # 竖直方向的"右侧"单位向量
    ang = math.atan2(r @ h_unit, r @ v_unit)
    if not right_positive:
        ang = -ang
    if unit == "rad":
        return ang
    return math.degrees(ang)


def fmt_angle(ang, unit="deg"):
    return f"{ang:.2f}"


# ---------------------------------------------------------------------------
# 第一遍：YOLOv8 检测 + ByteTrack 跟踪
# ---------------------------------------------------------------------------
def collect_tracks(model, cap, cfg, progress_callback=None):
    """逐帧跟踪，返回 (track_data, frame_boxes, meta)。"""
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1e-6 or math.isnan(fps):
        fps = 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    start_frame = int(round(cfg["skip_start_s"] * fps))
    end_frame = total - int(round(cfg["skip_end_s"] * fps))
    start_frame = max(0, min(start_frame, total - 1))
    end_frame = max(start_frame + 1, min(end_frame, total))

    interval = max(1, int(round(fps / cfg["target_fps"])))
    dt = interval / fps
    est_total = max(1, (end_frame - start_frame + interval - 1) // interval)

    classes = None
    if cfg["target_class_id"] is not None:
        classes = [int(cfg["target_class_id"])]

    track_data = {}
    frame_boxes = {}
    n = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    t0 = time.time()

    while True:
        frame = None
        for _ in range(interval):
            ok, f = cap.read()
            if not ok:
                break
            frame = f
        if frame is None:
            break
        abs_idx = start_frame + n * interval
        if abs_idx >= end_frame:
            break
        t = n * dt

        results = model.track(frame, persist=True, conf=cfg["conf"], iou=cfg["iou"],
                              imgsz=cfg["imgsz"], classes=classes,
                              tracker=cfg["tracker"], verbose=False)
        boxes = results[0].boxes
        fboxes = []
        if boxes is not None and boxes.id is not None:
            ids = boxes.id.cpu().numpy().astype(int)
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            for tid, box, c in zip(ids, xyxy, confs):
                cx = (box[0] + box[2]) / 2.0
                cy = (box[1] + box[3]) / 2.0
                track_data.setdefault(tid, []).append(
                    (n, t, float(cx), float(cy), box.tolist(), float(c)))
                fboxes.append((int(tid), box.tolist(), float(c)))
        frame_boxes[n] = fboxes
        n += 1

        if progress_callback and n % max(1, round(50 / interval)) == 0:
            progress_callback(n, est_total)
        if cfg["max_frames"] and n >= cfg["max_frames"]:
            break

    if progress_callback:
        progress_callback(n, est_total)

    meta = dict(fps=fps, start_frame=start_frame, interval=interval, dt=dt,
                n_frames=n, frame_size=(width, height),
                video_name=os.path.basename(cfg["video_path"]),
                total=total)
    return track_data, frame_boxes, meta


# ---------------------------------------------------------------------------
# 轨迹 -> 摆 绑定
# ---------------------------------------------------------------------------
def bind_tracks(track_data, origins):
    """每条轨迹的质心（中位点）离哪个圆心最近就归哪个摆（仅作参考统计）。"""
    centers = np.array(origins, dtype=float)
    bind_map = {}
    dists = {}
    for tid, pts in track_data.items():
        arr = np.array([(p[2], p[3]) for p in pts])  # (cx, cy)
        median_pt = np.median(arr, axis=0)
        d = np.linalg.norm(centers - median_pt, axis=1)
        pi = int(np.argmin(d))
        bind_map[tid] = pi
        dists[tid] = float(d[pi])

    per_pendulum = {i: [] for i in range(len(origins))}
    for tid, pi in bind_map.items():
        per_pendulum[pi].append(tid)
    for tid, pi in bind_map.items():
        arr = np.array([(p[2], p[3]) for p in track_data[tid]])
        L = float(np.median(np.linalg.norm(arr - centers[pi], axis=1)))
        d = dists[tid]
        if L > 5 and abs(d - L) > 0.4 * L:
            print(f"[绑定] 注意：轨迹 {tid} -> P{pi + 1}，质心距圆心 {d:.1f}px，"
                  f"与摆长 {L:.1f}px 明显不符，可能横跨多个摆。")
    return bind_map


# ---------------------------------------------------------------------------
# 水平几何约束修正（匈牙利算法全局最优一对一匹配）
# ---------------------------------------------------------------------------
def enforce_horizontal_order(frame_boxes, origins, h_unit, meta):
    """把每个检测框分配给正确的摆：水平区间软约束 + 位置连续性全局匹配。"""
    N = len(origins)
    proj = np.array([o @ h_unit for o in origins])
    mids = [(proj[i] + proj[i + 1]) / 2.0 for i in range(N - 1)]
    zone_w = 2.5
    dt = meta["dt"]

    all_dist = []
    for n0 in range(meta["n_frames"]):
        for (tid, xyxy, conf) in frame_boxes.get(n0, []):
            cx = (xyxy[0] + xyxy[2]) / 2.0
            cy = (xyxy[1] + xyxy[3]) / 2.0
            all_dist.append(min(math.hypot(cx - o[0], cy - o[1]) for o in origins))
    L_est = float(np.median(all_dist)) if all_dist else 320.0
    LO, HI = 0.4 * L_est, 1.6 * L_est

    prev_pos = [None] * N
    prev_vel = [None] * N
    pend_pts = {i: [] for i in range(N)}
    frame_assign = {}
    n_frames = meta["n_frames"]

    def zone_pen(p, i):
        pen = 0.0
        if i > 0 and p < mids[i - 1]:
            pen = mids[i - 1] - p
        if i < N - 1 and p > mids[i]:
            pen = p - mids[i]
        return pen

    for n in range(n_frames):
        boxes = frame_boxes.get(n, [])
        if not boxes:
            frame_assign[n] = {}
            continue
        items = []
        for (tid, xyxy, conf) in boxes:
            cx = (xyxy[0] + xyxy[2]) / 2.0
            cy = (xyxy[1] + xyxy[3]) / 2.0
            items.append((tid, cx, cy, conf))
        M = len(items)

        if n == 0:
            chosen = {}
            for i in range(N):
                cands = [it for it in items if _interval_of(it, h_unit, mids) == i]
                if cands:
                    chosen[i] = cands[0]
        else:
            pred = np.zeros((N, 2))
            for i in range(N):
                if prev_pos[i] is not None:
                    if prev_vel[i] is not None:
                        pred[i] = prev_pos[i] + prev_vel[i] * dt
                    else:
                        pred[i] = prev_pos[i]
                else:
                    pred[i] = origins[i] + np.array([0.0, L_est])
            cost = np.zeros((N, M))
            for i in range(N):
                for j in range(M):
                    dx = items[j][1] - pred[i][0]
                    dy = items[j][2] - pred[i][1]
                    d_pos = math.sqrt(dx * dx + dy * dy)
                    p = items[j][1] * h_unit[0] + items[j][2] * h_unit[1]
                    cost[i][j] = d_pos + zone_w * zone_pen(p, i)
            rows, cols = linear_sum_assignment(cost)
            chosen = {r: items[c] for r, c in zip(rows, cols)}

        fassign = {}
        for i in range(N):
            it = chosen.get(i)
            if it is None:
                continue
            d_origin = math.hypot(it[1] - origins[i][0], it[2] - origins[i][1])
            if not (LO <= d_origin <= HI):
                continue
            pend_pts[i].append((n, it[1], it[2], it[3]))
            fassign[it[0]] = i
            if prev_pos[i] is not None:
                prev_vel[i] = np.array([(it[1] - prev_pos[i][0]) / dt,
                                        (it[2] - prev_pos[i][1]) / dt])
            prev_pos[i] = np.array([it[1], it[2]])
        frame_assign[n] = fassign

    return pend_pts, frame_assign


def _interval_of(it, h_unit, mids):
    p = it[1] * h_unit[0] + it[2] * h_unit[1]
    return int(np.searchsorted(mids, p))


# ---------------------------------------------------------------------------
# 角度时间序列 + 降噪
# ---------------------------------------------------------------------------
def build_angle_series(pend_pts, origins, v_unit, meta, cfg):
    n_frames = meta["n_frames"]
    unit = cfg["angle_unit"]
    rp = cfg["right_positive"]

    series = {}
    for pi in range(len(origins)):
        arr = np.full(n_frames, np.nan)
        origin = origins[pi]
        for (n, cx, cy, conf) in pend_pts[pi]:
            if 0 <= n < n_frames:
                arr[n] = compute_angle((cx, cy), origin, v_unit, rp, unit)
        series[pi] = arr
    return series


def denoise_series(arr, cfg):
    window = int(cfg["sg_window"])
    polyorder = int(cfg["sg_polyorder"])
    median_k = int(cfg["median_k"])
    max_nan = int(cfg["max_nan_fill"])

    y = arr.copy()
    s = pd.Series(y)
    y = s.interpolate(method="linear", limit=max_nan, limit_direction="both").to_numpy()
    if median_k > 1:
        y = _apply_piecewise(y, lambda seg: median_filter(seg, size=median_k))
    if window > 1:
        y = _apply_piecewise(y, lambda seg: savgol_filter(
            seg, min(window, len(seg) if len(seg) % 2 == 1 else len(seg) - 1), polyorder))
    return y


def _apply_piecewise(y, func):
    out = y.copy()
    n = len(y)
    i = 0
    while i < n:
        if np.isnan(y[i]):
            i += 1
            continue
        j = i
        while j < n and not np.isnan(y[j]):
            j += 1
        seg = y[i:j]
        if len(seg) >= 3:
            try:
                out[i:j] = func(seg)
            except Exception:
                out[i:j] = seg
        i = j
    return out


# ---------------------------------------------------------------------------
# 输出：CSV、标注视频、汇总图
# ---------------------------------------------------------------------------
def write_csvs(series, t_grid, origins, out_dir, prefix, unit="deg"):
    n = len(origins)
    files = []
    rows = {"t": t_grid}
    for pi in range(n):
        col = f"angle_{pi + 1}" + ("" if unit == "deg" else "_rad")
        rows[col] = series[pi]
        df = pd.DataFrame({"t": t_grid, col: series[pi]})
        path = os.path.join(out_dir, f"{prefix}_angle_{pi + 1}.csv")
        df.to_csv(path, index=False, encoding="utf-8-sig")
        files.append(path)
    df_all = pd.DataFrame(rows)
    path_all = os.path.join(out_dir, f"{prefix}_all_pendulums.csv")
    df_all.to_csv(path_all, index=False, encoding="utf-8-sig")
    files.append(path_all)
    return files


def write_annotated_video(cap, frame_boxes, frame_assign, origins, v_unit, h_unit,
                          out_path, meta, cfg, progress_callback=None,
                          progress_base=0, progress_total=1):
    width, height = meta["frame_size"]
    scale = cfg["video_output_scale"]
    w = int(width * scale)
    h = int(height * scale)
    fps = cfg["target_fps"]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp_path = tmp.name
    tmp.close()

    writer = cv2.VideoWriter(tmp_path, fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError("无法创建视频写入器（尝试 mp4v 编码失败）。")

    cap.set(cv2.CAP_PROP_POS_FRAMES, meta["start_frame"])
    interval = meta["interval"]
    n_frames = meta["n_frames"]
    palette = [(0, 0, 255), (0, 165, 255), (0, 255, 0),
               (255, 0, 0), (255, 255, 0), (255, 0, 255)]
    rp = cfg["right_positive"]
    unit = cfg["angle_unit"]

    for n in range(n_frames):
        frame = None
        for _ in range(interval):
            ok, f = cap.read()
            if not ok:
                break
            frame = f
        if frame is None:
            break
        if scale != 1.0:
            frame = cv2.resize(frame, (w, h))
        disp = frame.copy()
        assign = frame_assign.get(n, {})
        for (tid, xyxy, conf) in frame_boxes.get(n, []):
            pi = assign.get(tid)
            if pi is None:
                color = (128, 128, 128)
                label = f"id{tid}"
            else:
                color = palette[pi % len(palette)]
                cx = (xyxy[0] + xyxy[2]) / 2.0
                cy = (xyxy[1] + xyxy[3]) / 2.0
                ang = compute_angle((cx, cy), origins[pi], v_unit, rp, unit)
                suffix = " deg" if unit == "deg" else " rad"
                label = f"P{pi + 1} id{tid} {fmt_angle(ang, unit)}{suffix}"
            x1, y1, x2, y2 = (int(xyxy[0] * scale), int(xyxy[1] * scale),
                              int(xyxy[2] * scale), int(xyxy[3] * scale))
            cv2.rectangle(disp, (x1, y1), (x2, y2), color, 2)
            cv2.putText(disp, label, (x1, max(15, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.circle(disp, (int((x1 + x2) / 2), int((y1 + y2) / 2)), 3, color, -1)
        for i, o in enumerate(origins):
            po = (int(o[0] * scale), int(o[1] * scale))
            cv2.circle(disp, po, 5, palette[i % len(palette)], -1)
            cv2.putText(disp, f"O{i + 1}", (po[0] + 8, po[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, palette[i % len(palette)], 2)
        vo = (int(origins[0][0] * scale), int(origins[0][1] * scale))
        ve = (int((origins[0][0] + v_unit[0] * 200) * scale),
              int((origins[0][1] + v_unit[1] * 200) * scale))
        cv2.arrowedLine(disp, vo, ve, (255, 0, 255), 2, tipLength=0.08)
        he = (int((origins[0][0] + h_unit[0] * 200) * scale),
              int((origins[0][1] + h_unit[1] * 200) * scale))
        cv2.arrowedLine(disp, vo, he, (0, 255, 0), 2, tipLength=0.08)
        cv2.putText(disp, f"t={n * meta['dt']:.2f}s", (15, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        writer.write(disp)
        if progress_callback and n % max(1, round(50 / interval)) == 0:
            progress_callback(progress_base + n + 1, progress_total)

    writer.release()
    if os.path.exists(out_path):
        os.remove(out_path)
    os.replace(tmp_path, out_path)
    return out_path


def plot_all(series, t_grid, origins, out_path, video_name, unit="deg"):
    """科研风格多面板图：每个摆一个面板，纵向堆叠、共享时间轴。"""
    n = len(origins)
    ylab = "Angle (rad)" if unit == "rad" else "Angle (deg)"

    # 保存并临时设置科研风格字体，绘图后恢复，避免影响同进程其他实验的绘图
    _saved = {k: plt.rcParams[k] for k in ("font.family", "font.serif", "mathtext.fontset")}
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "SimSun", "SimHei",
                                  "Microsoft YaHei", "DejaVu Serif"]
    plt.rcParams["mathtext.fontset"] = "stix"
    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]

    fig, axes = plt.subplots(n, 1, figsize=(7.0, 2.3 * n), sharex=True,
                             constrained_layout=True)
    if n == 1:
        axes = [axes]

    plotted_any = False
    for i in range(n):
        ax = axes[i]
        s = series[i]
        if not np.all(np.isnan(s)):
            ax.plot(t_grid, s, color=colors[i % len(colors)], linewidth=1.0)
            plotted_any = True
        ax.text(0.0, 1.02, f"({chr(97 + i)})", transform=ax.transAxes,
                fontsize=11, fontweight="bold", va="bottom", ha="left")
        ax.text(0.055, 1.02, f"Pendulum {i + 1}", transform=ax.transAxes,
                fontsize=10, va="bottom", ha="left")
        ax.tick_params(direction="out", labelsize=9)
        ax.grid(True, which="major", linestyle="--", linewidth=0.4, alpha=0.35)
        ax.set_ylabel(ylab, fontsize=10)
        ax.set_xlim(t_grid[0], t_grid[-1])

    axes[-1].set_xlabel("Time (s)", fontsize=10)
    fig.suptitle(f"Magnetic Newton Cradle - {video_name}", fontsize=11)

    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    for _k, _v in _saved.items():
        plt.rcParams[_k] = _v
    if not plotted_any:
        print("[绘图] 警告：没有任何摆的有效数据，图表为空。")
    else:
        print(f"[绘图] 已输出: {out_path}")


# ---------------------------------------------------------------------------
# 主处理函数（Web 调用入口）
# ---------------------------------------------------------------------------
def process_ciliniudun(video_path, output_dir, model_path, calibration,
                       num_pivots=4,
                       skip_start_sec=2.0,
                       skip_end_sec=0.0,
                       target_fps=30,
                       conf=0.18,
                       iou=0.5,
                       imgsz=1536,
                       sg_window=11,
                       sg_polyorder=3,
                       median_k=0,
                       max_nan_fill=8,
                       right_positive=True,
                       gen_video=True,
                       video_output_scale=1.0,
                       model=None,
                       progress_callback=None):
    """
    Parameters
    ----------
    calibration : dict
        {
            "pivot_1".."pivot_N": [x, y],   （原始图像坐标）
            "vertical_1": [x, y],
            "vertical_2": [x, y],
            "horizontal_1": [x, y],
            "horizontal_2": [x, y],
        }
    """
    cfg = dict(DEFAULT_CFG)
    cfg.update({
        "video_path": video_path,
        "skip_start_s": float(skip_start_sec),
        "skip_end_s": float(skip_end_sec),
        "target_fps": max(1, int(target_fps)),
        "conf": float(conf),
        "iou": float(iou),
        "imgsz": int(imgsz),
        "sg_window": max(3, int(sg_window) | 1),
        "sg_polyorder": int(sg_polyorder),
        "median_k": max(0, int(median_k)),
        "max_nan_fill": max(0, int(max_nan_fill)),
        "right_positive": bool(right_positive),
        "video_output_scale": float(video_output_scale),
    })

    if not video_path or not os.path.exists(video_path):
        raise FileNotFoundError(f"找不到视频文件: {video_path}")
    if not model_path or not os.path.exists(model_path):
        raise FileNotFoundError(f"找不到模型文件: {model_path}")

    os.makedirs(output_dir, exist_ok=True)
    prefix = os.path.splitext(os.path.basename(video_path))[0]

    # ---- 标定解析：圆心（按水平投影从左到右排序）+ 竖直/水平参考轴 ----
    num_pivots = int(num_pivots)
    if num_pivots < 1 or num_pivots > 8:
        raise RuntimeError(f"摆球数量 num_pivots={num_pivots} 超出范围 (1~8)。")

    origins_raw = []
    for i in range(1, num_pivots + 1):
        key = f"pivot_{i}"
        if key not in calibration:
            raise RuntimeError(f"缺少标定点 {key}。")
        origins_raw.append(np.asarray(calibration[key], dtype=float))
    p1 = np.asarray(calibration.get("vertical_1"), dtype=float)
    p2 = np.asarray(calibration.get("vertical_2"), dtype=float)
    h1 = np.asarray(calibration.get("horizontal_1"), dtype=float)
    h2 = np.asarray(calibration.get("horizontal_2"), dtype=float)

    v_unit = normalize(p2 - p1)
    if v_unit[1] < 0:  # 自动保证竖直方向向下
        v_unit = -v_unit
    h_unit = normalize(h2 - h1)
    h_unit = h_unit - (h_unit @ v_unit) * v_unit
    hn = np.linalg.norm(h_unit)
    if hn < 1e-6:
        raise RuntimeError("水平方向与竖直方向几乎平行，请重新标记。")
    h_unit = h_unit / hn
    if h_unit[0] < 0:  # 自动保证水平方向向右
        h_unit = -h_unit

    # 按水平投影从左到右重排摆
    proj = np.array([o @ h_unit for o in origins_raw])
    order = np.argsort(proj)
    origins = [origins_raw[i] for i in order]
    print(f"[标定] 摆按水平方向从左到右编号: "
          + ", ".join(f"摆{i + 1}=原P{int(order[i]) + 1}" for i in range(len(order))))

    # ---- 打开视频 ----
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")

    # ---- 第一遍跟踪 ----
    if model is None:
        print(f"[模型] 加载 {model_path}")
        model = YOLO(model_path)

    def _on_progress(cur, total):
        if progress_callback:
            progress_callback(cur, total)

    track_data, frame_boxes, meta = collect_tracks(model, cap, cfg,
                                                   progress_callback=_on_progress)
    cap.release()
    if meta["n_frames"] < 2:
        raise RuntimeError("有效处理帧太少，请检查跳秒/视频配置。")

    n_track = meta["n_frames"]
    n_video = n_track if (gen_video and cfg["video_output_scale"] > 0) else 0
    total_progress = n_track + n_video

    # ---- 轨迹绑定（参考统计）----
    bind_tracks(track_data, origins)

    # ---- 水平几何约束修正 ----
    pend_pts, frame_assign = enforce_horizontal_order(frame_boxes, origins, h_unit, meta)

    # ---- 角度 + 降噪 ----
    unit = cfg["angle_unit"]
    series_raw = build_angle_series(pend_pts, origins, v_unit, meta, cfg)
    series = {pi: denoise_series(series_raw[pi], cfg) for pi in range(len(origins))}

    for pi in range(len(origins)):
        valid = np.sum(~np.isnan(series_raw[pi]))
        ang = series[pi][~np.isnan(series[pi])]
        rng = (ang.min(), ang.max()) if ang.size else (float("nan"), float("nan"))
        print(f"[数据] 摆 {pi + 1}: 有效帧 {valid}/{meta['n_frames']} "
              f"({100.0 * valid / max(meta['n_frames'], 1):.1f}%), 角度范围 "
              f"[{rng[0]:.2f}, {rng[1]:.2f}] {unit}")

    # ---- 输出 CSV ----
    t_grid = np.arange(meta["n_frames"]) * meta["dt"]
    csv_files = write_csvs(series, t_grid, origins, output_dir, prefix, unit)

    # ---- 输出标注视频（可选）----
    out_video = None
    if gen_video and cfg["video_output_scale"] > 0:
        cap2 = cv2.VideoCapture(video_path)
        if cap2.isOpened():
            out_video = os.path.join(output_dir, f"{prefix}_tracked_video.mp4")
            write_annotated_video(cap2, frame_boxes, frame_assign, origins, v_unit,
                                  h_unit, out_video, meta, cfg,
                                  progress_callback=_on_progress,
                                  progress_base=n_track, progress_total=total_progress)
            cap2.release()

    _on_progress(total_progress, total_progress)

    # ---- 输出汇总图 ----
    out_plot = os.path.join(output_dir, f"{prefix}_all_pendulums_plot.png")
    plot_all(series, t_grid, origins, out_plot, meta["video_name"], unit)

    per_pendulum_csvs = csv_files[:-1]  # 去掉最后一个是汇总 CSV
    return {
        "csv_path": csv_files[-1],           # 汇总 CSV
        "per_pendulum_csvs": per_pendulum_csvs,
        "plot_path": out_plot,
        "video_path": out_video,
        "num_pendulums": len(origins),
        "num_points": int(meta["n_frames"]),
        "dt": float(meta["dt"]),
    }


if __name__ == "__main__":
    try:
        # 本地快速调试入口
        import json
        calib = json.load(open(sys.argv[1], encoding="utf-8")) if len(sys.argv) > 1 else {}
        res = process_ciliniudun(
            video_path=r"C:\Users\MECHREV\Desktop\d01dd756d79697ef84b654ca3f6e2704.mp4",
            output_dir=r"C:\Users\MECHREV\Desktop\newmethod_web",
            model_path=r"C:\Users\MECHREV\Desktop\yolov8\runs\detect\ciliniudun\weights\best.pt",
            calibration=calib, num_pivots=len([k for k in calib if k.startswith("pivot_")]) or 4)
        print(json.dumps(res, ensure_ascii=False, indent=2))
    except Exception as e:
        print(f"\n[错误] {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
