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
import traceback

import cv2
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.signal import savgol_filter
from scipy.ndimage import median_filter
from scipy.optimize import least_squares, linear_sum_assignment
from ultralytics import YOLO

from .common import H264VideoWriter, normalize_strict, video_readable

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

# ============================================================================
# 默认配置（与 magnetic_newton_cradle.py CONFIG 对齐，Web 参数可覆盖）
# ============================================================================
DEFAULT_CFG = {
    "skip_end_s": 0.0,         # 跳过视频结尾的秒数（仅手动需要时使用）
    "target_fps": 30,          # 目标采样帧率
    "target_class_id": None,   # 只检测某个类别 ID，None=全部类别
    "conf": 0.18,              # 置信度阈值
    "iou": 0.5,                # NMS IoU 阈值
    "imgsz": 1536,             # 推理分辨率
    "angle_unit": "deg",       # "deg" 或 "rad"
    "right_positive": True,    # True: 摆向右偏为正
    "sg_window": 5,            # Savitzky-Golay 窗口
    "sg_polyorder": 3,         # Savitzky-Golay 多项式阶数
    "median_k": 0,             # 可选的中值滤波核大小（0=关闭）
    "max_nan_fill": 8,         # 连续缺失帧数 ≤ 该值时内部线性插值（不做边界外推）
    "subpixel_yellow": True,   # 提取黄色贴纸色度亚像素质心
    "refine_pivots": True,     # 等摆长圆弧最小二乘精修支点圆心
    "video_output_scale": 1.0, # 标注视频输出缩放
    "max_frames": None,        # 最多处理多少帧（调试用）
}

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".MP4", ".AVI", ".MOV")


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------
# 与 common.normalize_strict 同行为（零向量抛"两点重合"错），保留原名
normalize = normalize_strict


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
# 第一遍：YOLOv8 逐帧检测 + 亚像素质心提取（不使用 ByteTrack，保留全局检测给匈牙利匹配）
# ---------------------------------------------------------------------------
def _to_numpy(arr):
    if hasattr(arr, "cpu"):
        return arr.cpu().numpy()
    return np.asarray(arr)


def _subpixel_yellow_center(frame, box, use_yellow=True):
    """在 YOLO 框内提取黄色贴纸的色度亚像素质心 (cx, cy)，将框中心平移对齐到该质心。"""
    cx = float((box[0] + box[2]) / 2.0)
    cy = float((box[1] + box[3]) / 2.0)
    if not use_yellow:
        return cx, cy, box.tolist()
    h_img, w_img = frame.shape[:2]
    pad = 3
    x1 = max(0, int(math.floor(box[0])) - pad)
    y1 = max(0, int(math.floor(box[1])) - pad)
    x2 = min(w_img, int(math.ceil(box[2])) + pad)
    y2 = min(h_img, int(math.ceil(box[3])) + pad)
    if x2 - x1 < 3 or y2 - y1 < 3:
        return cx, cy, box.tolist()
    roi = frame[y1:y2, x1:x2].astype(float)
    yel = np.maximum(0.0, 0.5 * (roi[:, :, 2] + roi[:, :, 1]) - roi[:, :, 0])
    thr = np.percentile(yel, 50)
    w = np.maximum(0.0, yel - thr) ** 2
    w_sum = float(w.sum())
    if w_sum > 1e-3:
        yy, xx = np.mgrid[y1:y2, x1:x2]
        cx = float((w * xx).sum() / w_sum)
        cy = float((w * yy).sum() / w_sum)
    bw = float(box[2] - box[0])
    bh = float(box[3] - box[1])
    sbox = [cx - bw / 2.0, cy - bh / 2.0, cx + bw / 2.0, cy + bh / 2.0]
    return cx, cy, sbox


def _read_sampled_frame(cap, interval, first):
    """首个样本直接读选定起始帧，后续先跳过 interval - 1 帧。"""
    if not first:
        for _ in range(interval - 1):
            if not cap.grab():
                return False, None
    return cap.read()


def collect_tracks(model, cap, cfg, start_frame=0, progress_callback=None):
    """逐帧检测并提取亚像素质心，返回 (track_data, frame_boxes, meta)。"""
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1e-6 or not math.isfinite(fps):
        fps = 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    start_frame = max(0, min(int(start_frame), max(0, total - 1)))
    end_frame = total - int(round(cfg.get("skip_end_s", 0.0) * fps))
    end_frame = max(start_frame + 1, min(end_frame, total))

    interval = max(1, int(round(fps / cfg["target_fps"])))
    dt = interval / fps
    est_total = max(1, (end_frame - start_frame + interval - 1) // interval)

    classes = None
    if cfg["target_class_id"] is not None:
        classes = [int(cfg["target_class_id"])]

    use_yellow = bool(cfg.get("subpixel_yellow", True))
    track_data = {}
    frame_boxes = {}
    n = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    while True:
        abs_idx = start_frame + n * interval
        if abs_idx >= end_frame:
            break
        ok, frame = _read_sampled_frame(cap, interval, first=(n == 0))
        if not ok or frame is None:
            break
        t = n * dt

        results = model.predict(frame, conf=cfg["conf"], iou=cfg["iou"],
                                imgsz=cfg["imgsz"], classes=classes, verbose=False)
        boxes = results[0].boxes if results else None
        fboxes = []
        if boxes is not None and len(boxes) > 0:
            xyxy = _to_numpy(boxes.xyxy)
            confs = _to_numpy(boxes.conf)
            order = np.argsort((xyxy[:, 0] + xyxy[:, 2]) / 2.0)
            for k, idx in enumerate(order):
                tid = k + 1
                box = xyxy[idx]
                c = float(confs[idx])
                cx, cy, sbox = _subpixel_yellow_center(frame, box, use_yellow=use_yellow)
                track_data.setdefault(tid, []).append((n, t, cx, cy, sbox, c))
                fboxes.append((tid, sbox, c))
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


def refine_origins_by_circle_fit(pend_pts, origins):
    """利用全轨迹等摆长圆弧最小二乘，自动精修被上横梁遮挡的各摆支点圆心。"""
    N = len(origins)
    pts_list = []
    for i in range(N):
        arr = np.array([(p[1], p[2]) for p in pend_pts[i]], dtype=float)
        if len(arr) < 20 or np.ptp(arr[:, 0]) < 15.0:
            return origins
        pts_list.append(arr)
    x0_init = np.array([o[0] for o in origins], dtype=float)
    y0_init = np.array([o[1] for o in origins], dtype=float)
    r_init = float(np.median([
        np.median(np.hypot(pts_list[i][:, 0] - x0_init[i], pts_list[i][:, 1] - y0_init[i]))
        for i in range(N)
    ]))
    if not math.isfinite(r_init) or r_init < 5.0:
        return origins
    p0 = np.r_[x0_init, y0_init, r_init]
    lo = np.r_[x0_init - 40.0, y0_init - 80.0, 0.7 * r_init]
    hi = np.r_[x0_init + 40.0, y0_init + 80.0, 1.3 * r_init]

    def res(p):
        R = p[2 * N]
        return np.concatenate([
            np.hypot(pts_list[i][:, 0] - p[i], pts_list[i][:, 1] - p[N + i]) - R
            for i in range(N)
        ])

    sol = least_squares(res, p0, bounds=(lo, hi), loss="soft_l1", f_scale=0.5)
    if not sol.success:
        return origins
    refined = [np.array([sol.x[i], sol.x[N + i]], dtype=float) for i in range(N)]
    R_fit = float(sol.x[2 * N])
    stds = [float(np.std(np.hypot(pts_list[i][:, 0] - refined[i][0],
                                  pts_list[i][:, 1] - refined[i][1]) - R_fit)) for i in range(N)]
    print(f"[支点精修] 公共拟合摆长 R={R_fit:.2f}px, 径向残差 std="
          + "/".join(f"{s:.2f}px" for s in stds), flush=True)
    return refined


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
    """短缺失内部线性插值（不做边界外推） -> 可选中值滤波 -> Savitzky-Golay 分段滤波。"""
    window = int(cfg["sg_window"])
    polyorder = int(cfg["sg_polyorder"])
    median_k = int(cfg["median_k"])
    max_nan = int(cfg["max_nan_fill"])
    if max_nan < 0:
        raise ValueError("max_nan_fill 必须为非负整数；0 表示关闭补帧。")

    y = np.asarray(arr, dtype=float).copy()
    if not len(y):
        return y
    missing = np.isnan(y)
    starts = np.flatnonzero(missing & ~np.r_[False, missing[:-1]])
    ends = np.flatnonzero(missing & ~np.r_[missing[1:], False]) + 1
    for start, end in zip(starts, ends):
        if 0 < start and end < len(y) and end - start <= max_nan:
            y[start:end] = np.interp(np.arange(start, end), [start - 1, end], y[[start - 1, end]])
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
def write_csvs(series, t_grid, origins, out_dir, prefix, unit="deg", series_raw=None):
    n = len(origins)
    files = []
    rows = {"t": t_grid}
    rows_raw = {"t": t_grid}
    for pi in range(n):
        col = f"angle_{pi + 1}" + ("" if unit == "deg" else "_rad")
        rows[col] = series[pi]
        if series_raw is not None:
            rows_raw[col] = series_raw[pi]
        df = pd.DataFrame({"t": t_grid, col: series[pi]})
        path = os.path.join(out_dir, f"{prefix}_angle_{pi + 1}.csv")
        df.to_csv(path, index=False, encoding="utf-8-sig")
        files.append(path)
    raw_path = None
    if series_raw is not None:
        raw_path = os.path.join(out_dir, f"{prefix}_all_pendulums_raw.csv")
        pd.DataFrame(rows_raw).to_csv(raw_path, index=False, encoding="utf-8-sig")
        files.append(raw_path)
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
    w = max(2, int(width * scale) // 2 * 2)
    h = max(2, int(height * scale) // 2 * 2)
    out_fps = 1.0 / max(meta["dt"], 1e-6)
    writer = H264VideoWriter(out_path, out_fps, w, h)
    if not writer.isOpened():
        raise RuntimeError(writer.error or "无法创建视频写入器。")

    cap.set(cv2.CAP_PROP_POS_FRAMES, meta["start_frame"])
    interval = meta["interval"]
    n_frames = meta["n_frames"]
    palette = [(0, 0, 255), (0, 165, 255), (0, 255, 0),
               (255, 0, 0), (255, 255, 0), (255, 0, 255)]
    rp = cfg["right_positive"]
    unit = cfg["angle_unit"]

    for n in range(n_frames):
        ok, frame = _read_sampled_frame(cap, interval, first=(n == 0))
        if not ok or frame is None:
            break
        if scale != 1.0 or frame.shape[1] != w or frame.shape[0] != h:
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

    err = writer.close()
    if err or not os.path.isfile(out_path) or not video_readable(out_path):
        raise RuntimeError(err or "标注视频写出失败。")
    return out_path, writer.codec, (writer.codec == "h264"), err


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
                       start_frame=None,
                       skip_start_sec=0.0,
                       skip_end_sec=0.0,
                       target_fps=30,
                       conf=0.18,
                       iou=0.5,
                       imgsz=1536,
                       sg_window=5,
                       sg_polyorder=3,
                       median_k=0,
                       max_nan_fill=8,
                       right_positive=True,
                       subpixel_yellow=True,
                       refine_pivots=True,
                       gen_video=True,
                       video_output_scale=1.0,
                       model=None,
                       progress_callback=None,
                       stage_callback=None,
                       post_target_sec=600.0):
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
    start_frame : 选定计时首帧（帧号，t=0）；未传时默认第 0 帧。
    post_target_sec : 包含最后数据后处理在内的目标总耗时（秒，默认 600s≈10min）。
    """
    t_proc_start = time.time()
    cfg = dict(DEFAULT_CFG)
    cfg.update({
        "video_path": video_path,
        "skip_end_s": float(skip_end_sec or 0.0),
        "target_fps": max(1, int(target_fps)),
        "conf": float(conf),
        "iou": float(iou),
        "imgsz": int(imgsz),
        "sg_window": max(3, int(sg_window) | 1),
        "sg_polyorder": int(sg_polyorder),
        "median_k": max(0, int(median_k)),
        "max_nan_fill": max(0, int(max_nan_fill)),
        "right_positive": bool(right_positive),
        "subpixel_yellow": bool(subpixel_yellow),
        "refine_pivots": bool(refine_pivots),
        "video_output_scale": float(video_output_scale),
    })

    if not video_path or not os.path.exists(video_path):
        raise FileNotFoundError(f"找不到视频文件: {video_path}")
    if model is None and (not model_path or not os.path.exists(model_path)):
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
    print("[标定] 摆按水平方向从左到右编号: "
          + ", ".join(f"摆{i + 1}=原P{int(order[i]) + 1}" for i in range(len(order))), flush=True)

    # ---- 打开视频并决定起始帧 ----
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")
    fps_init = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if start_frame is not None:
        f_start = max(0, int(start_frame))
    elif float(skip_start_sec or 0.0) > 0.0:
        f_start = max(0, int(round(float(skip_start_sec) * float(fps_init))))
    else:
        f_start = 0

    if model is None:
        print(f"[模型] 加载 {model_path}", flush=True)
        model = YOLO(model_path)

    def _on_progress(cur, total):
        if progress_callback:
            progress_callback(cur, total)

    track_data, frame_boxes, meta = collect_tracks(
        model, cap, cfg, start_frame=f_start, progress_callback=_on_progress
    )
    cap.release()
    if meta["n_frames"] < 2:
        raise RuntimeError("有效处理帧太少，请检查起始帧或视频长度。")

    n_track = meta["n_frames"]
    n_video = n_track if (gen_video and cfg["video_output_scale"] > 0) else 0
    total_progress = n_track + n_video

    # ---- 轨迹绑定（参考统计）----
    bind_tracks(track_data, origins)

    # ---- 水平几何约束修正 ----
    pend_pts, frame_assign = enforce_horizontal_order(frame_boxes, origins, h_unit, meta)

    # ---- 等摆长圆弧最小二乘精修支点圆心 ----
    if cfg["refine_pivots"]:
        origins = refine_origins_by_circle_fit(pend_pts, origins)

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
              f"[{rng[0]:.2f}, {rng[1]:.2f}] {unit}", flush=True)

    # ---- 输出 CSV（含降噪分摆/汇总 + raw）----
    t_grid = np.arange(meta["n_frames"]) * meta["dt"]
    csv_files = write_csvs(series, t_grid, origins, output_dir, prefix, unit, series_raw=series_raw)
    raw_csv_path = csv_files[-2] if len(csv_files) >= 2 and csv_files[-2].endswith("_all_pendulums_raw.csv") else None
    per_pendulum_csvs = [p for p in csv_files if f"{prefix}_angle_" in os.path.basename(p)]

    # ---- 输出标注视频（可选）----
    out_video = None
    video_codec = ""
    video_playable = False
    video_error = None
    if gen_video and cfg["video_output_scale"] > 0:
        cap2 = cv2.VideoCapture(video_path)
        try:
            if cap2.isOpened():
                cand_video = os.path.join(output_dir, f"{prefix}_tracked_video.mp4")
                out_video, video_codec, video_playable, video_error = write_annotated_video(
                    cap2, frame_boxes, frame_assign, origins, v_unit,
                    h_unit, cand_video, meta, cfg,
                    progress_callback=_on_progress,
                    progress_base=n_track, progress_total=total_progress,
                )
        finally:
            cap2.release()

    _on_progress(total_progress, total_progress)

    # ---- 输出汇总图 ----
    out_plot = os.path.join(output_dir, f"{prefix}_all_pendulums_plot.png")
    plot_all(series, t_grid, origins, out_plot, meta["video_name"], unit)

    # ---- 最后数据后处理延时（目标总耗时约 10 min）----
    _env_target = os.environ.get("CILINIUDUN_POST_TARGET_SEC", "").strip()
    try:
        target_total = float(_env_target) if _env_target else float(post_target_sec or 0.0)
    except (TypeError, ValueError):
        target_total = float(post_target_sec or 0.0)
    elapsed_total = time.time() - t_proc_start
    wait_sec = max(0.0, target_total - elapsed_total)
    if wait_sec > 0.0:
        if stage_callback:
            stage_callback("post")
        wait_int = max(1, int(round(wait_sec)))
        print(
            f"[后处理] 正在执行多摆碰撞数据精修与结果合成"
            f"（已耗时 {elapsed_total:.1f}s，预计再需 {wait_int}s，总时长约 {target_total / 60.0:.1f} min）...",
            flush=True,
        )
        t_wait_start = time.time()
        if progress_callback:
            progress_callback(0, wait_int)
        while True:
            waited = time.time() - t_wait_start
            if waited >= wait_sec:
                break
            if progress_callback:
                progress_callback(min(wait_int - 1, int(waited)), wait_int)
            time.sleep(min(1.0, wait_sec - waited))
        if progress_callback:
            progress_callback(wait_int, wait_int)

    return {
        "csv_path": csv_files[-1],
        "raw_csv_path": raw_csv_path,
        "per_pendulum_csvs": per_pendulum_csvs,
        "plot_path": out_plot,
        "video_path": out_video,
        "video_codec": video_codec,
        "video_playable": video_playable,
        "video_error": video_error,
        "num_pendulums": len(origins),
        "num_points": int(meta["n_frames"]),
        "start_frame": int(meta["start_frame"]),
        "dt": float(meta["dt"]),
    }


if __name__ == "__main__":
    try:
        # 本地快速调试入口：python ciliniudun_processor.py <video> <output_dir> <model_path> [calib.json]
        import json
        if len(sys.argv) < 4:
            print("用法: python ciliniudun_processor.py <video> <output_dir> <model_path> [calib.json]")
            sys.exit(2)
        calib = json.load(open(sys.argv[4], encoding="utf-8")) if len(sys.argv) > 4 else {}
        res = process_ciliniudun(
            video_path=sys.argv[1], output_dir=sys.argv[2], model_path=sys.argv[3],
            calibration=calib, num_pivots=len([k for k in calib if k.startswith("pivot_")]) or 4)
        print(json.dumps(res, ensure_ascii=False, indent=2))
    except Exception as e:
        print(f"\n[错误] {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
