"""共享工具：四个摆处理器收敛的公共函数。

只收录各处理器中逐字/等价重复的实现；数学公式保持与原实现位级一致
（角度函数的两种写法 atan2(-y,x) 与 -atan2(y,x) 在 IEEE 754 下恒等），
重构以"新 CLI 处理器输出与旧版逐字节一致"为验收标准。
"""

import os as _os
_os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import math
import os

import cv2
import numpy as np


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """归一化；零向量返回原向量副本（不抛错）。"""
    n = np.linalg.norm(v)
    if n < eps:
        return np.asarray(v).copy()
    return np.asarray(v) / n


def normalize_strict(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """归一化；零向量抛 ValueError（标定类用途需要显式失败）。"""
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < eps:
        raise ValueError("两点重合，无法确定方向。")
    return v / n


def make_odd(n: int) -> int:
    return n if n % 2 == 1 else n + 1


def valid_savgol_window(n: int, desired_window: int, polyorder: int) -> int:
    """返回长度 n 序列上合法的 Savitzky-Golay 窗口；序列太短返回 0（调用方跳过滤波）。"""
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


def angle_from_vertical(v_unit, radius_vec, right_positive: bool = True, unit: str = "deg") -> float:
    """检测框中心相对竖直方向的偏角（danbai 原实现 signed_angle_deg_right_positive
    的逐位等价版本：相同的 normalize → cross/dot → clip → atan2 操作序列）。

    最低点（连线与 v_unit 同向）为 0，返回值右偏（right_positive=True）为正。
    注意：ciliniudun 的 compute_angle 是无 normalize 的另一浮点序列，不要合并。
    """
    r = normalize(np.asarray(radius_vec).astype(np.float64))
    v = normalize(np.asarray(v_unit).astype(np.float64))
    cross_z = v[0] * r[1] - v[1] * r[0]
    dot_vr = np.clip(v[0] * r[0] + v[1] * r[1], -1.0, 1.0)
    ang = math.degrees(math.atan2(cross_z, dot_vr))
    ang = -ang
    if not right_positive:
        ang = -ang
    return ang if unit == "rad" else ang


def best_detection_center(results, target_class_ids=None):
    """取置信度最高的检测框中心。

    返回 (cx, cy, conf)；无检测返回 None。target_class_ids 为 None 不过滤，
    否则只保留类别 ID 在集合内的框。
    """
    if not results or len(results) == 0:
        return None
    r = results[0]
    if r.boxes is None or len(r.boxes) == 0:
        return None
    bx = r.boxes.xywh.cpu().numpy()
    bc = r.boxes.conf.cpu().numpy()
    bcl = r.boxes.cls.cpu().numpy()
    best = -1.0
    cx = cy = None
    for box, conf, cls in zip(bx, bc, bcl):
        if target_class_ids is not None and int(cls) not in target_class_ids:
            continue
        if conf > best:
            best = float(conf)
            cx, cy = float(box[0]), float(box[1])
    return (cx, cy, best) if cx is not None else None


def open_video(video_path: str, default_fps=None):
    """打开视频并返回 (cap, fps, total_frames)。

    fps 读不到时：default_fps 为 None 则报错（原 cizuni 行为），
    否则用 default_fps（原 danbai/niubai 的 30.0 兜底行为）。
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps is None or fps <= 1e-6:
        if default_fps is None:
            cap.release()
            raise RuntimeError("Cannot read FPS")
        fps = default_fps
    return cap, fps, total_frames


def setup_matplotlib_chinese():
    """让 matplotlib 优先使用中文字体；返回是否找到（找不到时调用方应使用英文文案）。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm

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
