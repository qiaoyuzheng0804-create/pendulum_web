"""共享工具：四个摆处理器收敛的公共函数。

只收录各处理器中逐字/等价重复的实现；数学公式保持与原实现位级一致
（角度函数的两种写法 atan2(-y,x) 与 -atan2(y,x) 在 IEEE 754 下恒等），
重构以"新 CLI 处理器输出与旧版逐字节一致"为验收标准。
"""

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
    return math.radians(ang) if unit == "rad" else ang


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


# ================================================================
#  共享 H.264 标注视频写入器（优先浏览器可内联播放的 H.264）
# ================================================================
_ffmpeg_cache = None
_ffmpeg_probed = False
_h264_probe_cache = {}


def _ffmpeg_exe():
    """本机可用的 ffmpeg 可执行文件；没有则返回 None（与 pi_manager 同一套探测）。"""
    global _ffmpeg_cache, _ffmpeg_probed
    if not _ffmpeg_probed:
        _ffmpeg_probed = True
        try:
            import imageio_ffmpeg
            _ffmpeg_cache = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            _ffmpeg_cache = None
    return _ffmpeg_cache


def video_readable(path):
    """视频能不能解码：写盘后自检用（打不开或读不出首帧都算坏）。"""
    try:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return False
        ok = bool(cap.read()[0])
        cap.release()
        return ok
    except Exception:
        return False


def _probe_cv_h264(api, fourcc, tag):
    """在临时文件上试写几帧再读回，判断某后端能否真的写出 H.264。"""
    import os
    import tempfile
    import uuid
    import numpy as np

    key = (api, tag)
    if key in _h264_probe_cache:
        return _h264_probe_cache[key]
    ok = False
    tmp = os.path.join(tempfile.gettempdir(), "pendulum_probe_%s.mp4" % uuid.uuid4().hex)
    writer = None
    try:
        writer = cv2.VideoWriter(tmp, api, cv2.VideoWriter_fourcc(*fourcc), 30.0, (320, 240))
        if writer.isOpened():
            for _ in range(3):
                writer.write(np.full((240, 320, 3), 90, np.uint8))
            writer.release()
            writer = None
            ok = os.path.getsize(tmp) > 512 and video_readable(tmp)
    except Exception:
        ok = False
    finally:
        if writer is not None:
            try:
                writer.release()
            except Exception:
                pass
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass
    _h264_probe_cache[key] = ok
    if not ok:
        print("[视频] 后端 %s 写不出 H.264（探针未通过）" % tag, flush=True)
    return ok


def _is_ascii_path(path):
    try:
        path.encode("ascii")
        return True
    except UnicodeEncodeError:
        return False


class H264VideoWriter:
    """逐帧写标注视频；codec 为 "h264"（浏览器可播放）或 "mp4v"（仅落盘）。"""

    def __init__(self, path, fps, width, height):
        import os
        import subprocess
        import tempfile
        import uuid

        self.path = str(path)
        self.fps = float(fps)
        self.width, self.height = int(width), int(height)
        self.codec = ""
        self.backend = ""
        self.error = None
        self.written = 0
        self._proc = None
        self._writer = None
        self._tmp_path = None
        self._errlog_path = None
        self._errlog = None

        exe = _ffmpeg_exe()
        if exe:
            try:
                self._errlog_path = self.path + ".ffmpeg.log"
                self._errlog = open(self._errlog_path, "wb")
                flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
                self._proc = subprocess.Popen(
                    [exe, "-hide_banner", "-loglevel", "error", "-y",
                     "-f", "rawvideo", "-pix_fmt", "bgr24",
                     "-s", "%dx%d" % (self.width, self.height),
                     "-framerate", "%.6f" % self.fps, "-i", "pipe:0",
                     "-an", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "20",
                     "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                     "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                     self.path],
                    stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                    stderr=self._errlog, creationflags=flags)
                self.backend, self.codec = "ffmpeg/libx264", "h264"
                return
            except Exception as exc:
                self._proc = None
                self._close_errlog()
                print("[视频] ffmpeg 启动失败（%s），改用其他编码器" % exc, flush=True)

        if os.name == "nt":
            api = getattr(cv2, "CAP_MSMF", None)
            if api is not None and _probe_cv_h264(api, "avc1", "MSMF/avc1"):
                target = self.path if _is_ascii_path(self.path) else \
                    os.path.join(tempfile.gettempdir(), "pendulum_annot_%s.mp4" % uuid.uuid4().hex)
                w = cv2.VideoWriter(target, api, cv2.VideoWriter_fourcc(*"avc1"),
                                    self.fps, (self.width, self.height))
                if w.isOpened():
                    self._writer = w
                    self._tmp_path = None if target == self.path else target
                    self.backend, self.codec = "MSMF/avc1", "h264"
                    return
                w.release()

        for fcc in ("avc1", "H264", "X264"):
            if _probe_cv_h264(cv2.CAP_ANY, fcc, "cv2/%s" % fcc):
                w = cv2.VideoWriter(self.path, cv2.CAP_ANY, cv2.VideoWriter_fourcc(*fcc),
                                    self.fps, (self.width, self.height))
                if w.isOpened():
                    self._writer = w
                    self.backend, self.codec = "cv2/%s" % fcc, "h264"
                    return
                w.release()

        self._writer = cv2.VideoWriter(self.path, cv2.VideoWriter_fourcc(*"mp4v"),
                                       self.fps, (self.width, self.height))
        self.backend, self.codec = "cv2/mp4v", "mp4v"
        if not self._writer.isOpened():
            self._writer = None
            self.error = "找不到可用的视频编码器"
            print("[视频] %s，标注视频将不会生成。" % self.error, flush=True)

    def isOpened(self):
        return (self._proc is not None) or (self._writer is not None and self._writer.isOpened())

    def write(self, frame):
        """写一帧（BGR）；失败返回 False 并把原因记进 self.error。"""
        import numpy as np

        if self._proc is not None:
            try:
                self._proc.stdin.write(np.ascontiguousarray(frame).tobytes())
                self.written += 1
                return True
            except Exception as exc:
                self.error = "写入 ffmpeg 失败: %s" % exc
                return False
        if self._writer is not None:
            try:
                self._writer.write(frame)
                self.written += 1
                return True
            except Exception as exc:
                self.error = "写入视频失败: %s" % exc
                return False
        return False

    def close(self):
        """收尾；返回错误信息（None 表示正常）。"""
        import shutil

        if self._proc is not None:
            proc, self._proc = self._proc, None
            try:
                if proc.stdin is not None:
                    proc.stdin.close()
                code = proc.wait(timeout=120)
            except Exception as exc:
                try:
                    proc.kill()
                except Exception:
                    pass
                self._close_errlog()
                return "ffmpeg 收尾失败: %s" % exc
            detail = "" if code == 0 else self._read_errlog()
            self._close_errlog()
            if code != 0:
                return "ffmpeg 退出码 %s%s" % (code, ("：" + detail) if detail else "")
            return None

        if self._writer is not None:
            self._writer.release()
            self._writer = None
            if self._tmp_path:
                tmp, self._tmp_path = self._tmp_path, None
                try:
                    shutil.move(tmp, self.path)
                except Exception as exc:
                    self.error = "临时视频搬回 %s 失败: %s" % (self.path, exc)
                    self.path = tmp
                    return self.error
            return None
        return self.error

    def _read_errlog(self):
        try:
            with open(self._errlog_path, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()
        except Exception:
            return ""
        return " / ".join(ln.strip() for ln in text.splitlines() if ln.strip())[-400:]

    def _close_errlog(self):
        import os

        if self._errlog is not None:
            try:
                self._errlog.close()
            except Exception:
                pass
            self._errlog = None
        try:
            if self._errlog_path and os.path.exists(self._errlog_path):
                os.remove(self._errlog_path)
        except Exception:
            pass

