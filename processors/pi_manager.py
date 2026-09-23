"""树莓派 5 摄像头管理器：局域网 JPEG 帧流（RPJ1 协议）。

对端：树莓派套件 raspberry/capture_stream.py（start_stream.sh 走的是 --no-record 纯传输模式，
树莓派本地不落任何文件）。每个帧包的线协议是：

    24 字节头 struct("!4sIQQHH") = magic b"RPJ1" / JPEG 字节数 / 帧号 /
                                   传感器时间戳(ns) / 宽 / 高
    紧随其后是 JPEG 负载

与 openmv_manager 的分工：那边是 USB 串口取帧（USBDBG V1），这边是 TCP 收帧。
摄像头留在树莓派上、YOLO 跑在本机，所以帧要整段搬过来 —— 这也是为什么录制的落盘
不在这里拼时间戳，而是走「恒定帧率网格」（CFR）：

    收到的帧按 sensor_timestamp_ns 折算成第几个 1/fps 时间格，
    空格用上一帧补齐（filled）、同一格多出来的帧丢弃（late）。

这样写出的视频「容器帧率 == 真实时间轴」，四个处理器里既有的「帧号 ÷ 帧率」时间基准
不用改一行就是对的；反过来，如果按到达顺序原样写盘，树莓派 JPEG 编码跟不上 60 FPS 时
（实测会掉到 40 上下）整条时间轴会被整体压缩，阻尼系数直接错。
逐帧时间戳另外落一份 CSV（配对文件），供后续要按真实采样时刻拟合时使用。
"""

from __future__ import annotations

import csv
import os
import queue
import socket
import struct
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

# ─── 线协议常量（与树莓派端 capture_stream.py 保持一致）─────────────────────
# 现行（云台版套件）RPJ2：33 字节头，帧头里顺带回传 pan/tilt 脉宽与限位码
#   4s magic / I JPEG长度 / Q 帧号 / Q 传感器时间戳ns / H 宽 / H 高
#   / H pan_pulse_us / H tilt_pulse_us / B limit_code
# 旧版（无云台）RPJ1 仍接受：24 字节头，只有前 6 个字段。

PI_MAGIC = b"RPJ2"
PI_HEADER = struct.Struct("!4sIQQHHHHB")
PI_MAGIC_LEGACY = b"RPJ1"
PI_HEADER_LEGACY = struct.Struct("!4sIQQHH")
MAX_JPEG_BYTES = 20 * 1024 * 1024
DEFAULT_PORT = 9000

# 云台限位码（树莓派端 LIMIT_* 常量）
LIMIT_NONE = 0
LIMIT_TEXT = {
    0: "READY",
    1: "PAN MIN LIMIT",
    2: "PAN MAX LIMIT",
    3: "TILT MIN LIMIT",
    4: "TILT MAX LIMIT",
}
GIMBAL_COMMANDS = ("U", "D", "L", "R", "C")   # 单字节命令，直接写回同一条 TCP
LEGACY_GIMBAL_ERROR = "当前树莓派端是旧版协议（RPJ1，无云台），请先更新树莓派端代码。"

READ_TIMEOUT_S = 5.0      # 单次 recv 等待上限：超时视为断流，重连
RECONNECT_DELAY_S = 1.0
FPS_WINDOW = 30           # 用最近 N 帧的传感器时间戳估算实际帧率

# 断流后最多补这么多秒的重复帧；超出部分压缩掉并记进 trimmed
# （>3 秒的断流实验本身已经废了，宁可记账也不要凭空多出几百帧重复画面）
GAP_CAP_SECONDS = 3.0
REC_QUEUE_MAX = 240       # 收帧线程 → 写盘线程的有界队列（满则丢最新的，网格会补重复帧）

_ffmpeg_cache: Optional[str] = None
_ffmpeg_probed = False


def _ffmpeg_exe() -> Optional[str]:
    """本机可用的 ffmpeg 可执行文件；没有则返回 None（录制退回 cv2 mp4v）。"""
    global _ffmpeg_cache, _ffmpeg_probed
    if not _ffmpeg_probed:
        _ffmpeg_probed = True
        try:
            import imageio_ffmpeg
            _ffmpeg_cache = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            _ffmpeg_cache = None
    return _ffmpeg_cache


@dataclass(frozen=True)
class FramePacket:
    """一帧：JPEG 负载 + 树莓派侧的帧号与传感器时间戳（RPJ2 还带云台状态）。"""
    frame_id: int
    timestamp_ns: int
    width: int
    height: int
    jpeg: bytes
    pan_us: Optional[int] = None
    tilt_us: Optional[int] = None
    limit_code: Optional[int] = None


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    """收满 size 字节；对端关闭或超时都抛错，由调用方重连。"""
    data = bytearray()
    while len(data) < size:
        chunk = sock.recv(size - len(data))
        if not chunk:
            raise ConnectionError("树莓派关闭了视频流连接")
        data.extend(chunk)
    return bytes(data)


# ─── 视频落盘 ──────────────────────────────────────────────────────────────────

class _VideoSink:
    """把一帧帧 JPEG 写成 MP4。

    优先用 ffmpeg 直接转 H.264：转出来的 MP4 浏览器能内联播放，标定窗口的
    "实时预览"走原生 <video>（mp4v 那条路浏览器不认，只能退回服务端逐帧取图）。
    没有 imageio-ffmpeg 时退回 cv2.VideoWriter(mp4v)，行为与 openmv_manager 一致。
    """

    def __init__(self, path: str, fps: float, width: int, height: int) -> None:
        self.path = path
        self.fps = float(fps)
        self.width = int(width)
        self.height = int(height)
        self.error: Optional[str] = None
        self.backend = ""
        self._proc: Optional[subprocess.Popen] = None
        self._writer: Optional[cv2.VideoWriter] = None
        self._errlog_path = path + ".ffmpeg.log"
        self._errlog = None

        exe = _ffmpeg_exe()
        if exe:
            command = [
                exe, "-hide_banner", "-loglevel", "error", "-y",
                "-f", "mjpeg", "-framerate", "%.6f" % self.fps, "-i", "pipe:0",
                "-an", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "20",
                "-pix_fmt", "yuv420p",
                # 奇数边长会让 libx264 直接报错，偶数化一下（正常 1280x720 是恒等变换）
                "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                path,
            ]
            flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            try:
                # stderr 落文件而不是 PIPE：PIPE 满了会把 ffmpeg 卡死进而卡死写盘线程
                self._errlog = open(self._errlog_path, "wb")
                self._proc = subprocess.Popen(
                    command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                    stderr=self._errlog, creationflags=flags)
                self.backend = "ffmpeg/h264"
                return
            except Exception as exc:
                self._proc = None
                self._close_errlog()
                self.error = f"ffmpeg 启动失败（{exc}），改用 mp4v"
                print(f"[Pi] {self.error}", flush=True)

        self.backend = "cv2/mp4v"
        self._writer = cv2.VideoWriter(
            path, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (self.width, self.height))
        if not self._writer.isOpened():
            self._writer = None
            raise RuntimeError("无法创建录制文件（ffmpeg 不可用且 OpenCV 编码器初始化失败）")

    def write(self, jpeg: bytes) -> bool:
        """写一帧；失败返回 False 并把原因记进 self.error。"""
        if self._proc is not None:
            try:
                assert self._proc.stdin is not None
                self._proc.stdin.write(jpeg)
                return True
            except Exception as exc:
                self.error = f"写入 ffmpeg 失败: {exc}"
                return False
        if self._writer is not None:
            frame = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                return True  # 单帧解码失败不该终止整段录制
            if frame.shape[1] != self.width or frame.shape[0] != self.height:
                frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
            try:
                self._writer.write(frame)
                return True
            except Exception as exc:
                self.error = f"写入视频失败: {exc}"
                return False
        return False

    def close(self) -> Optional[str]:
        """正常收尾；返回错误信息，None 表示成功。"""
        if self._proc is not None:
            proc, self._proc = self._proc, None
            try:
                if proc.stdin is not None:
                    proc.stdin.close()
                code = proc.wait(timeout=30)
            except Exception as exc:
                try:
                    proc.kill()
                except Exception:
                    pass
                self._close_errlog()
                return f"ffmpeg 收尾失败: {exc}"
            detail = ""
            if code != 0:
                detail = self._read_errlog()
            self._close_errlog()
            if code != 0:
                return "ffmpeg 退出码 %s%s" % (code, ("：" + detail) if detail else "")
            return None
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        return None

    def _read_errlog(self) -> str:
        try:
            text = Path(self._errlog_path).read_text(encoding="utf-8", errors="replace")
        except Exception:
            return ""
        return " / ".join(line.strip() for line in text.splitlines() if line.strip())[-400:]

    def _close_errlog(self) -> None:
        if self._errlog is not None:
            try:
                self._errlog.close()
            except Exception:
                pass
            self._errlog = None
        # 成功路径上也删掉日志，别在实验产物里留垃圾
        try:
            if os.path.exists(self._errlog_path):
                os.remove(self._errlog_path)
        except Exception:
            pass


# ─── 收帧线程 ──────────────────────────────────────────────────────────────────

class _ReceiverThread(threading.Thread):
    """TCP 收帧：连接 → 逐包解析 → 交给回调；断开后退避重连，直到 stop。

    云台命令走的是同一条连接（反向单字节），所以这里要留着 socket 供发送线程用。
    """

    def __init__(self, host: str, port: int, stop_event: threading.Event, on_frame) -> None:
        super().__init__(name="pi-receiver", daemon=True)
        self.host = host
        self.port = int(port)
        self.stop_event = stop_event
        self._on_frame = on_frame
        self.connected = False
        self.last_error: Optional[str] = None
        self.total_frames = 0
        self.last_frame_time = 0.0
        self.width = 0
        self.height = 0
        # 云台状态（RPJ2 每帧回传；RPJ1 恒为 None）
        self.protocol = ""
        self.pan_us: Optional[int] = None
        self.tilt_us: Optional[int] = None
        self.limit_code: Optional[int] = None
        self._socket: Optional[socket.socket] = None

    def run(self) -> None:
        while not self.stop_event.is_set():
            try:
                with socket.create_connection((self.host, self.port), timeout=5.0) as sock:
                    sock.settimeout(READ_TIMEOUT_S)
                    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    self._socket = sock
                    self.connected = True
                    self.last_error = None
                    print(f"[Pi] 已连接树莓派视频流 {self.host}:{self.port}", flush=True)
                    self._read_stream(sock)
            except Exception as exc:
                self.last_error = str(exc) or exc.__class__.__name__
            finally:
                self._socket = None
                self.connected = False
                self.pan_us = self.tilt_us = self.limit_code = None
            if not self.stop_event.is_set():
                print(f"[Pi] 视频流中断（{self.last_error}），{RECONNECT_DELAY_S:g}s 后重连", flush=True)
                self.stop_event.wait(RECONNECT_DELAY_S)

    def send(self, data: bytes) -> bool:
        """往同一条连接反向写命令（云台用）；未连接或写失败返回 False。"""
        sock = self._socket
        if sock is None or not self.connected:
            return False
        try:
            sock.sendall(data)
            return True
        except OSError:
            return False

    def _read_stream(self, sock: socket.socket) -> None:
        while not self.stop_event.is_set():
            # 先读 4 字节 magic 再决定是哪种帧头：新旧套件的头长度不同（33 / 24 字节）
            magic = _recv_exact(sock, 4)
            pan_us = tilt_us = limit_code = None
            if magic == PI_MAGIC:
                (_, size, frame_id, ts_ns, width, height,
                 pan_us, tilt_us, limit_code) = PI_HEADER.unpack(
                    magic + _recv_exact(sock, PI_HEADER.size - 4))
                if self.protocol != "RPJ2":
                    self.protocol = "RPJ2"
                    print("[Pi] 协议 RPJ2（含云台状态）", flush=True)
            elif magic == PI_MAGIC_LEGACY:
                (_, size, frame_id, ts_ns, width, height) = PI_HEADER_LEGACY.unpack(
                    magic + _recv_exact(sock, PI_HEADER_LEGACY.size - 4))
                if self.protocol != "RPJ1":
                    self.protocol = "RPJ1"
                    print("[Pi] 检测到旧版协议 RPJ1（无云台字段），云台控制不可用", flush=True)
            else:
                raise ValueError(f"视频流头部标识异常（{magic!r}），请确认树莓派端跑的是 capture_stream.py")
            if not 0 < size <= MAX_JPEG_BYTES:
                raise ValueError(f"JPEG 长度异常: {size}")

            payload = _recv_exact(sock, size)
            self.total_frames += 1
            self.last_frame_time = time.monotonic()
            self.width, self.height = int(width), int(height)
            if pan_us is not None:
                self.pan_us, self.tilt_us, self.limit_code = int(pan_us), int(tilt_us), int(limit_code)
            self._on_frame(FramePacket(int(frame_id), int(ts_ns), int(width), int(height),
                                       payload, pan_us, tilt_us, limit_code))


# ─── 管理器 ────────────────────────────────────────────────────────────────────

class PiCameraManager:
    """树莓派摄像头的连接 / 预览 / 录制（线程安全）。

    与 OpenMVManager 的公开接口保持同一形状（connect_xxx / get_status /
    get_latest_jpeg / start_recording / stop_recording / disconnect），
    让 app.py 与前端能以相同的节奏处理两种相机。
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._receiver: Optional[_ReceiverThread] = None
        self._host = ""
        self._port = DEFAULT_PORT

        self._jpeg_lock = threading.Lock()
        self._latest_jpeg: Optional[bytes] = None
        self._latest_frame: Optional[FramePacket] = None

        # 云台指令展示用计数（真值以帧头回传的 pan/tilt 为准）
        self._gimbal_last_command = ""
        self._gimbal_command_count = 0

        self._ts_lock = threading.Lock()
        self._ts_window: List[int] = []

        # 录制状态：_rec_lock 只保护标志与路径，绝不跨 join 持有
        self._rec_lock = threading.Lock()
        self._recording = False
        self._rec_thread: Optional[threading.Thread] = None
        self._rec_queue: Optional[queue.Queue] = None
        self._rec_stop = threading.Event()
        self._rec_sink: Optional[_VideoSink] = None
        self._rec_upload_dir = ""
        self._rec_tmp_path: Optional[str] = None
        self._rec_csv_path: Optional[str] = None
        self._rec_error: Optional[str] = None
        self._record_fps = 30.0
        self._rec_start_time = 0.0
        self._rec_frames = 0      # 写进视频的帧数（含补齐的重复帧）
        self._rec_samples = 0     # 真实收到的样本数
        self._rec_filled = 0      # 补齐的重复帧数
        self._rec_late = 0        # 迟到（时间格已被占）而丢弃的帧数
        self._rec_dropped = 0     # 写盘队列满而丢弃的帧数
        self._rec_trimmed = 0     # 断流超过上限、被压缩掉的时间格数
        self._rec_t0_ns: Optional[int] = None
        self._rec_ts_first = 0
        self._rec_ts_last = 0
        # 录制期间云台姿态的取值范围（中途动过云台 → 这段的标定作废）
        self._rec_pan_min: Optional[int] = None
        self._rec_pan_max: Optional[int] = None
        self._rec_tilt_min: Optional[int] = None
        self._rec_tilt_max: Optional[int] = None

    # ── 连接 ───────────────────────────────────────────────────────────────

    @property
    def connected(self) -> bool:
        return self._receiver is not None and self._receiver.connected

    @property
    def recording(self) -> bool:
        return self._recording

    def connect(self, host: str, port: int = DEFAULT_PORT) -> dict:
        """连上树莓派视频流，等到第一帧到达才算成功。"""
        host = (host or "").strip()
        if not host:
            return {"error": "请填写树莓派地址（IP 或 raspberrypi.local）。"}
        try:
            port = int(port)
        except (TypeError, ValueError):
            return {"error": "端口必须是数字。"}
        if not (1 <= port <= 65535):
            return {"error": "端口超出 1~65535 范围。"}

        with self._lock:
            if self.connected:
                return {"error": f"已连接 {self._host}:{self._port}，请先断开。"}
            self._stop.clear()
            self._latest_jpeg = None
            self._latest_frame = None
            with self._ts_lock:
                self._ts_window = []
            self._host, self._port = host, port
            receiver = _ReceiverThread(host, port, self._stop, self._on_frame)
            self._receiver = receiver
            receiver.start()

            # 等第一帧：连上了但没画面（树莓派端没启动传输）与连不上要分开报
            deadline = time.monotonic() + 8.0
            while time.monotonic() < deadline:
                time.sleep(0.1)
                if receiver.total_frames > 0:
                    print(f"[Pi] 首帧到达 {receiver.width}x{receiver.height}", flush=True)
                    return {"success": True, "host": host, "port": port,
                            "width": receiver.width, "height": receiver.height}
                if receiver.last_error and not receiver.connected:
                    break
                if not receiver.is_alive():
                    break

            err = receiver.last_error or "等待画面超时"
            reached = receiver.connected
            self._cleanup_receiver()
            if reached:
                return {"error": f"已连上 {host}:{port} 但没收到画面，请确认树莓派端已开始传输。"}
            return {"error": f"无法连接 {host}:{port}：{err}"}

    def disconnect(self) -> dict:
        """断开视频流；正在录制的话先正常收尾落盘，不丢已收到的数据。"""
        note = ""
        if self._recording or self._rec_thread is not None:
            result = self.stop_recording()
            if result.get("success"):
                note = f"录制中断，已保存 {result['frame_count']} 帧到 {result['video_path']}"
                print(f"[Pi] {note}", flush=True)
        with self._lock:
            if self._receiver is not None:
                self._stop.set()
                self._receiver.join(timeout=3.0)
                if self._receiver.is_alive():
                    print("[Pi] 收帧线程未在 3s 内退出", flush=True)
            self._cleanup_receiver()
        out = {"success": True}
        if note:
            out["note"] = note
        return out

    def _cleanup_receiver(self) -> None:
        self._stop.set()
        self._receiver = None
        with self._jpeg_lock:
            self._latest_jpeg = None
            self._latest_frame = None

    # ── 状态 / 预览 ────────────────────────────────────────────────────────

    def get_status(self) -> dict:
        result: dict = {
            "connected": self.connected,
            "host": self._host,
            "port": self._port,
            "recording": self._recording,
            "fps": 0.0,
            "total_frames": 0,
            "width": 0,
            "height": 0,
            "last_error": None,
        }
        receiver = self._receiver
        if receiver is not None:
            result.update({
                "total_frames": receiver.total_frames,
                "width": receiver.width,
                "height": receiver.height,
                "last_error": receiver.last_error,
                "protocol": receiver.protocol,
                "last_frame_age": (round(time.monotonic() - receiver.last_frame_time, 2)
                                   if receiver.last_frame_time else None),
            })
            if receiver.pan_us is not None:
                result.update({
                    "gimbal_pan_us": receiver.pan_us,
                    "gimbal_tilt_us": receiver.tilt_us,
                    "gimbal_limit_code": receiver.limit_code,
                    "gimbal_limit": LIMIT_TEXT.get(int(receiver.limit_code or 0), ""),
                    "gimbal_last_command": self._gimbal_last_command,
                    "gimbal_command_count": self._gimbal_command_count,
                })
        with self._ts_lock:
            window = list(self._ts_window)
        # 实际帧率只认传感器时间戳，不认"到达速率"（网络上会抖）
        if len(window) >= 2 and window[-1] > window[0]:
            result["fps"] = round((len(window) - 1) * 1e9 / (window[-1] - window[0]), 1)
        if self._recording:
            result.update({
                "record_elapsed": round(time.monotonic() - self._rec_start_time, 1),
                "recorded_frames": self._rec_frames,
                "record_samples": self._rec_samples,
                "record_filled": self._rec_filled,
                "record_dropped": self._rec_dropped,
                "record_fps": self._record_fps,
            })
        return result

    def get_latest_jpeg(self) -> Optional[bytes]:
        """最新一帧的原始 JPEG（树莓派已经编过码，这里直接转发，不再重编码）。"""
        with self._jpeg_lock:
            return self._latest_jpeg

    def _on_frame(self, frame: FramePacket) -> None:
        """收帧线程回调：刷新预览、记时间戳窗口，录制中则折算时间格入队。"""
        with self._ts_lock:
            self._ts_window.append(frame.timestamp_ns)
            if len(self._ts_window) > FPS_WINDOW:
                del self._ts_window[0]
        with self._jpeg_lock:
            self._latest_jpeg = frame.jpeg
            self._latest_frame = frame
        if not self._recording:
            return
        rec_queue = self._rec_queue
        if rec_queue is None:
            return
        if self._rec_t0_ns is None:
            self._rec_t0_ns = frame.timestamp_ns
            self._rec_ts_first = frame.timestamp_ns
        self._rec_ts_last = frame.timestamp_ns
        slot = int(round((frame.timestamp_ns - self._rec_t0_ns) / 1e9 * self._record_fps))
        if slot < 0:
            return
        try:
            rec_queue.put_nowait((slot, frame))
        except queue.Full:
            # 写盘跟不上：丢最新的，网格会用上一帧补格，时间轴仍然正确
            self._rec_dropped += 1

    # ── 云台（二维 MG90S，树莓派 GPIO 硬件 PWM）─────────────────────────────
    # 命令不走串口：往收帧的同一条 TCP 连接反向写一个字节（U/D/L/R/C），
    # 树莓派端 gimbal-commands 线程 recv 到就执行。方向/步长/软限位都在树莓派侧
    # （start_stream.sh 已带 --pan-invert --tilt-invert，方向即画面方向），
    # 本机只发方向键，拿回的是每帧头部里的 pan/tilt 脉宽与限位码。

    def send_gimbal_command(self, command: str) -> dict:
        cmd = (command or "").strip().upper()
        if cmd not in GIMBAL_COMMANDS:
            return {"error": f"无效命令 '{command}'，支持: {', '.join(GIMBAL_COMMANDS)}"}
        receiver = self._receiver
        if receiver is None or not self.connected:
            return {"error": "树莓派未连接。"}
        if receiver.protocol == "RPJ1":
            return {"error": LEGACY_GIMBAL_ERROR}
        if not receiver.send(cmd.encode("ascii")):
            return {"error": "云台指令发送失败，连接可能已断开。"}
        self._gimbal_command_count += 1
        self._gimbal_last_command = cmd
        return {"success": True, "command": cmd}

    # ── 录制 ───────────────────────────────────────────────────────────────

    def start_recording(self, upload_dir: str, fps: float = 30.0) -> dict:
        """开始录制：帧写到临时文件，停止时才定名（中途出错就丢掉残片）。"""
        if not self.connected:
            return {"error": "树莓派未连接。"}
        try:
            fps = float(fps)
        except (TypeError, ValueError):
            return {"error": "录制帧率必须是数字。"}
        if not (1.0 <= fps <= 240.0):
            return {"error": "录制帧率需在 1~240 之间。"}

        with self._rec_lock:
            if self._recording or self._rec_thread is not None:
                return {"error": "已在录制中。"}
            frame = self._latest_frame
            if frame is None:
                return {"error": "还没收到画面，无法开始录制。"}

            stamp = int(time.time())
            self._rec_upload_dir = str(upload_dir)
            self._rec_tmp_path = str(Path(upload_dir) / f"pi_rec_{stamp}_tmp.mp4")
            self._rec_csv_path = str(Path(upload_dir) / f"pi_frames_{stamp}_tmp.csv")
            self._record_fps = fps
            self._rec_error = None
            self._rec_frames = self._rec_samples = self._rec_filled = 0
            self._rec_late = self._rec_dropped = self._rec_trimmed = 0
            self._rec_t0_ns = None
            self._rec_ts_first = self._rec_ts_last = 0
            self._rec_pan_min = self._rec_pan_max = None
            self._rec_tilt_min = self._rec_tilt_max = None
            try:
                sink = _VideoSink(self._rec_tmp_path, fps, frame.width, frame.height)
            except Exception as exc:
                self._rec_upload_dir = ""
                self._rec_tmp_path = self._rec_csv_path = None
                return {"error": str(exc)}

            self._rec_sink = sink
            self._rec_queue = queue.Queue(maxsize=REC_QUEUE_MAX)
            self._rec_stop.clear()
            self._rec_start_time = time.monotonic()
            self._recording = True
            self._rec_thread = threading.Thread(
                target=self._record_loop, args=(sink,), name="pi-writer", daemon=True)
            self._rec_thread.start()
            print(f"[Pi] 开始录制 {frame.width}x{frame.height} @ {fps:g} FPS（{sink.backend}）", flush=True)
            return {"success": True, "fps": fps, "width": frame.width, "height": frame.height,
                    "backend": sink.backend}

    def stop_recording(self, upload_dir: Optional[str] = None) -> dict:
        """停止录制：等写盘线程收尾，把临时文件定名成最终产物。"""
        with self._rec_lock:
            if not self._recording and self._rec_thread is None:
                return {"error": "未在录制中。"}
            self._recording = False
            thread, self._rec_thread = self._rec_thread, None
            rec_queue = self._rec_queue
            upload_dir = upload_dir or self._rec_upload_dir
        self._rec_stop.set()
        if thread is not None:
            thread.join(timeout=60)
            if thread.is_alive():
                print("[Pi] 写盘线程未在 60s 内退出", flush=True)
        # 写盘线程若被超时中断，队列里可能还压着帧；这段视频不再定名，残余直接丢掉
        if rec_queue is not None:
            while True:
                try:
                    rec_queue.get_nowait()
                except queue.Empty:
                    break
        with self._rec_lock:
            tmp = self._rec_tmp_path
            csv_tmp = self._rec_csv_path
            count, samples = self._rec_frames, self._rec_samples
            filled, late = self._rec_filled, self._rec_late
            dropped, trimmed = self._rec_dropped, self._rec_trimmed
            err = self._rec_error
            ts_first, ts_last = self._rec_ts_first, self._rec_ts_last
            pan_min, pan_max = self._rec_pan_min, self._rec_pan_max
            tilt_min, tilt_max = self._rec_tilt_min, self._rec_tilt_max
            used_fps = self._record_fps
            self._rec_queue = None
            self._rec_sink = None
            self._rec_tmp_path = self._rec_csv_path = None

        def _drop_temp() -> None:
            for path in (tmp, csv_tmp):
                if path and os.path.exists(path):
                    try:
                        os.remove(path)
                    except Exception:
                        pass

        if err:
            _drop_temp()
            return {"error": err}
        if count == 0:
            _drop_temp()
            return {"error": "录制帧数为 0，未保存视频。"}

        stamp = int(time.time())
        video_name = f"pi_capture_{stamp}.mp4"
        video_path = str(Path(upload_dir) / video_name)
        csv_name = f"pi_capture_{stamp}_frames.csv"
        csv_path = str(Path(upload_dir) / csv_name)
        try:
            os.replace(tmp, video_path)
            if csv_tmp and os.path.exists(csv_tmp):
                os.replace(csv_tmp, csv_path)
        except Exception as exc:
            return {"error": f"录制文件落盘失败: {exc}"}

        real_duration = (ts_last - ts_first) / 1e9 if ts_last > ts_first else 0.0
        spans = [d for d in (
            None if pan_min is None or pan_max is None else pan_max - pan_min,
            None if tilt_min is None or tilt_max is None else tilt_max - tilt_min,
        ) if d is not None]
        gimbal_span = max(spans) if spans else 0
        return {
            "success": True,
            "video_path": video_path,
            "video_name": video_name,
            "frame_count": count,                       # 视频总帧数（含补齐的重复帧）
            "duration": round(count / used_fps, 2) if used_fps else 0.0,
            "fps": used_fps,
            "samples": samples,                         # 真实收到的样本数
            "filled": filled,                           # 补齐重复帧数
            "late": late,
            "dropped": dropped,
            "trimmed": trimmed,                         # 断流压缩掉的时间格数（>0 说明时间轴有断点）
            "real_duration": round(real_duration, 2),   # 传感器时间戳算出的真实时长
            "metadata_path": csv_path if os.path.exists(csv_path) else "",
            # 录制期间云台动过 → 相机姿态变过，这一段标定不能用了
            "gimbal_span_us": gimbal_span,
            "gimbal_moved": gimbal_span > 0,
        }

    def _record_loop(self, sink: _VideoSink) -> None:
        """写盘线程：按时间格补帧并落盘，同时写逐帧时间戳 CSV。"""
        rec_queue = self._rec_queue
        if rec_queue is None:
            return
        gap_cap = max(1, int(round(self._record_fps * GAP_CAP_SECONDS)))
        next_slot = 0
        last_jpeg: Optional[bytes] = None
        csv_file = None
        try:
            csv_file = open(self._rec_csv_path, "w", newline="", encoding="utf-8-sig")
            writer = csv.DictWriter(csv_file, fieldnames=(
                "video_frame", "frame_id", "sensor_timestamp_ns",
                "elapsed_s", "video_time_s", "filled_before",
                "pan_pulse_us", "tilt_pulse_us", "gimbal_limit_code"))
            writer.writeheader()

            while True:
                try:
                    item = rec_queue.get(timeout=0.2)
                except queue.Empty:
                    if self._rec_stop.is_set():
                        break
                    continue
                slot, frame = item
                if slot < next_slot:
                    self._rec_late += 1      # 它的时间格已经过去了，丢掉
                    continue

                fill = slot - next_slot
                if fill > 0 and last_jpeg is None:
                    # 还没有可复制的画面（首帧之前的空档没法补）
                    self._rec_trimmed += fill
                    fill = 0
                if fill > gap_cap:
                    self._rec_trimmed += fill - gap_cap
                    fill = gap_cap
                filled_ok = True
                for _ in range(fill):
                    if not sink.write(last_jpeg):
                        filled_ok = False
                        break
                if not filled_ok:
                    self._rec_error = sink.error or "视频写入失败"
                    break
                self._rec_filled += fill
                if not sink.write(frame.jpeg):
                    self._rec_error = sink.error or "视频写入失败"
                    break

                self._rec_frames += fill + 1
                next_slot = slot + 1        # 网格对齐到传感器钟；被压缩的部分记在 trimmed
                last_jpeg = frame.jpeg
                self._rec_samples += 1
                if frame.pan_us is not None:
                    self._rec_pan_min = frame.pan_us if self._rec_pan_min is None else min(self._rec_pan_min, frame.pan_us)
                    self._rec_pan_max = frame.pan_us if self._rec_pan_max is None else max(self._rec_pan_max, frame.pan_us)
                    self._rec_tilt_min = frame.tilt_us if self._rec_tilt_min is None else min(self._rec_tilt_min, frame.tilt_us)
                    self._rec_tilt_max = frame.tilt_us if self._rec_tilt_max is None else max(self._rec_tilt_max, frame.tilt_us)
                writer.writerow({
                    "video_frame": self._rec_frames,
                    "frame_id": frame.frame_id,
                    "sensor_timestamp_ns": frame.timestamp_ns,
                    "elapsed_s": "%.9f" % ((frame.timestamp_ns - self._rec_t0_ns) / 1e9),
                    "video_time_s": "%.9f" % ((self._rec_frames - 1) / self._record_fps),
                    "filled_before": fill,
                    # 相机姿态逐帧留档：中途动过云台，该段标定就作废了（与套件 CSV 同名列）
                    "pan_pulse_us": "" if frame.pan_us is None else frame.pan_us,
                    "tilt_pulse_us": "" if frame.tilt_us is None else frame.tilt_us,
                    "gimbal_limit_code": "" if frame.limit_code is None else frame.limit_code,
                })
                if self._rec_samples % 30 == 0:
                    csv_file.flush()

                if self._rec_stop.is_set() and rec_queue.empty():
                    break
        except Exception as exc:
            self._rec_error = f"写盘线程异常: {exc}"
        finally:
            if csv_file is not None:
                try:
                    csv_file.flush()
                    csv_file.close()
                except Exception:
                    pass
            close_error = sink.close()
            if close_error and not self._rec_error:
                self._rec_error = close_error


# ─── 全局单例 ──────────────────────────────────────────────────────────────────

pi_manager = PiCameraManager()
