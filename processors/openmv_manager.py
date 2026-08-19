"""OpenMV camera manager: USBDBG V1 frame capture, recording, and gimbal control.

Provides a thread-safe singleton that manages the OpenMV serial connection,
continuously reads JPEG frames, supports MJPEG streaming to browsers, records
frames to MP4 video files, and sends gimbal servo commands via TX_INPUT.
"""

from __future__ import annotations

import io
import queue
import struct
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

try:
    import serial
    _SERIAL_AVAILABLE = True
except ImportError:
    serial = None  # type: ignore
    _SERIAL_AVAILABLE = False

# ─── USBDBG V1 protocol constants (from OpenMV 4.6.20 firmware) ────────────────

USBDBG_COMMAND = 0x30
USBDBG_FW_VERSION = 0x80
USBDBG_FRAME_SIZE = 0x81
USBDBG_FRAME_DUMP = 0x82
USBDBG_FB_ENABLE = 0x0D
USBDBG_TX_INPUT = 0x11

COMMAND_HEADER = struct.Struct("<BBI")
TRIPLE_U32 = struct.Struct("<III")
MAX_FRAME_BYTES = 4 * 1024 * 1024

OPENMV_USB_VID = 0x37C5
DEFAULT_BAUD = 921600


class USBDBGError(RuntimeError):
    """OpenMV USBDBG protocol error."""
    pass


def _read_exact(link: serial.Serial, size: int) -> bytes:
    """Read exactly *size* bytes from *link*, raising on timeout."""
    data = bytearray()
    stall_timeout = max(link.timeout or 1.0, 1.0)
    deadline = time.monotonic() + stall_timeout
    while len(data) < size:
        chunk = link.read(size - len(data))
        if chunk:
            data.extend(chunk)
            deadline = time.monotonic() + stall_timeout
        elif time.monotonic() >= deadline:
            raise USBDBGError(
                f"OpenMV response timed out ({len(data)}/{size} bytes)"
            )
    return bytes(data)


class USBDBGV1:
    """Ordered request/response access to one OpenMV serial connection."""

    def __init__(self, link: serial.Serial) -> None:
        self.link = link

    def _request(self, opcode: int, response_size: int) -> bytes:
        self.link.write(COMMAND_HEADER.pack(USBDBG_COMMAND, opcode, response_size))
        self.link.flush()
        return _read_exact(self.link, response_size)

    def _write(self, opcode: int, payload: bytes = b"") -> None:
        self.link.write(COMMAND_HEADER.pack(USBDBG_COMMAND, opcode, len(payload)))
        if payload:
            self.link.write(payload)
        self.link.flush()

    def firmware_version(self) -> tuple[int, int, int]:
        return TRIPLE_U32.unpack(self._request(USBDBG_FW_VERSION, TRIPLE_U32.size))

    def framebuffer_enable(self, enabled: bool) -> None:
        self._write(USBDBG_FB_ENABLE, struct.pack("<I", int(enabled)))

    def frame_size(self) -> tuple[int, int, int]:
        return TRIPLE_U32.unpack(self._request(USBDBG_FRAME_SIZE, TRIPLE_U32.size))

    def frame_dump(self, size: int) -> bytes:
        if not 0 < size <= MAX_FRAME_BYTES:
            raise USBDBGError(f"Invalid framebuffer size: {size}")
        return self._request(USBDBG_FRAME_DUMP, size)

    def tx_input(self, data: bytes) -> None:
        """Send input data (gimbal commands) to OpenMV via TX_INPUT channel."""
        if data:
            self._write(USBDBG_TX_INPUT, data)


# ─── Frame decoding ────────────────────────────────────────────────────────────

def decode_framebuffer(data: bytes, width: int, height: int) -> Optional[np.ndarray]:
    """Decode raw framebuffer bytes to RGB numpy array.

    Supports JPEG, RGB565, and grayscale formats.
    Returns None if the data cannot be decoded.
    """
    if data.startswith(b"\xff\xd8"):
        with Image.open(io.BytesIO(data)) as image:
            return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()

    pixels = width * height
    if len(data) == pixels * 2:
        rgb565 = np.frombuffer(data, dtype="<u2").reshape(height, width)
        red = ((rgb565 >> 11) & 0x1F).astype(np.uint8)
        green = ((rgb565 >> 5) & 0x3F).astype(np.uint8)
        blue = (rgb565 & 0x1F).astype(np.uint8)
        return np.stack(
            (
                (red << 3) | (red >> 2),
                (green << 2) | (green >> 4),
                (blue << 3) | (blue >> 2),
            ),
            axis=-1,
        )

    if len(data) == pixels:
        gray = np.frombuffer(data, dtype=np.uint8).reshape(height, width)
        return np.repeat(gray[:, :, None], 3, axis=2)

    return None


def rgb_to_jpeg_bytes(rgb: np.ndarray, quality: int = 80) -> bytes:
    """Encode an RGB numpy array to JPEG bytes."""
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return b""
    return buf.tobytes()


# ─── Camera worker thread ──────────────────────────────────────────────────────

@dataclass
class FrameInfo:
    """Metadata for one captured frame."""
    number: int
    width: int
    height: int
    rgb: np.ndarray
    captured_at: float
    jpeg_bytes: int


class _CameraWorker(threading.Thread):
    """Background thread that continuously reads frames from OpenMV."""

    def __init__(
        self,
        port: str,
        baud: int,
        frame_queue: queue.Queue,
        stop_event: threading.Event,
    ) -> None:
        super().__init__(daemon=True)
        self.port = port
        self.baud = baud
        self.frame_queue = frame_queue
        self.stop_event = stop_event
        self.version: Optional[tuple[int, int, int]] = None
        self.dropped_frames = 0
        self.total_frames = 0
        self.last_frame_time: float = 0.0
        self.fps: float = 0.0
        self.last_error: Optional[str] = None
        self.connected = False

    def run(self) -> None:
        while not self.stop_event.is_set():
            try:
                with serial.Serial(
                    self.port,
                    self.baud,
                    timeout=2.0,
                    write_timeout=1.0,
                ) as link:
                    self._run_camera(link)
            except Exception as exc:
                self.last_error = str(exc)
                self.connected = False
                if not self.stop_event.is_set():
                    time.sleep(0.5)

    def _run_camera(self, link: serial.Serial) -> None:
        time.sleep(0.3)
        link.reset_input_buffer()
        camera = USBDBGV1(link)
        self.version = camera.firmware_version()
        camera.framebuffer_enable(True)
        self.connected = True
        self.last_error = None
        self._camera = camera  # expose for gimbal commands

        while not self.stop_event.is_set():
            width, height, size = camera.frame_size()
            if size == 0:
                time.sleep(0.01)
                continue
            jpeg = camera.frame_dump(size)
            rgb = decode_framebuffer(jpeg, width, height)
            if rgb is None:
                self.dropped_frames += 1
                time.sleep(0.005)
                continue

            now = time.monotonic()
            self.total_frames += 1
            if self.last_frame_time > 0:
                dt = now - self.last_frame_time
                if dt > 0:
                    instant = 1.0 / dt
                    self.fps = instant if self.fps == 0 else self.fps * 0.85 + instant * 0.15
            self.last_frame_time = now

            frame = FrameInfo(
                number=self.total_frames,
                width=width,
                height=height,
                rgb=rgb,
                captured_at=now,
                jpeg_bytes=len(jpeg),
            )
            # Non-blocking put: drop oldest if queue full
            try:
                self.frame_queue.put_nowait(frame)
            except queue.Full:
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
                self.frame_queue.put_nowait(frame)

    def get_camera(self) -> Optional[USBDBGV1]:
        """Return the current USBDBGV1 instance for sending commands."""
        return getattr(self, "_camera", None)


# ─── OpenMV Manager ────────────────────────────────────────────────────────────

class OpenMVManager:
    """Thread-safe manager for OpenMV camera connection, recording, and gimbal.

    Typical usage::

        mgr = OpenMVManager()
        mgr.connect("COM12")
        jpeg = mgr.get_latest_jpeg()   # for MJPEG stream
        mgr.start_recording()
        ...
        video_path = mgr.stop_recording(upload_dir)
        mgr.send_gimbal_command("U")
        mgr.disconnect()
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._worker: Optional[_CameraWorker] = None
        self._stop_event = threading.Event()
        self._frame_queue: queue.Queue = queue.Queue(maxsize=2)
        self._recording = False
        self._recorded_frames: list[np.ndarray] = []
        self._record_start_time: float = 0.0

    # ── Connection ─────────────────────────────────────────────────────────

    @property
    def connected(self) -> bool:
        return self._worker is not None and self._worker.connected

    def connect(self, port: str, baud: int = DEFAULT_BAUD) -> dict:
        """Connect to OpenMV camera and start frame capture thread."""
        if not _SERIAL_AVAILABLE:
            return {"error": "未安装 pyserial，请先执行: pip install pyserial"}
        with self._lock:
            if self.connected:
                return {"error": f"已连接到 {self._worker.port}，请先断开。"}

            self._stop_event.clear()
            self._frame_queue = queue.Queue(maxsize=2)
            self._worker = _CameraWorker(
                port, baud, self._frame_queue, self._stop_event
            )
            self._worker.start()

            # Wait up to 3 seconds for connection
            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline and not self._worker.connected:
                time.sleep(0.1)
                if self._worker.last_error and not self._worker.connected:
                    err = self._worker.last_error
                    self._cleanup_worker()
                    return {"error": f"连接失败: {err}"}

            if not self._worker.connected:
                err = self._worker.last_error or "连接超时"
                self._cleanup_worker()
                return {"error": f"连接超时: {err}"}

            version = self._worker.version or (0, 0, 0)
            return {
                "success": True,
                "port": port,
                "firmware": f"{version[0]}.{version[1]}.{version[2]}",
            }

    def disconnect(self) -> dict:
        """Disconnect from OpenMV camera."""
        with self._lock:
            if self._recording:
                self._recording = False
                self._recorded_frames = []
            if not self.connected:
                self._cleanup_worker()
                return {"success": True, "already": True}
            self._stop_event.set()
            self._worker.join(timeout=2.0)
            self._cleanup_worker()
            return {"success": True}

    def _cleanup_worker(self) -> None:
        self._worker = None
        # Drain queue
        while not self._frame_queue.empty():
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                break

    # ── Status ──────────────────────────────────────────────────────────────

    def get_status(self) -> dict:
        """Return current connection and recording status."""
        if not self.connected:
            return {"connected": False, "recording": False}
        w = self._worker
        version = w.version or (0, 0, 0)
        return {
            "connected": True,
            "port": w.port,
            "firmware": f"{version[0]}.{version[1]}.{version[2]}",
            "fps": round(w.fps, 1),
            "total_frames": w.total_frames,
            "dropped_frames": w.dropped_frames,
            "recording": self._recording,
            "recorded_frames": len(self._recorded_frames) if self._recording else 0,
            "record_elapsed": round(time.monotonic() - self._record_start_time, 1) if self._recording else 0,
        }

    # ── Frame access ───────────────────────────────────────────────────────

    def get_latest_frame(self) -> Optional[FrameInfo]:
        """Get the most recent frame (non-blocking)."""
        latest = None
        while True:
            try:
                latest = self._frame_queue.get_nowait()
            except queue.Empty:
                break
        return latest

    def get_latest_jpeg(self, quality: int = 75) -> Optional[bytes]:
        """Get the most recent frame as JPEG bytes."""
        frame = self.get_latest_frame()
        if frame is None:
            return None
        return rgb_to_jpeg_bytes(frame.rgb, quality)

    # ── Recording ──────────────────────────────────────────────────────────

    def start_recording(self) -> dict:
        """Start recording frames."""
        with self._lock:
            if not self.connected:
                return {"error": "OpenMV 未连接。"}
            if self._recording:
                return {"error": "已在录制中。"}
            self._recorded_frames = []
            self._record_start_time = time.monotonic()
            self._recording = True
            return {"success": True}

    def stop_recording(self, upload_dir: str, fps: float = 13.0) -> dict:
        """Stop recording and save frames as MP4 video.

        Args:
            upload_dir: Directory to save the video file.
            fps: Output video frame rate.

        Returns:
            dict with video_path and frame count, or error.
        """
        with self._lock:
            if not self._recording:
                return {"error": "未在录制中。"}
            self._recording = False
            frames = self._recorded_frames.copy()
            self._recorded_frames = []

        if not frames:
            return {"error": "录制帧数为 0，未保存视频。"}

        # Generate output path
        ts = int(time.time())
        video_name = f"openmv_capture_{ts}.mp4"
        video_path = str(Path(upload_dir) / video_name)

        # Write video with OpenCV
        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(video_path, fourcc, fps, (w, h))
        for rgb_frame in frames:
            bgr = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
            writer.write(bgr)
        writer.release()

        return {
            "success": True,
            "video_path": video_path,
            "video_name": video_name,
            "frame_count": len(frames),
            "duration": round(len(frames) / fps, 2),
        }

    def _record_frame(self, rgb: np.ndarray) -> None:
        """Append frame to recording buffer if recording."""
        if self._recording:
            self._recorded_frames.append(rgb.copy())

    # ── Gimbal control ─────────────────────────────────────────────────────

    def send_gimbal_command(self, command: str) -> dict:
        """Send a gimbal command to OpenMV via TX_INPUT.

        Args:
            command: One of 'U' (up), 'D' (down), 'L' (left), 'R' (right), 'C' (center).

        Returns:
            dict with success status.
        """
        valid = {"U", "D", "L", "R", "C"}
        if command not in valid:
            return {"error": f"无效命令 '{command}'，支持: {', '.join(sorted(valid))}"}

        if not self.connected:
            return {"error": "OpenMV 未连接。"}

        camera = self._worker.get_camera()
        if camera is None:
            return {"error": "摄像头实例不可用。"}

        try:
            camera.tx_input(command.encode("ascii"))
            return {"success": True, "command": command}
        except Exception as e:
            return {"error": f"发送命令失败: {e}"}


# ─── Global singleton ─────────────────────────────────────────────────────────

openmv_manager = OpenMVManager()
