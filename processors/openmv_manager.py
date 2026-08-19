"""OpenMV camera manager: dual-serial-port architecture.

Port 1 — OpenMV USB-C (USBDBG V1 @ 921600):
    Frame capture thread reads JPEG frames from the framebuffer.

Port 2 — USB-TTL adapter (UART @ 115200):
    Gimbal control thread sends single-byte commands (L/R/U/D/C)
    to OpenMV UART3 (P4 TX / P5 RX), which drives two MG90S servos.

This matches the architecture in the source project:
    openmv拍摄/实时画面与键盘云台/电脑端/controller.py
"""

from __future__ import annotations

import io
import queue
import struct
import threading
import time
from dataclasses import dataclass
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

COMMAND_HEADER = struct.Struct("<BBI")
TRIPLE_U32 = struct.Struct("<III")
MAX_FRAME_BYTES = 4 * 1024 * 1024

OPENMV_USB_VID = 0x37C5
CAMERA_BAUD = 921600
CONTROL_BAUD = 115200

# ─── Servo limits (from source controller.py) ─────────────────────────────────

SERVO_STEP_US = 10
SERVO_CENTER_US = 1500
PAN_MIN_US = 835
PAN_MAX_US = 2165
TILT_MIN_US = 835
TILT_MAX_US = 2165
PAN_RIGHT_SIGN = -1
TILT_UP_SIGN = -1


class USBDBGError(RuntimeError):
    """OpenMV USBDBG protocol error."""
    pass


# ─── Low-level USBDBG V1 helpers ──────────────────────────────────────────────

def _read_exact(link: serial.Serial, size: int) -> bytes:
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


# ─── Frame decoding ────────────────────────────────────────────────────────────

def decode_framebuffer(data: bytes, width: int, height: int) -> Optional[np.ndarray]:
    """Decode raw framebuffer bytes to RGB numpy array (JPEG / RGB565 / gray)."""
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


# ─── Data structures ──────────────────────────────────────────────────────────

@dataclass
class FrameInfo:
    """Metadata for one captured frame."""
    number: int
    width: int
    height: int
    rgb: np.ndarray
    captured_at: float
    jpeg_bytes: int


# ─── Camera worker (USBDBG V1 frame reading) ─────────────────────────────────

class _CameraWorker(threading.Thread):
    """Background thread: reads frames from OpenMV via USBDBG V1."""

    def __init__(
        self,
        port: str,
        baud: int,
        frame_queue: queue.Queue,
        stop_event: threading.Event,
        record_callback=None,
    ) -> None:
        super().__init__(daemon=True)
        self.port = port
        self.baud = baud
        self.frame_queue = frame_queue
        self.stop_event = stop_event
        self._record_callback = record_callback
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
        time.sleep(0.2)  # give camera firmware time to start filling framebuffer
        self.connected = True
        self.last_error = None
        print(f"[OpenMV] Camera connected: firmware {self.version}, port {self.port}", flush=True)

        zero_count = 0
        while not self.stop_event.is_set():
            try:
                width, height, size = camera.frame_size()
                if size == 0:
                    zero_count += 1
                    if zero_count == 50:
                        print(f"[OpenMV] frame_size() returned 0 fifty times in a row — camera sensor may not be capturing", flush=True)
                    time.sleep(0.01)
                    continue
                if zero_count > 0:
                    print(f"[OpenMV] Got first frame after {zero_count} empty polls: {width}x{height} size={size}", flush=True)
                    zero_count = 0
                jpeg = camera.frame_dump(size)
            except (USBDBGError, OSError) as exc:
                self.last_error = str(exc)
                self.connected = False
                return

            rgb = decode_framebuffer(jpeg, width, height)
            if rgb is None:
                self.dropped_frames += 1
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
            try:
                self.frame_queue.put_nowait(frame)
            except queue.Full:
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
                self.frame_queue.put_nowait(frame)

            # Recording: append frame copy to buffer if active
            if self._record_callback:
                self._record_callback(rgb)


# ─── Gimbal control worker (UART @ 115200) ────────────────────────────────────

class _ControlWorker(threading.Thread):
    """Background thread: sends gimbal commands via USB-TTL UART.

    Protocol: single ASCII bytes (L/R/U/D/C) at 115200 baud.
    OpenMV UART3 receives them and drives P1/P9 servos.
    Software tracks pan/tilt pulse widths and enforces mechanical limits.
    """

    def __init__(
        self,
        port: str,
        commands: queue.Queue,
        stop_event: threading.Event,
    ) -> None:
        super().__init__(daemon=True)
        self.port = port
        self.commands = commands
        self.stop_event = stop_event
        self.connected = False
        self.last_error: Optional[str] = None
        self.command_count = 0
        self.blocked_count = 0
        self.last_command = "-"
        self.pan_us = SERVO_CENTER_US
        self.tilt_us = SERVO_CENTER_US
        self.limit_message = ""

    def run(self) -> None:
        try:
            with serial.Serial(
                self.port,
                CONTROL_BAUD,
                timeout=0,
                write_timeout=1.0,
            ) as control:
                self.connected = True
                self.last_error = None
                while not self.stop_event.is_set():
                    try:
                        command = self.commands.get(timeout=0.05)
                    except queue.Empty:
                        continue
                    self._send_command(control, command)
        except Exception as exc:
            self.last_error = str(exc)
            self.connected = False

    def _next_position(self, command: bytes) -> tuple[int, int, str]:
        pan_us = self.pan_us
        tilt_us = self.tilt_us
        msg = ""
        if command == b"L":
            pan_us -= PAN_RIGHT_SIGN * SERVO_STEP_US
            msg = "下方舵机已到左侧限制"
        elif command == b"R":
            pan_us += PAN_RIGHT_SIGN * SERVO_STEP_US
            msg = "下方舵机已到右侧限制"
        elif command == b"U":
            tilt_us += TILT_UP_SIGN * SERVO_STEP_US
            msg = "上方舵机已到上侧限制"
        elif command == b"D":
            tilt_us -= TILT_UP_SIGN * SERVO_STEP_US
            msg = "上方舵机已到下侧限制"
        elif command == b"C":
            pan_us = SERVO_CENTER_US
            tilt_us = SERVO_CENTER_US
        return pan_us, tilt_us, msg

    def _send_command(self, control: serial.Serial, command: bytes) -> None:
        pan_us, tilt_us, msg = self._next_position(command)
        if not (PAN_MIN_US < pan_us < PAN_MAX_US and TILT_MIN_US < tilt_us < TILT_MAX_US):
            self.blocked_count += 1
            self.limit_message = msg
            return
        control.write(command)
        control.flush()
        self.pan_us = pan_us
        self.tilt_us = tilt_us
        self.limit_message = ""
        self.command_count += 1
        self.last_command = command.decode("ascii", errors="replace")


# ─── OpenMV Manager ────────────────────────────────────────────────────────────

class OpenMVManager:
    """Thread-safe manager for OpenMV camera + gimbal (dual serial port).

    Port 1 (camera):  OpenMV USB-C, USBDBG V1 @ 921600  → frame capture
    Port 2 (gimbal):  USB-TTL adapter, UART @ 115200    → servo commands
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._camera_worker: Optional[_CameraWorker] = None
        self._control_worker: Optional[_ControlWorker] = None
        self._stop_event = threading.Event()
        self._frame_queue: queue.Queue = queue.Queue(maxsize=2)
        self._command_queue: queue.Queue = queue.Queue()
        self._recording = False
        self._recorded_frames: list[np.ndarray] = []
        self._record_start_time: float = 0.0

    # ── Connection ─────────────────────────────────────────────────────────

    @property
    def camera_connected(self) -> bool:
        return self._camera_worker is not None and self._camera_worker.connected

    @property
    def control_connected(self) -> bool:
        return self._control_worker is not None and self._control_worker.connected

    @property
    def connected(self) -> bool:
        return self.camera_connected

    def connect_camera(self, port: str, baud: int = CAMERA_BAUD) -> dict:
        """Connect to OpenMV camera for frame capture."""
        if not _SERIAL_AVAILABLE:
            return {"error": "未安装 pyserial，请先执行: pip install pyserial"}
        with self._lock:
            if self.camera_connected:
                return {"error": f"摄像头已连接: {self._camera_worker.port}，请先断开。"}
            self._stop_event.clear()
            self._frame_queue = queue.Queue(maxsize=2)
            self._camera_worker = _CameraWorker(
                port, baud, self._frame_queue, self._stop_event,
                record_callback=self._record_frame_if_active,
            )
            self._camera_worker.start()

            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline and not self._camera_worker.connected:
                time.sleep(0.1)
                if self._camera_worker.last_error and not self._camera_worker.connected:
                    err = self._camera_worker.last_error
                    self._cleanup_camera()
                    return {"error": f"连接失败: {err}"}

            if not self._camera_worker.connected:
                err = self._camera_worker.last_error or "连接超时"
                self._cleanup_camera()
                return {"error": f"连接超时: {err}"}

            version = self._camera_worker.version or (0, 0, 0)
            return {
                "success": True,
                "port": port,
                "firmware": f"{version[0]}.{version[1]}.{version[2]}",
            }

    def connect_control(self, port: str) -> dict:
        """Connect to USB-TTL adapter for gimbal control."""
        if not _SERIAL_AVAILABLE:
            return {"error": "未安装 pyserial，请先执行: pip install pyserial"}
        with self._lock:
            if self.control_connected:
                return {"error": f"云台已连接: {self._control_worker.port}，请先断开。"}
            self._command_queue = queue.Queue()
            self._control_worker = _ControlWorker(
                port, self._command_queue, self._stop_event
            )
            self._control_worker.start()

            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline and not self._control_worker.connected:
                time.sleep(0.1)
                if self._control_worker.last_error and not self._control_worker.connected:
                    err = self._control_worker.last_error
                    self._cleanup_control()
                    return {"error": f"云台连接失败: {err}"}

            if not self._control_worker.connected:
                err = self._control_worker.last_error or "连接超时"
                self._cleanup_control()
                return {"error": f"云台连接超时: {err}"}

            return {"success": True, "port": port}

    def disconnect_camera(self) -> dict:
        """Disconnect camera."""
        with self._lock:
            if self._recording:
                self._recording = False
                self._recorded_frames = []
            if not self.camera_connected:
                self._cleanup_camera()
                return {"success": True, "already": True}
            self._stop_event.set()
            self._camera_worker.join(timeout=2.0)
            self._cleanup_camera()
            return {"success": True}

    def disconnect_control(self) -> dict:
        """Disconnect gimbal control."""
        with self._lock:
            if not self.control_connected:
                self._cleanup_control()
                return {"success": True, "already": True}
            self._stop_event.set()
            self._control_worker.join(timeout=2.0)
            self._cleanup_control()
            return {"success": True}

    def disconnect(self) -> dict:
        """Disconnect both camera and gimbal."""
        self.disconnect_control()
        self.disconnect_camera()
        # Reset stop event if camera was the last thing connected
        if not self.camera_connected and not self.control_connected:
            self._stop_event.clear()
        return {"success": True}

    def _cleanup_camera(self) -> None:
        self._camera_worker = None
        while not self._frame_queue.empty():
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                break

    def _cleanup_control(self) -> None:
        self._control_worker = None
        while not self._command_queue.empty():
            try:
                self._command_queue.get_nowait()
            except queue.Empty:
                break

    # ── Status ──────────────────────────────────────────────────────────────

    def get_status(self) -> dict:
        """Return connection and recording status for both ports."""
        result: dict = {
            "camera_connected": self.camera_connected,
            "control_connected": self.control_connected,
            "connected": self.camera_connected,
            "recording": self._recording,
        }
        if self.camera_connected:
            w = self._camera_worker
            version = w.version or (0, 0, 0)
            result.update({
                "camera_port": w.port,
                "firmware": f"{version[0]}.{version[1]}.{version[2]}",
                "fps": round(w.fps, 1),
                "total_frames": w.total_frames,
                "dropped_frames": w.dropped_frames,
            })
        if self.control_connected:
            c = self._control_worker
            result.update({
                "control_port": c.port,
                "gimbal_command_count": c.command_count,
                "gimbal_last_command": c.last_command,
                "gimbal_pan_us": c.pan_us,
                "gimbal_tilt_us": c.tilt_us,
                "gimbal_limit": c.limit_message,
            })
        if self._recording:
            result["recorded_frames"] = len(self._recorded_frames)
            result["record_elapsed"] = round(time.monotonic() - self._record_start_time, 1)
        return result

    # ── Frame access ───────────────────────────────────────────────────────

    def get_latest_frame(self) -> Optional[FrameInfo]:
        """Get the most recent frame (non-blocking, drains queue)."""
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
            if not self.camera_connected:
                return {"error": "OpenMV 摄像头未连接。"}
            if self._recording:
                return {"error": "已在录制中。"}
            self._recorded_frames = []
            self._record_start_time = time.monotonic()
            self._recording = True
            return {"success": True}

    def stop_recording(self, upload_dir: str, fps: float = 13.0) -> dict:
        """Stop recording and save frames as MP4 video."""
        with self._lock:
            if not self._recording:
                return {"error": "未在录制中。"}
            self._recording = False
            frames = self._recorded_frames.copy()
            self._recorded_frames = []

        if not frames:
            return {"error": "录制帧数为 0，未保存视频。"}

        ts = int(time.time())
        video_name = f"openmv_capture_{ts}.mp4"
        video_path = str(Path(upload_dir) / video_name)

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

    def _record_frame_if_active(self, rgb: np.ndarray) -> None:
        """Callback for camera worker: append frame if recording is active."""
        if self._recording:
            self._recorded_frames.append(rgb.copy())

    # ── Gimbal control ─────────────────────────────────────────────────────

    def send_gimbal_command(self, command: str) -> dict:
        """Send a gimbal command via the control serial port.

        Args:
            command: One of 'U' (up), 'D' (down), 'L' (left), 'R' (right), 'C' (center).
        """
        valid = {"U", "D", "L", "R", "C"}
        if command not in valid:
            return {"error": f"无效命令 '{command}'，支持: {', '.join(sorted(valid))}"}
        if not self.control_connected:
            return {"error": "云台串口未连接。请先连接 USB-TTL 适配器。"}
        self._command_queue.put(command.encode("ascii"))
        return {"success": True, "command": command}


# ─── Global singleton ─────────────────────────────────────────────────────────

openmv_manager = OpenMVManager()
