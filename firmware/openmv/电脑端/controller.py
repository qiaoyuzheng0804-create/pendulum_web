#!/usr/bin/env python3
"""OpenMV USBDBG V1 viewer, YOLO input pipeline, and gimbal controller."""

from __future__ import annotations

import argparse
import io
import queue
import threading
import time
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import serial
from PIL import Image, ImageTk
from serial.tools import list_ports

from openmv_v1 import USBDBGError, USBDBGV1


DEFAULT_BAUD = 921600
OPENMV_USB_VID = 0x37C5
DEFAULT_DANBAI_MODEL = (
    Path(__file__).resolve().parents[3]
    / "models"
    / "danbai_best.pt"
)
SERVO_STEP_US = 10
SERVO_CENTER_US = 1500
PAN_MIN_US = 835
PAN_MAX_US = 2165
TILT_MIN_US = 835
TILT_MAX_US = 2165
PAN_RIGHT_SIGN = -1
TILT_UP_SIGN = -1


@dataclass
class VideoFrame:
    number: int
    width: int
    height: int
    rgb: np.ndarray
    captured_at: float
    jpeg_bytes: int
    detections: int | None = None


def decode_framebuffer(data: bytes, width: int, height: int) -> np.ndarray | None:
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


def replace_latest(target: queue.Queue, item: object) -> None:
    try:
        target.put_nowait(item)
    except queue.Full:
        try:
            target.get_nowait()
        except queue.Empty:
            pass
        target.put_nowait(item)


def serial_ports() -> list:
    return list(list_ports.comports())


def choose_port(requested: str | None) -> str:
    if requested:
        return requested
    candidates = [
        port for port in serial_ports()
        if port.vid == OPENMV_USB_VID
        or "openmv" in (port.description or "").lower()
    ]
    if len(candidates) == 1:
        return candidates[0].device
    found = "\n".join(
        f"  {port.device}: {port.description}" for port in serial_ports()
    ) or "  (none)"
    raise RuntimeError("Cannot identify one OpenMV port; use --port COMx.\n" + found)


class CameraWorker(threading.Thread):
    def __init__(
        self,
        port: str,
        baud: int,
        frames: queue.Queue[VideoFrame],
        errors: queue.Queue[Exception],
        stop: threading.Event,
    ) -> None:
        super().__init__(daemon=True)
        self.port = port
        self.baud = baud
        self.frames = frames
        self.errors = errors
        self.stop = stop
        self.version: tuple[int, int, int] | None = None
        self.dropped_frames = 0
        self.reconnect_count = 0

    def run(self) -> None:
        while not self.stop.is_set():
            try:
                with serial.Serial(
                    self.port,
                    self.baud,
                    timeout=2.0,
                    write_timeout=1.0,
                ) as link:
                    self._run_camera(link)
            except Exception as exc:
                self.reconnect_count += 1
                replace_latest(self.errors, exc)
                if not self.stop.is_set():
                    time.sleep(0.5)

    def _run_camera(self, link: serial.Serial) -> None:
        time.sleep(0.3)
        link.reset_input_buffer()
        camera = USBDBGV1(link)
        self.version = camera.firmware_version()
        camera.framebuffer_enable(True)
        frame_number = 0

        while not self.stop.is_set():
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
            frame_number += 1
            replace_latest(
                self.frames,
                VideoFrame(
                    frame_number,
                    width,
                    height,
                    rgb,
                    time.monotonic(),
                    size,
                ),
            )


class ControlWorker(threading.Thread):
    def __init__(
        self,
        port: str,
        commands: queue.Queue[bytes],
        errors: queue.Queue[Exception],
        stop: threading.Event,
    ) -> None:
        super().__init__(daemon=True)
        self.port = port
        self.commands = commands
        self.errors = errors
        self.stop = stop
        self.command_count = 0
        self.blocked_count = 0
        self.last_command = "-"
        self.pan_us = SERVO_CENTER_US
        self.tilt_us = SERVO_CENTER_US
        self.limit_message = ""

    def next_position(self, command: bytes) -> tuple[int, int, str]:
        pan_us = self.pan_us
        tilt_us = self.tilt_us
        limit_message = ""

        if command == b"L":
            pan_us -= PAN_RIGHT_SIGN * SERVO_STEP_US
            limit_message = "下方舵机已到左侧限制"
        elif command == b"R":
            pan_us += PAN_RIGHT_SIGN * SERVO_STEP_US
            limit_message = "下方舵机已到右侧限制"
        elif command == b"U":
            tilt_us += TILT_UP_SIGN * SERVO_STEP_US
            limit_message = "上方舵机已到上侧限制"
        elif command == b"D":
            tilt_us -= TILT_UP_SIGN * SERVO_STEP_US
            limit_message = "上方舵机已到下侧限制"
        elif command == b"C":
            pan_us = SERVO_CENTER_US
            tilt_us = SERVO_CENTER_US

        return pan_us, tilt_us, limit_message

    def send_command(self, control: serial.Serial, command: bytes) -> None:
        pan_us, tilt_us, limit_message = self.next_position(command)
        if not (
            PAN_MIN_US < pan_us < PAN_MAX_US
            and TILT_MIN_US < tilt_us < TILT_MAX_US
        ):
            self.blocked_count += 1
            self.limit_message = limit_message
            return
        control.write(command)
        control.flush()
        self.pan_us = pan_us
        self.tilt_us = tilt_us
        self.limit_message = ""
        self.command_count += 1
        self.last_command = command.decode("ascii", errors="replace")

    def run(self) -> None:
        try:
            with serial.Serial(
                self.port,
                115200,
                timeout=0,
                write_timeout=1.0,
            ) as control:
                while not self.stop.is_set():
                    try:
                        command = self.commands.get(timeout=0.05)
                    except queue.Empty:
                        continue
                    self.send_command(control, command)
        except Exception as exc:
            replace_latest(self.errors, exc)
            self.stop.set()


class YoloWorker(threading.Thread):
    def __init__(
        self,
        model_path: Path,
        confidence: float,
        image_size: int,
        source_frames: queue.Queue[VideoFrame],
        output_frames: queue.Queue[VideoFrame],
        errors: queue.Queue[Exception],
        stop: threading.Event,
    ) -> None:
        super().__init__(daemon=True)
        self.model_path = model_path
        self.confidence = confidence
        self.image_size = image_size
        self.source_frames = source_frames
        self.output_frames = output_frames
        self.errors = errors
        self.stop = stop

    def run(self) -> None:
        try:
            try:
                from ultralytics import YOLO
            except ImportError as exc:
                raise RuntimeError(
                    "YOLO requested but ultralytics is not installed"
                ) from exc
            model = YOLO(str(self.model_path))
            while not self.stop.is_set():
                try:
                    frame = self.source_frames.get(timeout=0.1)
                except queue.Empty:
                    continue
                # Ultralytics treats numpy image input as OpenCV-style BGR.
                source_bgr = np.ascontiguousarray(frame.rgb[:, :, ::-1])
                results = model.predict(
                    source=source_bgr,
                    conf=self.confidence,
                    iou=0.45,
                    imgsz=self.image_size,
                    max_det=1,
                    verbose=False,
                )
                result = results[0]
                plotted_bgr = result.plot()
                frame.rgb = np.ascontiguousarray(plotted_bgr[:, :, ::-1])
                frame.detections = len(result.boxes) if result.boxes is not None else 0
                replace_latest(self.output_frames, frame)
        except Exception as exc:
            replace_latest(self.errors, exc)
            self.stop.set()


class Viewer:
    def __init__(self, args: argparse.Namespace, port: str) -> None:
        self.args = args
        self.stop = threading.Event()
        self.commands: queue.Queue[bytes] = queue.Queue()
        self.camera_frames: queue.Queue[VideoFrame] = queue.Queue(maxsize=1)
        self.display_frames: queue.Queue[VideoFrame] = queue.Queue(maxsize=1)
        self.errors: queue.Queue[Exception] = queue.Queue(maxsize=1)
        self.keys_down: set[str] = set()
        self.pending_releases: dict[str, str] = {}
        self.photo: ImageTk.PhotoImage | None = None
        self.last_frame_time: float | None = None
        self.fps = 0.0

        camera_output = self.camera_frames if args.model else self.display_frames
        self.camera = CameraWorker(
            port,
            args.baud,
            camera_output,
            self.errors,
            self.stop,
        )
        self.control = ControlWorker(
            args.control_port,
            self.commands,
            self.errors,
            self.stop,
        )
        self.yolo: YoloWorker | None = None
        if args.model:
            self.yolo = YoloWorker(
                args.model,
                args.confidence,
                args.imgsz,
                self.camera_frames,
                self.display_frames,
                self.errors,
                self.stop,
            )

        self.root = tk.Tk()
        self.root.title(f"OpenMV experiment camera - {port}")
        self.root.configure(bg="#202124")
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.image_label = tk.Label(self.root, bg="black")
        self.image_label.pack(fill="both", expand=True, padx=8, pady=(8, 4))
        self.status = tk.StringVar(value="Connecting to OpenMV...")
        tk.Label(
            self.root,
            textvariable=self.status,
            anchor="w",
            bg="#202124",
            fg="white",
            padx=8,
            pady=6,
        ).pack(fill="x")
        self.root.bind("<KeyPress>", self.key_press)
        self.root.bind("<KeyRelease>", self.key_release)
        self.root.focus_force()

    def key_press(self, event: tk.Event) -> None:
        key = event.keysym
        pending = self.pending_releases.pop(key, None)
        if pending is not None:
            self.root.after_cancel(pending)
        if key in self.keys_down:
            return
        self.keys_down.add(key)
        command = {
            "Up": b"U",
            "Down": b"D",
            "Left": b"L",
            "Right": b"R",
            "space": b"C",
        }.get(key)
        if command:
            self.commands.put(command)

    def key_release(self, event: tk.Event) -> None:
        key = event.keysym
        pending = self.pending_releases.pop(key, None)
        if pending is not None:
            self.root.after_cancel(pending)
        self.pending_releases[key] = self.root.after(
            40, lambda released=key: self.finish_release(released)
        )

    def finish_release(self, key: str) -> None:
        self.pending_releases.pop(key, None)
        self.keys_down.discard(key)

    def update(self) -> None:
        if not self.errors.empty():
            self.status.set(f"Error: {self.errors.get_nowait()}")

        latest = None
        while True:
            try:
                latest = self.display_frames.get_nowait()
            except queue.Empty:
                break

        if latest is not None:
            now = latest.captured_at
            if self.last_frame_time is not None:
                instant = 1.0 / max(now - self.last_frame_time, 0.001)
                self.fps = instant if self.fps == 0 else self.fps * 0.85 + instant * 0.15
            self.last_frame_time = now

            image = Image.fromarray(latest.rgb, "RGB")
            if self.args.scale != 1:
                image = image.resize(
                    (image.width * self.args.scale, image.height * self.args.scale),
                    Image.Resampling.NEAREST,
                )
            self.photo = ImageTk.PhotoImage(image)
            self.image_label.configure(image=self.photo)
            version = self.camera.version or (0, 0, 0)
            detection_text = (
                f" | detections {latest.detections}"
                if latest.detections is not None else ""
            )
            self.status.set(
                f"{latest.width}x{latest.height} | {latest.jpeg_bytes / 1024:.1f} KiB "
                f"| {self.fps:.1f} FPS | firmware {version[0]}.{version[1]}.{version[2]}"
                f"{detection_text} | sent {self.control.command_count} "
                f"({self.control.last_command}) | P {self.control.pan_us} "
                f"T {self.control.tilt_us} | dropped {self.camera.dropped_frames} "
                f"| reconnect {self.camera.reconnect_count} | "
                f"{self.control.limit_message or '方向键控制云台，空格回中'}"
            )

        if not self.stop.is_set():
            self.root.after(10, self.update)

    def close(self) -> None:
        self.stop.set()
        self.root.destroy()

    def run(self) -> None:
        self.camera.start()
        self.control.start()
        if self.yolo:
            self.yolo.start()
        self.root.after(10, self.update)
        try:
            self.root.mainloop()
        finally:
            self.stop.set()
            self.camera.join(timeout=2.0)
            self.control.join(timeout=2.0)
            if self.yolo:
                self.yolo.join(timeout=2.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OpenMV USB viewer, YOLO pipeline, and gimbal controller"
    )
    parser.add_argument("--port", help="OpenMV port, for example COM12")
    parser.add_argument("--control-port", required=True, help="USB-TTL control port, for example COM5")
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD)
    parser.add_argument("--scale", type=int, default=1, choices=range(1, 5))
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_DANBAI_MODEL,
        help="Ultralytics model path (default: pendulum_web danbai_best.pt)",
    )
    parser.add_argument("--confidence", type=float, default=0.12)
    parser.add_argument(
        "--imgsz",
        type=int,
        default=960,
        help="YOLO inference size; use 1280 for maximum small-object sensitivity",
    )
    parser.add_argument("--list", action="store_true", help="List ports and exit")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.list:
        for port in serial_ports():
            print(f"{port.device}: {port.description} [{port.hwid}]")
        return 0
    try:
        port = choose_port(args.port)
        Viewer(args, port).run()
    except (RuntimeError, serial.SerialException, USBDBGError) as exc:
        print(f"Error: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
