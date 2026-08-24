#!/usr/bin/env python3
"""Read-only compatibility probe for the OpenMV USBDBG V1 protocol.

This tool never uploads a script, resets the camera, enters the bootloader,
or writes to the camera filesystem. It only reads version/framebuffer data.
"""

from __future__ import annotations

import argparse
import io
import struct
import time
from pathlib import Path

import serial
from PIL import Image
from serial.tools import list_ports


USBDBG_COMMAND = 0x30
USBDBG_FW_VERSION = 0x80
USBDBG_FRAME_SIZE = 0x81
USBDBG_FRAME_DUMP = 0x82
USBDBG_FB_ENABLE = 0x0D

COMMAND_HEADER = struct.Struct("<BBI")
VERSION_RESPONSE = struct.Struct("<III")
FRAME_SIZE_RESPONSE = struct.Struct("<III")

DEFAULT_BAUD = 921600
MAX_FRAME_BYTES = 4 * 1024 * 1024


class ProbeError(RuntimeError):
    pass


def read_exact(link: serial.Serial, size: int) -> bytes:
    result = bytearray()
    deadline = time.monotonic() + max(link.timeout or 1.0, 1.0)
    while len(result) < size:
        chunk = link.read(size - len(result))
        if chunk:
            result.extend(chunk)
            deadline = time.monotonic() + max(link.timeout or 1.0, 1.0)
        elif time.monotonic() >= deadline:
            raise ProbeError(
                "OpenMV response timed out "
                f"({len(result)}/{size} bytes received)"
            )
    return bytes(result)


class USBDBGV1:
    def __init__(self, link: serial.Serial) -> None:
        self.link = link

    def _request(self, opcode: int, response_size: int) -> bytes:
        self.link.write(COMMAND_HEADER.pack(USBDBG_COMMAND, opcode, response_size))
        self.link.flush()
        return read_exact(self.link, response_size)

    def _write(self, opcode: int, payload: bytes) -> None:
        self.link.write(COMMAND_HEADER.pack(USBDBG_COMMAND, opcode, len(payload)))
        if payload:
            self.link.write(payload)
        self.link.flush()

    def firmware_version(self) -> tuple[int, int, int]:
        return VERSION_RESPONSE.unpack(
            self._request(USBDBG_FW_VERSION, VERSION_RESPONSE.size)
        )

    def framebuffer_enable(self, enabled: bool) -> None:
        self._write(USBDBG_FB_ENABLE, struct.pack("<I", int(enabled)))

    def frame_size(self) -> tuple[int, int, int]:
        return FRAME_SIZE_RESPONSE.unpack(
            self._request(USBDBG_FRAME_SIZE, FRAME_SIZE_RESPONSE.size)
        )

    def frame_dump(self, size: int) -> bytes:
        if not 0 < size <= MAX_FRAME_BYTES:
            raise ProbeError(f"Invalid framebuffer size reported by camera: {size}")
        return self._request(USBDBG_FRAME_DUMP, size)


def print_ports() -> None:
    ports = list(list_ports.comports())
    if not ports:
        print("No serial ports found.")
        return
    for port in ports:
        print(f"{port.device}: {port.description} [{port.hwid}]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only OpenMV firmware/framebuffer compatibility probe"
    )
    parser.add_argument("--port", help="OpenMV data port, for example COM12")
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD)
    parser.add_argument("--frames", type=int, default=20)
    parser.add_argument("--list", action="store_true", help="List ports and exit")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("probe_frame.jpg"),
        help="Where to save the first successfully decoded JPEG",
    )
    return parser.parse_args()


def run_probe(args: argparse.Namespace) -> int:
    if not args.port:
        raise ProbeError("Use --port COMx to specify the OpenMV data port")
    if args.frames < 1:
        raise ProbeError("--frames must be at least 1")

    print("Safety mode: no script upload, reset, bootloader, or filesystem write.")
    with serial.Serial(
        port=args.port,
        baudrate=args.baud,
        timeout=1.0,
        write_timeout=1.0,
    ) as link:
        time.sleep(0.3)
        link.reset_input_buffer()
        camera = USBDBGV1(link)

        version = camera.firmware_version()
        print(f"Firmware response: {version[0]}.{version[1]}.{version[2]}")
        camera.framebuffer_enable(True)

        received = 0
        decoded = 0
        first_saved = False
        started = time.monotonic()
        deadline = started + 10.0

        while received < args.frames and time.monotonic() < deadline:
            width, height, size = camera.frame_size()
            if size == 0:
                time.sleep(0.05)
                continue

            payload = camera.frame_dump(size)
            received += 1
            is_jpeg = payload.startswith(b"\xff\xd8") and payload.endswith(b"\xff\xd9")
            print(
                f"Frame {received}: {width}x{height}, {size} bytes, "
                f"JPEG={'yes' if is_jpeg else 'no'}"
            )

            if is_jpeg:
                with Image.open(io.BytesIO(payload)) as image:
                    image.load()
                    decoded += 1
                    if not first_saved:
                        args.output.parent.mkdir(parents=True, exist_ok=True)
                        image.convert("RGB").save(args.output, quality=95)
                        print(f"First decoded frame saved to: {args.output.resolve()}")
                        first_saved = True

            time.sleep(0.05)

        elapsed = max(time.monotonic() - started, 0.001)
        print(
            f"Result: {received} frames, {decoded} JPEG frames, "
            f"average polling rate {received / elapsed:.2f} FPS"
        )
        if received == 0:
            raise ProbeError(
                "The USBDBG handshake worked, but no framebuffer became available"
            )
        if decoded == 0:
            raise ProbeError(
                "Frames arrived but were not JPEG; raw-frame decoding must be added "
                "before using them with YOLO"
            )
    return 0


def main() -> int:
    args = parse_args()
    if args.list:
        print_ports()
        return 0
    try:
        return run_probe(args)
    except (ProbeError, serial.SerialException, OSError, ValueError) as exc:
        print(f"Probe failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
