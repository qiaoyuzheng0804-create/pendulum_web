"""Minimal host client for the OpenMV USBDBG V1 protocol used by 4.6.20."""

from __future__ import annotations

import struct
import time

import serial


USBDBG_COMMAND = 0x30
USBDBG_FW_VERSION = 0x80
USBDBG_FRAME_SIZE = 0x81
USBDBG_FRAME_DUMP = 0x82
USBDBG_FB_ENABLE = 0x0D
USBDBG_TX_INPUT = 0x11

COMMAND_HEADER = struct.Struct("<BBI")
TRIPLE_U32 = struct.Struct("<III")
MAX_FRAME_BYTES = 4 * 1024 * 1024


class USBDBGError(RuntimeError):
    pass


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

    def tx_input(self, data: bytes) -> None:
        if data:
            self._write(USBDBG_TX_INPUT, data)
