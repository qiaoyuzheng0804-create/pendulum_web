#!/usr/bin/env python3
"""通过 USB-TTL 串口控制 STM32 单摆电磁铁。"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
import time

import serial
from serial.tools import list_ports


BAUD_RATE = 115200


def default_port() -> str:
    return "COM3" if os.name == "nt" else "/dev/ttyUSB0"


def wait_for_toggle_key() -> None:
    """等待回车或空格；Windows 下按键无需再按一次回车。"""
    print("按 Enter（或空格）切换电磁铁，按 Ctrl+C 退出。", flush=True)

    if os.name == "nt":
        import msvcrt

        while True:
            key = msvcrt.getwch()
            if key in ("\r", "\n", " "):
                return
            if key == "\x03":
                raise KeyboardInterrupt

    if not sys.stdin.isatty():
        input()
        return

    import termios
    import tty

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while True:
            key = sys.stdin.read(1)
            if key in ("\r", "\n", " "):
                print()
                return
            if key == "\x03":
                raise KeyboardInterrupt
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def format_wall_time(timestamp_ns: int) -> str:
    seconds, nanoseconds = divmod(timestamp_ns, 1_000_000_000)
    local_time = dt.datetime.fromtimestamp(seconds).astimezone()
    return f"{local_time:%Y-%m-%d %H:%M:%S}.{nanoseconds:09d} {local_time:%z}"


def print_ports() -> None:
    ports = list(list_ports.comports())
    if not ports:
        print("未发现串口设备。")
        return
    for port in ports:
        print(f"{port.device}: {port.description}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="STM32 单摆电磁铁串口控制器")
    parser.add_argument(
        "port",
        nargs="?",
        default=default_port(),
        help=f"串口名称（默认：{default_port()}）",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="列出可用串口后退出",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.list:
        print_ports()
        return 0

    electromagnet_on = False
    try:
        with serial.Serial(
            port=args.port,
            baudrate=BAUD_RATE,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=1.0,
            write_timeout=1.0,
        ) as connection:
            # Some USB-TTL adapters reset the board when the port opens.
            time.sleep(1.5)
            connection.reset_input_buffer()
            try:
                # The protocol has no state query. Establish a known safe state.
                connection.write(b"0")
                connection.flush()
                electromagnet_on = False
                print(f"已连接 {args.port}（115200 8N1）。")
                print("电磁铁当前状态：断电（无磁力）。")

                while True:
                    wait_for_toggle_key()
                    next_state = not electromagnet_on
                    command = b"1" if next_state else b"0"

                    # Set the safety flag before sending: a failed write still
                    # takes the exception path, which attempts to send '0'.
                    if next_state:
                        electromagnet_on = True
                    connection.write(command)
                    toggle_monotonic_ns = time.perf_counter_ns()
                    toggle_wall_ns = time.time_ns()
                    connection.flush()
                    electromagnet_on = next_state

                    state_text = "通电（有磁力）" if next_state else "断电（无磁力）"
                    print(f"切换指令已发送，预期状态：{state_text}。")
                    print(f"指令时间：{format_wall_time(toggle_wall_ns)}")
                    print(f"单调时钟计数：{toggle_monotonic_ns} ns")
            finally:
                # De-energize before closing the current serial connection.
                if electromagnet_on and connection.is_open:
                    try:
                        connection.write(b"0")
                        connection.flush()
                        electromagnet_on = False
                        print("已执行异常断电保护。", file=sys.stderr)
                    except serial.SerialException:
                        print("警告：异常断电指令发送失败，请手动断电。", file=sys.stderr)

    except KeyboardInterrupt:
        print("\n操作已取消。", file=sys.stderr)
    except EOFError:
        print("\n输入已结束。", file=sys.stderr)
    except serial.SerialException as exc:
        print(f"串口错误：{exc}", file=sys.stderr)

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
