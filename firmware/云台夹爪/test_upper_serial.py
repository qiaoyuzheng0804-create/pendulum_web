"""Test the upper gimbal motor with the computer Left/Right keys."""
import sys
import tkinter as tk

import serial


if len(sys.argv) != 2:
    print("用法: python test_upper_serial.py COM3")
    raise SystemExit(2)

try:
    port = serial.Serial(sys.argv[1], 115200, timeout=0.2)
except serial.SerialException as exc:
    print(f"无法打开串口: {exc}")
    raise SystemExit(1)


root = tk.Tk()
root.title("上方云台电机串口测试")
root.geometry("430x150")
tk.Label(
    root,
    text="保持本窗口焦点：按电脑左键/右键测试上方电机\n左键发送 U，右键发送 D，每次约 2°",
).pack(expand=True)

pressed = set()


def key_down(event):
    if event.keysym in pressed:
        return
    pressed.add(event.keysym)
    command = {"Left": b"U", "Right": b"D"}.get(event.keysym)
    if command is not None:
        port.write(command)
        port.flush()


def key_up(event):
    pressed.discard(event.keysym)


def close_window():
    port.close()
    root.destroy()


root.bind("<KeyPress>", key_down)
root.bind("<KeyRelease>", key_up)
root.protocol("WM_DELETE_WINDOW", close_window)
root.mainloop()
