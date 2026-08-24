"""Send one command for each arrow or gripper key press."""
import sys
import tkinter as tk
import serial

if len(sys.argv) != 2:
    print("用法: python pc_keyboard.py COM3")
    raise SystemExit(2)

try:
    port = serial.Serial(sys.argv[1], 115200, timeout=0.2)
except serial.SerialException as exc:
    print(f"无法打开串口: {exc}")
    raise SystemExit(1)

root = tk.Tk()
root.title("二维云台方向键控制")
root.geometry("420x140")
tk.Label(root, text="保持本窗口焦点：方向键控制云台，O 开夹，C 关夹\n每次按键执行一次动作").pack(expand=True)
pressed = set()

def key_down(event):
    if event.keysym in pressed:
        return
    pressed.add(event.keysym)
    table = {
        "Left": b"L", "Right": b"R", "Up": b"U", "Down": b"D",
        "o": b"O", "O": b"O", "c": b"C", "C": b"C",
    }
    if event.keysym in table:
        port.write(table[event.keysym])

def key_up(event):
    pressed.discard(event.keysym)

root.bind("<KeyPress>", key_down)
root.bind("<KeyRelease>", key_up)
root.protocol("WM_DELETE_WINDOW", lambda: (port.close(), root.destroy()))
root.mainloop()
