"""OpenMV diagnostic: check firmware, script status, and try to run camera."""
import serial
import struct
import time
import sys

USBDBG_COMMAND = 0x30
USBDBG_FW_VERSION = 0x80
USBDBG_FRAME_SIZE = 0x81
USBDBG_FRAME_DUMP = 0x82
USBDBG_FB_ENABLE = 0x0D
USBDBG_SCRIPT_SAVE = 0x08
USBDBG_SCRIPT_RUNNING = 0x09
USBDBG_TX_INPUT = 0x11

COMMAND_HEADER = struct.Struct("<BBI")
TRIPLE_U32 = struct.Struct("<III")

def read_exact(link, size):
    data = bytearray()
    deadline = time.monotonic() + 2.0
    while len(data) < size:
        chunk = link.read(size - len(data))
        if chunk:
            data.extend(chunk)
            deadline = time.monotonic() + 2.0
        elif time.monotonic() >= deadline:
            raise RuntimeError(f"Timeout ({len(data)}/{size})")
    return bytes(data)

def usbdbg_request(link, opcode, resp_size):
    link.write(COMMAND_HEADER.pack(USBDBG_COMMAND, opcode, resp_size))
    link.flush()
    return read_exact(link, resp_size)

def usbdbg_write(link, opcode, payload=b""):
    link.write(COMMAND_HEADER.pack(USBDBG_COMMAND, opcode, len(payload)))
    if payload:
        link.write(payload)
    link.flush()

port = "COM5"
if len(sys.argv) > 1:
    port = sys.argv[1]

print(f"Connecting to {port}...")
try:
    with serial.Serial(port, 921600, timeout=2.0, write_timeout=1.0) as link:
        time.sleep(0.3)
        link.reset_input_buffer()

        # 1. Firmware version
        ver = TRIPLE_U32.unpack(usbdbg_request(link, USBDBG_FW_VERSION, 12))
        print(f"[1] Firmware: {ver[0]}.{ver[1]}.{ver[2]}")

        # 2. Check if script is running
        try:
            running = struct.unpack("<I", usbdbg_request(link, USBDBG_SCRIPT_RUNNING, 4))[0]
            print(f"[2] Script running: {running}")
        except Exception as e:
            print(f"[2] Script running check failed: {e}")

        # 3. Enable framebuffer
        usbdbg_write(link, USBDBG_FB_ENABLE, struct.pack("<I", 1))
        print("[3] framebuffer_enable(True) sent")
        time.sleep(0.5)

        # 4. Check frame size
        w, h, sz = TRIPLE_U32.unpack(usbdbg_request(link, USBDBG_FRAME_SIZE, 12))
        print(f"[4] frame_size: {w}x{h}, {sz} bytes")

        if sz > 0:
            print("    Camera is producing frames!")
        else:
            print("    No frames. Trying to save a startup script...")
            # 5. Save a minimal camera script to the OpenMV
            startup_script = b"""import sensor
sensor.reset()
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QVGA)
sensor.skip_frames(time=1000)
print("Camera initialized OK")
while True:
    sensor.snapshot()
"""
            usbdbg_write(link, USBDBG_SCRIPT_SAVE, startup_script)
            print("[5] Script saved to OpenMV. Waiting for restart...")
            time.sleep(3.0)

            # 6. Re-check
            link.reset_input_buffer()
            usbdbg_write(link, USBDBG_FB_ENABLE, struct.pack("<I", 1))
            time.sleep(0.5)
            w, h, sz = TRIPLE_U32.unpack(usbdbg_request(link, USBDBG_FRAME_SIZE, 12))
            print(f"[6] frame_size after script save: {w}x{h}, {sz} bytes")
            if sz > 0:
                print("    SUCCESS! Camera is now producing frames.")
            else:
                print("    Still no frames. Possible hardware issue.")

except Exception as e:
    print(f"Error: {e}")
