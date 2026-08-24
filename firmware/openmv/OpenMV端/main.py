"""OpenMV4 H7 Plus: USB camera and UART-controlled two-axis MG90S gimbal."""

import sensor
from pyb import Pin, Timer, UART


# One key press changes 10 us, about 0.9 degrees for the nominal
# 500..2500 us / 180 degree mapping.
SERVO_STEP_US = 10
SERVO_CENTER_US = 1500

# The vendor sample limits both axes to about 30..150 degrees. These pulse
# limits retain that safer mechanical range instead of commanding 0..180.
PAN_MIN_US = 835
PAN_MAX_US = 2165
TILT_MIN_US = 835
TILT_MAX_US = 2165

# Flip either sign if the physical direction is reversed on this mount.
PAN_RIGHT_SIGN = -1
TILT_UP_SIGN = -1


pan_pin = Pin("P1", Pin.OUT_PP)    # lower/base yaw servo
tilt_pin = Pin("P9", Pin.OUT_PP)  # upper pitch servo
pan_pulse_us = SERVO_CENTER_US
tilt_pulse_us = SERVO_CENTER_US
command_count = 0
received_count = 0
last_command = "-"


def clamp(value, minimum, maximum):
    if value < minimum:
        return minimum
    if value > maximum:
        return maximum
    return value


# This is the timer arrangement supplied for the 1.8-inch LCD version. A
# prescaler value of pulse_us * 10 - 1 produces the requested servo high time.
pan_pulse_timer = Timer(12)
pan_period_timer = Timer(13, freq=50)
tilt_pulse_timer = Timer(14)
tilt_period_timer = Timer(15, freq=50)


def pan_low(_timer):
    pan_pin.low()
    pan_pulse_timer.deinit()


def pan_period(_timer):
    pan_pin.high()
    pan_pulse_timer.init(prescaler=(pan_pulse_us * 10) - 1, period=23)
    pan_pulse_timer.callback(pan_low)


def tilt_low(_timer):
    tilt_pin.low()
    tilt_pulse_timer.deinit()


def tilt_period(_timer):
    tilt_pin.high()
    tilt_pulse_timer.init(prescaler=(tilt_pulse_us * 10) - 1, period=23)
    tilt_pulse_timer.callback(tilt_low)


pan_period_timer.callback(pan_period)
tilt_period_timer.callback(tilt_period)


# UART3 uses P4 (TX) and P5 (RX). The computer sends one ASCII command byte
# per key press through the external USB-TTL adapter.
CONTROL_BAUD = 115200
control_uart = UART(3, CONTROL_BAUD, timeout=0, timeout_char=0)


def apply_command(command):
    global pan_pulse_us, tilt_pulse_us, command_count, last_command

    recognized = True

    if command == ord("L"):
        pan_pulse_us = clamp(
            pan_pulse_us - (PAN_RIGHT_SIGN * SERVO_STEP_US),
            PAN_MIN_US,
            PAN_MAX_US,
        )
    elif command == ord("R"):
        pan_pulse_us = clamp(
            pan_pulse_us + (PAN_RIGHT_SIGN * SERVO_STEP_US),
            PAN_MIN_US,
            PAN_MAX_US,
        )
    elif command == ord("U"):
        tilt_pulse_us = clamp(
            tilt_pulse_us + (TILT_UP_SIGN * SERVO_STEP_US),
            TILT_MIN_US,
            TILT_MAX_US,
        )
    elif command == ord("D"):
        tilt_pulse_us = clamp(
            tilt_pulse_us - (TILT_UP_SIGN * SERVO_STEP_US),
            TILT_MIN_US,
            TILT_MAX_US,
        )
    elif command == ord("C"):
        pan_pulse_us = SERVO_CENTER_US
        tilt_pulse_us = SERVO_CENTER_US
    else:
        recognized = False

    if recognized:
        command_count += 1
        last_command = chr(command)


def receive_commands():
    global received_count

    available = control_uart.any()
    if available:
        data = control_uart.read(available)
        if data:
            received_count += len(data)
            for command in data:
                apply_command(command)


sensor.reset()
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.VGA)
sensor.set_windowing((560, 400))  # More detail than HVGA, less USB load than VGA.
sensor.skip_frames(time=1000)

while True:
    receive_commands()
    sensor.snapshot()
    # The computer reads this framebuffer through USBDBG V1, without the IDE.
