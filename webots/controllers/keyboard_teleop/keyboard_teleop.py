"""Manual keyboard-teleop test controller for the youBot mecanum base.

**Standalone diagnostic tool, not part of the ROS2/coordinator pipeline.** Run this as a robot's
*embedded* Webots controller (no ROS2, no WSL, no extern-controller setup needed at all -- this
runs directly inside Webots on Windows) to manually drive one youBot with the keyboard and
directly see whether the corrected wheel-mapping/kinematics (see
docs/webots_ros2_integration.md sections 8a/8d) actually behaves right, decoupled from the
lock-step/ROS2 pipeline that was making it hard to tell which layer a problem was in.

Reuses ``_inverse_kinematics``/``_WHEEL_RADIUS_M``/``_HALF_LENGTH_M``/``_HALF_WIDTH_M`` directly
from ``youbot_mecanum.py`` (imported from the sibling controller directory) rather than
duplicating them, so this test always reflects the exact same, already-tested kinematics the real
controller uses -- see ``tests/test_youbot_kinematics.py``.

Controls (arrow keys for translation, Q/E to rotate -- hold multiple at once for diagonal
strafing, e.g. Up+Right):

    Up / Down     -> drive forward / backward
    Left / Right  -> strafe left / right (true sideways motion, no turning -- this is the
                     part worth watching closely: a correct mecanum robot moves sideways
                     without rotating at all)
    Q / E         -> rotate left / right in place
    (release all keys to stop)

Setup: generate a single-robot world with this as the controller, no ROS2/WSL involved:

    .venv\\Scripts\\python.exe scripts\\generate_webots_world.py --robots 1 \\
        --controller keyboard_teleop --output webots\\worlds\\keyboard_test.wbt

Open that world in Webots, press Play, then **click on the 3D view first** so the Webots window
has keyboard focus -- key presses are silently ignored otherwise, which looks identical to "it's
not working."
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "youbot_mecanum"))
from youbot_mecanum import (
    _HALF_LENGTH_M,
    _HALF_WIDTH_M,
    _WHEEL_RADIUS_M,
    _inverse_kinematics,
)

TIME_STEP_MS = 32
SPEED_MPS = 0.3
YAW_RATE_RAD_S = 1.0

_MOTOR_NAMES = ["wheel1", "wheel2", "wheel3", "wheel4"]


def main() -> None:
    from controller import Keyboard, Robot

    robot = Robot()

    motors = []
    for name in _MOTOR_NAMES:
        motor = robot.getDevice(name)
        if motor is None:
            print(f"[keyboard_teleop] FATAL: wheel motor device '{name}' not found on this robot.")
            return
        motor.setPosition(float("inf"))
        motor.setVelocity(0.0)
        motors.append(motor)

    keyboard = robot.getKeyboard()
    keyboard.enable(TIME_STEP_MS)

    print(
        "[keyboard_teleop] Ready. Click the 3D view for keyboard focus, then use "
        "Up/Down/Left/Right to strafe and Q/E to rotate. Release all keys to stop."
    )
    print(
        f"[keyboard_teleop] wheel radius={_WHEEL_RADIUS_M} half_length={_HALF_LENGTH_M} "
        f"half_width={_HALF_WIDTH_M} (from youbot_mecanum.py)"
    )

    while robot.step(TIME_STEP_MS) != -1:
        pressed = set()
        key = keyboard.getKey()
        while key != -1:
            pressed.add(key)
            key = keyboard.getKey()

        vx = vy = omega = 0.0
        if Keyboard.UP in pressed:
            vx += SPEED_MPS
        if Keyboard.DOWN in pressed:
            vx -= SPEED_MPS
        if Keyboard.LEFT in pressed:
            vy += SPEED_MPS
        if Keyboard.RIGHT in pressed:
            vy -= SPEED_MPS
        if ord("Q") in pressed:
            omega += YAW_RATE_RAD_S
        if ord("E") in pressed:
            omega -= YAW_RATE_RAD_S

        wheel_speeds = _inverse_kinematics(vx, vy, omega)
        for motor, speed in zip(motors, wheel_speeds):
            motor.setVelocity(speed)


if __name__ == "__main__":
    main()
