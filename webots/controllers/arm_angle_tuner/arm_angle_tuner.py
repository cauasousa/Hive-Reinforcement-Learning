"""Interactive arm-joint-angle tuning tool (standalone, no ROS2 needed).

**Diagnostic tool, not part of the ROS2/coordinator pipeline.** Assign this as a robot's embedded
Webots controller to manually move each of the youBot's 5 arm joints with the keyboard and watch
the pose live, so a good pose (e.g. a replacement for _ARM_POSE_CARRY_RAD) can be found by feel
instead of guessing numbers blind -- created per direct user request while tuning the pickup-grab
pose in youbot_mecanum.py.

Reuses joint limits/motor names/existing poses from youbot_mecanum.py (imported from the sibling
controller directory) rather than duplicating them.

Also drives the gripper-mounted pickup camera (see youbot_mecanum.py's _gripper_camera_pose
docstring for why it's a bodySlot child with its pose overwritten every step rather than a real
child of the arm) from whatever pose is currently commanded here -- so tuning a pose with this
tool shows the same live camera view youbot_mecanum.py would produce for that pose, not a stale
placeholder position. Needs Supervisor (not plain Robot) to reach the camera node's fields, and
needs this robot's own id (parsed from its ``name`` field, e.g. "robot_3" -> 3, since this
controller -- unlike youbot_mecanum.py -- takes no ``--robot-id`` controllerArgs) to look up its
own camera rather than another robot's.

Reads real arm/finger position sensors (added for this -- see the module's earlier state) so the
gripper camera's pose (and the T auto-grab sequence's "did it actually get there" checks) reflect
where the arm *physically* is mid-motion, not an instantaneous jump to whatever was last commanded
-- confirmed live to matter: with only the commanded target available, the camera used to snap
straight to the destination the instant a key was pressed, while the visible arm mesh was still
smoothly interpolating there over the next second or so.

Also prints live feedback whenever the pickup camera recognizes something in view (resolved back
to a real Webots node via ``Supervisor.getFromId`` on the recognized object's id, so it can name
whatever it sees -- typically a ``PACKAGE_<row>_<col>`` box -- without needing to already know
which one to expect).

Controls (click the 3D view first for keyboard focus):
    1-5           -> select which arm joint (arm1..arm5) Up/Down controls
    Up / Down     -> increase / decrease the selected joint's angle (clamped to its real limits)
    G             -> toggle the gripper open/closed
    R             -> snap to _ARM_POSE_REST_RAD (youbot_mecanum.py's current idle/driving pose)
    Y             -> snap to _ARM_POSE_CARRY_RAD (gripper stays open, auto-closes after
                     GRIP_AUTO_CLOSE_SECONDS of real time)
    L             -> snap to _ARM_POSE_LIFTED_RAD (youbot_mecanum.py's current lift/carry pose)
    T             -> autonomous grab attempt: CARRY pose -> (once sensor-confirmed arrived, after
                     a GRAB_DELAY_SECONDS beat) close the gripper -> LIFT pose -- the same staged,
                     sensor-confirmed sequence youbot_mecanum.py runs during a real pickup, just
                     triggered by a single keypress here instead of driving there. Meant to be
                     pressed once the operator has manually parked the arm/base somewhere the
                     pickup camera is already reporting the box (see the recognition feedback
                     above) -- this key does NOT drive the base or otherwise move toward the box
                     itself, only the arm/gripper.
    P             -> print the current pose as a ready-to-paste Python dict

The selected joint and current pose are printed every time something changes. Manual joint/G/R/Y/L
keys are ignored while a T sequence is in progress (to avoid fighting it) -- Y with a pending
auto-close still cancels other things as before.
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "youbot_mecanum"))
from youbot_mecanum import (
    _ARM_JOINT_LIMITS_RAD,
    _ARM_MAX_VELOCITY_RAD_S,
    _ARM_MOTOR_NAMES,
    _ARM_POSE_CARRY_RAD,
    _ARM_POSE_LIFTED_RAD,
    _ARM_POSE_REST_RAD,
    _ARM_POSE_TOLERANCE_RAD,
    _CAMERA_LOOK_ALIGNMENT,
    _CARRY_GRIPPER_OPEN,
    _FINGER_CLOSED_M,
    _FINGER_MAX_VELOCITY_M_S,
    _FINGER_MOTOR_NAMES,
    _FINGER_OPEN_M,
    _FINGER_TOLERANCE_M,
    _POSE_TIMEOUT_STEPS,
    GRAB_DELAY_SECONDS,
    _camera_point_to_world,
    _gripper_chassis_local_pose,
    _mat_mul,
    _mat_to_axis_angle,
    _mat_vec,
)

TIME_STEP_MS = 32
ANGLE_STEP_RAD = 0.05
GRIP_AUTO_CLOSE_SECONDS = 3.0
RECOGNITION_PRINT_INTERVAL_STEPS = 15
DIAGNOSTIC_PRINT_INTERVAL_STEPS = 30


def main() -> None:
    from controller import Keyboard, Supervisor

    robot = Supervisor()

    robot_name = robot.getName()
    robot_id = None
    if robot_name.startswith("robot_"):
        try:
            robot_id = int(robot_name.rsplit("_", 1)[-1])
        except ValueError:
            robot_id = None
    pickup_camera_node = robot.getFromDef(f"PICKUP_CAMERA_{robot_id}") if robot_id is not None else None
    pickup_camera_translation_field = None
    pickup_camera_rotation_field = None
    if pickup_camera_node is not None:
        pickup_camera_translation_field = pickup_camera_node.getField("translation")
        pickup_camera_rotation_field = pickup_camera_node.getField("rotation")
    else:
        print(
            f"[arm_angle_tuner] WARNING: couldn't resolve this robot's own pickup camera "
            f"(name={robot_name!r}, robot_id={robot_id!r}) -- the camera won't track the arm in "
            f"this session (world file may be stale, regenerate it).",
            flush=True,
        )

    pickup_camera = robot.getDevice("pickup_camera")
    if pickup_camera is not None:
        pickup_camera.enable(TIME_STEP_MS)
        pickup_camera.recognitionEnable(TIME_STEP_MS)
        print(
            f"[arm_angle_tuner] pickup_camera.hasRecognition()="
            f"{pickup_camera.hasRecognition()}, getRecognitionSamplingPeriod()="
            f"{pickup_camera.getRecognitionSamplingPeriod()}",
            flush=True,
        )
    else:
        print(
            f"[arm_angle_tuner] WARNING: no 'pickup_camera' device found -- no box-recognition "
            f"feedback available this session.",
            flush=True,
        )

    arm_motors = {}
    for name in _ARM_MOTOR_NAMES:
        motor = robot.getDevice(name)
        if motor is None:
            print(f"[arm_angle_tuner] FATAL: arm motor '{name}' not found on this robot.")
            return
        motor.setVelocity(_ARM_MAX_VELOCITY_RAD_S)
        arm_motors[name] = motor

    finger_motors = {}
    for name in _FINGER_MOTOR_NAMES:
        motor = robot.getDevice(name)
        if motor is None:
            print(f"[arm_angle_tuner] FATAL: gripper motor '{name}' not found on this robot.")
            return
        motor.setVelocity(_FINGER_MAX_VELOCITY_M_S)
        finger_motors[name] = motor

    arm_sensors = {}
    for name in _ARM_MOTOR_NAMES:
        sensor = robot.getDevice(f"{name}sensor")
        if sensor is not None:
            sensor.enable(TIME_STEP_MS)
            arm_sensors[name] = sensor
    if len(arm_sensors) != len(arm_motors):
        arm_sensors = {}

    finger_sensors = {}
    for name in _FINGER_MOTOR_NAMES:
        sensor = robot.getDevice(f"{name}sensor")
        if sensor is not None:
            sensor.enable(TIME_STEP_MS)
            finger_sensors[name] = sensor
    if len(finger_sensors) != len(finger_motors):
        finger_sensors = {}

    pose = dict(_ARM_POSE_REST_RAD)
    gripper_open = True

    def _apply() -> None:
        for name, motor in arm_motors.items():
            motor.setPosition(pose[name])
        target = _FINGER_OPEN_M if gripper_open else _FINGER_CLOSED_M
        for motor in finger_motors.values():
            motor.setPosition(target)

    def _current_arm_angles() -> dict[str, float]:
        """Best available estimate of the arm's real current joint angles -- prefers live
        position sensors (the actual mid-motion position) and falls back to the commanded target
        only when sensors are unavailable."""
        if arm_sensors:
            return {name: arm_sensors[name].getValue() for name in _ARM_MOTOR_NAMES}
        return dict(pose)

    def _arm_at_pose(target: dict[str, float]) -> bool:
        if not arm_sensors:
            return False
        return all(
            abs(arm_sensors[name].getValue() - target[name]) <= _ARM_POSE_TOLERANCE_RAD
            for name in target
        )

    def _fingers_at(target_m: float) -> bool:
        if not finger_sensors:
            return False
        return all(
            abs(sensor.getValue() - target_m) <= _FINGER_TOLERANCE_M
            for sensor in finger_sensors.values()
        )

    def _print_pose(label: str) -> None:
        formatted = ", ".join(f'"{k}": {round(v, 4)}' for k, v in pose.items())
        print(
            f"[arm_angle_tuner] {label}: {{{formatted}}}  gripper_open={gripper_open}",
            flush=True,
        )

    _apply()
    _print_pose("initial pose (REST)")

    keyboard = robot.getKeyboard()
    keyboard.enable(TIME_STEP_MS)

    print(
        "[arm_angle_tuner] Ready. Click the 3D view for keyboard focus. Keys: 1-5 select joint, "
        "Up/Down adjust it, G=toggle gripper, R=rest pose, Y=carry pose, L=lifted pose, "
        "T=autonomous grab attempt, P=print pose.",
        flush=True,
    )

    selected = "arm1"
    prev_pressed: set[int] = set()
    grip_close_deadline = None

    auto_grab_phase = None
    auto_grab_step = 0
    auto_grab_reach_confirmed_at = None

    last_recognized_id = None
    recognition_print_countdown = 0
    diagnostic_print_countdown = 0

    while robot.step(TIME_STEP_MS) != -1:
        gripper_pos, gripper_rot = _gripper_chassis_local_pose(_current_arm_angles())
        camera_rot = _mat_mul(gripper_rot, _CAMERA_LOOK_ALIGNMENT)
        if pickup_camera_translation_field is not None:
            pickup_camera_translation_field.setSFVec3f(list(gripper_pos))
        if pickup_camera_rotation_field is not None:
            pickup_camera_rotation_field.setSFRotation(list(_mat_to_axis_angle(camera_rot)))

        recognition_print_countdown -= 1
        if pickup_camera is not None:
            recognized = pickup_camera.getRecognitionObjects()
            current_id = recognized[0].getId() if recognized else None
            if current_id is not None and (current_id != last_recognized_id or recognition_print_countdown <= 0):
                fx, fy, fz = recognized[0].getPosition()
                chassis_local = _camera_point_to_world(fx, fy, fz, gripper_pos, camera_rot, 0.0, 0.0, 0.0)
                node = robot.getFromId(current_id)
                label = node.getDef() if node is not None and node.getDef() else f"id={current_id}"
                print(
                    f"[arm_angle_tuner] camera sees '{label}' at chassis-local "
                    f"({chassis_local[0]:.4f}, {chassis_local[1]:.4f}).",
                    flush=True,
                )
                recognition_print_countdown = RECOGNITION_PRINT_INTERVAL_STEPS
            elif current_id is None and last_recognized_id is not None:
                print("[arm_angle_tuner] camera no longer sees anything.", flush=True)
            last_recognized_id = current_id

        diagnostic_print_countdown -= 1
        if diagnostic_print_countdown <= 0:
            raw_count = len(pickup_camera.getRecognitionObjects()) if pickup_camera is not None else -1
            axis_angle = _mat_to_axis_angle(camera_rot)
            look_dir = _mat_vec(camera_rot, (1.0, 0.0, 0.0))
            print(
                f"[arm_angle_tuner] DIAG camera chassis-local pos=({gripper_pos[0]:.4f}, "
                f"{gripper_pos[1]:.4f}, {gripper_pos[2]:.4f}) look_dir=({look_dir[0]:.3f}, "
                f"{look_dir[1]:.3f}, {look_dir[2]:.3f}) axis-angle=("
                f"{axis_angle[0]:.3f}, {axis_angle[1]:.3f}, {axis_angle[2]:.3f}, "
                f"{axis_angle[3]:.3f}) recognized_count={raw_count}",
                flush=True,
            )
            diagnostic_print_countdown = DIAGNOSTIC_PRINT_INTERVAL_STEPS

        pressed = set()
        key = keyboard.getKey()
        while key != -1:
            pressed.add(key)
            key = keyboard.getKey()

        just_pressed = pressed - prev_pressed
        changed = False

        if ord("T") in just_pressed and auto_grab_phase is None:
            print(
                "[arm_angle_tuner] T: starting autonomous grab attempt -- CARRY pose, then "
                "close gripper, then LIFT, each step sensor-confirmed.",
                flush=True,
            )
            auto_grab_phase = "reach"
            auto_grab_step = 0
            auto_grab_reach_confirmed_at = None
            grip_close_deadline = None
            pose = dict(_ARM_POSE_CARRY_RAD)
            gripper_open = _CARRY_GRIPPER_OPEN
            changed = True

        if auto_grab_phase is None:
            for i in range(1, 6):
                if ord(str(i)) in just_pressed:
                    selected = f"arm{i}"
                    print(f"[arm_angle_tuner] selected joint = {selected}", flush=True)

            if Keyboard.UP in pressed:
                lo, hi = _ARM_JOINT_LIMITS_RAD[selected]
                pose[selected] = min(hi, pose[selected] + ANGLE_STEP_RAD)
                changed = True
            if Keyboard.DOWN in pressed:
                lo, hi = _ARM_JOINT_LIMITS_RAD[selected]
                pose[selected] = max(lo, pose[selected] - ANGLE_STEP_RAD)
                changed = True
            if ord("G") in just_pressed:
                gripper_open = not gripper_open
                grip_close_deadline = None
                changed = True
            if ord("R") in just_pressed:
                pose = dict(_ARM_POSE_REST_RAD)
                gripper_open = True
                grip_close_deadline = None
                changed = True
            if ord("Y") in just_pressed:
                pose = dict(_ARM_POSE_CARRY_RAD)
                gripper_open = _CARRY_GRIPPER_OPEN
                grip_close_deadline = time.time() + GRIP_AUTO_CLOSE_SECONDS
                changed = True
            if ord("L") in just_pressed:
                pose = dict(_ARM_POSE_LIFTED_RAD)
                gripper_open = False
                grip_close_deadline = None
                changed = True

            if grip_close_deadline is not None and time.time() >= grip_close_deadline:
                print(
                    f"[arm_angle_tuner] auto-closing gripper ({GRIP_AUTO_CLOSE_SECONDS:.1f}s "
                    f"after Y).",
                    flush=True,
                )
                gripper_open = False
                grip_close_deadline = None
                changed = True

        if ord("P") in just_pressed:
            _print_pose("current pose")

        if auto_grab_phase == "reach":
            auto_grab_step += 1
            if auto_grab_reach_confirmed_at is None:
                if _arm_at_pose(_ARM_POSE_CARRY_RAD) or auto_grab_step >= _POSE_TIMEOUT_STEPS:
                    auto_grab_reach_confirmed_at = time.time()
                    print(
                        f"[arm_angle_tuner] T: reached CARRY pose, waiting "
                        f"{GRAB_DELAY_SECONDS:.1f}s before closing gripper.",
                        flush=True,
                    )
            elif time.time() - auto_grab_reach_confirmed_at >= GRAB_DELAY_SECONDS:
                print("[arm_angle_tuner] T: closing gripper.", flush=True)
                gripper_open = False
                _apply()
                auto_grab_phase, auto_grab_step = "close_gripper", 0
                auto_grab_reach_confirmed_at = None
        elif auto_grab_phase == "close_gripper":
            auto_grab_step += 1
            if _fingers_at(_FINGER_CLOSED_M) or auto_grab_step >= _POSE_TIMEOUT_STEPS:
                print("[arm_angle_tuner] T: lifting.", flush=True)
                pose = dict(_ARM_POSE_LIFTED_RAD)
                _apply()
                auto_grab_phase, auto_grab_step = "lift", 0
        elif auto_grab_phase == "lift":
            auto_grab_step += 1
            if _arm_at_pose(_ARM_POSE_LIFTED_RAD) or auto_grab_step >= _POSE_TIMEOUT_STEPS:
                print(
                    "[arm_angle_tuner] T: GRAB ATTEMPT COMPLETE -- manual controls re-enabled.",
                    flush=True,
                )
                auto_grab_phase = None

        if changed:
            _apply()
            _print_pose("current pose")

        prev_pressed = pressed


if __name__ == "__main__":
    main()
