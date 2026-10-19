"""Standalone test controller (no ROS2/coordinator needed): drives ONE robot straight to a pickup
station, runs the same approach -> rotate -> grab sequence youbot_mecanum.py uses, then holds
position -- for interactively watching/tuning that sequence without needing the full WSL/ROS2
stack running. Created per direct user request to speed up iteration on the pickup geometry fix.

Run *by Webots itself* -- assign this as a robot's ``controller`` field (e.g. robot_4) directly
in the world file / Webots' scene tree. Reuses youbot_mecanum.py's pure kinematics/pose/constants
via import (not duplicated), so any tuning done there (arm poses, PICKUP_STOP_OFFSET_M,
PICKUP_FACE_THETA_RAD, etc.) is picked up here automatically -- no need to keep two copies in
sync.

The world's ``controllerArgs`` (``--robot-id``/``--start-row``/``--start-col``) are read exactly
like youbot_mecanum.py does, so this only needs the robot's ``controller`` field changed -- no
world regeneration needed. An optional extra ``--pickup-row`` controllerArgs entry (default 1)
selects *which* pickup station to target -- this project's three pickup pads sit at rows 1, 6, 11
(column 0 always -- see warehouse_grid.generate_warehouse_map/np.linspace), so passing
``--pickup-row 6`` or ``--pickup-row 11`` runs the exact same proven sequence against the other
two stations, letting each be verified independently on its own robot (per direct user request,
after confirming it worked correctly at row 1).
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "youbot_mecanum"))
import youbot_mecanum as ybm


def _parse_pickup_row() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pickup-row", type=int, default=1)
    args, _unknown = parser.parse_known_args(sys.argv[1:])
    return args.pickup_row


MANUAL_ANGLE_STEP_RAD = 0.05
GRIP_AUTO_CLOSE_SECONDS = 10.0


def main() -> None:
    from controller import Keyboard, Supervisor

    args = ybm._parse_args()
    robot_id = args.robot_id
    target_pickup_cell = (_parse_pickup_row(), 0)

    robot = Supervisor()
    self_node = robot.getSelf()
    keyboard = robot.getKeyboard()
    keyboard.enable(ybm.TIME_STEP_MS)

    motors = []
    for candidates, label in zip(ybm._WHEEL_MOTOR_CANDIDATES, ybm._WHEEL_NAMES):
        motor, _matched = ybm._get_device_any(robot, candidates)
        if motor is None:
            print(f"[goto_pickup_test robot_{robot_id}] FATAL: no wheel motor device found for "
                  f"the {label} wheel.", flush=True)
            raise RuntimeError(f"no wheel motor device found for {label} wheel")
        motor.setPosition(float("inf"))
        motor.setVelocity(0.0)
        motors.append(motor)

    sensors = []
    encoders_available = True
    for candidates, label in zip(ybm._WHEEL_MOTOR_CANDIDATES, ybm._WHEEL_NAMES):
        sensor_candidates = [f"{c}sensor" for c in candidates] + [f"{c} sensor" for c in candidates]
        sensor, _matched = ybm._get_device_any(robot, sensor_candidates)
        if sensor is None:
            encoders_available = False
        else:
            sensor.enable(ybm.TIME_STEP_MS)
        sensors.append(sensor)
    if not encoders_available:
        sensors = [None, None, None, None]

    arm_motors = {}
    for name in ybm._ARM_MOTOR_NAMES:
        motor = robot.getDevice(name)
        if motor is None:
            arm_motors = {}
            break
        motor.setVelocity(ybm._ARM_MAX_VELOCITY_RAD_S)
        arm_motors[name] = motor

    finger_motors = {}
    if arm_motors:
        for name in ybm._FINGER_MOTOR_NAMES:
            motor = robot.getDevice(name)
            if motor is None:
                arm_motors, finger_motors = {}, {}
                break
            motor.setVelocity(ybm._FINGER_MAX_VELOCITY_M_S)
            finger_motors[name] = motor

    arm_sensors = {}
    for name in arm_motors:
        sensor = robot.getDevice(f"{name}sensor")
        if sensor is not None:
            sensor.enable(ybm.TIME_STEP_MS)
            arm_sensors[name] = sensor
    if len(arm_sensors) != len(arm_motors):
        arm_sensors = {}

    finger_sensors = {}
    for name in finger_motors:
        sensor = robot.getDevice(f"{name}sensor")
        if sensor is not None:
            sensor.enable(ybm.TIME_STEP_MS)
            finger_sensors[name] = sensor
    if len(finger_sensors) != len(finger_motors):
        finger_sensors = {}

    def _arm_at_pose(target: dict[str, float]) -> bool:
        if not arm_sensors:
            return False
        return all(
            abs(arm_sensors[name].getValue() - target[name]) <= ybm._ARM_POSE_TOLERANCE_RAD
            for name in target
        )

    def _fingers_at(target_m: float) -> bool:
        if not finger_sensors:
            return False
        return all(
            abs(sensor.getValue() - target_m) <= ybm._FINGER_TOLERANCE_M
            for sensor in finger_sensors.values()
        )

    last_commanded_arm_pose = [dict(ybm._ARM_POSE_REST_RAD)]

    def _apply_arm_pose(pose_rad: dict[str, float], gripper_open: bool) -> None:
        for name, motor in arm_motors.items():
            motor.setPosition(pose_rad[name])
        target = ybm._FINGER_OPEN_M if gripper_open else ybm._FINGER_CLOSED_M
        for motor in finger_motors.values():
            motor.setPosition(target)
        last_commanded_arm_pose[0] = dict(pose_rad)

    if arm_motors:
        _apply_arm_pose(ybm._ARM_POSE_REST_RAD, gripper_open=True)

    pickup_camera = robot.getDevice("pickup_camera")
    pickup_camera_node = robot.getFromDef(f"PICKUP_CAMERA_{robot_id}")
    pickup_camera_translation_field = None
    pickup_camera_rotation_field = None
    if pickup_camera is not None:
        pickup_camera.enable(ybm.TIME_STEP_MS)
        pickup_camera.recognitionEnable(ybm.TIME_STEP_MS)
    if pickup_camera_node is not None:
        pickup_camera_translation_field = pickup_camera_node.getField("translation")
        pickup_camera_rotation_field = pickup_camera_node.getField("rotation")

    start_x, start_y = ybm._cell_to_world((args.start_row, args.start_col))
    pose = {"x": start_x, "y": start_y, "theta": 0.0}
    last_wheel_positions = [None, None, None, None]

    row0, col0 = target_pickup_cell
    expected_pickup_node = robot.getFromDef(f"PACKAGE_{row0}_{col0}")

    def _current_arm_angles() -> dict[str, float]:
        if arm_sensors:
            return {name: arm_sensors[name].getValue() for name in ybm._ARM_MOTOR_NAMES}
        return dict(last_commanded_arm_pose[0])

    def _locate_box_via_camera(camera_chassis_pos, camera_rot) -> tuple[float, float] | None:
        if pickup_camera is None or expected_pickup_node is None:
            return None
        expected_id = expected_pickup_node.getId()
        for obj in pickup_camera.getRecognitionObjects():
            if obj.getId() != expected_id:
                continue
            fx, fy, fz = obj.getPosition()
            return ybm._camera_point_to_world(
                fx, fy, fz, camera_chassis_pos, camera_rot,
                pose["x"], pose["y"], pose["theta"],
            )
        return None

    sensed_box_xy = None
    box_x, box_y = ybm._cell_to_world(target_pickup_cell)
    stop_x, stop_y = box_x + ybm.PICKUP_STOP_OFFSET_M, box_y
    face_theta = ybm.PICKUP_FACE_THETA_RAD
    phase = "approach"
    phase_step = 0
    package_node = None
    reach_confirmed_at = None

    manual_pose = dict(ybm._ARM_POSE_REST_RAD)
    manual_gripper_open = True
    manual_selected = "arm1"
    manual_prev_pressed: set[int] = set()
    manual_grip_close_deadline = None
    print(
        f"[goto_pickup_test robot_{robot_id}] start pose ({start_x:.4f}, {start_y:.4f}), "
        f"target pickup cell {target_pickup_cell}, box at ({box_x:.4f}, {box_y:.4f}), "
        f"stopping short at ({stop_x:.4f}, {stop_y:.4f})",
        flush=True,
    )

    while robot.step(ybm.TIME_STEP_MS) != -1:
        vx_cmd = vy_cmd = omega_cmd = 0.0

        gripper_pos, gripper_rot = ybm._gripper_chassis_local_pose(_current_arm_angles())
        camera_rot = ybm._mat_mul(gripper_rot, ybm._CAMERA_LOOK_ALIGNMENT)
        if pickup_camera_translation_field is not None:
            pickup_camera_translation_field.setSFVec3f(list(gripper_pos))
        if pickup_camera_rotation_field is not None:
            pickup_camera_rotation_field.setSFRotation(list(ybm._mat_to_axis_angle(camera_rot)))

        if phase == "approach":
            detected = _locate_box_via_camera(gripper_pos, camera_rot)
            if detected is not None:
                sensed_box_xy = detected
            box_x, box_y = sensed_box_xy if sensed_box_xy is not None else ybm._cell_to_world(target_pickup_cell)
            stop_x, stop_y = box_x + ybm.PICKUP_STOP_OFFSET_M, box_y
            dxs, dys = stop_x - pose["x"], stop_y - pose["y"]
            dists = math.hypot(dxs, dys)
            if dists <= ybm.PICKUP_STOP_TOLERANCE_M:
                phase = "rotate"
                if sensed_box_xy is not None:
                    face_theta = math.atan2(box_y - pose["y"], box_x - pose["x"])
                else:
                    face_theta = ybm.PICKUP_FACE_THETA_RAD
                print(
                    f"[goto_pickup_test robot_{robot_id}] stopped short of box at "
                    f"({pose['x']:.4f}, {pose['y']:.4f}), sensed box at ({box_x:.4f}, {box_y:.4f}) "
                    f"[camera={'yes' if sensed_box_xy is not None else 'no, cell-center fallback'}] "
                    f"-- now rotating to face it (target theta={face_theta:.4f}).",
                    flush=True,
                )
            else:
                speed = min(ybm.MAX_BODY_SPEED_MPS, max(ybm.PICKUP_APPROACH_MIN_SPEED_MPS, ybm.KP_LINEAR * dists))
                vx_world = speed * dxs / dists
                vy_world = speed * dys / dists
                vx_cmd, vy_cmd = ybm._world_to_body(vx_world, vy_world, pose["theta"])
        elif phase == "rotate":
            ajust = ybm._wrap_angle(face_theta + 0.003)
            angle_diff = ybm._wrap_angle(ajust - pose["theta"])
            if abs(angle_diff) <= ybm.ANGLE_TOLERANCE_RAD:
                row, col = target_pickup_cell
                def_name = f"PACKAGE_{row}_{col}"
                package_node = robot.getFromDef(def_name)
                print(
                    f"[goto_pickup_test robot_{robot_id}] facing box (DEF {def_name} found="
                    f"{package_node is not None}) -- reaching.",
                    flush=True,
                )
                if arm_motors:
                    _apply_arm_pose(ybm._ARM_POSE_CARRY_RAD, gripper_open=True)
                phase, phase_step = "reach", 0
            else:
                omega_cmd = max(
                    -ybm.MAX_ANGULAR_SPEED_RAD_S,
                    min(ybm.MAX_ANGULAR_SPEED_RAD_S, ybm.KP_ANGULAR * angle_diff),
                )
        elif phase == "reach":
            phase_step += 1
            if reach_confirmed_at is None:
                if _arm_at_pose(ybm._ARM_POSE_CARRY_RAD) or phase_step >= ybm._POSE_TIMEOUT_STEPS:
                    reach_confirmed_at = time.time()
                    print(
                        f"[goto_pickup_test robot_{robot_id}] reached CARRY pose, waiting "
                        f"{ybm.GRAB_DELAY_SECONDS:.1f}s before closing gripper.",
                        flush=True,
                    )
            elif time.time() - reach_confirmed_at >= ybm.GRAB_DELAY_SECONDS:
                print(f"[goto_pickup_test robot_{robot_id}] closing gripper.", flush=True)
                if arm_motors:
                    _apply_arm_pose(ybm._ARM_POSE_CARRY_RAD, gripper_open=False)
                phase, phase_step = "close_gripper", 0
                reach_confirmed_at = None
        elif phase == "close_gripper":
            phase_step += 1
            if _fingers_at(ybm._FINGER_CLOSED_M) or phase_step >= ybm._POSE_TIMEOUT_STEPS:
                print(f"[goto_pickup_test robot_{robot_id}] lifting.", flush=True)
                if arm_motors:
                    _apply_arm_pose(ybm._ARM_POSE_LIFTED_RAD, gripper_open=False)
                phase, phase_step = "lift", 0
        elif phase == "lift":
            phase_step += 1
            if _arm_at_pose(ybm._ARM_POSE_LIFTED_RAD) or phase_step >= ybm._POSE_TIMEOUT_STEPS:
                print(
                    f"[goto_pickup_test robot_{robot_id}] PICKUP COMPLETE\n"
                    f"  robot pose x, y, theta(rad)  = "
                    f"({pose['x']:.4f}, {pose['y']:.4f}, {pose['theta']:.4f})\n"
                    f"  package_node found            = {package_node is not None}\n"
                    f"  manual control still available -- 1-5 select joint, Up/Down adjust, "
                    f"G=gripper, R=rest, Y=carry (auto-grabs {GRIP_AUTO_CLOSE_SECONDS:.0f}s "
                    f"later), L=lifted, "
                    f"P=print pose\n"
                    f"  HOLDING position here (this controller never continues to dropoff)",
                    flush=True,
                )
                if package_node is None:
                    print(
                        f"[goto_pickup_test robot_{robot_id}] WARNING: no PACKAGE node found -- "
                        f"world file may be stale, regenerate it.",
                        flush=True,
                    )
                manual_pose = dict(ybm._ARM_POSE_LIFTED_RAD)
                manual_gripper_open = True
                if arm_motors:
                    _apply_arm_pose(manual_pose, gripper_open=manual_gripper_open)
                phase = "done"

        if phase == "done" and arm_motors:
            pressed = set()
            key = keyboard.getKey()
            while key != -1:
                pressed.add(key)
                key = keyboard.getKey()
            just_pressed = pressed - manual_prev_pressed
            manual_changed = False

            for i in range(1, 6):
                if ord(str(i)) in just_pressed:
                    manual_selected = f"arm{i}"
                    print(f"[goto_pickup_test robot_{robot_id}] selected joint = "
                          f"{manual_selected}", flush=True)

            if Keyboard.UP in pressed:
                lo, hi = ybm._ARM_JOINT_LIMITS_RAD[manual_selected]
                manual_pose[manual_selected] = min(hi, manual_pose[manual_selected] + MANUAL_ANGLE_STEP_RAD)
                manual_changed = True
            if Keyboard.DOWN in pressed:
                lo, hi = ybm._ARM_JOINT_LIMITS_RAD[manual_selected]
                manual_pose[manual_selected] = max(lo, manual_pose[manual_selected] - MANUAL_ANGLE_STEP_RAD)
                manual_changed = True
            if ord("G") in just_pressed:
                manual_gripper_open = not manual_gripper_open
                manual_grip_close_deadline = None
                manual_changed = True
            if ord("R") in just_pressed:
                manual_pose = dict(ybm._ARM_POSE_REST_RAD)
                manual_gripper_open = True
                manual_grip_close_deadline = None
                manual_changed = True
            if ord("Y") in just_pressed:
                manual_pose = dict(ybm._ARM_POSE_CARRY_RAD)
                manual_gripper_open = ybm._CARRY_GRIPPER_OPEN
                manual_grip_close_deadline = time.time() + GRIP_AUTO_CLOSE_SECONDS
                manual_changed = True
            if ord("L") in just_pressed:
                manual_pose = dict(ybm._ARM_POSE_LIFTED_RAD)
                manual_gripper_open = False
                manual_grip_close_deadline = None
                manual_changed = True
            if ord("P") in just_pressed:
                formatted = ", ".join(f'"{k}": {round(v, 4)}' for k, v in manual_pose.items())
                print(f"[goto_pickup_test robot_{robot_id}] current pose: {{{formatted}}}  "
                      f"gripper_open={manual_gripper_open}", flush=True)

            if manual_grip_close_deadline is not None and time.time() >= manual_grip_close_deadline:
                print(
                    f"[goto_pickup_test robot_{robot_id}] auto-closing gripper "
                    f"({GRIP_AUTO_CLOSE_SECONDS:.1f}s after Y).",
                    flush=True,
                )
                manual_gripper_open = False
                manual_grip_close_deadline = None
                manual_changed = True

            if manual_changed:
                _apply_arm_pose(manual_pose, gripper_open=manual_gripper_open)
                formatted = ", ".join(f'"{k}": {round(v, 4)}' for k, v in manual_pose.items())
                print(f"[goto_pickup_test robot_{robot_id}] current pose: {{{formatted}}}  "
                      f"gripper_open={manual_gripper_open}", flush=True)

            manual_prev_pressed = pressed

        wheel_speeds = ybm._inverse_kinematics(vx_cmd, vy_cmd, omega_cmd)
        for motor, speed in zip(motors, wheel_speeds):
            motor.setVelocity(speed)

        if encoders_available:
            deltas = []
            for i, sensor in enumerate(sensors):
                pos = sensor.getValue()
                if last_wheel_positions[i] is None:
                    deltas.append(0.0)
                else:
                    deltas.append((pos - last_wheel_positions[i]) * ybm._WHEEL_RADIUS_M)
                last_wheel_positions[i] = pos
            body_dx, body_dy, dtheta = ybm._forward_kinematics(deltas)
            cos_t, sin_t = math.cos(pose["theta"]), math.sin(pose["theta"])
            pose["x"] += body_dx * cos_t - body_dy * sin_t
            pose["y"] += body_dx * sin_t + body_dy * cos_t
            pose["theta"] += dtheta
        else:
            pose["x"] += vx_cmd * ybm.TIME_STEP_S
            pose["y"] += vy_cmd * ybm.TIME_STEP_S

        if self_node is not None:
            true_x, true_y, _true_z = self_node.getField("translation").getSFVec3f()
            pose["x"], pose["y"] = true_x, true_y


if __name__ == "__main__":
    main()
