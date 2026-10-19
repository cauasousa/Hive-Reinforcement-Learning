"""Webots extern controller for one KUKA youBot (mecanum base + 5-DOF arm), bridged to ROS2.

Topics under /hiverl/robot_<id>: command (goal cell, status, pickup trigger), cmd_vel (RL velocity
for TO_ITEM/CARRYING), reset (teleport) -> arrived, pose. Velocity commands go through a directional
LiDAR brake and an optional yield nudge (right -> back -> left). IDLE robots drive to goal cells;
PICKING_UP runs the arm pickup sequence and the box is carried to the dropoff.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time


# Physics step.
TIME_STEP_MS = 32
TIME_STEP_S = TIME_STEP_MS / 1000.0

# Mecanum base geometry.
_WHEEL_RADIUS_M = 0.05
_HALF_LENGTH_M = 0.228
_HALF_WIDTH_M = 0.158
_K = _HALF_LENGTH_M + _HALF_WIDTH_M

# Goal-cell autopilot.
MAX_BODY_SPEED_MPS = 0.4
KP_LINEAR = 1.5
ARRIVAL_TOLERANCE_M = 0.30

# A cmd_vel older than this is ignored (robot holds still).
CMD_VEL_STALE_S = 2.0

# Pickup approach and heading control.
PICKUP_STOP_OFFSET_M = 0.40
PICKUP_STOP_TOLERANCE_M = 0.01
PICKUP_APPROACH_MIN_SPEED_MPS = 0.02
PICKUP_FACE_THETA_RAD = math.pi
KP_ANGULAR = 2.0
MAX_ANGULAR_SPEED_RAD_S = 1.0
ANGLE_TOLERANCE_RAD = 0.05
HEADING_REALIGN_TOLERANCE_RAD = 0.15


def _wrap_angle(angle_rad: float) -> float:
    """Wrap an angle to [-pi, pi]."""
    return (angle_rad + math.pi) % (2 * math.pi) - math.pi


def _heading_realign_omega(expected_theta: float, true_theta: float) -> float:
    """Angular correction toward expected_theta (0 inside the tolerance)."""
    angle_diff = _wrap_angle(expected_theta - true_theta)
    if abs(angle_diff) <= HEADING_REALIGN_TOLERANCE_RAD:
        return 0.0
    return max(-MAX_ANGULAR_SPEED_RAD_S, min(MAX_ANGULAR_SPEED_RAD_S, KP_ANGULAR * angle_diff))


# LiDAR brake: slow down below SLOW, stop at STOP (meters from the chassis center).
LIDAR_MAX_RANGE_M = 1.5
LIDAR_SAFETY_STOP_M = 0.35
LIDAR_SAFETY_SLOW_M = 0.9
LIDAR_SELF_DYNAMIC_FLOOR_M = 0.10
LIDAR_SELF_CALIBRATION_MARGIN_M = 0.10
# Startup self-mask only masks rays closer than this.
LIDAR_SELF_STARTUP_MAX_M = 0.25
LIDAR_SELF_CALIBRATION_WINDOW_STEPS = 10

# Reverse is never scaled below this by the brake.
LIDAR_REVERSE_SAFETY_FLOOR = 0.3
# RL mode: only returns in this cone around the direction of travel brake; any return under
# CONTACT_STOP stops.
LIDAR_DIRECTIONAL_HALF_WIDTH_DEG = 60.0
LIDAR_CONTACT_STOP_M = 0.25
# Returns closer than this are inside the robot itself and are ignored.
LIDAR_INSIDE_CHASSIS_M = 0.15


def _lidar_scan_mask(ranges, self_mask: list[bool] | None) -> list[bool]:
    """Self-mask plus, for this scan only, rays reading inside the chassis."""
    if self_mask is None:
        self_mask = [False] * len(ranges)
    return [
        (i < len(self_mask) and self_mask[i]) or (r == r and r < LIDAR_INSIDE_CHASSIS_M)
        for i, r in enumerate(ranges)
    ]


def _lidar_min_range(
    ranges, self_mask: list[bool] | None = None, dynamic_floor_m: float = 0.0
) -> float:
    """Closest valid return over the scan (masked rays ignored), clamped to dynamic_floor_m."""
    if self_mask is None:
        self_mask = [False] * len(ranges)
    finite = [
        r for r, is_self in zip(ranges, self_mask)
        if not is_self and r == r and r != float("inf")
    ]
    if not finite:
        return float("inf")
    return max(dynamic_floor_m, min(finite))


def _lidar_calibrate_self_mask(ranges, max_m: float | None = None) -> list[bool]:
    """Rays closer than the cutoff at calibration time (self-hits to ignore forever)."""
    threshold = LIDAR_SAFETY_STOP_M + LIDAR_SELF_CALIBRATION_MARGIN_M if max_m is None else max_m
    return [(r == r and r < threshold) for r in ranges]


def _lidar_safety_scale(min_range_m: float) -> float:
    """Speed scale: 1 beyond LIDAR_SAFETY_SLOW_M, 0 at LIDAR_SAFETY_STOP_M, linear in between."""
    if min_range_m <= LIDAR_SAFETY_STOP_M:
        return 0.0
    if min_range_m >= LIDAR_SAFETY_SLOW_M:
        return 1.0
    return (min_range_m - LIDAR_SAFETY_STOP_M) / (LIDAR_SAFETY_SLOW_M - LIDAR_SAFETY_STOP_M)


# Goal-cell mode stall recovery.
LIDAR_STALL_RECOVERY_STEPS = 94
LIDAR_STALL_TRIGGER_SCALE = 0.1
LIDAR_STALL_CREEP_SCALE = 0.15
LIDAR_STALL_CREEP_DURATION_STEPS = 31

LIDAR_STALL_RETREAT_RANGE_M = 0.20

HARD_STUCK_RECOVERY_STEPS = 470

HARD_STUCK_RETREAT_DISTANCE_M = 0.4


def _lidar_stall_recovery_scale(
    safety_scale: float, stall_steps: int, creep_remaining: int, retreat: bool = False
) -> tuple[float, int, int]:
    """One step of the goal-cell stall recovery (creep forward after a long brake)."""
    creep_scale = -LIDAR_STALL_CREEP_SCALE if retreat else LIDAR_STALL_CREEP_SCALE
    if creep_remaining > 0:
        return creep_scale, 0, creep_remaining - 1
    if safety_scale > LIDAR_STALL_TRIGGER_SCALE:
        return safety_scale, 0, 0
    stall_steps += 1
    if stall_steps >= LIDAR_STALL_RECOVERY_STEPS:
        return creep_scale, 0, LIDAR_STALL_CREEP_DURATION_STEPS - 1
    return safety_scale, stall_steps, 0


# Goal-cell mode lateral dodge.
LIDAR_AVOIDANCE_LATERAL_SPEED_MPS = 0.2
LIDAR_AVOIDANCE_WORSENING_EPS_M = 0.01
LIDAR_AVOIDANCE_WORSENING_STEPS = 5

# Position-based stall check (goal-cell mode).
POSITION_STALL_CHECK_STEPS = 94
POSITION_STALL_DISTANCE_M = 0.05


def _position_stall_detected(
    pos: tuple[float, float], ref_pos: tuple[float, float] | None, steps_since_ref: int
) -> tuple[bool, tuple[float, float], int]:
    """One step of the position-based stall check."""
    if ref_pos is None:
        return False, pos, 0
    steps_since_ref += 1
    if steps_since_ref < POSITION_STALL_CHECK_STEPS:
        return False, ref_pos, steps_since_ref
    moved_m = math.hypot(pos[0] - ref_pos[0], pos[1] - ref_pos[1])
    return moved_m < POSITION_STALL_DISTANCE_M, pos, 0


def _lidar_avoidance_lateral_cmd(
    lidar_range: float, prev_range: float, probe_sign: int, worsening_steps: int
) -> tuple[float, float, int, int]:
    """One step of the goal-cell lateral dodge (hill-climb on the min range)."""
    restriction = 1.0 - _lidar_safety_scale(lidar_range)
    if restriction <= 0.0:
        return 0.0, lidar_range, probe_sign, 0
    if lidar_range < prev_range - LIDAR_AVOIDANCE_WORSENING_EPS_M:
        worsening_steps += 1
    else:
        worsening_steps = 0
    if worsening_steps >= LIDAR_AVOIDANCE_WORSENING_STEPS:
        probe_sign = -probe_sign
        worsening_steps = 0
    lateral_cmd = LIDAR_AVOIDANCE_LATERAL_SPEED_MPS * probe_sign * restriction
    return lateral_cmd, lidar_range, probe_sign, worsening_steps


def _ray_index_for_local_angle_deg(
    angle_deg: float, horizontal_resolution: int, field_of_view_deg: float = 360.0
) -> int:
    """Ray index of a local angle (0 = front, +90 = left)."""
    step_deg = field_of_view_deg / horizontal_resolution
    return int(round((180.0 - angle_deg) / step_deg)) % horizontal_resolution


def _local_angle_deg_for_ray_index(
    index: int, horizontal_resolution: int, field_of_view_deg: float = 360.0
) -> float:
    """Local angle of a ray index."""
    step_deg = field_of_view_deg / horizontal_resolution
    return (180.0 - step_deg * index) % 360.0


def _sector_ray_indices(
    center_deg: float, half_width_deg: float, horizontal_resolution: int, field_of_view_deg: float = 360.0
) -> list[int]:
    """Ray indices covering [center - half_width, center + half_width] (local frame)."""
    idx_start = _ray_index_for_local_angle_deg(
        center_deg + half_width_deg, horizontal_resolution, field_of_view_deg
    )
    idx_end = _ray_index_for_local_angle_deg(
        center_deg - half_width_deg, horizontal_resolution, field_of_view_deg
    )
    if idx_end >= idx_start:
        return list(range(idx_start, idx_end + 1))
    return list(range(idx_start, horizontal_resolution)) + list(range(0, idx_end + 1))


def _sector_min_range(
    ranges,
    self_mask: list[bool] | None,
    center_deg: float,
    half_width_deg: float,
    horizontal_resolution: int,
    field_of_view_deg: float = 360.0,
) -> float:
    """Closest valid return in a sector (inf if none)."""
    if self_mask is None:
        self_mask = [False] * len(ranges)
    best = float("inf")
    for i in _sector_ray_indices(center_deg, half_width_deg, horizontal_resolution, field_of_view_deg):
        if i >= len(ranges) or (i < len(self_mask) and self_mask[i]):
            continue
        r = ranges[i]
        if r == r and r < best:
            best = r
    return best


# Goal-cell mode rotational scan.
LIDAR_ROTATIONAL_SCAN_OMEGA_RADPS = 0.3
LIDAR_ROTATIONAL_SCAN_SECTOR_HALF_WIDTH_DEG = 35.0
LIDAR_ROTATIONAL_SCAN_CLEARANCE_M = LIDAR_SAFETY_SLOW_M
LIDAR_ROTATIONAL_SCAN_FORWARD_FLANK_CENTER_DEG = 35.0
LIDAR_ROTATIONAL_SCAN_FORWARD_FLANK_HALF_WIDTH_DEG = 15.0


def _lidar_rotational_scan_cmd(
    ranges,
    self_mask: list[bool] | None,
    horizontal_resolution: int,
    field_of_view_deg: float = 360.0,
) -> tuple[float, bool]:
    """Goal-cell escape: turn right until the right side or a forward flank is clear."""
    right_range = _sector_min_range(
        ranges, self_mask, -90.0, LIDAR_ROTATIONAL_SCAN_SECTOR_HALF_WIDTH_DEG,
        horizontal_resolution, field_of_view_deg,
    )
    if right_range >= LIDAR_ROTATIONAL_SCAN_CLEARANCE_M:
        return -LIDAR_ROTATIONAL_SCAN_OMEGA_RADPS, True

    left_flank = _sector_min_range(
        ranges, self_mask,
        LIDAR_ROTATIONAL_SCAN_FORWARD_FLANK_CENTER_DEG, LIDAR_ROTATIONAL_SCAN_FORWARD_FLANK_HALF_WIDTH_DEG,
        horizontal_resolution, field_of_view_deg,
    )
    right_flank = _sector_min_range(
        ranges, self_mask,
        -LIDAR_ROTATIONAL_SCAN_FORWARD_FLANK_CENTER_DEG, LIDAR_ROTATIONAL_SCAN_FORWARD_FLANK_HALF_WIDTH_DEG,
        horizontal_resolution, field_of_view_deg,
    )
    if max(left_flank, right_flank) >= LIDAR_ROTATIONAL_SCAN_CLEARANCE_M:
        return 0.0, True

    return -LIDAR_ROTATIONAL_SCAN_OMEGA_RADPS, False


# Yield nudge in RL mode (disable with HIVERL_LIDAR_YIELD=0).
LIDAR_YIELD_ENABLED = os.environ.get("HIVERL_LIDAR_YIELD", "1") != "0"
# Nudge speed, max duration and cooldown.
YIELD_SPEED_MPS = 0.12
YIELD_DURATION_S = 0.8
YIELD_COOLDOWN_S = 1.0
# Fires when a forward command is braked to TRIGGER_SCALE; released above RELEASE_SCALE.
YIELD_MIN_FORWARD_MPS = 0.1
YIELD_TRIGGER_SCALE = 0.15
YIELD_RELEASE_SCALE = 0.6
YIELD_SECTOR_HALF_WIDTH_DEG = 30.0
# A side is free beyond this (a centered 1-cell aisle reads ~0.55 m, so it is not).
YIELD_SIDE_CLEAR_M = 0.6
YIELD_BACK_CLEAR_M = 0.6
# Stop the nudge if the chosen side gets this close.
YIELD_SIDE_ABORT_M = 0.32
YIELD_BACK_ABORT_M = 0.38
# Backing up waits base + per_id * robot_id, so the lower id backs off first.
YIELD_BACK_BASE_DELAY_S = 0.3
YIELD_BACK_DELAY_PER_ID_S = 0.4

_YIELD_SECTORS = {
    "right": (-90.0, YIELD_SIDE_CLEAR_M, YIELD_SIDE_ABORT_M),
    "back": (180.0, YIELD_BACK_CLEAR_M, YIELD_BACK_ABORT_M),
    "left": (90.0, YIELD_SIDE_CLEAR_M, YIELD_SIDE_ABORT_M),
}


def _sector_has_information(
    self_mask: list[bool] | None, center_deg: float, half_width_deg: float,
    horizontal_resolution: int, field_of_view_deg: float = 360.0,
) -> bool:
    """False if every ray of the sector is self-masked."""
    if self_mask is None:
        return True
    return any(
        not (i < len(self_mask) and self_mask[i])
        for i in _sector_ray_indices(center_deg, half_width_deg, horizontal_resolution, field_of_view_deg)
    )


def _lidar_yield_direction(
    ranges, self_mask: list[bool] | None, horizontal_resolution: int, field_of_view_deg: float = 360.0
) -> str | None:
    """First free direction in order right -> back -> left, or None."""
    for name in ("right", "back", "left"):
        center, clear_m, _ = _YIELD_SECTORS[name]
        if not _sector_has_information(
            self_mask, center, YIELD_SECTOR_HALF_WIDTH_DEG, horizontal_resolution, field_of_view_deg
        ):
            continue
        rng = _sector_min_range(
            ranges, self_mask, center, YIELD_SECTOR_HALF_WIDTH_DEG, horizontal_resolution, field_of_view_deg
        )
        if rng >= clear_m:
            return name
    return None


def _yield_body_velocity(direction: str) -> tuple[float, float]:
    """Body-frame (vx, vy) of a yield nudge (+vy is left)."""
    if direction == "right":
        return 0.0, -YIELD_SPEED_MPS
    if direction == "left":
        return 0.0, YIELD_SPEED_MPS
    return -YIELD_SPEED_MPS, 0.0


class _LidarYieldController:
    """Per-robot yield state machine: when the front blocks a forward command, nudge right/back/left
    briefly, then cool down. step() returns the (vx, vy) override or None.
    """

    def __init__(self, robot_id: int, time_step_s: float):
        self.time_step_s = time_step_s
        self.duration_steps = max(1, int(round(YIELD_DURATION_S / time_step_s)))
        self.cooldown_total_steps = max(1, int(round(YIELD_COOLDOWN_S / time_step_s)))
        self.back_delay_steps = max(
            1, int(round((YIELD_BACK_BASE_DELAY_S + YIELD_BACK_DELAY_PER_ID_S * robot_id) / time_step_s))
        )
        self.mode: str | None = None
        self.steps_left = 0
        self.cooldown_steps = 0
        self.blocked_steps = 0
        self.just_started: str | None = None

    def _finish(self) -> None:
        self.mode = None
        self.steps_left = 0
        self.blocked_steps = 0
        self.cooldown_steps = self.cooldown_total_steps

    def reset(self) -> None:
        self.mode = None
        self.steps_left = 0
        self.cooldown_steps = 0
        self.blocked_steps = 0

    def step(
        self, forward_requested: bool, front_scale: float, ranges, self_mask: list[bool] | None,
        horizontal_resolution: int, field_of_view_deg: float = 360.0,
    ) -> tuple[float, float] | None:
        self.just_started = None
        if self.mode is not None:
            center, _, abort_m = _YIELD_SECTORS[self.mode]
            rng = _sector_min_range(
                ranges, self_mask, center, YIELD_SECTOR_HALF_WIDTH_DEG, horizontal_resolution, field_of_view_deg
            )
            self.steps_left -= 1
            if (
                (forward_requested and front_scale >= YIELD_RELEASE_SCALE)
                or rng < abort_m
                or self.steps_left <= 0
            ):
                self._finish()
                return None
            return _yield_body_velocity(self.mode)

        if self.cooldown_steps > 0:
            self.cooldown_steps -= 1
            return None
        if not (forward_requested and front_scale <= YIELD_TRIGGER_SCALE):
            self.blocked_steps = 0
            return None
        self.blocked_steps += 1
        direction = _lidar_yield_direction(ranges, self_mask, horizontal_resolution, field_of_view_deg)
        if direction is None:
            return None
        if direction == "back" and self.blocked_steps < self.back_delay_steps:
            return None
        self.mode = direction
        self.steps_left = self.duration_steps
        self.just_started = direction
        return _yield_body_velocity(direction)


# Arm geometry (joint anchors) for forward kinematics.
_ARM_BASE_OFFSET_M = (0.156, 0.0, 0.0)
_ARM1_ANCHOR_M = (0.0, 0.0, 0.077)
_ARM2_ANCHOR_M = (0.033, 0.0, 0.07)
_ARM3_ANCHOR_M = (0.0, 0.0, 0.155)
_ARM4_ANCHOR_M = (0.0, 0.0, 0.135)
_ARM5_ANCHOR_M = (0.0, 0.0, 0.081)
_GRIPPER_MOUNT_OFFSET_M = (0.0, 0.0, 0.16)


def _vec_add(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _mat_identity() -> tuple:
    return ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


def _mat_rot_z(theta: float) -> tuple:
    c, s = math.cos(theta), math.sin(theta)
    return ((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0))


def _mat_rot_y(theta: float) -> tuple:
    c, s = math.cos(theta), math.sin(theta)
    return ((c, 0.0, s), (0.0, 1.0, 0.0), (-s, 0.0, c))


def _mat_mul(a: tuple, b: tuple) -> tuple:
    return tuple(
        tuple(sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3))
        for i in range(3)
    )


def _mat_vec(a: tuple, v: tuple[float, float, float]) -> tuple[float, float, float]:
    return tuple(sum(a[i][k] * v[k] for k in range(3)) for i in range(3))


def _mat_to_axis_angle(r: tuple) -> tuple[float, float, float, float]:
    """3x3 rotation matrix -> Webots axis-angle rotation."""
    trace = r[0][0] + r[1][1] + r[2][2]
    cos_angle = max(-1.0, min(1.0, (trace - 1.0) / 2.0))
    angle = math.acos(cos_angle)
    if angle < 1e-6:
        return (1.0, 0.0, 0.0, 0.0)
    if angle > math.pi - 1e-6:
        axis = (
            math.sqrt(max(0.0, (r[0][0] + 1) / 2)),
            math.sqrt(max(0.0, (r[1][1] + 1) / 2)),
            math.sqrt(max(0.0, (r[2][2] + 1) / 2)),
        )
        norm = math.sqrt(sum(c * c for c in axis))
        if norm < 1e-9:
            return (1.0, 0.0, 0.0, angle)
        return (axis[0] / norm, axis[1] / norm, axis[2] / norm, angle)
    sin_angle = math.sin(angle)
    return (
        (r[2][1] - r[1][2]) / (2 * sin_angle),
        (r[0][2] - r[2][0]) / (2 * sin_angle),
        (r[1][0] - r[0][1]) / (2 * sin_angle),
        angle,
    )


def _gripper_chassis_local_pose(angles: dict[str, float]) -> tuple[tuple[float, float, float], tuple]:
    """Arm forward kinematics: gripper position and rotation in the chassis frame."""
    pos = _ARM_BASE_OFFSET_M
    rot = _mat_identity()

    pos = _vec_add(pos, _mat_vec(rot, _ARM1_ANCHOR_M))
    rot = _mat_mul(rot, _mat_rot_z(angles["arm1"]))

    pos = _vec_add(pos, _mat_vec(rot, _ARM2_ANCHOR_M))
    rot = _mat_mul(rot, _mat_rot_y(-angles["arm2"]))

    pos = _vec_add(pos, _mat_vec(rot, _ARM3_ANCHOR_M))
    rot = _mat_mul(rot, _mat_rot_y(-angles["arm3"]))

    pos = _vec_add(pos, _mat_vec(rot, _ARM4_ANCHOR_M))
    rot = _mat_mul(rot, _mat_rot_y(-angles["arm4"]))

    pos = _vec_add(pos, _mat_vec(rot, _ARM5_ANCHOR_M))
    rot = _mat_mul(rot, _mat_rot_z(angles["arm5"]))

    pos = _vec_add(pos, _mat_vec(rot, _GRIPPER_MOUNT_OFFSET_M))
    return pos, rot


_CAMERA_LOOK_ALIGNMENT = _mat_rot_y(-math.pi / 2)


def _gripper_camera_pose(angles: dict[str, float]) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    """Gripper camera translation and rotation in the chassis frame."""
    pos, rot = _gripper_chassis_local_pose(angles)
    camera_rot = _mat_mul(rot, _CAMERA_LOOK_ALIGNMENT)
    return pos, _mat_to_axis_angle(camera_rot)


def _camera_point_to_world(
    fx: float, fy: float, fz: float, camera_chassis_pos: tuple[float, float, float], camera_rot: tuple,
    robot_x: float, robot_y: float, robot_theta: float,
) -> tuple[float, float]:
    """Point in the camera frame -> world (x, y)."""
    chassis_point = _vec_add(camera_chassis_pos, _mat_vec(camera_rot, (fx, fy, fz)))
    cos_t, sin_t = math.cos(robot_theta), math.sin(robot_theta)
    world_x = robot_x + chassis_point[0] * cos_t - chassis_point[1] * sin_t
    world_y = robot_y + chassis_point[0] * sin_t + chassis_point[1] * cos_t
    return world_x, world_y


# Arm and gripper devices, limits and poses.
_ARM_MOTOR_NAMES = ["arm1", "arm2", "arm3", "arm4", "arm5"]
_ARM_JOINT_LIMITS_RAD = {
    "arm1": (-2.99, 2.9496),
    "arm2": (-1.13446, 1.5708),
    "arm3": (-2.63545, 2.54818),
    "arm4": (-1.78024, 1.78024),
    "arm5": (-2.92343, 2.92343),
}
_ARM_MAX_VELOCITY_RAD_S = 1.5708
_FINGER_MOTOR_NAMES = ["finger::left", "finger::right"]
_FINGER_OPEN_M = 0.025
_FINGER_CLOSED_M = 0.0
_FINGER_MAX_VELOCITY_M_S = 0.01

_ARM_POSE_REST_RAD = {"arm1": 0.0, "arm2": 1.4, "arm3": -2.5, "arm4": 1.6, "arm5": 0.0}
_ARM_POSE_CARRY_RAD = {
    "arm1": 0.24, "arm2": -1.13446, "arm3": -0.9, "arm4": -1.1098, "arm5": 0.25,
}
_CARRY_GRIPPER_OPEN = True
_ARM_POSE_LIFTED_RAD = {"arm1": -0.1004, "arm2": 0.6708, "arm3": 0.6, "arm4": 1.6302, "arm5": 0.0}

_ARM_POSE_TOLERANCE_RAD = 0.03
_FINGER_TOLERANCE_M = 0.003
_POSE_TIMEOUT_STEPS = 150
GRAB_DELAY_SECONDS = 3.0

# Height of a box resting on the floor.
_PACKAGE_REST_Z_M = 0.032
_PACKAGE_DROP_GAP_M = 0.01

_WHEEL_MOTOR_CANDIDATES = [
    ["wheel1", "front right wheel", "fr_wheel", "wheel_front_right"],
    ["wheel2", "front left wheel", "fl_wheel", "wheel_front_left"],
    ["wheel3", "back right wheel", "br_wheel", "wheel_back_right", "rear right wheel"],
    ["wheel4", "back left wheel", "bl_wheel", "wheel_back_left", "rear left wheel"],
]
_WHEEL_NAMES = ["front-right", "front-left", "back-right", "back-left"]


def _get_device_any(robot: Robot, candidates: list[str]):
    for name in candidates:
        device = robot.getDevice(name)
        if device is not None:
            return device, name
    return None, None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot-id", type=int, required=True)
    parser.add_argument("--start-row", type=int, required=True)
    parser.add_argument("--start-col", type=int, required=True)
    args, _unknown = parser.parse_known_args(sys.argv[1:])
    return args


def _cell_to_world(cell: tuple[int, int]) -> tuple[float, float]:
    row, col = cell
    return (col + 0.5) * 1.0, -(row + 0.5) * 1.0


def _inverse_kinematics(vx: float, vy: float, omega: float) -> list[float]:
    """Body (vx, vy, omega) -> wheel speeds [FR, FL, BR, BL]."""
    r = _WHEEL_RADIUS_M
    return [
        (vx + vy + _K * omega) / r,
        (vx - vy - _K * omega) / r,
        (vx - vy + _K * omega) / r,
        (vx + vy - _K * omega) / r,
    ]


# Max growth of the linear speed per step.
MAX_VELOCITY_RAMP_MPS_PER_STEP = 0.04
# Max change of omega per step.
MAX_OMEGA_RAMP_RADPS_PER_STEP = 0.25


def _rate_limit_velocity_2d(
    vx: float, vy: float, prev_vx: float, prev_vy: float, max_step_mps: float
) -> tuple[float, float]:
    """Limit how fast the (vx, vy) speed can grow per step (braking is instant)."""
    speed = math.hypot(vx, vy)
    prev_speed = math.hypot(prev_vx, prev_vy)
    if speed <= prev_speed:
        return vx, vy
    dvx, dvy = vx - prev_vx, vy - prev_vy
    step = math.hypot(dvx, dvy)
    if step <= max_step_mps:
        return vx, vy
    scale = max_step_mps / step
    return prev_vx + dvx * scale, prev_vy + dvy * scale


def _rate_limit_scalar(value: float, prev_value: float, max_step: float) -> float:
    """Limit the change of a value per step in both directions."""
    delta = value - prev_value
    if abs(delta) <= max_step:
        return value
    return prev_value + math.copysign(max_step, delta)


_WHEEL_SCALE_SAFETY_MARGIN = 0.999


def _scale_wheel_speeds_to_limit(wheel_speeds: list[float], max_velocity: float) -> list[float]:
    """Scale all wheel speeds down together if one exceeds the limit."""
    peak = max(abs(s) for s in wheel_speeds)
    if peak <= max_velocity or peak == 0.0:
        return wheel_speeds
    scale = (max_velocity * _WHEEL_SCALE_SAFETY_MARGIN) / peak
    return [s * scale for s in wheel_speeds]


def _world_to_body(vx_world: float, vy_world: float, theta: float) -> tuple[float, float]:
    """Rotate a world-frame velocity into the body frame."""
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    vx_body = vx_world * cos_t + vy_world * sin_t
    vy_body = -vx_world * sin_t + vy_world * cos_t
    return vx_body, vy_body


def _forward_kinematics(wheel_deltas: list[float]) -> tuple[float, float, float]:
    """Wheel distance deltas [FR, FL, BR, BL] -> body (dx, dy, dtheta)."""
    w1, w2, w3, w4 = wheel_deltas
    dx = (w1 + w2 + w3 + w4) / 4.0
    dy = (w1 - w2 - w3 + w4) / 4.0
    dtheta = (w1 - w2 + w3 - w4) / (4.0 * _K)
    return dx, dy, dtheta


def main() -> None:
    """Connect to Webots and ROS2, then run the control loop."""
    from controller import Supervisor

    args = _parse_args()
    robot_id = args.robot_id

    robot = Supervisor()
    self_node = robot.getSelf()
    if self_node is None:
        print(
            f"[youbot_mecanum robot_{robot_id}] WARNING: getSelf() returned None -- falling back "
            f"to wheel-encoder dead reckoning for x/y, which drifts over a long drive.",
            flush=True,
        )

    motors = []
    for candidates, label in zip(_WHEEL_MOTOR_CANDIDATES, _WHEEL_NAMES):
        motor, matched = _get_device_any(robot, candidates)
        if motor is None:
            print(
                f"[youbot_mecanum robot_{robot_id}] FATAL: no wheel motor device found for the "
                f"{label} wheel. Tried: {candidates}. Open this robot in Webots' scene tree, find "
                f"the real device name, and add it to _WHEEL_MOTOR_CANDIDATES in "
                f"webots/controllers/youbot_mecanum/youbot_mecanum.py.",
                flush=True,
            )
            raise RuntimeError(f"no wheel motor device found for {label} wheel")
        motor.setPosition(float("inf"))
        motor.setVelocity(0.0)
        motors.append(motor)
    wheel_max_velocity = min(m.getMaxVelocity() for m in motors)

    sensors = []
    encoders_available = True
    for candidates, label in zip(_WHEEL_MOTOR_CANDIDATES, _WHEEL_NAMES):
        sensor_candidates = [f"{c}sensor" for c in candidates] + [f"{c} sensor" for c in candidates]
        sensor, matched = _get_device_any(robot, sensor_candidates)
        if sensor is None:
            encoders_available = False
            print(
                f"[youbot_mecanum robot_{robot_id}] WARNING: no encoder found for the {label} "
                f"wheel (tried: {sensor_candidates}). Falling back to open-loop odometry "
                f"(integrating commanded velocity, no slip compensation) for all wheels.",
                flush=True,
            )
        else:
            sensor.enable(TIME_STEP_MS)
        sensors.append(sensor)
    if not encoders_available:
        sensors = [None, None, None, None]

    start_x, start_y = _cell_to_world((args.start_row, args.start_col))
    pose = {"x": start_x, "y": start_y, "theta": 0.0}
    last_wheel_positions = [None, None, None, None]

    led_to_item = robot.getDevice("led_to_item")
    led_carrying = robot.getDevice("led_carrying")
    status = [0]

    lidar = robot.getDevice("lidar")
    if lidar is not None:
        lidar.enable(TIME_STEP_MS)
    else:
        print(
            f"[youbot_mecanum robot_{robot_id}] WARNING: no 'lidar' device found -- this robot has "
            f"no local proximity safety brake during transit (world file may be stale, regenerate "
            f"it).",
            flush=True,
        )

    pickup_camera = robot.getDevice("pickup_camera")
    pickup_camera_node = robot.getFromDef(f"PICKUP_CAMERA_{robot_id}")
    pickup_camera_translation_field = None
    pickup_camera_rotation_field = None
    if pickup_camera is not None:
        pickup_camera.enable(TIME_STEP_MS)
        pickup_camera.recognitionEnable(TIME_STEP_MS)
    else:
        print(
            f"[youbot_mecanum robot_{robot_id}] WARNING: no 'pickup_camera' device found -- "
            f"falling back to the assumed cell-center box position (world file may be stale, "
            f"regenerate it).",
            flush=True,
        )
    if pickup_camera_node is not None:
        pickup_camera_translation_field = pickup_camera_node.getField("translation")
        pickup_camera_rotation_field = pickup_camera_node.getField("rotation")
    elif pickup_camera is not None:
        print(
            f"[youbot_mecanum robot_{robot_id}] WARNING: no 'PICKUP_CAMERA_{robot_id}' DEF node "
            f"found -- the camera device exists but this controller can't move it to track the "
            f"gripper (world file may be stale, regenerate it).",
            flush=True,
        )

    def _apply_led_pattern() -> None:
        if led_to_item is not None:
            led_to_item.set(1 if status[0] == 1 else 0)
        if led_carrying is not None:
            led_carrying.set(1 if status[0] == 3 else 0)

    arm_motors = {}
    for name in _ARM_MOTOR_NAMES:
        motor = robot.getDevice(name)
        if motor is None:
            print(
                f"[youbot_mecanum robot_{robot_id}] WARNING: arm motor '{name}' not found -- "
                f"arm/gripper animation disabled for this robot (driving is unaffected).",
                flush=True,
            )
            arm_motors = {}
            break
        motor.setVelocity(_ARM_MAX_VELOCITY_RAD_S)
        arm_motors[name] = motor

    finger_motors = {}
    if arm_motors:
        for name in _FINGER_MOTOR_NAMES:
            motor = robot.getDevice(name)
            if motor is None:
                print(
                    f"[youbot_mecanum robot_{robot_id}] WARNING: gripper motor '{name}' not "
                    f"found -- arm/gripper animation disabled for this robot.",
                    flush=True,
                )
                arm_motors, finger_motors = {}, {}
                break
            motor.setVelocity(_FINGER_MAX_VELOCITY_M_S)
            finger_motors[name] = motor

    arm_sensors = {}
    for name in arm_motors:
        sensor = robot.getDevice(f"{name}sensor")
        if sensor is not None:
            sensor.enable(TIME_STEP_MS)
            arm_sensors[name] = sensor
    if len(arm_sensors) != len(arm_motors):
        print(
            f"[youbot_mecanum robot_{robot_id}] WARNING: not all arm position sensors found "
            f"({len(arm_sensors)}/{len(arm_motors)}) -- pickup animation stages will fall back "
            f"to the _POSE_TIMEOUT_STEPS timeout instead of confirming arrival.",
            flush=True,
        )
        arm_sensors = {}

    finger_sensors = {}
    for name in finger_motors:
        sensor = robot.getDevice(f"{name}sensor")
        if sensor is not None:
            sensor.enable(TIME_STEP_MS)
            finger_sensors[name] = sensor
    if len(finger_sensors) != len(finger_motors):
        finger_sensors = {}

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

    last_commanded_arm_pose = [dict(_ARM_POSE_REST_RAD)]

    def _apply_arm_pose(pose_rad: dict[str, float], gripper_open: bool) -> None:
        for name, motor in arm_motors.items():
            motor.setPosition(pose_rad[name])
        target = _FINGER_OPEN_M if gripper_open else _FINGER_CLOSED_M
        for motor in finger_motors.values():
            motor.setPosition(target)
        last_commanded_arm_pose[0] = dict(pose_rad)

    if arm_motors:
        _apply_arm_pose(_ARM_POSE_REST_RAD, gripper_open=True)

    def _current_arm_angles() -> dict[str, float]:
        """Current arm joint angles (position sensors, else last commanded)."""
        if arm_sensors:
            return {name: arm_sensors[name].getValue() for name in _ARM_MOTOR_NAMES}
        return dict(last_commanded_arm_pose[0])

    def _locate_box_via_camera(
        expected_node, robot_pose: dict,
        camera_chassis_pos: tuple[float, float, float], camera_rot: tuple,
    ) -> tuple[float, float] | None:
        """World (x, y) of the pickup box seen by the gripper camera, or None."""
        if pickup_camera is None or expected_node is None:
            return None
        expected_id = expected_node.getId()
        for obj in pickup_camera.getRecognitionObjects():
            if obj.getId() != expected_id:
                continue
            fx, fy, fz = obj.getPosition()
            return _camera_point_to_world(
                fx, fy, fz, camera_chassis_pos, camera_rot,
                robot_pose["x"], robot_pose["y"], robot_pose["theta"],
            )
        return None

    goal_cell = [args.start_row, args.start_col]

    carried_package_node = [None]
    carried_package_home = [None]
    pending_pickup_cell = [None]
    pending_dropoff_cell = [None]
    expected_pickup_node = [None]
    sensed_box_xy = [None]
    pickup_face_theta = [PICKUP_FACE_THETA_RAD]
    expected_theta = [0.0]
    pickup_phase = [None]
    pickup_phase_step = [0]
    reach_confirmed_at = [None]
    last_pickup_completed_cell = [None]

    import rclpy
    from geometry_msgs.msg import Pose2D, Twist
    from std_msgs.msg import Bool, Int32MultiArray

    rclpy.init(args=None)
    node = rclpy.create_node(f"robot_{robot_id}_controller")
    topic_prefix = f"/hiverl/robot_{robot_id}"

    def _on_command(msg: Int32MultiArray) -> None:
        if len(msg.data) < 3:
            return
        if int(msg.data[2]) == -1:
            row, col = int(msg.data[0]), int(msg.data[1])
            x, y = _cell_to_world((row, col))
            goal_cell[0], goal_cell[1] = row, col
            if self_node is not None:
                _old_x, _old_y, z = self_node.getField("translation").getSFVec3f()
                self_node.getField("translation").setSFVec3f([x, y, z])
                self_node.getField("rotation").setSFRotation([0.0, 0.0, 1.0, 0.0])
                self_node.resetPhysics()
            pose["x"], pose["y"], pose["theta"] = x, y, 0.0
            for motor in motors:
                motor.setVelocity(0.0)
            prev_vx_cmd[0] = prev_vy_cmd[0] = prev_omega_cmd[0] = 0.0
            pickup_phase[0] = None
            pending_pickup_cell[0] = None
            pending_dropoff_cell[0] = None
            last_pickup_completed_cell[0] = None
            if carried_package_node[0] is not None and carried_package_home[0] is not None:
                hx, hy = _cell_to_world(carried_package_home[0])
                carried_package_node[0].getField("translation").setSFVec3f(
                    [hx, hy, _PACKAGE_REST_Z_M + _PACKAGE_DROP_GAP_M]
                )
                carried_package_node[0].resetPhysics()
            carried_package_node[0] = None
            carried_package_home[0] = None
            if arm_motors:
                _apply_arm_pose(_ARM_POSE_REST_RAD, gripper_open=True)
            status[0] = 0
            _apply_led_pattern()
            print(
                f"[youbot_mecanum robot_{robot_id}] RESET ACK: teleported to cell "
                f"({row}, {col}) at ({x:.2f}, {y:.2f}).",
                flush=True,
            )
            return
        goal_cell[0], goal_cell[1] = int(msg.data[0]), int(msg.data[1])
        previous_status = status[0]
        status[0] = int(msg.data[2])
        _apply_led_pattern()
        if previous_status != 2 and status[0] == 2:
            pending_pickup_cell[0] = tuple(goal_cell)
            pickup_phase[0] = "approach"
            last_pickup_completed_cell[0] = None
            row, col = pending_pickup_cell[0]
            expected_pickup_node[0] = robot.getFromDef(f"PACKAGE_{row}_{col}")
            sensed_box_xy[0] = None
            pickup_face_theta[0] = PICKUP_FACE_THETA_RAD
            print(
                f"[youbot_mecanum robot_{robot_id}] PICKUP TRIGGERED: entering 'approach' phase "
                f"for cell ({row}, {col}), current pose=({pose['x']:.3f}, {pose['y']:.3f}, "
                f"{pose['theta']:.3f}), package_node found={expected_pickup_node[0] is not None}.",
                flush=True,
            )
        elif previous_status == 3 and status[0] != 3:
            pending_dropoff_cell[0] = tuple(goal_cell)
        if status[0] in (0, 1) and pickup_phase[0] is not None:
            print(
                f"[youbot_mecanum robot_{robot_id}] PICKUP CANCELLED: status={status[0]} while "
                f"phase '{pickup_phase[0]}' for cell {pending_pickup_cell[0]} was still pending.",
                flush=True,
            )
            pickup_phase[0] = None
            pending_pickup_cell[0] = None
            if arm_motors:
                _apply_arm_pose(_ARM_POSE_REST_RAD, gripper_open=True)

    node.create_subscription(Int32MultiArray, f"{topic_prefix}/command", _on_command, 10)
    arrived_pub = node.create_publisher(Bool, f"{topic_prefix}/arrived", 10)
    pose_pub = node.create_publisher(Pose2D, f"{topic_prefix}/pose", 10)

    cmd_vel_v = [0.0]
    cmd_vel_omega = [0.0]
    # linear.z = 1: the trainer's shield stopped this robot while it wanted to go forward.
    cmd_vel_shield_blocked = [False]
    cmd_vel_last_time = [-1e9]
    continuous_diag_countdown = [0]

    def _on_cmd_vel(msg: Twist) -> None:
        cmd_vel_v[0] = float(msg.linear.x)
        cmd_vel_omega[0] = float(msg.angular.z)
        cmd_vel_shield_blocked[0] = float(msg.linear.z) > 0.5
        cmd_vel_last_time[0] = time.time()

    node.create_subscription(Twist, f"{topic_prefix}/cmd_vel", _on_cmd_vel, 10)

    gripper_camera_print_countdown = [0]
    lidar_diag_countdown = [0]
    lidar_stall_steps = [0]
    lidar_creep_remaining = [0]
    lidar_avoidance_prev_range = [float("inf")]
    lidar_avoidance_probe_sign = [1]
    lidar_avoidance_worsening_steps = [0]
    position_stall_ref = [None]
    position_stall_steps = [0]
    hard_stuck_steps = [0]
    prev_vx_cmd = [0.0]
    prev_vy_cmd = [0.0]
    prev_omega_cmd = [0.0]
    lidar_self_mask = [None]
    lidar_self_mask_accum = [None]
    lidar_calibration_step = [0]
    lidar_arm_settled_at = [None]
    lidar_carry_mask_accum = [None]
    lidar_carry_mask_done = [False]
    lidar_yield = _LidarYieldController(robot_id, TIME_STEP_S)

    # Control loop: one physics step per iteration.
    while robot.step(TIME_STEP_MS) != -1:
        rclpy.spin_once(node, timeout_sec=0.0)

        # LiDAR self-mask calibration right after startup.
        if lidar is not None and lidar_self_mask[0] is None:
            lidar_calibration_step[0] += 1
            if lidar_arm_settled_at[0] is None:
                if (
                    not arm_motors
                    or _arm_at_pose(_ARM_POSE_REST_RAD)
                    or lidar_calibration_step[0] >= _POSE_TIMEOUT_STEPS
                ):
                    lidar_arm_settled_at[0] = lidar_calibration_step[0]
            else:
                ranges = lidar.getRangeImage()
                
                frame_mask = _lidar_calibrate_self_mask(ranges, LIDAR_SELF_STARTUP_MAX_M)
                if lidar_self_mask_accum[0] is None:
                    lidar_self_mask_accum[0] = frame_mask
                else:
                    lidar_self_mask_accum[0] = [
                        a or b for a, b in zip(lidar_self_mask_accum[0], frame_mask)
                    ]
                if (
                    lidar_calibration_step[0] - lidar_arm_settled_at[0]
                    >= LIDAR_SELF_CALIBRATION_WINDOW_STEPS
                ):
                    lidar_self_mask[0] = lidar_self_mask_accum[0]
                    num_excluded = sum(lidar_self_mask[0])
                    print(  
                        f"[youbot_mecanum robot_{robot_id}] LiDAR self-detection calibrated: "
                        f"{num_excluded}/{len(ranges)} rays excluded as permanent self-hits "
                        f"(closer than {LIDAR_SELF_STARTUP_MAX_M}m, "
                        f"arm settled at rest after {lidar_arm_settled_at[0]} steps).",
                        flush=True,
                    )

        gripper_pos, gripper_rot = _gripper_chassis_local_pose(_current_arm_angles())
        # Keep the gripper camera attached to the arm.
        camera_rot = _mat_mul(gripper_rot, _CAMERA_LOOK_ALIGNMENT)
        if pickup_camera_translation_field is not None:
            pickup_camera_translation_field.setSFVec3f(list(gripper_pos))
        if pickup_camera_rotation_field is not None:
            pickup_camera_rotation_field.setSFRotation(list(_mat_to_axis_angle(camera_rot)))

        target_x, target_y = _cell_to_world(tuple(goal_cell))
        dx, dy = target_x - pose["x"], target_y - pose["y"]
        dist = math.hypot(dx, dy)

        arrived = dist <= ARRIVAL_TOLERANCE_M

        continuous_cmd = None
        yield_cmd = None
        # RL control (TO_ITEM/CARRYING): cmd_vel with the directional LiDAR brake and the yield
        # nudge.
        if status[0] in (1, 3) and (time.time() - cmd_vel_last_time[0]) <= CMD_VEL_STALE_S:
            vx_body = cmd_vel_v[0]
            omega_body = -cmd_vel_omega[0]
            if lidar is not None:
                ranges = lidar.getRangeImage()
                scan_mask = _lidar_scan_mask(ranges, lidar_self_mask[0])
                lidar_range = _lidar_min_range(ranges, scan_mask, LIDAR_SELF_DYNAMIC_FLOOR_M)
                # Directional brake.
                travel_center_deg = 0.0 if vx_body >= 0.0 else 180.0
                ahead_range = _sector_min_range(
                    ranges, scan_mask, travel_center_deg, LIDAR_DIRECTIONAL_HALF_WIDTH_DEG,
                    lidar.getHorizontalResolution(), math.degrees(lidar.getFov()),
                )
                if ahead_range == float("inf"):
                    ahead_range = lidar_range
                safety_scale = _lidar_safety_scale(max(ahead_range, LIDAR_SELF_DYNAMIC_FLOOR_M))
                if lidar_range <= LIDAR_CONTACT_STOP_M:
                    safety_scale = 0.0
                floor = LIDAR_REVERSE_SAFETY_FLOOR if vx_body < 0.0 else 0.0
                requested_vx = vx_body
                vx_body *= max(floor, safety_scale)
                # Yield nudge when the front is blocked.
                if LIDAR_YIELD_ENABLED and lidar_self_mask[0] is not None:
                    # A shield stop counts as a blocked forward command, whatever the LiDAR reads.
                    shield_blocked = cmd_vel_shield_blocked[0]
                    yield_cmd = lidar_yield.step(
                        requested_vx >= YIELD_MIN_FORWARD_MPS or shield_blocked,
                        0.0 if shield_blocked else safety_scale, ranges, scan_mask,
                        lidar.getHorizontalResolution(), math.degrees(lidar.getFov()),
                    )
                    if lidar_yield.just_started is not None:
                        print(
                            f"[youbot_mecanum robot_{robot_id}] LIDAR YIELD: "
                            f"{'shield stop' if shield_blocked else 'front blocked'} "
                            f"(ahead={ahead_range:.2f}m) -- nudging {lidar_yield.just_started} at "
                            f"{YIELD_SPEED_MPS} m/s for up to {YIELD_DURATION_S}s.",
                            flush=True,
                        )
            if yield_cmd is not None:
                continuous_cmd = (yield_cmd[0], yield_cmd[1], 0.0)
            else:
                continuous_cmd = (vx_body, 0.0, omega_body)
        # Stale cmd_vel: hold still.
        elif status[0] in (1, 3):
            continuous_cmd = (0.0, 0.0, 0.0)
        if continuous_cmd is None or (yield_cmd is None and lidar_yield.mode is not None):
            lidar_yield.reset()

        # Drop the carried box at the dropoff (or when a new task starts).
        if pending_dropoff_cell[0] is not None:
            dxp, dyp = _cell_to_world(pending_dropoff_cell[0])
            if (
                math.hypot(dxp - pose["x"], dyp - pose["y"]) <= ARRIVAL_TOLERANCE_M
                or status[0] in (1, 2)
            ):
                if arm_motors:
                    _apply_arm_pose(_ARM_POSE_REST_RAD, gripper_open=True)
                if carried_package_node[0] is not None:
                    carried_package_node[0].getField("translation").setSFVec3f(
                        [dxp, dyp, _PACKAGE_REST_Z_M + _PACKAGE_DROP_GAP_M]
                    )
                    carried_package_node[0].resetPhysics()
                    carried_package_node[0] = None
                pending_dropoff_cell[0] = None

        if (
            lidar is not None
            and lidar_self_mask[0] is not None
            and not lidar_carry_mask_done[0]
            and pickup_phase[0] in ("close_gripper", "lift")
        ):
            frame_mask = _lidar_calibrate_self_mask(lidar.getRangeImage())
            if lidar_carry_mask_accum[0] is None:
                lidar_carry_mask_accum[0] = frame_mask
            else:
                lidar_carry_mask_accum[0] = [
                    a or b for a, b in zip(lidar_carry_mask_accum[0], frame_mask)
                ]

        pickup_vx_cmd = pickup_vy_cmd = pickup_omega_cmd = None
        # Pickup sequence: approach, rotate, reach, close gripper, lift, reorient.
        if pickup_phase[0] == "approach":
            detected = _locate_box_via_camera(expected_pickup_node[0], pose, gripper_pos, camera_rot)
            if detected is not None:
                sensed_box_xy[0] = detected
            box_x, box_y = sensed_box_xy[0] if sensed_box_xy[0] is not None else _cell_to_world(pending_pickup_cell[0])
            stop_x, stop_y = box_x + PICKUP_STOP_OFFSET_M, box_y
            dxs, dys = stop_x - pose["x"], stop_y - pose["y"]
            dists = math.hypot(dxs, dys)
            if dists <= PICKUP_STOP_TOLERANCE_M:
                pickup_vx_cmd, pickup_vy_cmd, pickup_omega_cmd = 0.0, 0.0, 0.0
                pickup_phase[0] = "rotate"
                pickup_phase_step[0] = 0
                if sensed_box_xy[0] is not None:
                    pickup_face_theta[0] = math.atan2(box_y - pose["y"], box_x - pose["x"])
                else:
                    pickup_face_theta[0] = PICKUP_FACE_THETA_RAD
                print(
                    f"[youbot_mecanum robot_{robot_id}] pickup: stopped short of box at "
                    f"({pose['x']:.4f}, {pose['y']:.4f}), sensed box at ({box_x:.4f}, "
                    f"{box_y:.4f}) [camera={'yes' if sensed_box_xy[0] is not None else 'no, cell-center fallback'}] "
                    f"-- now rotating to face it (target theta={pickup_face_theta[0]:.4f}).",
                    flush=True,
                )
            else:
                speed = min(MAX_BODY_SPEED_MPS, max(PICKUP_APPROACH_MIN_SPEED_MPS, KP_LINEAR * dists))
                vx_world = speed * dxs / dists
                vy_world = speed * dys / dists
                pickup_vx_cmd, pickup_vy_cmd = _world_to_body(vx_world, vy_world, pose["theta"])
                pickup_omega_cmd = 0.0
        elif pickup_phase[0] == "rotate":
            pickup_vx_cmd, pickup_vy_cmd = 0.0, 0.0
            pickup_phase_step[0] += 1
            angle_diff = _wrap_angle(pickup_face_theta[0] - pose["theta"])
            if abs(angle_diff) <= ANGLE_TOLERANCE_RAD or pickup_phase_step[0] >= _POSE_TIMEOUT_STEPS:
                pickup_omega_cmd = 0.0
                row, col = pending_pickup_cell[0]
                px, py = _cell_to_world((row, col))
                print(
                    f"[youbot_mecanum robot_{robot_id}] pickup: facing box at "
                    f"({px:.4f}, {py:.4f}) -- reaching.",
                    flush=True,
                )
                if arm_motors:
                    _apply_arm_pose(_ARM_POSE_CARRY_RAD, gripper_open=_CARRY_GRIPPER_OPEN)
                expected_theta[0] = pickup_face_theta[0]
                pickup_phase[0] = "reach"
                pickup_phase_step[0] = 0
            else:
                pickup_omega_cmd = max(
                    -MAX_ANGULAR_SPEED_RAD_S, min(MAX_ANGULAR_SPEED_RAD_S, KP_ANGULAR * angle_diff)
                )
        elif pickup_phase[0] == "reach":
            pickup_vx_cmd, pickup_vy_cmd, pickup_omega_cmd = 0.0, 0.0, 0.0
            pickup_phase_step[0] += 1
            if reach_confirmed_at[0] is None:
                if _arm_at_pose(_ARM_POSE_CARRY_RAD) or pickup_phase_step[0] >= _POSE_TIMEOUT_STEPS:
                    reach_confirmed_at[0] = robot.getTime()
                    print(
                        f"[youbot_mecanum robot_{robot_id}] pickup: reached CARRY pose, waiting "
                        f"{GRAB_DELAY_SECONDS:.1f}s before closing gripper.",
                        flush=True,
                    )
            elif robot.getTime() - reach_confirmed_at[0] >= GRAB_DELAY_SECONDS:
                print(f"[youbot_mecanum robot_{robot_id}] pickup: closing gripper.", flush=True)
                if arm_motors:
                    _apply_arm_pose(_ARM_POSE_CARRY_RAD, gripper_open=False)
                pickup_phase[0] = "close_gripper"
                pickup_phase_step[0] = 0
                reach_confirmed_at[0] = None
        elif pickup_phase[0] == "close_gripper":
            pickup_vx_cmd, pickup_vy_cmd, pickup_omega_cmd = 0.0, 0.0, 0.0
            pickup_phase_step[0] += 1
            if _fingers_at(_FINGER_CLOSED_M) or pickup_phase_step[0] >= _POSE_TIMEOUT_STEPS:
                print(f"[youbot_mecanum robot_{robot_id}] pickup: lifting.", flush=True)
                if arm_motors:
                    _apply_arm_pose(_ARM_POSE_LIFTED_RAD, gripper_open=False)
                pickup_phase[0] = "lift"
                pickup_phase_step[0] = 0
        elif pickup_phase[0] == "lift":
            pickup_vx_cmd, pickup_vy_cmd, pickup_omega_cmd = 0.0, 0.0, 0.0
            pickup_phase_step[0] += 1
            if _arm_at_pose(_ARM_POSE_LIFTED_RAD) or pickup_phase_step[0] >= _POSE_TIMEOUT_STEPS:
                row, col = pending_pickup_cell[0]
                def_name = f"PACKAGE_{row}_{col}"
                package_node = robot.getFromDef(def_name)
                print(
                    f"[youbot_mecanum robot_{robot_id}] PICKUP COMPLETE\n"
                    f"  pickup cell (row, col)            = ({row}, {col})\n"
                    f"  robot pose x, y, theta(rad)       = "
                    f"({pose['x']:.4f}, {pose['y']:.4f}, {pose['theta']:.4f})\n"
                    f"  status code (0=IDLE/1=TO_ITEM/2=PICKING_UP/3=CARRYING) = {status[0]}\n"
                    f"  package_node found                = {package_node is not None}",
                    flush=True,
                )
                if package_node is not None:
                    carried_package_node[0] = package_node
                    carried_package_home[0] = (row, col)
                    package_node.resetPhysics()
                else:
                    print(
                        f"[youbot_mecanum robot_{robot_id}] WARNING: no PACKAGE_{row}_{col} node "
                        f"found at pickup -- world file may be stale, regenerate it.",
                        flush=True,
                    )
                if arm_motors:
                    _apply_arm_pose(_ARM_POSE_REST_RAD, gripper_open=True)
                last_pickup_completed_cell[0] = (row, col)
                pending_pickup_cell[0] = None
                pickup_phase[0] = "reorient"
                pickup_phase_step[0] = 0
                if not lidar_carry_mask_done[0] and lidar_carry_mask_accum[0] is not None:
                    lidar_self_mask[0] = [
                        a or b for a, b in zip(lidar_self_mask[0], lidar_carry_mask_accum[0])
                    ]
                    lidar_carry_mask_done[0] = True
                    print(
                        f"[youbot_mecanum robot_{robot_id}] LiDAR CARRY-pose self-detection "
                        f"calibrated: {sum(lidar_carry_mask_accum[0])} rays flagged as self-hits "
                        f"while arm-raised (union'd into the existing self-mask). See "
                        f"lidar_carry_mask_accum's comment above -- this is what prevents the "
                        f"raised arm/package from being mistaken for a nearby obstacle for the "
                        f"rest of this robot's CARRYING trips.",
                        flush=True,
                    )
        elif pickup_phase[0] == "reorient":
            pickup_vx_cmd, pickup_vy_cmd = 0.0, 0.0
            pickup_phase_step[0] += 1
            target_x, target_y = _cell_to_world(tuple(goal_cell))
            reorient_theta = math.atan2(target_y - pose["y"], target_x - pose["x"])
            angle_diff = _wrap_angle(reorient_theta - pose["theta"])
            if abs(angle_diff) <= ANGLE_TOLERANCE_RAD or pickup_phase_step[0] >= _POSE_TIMEOUT_STEPS:
                pickup_omega_cmd = 0.0
                expected_theta[0] = reorient_theta
                pickup_phase[0] = None
            else:
                pickup_omega_cmd = max(
                    -MAX_ANGULAR_SPEED_RAD_S, min(MAX_ANGULAR_SPEED_RAD_S, KP_ANGULAR * angle_diff)
                )

        if pickup_phase[0] in ("reach", "close_gripper", "lift"):
            gripper_camera_print_countdown[0] -= 1
            detected = _locate_box_via_camera(expected_pickup_node[0], pose, gripper_pos, camera_rot)
            if detected is not None and gripper_camera_print_countdown[0] <= 0:
                print(
                    f"[youbot_mecanum robot_{robot_id}] gripper camera sees box at "
                    f"({detected[0]:.4f}, {detected[1]:.4f}) during '{pickup_phase[0]}'.",
                    flush=True,
                )
                gripper_camera_print_countdown[0] = 15

        if tuple(goal_cell) == last_pickup_completed_cell[0]:
            arrived = True

        # Command source: RL velocity, pickup sequence, stop when arrived, or goal-cell autopilot.
        if continuous_cmd is not None:
            vx_cmd, vy_cmd, omega_cmd = continuous_cmd
        elif pickup_vx_cmd is not None:
            vx_cmd, vy_cmd, omega_cmd = pickup_vx_cmd, pickup_vy_cmd, pickup_omega_cmd
        elif arrived:
            vx_cmd, vy_cmd, omega_cmd = 0.0, 0.0, 0.0
        else:
            speed = min(MAX_BODY_SPEED_MPS, max(0.1, KP_LINEAR * dist))
            vx_world = speed * dx / dist
            vy_world = speed * dy / dist
            vx_cmd, vy_cmd = _world_to_body(vx_world, vy_world, pose["theta"])
            omega_cmd = 0.0
            if lidar is not None:
                lidar_range = _lidar_min_range(
                    lidar.getRangeImage(), lidar_self_mask[0], LIDAR_SELF_DYNAMIC_FLOOR_M
                )
                safety_scale = _lidar_safety_scale(lidar_range)

                if 0.0 < safety_scale <= LIDAR_STALL_TRIGGER_SCALE:
                    hard_stuck_steps[0] += 1
                else:
                    hard_stuck_steps[0] = 0

                if hard_stuck_steps[0] >= HARD_STUCK_RECOVERY_STEPS:
                    vx_cmd, vy_cmd, omega_cmd = 0.0, 0.0, 0.0
                    print(
                        f"[youbot_mecanum robot_{robot_id}] HARD STUCK RECOVERY: blocked for "
                        f"~{HARD_STUCK_RECOVERY_STEPS * TIME_STEP_MS / 1000:.1f}s straight "
                        f"(min_range={lidar_range:.3f}m) -- holding position; no teleport or "
                        f"physics reset. pose=("
                        f"{pose['x']:.4f}, {pose['y']:.4f}, {pose['theta']:.4f}) "
                        f"goal_cell={tuple(goal_cell)} (see HARD_STUCK_RECOVERY_STEPS' comment).",
                        flush=True,
                    )
                    hard_stuck_steps[0] = 0
                    lidar_stall_steps[0] = 0
                    lidar_creep_remaining[0] = 0
                    position_stall_ref[0] = None
                    position_stall_steps[0] = 0
                else:
                    if safety_scale >= 1.0:
                        omega_cmd = _heading_realign_omega(expected_theta[0], pose["theta"])
                    position_stalled, position_stall_ref[0], position_stall_steps[0] = (
                        _position_stall_detected(
                            (pose["x"], pose["y"]), position_stall_ref[0], position_stall_steps[0]
                        )
                    )
                    if position_stalled:
                        lidar_stall_steps[0] = 0
                        lidar_creep_remaining[0] = LIDAR_STALL_CREEP_DURATION_STEPS
                        print(
                            f"[youbot_mecanum robot_{robot_id}] POSITION STALL DETECTED: moved less "
                            f"than {POSITION_STALL_DISTANCE_M}m in "
                            f"~{POSITION_STALL_CHECK_STEPS * TIME_STEP_MS / 1000:.1f}s while actively "
                            f"transiting -- forcing stall-recovery creep regardless of what the LiDAR "
                            f"scale currently reads. min_range={lidar_range:.3f}m "
                            f"pose=({pose['x']:.4f}, {pose['y']:.4f}, {pose['theta']:.4f}) "
                            f"goal_cell={tuple(goal_cell)} (see POSITION_STALL_CHECK_STEPS' comment).",
                            flush=True,
                        )
                    retreat = lidar_range < LIDAR_STALL_RETREAT_RANGE_M
                    effective_scale, lidar_stall_steps[0], lidar_creep_remaining[0] = (
                        _lidar_stall_recovery_scale(
                            safety_scale, lidar_stall_steps[0], lidar_creep_remaining[0], retreat
                        )
                    )
                    rotational_scan_omega = None
                    if safety_scale <= 0.0:
                        effective_scale = 0.0
                        lidar_stall_steps[0] = 0
                        lidar_creep_remaining[0] = 0
                        rotational_scan_omega, _rotational_scan_done = _lidar_rotational_scan_cmd(
                            lidar.getRangeImage(), lidar_self_mask[0], lidar.getHorizontalResolution(),
                            math.degrees(lidar.getFov()),
                        )
                        if lidar_diag_countdown[0] <= 0:
                            print(
                                f"[youbot_mecanum robot_{robot_id}] LIDAR ROTATIONAL SCAN: fully "
                                f"braked (min_range={lidar_range:.3f}m) -- turning at omega="
                                f"{rotational_scan_omega:.3f}rad/s looking for a clear direction "
                                f"({'found one, resuming' if _rotational_scan_done else 'still scanning'}). "
                                f"pose=({pose['x']:.4f}, {pose['y']:.4f}, {pose['theta']:.4f}) "
                                f"goal_cell={tuple(goal_cell)} (see _lidar_rotational_scan_cmd's "
                                f"docstring).",
                                flush=True,
                            )
                    if effective_scale != safety_scale and lidar_diag_countdown[0] <= 0:
                        print(
                            f"[youbot_mecanum robot_{robot_id}] LIDAR STALL RECOVERY: fully braked "
                            f"for ~{LIDAR_STALL_RECOVERY_STEPS * TIME_STEP_MS / 1000:.1f}s straight "
                            f"-- {'retreating' if effective_scale < 0 else 'creeping forward'} at "
                            f"scale={effective_scale:.2f} for "
                            f"~{LIDAR_STALL_CREEP_DURATION_STEPS * TIME_STEP_MS / 1000:.1f}s to try "
                            f"to break the deadlock. min_range={lidar_range:.3f}m "
                            f"pose=({pose['x']:.4f}, {pose['y']:.4f}, {pose['theta']:.4f}) "
                            f"goal_cell={tuple(goal_cell)} (see LIDAR_STALL_RECOVERY_STEPS' comment).",
                            flush=True,
                        )
                    vx_cmd *= effective_scale
                    vy_cmd *= effective_scale
                    if rotational_scan_omega is not None:
                        omega_cmd = rotational_scan_omega
                    if safety_scale <= 0.0:
                        lateral_cmd = 0.0
                        lidar_avoidance_prev_range[0] = lidar_range
                        lidar_avoidance_worsening_steps[0] = 0
                    else:
                        lateral_cmd, lidar_avoidance_prev_range[0], lidar_avoidance_probe_sign[0], (
                            lidar_avoidance_worsening_steps[0]
                        ) = _lidar_avoidance_lateral_cmd(
                            lidar_range,
                            lidar_avoidance_prev_range[0],
                            lidar_avoidance_probe_sign[0],
                            lidar_avoidance_worsening_steps[0],
                        )
                    if lateral_cmd != 0.0 and lidar_diag_countdown[0] <= 0:
                        print(
                            f"[youbot_mecanum robot_{robot_id}] LIDAR AVOIDANCE DODGE: min_range="
                            f"{lidar_range:.3f}m -- nudging sideways (body-frame vy "
                            f"{'+' if lateral_cmd > 0 else ''}{lateral_cmd:.3f} m/s) to try to open "
                            f"up clearance before the stop threshold (see "
                            f"LIDAR_AVOIDANCE_LATERAL_SPEED_MPS' comment); flips side automatically "
                            f"if this direction doesn't help.",
                            flush=True,
                        )
                    vy_cmd = max(-MAX_BODY_SPEED_MPS, min(MAX_BODY_SPEED_MPS, vy_cmd + lateral_cmd))
                    lidar_diag_countdown[0] -= 1
                    if safety_scale < 1.0 and lidar_diag_countdown[0] <= 0:
                        print(
                            f"[youbot_mecanum robot_{robot_id}] LIDAR BRAKE ACTIVE: "
                            f"min_range(self-hits excluded)={lidar_range:.3f}m "
                            f"(stop<={LIDAR_SAFETY_STOP_M}, slow<={LIDAR_SAFETY_SLOW_M}) "
                            f"safety_scale={safety_scale:.2f} status={status[0]} "
                            f"pose=({pose['x']:.4f}, {pose['y']:.4f}, "
                            f"{pose['theta']:.4f}) goal_cell={tuple(goal_cell)} -- this only fires on "
                            f"the discrete goal-cell fallback path (continuous_cmd is None -- see its "
                            f"comment), so status=1(TO_ITEM)/3(CARRYING) here means the RL-driven "
                            f"cmd_vel went stale (>CMD_VEL_STALE_S with no fresh command), not that "
                            f"the policy itself commanded this; status=0(IDLE)/2(PICKING_UP) is the "
                            f"expected/by-design case for this path firing at all. If this never "
                            f"clears while the goal is genuinely reachable and no other robot is "
                            f"nearby, either a real non-self obstacle is actually there (cross-check "
                            f"pose against the map layout -- a centered robot in a 1-wide corridor "
                            f"should see ~0.55m to the nearest shelf face, so a persistent reading "
                            f"well below that plus this pose likely means it isn't well-centered, or "
                            f"is next to a corner/multiple shelf faces at once), or the self-mask "
                            f"calibration missed a self-hit ray -- see "
                            f"_lidar_calibrate_self_mask's docstring for that residual blind spot.",
                            flush=True,
                        )
                        lidar_diag_countdown[0] = 15

        # Ramp limits, then mecanum wheel speeds.
        vx_cmd, vy_cmd = _rate_limit_velocity_2d(
            vx_cmd, vy_cmd, prev_vx_cmd[0], prev_vy_cmd[0], MAX_VELOCITY_RAMP_MPS_PER_STEP
        )
        omega_cmd = _rate_limit_scalar(omega_cmd, prev_omega_cmd[0], MAX_OMEGA_RAMP_RADPS_PER_STEP)
        prev_vx_cmd[0], prev_vy_cmd[0] = vx_cmd, vy_cmd
        prev_omega_cmd[0] = omega_cmd
        wheel_speeds = _inverse_kinematics(vx_cmd, vy_cmd, omega_cmd)
        wheel_speeds = _scale_wheel_speeds_to_limit(wheel_speeds, wheel_max_velocity)
        for motor, speed in zip(motors, wheel_speeds):
            motor.setVelocity(speed)

        if continuous_cmd is not None:
            continuous_diag_countdown[0] -= 1
            if continuous_diag_countdown[0] <= 0:
                continuous_diag_countdown[0] = int(round(1.0 / TIME_STEP_S))
                print(
                    f"[youbot_mecanum robot_{robot_id}] CONTINUOUS DIAG: "
                    f"raw_cmd_vel=(v={cmd_vel_v[0]:.3f}, w={cmd_vel_omega[0]:.3f}) "
                    f"age={time.time() - cmd_vel_last_time[0]:.3f}s "
                    f"lidar_ready={lidar_self_mask[0] is not None} "
                    f"continuous_cmd=(vx={continuous_cmd[0]:.3f}, w={continuous_cmd[2]:.3f}) "
                    f"post_ramp=(vx={vx_cmd:.3f}, vy={vy_cmd:.3f}, w={omega_cmd:.3f}) "
                    f"wheel_speeds={[round(s, 3) for s in wheel_speeds]} "
                    f"wheel_max_velocity={wheel_max_velocity:.3f} "
                    f"pose=({pose['x']:.3f}, {pose['y']:.3f}, {pose['theta']:.3f})",
                    flush=True,
                )

        # Odometry from the encoders; replaced by the true pose when available.
        if encoders_available:
            deltas = []
            for i, sensor in enumerate(sensors):
                pos = sensor.getValue()
                if last_wheel_positions[i] is None:
                    deltas.append(0.0)
                else:
                    deltas.append((pos - last_wheel_positions[i]) * _WHEEL_RADIUS_M)
                last_wheel_positions[i] = pos
            body_dx, body_dy, dtheta = _forward_kinematics(deltas)
            cos_t, sin_t = math.cos(pose["theta"]), math.sin(pose["theta"])
            pose["x"] += body_dx * cos_t - body_dy * sin_t
            pose["y"] += body_dx * sin_t + body_dy * cos_t
            pose["theta"] += dtheta
        else:
            pose["x"] += vx_cmd * TIME_STEP_S
            pose["y"] += vy_cmd * TIME_STEP_S

        if self_node is not None:
            true_x, true_y, _true_z = self_node.getField("translation").getSFVec3f()
            pose["x"], pose["y"] = true_x, true_y
            orientation = self_node.getOrientation()
            pose["theta"] = math.atan2(orientation[3], orientation[0])

        if pickup_phase[0] is not None:
            arrived = False
        elif tuple(goal_cell) == last_pickup_completed_cell[0]:
            arrived = True

        arrived_pub.publish(Bool(data=arrived))
        pose_pub.publish(Pose2D(x=pose["x"], y=pose["y"], theta=pose["theta"]))

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
