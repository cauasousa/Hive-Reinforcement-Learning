"""youBot controller helpers: mecanum kinematics, LiDAR sectors, brake and yield logic."""
import importlib.util
import math
import os

import pytest

_MODULE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "webots", "controllers", "youbot_mecanum", "youbot_mecanum.py"
)
_spec = importlib.util.spec_from_file_location("youbot_mecanum", _MODULE_PATH)
youbot_mecanum = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(youbot_mecanum)

inverse_kinematics = youbot_mecanum._inverse_kinematics
scale_wheel_speeds_to_limit = youbot_mecanum._scale_wheel_speeds_to_limit
position_stall_detected = youbot_mecanum._position_stall_detected
rate_limit_velocity_2d = youbot_mecanum._rate_limit_velocity_2d
forward_kinematics = youbot_mecanum._forward_kinematics
world_to_body = youbot_mecanum._world_to_body
K = youbot_mecanum._K
R = youbot_mecanum._WHEEL_RADIUS_M
lidar_min_range = youbot_mecanum._lidar_min_range
lidar_safety_scale = youbot_mecanum._lidar_safety_scale
lidar_calibrate_self_mask = youbot_mecanum._lidar_calibrate_self_mask
lidar_stall_recovery_scale = youbot_mecanum._lidar_stall_recovery_scale
lidar_avoidance_lateral_cmd = youbot_mecanum._lidar_avoidance_lateral_cmd
heading_realign_omega = youbot_mecanum._heading_realign_omega
ray_index_for_local_angle_deg = youbot_mecanum._ray_index_for_local_angle_deg
local_angle_deg_for_ray_index = youbot_mecanum._local_angle_deg_for_ray_index
sector_ray_indices = youbot_mecanum._sector_ray_indices
sector_min_range = youbot_mecanum._sector_min_range
lidar_rotational_scan_cmd = youbot_mecanum._lidar_rotational_scan_cmd
wrap_angle = youbot_mecanum._wrap_angle
camera_point_to_world = youbot_mecanum._camera_point_to_world
gripper_chassis_local_pose = youbot_mecanum._gripper_chassis_local_pose
mat_to_axis_angle = youbot_mecanum._mat_to_axis_angle
mat_rot_z = youbot_mecanum._mat_rot_z
mat_rot_y = youbot_mecanum._mat_rot_y
mat_mul = youbot_mecanum._mat_mul
mat_vec = youbot_mecanum._mat_vec


@pytest.mark.parametrize(
    "vx,vy,omega",
    [
        (0.3, 0.0, 0.0),
        (0.0, 0.3, 0.0),
        (0.0, 0.0, 0.5),
        (0.2, -0.15, 0.3),
        (-0.25, 0.1, -0.4),
        (0.0, 0.0, 0.0),
    ],
)
def test_forward_is_exact_algebraic_inverse_of_inverse_kinematics(vx, vy, omega):
    wheel_speeds = inverse_kinematics(vx, vy, omega)
    wheel_deltas = [w * R for w in wheel_speeds]
    dx, dy, dtheta = forward_kinematics(wheel_deltas)

    assert dx == pytest.approx(vx, abs=1e-9)
    assert dy == pytest.approx(vy, abs=1e-9)
    assert dtheta == pytest.approx(omega, abs=1e-9)


def test_pure_forward_motion_spins_all_wheels_the_same_sign():
    w1, w2, w3, w4 = inverse_kinematics(vx=0.3, vy=0.0, omega=0.0)
    assert w1 > 0 and w2 > 0 and w3 > 0 and w4 > 0
    assert w1 == pytest.approx(w2) == pytest.approx(w3) == pytest.approx(w4)


def test_pure_sideways_motion_spins_diagonal_pairs_oppositely():
    w1, w2, w3, w4 = inverse_kinematics(vx=0.0, vy=0.3, omega=0.0)
    assert w1 > 0 and w4 > 0
    assert w2 < 0 and w3 < 0
    assert w1 == pytest.approx(w4)
    assert w2 == pytest.approx(w3)


def test_pure_rotation_spins_right_and_left_side_wheels_oppositely():
    w1, w2, w3, w4 = inverse_kinematics(vx=0.0, vy=0.0, omega=0.5)
    assert w1 > 0 and w3 > 0
    assert w2 < 0 and w4 < 0
    assert w1 == pytest.approx(w3)
    assert w2 == pytest.approx(w4)


def test_zero_velocity_gives_zero_wheel_speeds():
    assert inverse_kinematics(0.0, 0.0, 0.0) == [0.0, 0.0, 0.0, 0.0]


def test_wheel_geometry_constants_match_the_real_proto():
    assert R == pytest.approx(0.05)
    assert youbot_mecanum._HALF_LENGTH_M == pytest.approx(0.228)
    assert youbot_mecanum._HALF_WIDTH_M == pytest.approx(0.158)


def test_wheel_index_to_position_labels_match_the_real_proto():
    assert youbot_mecanum._WHEEL_NAMES == ["front-right", "front-left", "back-right", "back-left"]


def test_world_to_body_is_identity_at_zero_yaw():
    vx, vy = world_to_body(0.3, -0.2, theta=0.0)
    assert vx == pytest.approx(0.3)
    assert vy == pytest.approx(-0.2)


@pytest.mark.parametrize("theta", [0.0, 0.3, -0.7, 1.57, 3.0, -3.0])
def test_world_to_body_round_trips_through_the_odometry_rotation(theta):
    vx_world, vy_world = 0.25, -0.35
    vx_body, vy_body = world_to_body(vx_world, vy_world, theta)

    cos_t, sin_t = math.cos(theta), math.sin(theta)
    vx_recovered = vx_body * cos_t - vy_body * sin_t
    vy_recovered = vx_body * sin_t + vy_body * cos_t

    assert vx_recovered == pytest.approx(vx_world, abs=1e-9)
    assert vy_recovered == pytest.approx(vy_world, abs=1e-9)


def test_world_to_body_at_90_degrees_swaps_and_negates_axes():
    vx, vy = world_to_body(1.0, 0.0, theta=math.pi / 2)
    assert vx == pytest.approx(0.0, abs=1e-9)
    assert vy == pytest.approx(-1.0, abs=1e-9)


@pytest.mark.parametrize("pose_name", ["_ARM_POSE_REST_RAD", "_ARM_POSE_CARRY_RAD"])
def test_arm_pose_presets_stay_within_the_real_declared_joint_limits(pose_name):
    pose = getattr(youbot_mecanum, pose_name)
    limits = youbot_mecanum._ARM_JOINT_LIMITS_RAD
    for joint_name, angle in pose.items():
        low, high = limits[joint_name]
        assert low <= angle <= high, f"{pose_name}[{joint_name}]={angle} outside [{low}, {high}]"


def test_arm_poses_and_limits_cover_the_same_five_joints():
    joint_names = set(youbot_mecanum._ARM_MOTOR_NAMES)
    assert set(youbot_mecanum._ARM_JOINT_LIMITS_RAD.keys()) == joint_names
    assert set(youbot_mecanum._ARM_POSE_REST_RAD.keys()) == joint_names
    assert set(youbot_mecanum._ARM_POSE_CARRY_RAD.keys()) == joint_names


def test_lidar_min_range_ignores_inf_and_nan_no_return_sentinels():
    ranges = [float("inf"), 0.9, float("nan"), 0.4, float("inf")]
    assert lidar_min_range(ranges) == pytest.approx(0.4)


def test_lidar_min_range_is_inf_when_nothing_returns_anything():
    assert lidar_min_range([float("inf"), float("nan"), float("inf")]) == float("inf")


def test_lidar_safety_scale_is_full_speed_when_clear():
    assert lidar_safety_scale(youbot_mecanum.LIDAR_SAFETY_SLOW_M + 1.0) == 1.0
    assert lidar_safety_scale(float("inf")) == 1.0


def test_lidar_safety_scale_is_zero_at_and_below_the_stop_threshold():
    assert lidar_safety_scale(youbot_mecanum.LIDAR_SAFETY_STOP_M) == 0.0
    assert lidar_safety_scale(0.0) == 0.0


def test_lidar_safety_scale_ramps_linearly_between_the_two_thresholds():
    stop, slow = youbot_mecanum.LIDAR_SAFETY_STOP_M, youbot_mecanum.LIDAR_SAFETY_SLOW_M
    mid = (stop + slow) / 2
    assert lidar_safety_scale(mid) == pytest.approx(0.5)


def test_lidar_safety_slow_threshold_is_farther_out_than_the_stop_threshold():
    assert youbot_mecanum.LIDAR_SAFETY_SLOW_M > youbot_mecanum.LIDAR_SAFETY_STOP_M


def test_lidar_calibrate_self_mask_flags_rays_closer_than_the_stop_threshold():
    stop = youbot_mecanum.LIDAR_SAFETY_STOP_M
    margin = youbot_mecanum.LIDAR_SELF_CALIBRATION_MARGIN_M
    ranges = [stop - 0.1, stop + margin + 0.1, float("inf"), float("nan"), 0.01]
    assert lidar_calibrate_self_mask(ranges) == [True, False, False, False, True]


def test_lidar_calibrate_self_mask_catches_a_self_hit_reading_just_above_the_stop_threshold():
    stop = youbot_mecanum.LIDAR_SAFETY_STOP_M
    ranges = [stop + 0.001]
    assert lidar_calibrate_self_mask(ranges) == [True]


def test_lidar_min_range_excludes_self_masked_rays_even_though_theyre_the_scan_minimum():
    ranges = [0.067, 0.067, 0.5, float("inf")]
    self_mask = [True, True, False, False]
    assert lidar_min_range(ranges) == pytest.approx(0.067)
    assert lidar_min_range(ranges, self_mask) == pytest.approx(0.5)


def test_lidar_min_range_self_mask_none_matches_pre_calibration_behavior():
    ranges = [float("inf"), 0.9, float("nan"), 0.4, float("inf")]
    assert lidar_min_range(ranges, None) == lidar_min_range(ranges) == pytest.approx(0.4)


def test_lidar_min_range_dynamic_floor_clamps_a_self_hit_the_calibration_missed():
    ranges = [0.067, 0.5, float("inf")]
    self_mask = [False, False, False]
    assert lidar_min_range(ranges, self_mask) == pytest.approx(0.067)
    assert lidar_min_range(ranges, self_mask, dynamic_floor_m=0.1) == pytest.approx(0.1)


def test_lidar_min_range_dynamic_floor_does_not_hide_a_real_close_obstacle_above_the_floor():
    ranges = [0.15, 0.5]
    assert lidar_min_range(ranges, None, dynamic_floor_m=0.1) == pytest.approx(0.15)


def test_lidar_min_range_dynamic_floor_never_reports_clear_during_a_real_deep_collision():
    ranges = [0.02, 0.03, 0.01, float("inf"), float("nan")]
    result = lidar_min_range(ranges, None, dynamic_floor_m=0.1)
    assert result == pytest.approx(0.1)
    assert result != float("inf")


def test_lidar_stall_recovery_stays_at_the_raw_scale_while_not_fully_stalled():
    scale, stall, creep = 0.4, 0, 0
    for _ in range(youbot_mecanum.LIDAR_STALL_RECOVERY_STEPS + 5):
        scale, stall, creep = lidar_stall_recovery_scale(0.4, stall, creep)
    assert scale == pytest.approx(0.4)
    assert creep == 0


def test_lidar_stall_recovery_creeps_after_sustained_full_stop_then_returns_to_normal_braking():
    stall, creep = 0, 0
    effective_scale = None
    for _ in range(youbot_mecanum.LIDAR_STALL_RECOVERY_STEPS):
        effective_scale, stall, creep = lidar_stall_recovery_scale(0.0, stall, creep)
    assert effective_scale == pytest.approx(youbot_mecanum.LIDAR_STALL_CREEP_SCALE)
    assert creep == youbot_mecanum.LIDAR_STALL_CREEP_DURATION_STEPS - 1

    for _ in range(youbot_mecanum.LIDAR_STALL_CREEP_DURATION_STEPS - 1):
        effective_scale, stall, creep = lidar_stall_recovery_scale(0.0, stall, creep)
    assert effective_scale == pytest.approx(youbot_mecanum.LIDAR_STALL_CREEP_SCALE)
    assert creep == 0

    effective_scale, stall, creep = lidar_stall_recovery_scale(0.0, stall, creep)
    assert effective_scale == 0.0
    assert stall == 1
    assert creep == 0


def test_lidar_stall_recovery_triggers_on_a_sustained_near_zero_crawl_not_just_exact_zero():
    stall, creep = 0, 0
    effective_scale = None
    crawling_scales = [0.03, 0.05, 0.08, 0.06, 0.02, 0.04][:]
    for i in range(youbot_mecanum.LIDAR_STALL_RECOVERY_STEPS):
        effective_scale, stall, creep = lidar_stall_recovery_scale(
            crawling_scales[i % len(crawling_scales)], stall, creep
        )
    assert effective_scale == pytest.approx(youbot_mecanum.LIDAR_STALL_CREEP_SCALE)
    assert creep == youbot_mecanum.LIDAR_STALL_CREEP_DURATION_STEPS - 1


def test_lidar_stall_recovery_retreats_instead_of_advancing_when_the_obstacle_is_very_close():
    stall, creep = 0, 0
    effective_scale = None
    for _ in range(youbot_mecanum.LIDAR_STALL_RECOVERY_STEPS):
        effective_scale, stall, creep = lidar_stall_recovery_scale(0.0, stall, creep, retreat=True)
    assert effective_scale == pytest.approx(-youbot_mecanum.LIDAR_STALL_CREEP_SCALE)
    assert creep == youbot_mecanum.LIDAR_STALL_CREEP_DURATION_STEPS - 1

    for _ in range(youbot_mecanum.LIDAR_STALL_CREEP_DURATION_STEPS - 1):
        effective_scale, stall, creep = lidar_stall_recovery_scale(0.0, stall, creep, retreat=True)
    assert effective_scale == pytest.approx(-youbot_mecanum.LIDAR_STALL_CREEP_SCALE)
    assert creep == 0


def test_lidar_stall_recovery_resets_the_stall_counter_the_moment_it_clears():
    stall, creep = 0, 0
    for _ in range(youbot_mecanum.LIDAR_STALL_RECOVERY_STEPS - 1):
        _, stall, creep = lidar_stall_recovery_scale(0.0, stall, creep)
    assert stall == youbot_mecanum.LIDAR_STALL_RECOVERY_STEPS - 1
    scale, stall, creep = lidar_stall_recovery_scale(1.0, stall, creep)
    assert scale == pytest.approx(1.0)
    assert stall == 0


def test_camera_point_to_world_straight_ahead_identity_camera():
    identity = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    x, y = camera_point_to_world(1.0, 0.0, 0.0, (0.0, 0.0, 0.0), identity, 0.0, 0.0, 0.0)
    assert x == pytest.approx(1.0)
    assert y == pytest.approx(0.0)


def test_camera_point_to_world_accounts_for_camera_mount_offset_and_robot_yaw():
    identity = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    x, y = camera_point_to_world(1.0, 0.0, 0.0, (0.1, 0.0, 0.0), identity, 0.0, 0.0, math.pi / 2)
    assert x == pytest.approx(0.0, abs=1e-9)
    assert y == pytest.approx(1.1, abs=1e-9)


def test_gripper_chassis_local_pose_stays_within_a_plausible_reach_envelope():
    max_reach = (
        math.hypot(youbot_mecanum._ARM_BASE_OFFSET_M[0], youbot_mecanum._ARM_BASE_OFFSET_M[2])
        + youbot_mecanum._ARM1_ANCHOR_M[2]
        + math.hypot(youbot_mecanum._ARM2_ANCHOR_M[0], youbot_mecanum._ARM2_ANCHOR_M[2])
        + youbot_mecanum._ARM3_ANCHOR_M[2]
        + youbot_mecanum._ARM4_ANCHOR_M[2]
        + youbot_mecanum._ARM5_ANCHOR_M[2]
        + youbot_mecanum._GRIPPER_MOUNT_OFFSET_M[2]
    )
    for pose_name in ("_ARM_POSE_REST_RAD", "_ARM_POSE_CARRY_RAD", "_ARM_POSE_LIFTED_RAD"):
        angles = getattr(youbot_mecanum, pose_name)
        pos, rot = gripper_chassis_local_pose(angles)
        assert math.sqrt(sum(c * c for c in pos)) <= max_reach + 1e-6
        for row in rot:
            assert math.sqrt(sum(c * c for c in row)) == pytest.approx(1.0, abs=1e-6)


def test_mat_to_axis_angle_round_trips_known_rotations():
    for theta in (0.3, -0.7, 1.2, -2.5):
        for mat_fn in (mat_rot_z, mat_rot_y):
            m = mat_fn(theta)
            axis, angle = mat_to_axis_angle(m)[:3], mat_to_axis_angle(m)[3]
            assert angle == pytest.approx(abs(theta), abs=1e-6)
            rotated_axis = mat_vec(m, axis)
            assert rotated_axis[0] == pytest.approx(axis[0], abs=1e-6)
            assert rotated_axis[1] == pytest.approx(axis[1], abs=1e-6)
            assert rotated_axis[2] == pytest.approx(axis[2], abs=1e-6)


def test_mat_to_axis_angle_identity_is_zero_angle():
    identity = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    assert mat_to_axis_angle(identity)[3] == pytest.approx(0.0, abs=1e-9)


def test_rest_and_carry_poses_are_meaningfully_different():
    rest = youbot_mecanum._ARM_POSE_REST_RAD
    carry = youbot_mecanum._ARM_POSE_CARRY_RAD
    total_change = sum(abs(rest[j] - carry[j]) for j in rest)
    assert total_change > 1.0


def test_rate_limit_velocity_2d_lets_braking_through_instantly():
    vx, vy = rate_limit_velocity_2d(0.0, 0.0, 0.4, 0.0, max_step_mps=0.01)
    assert (vx, vy) == (0.0, 0.0)


def test_rate_limit_velocity_2d_caps_a_sudden_acceleration():
    vx, vy = rate_limit_velocity_2d(0.4, 0.0, 0.0, 0.0, max_step_mps=0.04)
    assert math.hypot(vx, vy) == pytest.approx(0.04)
    assert vx == pytest.approx(0.04)
    assert vy == pytest.approx(0.0)


def test_rate_limit_velocity_2d_preserves_direction_when_capping_a_diagonal_step():
    vx, vy = rate_limit_velocity_2d(0.3, 0.4, 0.0, 0.0, max_step_mps=0.05)
    assert math.hypot(vx, vy) == pytest.approx(0.05)
    assert vx == pytest.approx(0.05 * 0.6)
    assert vy == pytest.approx(0.05 * 0.8)


def test_rate_limit_velocity_2d_is_a_no_op_when_the_step_is_already_within_the_limit():
    vx, vy = rate_limit_velocity_2d(0.01, 0.0, 0.0, 0.0, max_step_mps=0.04)
    assert (vx, vy) == (0.01, 0.0)


def test_scale_wheel_speeds_to_limit_is_a_no_op_within_the_limit():
    speeds = [5.0, -3.0, 4.0, -1.0]
    assert scale_wheel_speeds_to_limit(speeds, 14.81) == speeds


def test_scale_wheel_speeds_to_limit_preserves_the_ratio_between_wheels():
    speeds = [20.0, -10.0, 16.0, -4.0]
    scaled = scale_wheel_speeds_to_limit(speeds, 14.81)
    assert max(abs(s) for s in scaled) < 14.81
    assert max(abs(s) for s in scaled) == pytest.approx(14.81, rel=0.01)
    ratio = scaled[0] / speeds[0]
    for original, result in zip(speeds, scaled):
        assert result == pytest.approx(original * ratio)


def test_scale_wheel_speeds_to_limit_handles_all_zero_speeds():
    assert scale_wheel_speeds_to_limit([0.0, 0.0, 0.0, 0.0], 14.81) == [0.0, 0.0, 0.0, 0.0]


def test_position_stall_detected_first_call_only_seeds_the_reference():
    stalled, ref, steps = position_stall_detected((1.0, 1.0), None, 0)
    assert stalled is False
    assert ref == (1.0, 1.0)
    assert steps == 0


def test_position_stall_detected_does_not_flag_normal_progress():
    ref, steps = (0.0, 0.0), 0
    stalled = None
    for i in range(youbot_mecanum.POSITION_STALL_CHECK_STEPS):
        pos = (i * 0.01, 0.0)
        stalled, ref, steps = position_stall_detected(pos, ref, steps)
    assert stalled is False


def test_position_stall_detected_flags_genuinely_not_moving():
    ref, steps = (5.0, -3.0), 0
    stalled = None
    for _ in range(youbot_mecanum.POSITION_STALL_CHECK_STEPS):
        stalled, ref, steps = position_stall_detected((5.0, -3.0), ref, steps)
    assert stalled is True
    assert steps == 0


def test_position_stall_detected_tolerates_small_jitter_below_the_distance_threshold():
    ref, steps = (0.0, 0.0), 0
    stalled = None
    for _ in range(youbot_mecanum.POSITION_STALL_CHECK_STEPS):
        stalled, ref, steps = position_stall_detected(
            (youbot_mecanum.POSITION_STALL_DISTANCE_M * 0.1, 0.0), ref, steps
        )
    assert stalled is True


def test_lidar_avoidance_lateral_cmd_is_a_no_op_when_nothing_is_close():
    cmd, prev, sign, worsening = lidar_avoidance_lateral_cmd(1.5, float("inf"), 1, 0)
    assert cmd == 0.0
    assert sign == 1
    assert worsening == 0


def test_lidar_avoidance_lateral_cmd_dodges_toward_the_probe_sign_while_range_improves():
    prev, sign, worsening = 0.45, 1, 0
    cmd = None
    ranges = [0.5, 0.55, 0.6, 0.65]
    for r in ranges:
        cmd, prev, sign, worsening = lidar_avoidance_lateral_cmd(r, prev, sign, worsening)
        assert sign == 1
        assert worsening == 0
    assert cmd == pytest.approx(
        youbot_mecanum.LIDAR_AVOIDANCE_LATERAL_SPEED_MPS
        * (1.0 - lidar_safety_scale(ranges[-1]))
    )


def test_lidar_avoidance_lateral_cmd_flips_the_probe_sign_when_the_wrong_side_was_tried():
    prev, sign, worsening = 0.5, 1, 0
    worsening_ranges = [0.5 - 0.05 * (i + 1) for i in range(youbot_mecanum.LIDAR_AVOIDANCE_WORSENING_STEPS)]
    for r in worsening_ranges:
        cmd, prev, sign, worsening = lidar_avoidance_lateral_cmd(r, prev, sign, worsening)
    assert sign == -1
    assert worsening == 0


def test_lidar_avoidance_lateral_cmd_scales_with_how_restricted_the_brake_ramp_already_is():
    cmd_far, *_ = lidar_avoidance_lateral_cmd(0.59, 0.59, 1, 0)
    cmd_near, *_ = lidar_avoidance_lateral_cmd(0.36, 0.59, 1, 0)
    assert 0.0 < cmd_far < cmd_near < youbot_mecanum.LIDAR_AVOIDANCE_LATERAL_SPEED_MPS


def test_heading_realign_omega_is_a_no_op_within_tolerance():
    assert heading_realign_omega(0.0, 0.05) == 0.0
    assert heading_realign_omega(0.0, -youbot_mecanum.HEADING_REALIGN_TOLERANCE_RAD) == 0.0


def test_heading_realign_omega_corrects_toward_expected_theta_after_a_knock():
    omega = heading_realign_omega(0.0, 1.0)
    assert omega < 0.0
    assert abs(omega) <= youbot_mecanum.MAX_ANGULAR_SPEED_RAD_S


def test_heading_realign_omega_wraps_correctly_across_the_pi_boundary():
    omega = heading_realign_omega(math.pi, -math.pi + 0.3)
    assert omega == pytest.approx(-0.6)
    assert abs(omega) < 1.0


_LIDAR_HORIZONTAL_RESOLUTION = 180
_LIDAR_FIELD_OF_VIEW_DEG = 360.0

_MEASURED_ANGLE_TO_INDEX = [
    (15, 81), (30, 76), (45, 67), (60, 58), (75, 50), (90, 44), (105, 39), (120, 30),
    (135, 23), (150, 13), (165, 5), (180, 0), (195, 174), (210, 166), (225, 158),
    (240, 148), (255, 140), (270, 134), (285, 129), (300, 121), (315, 113), (330, 103),
    (345, 100),
]


@pytest.mark.parametrize("angle_deg,measured_index", _MEASURED_ANGLE_TO_INDEX)
def test_ray_index_for_local_angle_deg_matches_the_live_calibration_sweep(angle_deg, measured_index):
    predicted = ray_index_for_local_angle_deg(
        angle_deg, _LIDAR_HORIZONTAL_RESOLUTION, _LIDAR_FIELD_OF_VIEW_DEG
    )
    diff = min((predicted - measured_index) % 180, (measured_index - predicted) % 180)
    assert diff <= 3, f"angle={angle_deg} predicted={predicted} measured={measured_index}"


def test_ray_index_for_local_angle_deg_known_anchors():
    assert ray_index_for_local_angle_deg(180.0, 180, 360.0) == 0
    assert ray_index_for_local_angle_deg(0.0, 180, 360.0) == 90
    assert ray_index_for_local_angle_deg(90.0, 180, 360.0) == 45
    assert ray_index_for_local_angle_deg(270.0, 180, 360.0) == 135


def test_local_angle_deg_for_ray_index_is_the_inverse_of_ray_index_for_local_angle_deg():
    for index in range(_LIDAR_HORIZONTAL_RESOLUTION):
        angle = local_angle_deg_for_ray_index(index, _LIDAR_HORIZONTAL_RESOLUTION, _LIDAR_FIELD_OF_VIEW_DEG)
        round_tripped = ray_index_for_local_angle_deg(
            angle, _LIDAR_HORIZONTAL_RESOLUTION, _LIDAR_FIELD_OF_VIEW_DEG
        )
        assert round_tripped == index


def test_sector_ray_indices_does_not_wrap_when_the_sector_is_away_from_index_zero():
    indices = sector_ray_indices(90.0, 10.0, _LIDAR_HORIZONTAL_RESOLUTION, _LIDAR_FIELD_OF_VIEW_DEG)
    assert indices == list(range(40, 51))


def test_sector_ray_indices_wraps_around_the_array_boundary_for_a_sector_centered_behind():
    indices = sector_ray_indices(180.0, 10.0, _LIDAR_HORIZONTAL_RESOLUTION, _LIDAR_FIELD_OF_VIEW_DEG)
    assert indices[0] == 175
    assert indices[-1] == 5
    assert 0 in indices
    assert len(indices) == 11


def test_sector_min_range_ignores_self_masked_and_inf_rays():
    ranges = [float("inf")] * _LIDAR_HORIZONTAL_RESOLUTION
    ranges[45] = 0.3
    self_mask = [False] * _LIDAR_HORIZONTAL_RESOLUTION
    self_mask[44] = True
    ranges[44] = 0.1
    result = sector_min_range(
        ranges, self_mask, 90.0, 10.0, _LIDAR_HORIZONTAL_RESOLUTION, _LIDAR_FIELD_OF_VIEW_DEG
    )
    assert result == pytest.approx(0.3)


def test_sector_min_range_is_inf_when_the_whole_sector_is_self_masked():
    ranges = [0.2] * _LIDAR_HORIZONTAL_RESOLUTION
    self_mask = [False] * _LIDAR_HORIZONTAL_RESOLUTION
    for i in sector_ray_indices(0.0, 10.0, _LIDAR_HORIZONTAL_RESOLUTION, _LIDAR_FIELD_OF_VIEW_DEG):
        self_mask[i] = True
    result = sector_min_range(
        ranges, self_mask, 0.0, 10.0, _LIDAR_HORIZONTAL_RESOLUTION, _LIDAR_FIELD_OF_VIEW_DEG
    )
    assert result == float("inf")


def test_lidar_rotational_scan_cmd_turns_right_immediately_when_right_is_already_clear():
    ranges = [1.5] * _LIDAR_HORIZONTAL_RESOLUTION
    self_mask = [False] * _LIDAR_HORIZONTAL_RESOLUTION
    omega, done = lidar_rotational_scan_cmd(
        ranges, self_mask, _LIDAR_HORIZONTAL_RESOLUTION, _LIDAR_FIELD_OF_VIEW_DEG
    )
    assert done is True
    assert omega < 0.0


def test_lidar_rotational_scan_cmd_keeps_scanning_when_boxed_in_on_every_side():
    ranges = [0.1] * _LIDAR_HORIZONTAL_RESOLUTION
    self_mask = [False] * _LIDAR_HORIZONTAL_RESOLUTION
    omega, done = lidar_rotational_scan_cmd(
        ranges, self_mask, _LIDAR_HORIZONTAL_RESOLUTION, _LIDAR_FIELD_OF_VIEW_DEG
    )
    assert done is False
    assert omega == pytest.approx(-youbot_mecanum.LIDAR_ROTATIONAL_SCAN_OMEGA_RADPS)


def test_lidar_rotational_scan_cmd_stops_once_a_forward_flank_clears_even_if_right_never_did():
    ranges = [0.1] * _LIDAR_HORIZONTAL_RESOLUTION
    self_mask = [False] * _LIDAR_HORIZONTAL_RESOLUTION
    for i in sector_ray_indices(35.0, 15.0, _LIDAR_HORIZONTAL_RESOLUTION, _LIDAR_FIELD_OF_VIEW_DEG):
        ranges[i] = 1.5
    omega, done = lidar_rotational_scan_cmd(
        ranges, self_mask, _LIDAR_HORIZONTAL_RESOLUTION, _LIDAR_FIELD_OF_VIEW_DEG
    )
    assert done is True
    assert omega == 0.0


lidar_yield_direction = youbot_mecanum._lidar_yield_direction
LidarYieldController = youbot_mecanum._LidarYieldController


def _scan(default=1.5, **sectors):
    centers = {"right": -90.0, "back": 180.0, "left": 90.0, "front": 0.0}
    ranges = [default] * _LIDAR_HORIZONTAL_RESOLUTION
    for name, value in sectors.items():
        for i in sector_ray_indices(
            centers[name], youbot_mecanum.YIELD_SECTOR_HALF_WIDTH_DEG,
            _LIDAR_HORIZONTAL_RESOLUTION, _LIDAR_FIELD_OF_VIEW_DEG,
        ):
            ranges[i] = value
    return ranges


def _no_mask():
    return [False] * _LIDAR_HORIZONTAL_RESOLUTION


def test_yield_direction_prefers_right_when_everything_is_free():
    assert lidar_yield_direction(_scan(), _no_mask(), _LIDAR_HORIZONTAL_RESOLUTION) == "right"


def test_yield_direction_backs_up_when_right_is_occupied():
    assert lidar_yield_direction(_scan(right=0.4), _no_mask(), _LIDAR_HORIZONTAL_RESOLUTION) == "back"


def test_yield_direction_goes_left_when_right_and_back_are_occupied():
    ranges = _scan(right=0.4, back=0.4)
    assert lidar_yield_direction(ranges, _no_mask(), _LIDAR_HORIZONTAL_RESOLUTION) == "left"


def test_yield_direction_is_none_when_boxed_in():
    ranges = _scan(right=0.4, back=0.4, left=0.4)
    assert lidar_yield_direction(ranges, _no_mask(), _LIDAR_HORIZONTAL_RESOLUTION) is None


def test_yield_direction_treats_a_centered_one_cell_aisle_side_as_occupied():
    ranges = _scan(right=0.55, left=0.55)
    assert lidar_yield_direction(ranges, _no_mask(), _LIDAR_HORIZONTAL_RESOLUTION) == "back"


def test_yield_direction_never_treats_a_fully_self_masked_sector_as_free():
    mask = _no_mask()
    for i in sector_ray_indices(-90.0, youbot_mecanum.YIELD_SECTOR_HALF_WIDTH_DEG, _LIDAR_HORIZONTAL_RESOLUTION):
        mask[i] = True
    ranges = _scan(default=float("inf"))
    assert lidar_yield_direction(ranges, mask, _LIDAR_HORIZONTAL_RESOLUTION) == "back"


def test_yield_right_nudge_is_negative_vy_and_does_not_move_forward():
    ctrl = LidarYieldController(robot_id=0, time_step_s=0.032)
    cmd = ctrl.step(True, 0.0, _scan(front=0.3), _no_mask(), _LIDAR_HORIZONTAL_RESOLUTION)
    assert ctrl.just_started == "right"
    assert cmd == pytest.approx((0.0, -youbot_mecanum.YIELD_SPEED_MPS))


def test_yield_does_not_fire_when_not_driving_forward_or_front_is_open():
    ctrl = LidarYieldController(robot_id=0, time_step_s=0.032)
    assert ctrl.step(False, 0.0, _scan(front=0.3), _no_mask(), _LIDAR_HORIZONTAL_RESOLUTION) is None
    assert ctrl.step(True, 1.0, _scan(), _no_mask(), _LIDAR_HORIZONTAL_RESOLUTION) is None
    assert ctrl.mode is None


def test_yield_nudge_is_brief_then_cools_down():
    dt = 0.032
    ctrl = LidarYieldController(robot_id=0, time_step_s=dt)
    ranges = _scan(front=0.3)
    cmds = [ctrl.step(True, 0.0, ranges, _no_mask(), _LIDAR_HORIZONTAL_RESOLUTION) for _ in range(200)]
    active = [c is not None for c in cmds]
    first_run = active.index(False)
    assert first_run == ctrl.duration_steps
    assert not any(active[first_run:first_run + ctrl.cooldown_total_steps])
    assert any(active[first_run + ctrl.cooldown_total_steps:])


def test_yield_releases_as_soon_as_the_front_opens():
    ctrl = LidarYieldController(robot_id=0, time_step_s=0.032)
    assert ctrl.step(True, 0.0, _scan(front=0.3), _no_mask(), _LIDAR_HORIZONTAL_RESOLUTION) is not None
    assert ctrl.step(True, 0.9, _scan(), _no_mask(), _LIDAR_HORIZONTAL_RESOLUTION) is None
    assert ctrl.mode is None


def test_yield_aborts_when_the_chosen_side_closes_up():
    ctrl = LidarYieldController(robot_id=0, time_step_s=0.032)
    assert ctrl.step(True, 0.0, _scan(front=0.3), _no_mask(), _LIDAR_HORIZONTAL_RESOLUTION) is not None
    closed = _scan(front=0.3, right=youbot_mecanum.YIELD_SIDE_ABORT_M - 0.05)
    assert ctrl.step(True, 0.0, closed, _no_mask(), _LIDAR_HORIZONTAL_RESOLUTION) is None
    assert ctrl.mode is None


def test_backing_up_waits_longer_for_a_higher_robot_id():
    dt = 0.032
    ranges = _scan(front=0.3, right=0.4)

    def steps_until_back(robot_id):
        ctrl = LidarYieldController(robot_id=robot_id, time_step_s=dt)
        for n in range(1, 500):
            cmd = ctrl.step(True, 0.0, ranges, _no_mask(), _LIDAR_HORIZONTAL_RESOLUTION)
            if cmd is not None:
                assert cmd == pytest.approx((-youbot_mecanum.YIELD_SPEED_MPS, 0.0))
                return n
        raise AssertionError("never backed up")

    assert steps_until_back(0) < steps_until_back(1) < steps_until_back(4)


def test_left_is_immediate_no_back_delay():
    ctrl = LidarYieldController(robot_id=4, time_step_s=0.032)
    cmd = ctrl.step(True, 0.0, _scan(front=0.3, right=0.4, back=0.4), _no_mask(), _LIDAR_HORIZONTAL_RESOLUTION)
    assert cmd == pytest.approx((0.0, youbot_mecanum.YIELD_SPEED_MPS))


def test_scan_mask_drops_returns_inside_the_chassis_but_keeps_real_contacts():
    ranges = [1.5] * _LIDAR_HORIZONTAL_RESOLUTION
    ranges[90] = 0.07
    ranges[45] = 0.20
    mask = youbot_mecanum._lidar_scan_mask(ranges, [False] * _LIDAR_HORIZONTAL_RESOLUTION)
    assert mask[90] is True
    assert mask[45] is False
    front = sector_min_range(ranges, mask, 0.0, 60.0, _LIDAR_HORIZONTAL_RESOLUTION, _LIDAR_FIELD_OF_VIEW_DEG)
    assert front == pytest.approx(1.5)


def test_scan_mask_keeps_the_calibrated_self_mask_and_handles_none():
    ranges = [1.5] * _LIDAR_HORIZONTAL_RESOLUTION
    self_mask = [False] * _LIDAR_HORIZONTAL_RESOLUTION
    self_mask[3] = True
    assert youbot_mecanum._lidar_scan_mask(ranges, self_mask)[3] is True
    assert youbot_mecanum._lidar_scan_mask(ranges, None) == [False] * _LIDAR_HORIZONTAL_RESOLUTION


def test_startup_calibration_cutoff_masks_the_chassis_self_hit_but_not_a_nearby_shelf():
    ranges = [1.5] * _LIDAR_HORIZONTAL_RESOLUTION
    ranges[90] = 0.067
    for i in range(20, 50):
        ranges[i] = 0.44
    startup = lidar_calibrate_self_mask(ranges, youbot_mecanum.LIDAR_SELF_STARTUP_MAX_M)
    assert startup[90] is True
    assert sum(startup) == 1
    assert sum(lidar_calibrate_self_mask(ranges)) == 31


def test_yield_fires_on_a_shield_stop_even_with_a_clear_front():
    # The coordinator flags a shield stop; the controller passes front_scale=0 for it.
    ctrl = LidarYieldController(robot_id=0, time_step_s=0.032)
    cmd = ctrl.step(True, 0.0, _scan(), _no_mask(), _LIDAR_HORIZONTAL_RESOLUTION)
    assert ctrl.just_started == "right"
    assert cmd == pytest.approx((0.0, -youbot_mecanum.YIELD_SPEED_MPS))
