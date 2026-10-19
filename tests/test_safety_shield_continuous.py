"""Continuous safety shield."""
import math

import numpy as np

from hiverl.env.continuous_kinematics import LOCAL_SENSING_RANGE_M, V_MAX_MPS, cell_to_world
from hiverl.env.robot import Robot, RobotStatus
from hiverl.env.safety_shield_continuous import apply_shield_continuous
from hiverl.env.warehouse_grid import AISLE, SHELF, WarehouseMap

FORWARD = (V_MAX_MPS, 0.0)
STOP = (0.0, 0.0)


def _corridor_map(width: int = 6) -> WarehouseMap:
    grid = np.full((3, width), SHELF, dtype=np.int8)
    grid[1, :] = AISLE
    return WarehouseMap(grid=grid)


def _robot(robot_id: int, cell: tuple[int, int], theta: float = 0.0, status=RobotStatus.IDLE) -> Robot:
    x, y = cell_to_world(cell)
    r = Robot(robot_id, pos=cell, status=status)
    r.x, r.y, r.theta = x, y, theta
    r.continuous_trail = [(x, y)]
    return r


def _apply(robots: list[Robot], result) -> None:
    for r in robots:
        x, y, theta = result.final_poses[r.robot_id]
        moved = (x, y) != (r.x, r.y)
        r.x, r.y, r.theta = x, y, theta
        if moved:
            r.push_continuous_trail()
        r.sync_pos_from_continuous()


def test_two_robots_converging_on_the_same_point_never_both_arrive_there():
    grid = np.zeros((3, 3), dtype=np.int8)
    m = WarehouseMap(grid=grid)
    a = _robot(0, (0, 1), theta=math.pi / 2)
    b = _robot(1, (2, 1), theta=-math.pi / 2)
    result = apply_shield_continuous(
        [a, b], {0: FORWARD, 1: FORWARD}, m, dt=1.0,
        enable_veto=True, enable_deadlock_reversal=False, deadlock_wait_threshold=3,
    )
    moved = [rid for rid, v in result.final_velocities.items() if v != STOP]
    assert len(moved) == 1
    assert result.collision_events == set()
    assert len(result.veto_events) == 1
    ax, ay, _ = result.final_poses[0]
    bx, by, _ = result.final_poses[1]
    assert math.hypot(ax - bx, ay - by) > 0.0


def test_wall_check_accounts_for_robot_radius_not_just_center_cell():
    m = _corridor_map()
    r = _robot(0, (1, 2), theta=0.0, status=RobotStatus.TO_ITEM)
    r.y = 0.75
    result = apply_shield_continuous(
        [r], {0: FORWARD}, m, dt=1.0,
        enable_veto=True, enable_deadlock_reversal=False, deadlock_wait_threshold=3,
    )
    assert 0 in result.wall_block_events
    assert result.final_velocities[0] == STOP
    assert result.final_poses[0][:2] == (r.x, r.y)


def test_wall_check_allows_a_well_centered_aisle_move():
    m = _corridor_map()
    r = _robot(0, (1, 2), theta=0.0, status=RobotStatus.TO_ITEM)
    result = apply_shield_continuous(
        [r], {0: FORWARD}, m, dt=1.0,
        enable_veto=True, enable_deadlock_reversal=False, deadlock_wait_threshold=3,
    )
    assert result.wall_block_events == set()
    assert result.final_velocities[0] != STOP


def test_wall_check_allows_escape_from_an_already_violating_position():
    m = _corridor_map()
    r = _robot(0, (1, 2), theta=math.pi / 2, status=RobotStatus.TO_ITEM)
    r.y = 0.7
    result = apply_shield_continuous(
        [r], {0: (0.1, 0.0)}, m, dt=1.0,
        enable_veto=True, enable_deadlock_reversal=False, deadlock_wait_threshold=3,
    )
    assert 0 not in result.wall_block_events
    assert math.isclose(result.final_poses[0][1], 0.8, abs_tol=1e-6)


def test_wall_check_still_blocks_moving_deeper_into_an_already_violating_position():
    m = _corridor_map()
    r = _robot(0, (1, 2), theta=-math.pi / 2, status=RobotStatus.TO_ITEM)
    r.y = 0.7
    result = apply_shield_continuous(
        [r], {0: (0.1, 0.0)}, m, dt=1.0,
        enable_veto=True, enable_deadlock_reversal=False, deadlock_wait_threshold=3,
    )
    assert 0 in result.wall_block_events
    assert result.final_poses[0][:2] == (r.x, 0.7)


def test_veto_disabled_logs_collision_instead_of_choosing_a_winner():
    grid = np.zeros((3, 3), dtype=np.int8)
    m = WarehouseMap(grid=grid)
    a = _robot(0, (0, 1), theta=math.pi / 2)
    b = _robot(1, (2, 1), theta=-math.pi / 2)
    result = apply_shield_continuous(
        [a, b], {0: FORWARD, 1: FORWARD}, m, dt=1.0,
        enable_veto=False, enable_deadlock_reversal=False, deadlock_wait_threshold=3,
    )
    assert result.final_velocities[0] == STOP
    assert result.final_velocities[1] == STOP
    assert result.collision_events == {0, 1}


def test_head_on_corridor_deadlock_persists_without_reversal():
    m = _corridor_map()
    a = _robot(0, (1, 1), theta=0.0)
    b = _robot(1, (1, 2), theta=math.pi)
    for _ in range(10):
        result = apply_shield_continuous(
            [a, b], {0: FORWARD, 1: FORWARD}, m, dt=1.0,
            enable_veto=True, enable_deadlock_reversal=False, deadlock_wait_threshold=3,
        )
        assert result.final_velocities[0] == STOP
        assert result.final_velocities[1] == STOP
        _apply([a, b], result)
    assert a.pos == (1, 1)
    assert b.pos == (1, 2)


def test_veto_success_is_not_double_counted_as_a_collision():
    grid = np.zeros((3, 3), dtype=np.int8)
    m = WarehouseMap(grid=grid)
    a = _robot(0, (0, 1), theta=math.pi / 2)
    b = _robot(1, (1, 1))
    result = apply_shield_continuous(
        [a, b], {0: FORWARD, 1: STOP}, m, dt=1.0,
        enable_veto=True, enable_deadlock_reversal=False, deadlock_wait_threshold=3,
    )
    assert result.collision_events == set()
    assert result.veto_events == {0}
    assert result.reward_penalty_events == {0}


def test_corridor_clearance_vetoes_robots_ending_up_adjacent_in_a_corridor():
    m = _corridor_map()
    a = _robot(0, (1, 1), theta=0.0)
    b = _robot(1, (1, 4), theta=math.pi)
    result = apply_shield_continuous(
        [a, b], {0: FORWARD, 1: FORWARD}, m, dt=1.0,
        enable_veto=True, enable_deadlock_reversal=False, deadlock_wait_threshold=3,
    )
    moved = [rid for rid, v in result.final_velocities.items() if v != STOP]
    assert len(moved) == 1


def test_corridor_clearance_does_not_apply_in_open_floor_space():
    grid = np.zeros((5, 5), dtype=np.int8)
    m = WarehouseMap(grid=grid)
    a = _robot(0, (2, 1), theta=0.0)
    b = _robot(1, (2, 4), theta=math.pi)
    result = apply_shield_continuous(
        [a, b], {0: FORWARD, 1: FORWARD}, m, dt=1.0,
        enable_veto=True, enable_deadlock_reversal=False, deadlock_wait_threshold=3,
    )
    moved = [rid for rid, v in result.final_velocities.items() if v != STOP]
    assert len(moved) == 2


def test_pickup_clearance_vetoes_a_robot_moving_adjacent_to_a_picking_up_robot():
    grid = np.zeros((5, 5), dtype=np.int8)
    m = WarehouseMap(grid=grid)
    picking_up = _robot(0, (2, 2), status=RobotStatus.PICKING_UP)
    passer = _robot(1, (2, 0), theta=0.0)
    result = apply_shield_continuous(
        [picking_up, passer], {0: STOP, 1: FORWARD}, m, dt=1.0,
        enable_veto=True, enable_deadlock_reversal=False, deadlock_wait_threshold=3,
    )
    assert result.final_velocities[1] == STOP
    assert 1 in result.veto_events


def test_pickup_clearance_marks_a_real_collision_when_veto_is_disabled():
    grid = np.zeros((5, 5), dtype=np.int8)
    m = WarehouseMap(grid=grid)
    picking_up = _robot(0, (2, 2), status=RobotStatus.PICKING_UP)
    passer = _robot(1, (2, 0), theta=0.0)
    result = apply_shield_continuous(
        [picking_up, passer], {0: STOP, 1: FORWARD}, m, dt=1.0,
        enable_veto=False, enable_deadlock_reversal=False, deadlock_wait_threshold=3,
    )
    assert result.final_velocities[1] == STOP
    assert 1 in result.collision_events


def test_pickup_clearance_does_not_apply_to_a_robot_that_isnt_picking_up():
    grid = np.zeros((5, 5), dtype=np.int8)
    m = WarehouseMap(grid=grid)
    stationary = _robot(0, (2, 2), status=RobotStatus.CARRYING)
    passer = _robot(1, (2, 0), theta=0.0)
    result = apply_shield_continuous(
        [stationary, passer], {0: STOP, 1: FORWARD}, m, dt=1.0,
        enable_veto=True, enable_deadlock_reversal=False, deadlock_wait_threshold=3,
    )
    assert result.final_velocities[1] == FORWARD


def test_pickup_clearance_does_not_restrict_cells_beyond_the_immediate_ring():
    grid = np.zeros((6, 6), dtype=np.int8)
    m = WarehouseMap(grid=grid)
    picking_up = _robot(0, (2, 2), status=RobotStatus.PICKING_UP)
    passer = _robot(1, (2, 5), theta=math.pi)
    result = apply_shield_continuous(
        [picking_up, passer], {0: STOP, 1: FORWARD}, m, dt=1.0,
        enable_veto=True, enable_deadlock_reversal=False, deadlock_wait_threshold=3,
    )
    assert result.final_velocities[1] == FORWARD


def test_robots_beyond_local_sensing_range_never_veto_each_other():
    grid = np.zeros((3, 20), dtype=np.int8)
    m = WarehouseMap(grid=grid)
    far = _robot(0, (1, 0), theta=0.0)
    other = _robot(1, (1, 19), theta=math.pi)
    assert math.hypot(far.x - other.x, far.y - other.y) > LOCAL_SENSING_RANGE_M

    for _ in range(5):
        result = apply_shield_continuous(
            [far, other], {0: FORWARD, 1: FORWARD}, m, dt=1.0,
            enable_veto=True, enable_deadlock_reversal=False, deadlock_wait_threshold=3,
        )
        assert result.veto_events == set()
        assert result.collision_events == set()
        _apply([far, other], result)


def test_head_on_corridor_deadlock_triggers_repeated_reversal_without_corrupting_state():
    m = _corridor_map()
    a = _robot(0, (1, 1), theta=0.0)
    b = _robot(1, (1, 2), theta=math.pi)
    a.continuous_trail = [cell_to_world((1, 0)), (a.x, a.y)]
    b.continuous_trail = [cell_to_world((1, 3)), (b.x, b.y)]

    reversal_count = 0
    for _ in range(20):
        a_v = FORWARD if a.pos[1] < 4 else STOP
        b_v = FORWARD if b.pos[1] > 0 else STOP
        result = apply_shield_continuous(
            [a, b], {0: a_v, 1: b_v}, m, dt=1.0,
            enable_veto=True, enable_deadlock_reversal=True, deadlock_wait_threshold=3,
        )
        if result.reversal_events:
            reversal_count += 1
        _apply([a, b], result)
        assert math.hypot(a.x - b.x, a.y - b.y) > 1e-6

    assert reversal_count >= 3


def _open_map() -> WarehouseMap:
    return WarehouseMap(grid=np.zeros((5, 8), dtype=np.int8))


def _shield(robots, desired, active):
    return apply_shield_continuous(
        robots=robots, desired_velocities=desired, warehouse_map=_open_map(), dt=0.3,
        enable_veto=True, enable_deadlock_reversal=True, deadlock_wait_threshold=1,
        active_resolution=active,
    )


def test_active_resolution_keeps_rotation_of_a_stopped_robot():
    for active, expected_omega in ((False, 0.0), (True, 0.5)):
        robots = [_robot(0, (2, 4), theta=math.pi), _robot(1, (2, 3), theta=0.0)]
        res = _shield(robots, {0: (V_MAX_MPS, 0.5), 1: (V_MAX_MPS, 0.5)}, active)
        assert res.final_velocities[1][0] == 0.0
        assert res.final_velocities[1][1] == expected_omega


def test_active_resolution_lower_priority_robot_backs_away_from_the_winner():
    def run(active):
        winner = _robot(0, (2, 2), theta=0.0, status=RobotStatus.CARRYING)
        loser = _robot(1, (2, 3), theta=0.0, status=RobotStatus.TO_ITEM)
        return _shield([winner, loser], {0: FORWARD, 1: STOP}, active)

    default = run(False)
    assert default.final_velocities[1] == (0.0, 0.0) and not default.retreat_events
    active = run(True)
    assert 1 in active.retreat_events
    assert active.final_velocities[1][0] > 0.0  # moves forward = away from the robot behind it
    assert active.final_poses[1][0] > 3.0


def test_active_resolution_deadlock_reversal_works_within_one_cell():
    def run(active):
        a = _robot(0, (2, 4), theta=math.pi)
        b = _robot(1, (2, 3), theta=0.0)
        b.continuous_trail = [(2.6, 2.0), (2.8, 2.0), (3.0, 2.0)]  # all inside cell (2, 3)
        b.blocked_steps = 5
        return _shield([a, b], {0: FORWARD, 1: FORWARD}, active)

    assert 1 not in run(False).reversal_events
    res = run(True)
    assert 1 in res.reversal_events
    assert res.final_velocities[1][0] < 0.0


def test_custom_safety_radius_lets_robots_get_closer():
    # Robots 0.9 m apart driving apart-parallel: one moving toward the other ends ~0.7 m away.
    def run(radius):
        a = _robot(0, (2, 2), theta=0.0)
        b = _robot(1, (2, 3), theta=0.0)
        b.x = 2.0 + 0.9
        return apply_shield_continuous(
            robots=[a, b], desired_velocities={0: (0.7, 0.0), 1: STOP}, warehouse_map=_open_map(), dt=0.3,
            enable_veto=True, enable_deadlock_reversal=False, deadlock_wait_threshold=1,
            safety_radius_m=radius,
        )

    assert run(None).final_velocities[0][0] == 0.0  # default 0.8 m: blocked
    assert run(0.65).final_velocities[0][0] > 0.0  # 0.65 m: allowed
