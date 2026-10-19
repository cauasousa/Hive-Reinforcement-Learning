"""Grid safety shield."""
import numpy as np

from hiverl.env.robot import Robot, RobotStatus
from hiverl.env.safety_shield import LOCAL_SENSING_RANGE, WAIT_ACTION, apply_shield
from hiverl.env.warehouse_grid import ACTION_DELTAS, AISLE, SHELF, WarehouseMap

UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3


def _corridor_map(width: int = 6) -> WarehouseMap:
    grid = np.full((3, width), SHELF, dtype=np.int8)
    grid[1, :] = AISLE
    return WarehouseMap(grid=grid)


def _apply(robots, result):
    for r in robots:
        dr, dc = ACTION_DELTAS[result.final_actions[r.robot_id]]
        new_pos = (r.pos[0] + dr, r.pos[1] + dc)
        if new_pos != r.pos:
            r.pos = new_pos
            r.push_trail()


def test_two_robots_targeting_same_free_cell_is_a_reactive_collision_even_with_veto_enabled():
    grid = np.zeros((3, 3), dtype=np.int8)
    m = WarehouseMap(grid=grid)
    a = Robot(0, pos=(0, 1))
    b = Robot(1, pos=(2, 1))
    result = apply_shield([a, b], {0: DOWN, 1: UP}, m, enable_veto=True, enable_deadlock_reversal=False, deadlock_wait_threshold=3)
    moved = [rid for rid, act in result.final_actions.items() if act != WAIT_ACTION]
    assert moved == []
    assert result.collision_events == {0, 1}
    assert result.veto_events == set()


def test_veto_disabled_logs_collision_instead_of_choosing_a_winner():
    grid = np.zeros((3, 3), dtype=np.int8)
    m = WarehouseMap(grid=grid)
    a = Robot(0, pos=(0, 1))
    b = Robot(1, pos=(2, 1))
    result = apply_shield([a, b], {0: DOWN, 1: UP}, m, enable_veto=False, enable_deadlock_reversal=False, deadlock_wait_threshold=3)
    assert result.final_actions[0] == WAIT_ACTION
    assert result.final_actions[1] == WAIT_ACTION
    assert result.collision_events == {0, 1}


def test_head_on_corridor_deadlock_persists_without_reversal():
    m = _corridor_map()
    a = Robot(0, pos=(1, 1))
    b = Robot(1, pos=(1, 2))
    for _ in range(10):
        result = apply_shield(
            [a, b], {0: RIGHT, 1: LEFT}, m, enable_veto=True, enable_deadlock_reversal=False, deadlock_wait_threshold=3
        )
        assert result.final_actions[0] == WAIT_ACTION
        assert result.final_actions[1] == WAIT_ACTION
    assert a.pos == (1, 1)
    assert b.pos == (1, 2)


def test_veto_success_is_not_double_counted_as_a_collision():
    grid = np.zeros((3, 3), dtype=np.int8)
    m = WarehouseMap(grid=grid)
    a = Robot(0, pos=(0, 1))
    b = Robot(1, pos=(1, 1))
    result = apply_shield([a, b], {0: DOWN, 1: WAIT_ACTION}, m, enable_veto=True, enable_deadlock_reversal=False, deadlock_wait_threshold=3)
    assert result.collision_events == set()
    assert len(result.veto_events) == 1
    assert result.reward_penalty_events == result.veto_events


def test_corridor_clearance_vetoes_robots_ending_up_adjacent_in_a_corridor():
    m = _corridor_map()
    a = Robot(0, pos=(1, 1))
    b = Robot(1, pos=(1, 4))
    result = apply_shield(
        [a, b], {0: RIGHT, 1: LEFT}, m, enable_veto=True, enable_deadlock_reversal=False, deadlock_wait_threshold=3
    )
    moved = [rid for rid, act in result.final_actions.items() if act != WAIT_ACTION]
    assert len(moved) == 1


def test_corridor_clearance_does_not_apply_in_open_floor_space():
    grid = np.zeros((5, 5), dtype=np.int8)
    m = WarehouseMap(grid=grid)
    a = Robot(0, pos=(2, 1))
    b = Robot(1, pos=(2, 4))
    result = apply_shield(
        [a, b], {0: RIGHT, 1: LEFT}, m, enable_veto=True, enable_deadlock_reversal=False, deadlock_wait_threshold=3
    )
    moved = [rid for rid, act in result.final_actions.items() if act != WAIT_ACTION]
    assert len(moved) == 2


def test_pickup_clearance_vetoes_a_robot_moving_adjacent_to_a_picking_up_robot():
    grid = np.zeros((5, 5), dtype=np.int8)
    m = WarehouseMap(grid=grid)
    picking_up = Robot(0, pos=(2, 2), status=RobotStatus.PICKING_UP)
    passer = Robot(1, pos=(2, 0))
    result = apply_shield(
        [picking_up, passer], {0: WAIT_ACTION, 1: RIGHT}, m,
        enable_veto=True, enable_deadlock_reversal=False, deadlock_wait_threshold=3,
    )
    assert result.final_actions[1] == WAIT_ACTION
    assert 1 in result.veto_events


def test_pickup_clearance_marks_a_real_collision_when_veto_is_disabled():
    grid = np.zeros((5, 5), dtype=np.int8)
    m = WarehouseMap(grid=grid)
    picking_up = Robot(0, pos=(2, 2), status=RobotStatus.PICKING_UP)
    passer = Robot(1, pos=(2, 0))
    result = apply_shield(
        [picking_up, passer], {0: WAIT_ACTION, 1: RIGHT}, m,
        enable_veto=False, enable_deadlock_reversal=False, deadlock_wait_threshold=3,
    )
    assert result.final_actions[1] == WAIT_ACTION
    assert 1 in result.collision_events


def test_pickup_clearance_does_not_apply_to_a_robot_that_isnt_picking_up():
    grid = np.zeros((5, 5), dtype=np.int8)
    m = WarehouseMap(grid=grid)
    stationary = Robot(0, pos=(2, 2), status=RobotStatus.CARRYING)
    passer = Robot(1, pos=(2, 0))
    result = apply_shield(
        [stationary, passer], {0: WAIT_ACTION, 1: RIGHT}, m,
        enable_veto=True, enable_deadlock_reversal=False, deadlock_wait_threshold=3,
    )
    assert result.final_actions[1] == RIGHT


def test_pickup_clearance_does_not_restrict_cells_beyond_the_immediate_ring():
    grid = np.zeros((6, 6), dtype=np.int8)
    m = WarehouseMap(grid=grid)
    picking_up = Robot(0, pos=(2, 2), status=RobotStatus.PICKING_UP)
    passer = Robot(1, pos=(2, 5))
    result = apply_shield(
        [picking_up, passer], {0: WAIT_ACTION, 1: LEFT}, m,
        enable_veto=True, enable_deadlock_reversal=False, deadlock_wait_threshold=3,
    )
    assert result.final_actions[1] == LEFT


def test_robots_beyond_local_sensing_range_never_veto_each_other():
    grid = np.zeros((3, 20), dtype=np.int8)
    m = WarehouseMap(grid=grid)
    far = Robot(0, pos=(1, 0))
    other = Robot(1, pos=(1, 19))
    assert manhattan_pos(far.pos, other.pos) > LOCAL_SENSING_RANGE

    for _ in range(5):
        result = apply_shield(
            [far, other], {0: RIGHT, 1: LEFT}, m,
            enable_veto=True, enable_deadlock_reversal=False, deadlock_wait_threshold=3,
        )
        assert result.veto_events == set()
        assert result.collision_events == set()
        _apply([far, other], result)


def manhattan_pos(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def test_head_on_corridor_deadlock_triggers_repeated_reversal_without_corrupting_state():
    m = _corridor_map()
    a = Robot(0, pos=(1, 1))
    b = Robot(1, pos=(1, 2))
    a.trail = [(1, 0), (1, 1)]
    b.trail = [(1, 3), (1, 2)]

    reversal_count = 0
    for _ in range(20):
        a_action = RIGHT if a.pos[1] < 4 else WAIT_ACTION
        b_action = LEFT if b.pos[1] > 0 else WAIT_ACTION
        result = apply_shield(
            [a, b],
            {0: a_action, 1: b_action},
            m,
            enable_veto=True,
            enable_deadlock_reversal=True,
            deadlock_wait_threshold=3,
        )
        if result.reversal_events:
            reversal_count += 1
        _apply([a, b], result)
        assert a.pos != b.pos

    assert reversal_count >= 3
