"""Deterministic safety shield for the grid env.

Before actions are applied it vetoes moves that would hit a wall or another robot (priority decides
who waits) and reverses robots stuck in a deadlock.
"""
from __future__ import annotations

from collections import defaultdict

from hiverl.env.robot import Robot, RobotStatus
from hiverl.env.warehouse_grid import ACTION_DELTAS, WarehouseMap, manhattan

WAIT_ACTION = 4
_REVERSE_LOOKUP = {v: k for k, v in ACTION_DELTAS.items()}

LOCAL_SENSING_RANGE = 3


class ShieldResult:
    """Shield output: final actions plus veto/collision/reversal events."""
    def __init__(self):
        self.final_actions: dict[int, int] = {}
        self.collision_events: set[int] = set()
        self.veto_events: set[int] = set()
        self.reversal_events: set[int] = set()
        self.wall_block_events: set[int] = set()

    @property
    def intervention_events(self) -> set[int]:
        """Robots the shield actually steered (veto or reversal)."""
        return self.veto_events | self.reversal_events

    @property
    def reward_penalty_events(self) -> set[int]:
        """Robots that attempted an invalid move (used for the reward penalty)."""
        return self.collision_events | self.veto_events | self.wall_block_events


def is_corridor_cell(warehouse_map: WarehouseMap, pos: tuple[int, int]) -> bool:
    """True if exactly two opposite neighbors are walkable."""
    r, c = pos
    open_dirs = []
    for name, (dr, dc) in (("v", (-1, 0)), ("v", (1, 0)), ("h", (0, -1)), ("h", (0, 1))):
        if warehouse_map.is_walkable((r + dr, c + dc)):
            open_dirs.append(name)
    return len(open_dirs) == 2 and open_dirs[0] == open_dirs[1]


def _pick_winner(ids: list[int], robots_by_id: dict[int, Robot]) -> int:
    return max(ids, key=lambda rid: robots_by_id[rid].priority_key())


def _within_local_sensing_range(a: int, b: int, positions: dict[int, tuple[int, int]]) -> bool:
    """True if two robots are close enough to conflict this step."""
    return manhattan(positions[a], positions[b]) <= LOCAL_SENSING_RANGE


def apply_shield(
    robots: list[Robot],
    desired_actions: dict[int, int],
    warehouse_map: WarehouseMap,
    enable_veto: bool,
    enable_deadlock_reversal: bool,
    deadlock_wait_threshold: int,
) -> ShieldResult:
    """Return the safe actions for all robots."""
    result = ShieldResult()
    robots_by_id = {r.robot_id: r for r in robots}
    positions = {r.robot_id: r.pos for r in robots}
    final_actions = dict(desired_actions)
    desired_next: dict[int, tuple[int, int]] = {}

    for r in robots:
        dr, dc = ACTION_DELTAS[desired_actions[r.robot_id]]
        nxt = (r.pos[0] + dr, r.pos[1] + dc)
        if not warehouse_map.is_walkable(nxt):
            nxt = r.pos
            final_actions[r.robot_id] = WAIT_ACTION
            if (dr, dc) != (0, 0):
                result.wall_block_events.add(r.robot_id)
        desired_next[r.robot_id] = nxt

    blocked_by: dict[int, int] = {}
    ids = list(desired_next.keys())
    mark_as_collision = not enable_veto

    def _stop(rid: int, is_collision: bool):
        final_actions[rid] = WAIT_ACTION
        desired_next[rid] = positions[rid]
        if is_collision:
            result.collision_events.add(rid)
        else:
            result.veto_events.add(rid)

    for _ in range(len(ids) + 1):
        changed = False
        movers = [i for i in ids if desired_next[i] != positions[i]]
        stationary_cells = {positions[i] for i in ids if desired_next[i] == positions[i]}

        for i in movers:
            if desired_next[i] in stationary_cells:
                occupant = next(j for j in ids if positions[j] == desired_next[i])
                _stop(i, mark_as_collision)
                if not mark_as_collision:
                    blocked_by[i] = occupant
                changed = True
        movers = [i for i in ids if desired_next[i] != positions[i]]

        by_cell: dict[tuple[int, int], list[int]] = defaultdict(list)
        for i in movers:
            by_cell[desired_next[i]].append(i)
        for group in by_cell.values():
            if len(group) < 2:
                continue
            assert all(
                _within_local_sensing_range(group[0], j, positions) for j in group[1:]
            ), "vertex conflict pair fell outside LOCAL_SENSING_RANGE -- should be impossible"
            for i in group:
                _stop(i, True)
            for k, i in enumerate(group):
                blocked_by[i] = group[(k + 1) % len(group)]
            changed = True
        movers = [i for i in ids if desired_next[i] != positions[i]]

        for a_idx in range(len(movers)):
            for b_idx in range(a_idx + 1, len(movers)):
                a, b = movers[a_idx], movers[b_idx]
                if desired_next[a] == positions[b] and desired_next[b] == positions[a]:
                    assert _within_local_sensing_range(a, b, positions), (
                        "swap conflict pair fell outside LOCAL_SENSING_RANGE -- should be impossible"
                    )
                    if mark_as_collision:
                        _stop(a, True)
                        _stop(b, True)
                    else:
                        winner = _pick_winner([a, b], robots_by_id)
                        loser = b if winner == a else a
                        _stop(loser, False)
                        blocked_by[loser] = winner
                    changed = True
        movers = [i for i in ids if desired_next[i] != positions[i]]

        for a_idx in range(len(movers)):
            for b_idx in range(a_idx + 1, len(movers)):
                a, b = movers[a_idx], movers[b_idx]
                cell_a, cell_b = desired_next[a], desired_next[b]
                if manhattan(cell_a, cell_b) != 1:
                    continue
                if not (is_corridor_cell(warehouse_map, cell_a) and is_corridor_cell(warehouse_map, cell_b)):
                    continue
                assert _within_local_sensing_range(a, b, positions), (
                    "corridor-clearance pair fell outside LOCAL_SENSING_RANGE -- should be impossible"
                )
                if mark_as_collision:
                    _stop(a, True)
                    _stop(b, True)
                else:
                    winner = _pick_winner([a, b], robots_by_id)
                    loser = b if winner == a else a
                    _stop(loser, False)
                    blocked_by[loser] = winner
                changed = True
        movers = [i for i in ids if desired_next[i] != positions[i]]

        pickup_exclusion_cells: dict[tuple[int, int], int] = {}
        for j in ids:
            if robots_by_id[j].status == RobotStatus.PICKING_UP:
                pr, pc = positions[j]
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    pickup_exclusion_cells[(pr + dr, pc + dc)] = j
        for i in movers:
            blocker = pickup_exclusion_cells.get(desired_next[i])
            if blocker is None:
                continue
            assert _within_local_sensing_range(i, blocker, positions), (
                "pickup-clearance pair fell outside LOCAL_SENSING_RANGE -- should be impossible"
            )
            _stop(i, mark_as_collision)
            if not mark_as_collision:
                blocked_by[i] = blocker
            changed = True

        if not changed:
            break

    for r in robots:
        wanted_to_move = ACTION_DELTAS[desired_actions[r.robot_id]] != (0, 0)
        got_blocked = final_actions[r.robot_id] == WAIT_ACTION and wanted_to_move
        r.blocked_steps = r.blocked_steps + 1 if got_blocked else 0

    if enable_deadlock_reversal:
        for r in robots:
            if r.blocked_steps <= deadlock_wait_threshold:
                continue
            other_id = blocked_by.get(r.robot_id)
            if other_id is None:
                continue
            other = robots_by_id[other_id]
            if other.blocked_steps <= deadlock_wait_threshold:
                continue
            if not (is_corridor_cell(warehouse_map, r.pos) and is_corridor_cell(warehouse_map, other.pos)):
                continue

            mover = r if r.priority_key() < other.priority_key() else other
            if len(mover.trail) < 2:
                continue
            backward_cell = mover.trail[-2]
            if backward_cell == mover.pos or not warehouse_map.is_walkable(backward_cell):
                continue
            if backward_cell in desired_next.values():
                continue

            delta = (backward_cell[0] - mover.pos[0], backward_cell[1] - mover.pos[1])
            action = _REVERSE_LOOKUP.get(delta)
            if action is None:
                continue

            final_actions[mover.robot_id] = action
            desired_next[mover.robot_id] = backward_cell
            result.reversal_events.add(mover.robot_id)
            result.veto_events.discard(mover.robot_id)
            mover.blocked_steps = 0

    result.final_actions = final_actions
    return result
