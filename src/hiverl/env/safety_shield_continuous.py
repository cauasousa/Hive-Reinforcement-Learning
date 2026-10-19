"""Safety shield for the continuous env.

Predicts each robot's motion for one tick and stops moves that would get closer than the safety
radius to another robot or touch a wall. Robots already too close may still move apart, and a robot
blocked for too long backs up along its trail.
"""
from __future__ import annotations

import math

from hiverl.env.continuous_kinematics import (
    CELL_SIZE_M,
    CORRIDOR_SAFETY_RADIUS_M,
    LOCAL_SENSING_RANGE_M,
    PICKUP_EXCLUSION_RADIUS_M,
    ROBOT_RADIUS_M,
    SAFETY_RADIUS_M,
    V_MAX_MPS,
    cell_to_world,
    clip_velocity,
    integrate_unicycle,
    world_to_cell,
)
from hiverl.env.robot import Robot, RobotStatus
from hiverl.env.safety_shield import is_corridor_cell
from hiverl.env.warehouse_grid import WarehouseMap

_CCD_SAMPLES = 4
# Active resolution (active_resolution=True, used when the shield really drives the robots):
# speed of the lower-priority robot backing away from the one it blocks, and the minimum distance
# of the deadlock-reversal target behind a robot.
_RETREAT_V_MPS = 0.3
_DEADLOCK_BACK_MIN_M = 0.3
_RL_STATUSES = (RobotStatus.TO_ITEM, RobotStatus.CARRYING)
# Extra margin beyond the robot radius when checking walls.
_WALL_CLEARANCE_MARGIN_M = 0.05


def _circle_aabb_clearance(px: float, py: float, cell: tuple[int, int]) -> float:
    """Distance from a point to a cell's square (0 inside it)."""
    cx, cy = cell_to_world(cell)
    half = CELL_SIZE_M / 2
    dx = max(abs(px - cx) - half, 0.0)
    dy = max(abs(py - cy) - half, 0.0)
    return math.hypot(dx, dy)


_WALL_CLEARANCE_THRESHOLD_M = ROBOT_RADIUS_M + _WALL_CLEARANCE_MARGIN_M


def _min_wall_clearance(warehouse_map: WarehouseMap, px: float, py: float) -> float:
    """Distance from a point to the nearest non-walkable cell around it."""
    center_cell = world_to_cell(px, py)
    best = math.inf
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            neighbor = (center_cell[0] + dr, center_cell[1] + dc)
            if warehouse_map.in_bounds(neighbor) and warehouse_map.is_walkable(neighbor):
                continue
            best = min(best, _circle_aabb_clearance(px, py, neighbor))
    return best


def _violates_wall_clearance(warehouse_map: WarehouseMap, px: float, py: float) -> bool:
    """True if a robot centered at (px, py) would overlap a wall."""
    return _min_wall_clearance(warehouse_map, px, py) < _WALL_CLEARANCE_THRESHOLD_M


class ShieldResultContinuous:
    """Shield output: final velocities/poses plus veto/collision/reversal events."""
    def __init__(self):
        self.final_velocities: dict[int, tuple[float, float]] = {}
        self.final_poses: dict[int, tuple[float, float, float]] = {}
        self.position_overrides: dict[int, tuple[float, float, float]] = {}
        self.collision_events: set[int] = set()
        self.veto_events: set[int] = set()
        self.reversal_events: set[int] = set()
        self.wall_block_events: set[int] = set()
        self.retreat_events: set[int] = set()

    @property
    def intervention_events(self) -> set[int]:
        return self.veto_events | self.reversal_events

    @property
    def reward_penalty_events(self) -> set[int]:
        return self.collision_events | self.veto_events | self.wall_block_events


def _pick_winner(ids: list[int], robots_by_id: dict[int, Robot]) -> int:
    return max(ids, key=lambda rid: robots_by_id[rid].priority_key())


def _within_local_sensing_range(a: Robot, b: Robot) -> bool:
    return math.hypot(a.x - b.x, a.y - b.y) <= LOCAL_SENSING_RANGE_M


def _safety_radius_for_pair(
    a: Robot,
    pred_a: tuple[float, float, float],
    b: Robot,
    pred_b: tuple[float, float, float],
    warehouse_map: WarehouseMap,
    safety_radius_m: float | None = None,
    pickup_radius_m: float | None = None,
) -> float:
    """Safety radius for a pair (pickup > corridor > open floor); None uses the module constants."""
    if a.status == RobotStatus.PICKING_UP or b.status == RobotStatus.PICKING_UP:
        return PICKUP_EXCLUSION_RADIUS_M if pickup_radius_m is None else pickup_radius_m
    if is_corridor_cell(warehouse_map, world_to_cell(*pred_a[:2])) and is_corridor_cell(
        warehouse_map, world_to_cell(*pred_b[:2])
    ):
        return CORRIDOR_SAFETY_RADIUS_M
    return SAFETY_RADIUS_M if safety_radius_m is None else safety_radius_m


def _closest_approach(a: Robot, pred_a: tuple[float, float, float], b: Robot, pred_b: tuple[float, float, float]) -> float:
    """Minimum distance between two robots along their predicted motion this tick."""
    min_dist = math.inf
    for k in range(_CCD_SAMPLES + 1):
        t = k / _CCD_SAMPLES
        ax = a.x + (pred_a[0] - a.x) * t
        ay = a.y + (pred_a[1] - a.y) * t
        bx = b.x + (pred_b[0] - b.x) * t
        by = b.y + (pred_b[1] - b.y) * t
        min_dist = min(min_dist, math.hypot(ax - bx, ay - by))
    return min_dist


def apply_shield_continuous(
    robots: list[Robot],
    desired_velocities: dict[int, tuple[float, float]],
    warehouse_map: WarehouseMap,
    dt: float,
    enable_veto: bool,
    enable_deadlock_reversal: bool,
    deadlock_wait_threshold: int,
    active_resolution: bool = False,
    safety_radius_m: float | None = None,
    pickup_radius_m: float | None = None,
) -> ShieldResultContinuous:
    """Return the corrected velocities for all robots.

    active_resolution (for when the shield's velocities are executed): a stopped robot keeps its
    rotation, a lower-priority robot blocking a higher-priority one backs away from it, and the
    deadlock reversal targets a trail point at least _DEADLOCK_BACK_MIN_M behind.
    safety_radius_m / pickup_radius_m override the open-floor and PICKING_UP safety radii.
    """
    result = ShieldResultContinuous()
    robots_by_id = {r.robot_id: r for r in robots}
    ids = [r.robot_id for r in robots]
    final_v: dict[int, tuple[float, float]] = {}
    predicted: dict[int, tuple[float, float, float]] = {}

    for r in robots:
        if r.reversal_cooldown > 0:
            final_v[r.robot_id] = (0.0, 0.0)
            predicted[r.robot_id] = (r.x, r.y, r.theta)
            r.reversal_cooldown -= 1
            continue
        v, omega = clip_velocity(*desired_velocities[r.robot_id])
        nx, ny, ntheta = integrate_unicycle(r.x, r.y, r.theta, v, omega, dt)
        predicted_clearance = _min_wall_clearance(warehouse_map, nx, ny)
        if predicted_clearance < _WALL_CLEARANCE_THRESHOLD_M:
            current_clearance = _min_wall_clearance(warehouse_map, r.x, r.y)
            already_violating = current_clearance < _WALL_CLEARANCE_THRESHOLD_M
            if not (already_violating and predicted_clearance > current_clearance):
                nx, ny = r.x, r.y
                v = 0.0
                if abs(desired_velocities[r.robot_id][0]) > 1e-9:
                    result.wall_block_events.add(r.robot_id)
        final_v[r.robot_id] = (v, omega)
        predicted[r.robot_id] = (nx, ny, ntheta)

    mark_as_collision = not enable_veto
    blocked_by: dict[int, int] = {}
    retreating: set[int] = set()

    def _moving(rid: int) -> bool:
        if active_resolution:
            return abs(final_v[rid][0]) > 1e-9
        return final_v[rid] != (0.0, 0.0)

    def _stop(rid: int, is_collision: bool) -> None:
        r = robots_by_id[rid]
        if active_resolution:
            omega = final_v[rid][1]
            final_v[rid] = (0.0, omega)
            predicted[rid] = (r.x, r.y, integrate_unicycle(r.x, r.y, r.theta, 0.0, omega, dt)[2])
        else:
            final_v[rid] = (0.0, 0.0)
            predicted[rid] = (r.x, r.y, r.theta)
        if is_collision:
            result.collision_events.add(rid)
        else:
            result.veto_events.add(rid)

    def _try_retreat(rid: int, away_from: int) -> bool:
        """Move robot rid away from robot away_from (forward or reverse, whichever opens the gap)."""
        r, w = robots_by_id[rid], robots_by_id[away_from]
        if r.status not in _RL_STATUSES:
            return False
        d0 = math.hypot(r.x - w.x, r.y - w.y)
        best = None
        for sign in (-1.0, 1.0):
            nx, ny, nth = integrate_unicycle(r.x, r.y, r.theta, sign * _RETREAT_V_MPS, 0.0, dt)
            d1 = math.hypot(nx - w.x, ny - w.y)
            if d1 > d0 + 1e-6 and (best is None or d1 > best[0]):
                best = (d1, sign, (nx, ny, nth))
        if best is None:
            return False
        _, sign, pose = best
        clearance = _min_wall_clearance(warehouse_map, pose[0], pose[1])
        if clearance < _WALL_CLEARANCE_THRESHOLD_M and clearance <= _min_wall_clearance(warehouse_map, r.x, r.y):
            return False
        for oid in ids:
            if oid in (rid, away_from):
                continue
            o = robots_by_id[oid]
            if not _within_local_sensing_range(r, o):
                continue
            thr = _safety_radius_for_pair(r, pose, o, predicted[oid], warehouse_map, safety_radius_m, pickup_radius_m)
            appr = _closest_approach(r, pose, o, predicted[oid])
            now = math.hypot(r.x - o.x, r.y - o.y)
            if appr < thr and appr < now - 1e-6:
                return False
        final_v[rid] = (sign * _RETREAT_V_MPS, 0.0)
        predicted[rid] = pose
        retreating.add(rid)
        result.retreat_events.add(rid)
        result.veto_events.add(rid)
        return True

    for _ in range(len(ids) + 1):
        changed = False
        for a_idx in range(len(ids)):
            for b_idx in range(a_idx + 1, len(ids)):
                a_id, b_id = ids[a_idx], ids[b_idx]
                if not _moving(a_id) and not _moving(b_id):
                    continue
                a, b = robots_by_id[a_id], robots_by_id[b_id]
                if not _within_local_sensing_range(a, b):
                    continue
                threshold = _safety_radius_for_pair(
                    a, predicted[a_id], b, predicted[b_id], warehouse_map, safety_radius_m, pickup_radius_m,
                )
                approach = _closest_approach(a, predicted[a_id], b, predicted[b_id])
                if approach >= threshold:
                    continue
                current_d = math.hypot(a.x - b.x, a.y - b.y)
                if current_d < threshold:
                    end_d = math.hypot(
                        predicted[a_id][0] - predicted[b_id][0], predicted[a_id][1] - predicted[b_id][1]
                    )
                    if end_d > current_d + 1e-6 and approach >= current_d - 1e-6:
                        continue
                a_moving = _moving(a_id)
                b_moving = _moving(b_id)
                if active_resolution and (a_id in retreating or b_id in retreating):
                    # A retreating robot keeps opening the gap; only the other one waits.
                    other = b_id if a_id in retreating else a_id
                    if other in retreating or not _moving(other):
                        continue
                    _stop(other, False)
                    blocked_by[other] = a_id if other == b_id else b_id
                elif (
                    active_resolution and not mark_as_collision and a_moving != b_moving
                    and _pick_winner([a_id, b_id], robots_by_id) == (a_id if a_moving else b_id)
                    and _try_retreat(b_id if a_moving else a_id, a_id if a_moving else b_id)
                ):
                    pass  # the lower-priority stationary robot backs away; re-checked next pass
                elif mark_as_collision:
                    if a_moving:
                        _stop(a_id, True)
                    if b_moving:
                        _stop(b_id, True)
                elif a_moving and b_moving:
                    winner = _pick_winner([a_id, b_id], robots_by_id)
                    loser = b_id if winner == a_id else a_id
                    _stop(loser, False)
                    blocked_by[loser] = winner
                elif a_moving:
                    _stop(a_id, False)
                    blocked_by[a_id] = b_id
                else:
                    _stop(b_id, False)
                    blocked_by[b_id] = a_id
                changed = True
        if not changed:
            break

    for r in robots:
        if active_resolution:
            got_blocked = abs(final_v[r.robot_id][0]) <= 1e-9 and abs(desired_velocities[r.robot_id][0]) > 1e-9
        else:
            wanted_to_move = abs(desired_velocities[r.robot_id][0]) > 1e-9 or abs(desired_velocities[r.robot_id][1]) > 1e-9
            got_blocked = final_v[r.robot_id] == (0.0, 0.0) and wanted_to_move
        r.blocked_steps = r.blocked_steps + 1 if got_blocked else 0

    if enable_deadlock_reversal:
        reversed_into: set[tuple[int, int]] = set()
        for r in robots:
            if r.blocked_steps <= deadlock_wait_threshold:
                continue
            other_id = blocked_by.get(r.robot_id)
            if other_id is None:
                continue
            if len(r.continuous_trail) < 2:
                continue
            if active_resolution:
                # First trail point at least _DEADLOCK_BACK_MIN_M behind (with 0.3 s ticks the
                # previous point is usually in the same cell, which the default rule rejects).
                back = next(
                    (p for p in reversed(r.continuous_trail[:-1])
                     if math.hypot(p[0] - r.x, p[1] - r.y) >= _DEADLOCK_BACK_MIN_M),
                    None,
                )
                if back is None or _violates_wall_clearance(warehouse_map, *back):
                    continue
                if any(
                    o.robot_id != r.robot_id
                    and math.hypot(predicted[o.robot_id][0] - back[0], predicted[o.robot_id][1] - back[1])
                    < 2 * ROBOT_RADIUS_M
                    for o in robots
                ):
                    continue
                back_x, back_y = back
                result.position_overrides[r.robot_id] = (back_x, back_y, r.theta)
                back_dist = math.hypot(back_x - r.x, back_y - r.y)
                final_v[r.robot_id] = (-min(V_MAX_MPS, back_dist / max(dt, 1e-6)), 0.0)
                result.reversal_events.add(r.robot_id)
                result.veto_events.discard(r.robot_id)
                r.blocked_steps = 0
                r.reversal_cooldown = deadlock_wait_threshold
                continue
            back_x, back_y = r.continuous_trail[-2]
            back_cell = world_to_cell(back_x, back_y)
            if back_cell == r.pos or not warehouse_map.is_walkable(back_cell):
                continue
            if math.hypot(back_x - r.x, back_y - r.y) < 1e-6:
                continue
            if back_cell in reversed_into:
                continue
            if any(
                other_r.robot_id != r.robot_id and world_to_cell(*predicted[other_r.robot_id][:2]) == back_cell
                for other_r in robots
            ):
                continue

            result.position_overrides[r.robot_id] = (back_x, back_y, r.theta)
            back_dist = math.hypot(back_x - r.x, back_y - r.y)
            final_v[r.robot_id] = (-min(V_MAX_MPS, back_dist / max(dt, 1e-6)), 0.0)
            result.reversal_events.add(r.robot_id)
            result.veto_events.discard(r.robot_id)
            r.blocked_steps = 0
            r.reversal_cooldown = deadlock_wait_threshold
            reversed_into.add(back_cell)

    for rid, pose in result.position_overrides.items():
        predicted[rid] = pose

    result.final_velocities = final_v
    result.final_poses = predicted
    return result


__all__ = ["ShieldResultContinuous", "apply_shield_continuous"]
