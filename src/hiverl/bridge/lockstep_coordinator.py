"""Lockstep bridge between the HiveRL env and real/simulated robots (Webots over ROS2).

LockstepCoordinator runs the grid env: each logical tick waits until the robots physically reach
their cells. ContinuousLockstepCoordinator runs the continuous env: policy-controlled robots get (v,
omega) for one tick of real time and the env computes rewards from the real poses.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable

from hiverl.agents.scripted_astar_agent import scripted_actions
from hiverl.bridge.coords import env_pose_to_world, world_pose_to_env
from hiverl.bridge.robot_driver import RobotDriver
from hiverl.env.continuous_kinematics import (
    DT_SECONDS,
    OMEGA_MAX_RADPS,
    ROBOT_RADIUS_M,
    V_MAX_MPS,
    clip_velocity,
    integrate_unicycle,
)
from hiverl.env.robot import RobotStatus
from hiverl.env.safety_shield import WAIT_ACTION
from hiverl.env.warehouse_env import WarehouseEnv
from hiverl.env.warehouse_env_continuous import ContinuousWarehouseEnv

PolicyFn = Callable[[dict], dict[str, int]]
ContinuousPolicyFn = Callable[[dict], dict[str, "list[float] | tuple[float, float]"]]

# RobotStatus -> integer code sent to the robot controller.
STATUS_CODE = {
    RobotStatus.IDLE: 0,
    RobotStatus.TO_ITEM: 1,
    RobotStatus.PICKING_UP: 2,
    RobotStatus.CARRYING: 3,
}

# Policy forward speed above which a shield stop counts as "blocked" (controller yield).
SHIELD_BLOCKED_MIN_V_MPS = 0.1
# Scale applied to omega before sending it to the real robot.
REAL_OMEGA_DERATE = 0.5

# Stall handling: minimum share of the round before a stalled round may end, and the hard timeout
# factor.
MIN_TICKS_STALL_BREAK_FRAC = 0.5
HARD_STALL_TIMEOUT_MULTIPLIER = 3.0


@dataclass
class TickResult:
    """Summary of one logical tick."""
    tick: int
    deliveries: int
    collisions: int
    interventions: int
    arrived_in_time: bool


@dataclass
class LockstepStats:
    """Per-tick results of a run."""
    ticks: list[TickResult] = field(default_factory=list)

    @property
    def total_deliveries(self) -> int:
        return sum(t.deliveries for t in self.ticks)

    @property
    def total_collisions(self) -> int:
        return sum(t.collisions for t in self.ticks)

    @property
    def total_interventions(self) -> int:
        return sum(t.interventions for t in self.ticks)

    @property
    def timed_out_ticks(self) -> int:
        return sum(1 for t in self.ticks if not t.arrived_in_time)


class LockstepCoordinator:
    """Grid-env coordinator: one logical tick per physical move."""
    def __init__(
        self,
        env: WarehouseEnv,
        driver: RobotDriver,
        poll_interval_s: float = 0.05,
        arrival_timeout_s: float = 10.0,
        sleep_fn: Callable[[float], None] = time.sleep,
        on_tick: Callable[[int, TickResult], None] | None = None,
        physical_proximity_stop_m: float = 0.6,
        synchronize_fleet: bool = False,
    ):
        self.env = env
        self.driver = driver
        self.poll_interval_s = poll_interval_s
        self.arrival_timeout_s = arrival_timeout_s
        self._sleep = sleep_fn
        self.on_tick = on_tick
        self.physical_proximity_stop_m = physical_proximity_stop_m
        self.synchronize_fleet = synchronize_fleet
        self._busy: dict[int, bool] = {}
        self._deadline: dict[int, float] = {}
        self._warned: set[int] = set()
        self._real_pickup_in_progress: set[int] = set()
        self.last_rewards: dict[str, float] = {}
        self.last_terminations: dict[str, bool] = {}
        self.last_truncations: dict[str, bool] = {}
        self.last_physical_proximity_events: set[int] = set()

    def reset(self, seed: int | None = None) -> dict:
        obs, _infos = self.env.reset(seed=seed)
        self._busy = {robot_id: self.synchronize_fleet for robot_id in self.env.robots}
        self._deadline = {}
        self._warned = set()
        self._real_pickup_in_progress = set()
        for robot_id, robot in self.env.robots.items():
            self.driver.reset(robot_id, robot.pos)
            self.driver.set_status(robot_id, STATUS_CODE[robot.status])
            if self.synchronize_fleet:
                self._deadline[robot_id] = time.monotonic() + self.arrival_timeout_s
        return obs

    def run(self, num_ticks: int, policy_fn: PolicyFn | None = None, reset: bool = True) -> LockstepStats:
        """Run up to num_ticks logical ticks; policy_fn=None uses the scripted A* baseline."""
        obs = self.reset() if reset else {
            self.env._agent_name(i): self.env._get_obs(i) for i in self.env.robots
        }
        stats = LockstepStats()

        tick = 0
        last_progress_at = time.monotonic()
        while tick < num_ticks:
            self._poll_busy_robots()
            newly_overdue = self._newly_overdue_robots()
            if newly_overdue and self.on_tick is not None:
                self.on_tick(
                    tick,
                    TickResult(tick=tick, deliveries=0, collisions=0, interventions=0, arrived_in_time=False),
                )

            if all(self._busy.values()) or (self.synchronize_fleet and any(self._busy.values())):
                stall_elapsed = time.monotonic() - last_progress_at
                min_ticks_before_stall_break = max(1, int(num_ticks * MIN_TICKS_STALL_BREAK_FRAC))
                if stall_elapsed >= self.arrival_timeout_s and (
                    tick >= min_ticks_before_stall_break
                    or stall_elapsed >= self.arrival_timeout_s * HARD_STALL_TIMEOUT_MULTIPLIER
                ):
                    break
                self._sleep(self.poll_interval_s)
                continue

            last_progress_at = time.monotonic()
            desired_actions = policy_fn(obs) if policy_fn is not None else scripted_actions(self.env)
            for robot_id, busy in self._busy.items():
                if busy:
                    desired_actions[self.env._agent_name(robot_id)] = WAIT_ACTION

            self.last_physical_proximity_events = set()
            poses = {rid: self.driver.get_pose(rid) for rid in self.env.robots}
            known = [(rid, p) for rid, p in poses.items() if p is not None]
            for idx_a in range(len(known)):
                for idx_b in range(idx_a + 1, len(known)):
                    rid_a, (xa, ya, _) = known[idx_a]
                    rid_b, (xb, yb, _) = known[idx_b]
                    if math.hypot(xa - xb, ya - yb) < self.physical_proximity_stop_m:
                        desired_actions[self.env._agent_name(rid_a)] = WAIT_ACTION
                        desired_actions[self.env._agent_name(rid_b)] = WAIT_ACTION
                        self.last_physical_proximity_events.add(rid_a)
                        self.last_physical_proximity_events.add(rid_b)

            for robot_id in self._real_pickup_in_progress:
                self.env.robots[robot_id].status = RobotStatus.PICKING_UP

            pre_positions = {robot_id: robot.pos for robot_id, robot in self.env.robots.items()}
            pre_statuses = {robot_id: robot.status for robot_id, robot in self.env.robots.items()}
            obs, self.last_rewards, self.last_terminations, truncated, _info = self.env.step(desired_actions)
            self.last_truncations = truncated

            for robot_id, robot in self.env.robots.items():
                if not self._busy[robot_id]:
                    moved = robot.pos != pre_positions[robot_id]
                    status_changed = robot.status != pre_statuses[robot_id]
                    needs_physical_goal = (
                        moved
                        or (status_changed and robot.status == RobotStatus.PICKING_UP)
                        or not self.synchronize_fleet
                    )
                    if needs_physical_goal:
                        self.driver.send_goal(robot_id, robot.pos)
                        self._busy[robot_id] = True
                        self._deadline[robot_id] = time.monotonic() + self.arrival_timeout_s
                        if robot.status == RobotStatus.PICKING_UP:
                            self._real_pickup_in_progress.add(robot_id)
                    if status_changed or needs_physical_goal:
                        self.driver.set_status(robot_id, STATUS_CODE[robot.status])

            result = TickResult(
                tick=tick,
                deliveries=len(self.env.last_delivery_events),
                collisions=len(self.env.last_collision_events),
                interventions=len(self.env.last_intervention_events),
                arrived_in_time=not newly_overdue,
            )
            stats.ticks.append(result)
            if self.on_tick is not None:
                self.on_tick(tick, result)
            tick += 1

            if any(truncated.values()):
                break

        deadline = time.monotonic() + self.arrival_timeout_s
        self._poll_busy_robots()
        while any(self._busy.values()) and time.monotonic() < deadline:
            self._sleep(self.poll_interval_s)
            self._poll_busy_robots()

        return stats

    def _poll_busy_robots(self) -> None:
        for robot_id, busy in self._busy.items():
            if busy and self.driver.poll_arrived(robot_id):
                self._busy[robot_id] = False
                self._deadline.pop(robot_id, None)
                self._warned.discard(robot_id)
                self._real_pickup_in_progress.discard(robot_id)
                getattr(self.env, "hold_robot_ids", set()).discard(robot_id)

    def _newly_overdue_robots(self) -> list[int]:
        """Robots that just passed their arrival deadline (reported once)."""
        now = time.monotonic()
        newly = []
        for robot_id, busy in self._busy.items():
            if busy and robot_id in self._deadline and now >= self._deadline[robot_id]:
                if robot_id not in self._warned:
                    self._warned.add(robot_id)
                    newly.append(robot_id)
        return newly


_RL_CONTROLLED_STATUSES = (RobotStatus.TO_ITEM, RobotStatus.CARRYING)

# Samples along each robot's predicted path for the proximity check.
PHYSICAL_PROXIMITY_CCD_SAMPLES = 4


def _predicted_world_pos(
    world_pose: tuple[float, float, float], action
) -> tuple[float, float]:
    """Expected real position at the end of the tick for a normalized (v, omega) action."""
    x, y, theta = world_pose
    if action is None:
        return x, y
    ex, ey, etheta = world_pose_to_env(x, y, theta)
    v, omega = clip_velocity(
        float(action[0]) * V_MAX_MPS, float(action[1]) * OMEGA_MAX_RADPS * REAL_OMEGA_DERATE
    )
    pred_ex, pred_ey, pred_etheta = integrate_unicycle(ex, ey, etheta, v, omega, DT_SECONDS)
    pred_x, pred_y, _ = env_pose_to_world(pred_ex, pred_ey, pred_etheta)
    return pred_x, pred_y


def _swept_min_distance(
    pos_a: tuple[float, float],
    pred_a: tuple[float, float],
    pos_b: tuple[float, float],
    pred_b: tuple[float, float],
) -> float:
    """Minimum distance between two robots along their predicted paths this tick."""
    min_dist = math.inf
    for k in range(PHYSICAL_PROXIMITY_CCD_SAMPLES + 1):
        t = k / PHYSICAL_PROXIMITY_CCD_SAMPLES
        ax = pos_a[0] + (pred_a[0] - pos_a[0]) * t
        ay = pos_a[1] + (pred_a[1] - pos_a[1]) * t
        bx = pos_b[0] + (pred_b[0] - pos_b[0]) * t
        by = pos_b[1] + (pred_b[1] - pos_b[1]) * t
        min_dist = min(min_dist, math.hypot(ax - bx, ay - by))
    return min_dist


# Legacy retreats (enable_legacy_retreats): ticks two robots may stay too close before one backs
# off.
PHYSICAL_STALL_TICKS_THRESHOLD = 10
# Lidar-stall metric: commanded speed above this but real displacement below a fraction of expected.
_LIDAR_STALL_MIN_V_MPS = 0.05
_LIDAR_STALL_DISPLACEMENT_FRAC = 0.2
_LIDAR_STALL_SUSTAINED_STREAK_TICKS = 5
# Legacy: ticks of lidar stall before an automatic reverse.
LIDAR_STALL_RETREAT_TRIGGER_TICKS = 15
# Pileup: this many robots within this distance for this many ticks.
PILEUP_DISTANCE_M = 2 * ROBOT_RADIUS_M
PILEUP_MIN_ROBOTS = 2
PILEUP_HARD_BREAK_TICKS = 30
# Legacy retreat speed, max duration, progress threshold and direction-flip patience.
PHYSICAL_STALL_RETREAT_V_MPS = -0.3
PHYSICAL_STALL_RETREAT_MAX_TICKS = 40
PHYSICAL_STALL_PROGRESS_EPS_M = 0.02
PHYSICAL_STALL_SIGN_FLIP_PATIENCE_TICKS = 3


class ContinuousLockstepCoordinator(LockstepCoordinator):
    """Coordinator for ContinuousWarehouseEnv: TO_ITEM/CARRYING robots get (v, omega) commands,
    IDLE/PICKING_UP robots get goal cells.

    Options: respawn_on_collision teleports robots that really collide; execute_shield_velocities
    sends the shield-corrected velocities instead of the raw action; enable_legacy_retreats turns on
    the scripted proximity stop/retreat.
    """

    def __init__(
        self,
        *args,
        velocity_tick_sleep_s: float | None = None,
        enable_legacy_retreats: bool = True,
        global_collision_reset_m: float | None = None,
        respawn_on_collision: bool = False,
        execute_shield_velocities: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.velocity_tick_sleep_s = (
            velocity_tick_sleep_s if velocity_tick_sleep_s is not None else DT_SECONDS
        )
        self.enable_legacy_retreats = enable_legacy_retreats
        self.global_collision_reset_m = global_collision_reset_m
        self.respawn_on_collision = respawn_on_collision
        self.execute_shield_velocities = execute_shield_velocities
        self.last_overridden_ids: set[int] = set()
        self.env.tick_dt_s = min(DT_SECONDS, self.velocity_tick_sleep_s)
        self._physical_stall_ticks: dict[int, int] = {}
        self._physical_stall_retreat_elapsed: dict[int, int] = {}
        self._physical_stall_retreat_sign: dict[int, float] = {}
        self._physical_stall_retreat_last_dist: dict[int, float] = {}
        self._physical_stall_retreat_bad_ticks: dict[int, int] = {}
        self._lidar_stall_streak: dict[int, int] = {}
        self.last_lidar_stall_events: set[int] = set()
        self._lidar_stall_retreat_elapsed: dict[int, int] = {}
        self.last_lidar_stall_streak_ends: list[int] = []
        self._pileup_streak = 0

    def _respawn_robots(self, robot_ids: list[int], penalize: bool = True) -> bool:
        """Teleport these robots out of a collision (env picks the cells); returns False if not
        possible.
        """
        cells = {}
        for rid in robot_ids:
            cell = self.env.respawn_robot(rid, penalize=penalize)
            if cell is None:
                return False
            cells[rid] = cell
        now = time.monotonic()
        for rid, cell in cells.items():
            self.driver.reset(rid, cell)
            self.driver.set_status(rid, STATUS_CODE[RobotStatus.IDLE])
            self._busy[rid] = True
            self._deadline[rid] = now + self.arrival_timeout_s
            self._warned.discard(rid)
            self._real_pickup_in_progress.discard(rid)
            for per_robot in (
                self._physical_stall_ticks, self._physical_stall_retreat_elapsed,
                self._physical_stall_retreat_sign, self._physical_stall_retreat_last_dist,
                self._physical_stall_retreat_bad_ticks, self._lidar_stall_retreat_elapsed,
            ):
                per_robot.pop(rid, None)
        return True

    def run(
        self, num_ticks: int, policy_fn: ContinuousPolicyFn, reset: bool = True
    ) -> LockstepStats:
        """Run up to num_ticks ticks of real time with the continuous policy."""
        assert isinstance(self.env, ContinuousWarehouseEnv), (
            "ContinuousLockstepCoordinator requires a ContinuousWarehouseEnv -- use the plain "
            "LockstepCoordinator for the discrete WarehouseEnv."
        )
        obs = self.reset() if reset else {
            self.env._agent_name(i): self.env._get_obs(i) for i in self.env.robots
        }
        stats = LockstepStats()

        tick = 0
        last_progress_at = time.monotonic()
        while tick < num_ticks:
            self.driver.spin()
            self._poll_busy_robots()
            newly_overdue = self._newly_overdue_robots()
            if newly_overdue and self.on_tick is not None:
                self.on_tick(
                    tick,
                    TickResult(tick=tick, deliveries=0, collisions=0, interventions=0, arrived_in_time=False),
                )

            # Goal-cell robots still moving: wait (unless velocity-mode robots are active).
            goal_cell_busy = {
                rid: busy
                for rid, busy in self._busy.items()
                if self.env.robots[rid].status not in _RL_CONTROLLED_STATUSES
            }
            has_velocity_mode_robots = any(
                r.status in _RL_CONTROLLED_STATUSES for r in self.env.robots.values()
            )
            if not has_velocity_mode_robots and (
                (goal_cell_busy and all(goal_cell_busy.values()))
                or (self.synchronize_fleet and any(goal_cell_busy.values()))
            ):
                stall_elapsed = time.monotonic() - last_progress_at
                min_ticks_before_stall_break = max(1, int(num_ticks * MIN_TICKS_STALL_BREAK_FRAC))
                if stall_elapsed >= self.arrival_timeout_s and (
                    tick >= min_ticks_before_stall_break
                    or stall_elapsed >= self.arrival_timeout_s * HARD_STALL_TIMEOUT_MULTIPLIER
                ):
                    break
                self._sleep(self.poll_interval_s)
                continue

            last_progress_at = time.monotonic()
            # Policy actions for this tick.
            actions = policy_fn(obs)
            policy_actions = dict(actions)

            self.last_physical_proximity_events = set()
            self.last_lidar_stall_events = set()
            self.last_lidar_stall_streak_ends = []
            poses = {rid: self.driver.get_pose(rid) for rid in self.env.robots}
            known = [(rid, p) for rid, p in poses.items() if p is not None]

            held = getattr(self.env, "hold_robot_ids", set())
            active_known = [(rid, p) for rid, p in known if rid not in held]
            # Real collision: respawn the pair (or end the round).
            global_collision_pair: tuple[int, int, float] | None = None
            for idx_a in range(len(active_known)):
                for idx_b in range(idx_a + 1, len(active_known)):
                    rid_a, (xa, ya, _) = active_known[idx_a]
                    rid_b, (xb, yb, _) = active_known[idx_b]
                    distance = math.hypot(xa - xb, ya - yb)
                    if (
                        self.global_collision_reset_m is not None
                        and distance < self.global_collision_reset_m
                    ):
                        global_collision_pair = (rid_a, rid_b, distance)
                        break
                if global_collision_pair is not None:
                    break
            if self.global_collision_reset_m is not None and global_collision_pair is not None:
                rid_a, rid_b, distance = global_collision_pair
                self._pileup_streak = 0
                charged = (min(rid_a, rid_b), max(rid_a, rid_b)) in getattr(self.env, "_deep_pairs", set())
                if self.respawn_on_collision and self._respawn_robots([rid_a, rid_b], penalize=not charged):
                    print(
                        f"[lockstep] COLLISION RESPAWN: robots {rid_a} and {rid_b} are "
                        f"{distance:.3f}m apart (< {self.global_collision_reset_m:.2f}m) -- "
                        "penalized, episode terminated and teleported apart; round continues.",
                        flush=True,
                    )
                else:
                    print(
                        f"[lockstep] GLOBAL COLLISION RESET: robots {rid_a} and {rid_b} are "
                        f"{distance:.3f}m apart (< {self.global_collision_reset_m:.2f}m) -- "
                        "ending this round so the next round performs a physical reset.",
                        flush=True,
                    )
                    break
            pileup_ids: set[int] = set()
            for idx_a in range(len(active_known)):
                for idx_b in range(idx_a + 1, len(active_known)):
                    rid_a, (xa, ya, _) = active_known[idx_a]
                    rid_b, (xb, yb, _) = active_known[idx_b]
                    if math.hypot(xa - xb, ya - yb) < PILEUP_DISTANCE_M:
                        pileup_ids.add(rid_a)
                        pileup_ids.add(rid_b)
            if len(pileup_ids) >= PILEUP_MIN_ROBOTS:
                self._pileup_streak += 1
            else:
                self._pileup_streak = 0
            if (
                self._pileup_streak >= PILEUP_HARD_BREAK_TICKS
                and self.respawn_on_collision
                and self._respawn_robots(sorted(pileup_ids))
            ):
                print(
                    f"[lockstep] PILEUP RESPAWN: {len(pileup_ids)} robots ({sorted(pileup_ids)}) "
                    f"mutually within {PILEUP_DISTANCE_M:.2f}m for {self._pileup_streak} ticks -- "
                    "penalized, episode terminated and teleported apart; round continues.",
                    flush=True,
                )
                self._pileup_streak = 0
            elif self._pileup_streak >= PILEUP_HARD_BREAK_TICKS:
                print(
                    f"[lockstep] ANTI-PILEUP BREAK: {len(pileup_ids)} robots "
                    f"({sorted(pileup_ids)}) mutually within {PILEUP_DISTANCE_M:.2f}m real "
                    f"distance for {self._pileup_streak} consecutive ticks -- ending this round "
                    f"now rather than risk a real Webots crash (see PILEUP_HARD_BREAK_TICKS' "
                    f"comment); the caller's normal short-round handling will force a physical "
                    f"reset before the next round.",
                    flush=True,
                )
                self._pileup_streak = 0
                break
            # Legacy proximity stop/retreat (only with enable_legacy_retreats).
            predicted = {
                rid: _predicted_world_pos(
                    p,
                    actions.get(self.env._agent_name(rid))
                    if self.env.robots[rid].status in _RL_CONTROLLED_STATUSES
                    else None,
                )
                for rid, p in known
            }
            physically_close: set[int] = set()
            should_yield_from: dict[int, tuple[int, float]] = {}
            for idx_a in range(len(known)):
                for idx_b in range(idx_a + 1, len(known)):
                    rid_a, (xa, ya, _) = known[idx_a]
                    rid_b, (xb, yb, _) = known[idx_b]
                    dist = _swept_min_distance(
                        (xa, ya), predicted[rid_a], (xb, yb), predicted[rid_b]
                    )
                    if dist < self.physical_proximity_stop_m:
                        physically_close.add(rid_a)
                        physically_close.add(rid_b)
                        self.last_physical_proximity_events.add(rid_a)
                        self.last_physical_proximity_events.add(rid_b)
                        a_rl = self.env.robots[rid_a].status in _RL_CONTROLLED_STATUSES
                        b_rl = self.env.robots[rid_b].status in _RL_CONTROLLED_STATUSES
                        if a_rl and not b_rl:
                            yielder = rid_a
                        elif b_rl and not a_rl:
                            yielder = rid_b
                        elif a_rl and b_rl:
                            yielder = max(rid_a, rid_b)
                        else:
                            continue
                        if yielder not in should_yield_from or dist < should_yield_from[yielder][1]:
                            should_yield_from[yielder] = (rid_b if yielder == rid_a else rid_a, dist)
            for rid, robot in self.env.robots.items():
                if robot.status not in _RL_CONTROLLED_STATUSES:
                    continue
                agent = self.env._agent_name(rid)
                lidar_streak = self._lidar_stall_streak.get(rid, 0)
                lidar_retreat_elapsed = self._lidar_stall_retreat_elapsed.get(rid, 0)
                if not self.enable_legacy_retreats:
                    self._lidar_stall_retreat_elapsed[rid] = 0
                elif lidar_retreat_elapsed > 0:
                    if lidar_streak == 0:
                        self._lidar_stall_retreat_elapsed[rid] = 0
                    elif lidar_retreat_elapsed < PHYSICAL_STALL_RETREAT_MAX_TICKS:
                        actions[agent] = [-abs(PHYSICAL_STALL_RETREAT_V_MPS) / V_MAX_MPS, 0.0]
                        self._lidar_stall_retreat_elapsed[rid] = lidar_retreat_elapsed + 1
                        continue
                    else:
                        self._lidar_stall_retreat_elapsed[rid] = 0
                elif lidar_streak >= LIDAR_STALL_RETREAT_TRIGGER_TICKS:
                    self._lidar_stall_retreat_elapsed[rid] = 1
                    actions[agent] = [-abs(PHYSICAL_STALL_RETREAT_V_MPS) / V_MAX_MPS, 0.0]
                    continue

                is_close = rid in physically_close
                if not self.enable_legacy_retreats:
                    self._physical_stall_ticks[rid] = 0
                    self._physical_stall_retreat_elapsed[rid] = 0
                    self._physical_stall_retreat_sign.pop(rid, None)
                    self._physical_stall_retreat_last_dist.pop(rid, None)
                    self._physical_stall_retreat_bad_ticks.pop(rid, None)
                    continue
                elapsed = self._physical_stall_retreat_elapsed.get(rid, 0)
                if (
                    elapsed > 0
                    and is_close
                    and rid in should_yield_from
                    and elapsed < PHYSICAL_STALL_RETREAT_MAX_TICKS
                ):
                    _, current_dist = should_yield_from[rid]
                    last_dist = self._physical_stall_retreat_last_dist.get(rid, current_dist)
                    sign = self._physical_stall_retreat_sign.get(rid, -1.0)
                    if current_dist <= last_dist + PHYSICAL_STALL_PROGRESS_EPS_M:
                        bad_ticks = self._physical_stall_retreat_bad_ticks.get(rid, 0) + 1
                        if bad_ticks >= PHYSICAL_STALL_SIGN_FLIP_PATIENCE_TICKS:
                            sign = -sign
                            bad_ticks = 0
                        self._physical_stall_retreat_bad_ticks[rid] = bad_ticks
                    else:
                        self._physical_stall_retreat_bad_ticks[rid] = 0
                    self._physical_stall_retreat_sign[rid] = sign
                    self._physical_stall_retreat_last_dist[rid] = current_dist
                    actions[agent] = [sign * abs(PHYSICAL_STALL_RETREAT_V_MPS) / V_MAX_MPS, 0.0]
                    self._physical_stall_retreat_elapsed[rid] = elapsed + 1
                    continue
                self._physical_stall_retreat_elapsed[rid] = 0
                self._physical_stall_retreat_sign.pop(rid, None)
                self._physical_stall_retreat_last_dist.pop(rid, None)
                self._physical_stall_retreat_bad_ticks.pop(rid, None)
                if not is_close:
                    self._physical_stall_ticks[rid] = 0
                    continue
                self._physical_stall_ticks[rid] = self._physical_stall_ticks.get(rid, 0) + 1
                if (
                    rid in should_yield_from
                    and self._physical_stall_ticks[rid] >= PHYSICAL_STALL_TICKS_THRESHOLD
                ):
                    self._physical_stall_retreat_elapsed[rid] = 1
                    self._physical_stall_ticks[rid] = 0
                    self._physical_stall_retreat_sign[rid] = -1.0
                    self._physical_stall_retreat_last_dist[rid] = should_yield_from[rid][1]
                    self._physical_stall_retreat_bad_ticks[rid] = 0
                    actions[agent] = [-abs(PHYSICAL_STALL_RETREAT_V_MPS) / V_MAX_MPS, 0.0]
                else:
                    actions[agent] = [0.0, 0.0]

            for robot_id in self._real_pickup_in_progress:
                self.env.robots[robot_id].status = RobotStatus.PICKING_UP

            # Robots whose action was replaced by a coordinator rule this tick.
            self.last_overridden_ids = {
                rid for rid in self.env.robots
                if actions[self.env._agent_name(rid)] is not policy_actions.get(self.env._agent_name(rid))
            }
            pre_positions = {robot_id: robot.pos for robot_id, robot in self.env.robots.items()}
            pre_statuses = {robot_id: robot.status for robot_id, robot in self.env.robots.items()}
            sent_v_mps: dict[int, float] = {}

            def _execute_and_observe(final_velocities: dict[int, tuple[float, float]]) -> dict:
                """Env pose_provider: send the commands, wait one tick, return the real poses."""
                for robot_id, robot in self.env.robots.items():
                    if robot.status not in _RL_CONTROLLED_STATUSES:
                        continue
                    self._real_pickup_in_progress.discard(robot_id)
                    self._busy[robot_id] = False
                    if self.execute_shield_velocities and robot_id not in self.last_overridden_ids:
                        v_mps, omega_radps = final_velocities.get(robot_id, (0.0, 0.0))
                        # Shield stopped a robot that wanted to drive forward: let the controller
                        # nudge it aside (right -> back -> left).
                        wanted_v = float(actions[self.env._agent_name(robot_id)][0]) * V_MAX_MPS
                        self.driver.set_shield_blocked(
                            robot_id,
                            robot_id in self.env.last_veto_events
                            and abs(v_mps) < 1e-6 and wanted_v >= SHIELD_BLOCKED_MIN_V_MPS,
                        )
                    else:
                        self.driver.set_shield_blocked(robot_id, False)
                        agent_action = actions[self.env._agent_name(robot_id)]
                        v_mps = float(agent_action[0]) * V_MAX_MPS
                        omega_radps = float(agent_action[1]) * OMEGA_MAX_RADPS
                    self.env.last_executed_velocities[robot_id] = (v_mps, omega_radps)
                    sent_v_mps[robot_id] = v_mps
                    self.driver.send_velocity(robot_id, v_mps, omega_radps * REAL_OMEGA_DERATE)
                if sent_v_mps:
                    self._sleep(self.velocity_tick_sleep_s)
                self.driver.spin()
                observed: dict[int, tuple[float, float, float]] = {}
                for robot_id in self.env.robots:
                    pose = self.driver.get_pose(robot_id)
                    if pose is None:
                        continue
                    observed[robot_id] = world_pose_to_env(*pose)
                    pre_pose = poses.get(robot_id)
                    sent_v = sent_v_mps.get(robot_id)
                    stalled_this_tick = False
                    if pre_pose is not None and sent_v is not None and abs(sent_v) > _LIDAR_STALL_MIN_V_MPS:
                        displacement_m = math.hypot(pose[0] - pre_pose[0], pose[1] - pre_pose[1])
                        expected_m = abs(sent_v) * self.velocity_tick_sleep_s
                        stalled_this_tick = displacement_m < _LIDAR_STALL_DISPLACEMENT_FRAC * expected_m
                    prev_streak = self._lidar_stall_streak.get(robot_id, 0)
                    if stalled_this_tick:
                        self.last_lidar_stall_events.add(robot_id)
                        self._lidar_stall_streak[robot_id] = prev_streak + 1
                    else:
                        if prev_streak > 0:
                            self.last_lidar_stall_streak_ends.append(prev_streak)
                        self._lidar_stall_streak[robot_id] = 0
                return observed

            self.env.pose_provider = _execute_and_observe
            try:
                obs, self.last_rewards, self.last_terminations, truncated, _info = self.env.step(actions)
            finally:
                self.env.pose_provider = None
            self.last_truncations = truncated

            for robot_id, robot in self.env.robots.items():
                if pre_statuses[robot_id] == RobotStatus.CARRYING and robot.status != RobotStatus.CARRYING:
                    self.driver.send_goal(robot_id, robot.pos)

            for robot_id, robot in self.env.robots.items():
                if robot.status in _RL_CONTROLLED_STATUSES:
                    self._real_pickup_in_progress.discard(robot_id)
                    self._busy[robot_id] = False
                    if robot.status != pre_statuses[robot_id]:
                        self.driver.set_status(robot_id, STATUS_CODE[robot.status])
                    continue
                if not self._busy[robot_id]:
                    moved = robot.pos != pre_positions[robot_id]
                    status_changed = robot.status != pre_statuses[robot_id]
                    needs_physical_goal = (
                        moved
                        or (status_changed and robot.status == RobotStatus.PICKING_UP)
                        or not self.synchronize_fleet
                    )
                    if needs_physical_goal:
                        self.driver.send_goal(robot_id, robot.pos)
                        self._busy[robot_id] = True
                        self._deadline[robot_id] = time.monotonic() + self.arrival_timeout_s
                        if robot.status == RobotStatus.PICKING_UP:
                            self._real_pickup_in_progress.add(robot_id)
                    if status_changed or needs_physical_goal:
                        self.driver.set_status(robot_id, STATUS_CODE[robot.status])

            result = TickResult(
                tick=tick,
                deliveries=len(self.env.last_delivery_events),
                collisions=len(self.env.last_collision_events),
                interventions=len(self.env.last_intervention_events),
                arrived_in_time=not newly_overdue,
            )
            stats.ticks.append(result)
            if self.on_tick is not None:
                self.on_tick(tick, result)
            tick += 1

            if any(truncated.values()):
                break

        for robot_id, robot in self.env.robots.items():
            if robot.status in _RL_CONTROLLED_STATUSES:
                self.driver.send_velocity(robot_id, 0.0, 0.0)
        self._poll_busy_robots()
        return stats
