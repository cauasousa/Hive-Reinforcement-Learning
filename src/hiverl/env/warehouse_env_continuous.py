"""Continuous-action warehouse environment used for Webots training.

Same map, orders, auction and A* as the grid env, but robots have a continuous pose (x, y, theta)
and act with (v, omega) in [-1, 1]. TO_ITEM/CARRYING robots follow the policy; IDLE robots use a
simple autopilot. With a pose_provider (Webots), rewards are computed from the real poses after each
move.
"""
from __future__ import annotations

import functools
import inspect
import math

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from hiverl.env.continuous_kinematics import (
    ARRIVAL_TOLERANCE_M,
    CELL_SIZE_M,
    DT_SECONDS,
    OMEGA_MAX_RADPS,
    ROBOT_RADIUS_M,
    SAFETY_RADIUS_M,
    V_MAX_MPS,
    cell_to_world,
    world_to_cell,
)
from hiverl.env.robot import Robot, RobotStatus
from hiverl.env.safety_shield_continuous import apply_shield_continuous
from hiverl.env.task_manager import TaskManager
from hiverl.env.warehouse_env import (
    REWARD_IDLE_WAIT,
    proximity_penalty,
)
from hiverl.env.warehouse_grid import AISLE, DROPOFF, PICKUP, SHELF, WarehouseMap, generate_warehouse_map
from hiverl.planners.astar import astar
from hiverl.planners.auctioneer import run_auction

# Simulated lidar rays (evenly spaced over 360 degrees) and raycast step.
NUM_LIDAR_RAYS = 16
_RAY_STEP_M = 0.1
_MOVE_EPS = 1e-3
_K_OMEGA = 2.0
# A pickup/dropoff counts when the tick's path passes within this distance.
_ARRIVAL_SWEEP_TOLERANCE_M = 1 * CELL_SIZE_M

# Reward terms (per tick unless noted).
REWARD_TIME_COST_CONTINUOUS = -0.03
REWARD_COLLISION_CONTINUOUS = -0.2
# Charged once per deep overlap (real collision).
REWARD_PHYSICAL_COLLISION_CONTINUOUS = -30.0
# Size of the agent_state observation (see _agent_state).
AGENT_STATE_DIM = 8
_STATUS_INDEX = {
    RobotStatus.IDLE: 0, RobotStatus.TO_ITEM: 1, RobotStatus.PICKING_UP: 2, RobotStatus.CARRYING: 3,
}

# Respawn cells must be at least this far from every other robot.
_RESPAWN_CLEARANCE_M = 1.5
# Chassis contact: centers closer than this cost REWARD_CONTACT_CONTINUOUS every tick.
CONTACT_DISTANCE_M = 0.48
REWARD_CONTACT_CONTINUOUS = -1.0
# Graded proximity penalty between PROXIMITY_START_M and contact.
PROXIMITY_START_M = 1.2
REWARD_PROXIMITY_MAX_CONTINUOUS = -0.5
# Old flat proximity tiers (kept for reference/tests).
REWARD_PROXIMITY_TIERS_CONTINUOUS: tuple[tuple[float, float], ...] = (
    (1.0, -0.15),
    (1.5, -0.06),
)

# Potential-based progress reward: scale * (gamma * phi' - phi), phi = -remaining path.
REWARD_PROGRESS_SCALE_CONTINUOUS = 0.55

_PBRS_GAMMA = 1.0

# Lookahead distance of the vector observation.
_LOOKAHEAD_DISTANCE_M = 1.5

# Task rewards.
REWARD_PICKUP_CONTINUOUS = 30.0
REWARD_DELIVERY_CONTINUOUS = 50.0

# Quadratic penalty on |omega| above a dead zone (only away from obstacles).
REWARD_OMEGA_PENALTY_CONTINUOUS = -0.3
_OMEGA_PENALTY_DEADZONE = 0.5
_OMEGA_PENALTY_CLEARANCE_M = 0.35

# Penalty on the change of the omega action between ticks.
REWARD_OMEGA_RATE_CONTINUOUS = -0.02

# Penalty for spinning/pushing near an obstacle without progress.
REWARD_STUCK_TURN_CONTINUOUS = -0.15
_STUCK_TURN_CLEARANCE_M = 0.35
_STUCK_TURN_V_THRESHOLD = 0.6
_STUCK_TURN_OMEGA_THRESHOLD = 0.6

# Ticks of stuck-turning that count as a stuck episode.
_STUCK_TICKS_THRESHOLD = 15

# Bonus on the tick a robot leaves a stuck episode.
REWARD_UNSTUCK_BONUS_CONTINUOUS = 0.08

# Linear penalty when closer than _WALL_PROXIMITY_THRESHOLD_M to a wall.
REWARD_WALL_PROXIMITY_SCALE_CONTINUOUS = -0.3
_WALL_PROXIMITY_THRESHOLD_M = 0.5


def _wrap_angle(angle: float) -> float:
    """Wrap an angle to [-pi, pi]."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


def _graded_proximity_penalty(distance: float) -> float:
    """0 beyond PROXIMITY_START_M, growing quadratically to REWARD_PROXIMITY_MAX_CONTINUOUS at contact.
    """
    if distance >= PROXIMITY_START_M:
        return 0.0
    frac = (PROXIMITY_START_M - max(distance, CONTACT_DISTANCE_M)) / (PROXIMITY_START_M - CONTACT_DISTANCE_M)
    return REWARD_PROXIMITY_MAX_CONTINUOUS * frac * frac


def _wall_proximity_penalty(distance: float, threshold: float, scale: float) -> float:
    """Linear penalty from 0 at threshold to scale at distance 0."""
    if distance >= threshold:
        return 0.0
    return scale * (threshold - distance) / threshold


def _remaining_path_distance(pos: tuple[float, float], path: list[tuple[int, int]]) -> float:
    """Distance from pos along all remaining path cells."""
    if not path:
        return 0.0
    hx, hy = cell_to_world(path[0])
    total = math.hypot(pos[0] - hx, pos[1] - hy)
    prev = (hx, hy)
    for cell in path[1:]:
        cx, cy = cell_to_world(cell)
        total += math.hypot(cx - prev[0], cy - prev[1])
        prev = (cx, cy)
    return total


def _lookahead_point(pos: tuple[float, float], path: list[tuple[int, int]], lookahead_m: float) -> tuple[float, float]:
    """Point lookahead_m ahead along the path (pure pursuit)."""
    if not path:
        return pos
    remaining = lookahead_m
    prev = pos
    for cell in path:
        cx, cy = cell_to_world(cell)
        seg_dx, seg_dy = cx - prev[0], cy - prev[1]
        seg_len = math.hypot(seg_dx, seg_dy)
        if seg_len >= remaining:
            if seg_len < 1e-9:
                return (cx, cy)
            t = remaining / seg_len
            return (prev[0] + seg_dx * t, prev[1] + seg_dy * t)
        remaining -= seg_len
        prev = (cx, cy)
    return prev


def _autopilot_velocity(robot: Robot, target_xy: tuple[float, float]) -> tuple[float, float]:
    """Simple P-controller (v, omega) toward a target (used for IDLE robots)."""
    dx, dy = target_xy[0] - robot.x, target_xy[1] - robot.y
    dist = math.hypot(dx, dy)
    if dist < ARRIVAL_TOLERANCE_M:
        return (0.0, 0.0)
    angle_err = _wrap_angle(math.atan2(dy, dx) - robot.theta)
    omega = max(-OMEGA_MAX_RADPS, min(OMEGA_MAX_RADPS, _K_OMEGA * angle_err))
    if abs(angle_err) >= math.pi / 2:
        v = 0.0
    else:
        v = V_MAX_MPS * (1.0 - abs(angle_err) / (math.pi / 2))
        v = min(v, dist / DT_SECONDS)
    return (v, omega)


class ContinuousWarehouseEnv(ParallelEnv):
    """Multi-robot warehouse with continuous poses and (v, omega) actions."""
    metadata = {"name": "hiverl_warehouse_continuous_v0"}

    @classmethod
    def from_config(cls, env_cfg: dict, **overrides) -> "ContinuousWarehouseEnv":
        valid_names = set(inspect.signature(cls.__init__).parameters) - {"self"}
        kwargs = {**env_cfg, **overrides}
        return cls(**{k: v for k, v in kwargs.items() if k in valid_names})

    def __init__(
        self,
        width: int = 17,
        height: int = 13,
        num_shelf_blocks_x: int = 3,
        num_shelf_blocks_y: int = 2,
        shelf_block_w: int = 2,
        shelf_block_h: int = 3,
        num_pickup_stations: int = 3,
        num_dropoff_stations: int = 3,
        num_robots: int = 4,
        order_spawn_interval: int = 4,
        max_queue_size: int = 20,
        view_radius: int = 3,
        max_episode_steps: int = 300,
        deadlock_wait_threshold: int = 2,
        enable_veto: bool = True,
        enable_deadlock_reversal: bool = True,
        seed: int | None = None,
        team_reward_share: float = 0.0,
        physical_collision_distance_m: float = 0.40,
        shield_physical_walls_only: bool = False,
        shield_active_resolution: bool = False,
        shield_safety_radius_m: float | None = None,
        shield_pickup_radius_m: float | None = None,
    ):
        """team_reward_share pools that fraction of pickup/delivery rewards across all robots;
        physical_collision_distance_m is the deep-overlap distance charged -30;
        shield_physical_walls_only makes the shield ignore the map's shelf blocks (they only exist
        logically -- A* still routes around them) and check only the perimeter and other robots;
        shield_active_resolution turns on the shield's active conflict resolution (stopped robots keep
        turning, the lower-priority robot backs away, working deadlock reversal) -- meant for when the
        shield's velocities are executed; shield_safety_radius_m / shield_pickup_radius_m override the
        shield's robot-robot distance (open floor / around a robot picking up), None = defaults.
        """
        if not 0.0 <= team_reward_share <= 1.0:
            raise ValueError(f"team_reward_share must be in [0, 1], got {team_reward_share}")
        self.team_reward_share = float(team_reward_share)
        self.physical_collision_distance_m = float(physical_collision_distance_m)
        self.shield_physical_walls_only = bool(shield_physical_walls_only)
        self.shield_active_resolution = bool(shield_active_resolution)
        self.shield_safety_radius_m = shield_safety_radius_m
        self.shield_pickup_radius_m = shield_pickup_radius_m
        self._deep_pairs: set[tuple[int, int]] = set()
        self.last_contact_ids: set[int] = set()
        self.pose_provider = None
        self.last_policy_controlled_ids: set[int] = set()
        self._map_kwargs = dict(
            width=width,
            height=height,
            num_shelf_blocks_x=num_shelf_blocks_x,
            num_shelf_blocks_y=num_shelf_blocks_y,
            shelf_block_w=shelf_block_w,
            shelf_block_h=shelf_block_h,
            num_pickup_stations=num_pickup_stations,
            num_dropoff_stations=num_dropoff_stations,
        )
        self.num_robots = num_robots
        self.order_spawn_interval = order_spawn_interval
        self.max_queue_size = max_queue_size
        self.view_radius = view_radius
        self.max_episode_steps = max_episode_steps
        self.deadlock_wait_threshold = deadlock_wait_threshold
        self.enable_veto = enable_veto
        self.enable_deadlock_reversal = enable_deadlock_reversal
        self.tick_dt_s = DT_SECONDS

        self.possible_agents = [f"robot_{i}" for i in range(num_robots)]
        self.agents = list(self.possible_agents)

        self._rng = np.random.default_rng(seed)
        self.map: WarehouseMap = generate_warehouse_map(**self._map_kwargs)
        self.shield_map = self._build_shield_map()
        self.robots: dict[int, Robot] = {}
        self.task_manager: TaskManager | None = None
        self._step_count = 0
        self.last_intervention_events: set[int] = set()
        self.last_collision_events: set[int] = set()
        self.last_delivery_events: set[int] = set()
        self._pending_physical_collision: set[int] = set()
        self._pending_termination: set[int] = set()
        self.hold_robot_ids: set[int] = set()
        self.last_wall_block_events: set[int] = set()
        self.last_veto_events: set[int] = set()
        self.last_reversal_events: set[int] = set()
        self._stuck_ticks: dict[int, int] = {}
        self.last_stuck_events: set[int] = set()
        self._last_shield_intervened: dict[int, float] = {}
        self._last_v: dict[int, float] = {}
        self._last_omega: dict[int, float] = {}
        self.last_final_velocities: dict[int, tuple[float, float]] = {}

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return spaces.Dict(
            {
                "lidar": spaces.Box(low=0.0, high=1.0, shape=(NUM_LIDAR_RAYS,), dtype=np.float32),
                # vector: sin/cos of the heading error to the lookahead point, remaining path
                # length.
                "vector": spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32),
                # 1 if the shield overrode this robot last tick.
                "shield_intervened": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
                # Nearest robot: distance, relative x/y, relative speed.
                "nearest_robot": spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32),
                # Status one-hot, last executed (v, omega), measured motion.
                "agent_state": spaces.Box(low=-np.inf, high=np.inf, shape=(AGENT_STATE_DIM,), dtype=np.float32),
            }
        )

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

    def reset(self, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.map = generate_warehouse_map(**self._map_kwargs)
        self.shield_map = self._build_shield_map()
        self.agents = list(self.possible_agents)
        self._step_count = 0
        self._stuck_ticks = {}
        self._prev_omega_norm = {}
        self._pending_physical_collision = set()
        self._pending_termination = set()
        self.hold_robot_ids = set()
        self.last_policy_controlled_ids = set()
        self._last_motion = {}
        self._last_speed = {}
        self._deep_pairs = set()
        self.last_contact_ids = set()
        self.last_executed_velocities = {}
        self.last_stuck_events = set()
        self._last_shield_intervened = {}
        self._last_v = {}
        self._last_omega = {}
        self.last_final_velocities = {}

        free_cells = [c for c in self.map.free_cells() if self.map.grid[c] == 0]
        self._rng.shuffle(free_cells)
        start_cells = free_cells[: self.num_robots]

        self.robots = {}
        for i in range(self.num_robots):
            r = Robot(robot_id=i, pos=start_cells[i], status=RobotStatus.IDLE)
            r.x, r.y = cell_to_world(start_cells[i])
            r.theta = float(self._rng.uniform(-math.pi, math.pi))
            r.push_continuous_trail()
            self.robots[i] = r

        self.task_manager = TaskManager(
            pickup_stations=self.map.pickup_stations,
            dropoff_stations=self.map.dropoff_stations,
            order_spawn_interval=self.order_spawn_interval,
            max_queue_size=self.max_queue_size,
            rng=self._rng,
        )

        observations = {self._agent_name(i): self._get_obs(i) for i in self.robots}
        infos = {self._agent_name(i): {} for i in self.robots}
        return observations, infos

    def step(self, actions: dict[str, np.ndarray]):
        """Advance one tick: assign orders, run the shield, move robots (or read real poses), update
        paths and tasks, and compute rewards.
        """
        self._step_count += 1
        desired_velocities: dict[int, tuple[float, float]] = {
            i: (float(actions[self._agent_name(i)][0]) * V_MAX_MPS, float(actions[self._agent_name(i)][1]) * OMEGA_MAX_RADPS)
            for i in self.robots
        }

        # New orders and auction for idle robots.
        self.task_manager.step(self._step_count)
        idle_ids = [
            i for i, r in self.robots.items()
            if r.status == RobotStatus.IDLE and i not in self.hold_robot_ids
        ]
        for i in idle_ids:
            self.robots[i].idle_steps += 1
        occupied_cells = frozenset(r.pos for r in self.robots.values())
        claimed_pickups = frozenset(
            r.order.pickup
            for r in self.robots.values()
            if r.order is not None and r.status in (RobotStatus.TO_ITEM, RobotStatus.PICKING_UP)
        ) | (occupied_cells & frozenset(self.task_manager.pickup_stations))
        assignments = run_auction(
            self.task_manager, idle_ids, {i: self.robots[i].pos for i in idle_ids}, claimed_pickups,
            idle_steps={i: self.robots[i].idle_steps for i in idle_ids},
        )
        for robot_id, order in assignments.items():
            robot = self.robots[robot_id]
            robot.order = order
            robot.status = RobotStatus.TO_ITEM
            robot.idle_steps = 0
            robot.path = self._route(robot.pos, order.pickup)

        # Policy drives TO_ITEM/CARRYING robots; others use the autopilot or hold still.
        self.last_policy_controlled_ids = {
            i for i, r in self.robots.items()
            if r.status in (RobotStatus.TO_ITEM, RobotStatus.CARRYING) and i not in self.hold_robot_ids
        }
        for i, r in self.robots.items():
            if i in self.hold_robot_ids:
                desired_velocities[i] = (0.0, 0.0)
            elif r.status == RobotStatus.IDLE:
                target = cell_to_world(r.path[0]) if r.path else (r.x, r.y)
                desired_velocities[i] = _autopilot_velocity(r, target)
            elif r.status == RobotStatus.PICKING_UP:
                desired_velocities[i] = (0.0, 0.0)

        pre_potential = {i: -_remaining_path_distance((r.x, r.y), r.path) for i, r in self.robots.items()}
        pre_had_path = {i: bool(r.path) for i, r in self.robots.items()}
        intended_move = {
            i: abs(v) > _MOVE_EPS or abs(omega) > _MOVE_EPS for i, (v, omega) in desired_velocities.items()
        }

        # Shield prediction (its velocities are executed only with execute_shield).
        shield_result = apply_shield_continuous(
            robots=list(self.robots.values()),
            desired_velocities=desired_velocities,
            warehouse_map=self.shield_map,
            dt=self.tick_dt_s,
            enable_veto=self.enable_veto,
            enable_deadlock_reversal=self.enable_deadlock_reversal,
            deadlock_wait_threshold=self.deadlock_wait_threshold,
            active_resolution=self.shield_active_resolution,
            safety_radius_m=self.shield_safety_radius_m,
            pickup_radius_m=self.shield_pickup_radius_m,
        )
        self.last_intervention_events = shield_result.intervention_events
        self.last_collision_events = shield_result.collision_events
        self.last_wall_block_events = shield_result.wall_block_events
        self.last_veto_events = shield_result.veto_events
        self.last_reversal_events = shield_result.reversal_events
        self.last_final_velocities = shield_result.final_velocities
        self._last_v = {i: v for i, (v, _omega) in shield_result.final_velocities.items()}
        self._last_omega = {i: omega for i, (_v, omega) in shield_result.final_velocities.items()}
        overridden_this_tick = (
            shield_result.reward_penalty_events | shield_result.reversal_events
        )
        self._last_shield_intervened = {
            i: (1.0 if i in overridden_this_tick else 0.0) for i in self.robots
        }
        self.last_delivery_events = set()
        self.last_stuck_events = set()
        self.last_reward_breakdown = {
            "time_cost": 0.0, "collision": 0.0, "progress_pos": 0.0, "progress_neg": 0.0,
            "idle_wait": 0.0, "proximity": 0.0, "pickup": 0.0, "delivery": 0.0, "omega_penalty": 0.0,
            "stuck_turn": 0.0, "unstuck_bonus": 0.0, "wall_proximity": 0.0, "omega_reversal": 0.0,
            "physical_collision": 0.0, "contact": 0.0,
        }
        self.last_pickup_latency_steps: list[int] = []
        self.last_delivery_latency_steps: list[int] = []

        robot_moved: dict[int, bool] = {}
        pre_continuous_pos = {i: (r.x, r.y) for i, r in self.robots.items()}
        path_for_potential: dict[int, list[tuple[int, int]]] = {}
        # Real poses from Webots when a pose_provider is attached.
        final_poses = shield_result.final_poses
        if self.pose_provider is not None:
            self.last_executed_velocities = dict(shield_result.final_velocities)
            observed = self.pose_provider(dict(shield_result.final_velocities))
            final_poses = {i: observed.get(i, final_poses[i]) for i in final_poses}
        self._last_motion = {
            i: (
                math.hypot(final_poses[i][0] - r.x, final_poses[i][1] - r.y),
                _wrap_angle(final_poses[i][2] - r.theta),
            )
            for i, r in self.robots.items()
        }
        self._last_speed = {
            i: (
                (final_poses[i][0] - r.x) * math.cos(r.theta) + (final_poses[i][1] - r.y) * math.sin(r.theta)
            ) / max(self.tick_dt_s, 1e-6)
            for i, r in self.robots.items()
        }
        for i, r in self.robots.items():
            # Apply the new pose and keep the A* path in sync.
            x, y, theta = final_poses[i]
            moved = math.hypot(x - r.x, y - r.y) > _MOVE_EPS
            robot_moved[i] = moved
            r.x, r.y, r.theta = x, y, theta
            if moved:
                r.push_continuous_trail()
            r.sync_pos_from_continuous()
            path_for_potential[i] = r.path
            if r.path:
                if r.pos == r.path[0]:
                    r.path.pop(0)
                elif i in shield_result.intervention_events:
                    goal = r.path[-1]
                    soft_blocked = {
                        other.pos: 5.0
                        for j, other in self.robots.items()
                        if j != i and other.blocked_steps > 0
                    }
                    r.path = self._route(r.pos, goal, soft_blocked)
            if not r.path and r.order is not None and r.status in (RobotStatus.TO_ITEM, RobotStatus.CARRYING):
                goal = r.order.pickup if r.status == RobotStatus.TO_ITEM else r.order.dropoff
                r.path = self._route(r.pos, goal)

        # Per-robot rewards.
        brk = self.last_reward_breakdown
        rewards = {i: 0.0 for i in self.robots}
        task_reward = {i: 0.0 for i in self.robots}
        for i, r in self.robots.items():
            if intended_move[i]:
                rewards[i] += REWARD_TIME_COST_CONTINUOUS
                brk["time_cost"] += REWARD_TIME_COST_CONTINUOUS

            if i in shield_result.reward_penalty_events:
                rewards[i] += REWARD_COLLISION_CONTINUOUS
                brk["collision"] += REWARD_COLLISION_CONTINUOUS

            # Progress along the remaining path.
            progressed = False
            if r.status != RobotStatus.IDLE and pre_had_path[i]:
                post_potential = -_remaining_path_distance((r.x, r.y), path_for_potential[i])
                delta_dist = post_potential - pre_potential[i]
                pbrs_reward = REWARD_PROGRESS_SCALE_CONTINUOUS * (_PBRS_GAMMA * post_potential - pre_potential[i])
                if delta_dist > _MOVE_EPS:
                    progressed = True
                    rewards[i] += pbrs_reward
                    brk["progress_pos"] += pbrs_reward
                elif delta_dist < -_MOVE_EPS:
                    rewards[i] += pbrs_reward
                    brk["progress_neg"] += pbrs_reward

            if r.status in (RobotStatus.TO_ITEM, RobotStatus.CARRYING):
                v_norm = float(actions[self._agent_name(i)][0])
                omega_norm = float(actions[self._agent_name(i)][1])

                # Smoothness: omega rate, large omega, turning in place near obstacles.
                prev_omega_norm = self._prev_omega_norm.get(i)
                if prev_omega_norm is not None:
                    rate_pen = REWARD_OMEGA_RATE_CONTINUOUS * (omega_norm - prev_omega_norm) ** 2
                    rewards[i] += rate_pen
                    brk["omega_reversal"] += rate_pen
                self._prev_omega_norm[i] = omega_norm

                nearest_obstacle = self._nearest_obstacle_distance(
                    i, max(_OMEGA_PENALTY_CLEARANCE_M, _STUCK_TURN_CLEARANCE_M)
                )
                if nearest_obstacle >= _OMEGA_PENALTY_CLEARANCE_M:
                    omega_pen = REWARD_OMEGA_PENALTY_CONTINUOUS * max(0.0, abs(omega_norm) - _OMEGA_PENALTY_DEADZONE) ** 2
                    rewards[i] += omega_pen
                    brk["omega_penalty"] += omega_pen

                is_stuck_turn = (
                    nearest_obstacle < _STUCK_TURN_CLEARANCE_M
                    and abs(v_norm) > _STUCK_TURN_V_THRESHOLD
                    and abs(omega_norm) > _STUCK_TURN_OMEGA_THRESHOLD
                    and not progressed
                )
                if is_stuck_turn:
                    rewards[i] += REWARD_STUCK_TURN_CONTINUOUS
                    brk["stuck_turn"] += REWARD_STUCK_TURN_CONTINUOUS
                    self._stuck_ticks[i] = self._stuck_ticks.get(i, 0) + 1
                    if self._stuck_ticks[i] >= _STUCK_TICKS_THRESHOLD:
                        self.last_stuck_events.add(i)
                else:
                    was_stuck = self._stuck_ticks.get(i, 0) >= _STUCK_TICKS_THRESHOLD
                    if was_stuck and progressed and nearest_obstacle >= _STUCK_TURN_CLEARANCE_M:
                        rewards[i] += REWARD_UNSTUCK_BONUS_CONTINUOUS
                        brk["unstuck_bonus"] += REWARD_UNSTUCK_BONUS_CONTINUOUS
                    self._stuck_ticks[i] = 0

                # Wall proximity.
                nearest_wall = self._nearest_wall_distance(i, _WALL_PROXIMITY_THRESHOLD_M)
                wall_prox_pen = _wall_proximity_penalty(
                    nearest_wall, _WALL_PROXIMITY_THRESHOLD_M, REWARD_WALL_PROXIMITY_SCALE_CONTINUOUS
                )
                rewards[i] += wall_prox_pen
                brk["wall_proximity"] += wall_prox_pen
            else:
                self._stuck_ticks[i] = 0
                self._prev_omega_norm.pop(i, None)

            # Standing still without being blocked.
            if not intended_move[i] and r.status != RobotStatus.IDLE:
                blocked_ahead = i in shield_result.intervention_events or i in shield_result.reward_penalty_events
                if not blocked_ahead:
                    rewards[i] += REWARD_IDLE_WAIT
                    brk["idle_wait"] += REWARD_IDLE_WAIT

            prox = _graded_proximity_penalty(self._nearest_other_distance(i))
            rewards[i] += prox
            brk["proximity"] += prox

            # Pickup / delivery, checked along this tick's swept path.
            pickup_arrived = (
                r.status == RobotStatus.TO_ITEM
                and r.order is not None
                and self._segment_min_distance(
                    pre_continuous_pos[i], (r.x, r.y), cell_to_world(r.order.pickup)
                )
                <= _ARRIVAL_SWEEP_TOLERANCE_M
            )
            dropoff_arrived = (
                r.status == RobotStatus.CARRYING
                and r.order is not None
                and self._segment_min_distance(
                    pre_continuous_pos[i], (r.x, r.y), cell_to_world(r.order.dropoff)
                )
                <= _ARRIVAL_SWEEP_TOLERANCE_M
            )
            if pickup_arrived:
                r.pos = r.order.pickup
                rewards[i] += REWARD_PICKUP_CONTINUOUS
                task_reward[i] += REWARD_PICKUP_CONTINUOUS
                brk["pickup"] += REWARD_PICKUP_CONTINUOUS
                r.order.picked_up_step = self._step_count
                self.last_pickup_latency_steps.append(self._step_count - r.order.created_step)
                r.status = RobotStatus.PICKING_UP
            elif r.status == RobotStatus.PICKING_UP:
                r.status = RobotStatus.CARRYING
                r.path = self._route(r.pos, r.order.dropoff)
            elif dropoff_arrived:
                r.pos = r.order.dropoff
                rewards[i] += REWARD_DELIVERY_CONTINUOUS
                task_reward[i] += REWARD_DELIVERY_CONTINUOUS
                brk["delivery"] += REWARD_DELIVERY_CONTINUOUS
                self.task_manager.notify_delivered(r.order.dropoff)
                self.last_delivery_events.add(i)
                if r.order.picked_up_step is not None:
                    self.last_delivery_latency_steps.append(self._step_count - r.order.picked_up_step)
                r.status = RobotStatus.IDLE
                r.order = None
                r.path = []
                # Clear the station: send the idle robot to a free aisle cell.
                if self.map.grid[r.pos] in (PICKUP, DROPOFF):
                    occupied = {other.pos for j, other in self.robots.items() if j != i}
                    occupied |= {
                        other.path[-1] for j, other in self.robots.items()
                        if j != i and other.path
                    }
                    clearance_positions = [(other.x, other.y) for j, other in self.robots.items() if j != i]
                    clearance_positions += [
                        cell_to_world(other.path[-1]) for j, other in self.robots.items()
                        if j != i and other.path
                    ]
                    target = self._nearest_free_aisle_cell(
                        r.pos, occupied,
                        clearance_world_positions=clearance_positions, min_clearance_m=SAFETY_RADIUS_M,
                    )
                    if target is not None:
                        r.path = self._route(r.pos, target)

        # Contact penalty every tick and -30 once per new deep overlap.
        contact_ids: set[int] = set()
        deep_now: set[tuple[int, int]] = set()
        robot_ids = list(self.robots)
        for a_idx in range(len(robot_ids)):
            for b_idx in range(a_idx + 1, len(robot_ids)):
                a_id, b_id = robot_ids[a_idx], robot_ids[b_idx]
                ra, rb = self.robots[a_id], self.robots[b_id]
                d = math.hypot(ra.x - rb.x, ra.y - rb.y)
                if d < CONTACT_DISTANCE_M:
                    contact_ids.update((a_id, b_id))
                if d < self.physical_collision_distance_m:
                    deep_now.add((a_id, b_id))
        for i in contact_ids:
            rewards[i] += REWARD_CONTACT_CONTINUOUS
            brk["contact"] += REWARD_CONTACT_CONTINUOUS
        for pair in deep_now - self._deep_pairs:
            for i in pair:
                rewards[i] += REWARD_PHYSICAL_COLLISION_CONTINUOUS
                brk["physical_collision"] += REWARD_PHYSICAL_COLLISION_CONTINUOUS
        self._deep_pairs = deep_now
        self.last_contact_ids = contact_ids
        # Share part of the task rewards with the whole team.
        if self.team_reward_share > 0.0 and self.robots:
            team_pool = self.team_reward_share * sum(task_reward.values())
            for i in self.robots:
                rewards[i] += team_pool / len(self.robots) - self.team_reward_share * task_reward[i]
        truncated = self._step_count >= self.max_episode_steps
        observations = {self._agent_name(i): self._get_obs(i) for i in self.robots}
        rewards_out = {self._agent_name(i): rewards[i] for i in self.robots}
        terminations = {self._agent_name(i): False for i in self.robots}
        # Robots respawned by the coordinator: end their episode.
        for i in self._pending_physical_collision:
            rewards_out[self._agent_name(i)] += REWARD_PHYSICAL_COLLISION_CONTINUOUS
            brk["physical_collision"] += REWARD_PHYSICAL_COLLISION_CONTINUOUS
            terminations[self._agent_name(i)] = True
        for i in self._pending_termination:
            terminations[self._agent_name(i)] = True
        self._pending_physical_collision = set()
        self._pending_termination = set()
        truncations = {self._agent_name(i): truncated for i in self.robots}
        infos = {self._agent_name(i): {} for i in self.robots}
        if truncated:
            self.agents = []

        return observations, rewards_out, terminations, truncations, infos

    def _build_shield_map(self) -> WarehouseMap:
        """Map used by the shield: the env map, or (shield_physical_walls_only) the same map with
        the shelf blocks turned into free floor."""
        if not self.shield_physical_walls_only:
            return self.map
        grid = self.map.grid.copy()
        grid[grid == SHELF] = AISLE
        return WarehouseMap(grid, list(self.map.pickup_stations), list(self.map.dropoff_stations))

    @staticmethod
    def _agent_name(robot_id: int) -> str:
        return f"robot_{robot_id}"

    @staticmethod
    def _dist_to(robot: Robot, xy: tuple[float, float]) -> float:
        return math.hypot(robot.x - xy[0], robot.y - xy[1])

    @staticmethod
    def _segment_min_distance(
        p0: tuple[float, float], p1: tuple[float, float], target: tuple[float, float]
    ) -> float:
        """Minimum distance from target to the segment p0 -> p1."""
        x0, y0 = p0
        x1, y1 = p1
        tx, ty = target
        dx, dy = x1 - x0, y1 - y0
        seg_len_sq = dx * dx + dy * dy
        if seg_len_sq < 1e-12:
            return math.hypot(tx - x0, ty - y0)
        t = max(0.0, min(1.0, ((tx - x0) * dx + (ty - y0) * dy) / seg_len_sq))
        return math.hypot(tx - (x0 + t * dx), ty - (y0 + t * dy))

    def _nearest_obstacle_distance(self, robot_id: int, max_range: float) -> float:
        """Nearest wall or robot within max_range (lidar raycast)."""
        robot = self.robots[robot_id]
        other_positions = [(r.x, r.y) for j, r in self.robots.items() if j != robot_id]
        return min(
            self._raycast(robot.x, robot.y, robot.theta + k * (2 * math.pi / NUM_LIDAR_RAYS), max_range, other_positions)
            for k in range(NUM_LIDAR_RAYS)
        )

    def _nearest_wall_distance(self, robot_id: int, max_range: float) -> float:
        """Nearest wall within max_range (robots ignored)."""
        robot = self.robots[robot_id]
        return min(
            self._raycast(robot.x, robot.y, robot.theta + k * (2 * math.pi / NUM_LIDAR_RAYS), max_range, [])
            for k in range(NUM_LIDAR_RAYS)
        )

    def _nearest_other_distance(self, robot_id: int) -> float:
        """Distance to the closest other robot."""
        robot = self.robots[robot_id]
        others = [(r.x, r.y) for j, r in self.robots.items() if j != robot_id]
        if not others:
            return float("inf")
        return min(math.hypot(robot.x - ox, robot.y - oy) for ox, oy in others)

    def _nearest_robot_features(self, robot_id: int) -> tuple[float, float, float, float]:
        """(distance, relative x, relative y, relative speed) of the nearest robot, normalized."""
        robot = self.robots[robot_id]
        max_range = max(1, self.view_radius) * CELL_SIZE_M
        nearest_id, nearest_dist = None, float("inf")
        for j, other in self.robots.items():
            if j == robot_id:
                continue
            dist = math.hypot(robot.x - other.x, robot.y - other.y)
            if dist < nearest_dist:
                nearest_id, nearest_dist = j, dist
        if nearest_id is None:
            return (1.0, 0.0, 0.0, 0.0)
        other = self.robots[nearest_id]
        dx, dy = other.x - robot.x, other.y - robot.y
        cos_t, sin_t = math.cos(robot.theta), math.sin(robot.theta)
        rel_x = dx * cos_t + dy * sin_t
        rel_y = -dx * sin_t + dy * cos_t
        rel_v = self._last_speed.get(nearest_id, 0.0) - self._last_speed.get(robot_id, 0.0)
        return (
            float(np.clip(nearest_dist / max_range, 0.0, 1.0)),
            float(np.clip(rel_x / max_range, -1.0, 1.0)),
            float(np.clip(rel_y / max_range, -1.0, 1.0)),
            float(np.clip(rel_v / (2.0 * V_MAX_MPS), -1.0, 1.0)),
        )

    def _route(
        self, start: tuple[int, int], goal: tuple[int, int],
        soft_blocked: dict[tuple[int, int], float] | None = None,
    ) -> list[tuple[int, int]]:
        """A* waypoints from start to goal (start excluded), starting from a walkable cell."""
        origin = start if self.map.is_walkable(start) else self._nearest_walkable_cell(start)
        if origin is None:
            return []
        path = astar(self.map, origin, goal, soft_blocked)
        if not path:
            return []
        return path[1:] if origin == start else path

    def _nearest_walkable_cell(self, start: tuple[int, int]) -> tuple[int, int] | None:
        """Closest walkable cell to a cell that may be inside a shelf."""
        from collections import deque

        visited = {start}
        queue = deque([start])
        while queue:
            cur = queue.popleft()
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)):
                nxt = (cur[0] + dr, cur[1] + dc)
                if nxt in visited or not self.map.in_bounds(nxt):
                    continue
                if self.map.is_walkable(nxt):
                    return nxt
                visited.add(nxt)
                queue.append(nxt)
        return None

    def respawn_robot(self, robot_id: int, penalize: bool = True) -> tuple[int, int] | None:
        """Teleport one robot to a free aisle cell after a real collision (rest of the fleet keeps
        going). penalize=True charges -30 on the next step; otherwise only its episode ends.
        """
        robot = self.robots[robot_id]
        others = [r for j, r in self.robots.items() if j != robot_id]
        occupied = {r.pos for r in others} | {r.path[-1] for r in others if r.path}
        clearance = [(r.x, r.y) for r in others] + [cell_to_world(r.path[-1]) for r in others if r.path]
        cell = self._nearest_free_aisle_cell(
            robot.pos, occupied, clearance_world_positions=clearance, min_clearance_m=_RESPAWN_CLEARANCE_M,
        )
        if cell is None:
            return None
        if robot.order is not None:
            robot.order.picked_up_step = None
            self.task_manager.pending_orders.insert(0, robot.order)
        robot.order = None
        robot.status = RobotStatus.IDLE
        robot.path = []
        robot.pos = cell
        robot.x, robot.y = cell_to_world(cell)
        robot.blocked_steps = 0
        robot.reversal_cooldown = 0
        robot.idle_steps = 0
        for per_robot in (
            self._stuck_ticks, self._prev_omega_norm, self._last_shield_intervened, self._last_v,
            self._last_omega, self._last_motion, self.last_final_velocities, self._last_speed,
        ):
            per_robot.pop(robot_id, None)
        if penalize:
            self._pending_physical_collision.add(robot_id)
        else:
            self._pending_termination.add(robot_id)
        self.hold_robot_ids.add(robot_id)
        return cell

    def _nearest_free_aisle_cell(
        self,
        start: tuple[int, int],
        occupied: set[tuple[int, int]],
        clearance_world_positions: list[tuple[float, float]] | None = None,
        min_clearance_m: float = 0.0,
    ):
        """Closest free aisle cell, at least min_clearance_m from the given positions."""
        from collections import deque

        visited = {start}
        queue = deque([start])
        while queue:
            cur = queue.popleft()
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nxt = (cur[0] + dr, cur[1] + dc)
                if nxt in visited or not self.map.is_walkable(nxt):
                    continue
                visited.add(nxt)
                if self.map.grid[nxt] not in (PICKUP, DROPOFF) and nxt not in occupied:
                    if clearance_world_positions and min_clearance_m > 0:
                        wx, wy = cell_to_world(nxt)
                        if any(
                            math.hypot(wx - ox, wy - oy) < min_clearance_m
                            for ox, oy in clearance_world_positions
                        ):
                            queue.append(nxt)
                            continue
                    return nxt
                queue.append(nxt)
        return None

    def _raycast(self, x: float, y: float, angle: float, max_range: float, other_positions: list[tuple[float, float]]) -> float:
        """Distance along a ray to the first shelf/out-of-bounds cell or robot."""
        dx, dy = math.cos(angle), math.sin(angle)
        dist = 0.0
        while dist < max_range:
            dist += _RAY_STEP_M
            px, py = x + dx * dist, y + dy * dist
            cell = world_to_cell(px, py)
            if not self.map.in_bounds(cell) or self.map.grid[cell] == SHELF:
                return dist
            for ox, oy in other_positions:
                if math.hypot(px - ox, py - oy) <= ROBOT_RADIUS_M:
                    return dist
        return max_range

    def _get_obs(self, robot_id: int) -> dict:
        """Observation dict of one robot."""
        robot = self.robots[robot_id]
        other_positions = [(r.x, r.y) for j, r in self.robots.items() if j != robot_id]
        max_range = max(1, self.view_radius) * CELL_SIZE_M

        # Lidar distances normalized by max_range.
        lidar = np.zeros(NUM_LIDAR_RAYS, dtype=np.float32)
        for k in range(NUM_LIDAR_RAYS):
            ray_angle = robot.theta + k * (2 * math.pi / NUM_LIDAR_RAYS)
            lidar[k] = self._raycast(robot.x, robot.y, ray_angle, max_range, other_positions) / max_range

        # Direction and distance to the goal along the path.
        vector = np.zeros(3, dtype=np.float32)
        if robot.path:
            wx, wy = _lookahead_point((robot.x, robot.y), robot.path, _LOOKAHEAD_DISTANCE_M)
            dx, dy = wx - robot.x, wy - robot.y
            rel_angle = _wrap_angle(math.atan2(dy, dx) - robot.theta)
            total_dist = _remaining_path_distance((robot.x, robot.y), robot.path)
            vector[0] = math.sin(rel_angle)
            vector[1] = math.cos(rel_angle)
            vector[2] = float(np.clip(total_dist / (self.map.width + self.map.height), 0.0, 1.0))

        shield_intervened = np.array(
            [self._last_shield_intervened.get(robot_id, 0.0)], dtype=np.float32
        )
        nearest_robot = np.array(self._nearest_robot_features(robot_id), dtype=np.float32)
        return {
            "lidar": lidar, "vector": vector, "shield_intervened": shield_intervened,
            "nearest_robot": nearest_robot, "agent_state": self._agent_state(robot_id),
        }

    def _agent_state(self, robot_id: int) -> np.ndarray:
        """Own state: status one-hot (4), last executed (v, omega) normalized (2), measured motion of
        the last tick (2).
        """
        robot = self.robots[robot_id]
        state = np.zeros(AGENT_STATE_DIM, dtype=np.float32)
        state[_STATUS_INDEX[robot.status]] = 1.0
        v, omega = self.last_executed_velocities.get(
            robot_id, self.last_final_velocities.get(robot_id, (0.0, 0.0))
        )
        state[4] = v / V_MAX_MPS
        state[5] = omega / OMEGA_MAX_RADPS
        dist, dtheta = self._last_motion.get(robot_id, (0.0, 0.0))
        dt = max(self.tick_dt_s, 1e-6)
        state[6] = dist / (V_MAX_MPS * dt)
        state[7] = dtheta / (OMEGA_MAX_RADPS * dt)
        return state
