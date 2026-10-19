"""Grid warehouse environment (PettingZoo ParallelEnv).

Each step: spawn orders, auction them to idle robots, apply the shield to the policy's actions, move
robots, handle pickups/deliveries and compute rewards.
"""
from __future__ import annotations

import functools
import math

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from hiverl.env.robot import Robot, RobotStatus
from hiverl.env.safety_shield import WAIT_ACTION, apply_shield
from hiverl.env.task_manager import TaskManager
from hiverl.env.warehouse_grid import (
    ACTION_DELTAS,
    DROPOFF,
    NUM_ACTIONS,
    PICKUP,
    SHELF,
    WarehouseMap,
    generate_warehouse_map,
    manhattan,
)
from hiverl.planners.astar import astar
from hiverl.planners.auctioneer import run_auction

_REVERSE_ACTION_LOOKUP = {v: k for k, v in ACTION_DELTAS.items()}

REWARD_DELIVERY = 10.0
REWARD_PICKUP = 3.0
REWARD_PROGRESS = 0.2
REWARD_TIME_COST = -0.05
REWARD_IDLE_WAIT = -0.5
REWARD_BUFFER_ZONE = -0.5
REWARD_COLLISION = -1.0

# Near-miss penalty tiers (distance threshold, penalty), tightest first.
REWARD_PROXIMITY_TIERS: tuple[tuple[float, float], ...] = (
    (1.0, -0.08),
    (1.5, -0.03),
)


def proximity_penalty(
    distance: float, tiers: tuple[tuple[float, float], ...] = REWARD_PROXIMITY_TIERS
) -> float:
    """Penalty of the tightest distance tier that applies (tiers ordered tightest first)."""
    for threshold, penalty in tiers:
        if distance <= threshold:
            return penalty
    return 0.0


LIDAR_DIRECTIONS = [
    (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1),
]
NUM_LIDAR_RAYS = len(LIDAR_DIRECTIONS)

DEFAULT_SLIP_PROB = 0.1
_PERPENDICULAR_ACTIONS = {0: (2, 3), 1: (2, 3), 2: (0, 1), 3: (0, 1)}


class WarehouseEnv(ParallelEnv):
    """Multi-robot grid warehouse with a shared discrete action space."""
    metadata = {"name": "hiverl_warehouse_v0"}

    @classmethod
    def from_config(cls, env_cfg: dict, **overrides) -> "WarehouseEnv":
        """Build from a configs/env_*.yaml dict plus overrides."""
        import inspect

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
        deadlock_wait_threshold: int = 1,
        enable_veto: bool = True,
        enable_deadlock_reversal: bool = True,
        enable_slip: bool = False,
        slip_prob: float = DEFAULT_SLIP_PROB,
        seed: int | None = None,
    ):
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
        self.enable_slip = enable_slip
        self.slip_prob = slip_prob

        self.possible_agents = [f"robot_{i}" for i in range(num_robots)]
        self.agents = list(self.possible_agents)

        self._rng = np.random.default_rng(seed)
        self.map: WarehouseMap = generate_warehouse_map(**self._map_kwargs)
        self.robots: dict[int, Robot] = {}
        self.task_manager: TaskManager | None = None
        self._step_count = 0
        self.last_intervention_events: set[int] = set()
        self.last_collision_events: set[int] = set()
        self.last_delivery_events: set[int] = set()
        self.last_wall_block_events: set[int] = set()

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return spaces.Dict(
            {
                "lidar": spaces.Box(low=0.0, high=1.0, shape=(NUM_LIDAR_RAYS,), dtype=np.float32),
                "vector": spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32),
                "action_mask": spaces.Box(low=0.0, high=1.0, shape=(NUM_ACTIONS,), dtype=np.float32),
            }
        )

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return spaces.Discrete(NUM_ACTIONS)

    def reset(self, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.map = generate_warehouse_map(**self._map_kwargs)
        self.agents = list(self.possible_agents)
        self._step_count = 0

        free_cells = [c for c in self.map.free_cells() if self.map.grid[c] == 0]
        self._rng.shuffle(free_cells)
        start_cells = free_cells[: self.num_robots]

        self.robots = {
            i: Robot(robot_id=i, pos=start_cells[i], status=RobotStatus.IDLE)
            for i in range(self.num_robots)
        }
        for r in self.robots.values():
            r.push_trail()

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

    def step(self, actions: dict[str, int]):
        self._step_count += 1
        desired_actions = {i: int(actions[self._agent_name(i)]) for i in self.robots}

        self.task_manager.step(self._step_count)
        idle_ids = [i for i, r in self.robots.items() if r.status == RobotStatus.IDLE]
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
            path = astar(self.map, robot.pos, order.pickup) or [robot.pos]
            robot.path = path[1:]

        for i, r in self.robots.items():
            if r.status == RobotStatus.IDLE:
                if r.path:
                    delta = (r.path[0][0] - r.pos[0], r.path[0][1] - r.pos[1])
                    desired_actions[i] = _REVERSE_ACTION_LOOKUP.get(delta, WAIT_ACTION)
                else:
                    desired_actions[i] = WAIT_ACTION
            elif r.status == RobotStatus.PICKING_UP:
                desired_actions[i] = WAIT_ACTION

        pre_waypoint_dist = {
            i: (manhattan(r.pos, r.next_waypoint) if r.next_waypoint else None)
            for i, r in self.robots.items()
        }

        physical_actions = self._apply_slip(desired_actions)

        shield_result = apply_shield(
            robots=list(self.robots.values()),
            desired_actions=physical_actions,
            warehouse_map=self.map,
            enable_veto=self.enable_veto,
            enable_deadlock_reversal=self.enable_deadlock_reversal,
            deadlock_wait_threshold=self.deadlock_wait_threshold,
        )
        self.last_intervention_events = shield_result.intervention_events
        self.last_collision_events = shield_result.collision_events
        self.last_wall_block_events = shield_result.wall_block_events
        self.last_delivery_events = set()

        robot_moved: dict[int, bool] = {}
        for i, r in self.robots.items():
            dr, dc = ACTION_DELTAS[shield_result.final_actions[i]]
            new_pos = (r.pos[0] + dr, r.pos[1] + dc)
            moved = new_pos != r.pos
            robot_moved[i] = moved
            if moved:
                r.pos = new_pos
                r.push_trail()
            if r.path and r.pos == r.path[0]:
                r.path.pop(0)

        rewards = {i: 0.0 for i in self.robots}
        for i, r in self.robots.items():
            if desired_actions[i] != WAIT_ACTION:
                rewards[i] += REWARD_TIME_COST

            if i in shield_result.reward_penalty_events:
                rewards[i] += REWARD_COLLISION

            if r.status != RobotStatus.IDLE and pre_waypoint_dist[i] is not None:
                new_dist = manhattan(r.pos, r.next_waypoint) if r.next_waypoint else 0
                if new_dist < pre_waypoint_dist[i]:
                    rewards[i] += REWARD_PROGRESS
                elif new_dist > pre_waypoint_dist[i]:
                    rewards[i] -= REWARD_PROGRESS

            if desired_actions[i] == WAIT_ACTION and r.status != RobotStatus.IDLE:
                blocked_ahead = i in shield_result.intervention_events or i in shield_result.reward_penalty_events
                if not blocked_ahead:
                    rewards[i] += REWARD_IDLE_WAIT

            if self._stationary_neighbor(i, robot_moved):
                rewards[i] += REWARD_BUFFER_ZONE

            rewards[i] += proximity_penalty(self._nearest_other_distance(i))

            if r.status == RobotStatus.TO_ITEM and r.order is not None and r.pos == r.order.pickup:
                rewards[i] += REWARD_PICKUP
                r.status = RobotStatus.PICKING_UP
            elif r.status == RobotStatus.PICKING_UP:
                r.status = RobotStatus.CARRYING
                path = astar(self.map, r.pos, r.order.dropoff) or [r.pos]
                r.path = path[1:]
            elif r.status == RobotStatus.CARRYING and r.order is not None and r.pos == r.order.dropoff:
                rewards[i] += REWARD_DELIVERY
                self.task_manager.notify_delivered(r.order.dropoff)
                self.last_delivery_events.add(i)
                r.status = RobotStatus.IDLE
                r.order = None
                r.path = []
                if self.map.grid[r.pos] in (PICKUP, DROPOFF):
                    occupied = {other.pos for j, other in self.robots.items() if j != i}
                    target = self._nearest_free_aisle_cell(r.pos, occupied)
                    if target is not None:
                        path = astar(self.map, r.pos, target) or [r.pos]
                        r.path = path[1:]

        truncated = self._step_count >= self.max_episode_steps
        observations = {self._agent_name(i): self._get_obs(i) for i in self.robots}
        rewards_out = {self._agent_name(i): rewards[i] for i in self.robots}
        terminations = {self._agent_name(i): False for i in self.robots}
        truncations = {self._agent_name(i): truncated for i in self.robots}
        infos = {self._agent_name(i): {} for i in self.robots}
        if truncated:
            self.agents = []

        return observations, rewards_out, terminations, truncations, infos

    def render(self):
        from hiverl.utils.visualization import render_ascii

        return render_ascii(self)

    @staticmethod
    def _agent_name(robot_id: int) -> str:
        return f"robot_{robot_id}"

    def _nearest_other_distance(self, robot_id: int) -> float:
        """Distance (cells) to the closest other robot."""
        pos = self.robots[robot_id].pos
        others = [r.pos for j, r in self.robots.items() if j != robot_id]
        if not others:
            return float("inf")
        return min(math.hypot(pos[0] - p[0], pos[1] - p[1]) for p in others)

    def _stationary_neighbor(self, robot_id: int, robot_moved: dict[int, bool]) -> bool:
        """True if the robot did not move and has an adjacent robot that also did not move."""
        if robot_moved.get(robot_id, True):
            return False
        pos = self.robots[robot_id].pos
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            npos = (pos[0] + dr, pos[1] + dc)
            for j, r in self.robots.items():
                if j != robot_id and r.pos == npos and not robot_moved.get(j, True):
                    return True
        return False

    def _apply_slip(self, desired_actions: dict[int, int]) -> dict[int, int]:
        """Randomly drop or deflect a move (domain randomization)."""
        if not self.enable_slip:
            return desired_actions
        slipped = dict(desired_actions)
        for i, action in desired_actions.items():
            if action == WAIT_ACTION:
                continue
            if self._rng.random() < self.slip_prob:
                if self._rng.random() < 0.5:
                    slipped[i] = WAIT_ACTION
                else:
                    slipped[i] = int(self._rng.choice(_PERPENDICULAR_ACTIONS[action]))
        return slipped

    def _nearest_free_aisle_cell(
        self, start: tuple[int, int], occupied: set[tuple[int, int]]
    ) -> tuple[int, int] | None:
        """Closest free aisle cell (used to clear idle robots off stations)."""
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
                    return nxt
                queue.append(nxt)
        return None

    def _get_obs(self, robot_id: int) -> dict:
        robot = self.robots[robot_id]
        other_positions = {r.pos for i, r in self.robots.items() if i != robot_id}

        max_range = max(1, self.view_radius)
        lidar = np.zeros(NUM_LIDAR_RAYS, dtype=np.float32)
        for k, (dr, dc) in enumerate(LIDAR_DIRECTIONS):
            steps = 0
            while steps < max_range:
                cell = (robot.pos[0] + dr * (steps + 1), robot.pos[1] + dc * (steps + 1))
                if not self.map.in_bounds(cell) or self.map.grid[cell] == SHELF or cell in other_positions:
                    break
                steps += 1
            lidar[k] = steps / max_range

        waypoint = robot.next_waypoint
        vector = np.zeros(3, dtype=np.float32)
        if waypoint is not None:
            dr = waypoint[0] - robot.pos[0]
            dc = waypoint[1] - robot.pos[1]
            dist = manhattan(robot.pos, waypoint)
            vector[0] = np.sign(dr)
            vector[1] = np.sign(dc)
            vector[2] = float(np.clip(dist / (self.map.width + self.map.height), 0.0, 1.0))

        action_mask = np.ones(NUM_ACTIONS, dtype=np.float32)
        for a, (dr, dc) in ACTION_DELTAS.items():
            if a == WAIT_ACTION:
                continue
            if not self.map.is_walkable((robot.pos[0] + dr, robot.pos[1] + dc)):
                action_mask[a] = 0.0

        return {"lidar": lidar, "vector": vector, "action_mask": action_mask}
