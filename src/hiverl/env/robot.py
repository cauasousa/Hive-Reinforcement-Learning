"""Robot state shared by the grid and continuous environments."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class RobotStatus(Enum):
    """Task status of a robot."""
    IDLE = "idle"
    TO_ITEM = "to_item"
    PICKING_UP = "picking_up"
    CARRYING = "carrying"


@dataclass
class Robot:
    """One robot: grid position, task, path, continuous pose and shield bookkeeping."""
    robot_id: int
    pos: tuple[int, int]
    status: RobotStatus = RobotStatus.IDLE
    path: list[tuple[int, int]] = field(default_factory=list)
    order: object | None = None
    trail: list[tuple[int, int]] = field(default_factory=list)
    blocked_steps: int = 0
    trail_maxlen: int = 8
    # Consecutive idle ticks without an order (used by the auction fairness rule).
    idle_steps: int = 0

    # Continuous pose (continuous env only).
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0
    continuous_trail: list[tuple[float, float]] = field(default_factory=list)
    # Ticks to stay still after a shield deadlock reversal.
    reversal_cooldown: int = 0

    def push_trail(self):
        self.trail.append(self.pos)
        if len(self.trail) > self.trail_maxlen:
            self.trail.pop(0)

    def sync_pos_from_continuous(self) -> None:
        """Set the grid cell from the continuous pose."""
        from hiverl.env.continuous_kinematics import world_to_cell

        self.pos = world_to_cell(self.x, self.y)

    def push_continuous_trail(self) -> None:
        self.continuous_trail.append((self.x, self.y))
        if len(self.continuous_trail) > self.trail_maxlen:
            self.continuous_trail.pop(0)

    @property
    def next_waypoint(self) -> tuple[int, int] | None:
        return self.path[0] if self.path else None

    def priority_key(self) -> tuple[int, int, int]:
        """Shield priority: carrying/picking up > empty, then closer to goal, then lower id."""
        carrying_flag = 1 if self.status in (RobotStatus.CARRYING, RobotStatus.PICKING_UP) else 0
        return (carrying_flag, -len(self.path), -self.robot_id)
