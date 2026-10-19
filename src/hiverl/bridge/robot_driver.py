"""RobotDriver interface used by the lockstep coordinator, plus an in-memory fake for tests."""
from __future__ import annotations

import math
from abc import ABC, abstractmethod


class RobotDriver(ABC):
    """Moves one fleet of robots and reports poses/arrivals."""

    @abstractmethod
    def reset(self, robot_id: int, cell: tuple[int, int]) -> None:
        """Teleport/initialize a robot at a cell (episode start)."""

    @abstractmethod
    def send_goal(self, robot_id: int, cell: tuple[int, int]) -> None:
        """Drive the robot toward a grid cell."""

    @abstractmethod
    def poll_arrived(self, robot_id: int) -> bool:
        """Non-blocking: has the robot reached its last goal cell?"""

    def send_velocity(self, robot_id: int, v: float, omega: float) -> None:
        """Send a continuous (v, omega) command (RL-controlled robots)."""

    def set_shield_blocked(self, robot_id: int, blocked: bool) -> None:
        """Flag sent with the next velocity: the shield stopped a robot that wanted to go forward."""

    def set_status(self, robot_id: int, status: int) -> None:
        """Report the robot's task status (drives the LED and pickup logic)."""

    def shutdown(self) -> None:
        """Release resources."""

    def spin(self) -> None:
        """Process pending incoming messages."""

    def get_pose(self, robot_id: int) -> tuple[float, float, float] | None:
        """Latest real (x, y, theta), or None if unavailable."""
        return None

    def arrival_diagnostic(self, robot_id: int) -> str:
        """Short text shown when an arrival times out."""
        return "arrival state unavailable"


class FakeRobotDriver(RobotDriver):
    """In-memory driver; a goal is reached after ticks_to_arrive polls."""

    def __init__(self, ticks_to_arrive: int = 0):
        self.ticks_to_arrive = ticks_to_arrive
        self.positions: dict[int, tuple[int, int]] = {}
        self._goals: dict[int, tuple[int, int]] = {}
        self._polls_since_goal: dict[int, int] = {}
        self.goal_history: list[tuple[int, tuple[int, int]]] = []
        self.statuses: dict[int, int] = {}
        self.poses: dict[int, tuple[float, float, float]] = {}
        self.velocity_history: list[tuple[int, float, float]] = []

    def reset(self, robot_id: int, cell: tuple[int, int]) -> None:
        self.positions[robot_id] = cell
        self._goals[robot_id] = cell
        self._polls_since_goal[robot_id] = self.ticks_to_arrive
        from hiverl.bridge.coords import cell_to_world

        x, y = cell_to_world(cell)
        self.poses[robot_id] = (x, y, 0.0)

    def send_goal(self, robot_id: int, cell: tuple[int, int]) -> None:
        self._goals[robot_id] = cell
        self._polls_since_goal[robot_id] = 0
        self.goal_history.append((robot_id, cell))

    def poll_arrived(self, robot_id: int) -> bool:
        self._polls_since_goal[robot_id] += 1
        if self._polls_since_goal[robot_id] >= self.ticks_to_arrive:
            self.positions[robot_id] = self._goals[robot_id]
            return True
        return False

    def set_status(self, robot_id: int, status: int) -> None:
        self.statuses[robot_id] = status

    def send_velocity(self, robot_id: int, v: float, omega: float) -> None:
        """Integrate the fake pose by one tick."""
        self.velocity_history.append((robot_id, v, omega))
        from hiverl.env.continuous_kinematics import DT_SECONDS

        x, y, theta = self.poses.get(robot_id, (0.0, 0.0, 0.0))
        omega_world = -omega
        new_theta = theta + omega_world * DT_SECONDS
        new_x = x + v * math.cos(theta) * DT_SECONDS
        new_y = y + v * math.sin(theta) * DT_SECONDS
        self.poses[robot_id] = (new_x, new_y, new_theta)

    def get_pose(self, robot_id: int) -> tuple[float, float, float] | None:
        return self.poses.get(robot_id)
