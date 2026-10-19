"""Constants and unicycle kinematics shared by the continuous env and its shield."""
from __future__ import annotations

import math

CELL_SIZE_M = 1.0

# Duration of one env tick (full speed covers about one cell).
DT_SECONDS = 1.0
V_MAX_MPS = 1.0
OMEGA_MAX_RADPS = (math.pi)/2

ROBOT_RADIUS_M = 0.3
# Shield distance thresholds (center to center): open floor, 1-wide corridor, around a robot picking
# up.
SAFETY_RADIUS_M = 2 * ROBOT_RADIUS_M + 0.2
CORRIDOR_SAFETY_RADIUS_M = CELL_SIZE_M + 0.2
PICKUP_EXCLUSION_RADIUS_M = CELL_SIZE_M + ROBOT_RADIUS_M
ARRIVAL_TOLERANCE_M = 0.2
# Max distance at which two robots can still conflict within one tick.
LOCAL_SENSING_RANGE_M = 2 * V_MAX_MPS * DT_SECONDS + max(
    SAFETY_RADIUS_M, CORRIDOR_SAFETY_RADIUS_M, PICKUP_EXCLUSION_RADIUS_M
)


def clip_velocity(v: float, omega: float) -> tuple[float, float]:
    """Clamp (v, omega) to the robot limits."""
    return (
        max(-V_MAX_MPS, min(V_MAX_MPS, v)),
        max(-OMEGA_MAX_RADPS, min(OMEGA_MAX_RADPS, omega)),
    )


def integrate_unicycle(
    x: float, y: float, theta: float, v: float, omega: float, dt: float = DT_SECONDS
) -> tuple[float, float, float]:
    """Pose after moving with (v, omega) for dt seconds (theta=0 faces +x / increasing column)."""
    new_theta = theta + omega * dt
    if abs(omega) < 1e-5:
        new_x = x + v * math.cos(theta) * dt
        new_y = y + v * math.sin(theta) * dt
    else:
        radius = v / omega
        new_x = x + radius * (math.sin(new_theta) - math.sin(theta))
        new_y = y - radius * (math.cos(new_theta) - math.cos(theta))
    return new_x, new_y, new_theta


def _round_half_away_from_zero(v: float) -> int:
    """Round .5 away from zero (Python's round() is half-to-even)."""
    return math.floor(v + 0.5) if v >= 0 else math.ceil(v - 0.5)


def world_to_cell(x: float, y: float) -> tuple[int, int]:
    """Nearest grid cell (row, col) of a continuous position."""
    return (_round_half_away_from_zero(y / CELL_SIZE_M), _round_half_away_from_zero(x / CELL_SIZE_M))


def cell_to_world(cell: tuple[int, int]) -> tuple[float, float]:
    """Continuous position of a cell center."""
    row, col = cell
    return (col * CELL_SIZE_M, row * CELL_SIZE_M)
