"""Grid cell <-> Webots world coordinates (1 cell = 1 m, world y = -row)."""
from __future__ import annotations

CELL_SIZE_M = 1.0


def cell_to_world(cell: tuple[int, int]) -> tuple[float, float]:
    """(row, col) -> (x, y) of the cell center in meters."""
    row, col = cell
    x = (col + 0.5) * CELL_SIZE_M
    y = -(row + 0.5) * CELL_SIZE_M
    return x, y


def world_to_cell(x: float, y: float) -> tuple[int, int]:
    """(x, y) in meters -> nearest (row, col)."""
    col = int(x // CELL_SIZE_M)
    row = int(-y // CELL_SIZE_M)
    return row, col


def arrived_at(x: float, y: float, cell: tuple[int, int], tolerance_m: float = 0.08) -> bool:
    """True if (x, y) is within tolerance_m of the cell center."""
    tx, ty = cell_to_world(cell)
    return abs(x - tx) <= tolerance_m and abs(y - ty) <= tolerance_m


def env_pose_to_world(x: float, y: float, theta: float) -> tuple[float, float, float]:
    """Env-frame pose -> Webots world pose."""
    return (x + 0.5) * CELL_SIZE_M, -(y + 0.5) * CELL_SIZE_M, -theta


def world_pose_to_env(x: float, y: float, theta: float) -> tuple[float, float, float]:
    """Webots world pose -> env-frame pose."""
    return x / CELL_SIZE_M - 0.5, -y / CELL_SIZE_M - 0.5, -theta
