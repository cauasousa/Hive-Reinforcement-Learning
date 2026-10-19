"""Grid A* planner."""
import numpy as np

from hiverl.env.warehouse_grid import AISLE, SHELF, WarehouseMap
from hiverl.planners.astar import astar


def _map_from_ascii(rows: list[str]) -> WarehouseMap:
    grid = np.array([[SHELF if ch == "#" else AISLE for ch in row] for row in rows], dtype=np.int8)
    return WarehouseMap(grid=grid)


def test_astar_finds_direct_path_when_no_obstacles():
    m = _map_from_ascii(["....", "....", "...."])
    path = astar(m, (0, 0), (2, 3))
    assert path is not None
    assert path[0] == (0, 0)
    assert path[-1] == (2, 3)
    assert len(path) - 1 == 5


def test_astar_routes_around_a_wall():
    m = _map_from_ascii(
        [
            ".....",
            ".###.",
            ".....",
        ]
    )
    path = astar(m, (0, 0), (2, 0))
    assert path is not None
    for cell in path:
        assert m.grid[cell] != SHELF


def test_astar_returns_none_when_unreachable():
    m = _map_from_ascii(
        [
            ".#.",
            ".#.",
            ".#.",
        ]
    )
    assert astar(m, (0, 0), (0, 2)) is None


def test_astar_start_equals_goal():
    m = _map_from_ascii(["..", ".."])
    assert astar(m, (0, 0), (0, 0)) == [(0, 0)]
