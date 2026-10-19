"""Warehouse grid (row 0 at the top) and procedural map generation."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

AISLE = 0
SHELF = 1
PICKUP = 2
DROPOFF = 3

ACTION_DELTAS = {
    0: (-1, 0),
    1: (1, 0),
    2: (0, -1),
    3: (0, 1),
    4: (0, 0),
}
NUM_ACTIONS = 5


@dataclass
class WarehouseMap:
    """Grid of cell codes plus pickup/dropoff station lists."""
    grid: np.ndarray
    pickup_stations: list[tuple[int, int]] = field(default_factory=list)
    dropoff_stations: list[tuple[int, int]] = field(default_factory=list)

    @property
    def height(self) -> int:
        return self.grid.shape[0]

    @property
    def width(self) -> int:
        return self.grid.shape[1]

    def in_bounds(self, pos: tuple[int, int]) -> bool:
        r, c = pos
        return 0 <= r < self.height and 0 <= c < self.width

    def is_walkable(self, pos: tuple[int, int]) -> bool:
        return self.in_bounds(pos) and self.grid[pos[0], pos[1]] != SHELF

    def free_cells(self) -> list[tuple[int, int]]:
        rows, cols = np.where(self.grid != SHELF)
        return list(zip(rows.tolist(), cols.tolist()))


def generate_warehouse_map(
    width: int,
    height: int,
    num_shelf_blocks_x: int,
    num_shelf_blocks_y: int,
    shelf_block_w: int,
    shelf_block_h: int,
    num_pickup_stations: int,
    num_dropoff_stations: int,
) -> WarehouseMap:
    """Build a symmetric aisle/shelf-block layout with pickup and dropoff columns."""
    grid = np.full((height, width), AISLE, dtype=np.int8)

    left_margin = right_margin = 2
    top_margin = bottom_margin = 1

    available_w = width - left_margin - right_margin
    available_h = height - top_margin - bottom_margin
    block_area_w = num_shelf_blocks_x * shelf_block_w + (num_shelf_blocks_x - 1)
    block_area_h = num_shelf_blocks_y * shelf_block_h + (num_shelf_blocks_y - 1)
    if block_area_w > available_w or block_area_h > available_h:
        raise ValueError("Map too small for the requested shelf block layout")

    start_x = left_margin + (available_w - block_area_w) // 2
    start_y = top_margin + (available_h - block_area_h) // 2

    for i in range(num_shelf_blocks_x):
        x0 = start_x + i * (shelf_block_w + 1)
        for j in range(num_shelf_blocks_y):
            y0 = start_y + j * (shelf_block_h + 1)
            grid[y0 : y0 + shelf_block_h, x0 : x0 + shelf_block_w] = SHELF

    pickup_rows = np.linspace(1, height - 2, num_pickup_stations).astype(int)
    dropoff_rows = np.linspace(1, height - 2, num_dropoff_stations).astype(int)

    pickup_stations = []
    for r in pickup_rows:
        grid[r, 0] = PICKUP
        pickup_stations.append((int(r), 0))

    dropoff_stations = []
    for r in dropoff_rows:
        grid[r, width - 1] = DROPOFF
        dropoff_stations.append((int(r), width - 1))

    return WarehouseMap(grid=grid, pickup_stations=pickup_stations, dropoff_stations=dropoff_stations)


def manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])
