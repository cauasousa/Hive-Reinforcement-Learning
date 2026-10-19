"""4-connected grid A* (static obstacles only)."""
from __future__ import annotations

import heapq

from hiverl.env.warehouse_grid import WarehouseMap, manhattan

_NEIGHBOR_DELTAS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def astar(
    warehouse_map: WarehouseMap,
    start: tuple[int, int],
    goal: tuple[int, int],
    soft_blocked: dict[tuple[int, int], float] | None = None,
) -> list[tuple[int, int]] | None:
    """Path from start to goal (inclusive) or None; soft_blocked adds extra cost to some cells."""
    soft_blocked = soft_blocked or {}
    if start == goal:
        return [start]
    if not warehouse_map.is_walkable(start) or not warehouse_map.is_walkable(goal):
        return None

    open_heap: list[tuple[int, int, tuple[int, int]]] = []
    counter = 0
    heapq.heappush(open_heap, (manhattan(start, goal), counter, start))
    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    g_score = {start: 0}
    closed: set[tuple[int, int]] = set()

    while open_heap:
        _, _, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        if current == goal:
            return _reconstruct_path(came_from, current)
        closed.add(current)

        for dr, dc in _NEIGHBOR_DELTAS:
            neighbor = (current[0] + dr, current[1] + dc)
            if not warehouse_map.is_walkable(neighbor) or neighbor in closed:
                continue
            tentative_g = g_score[current] + 1 + soft_blocked.get(neighbor, 0.0)
            if tentative_g < g_score.get(neighbor, float("inf")):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + manhattan(neighbor, goal)
                counter += 1
                heapq.heappush(open_heap, (f_score, counter, neighbor))

    return None


def _reconstruct_path(
    came_from: dict[tuple[int, int], tuple[int, int]], current: tuple[int, int]
) -> list[tuple[int, int]]:
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path
