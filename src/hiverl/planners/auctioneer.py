"""Greedy order assignment: nearest idle robot wins, with a fairness rule for long-idle robots."""
from __future__ import annotations

from hiverl.env.task_manager import Order, TaskManager
from hiverl.env.warehouse_grid import manhattan

# Idle ticks after which a robot gets priority in the auction.
FAIRNESS_IDLE_THRESHOLD = 20


def run_auction(
    task_manager: TaskManager,
    idle_robot_ids: list[int],
    robot_positions: dict[int, tuple[int, int]],
    claimed_pickups: frozenset[tuple[int, int]] = frozenset(),
    idle_steps: dict[int, int] | None = None,
) -> dict[int, Order]:
    """Assign at most one pending order per idle robot; returns {robot_id: Order}."""
    assignments: dict[int, Order] = {}
    available_robots = list(idle_robot_ids)
    taken_pickups = set(claimed_pickups)
    idle_steps = idle_steps or {}

    for order in list(task_manager.pending_orders):
        if not available_robots:
            break
        if order.pickup in taken_pickups:
            continue
        starving = [rid for rid in available_robots if idle_steps.get(rid, 0) >= FAIRNESS_IDLE_THRESHOLD]
        if starving:
            best_robot = max(starving, key=lambda rid: (idle_steps.get(rid, 0), -manhattan(robot_positions[rid], order.pickup)))
        else:
            best_robot = min(
                available_robots,
                key=lambda rid: manhattan(robot_positions[rid], order.pickup),
            )
        assignments[best_robot] = order
        available_robots.remove(best_robot)
        task_manager.pop_order(order)
        taken_pickups.add(order.pickup)

    return assignments
