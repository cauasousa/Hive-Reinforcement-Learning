"""Lifelong order queue of pickup -> dropoff jobs."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Order:
    """One pickup -> dropoff job."""
    order_id: int
    pickup: tuple[int, int]
    dropoff: tuple[int, int]
    created_step: int
    # Tick when the order was picked up (for latency metrics).
    picked_up_step: int | None = None


class TaskManager:
    """Spawns orders on a fixed cadence, bounded by max_queue_size."""
    def __init__(
        self,
        pickup_stations: list[tuple[int, int]],
        dropoff_stations: list[tuple[int, int]],
        order_spawn_interval: int,
        max_queue_size: int,
        rng: np.random.Generator,
    ):
        self.pickup_stations = pickup_stations
        self.dropoff_stations = dropoff_stations
        self.order_spawn_interval = order_spawn_interval
        self.max_queue_size = max_queue_size
        self.rng = rng
        self.pending_orders: list[Order] = []
        self._next_order_id = 0
        self.total_spawned = 0
        self.total_delivered = 0
        self._dropoff_load: dict[tuple[int, int], int] = {d: 0 for d in dropoff_stations}

    def reset(self):
        self.pending_orders = []
        self._next_order_id = 0
        self.total_spawned = 0
        self.total_delivered = 0
        self._dropoff_load = {d: 0 for d in self.dropoff_stations}

    def step(self, current_step: int):
        """Spawn a new order if it is time and there is room."""
        if current_step % self.order_spawn_interval != 0:
            return
        pending_per_station = {p: 0 for p in self.pickup_stations}
        for o in self.pending_orders:
            pending_per_station[o.pickup] = pending_per_station.get(o.pickup, 0) + 1
        min_pending = min(pending_per_station.values())
        if len(self.pending_orders) >= self.max_queue_size and min_pending > 0:
            return
        least_pending = [p for p in self.pickup_stations if pending_per_station[p] == min_pending]
        pickup = least_pending[self.rng.integers(len(least_pending))]
        min_load = min(self._dropoff_load.values())
        least_loaded = [d for d in self.dropoff_stations if self._dropoff_load[d] == min_load]
        dropoff = least_loaded[self.rng.integers(len(least_loaded))]
        order = Order(self._next_order_id, pickup, dropoff, current_step)
        self.pending_orders.append(order)
        self._next_order_id += 1
        self.total_spawned += 1
        self._dropoff_load[dropoff] += 1

    def pop_order(self, order: Order):
        self.pending_orders.remove(order)

    def notify_delivered(self, dropoff: tuple[int, int]):
        self.total_delivered += 1
        self._dropoff_load[dropoff] -= 1
