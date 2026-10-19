"""Bookkeeping for the Milestone 4 comparison runs (throughput, interventions, makespan)."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RunStats:
    steps: int = 0
    deliveries: int = 0
    collisions: int = 0
    shield_interventions: int = 0
    delivery_step_log: list[int] = field(default_factory=list)

    def record_step(self, env) -> None:
        self.steps += 1
        deliveries_this_step = len(env.last_delivery_events)
        self.deliveries += deliveries_this_step
        for _ in range(deliveries_this_step):
            self.delivery_step_log.append(self.steps)
        self.collisions += len(env.last_collision_events)
        self.shield_interventions += len(env.last_intervention_events)

    def throughput_per_100_steps(self) -> float:
        if self.steps == 0:
            return 0.0
        return 100.0 * self.deliveries / self.steps

    def makespan_for_n_orders(self, n: int) -> float | None:
        """Steps elapsed until the n-th delivery, or None if fewer than n deliveries occurred."""
        if len(self.delivery_step_log) < n:
            return None
        return float(self.delivery_step_log[n - 1])

    def as_dict(self) -> dict:
        return {
            "steps": self.steps,
            "deliveries": self.deliveries,
            "collisions": self.collisions,
            "shield_interventions": self.shield_interventions,
            "throughput_per_100_steps": self.throughput_per_100_steps(),
        }
