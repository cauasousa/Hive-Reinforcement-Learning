#!/usr/bin/env python3
"""Scripted (non-learned) baseline for the same real-Webots continuous-control task the MAPPO
policy is trained on -- answers "what does a dumb-but-competent controller achieve here?" so the
trained policy's deliveries/round can be read against a real ceiling instead of in a vacuum.

Every robot always drives straight at its current A* waypoint via the same proportional
controller (``_autopilot_velocity``) this codebase already uses for IDLE/PICKING_UP robots'
goal-cell movement -- here it's used for TO_ITEM/CARRYING robots too, instead of a policy_fn
sampling from a trained network. No learning, no exploration noise, no PPO -- purely "go straight
at the next waypoint, slow down if badly misaligned first". The Safety Shield (the same one the
trained policy is subject to) still applies unchanged.

If this baseline's deliveries/round is CLOSE to the trained policy's, most of the achievable
throughput in this environment is a property of the environment (task spawn rate, map layout,
fleet size, physical cycle time) -- the policy already has little headroom left, and the next
big win is scale/environment changes, not more RL tuning. If this baseline is MUCH WORSE, the
policy has real headroom the training loop is still capturing.

Requires ROS2 (rclpy) and a running Webots session with extern-controller robots already
listening -- same two-terminal setup as scripts/train_mappo_webots.py / scripts/watch_webots.py:

    ./scripts/run_extern_controllers.sh webots/worlds/hiverl_warehouse_extern.wbt.robots.json localhost 1234
    # in another WSL terminal, once the controllers above report "connected":
    python scripts/run_scripted_baseline.py --robots 5 --rounds 5
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import yaml

from hiverl.bridge.lockstep_coordinator import ContinuousLockstepCoordinator, TickResult
from hiverl.env.continuous_kinematics import OMEGA_MAX_RADPS, V_MAX_MPS, cell_to_world
from hiverl.env.warehouse_env_continuous import ContinuousWarehouseEnv, _autopilot_velocity

ROOT = os.path.join(os.path.dirname(__file__), "..")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-config", default=os.path.join(ROOT, "configs", "env_webots.yaml"))
    parser.add_argument(
        "--robots", type=int, default=None,
        help="MUST match --robots used when the Webots world was generated (see "
        "run_lockstep_demo.py's matching note)",
    )
    parser.add_argument("--arrival-timeout", type=float, default=60.0)
    parser.add_argument("--ticks", type=int, default=1000, help="real Webots ticks per round before a physical reset")
    parser.add_argument("--rounds", type=int, default=20, help="stop after this many rounds")
    parser.add_argument(
        "--velocity-tick-sleep-s", type=float, default=0.3,
        help="match whatever the trained-policy run being compared against used -- the physical "
        "cycle time this baseline is measured against must be the same pacing.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--quiet-ticks", action="store_true",
        help="skip the per-tick print (still prints one summary line per round) -- this baseline "
        "can run many ticks with nothing changing status-wise, and the per-tick log this script "
        "would otherwise produce is mostly noise for a baseline run.",
    )
    args = parser.parse_args()

    with open(args.env_config) as f:
        env_cfg = yaml.safe_load(f)
    if args.robots is not None:
        env_cfg["num_robots"] = args.robots

    env = ContinuousWarehouseEnv.from_config(env_cfg, max_episode_steps=10**9, seed=args.seed)
    agents = env.possible_agents

    from hiverl.bridge.ros2_driver import Ros2RobotDriver

    driver = Ros2RobotDriver(num_robots=env_cfg["num_robots"])
    coordinator = ContinuousLockstepCoordinator(
        env, driver, arrival_timeout_s=args.arrival_timeout, synchronize_fleet=True,
        velocity_tick_sleep_s=args.velocity_tick_sleep_s,
    )

    def policy_fn(obs_dict: dict) -> dict:
        actions = {}
        for i, agent in enumerate(agents):
            robot = env.robots[i]
            target = cell_to_world(robot.path[0]) if robot.path else (robot.x, robot.y)
            v, omega = _autopilot_velocity(robot, target)
            actions[agent] = np.array([v / V_MAX_MPS, omega / OMEGA_MAX_RADPS], dtype=np.float32)
        return actions

    round_totals: list[dict] = []

    def on_tick(tick: int, result: TickResult) -> None:
        if not args.quiet_ticks:
            pos_str = ",".join(f"{i}={env.robots[i].pos}" for i in range(len(agents)))
            print(
                f"[round {round_idx} tick {tick}] deliveries={result.deliveries} "
                f"collisions={result.collisions} interventions={result.interventions} "
                f"pos={{{pos_str}}}",
                flush=True,
            )
        totals = round_totals[-1]
        totals["deliveries"] += result.deliveries
        totals["collisions"] += result.collisions
        totals["interventions"] += result.interventions
        totals["wall_block"] += len(getattr(env, "last_wall_block_events", ()))

    coordinator.on_tick = on_tick
    round_idx = 0
    try:
        while round_idx < args.rounds:
            round_idx += 1
            round_totals.append({"deliveries": 0, "collisions": 0, "interventions": 0, "wall_block": 0})
            print(f"[round {round_idx}] running {args.ticks} ticks (scripted baseline)...", flush=True)
            stats = coordinator.run(args.ticks, policy_fn, reset=True)
            totals = round_totals[-1]
            print(
                f"[round {round_idx}] SCRIPTED BASELINE done on {len(stats.ticks)} real ticks: "
                f"deliveries={totals['deliveries']} collisions={totals['collisions']} "
                f"interventions={totals['interventions']} wall_block={totals['wall_block']} "
                f"(wall_block/{len(stats.ticks)}ticks="
                f"{totals['wall_block'] / max(1, len(stats.ticks)):.2f})", flush=True,
            )
    except KeyboardInterrupt:
        print("\nInterrupted -- shutting down.", flush=True)
    finally:
        driver.shutdown()

    if round_totals:
        deliveries = [r["deliveries"] for r in round_totals]
        wall_blocks = [r["wall_block"] for r in round_totals]
        print(
            f"\nScripted baseline summary over {len(round_totals)} round(s): "
            f"deliveries={deliveries} mean={sum(deliveries) / len(deliveries):.2f} "
            f"max={max(deliveries)}\n"
            f"wall_block={wall_blocks} mean={sum(wall_blocks) / len(wall_blocks):.2f} "
            f"zero_delivery_rounds={sum(1 for d in deliveries if d == 0)}/{len(deliveries)}",
            flush=True,
        )


if __name__ == "__main__":
    main()
