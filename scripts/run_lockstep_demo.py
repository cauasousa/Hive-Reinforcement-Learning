#!/usr/bin/env python3
"""Run the lock-step coordinator end to end.

With ``--driver fake`` (the default) this needs nothing beyond the existing Python environment --
no Webots, no ROS2 -- and is the way to actually exercise and verify the lock-step wait/dispatch
protocol described in docs/webots_ros2_integration.md before ever touching a real simulator.

With ``--driver ros2`` it drives real Webots robots over ROS2 (see docs/run_with_webots_ros2.md
for the required setup); this is the same coordinator code path, just swapping which
``RobotDriver`` it talks to.

Usage:
    # scripted A* baseline, fake driver with a simulated 3-tick travel delay per goal
    python scripts/run_lockstep_demo.py --policy astar --ticks 100 --driver fake --fake-delay 3

    # trained MAPPO checkpoint, fake driver, no delay (fastest way to sanity-check the loop)
    python scripts/run_lockstep_demo.py --policy checkpoint --checkpoint outputs/mappo_checkpoint.pt --ticks 200

    # real Webots robots over ROS2 (see docs/run_with_webots_ros2.md)
    python scripts/run_lockstep_demo.py --policy checkpoint --checkpoint outputs/mappo_checkpoint.pt --driver ros2

    # continuous (v, omega) action space -- fake driver, no Webots needed for this smoke test
    python scripts/run_lockstep_demo.py --continuous-actions --policy checkpoint \\
        --checkpoint outputs/mappo_checkpoint_continuous.pt --ticks 50
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import yaml

from hiverl.agents.checkpoint import load_checkpoint
from hiverl.agents.networks import ActorCriticNet, ContinuousActorCriticNet
from hiverl.bridge.lockstep_coordinator import ContinuousLockstepCoordinator, LockstepCoordinator, TickResult
from hiverl.bridge.robot_driver import FakeRobotDriver
from hiverl.env.warehouse_env import WarehouseEnv
from hiverl.env.warehouse_env_continuous import ContinuousWarehouseEnv

ROOT = os.path.join(os.path.dirname(__file__), "..")


def _load_checkpoint_policy(checkpoint_path: str, env: WarehouseEnv, hidden_dim: int, temperature: float):
    import numpy as np

    obs_space = env.observation_space(env.possible_agents[0])
    net = ActorCriticNet(
        lidar_dim=obs_space["lidar"].shape[-1],
        vector_dim=obs_space["vector"].shape[-1],
        hidden_dim=hidden_dim,
        num_actions=env.action_space(env.possible_agents[0]).n,
    )
    load_checkpoint(net, checkpoint_path, device="cpu")
    net.eval()

    def policy(obs_dict: dict) -> dict:
        agents = env.possible_agents
        lidar = torch.tensor(np.stack([obs_dict[a]["lidar"] for a in agents]), dtype=torch.float32)
        vector = torch.tensor(np.stack([obs_dict[a]["vector"] for a in agents]), dtype=torch.float32)
        mask = torch.tensor(np.stack([obs_dict[a]["action_mask"] for a in agents]), dtype=torch.float32)
        with torch.no_grad():
            logits = net.logits(lidar, vector, mask)
            actions = torch.distributions.Categorical(logits=logits / temperature).sample()
        return {a: int(actions[i].item()) for i, a in enumerate(agents)}

    return policy


def _load_continuous_checkpoint_policy(checkpoint_path: str, env: ContinuousWarehouseEnv, hidden_dim: int):
    import numpy as np

    obs_space = env.observation_space(env.possible_agents[0])
    net = ContinuousActorCriticNet(
        lidar_dim=obs_space["lidar"].shape[-1],
        vector_dim=obs_space["vector"].shape[-1],
        hidden_dim=hidden_dim,
        action_dim=env.action_space(env.possible_agents[0]).shape[0],
    )
    load_checkpoint(net, checkpoint_path, device="cpu")
    net.eval()

    def policy(obs_dict: dict) -> dict:
        agents = env.possible_agents
        lidar = torch.tensor(np.stack([obs_dict[a]["lidar"] for a in agents]), dtype=torch.float32)
        vector = torch.tensor(np.stack([obs_dict[a]["vector"] for a in agents]), dtype=torch.float32)
        with torch.no_grad():
            actions, _, _, _ = net.act(lidar, vector)
        return {a: actions[i].numpy() for i, a in enumerate(agents)}

    return policy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-config", default=os.path.join(ROOT, "configs", "env_small.yaml"))
    parser.add_argument("--train-config", default=os.path.join(ROOT, "configs", "train_config.yaml"))
    parser.add_argument("--policy", choices=["astar", "checkpoint"], default="astar")
    parser.add_argument("--checkpoint", default=os.path.join(ROOT, "outputs", "mappo_checkpoint.pt"))
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--ticks", type=int, default=100)
    parser.add_argument("--driver", choices=["fake", "ros2"], default="fake")
    parser.add_argument(
        "--fake-delay", type=int, default=0,
        help="ticks_to_arrive for FakeRobotDriver -- simulates physical travel time per goal",
    )
    parser.add_argument("--arrival-timeout", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--robots", type=int, default=None,
        help="override num_robots from the env config -- MUST match the --robots value used when "
        "the Webots world was generated (scripts/generate_webots_world.py --robots N), or with "
        "--driver ros2 any Webots robot beyond this count silently never receives a single "
        "command and sits motionless forever (it's simply never subscribed to).",
    )
    parser.add_argument(
        "--continuous-actions", action="store_true",
        help="drive ContinuousWarehouseEnv + a ContinuousActorCriticNet checkpoint over "
        "send_velocity/cmd_vel instead of the discrete grid path -- see "
        "ContinuousLockstepCoordinator. Requires --policy checkpoint (no scripted-A* equivalent "
        "for continuous actions).",
    )
    args = parser.parse_args()

    if args.continuous_actions and args.policy != "checkpoint":
        parser.error("--continuous-actions requires --policy checkpoint")

    with open(args.env_config) as f:
        env_cfg = yaml.safe_load(f)
    with open(args.train_config) as f:
        train_cfg = yaml.safe_load(f)
    if args.robots is not None:
        env_cfg["num_robots"] = args.robots

    env_cls = ContinuousWarehouseEnv if args.continuous_actions else WarehouseEnv
    env = env_cls.from_config(
        env_cfg,
        max_episode_steps=10**9,
        seed=args.seed if args.seed is not None else env_cfg["seed"],
    )

    if args.continuous_actions:
        policy_fn = _load_continuous_checkpoint_policy(args.checkpoint, env, train_cfg["hidden_dim"])
    elif args.policy == "checkpoint":
        policy_fn = _load_checkpoint_policy(args.checkpoint, env, train_cfg["hidden_dim"], args.temperature)
    else:
        policy_fn = None

    if args.driver == "fake":
        driver = FakeRobotDriver(ticks_to_arrive=args.fake_delay)
    else:
        from hiverl.bridge.ros2_driver import Ros2RobotDriver

        driver = Ros2RobotDriver(num_robots=env_cfg["num_robots"])

    def _on_tick(tick: int, result: TickResult) -> None:
        if not result.arrived_in_time:
            print(f"[tick {tick}] WARNING: arrival timeout -- a robot never reached its goal cell")
        if result.deliveries or result.collisions:
            print(
                f"[tick {tick}] deliveries={result.deliveries} collisions={result.collisions} "
                f"interventions={result.interventions}"
            )

    coordinator_cls = ContinuousLockstepCoordinator if args.continuous_actions else LockstepCoordinator
    coordinator = coordinator_cls(
        env, driver, arrival_timeout_s=args.arrival_timeout, on_tick=_on_tick,
        synchronize_fleet=args.driver == "ros2",
    )
    stats = coordinator.run(args.ticks, policy_fn)
    driver.shutdown()

    print(
        f"\n{args.ticks} lock-step ticks complete: "
        f"deliveries={stats.total_deliveries} collisions={stats.total_collisions} "
        f"interventions={stats.total_interventions} timed_out_ticks={stats.timed_out_ticks}"
    )


if __name__ == "__main__":
    main()
