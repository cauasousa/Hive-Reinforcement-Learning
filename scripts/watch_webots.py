#!/usr/bin/env python3
"""Watch a trained continuous MAPPO checkpoint drive the Webots fleet (inference only, the checkpoint
is never modified).

Start the controllers first (scripts/run_extern_controllers.sh), then run:

    python scripts/watch_webots.py --checkpoint outputs/<run>_best.pt --robots 5
        --ticks 700 --velocity-tick-sleep-s 0.3 --disable-legacy-retreats
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch
import yaml

from hiverl.agents.checkpoint import load_checkpoint
from hiverl.agents.networks import ContinuousActorCriticNet, migrate_architecture_change
from hiverl.agents.ppo import PPOConfig
from hiverl.bridge.lockstep_coordinator import ContinuousLockstepCoordinator, TickResult
from hiverl.env.warehouse_env_continuous import ContinuousWarehouseEnv

ROOT = os.path.join(os.path.dirname(__file__), "..")


def main() -> None:
    """Load the checkpoint and run rounds until Ctrl+C (or --rounds)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="checkpoint to load (read-only -- never modified)")
    parser.add_argument("--env-config", default=os.path.join(ROOT, "configs", "env_webots.yaml"))
    parser.add_argument("--train-config", default=os.path.join(ROOT, "configs", "train_config.yaml"))
    parser.add_argument(
        "--robots", type=int, default=None,
        help="MUST match --robots used when the Webots world was generated (see "
        "run_lockstep_demo.py's matching note)",
    )
    parser.add_argument("--arrival-timeout", type=float, default=60.0)
    parser.add_argument("--ticks", type=int, default=1000, help="real Webots ticks per round before a physical reset")
    parser.add_argument(
        "--rounds", type=int, default=None,
        help="stop after this many rounds; default runs until Ctrl+C",
    )
    parser.add_argument(
        "--velocity-tick-sleep-s", type=float, default=None,
        help="see ContinuousLockstepCoordinator -- None keeps the default DT_SECONDS-locked pacing",
    )
    parser.add_argument(
        "--disable-legacy-retreats", action="store_true",
        help="same as scripts/train_mappo_webots.py's flag: turn off the coordinator's scripted "
        "proximity stop/retreat (forces (0, 0) for up to 10 ticks, then reverses) so what you see "
        "is the policy itself -- training runs with it disabled",
    )
    parser.add_argument(
        "--execute-shield", action="store_true",
        help="same as scripts/train_mappo_webots.py's flag: send the env shield's corrected "
        "velocities to the real robots instead of the policy's raw action. Off by default, as in "
        "training -- the policy was never trained this way and run 67 froze most robots with it",
    )
    parser.add_argument(
        "--shield-radius", type=float, default=0.65,
        help="with --execute-shield: robot-robot distance (centers, m) at which the shield acts on "
        "open floor (training default 0.8; robots touch at ~0.47)",
    )
    parser.add_argument(
        "--shield-pickup-radius", type=float, default=0.9,
        help="with --execute-shield: shield distance around a robot that is picking up (training default 1.3)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--actor-hidden-dim", type=int, default=64,
        help="must match the value the checkpoint was TRAINED with (see "
        "train_mappo_webots.py's --actor-hidden-dim) -- separate actor/critic encoders "
        "(2026-09-13) means this can no longer be inferred from a single shared hidden_dim.",
    )
    parser.add_argument(
        "--critic-hidden-dim", type=int, default=256,
        help="must match the value the checkpoint was TRAINED with (see "
        "train_mappo_webots.py's --critic-hidden-dim).",
    )
    args = parser.parse_args()

    with open(args.env_config) as f:
        env_cfg = yaml.safe_load(f)
    with open(args.train_config) as f:
        train_cfg = yaml.safe_load(f)
    if args.robots is not None:
        env_cfg["num_robots"] = args.robots

    # With --execute-shield the shield really drives the robots: it checks only physical obstacles
    # (perimeter walls and robots, not the map's logical shelf blocks) and resolves conflicts actively.
    env = ContinuousWarehouseEnv.from_config(
        env_cfg, max_episode_steps=10**9, seed=args.seed,
        shield_physical_walls_only=args.execute_shield, shield_active_resolution=args.execute_shield,
        shield_safety_radius_m=args.shield_radius if args.execute_shield else None,
        shield_pickup_radius_m=args.shield_pickup_radius if args.execute_shield else None,
    )
    agents = env.possible_agents
    device = "cpu"
    torch.manual_seed(args.seed)

    cfg = PPOConfig.from_dict(train_cfg)
    obs_space = env.observation_space(agents[0])
    arch = (torch.load(args.checkpoint, map_location="cpu", weights_only=False).get("extra") or {}).get("arch") or {}
    net = ContinuousActorCriticNet(
        lidar_dim=obs_space["lidar"].shape[-1],
        vector_dim=obs_space["vector"].shape[-1],
        actor_hidden_dim=arch.get("actor_hidden_dim", args.actor_hidden_dim),
        critic_hidden_dim=arch.get("critic_hidden_dim", args.critic_hidden_dim),
        action_dim=env.action_space(agents[0]).shape[0],
        num_robots=len(agents),
        shield_flag_dim=obs_space["shield_intervened"].shape[-1],
        nearest_robot_dim=obs_space["nearest_robot"].shape[-1],
        agent_state_dim=arch.get("agent_state_dim", 0),
        soft_log_std=arch.get("soft_log_std", False),
        layer_norm=arch.get("layer_norm", False),
        obs_norm=arch.get("obs_norm", False),
    ).to(device)
    try:
        load_checkpoint(net, args.checkpoint, optimizer=None, device=device)
    except RuntimeError as e:
        print(
            f"Strict checkpoint load failed ({e!r}) -- falling back to a partial (shape-filtered) "
            f"load with function-preserving migration, same as scripts/train_mappo_webots.py.",
            flush=True,
        )
        meta = load_checkpoint(net, args.checkpoint, optimizer=None, device=device, strict=False)
        migrate_architecture_change(net, meta["skipped_tensors"], num_robots=len(agents))
    net.eval()
    print(f"Loaded {args.checkpoint} (read-only -- this script never saves or updates it)", flush=True)

    from hiverl.bridge.ros2_driver import Ros2RobotDriver

    driver = Ros2RobotDriver(num_robots=env_cfg["num_robots"])
    coordinator_kwargs = dict(
        arrival_timeout_s=args.arrival_timeout, synchronize_fleet=True,
        velocity_tick_sleep_s=args.velocity_tick_sleep_s,
        enable_legacy_retreats=not args.disable_legacy_retreats,
        execute_shield_velocities=args.execute_shield,
    )
    coordinator = ContinuousLockstepCoordinator(env, driver, **coordinator_kwargs)

    def policy_fn(obs_dict: dict) -> dict:
        lidar = torch.tensor(np.stack([obs_dict[a]["lidar"] for a in agents]), dtype=torch.float32)
        vector = torch.tensor(np.stack([obs_dict[a]["vector"] for a in agents]), dtype=torch.float32)
        shield = torch.tensor(np.stack([obs_dict[a]["shield_intervened"] for a in agents]), dtype=torch.float32)
        nearest_robot = torch.tensor(np.stack([obs_dict[a]["nearest_robot"] for a in agents]), dtype=torch.float32)
        extra = torch.cat([shield, nearest_robot], dim=-1)
        if net.agent_state_dim > 0:
            state = torch.tensor(np.stack([obs_dict[a]["agent_state"] for a in agents]), dtype=torch.float32)
            extra = torch.cat([extra, state], dim=-1)
        with torch.no_grad():
            actions, _, _, _ = net.act(lidar, vector, extra)
        return {a: actions[i].cpu().numpy() for i, a in enumerate(agents)}

    def on_tick(tick: int, result: TickResult) -> None:
        pos_str = ",".join(f"{i}={env.robots[i].pos}" for i in range(len(agents)))
        print(
            f"[round {round_idx} tick {tick}] deliveries={result.deliveries} "
            f"collisions={result.collisions} interventions={result.interventions} "
            f"pos={{{pos_str}}}",
            flush=True,
        )

    coordinator.on_tick = on_tick
    round_idx = 0
    try:
        while args.rounds is None or round_idx < args.rounds:
            round_idx += 1
            print(f"[round {round_idx}] running {args.ticks} ticks...", flush=True)
            coordinator.run(args.ticks, policy_fn, reset=True)
    except KeyboardInterrupt:
        print("\nInterrupted -- shutting down.", flush=True)
    finally:
        driver.shutdown()


if __name__ == "__main__":
    main()
