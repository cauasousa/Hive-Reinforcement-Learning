"""Milestone 2: train a single robot with PPO on static-obstacle navigation only.

No other robots are present, so this run proves the PPO implementation itself learns to reach
its A*-guided waypoints -- it says nothing yet about multi-robot coordination (Milestone 3).

Usage:
    python scripts/train_single_agent.py --steps 20000
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import yaml

from hiverl.agents.ppo import PPOConfig, train_ppo
from hiverl.env.warehouse_env import WarehouseEnv
from hiverl.utils.visualization import plot_training_curve

ROOT = os.path.join(os.path.dirname(__file__), "..")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-config", default=os.path.join(ROOT, "configs", "env_small.yaml"))
    parser.add_argument("--train-config", default=os.path.join(ROOT, "configs", "train_config.yaml"))
    parser.add_argument("--steps", type=int, default=None, help="override total_timesteps")
    parser.add_argument("--seed", type=int, default=None, help="override the config seed")
    parser.add_argument("--output-dir", default=os.path.join(ROOT, "outputs"))
    parser.add_argument(
        "--device", default="auto",
        help="'auto' (cuda if available, else cpu), or an explicit torch device string",
    )
    parser.add_argument("--resume", default=None, help="resume from a checkpoint saved by a previous run")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.env_config) as f:
        env_cfg = yaml.safe_load(f)
    with open(args.train_config) as f:
        train_cfg_dict = yaml.safe_load(f)
    if args.steps is not None:
        train_cfg_dict["total_timesteps"] = args.steps
    if args.seed is not None:
        train_cfg_dict["seed"] = args.seed
        env_cfg["seed"] = args.seed

    env = WarehouseEnv.from_config(env_cfg, num_robots=1, enable_veto=False, enable_deadlock_reversal=False)

    config = PPOConfig.from_dict(train_cfg_dict)
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    def log_fn(stats):
        print(
            f"update {stats['update']:4d} | reward {stats['mean_episode_reward']:.2f} | "
            f"deliveries {stats['deliveries']:3d} | policy_loss {stats['policy_loss']:.4f} | "
            f"value_loss {stats['value_loss']:.4f} | entropy {stats['entropy']:.3f}"
        )

    checkpoint_path = os.path.join(args.output_dir, "single_agent_checkpoint.pt")
    history_csv_path = os.path.join(args.output_dir, "single_agent_training_history.csv")
    _, history = train_ppo(
        env,
        config,
        checkpoint_path=checkpoint_path,
        log_fn=log_fn,
        device=device,
        resume_path=args.resume,
        checkpoint_interval=25,
        history_csv_path=history_csv_path,
    )

    plot_training_curve(
        history, os.path.join(args.output_dir, "single_agent_training_curve.png"), title="Milestone 2: single-robot PPO"
    )
    print(f"\nSaved checkpoint to {checkpoint_path}")
    print(f"Saved training history to {history_csv_path}")
    print(f"Saved training curve to {args.output_dir}/single_agent_training_curve.png")


if __name__ == "__main__":
    main()
