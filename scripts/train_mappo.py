"""Milestone 3: train the shared-policy MAPPO team with the full Safety Shield enabled.

Usage:
    python scripts/train_mappo.py --steps 20000 --robots 4
    python scripts/train_mappo.py --steps 2000000 --checkpoint-interval 50   # long run, crash-safe
    python scripts/train_mappo.py --resume outputs/mappo_checkpoint.pt --steps 500000  # continue training
    python scripts/train_mappo.py --continuous-actions --steps 20000   # (v, omega) instead of grid moves
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
from hiverl.env.warehouse_env_continuous import ContinuousWarehouseEnv
from hiverl.utils.visualization import plot_training_curve

ROOT = os.path.join(os.path.dirname(__file__), "..")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-config", default=os.path.join(ROOT, "configs", "env_small.yaml"))
    parser.add_argument("--train-config", default=os.path.join(ROOT, "configs", "train_config.yaml"))
    parser.add_argument("--steps", type=int, default=None, help="override total_timesteps")
    parser.add_argument("--robots", type=int, default=None, help="override num_robots")
    parser.add_argument("--seed", type=int, default=None, help="override the config seed")
    parser.add_argument("--output-dir", default=os.path.join(ROOT, "outputs"))
    parser.add_argument(
        "--device", default="auto",
        help="'auto' (cuda if available, else cpu), or an explicit torch device string",
    )
    parser.add_argument("--resume", default=None, help="resume from a checkpoint saved by a previous run")
    parser.add_argument(
        "--checkpoint-interval", type=int, default=25,
        help="overwrite the checkpoint every N updates in addition to the final save "
        "(0 disables periodic checkpointing; still saves once at the end)",
    )
    parser.add_argument(
        "--no-best-checkpoint", action="store_true",
        help="skip tracking a separate best-mean-episode-reward checkpoint",
    )
    parser.add_argument(
        "--anneal-lr", action="store_true",
        help="linearly decay the learning rate to 0 over the run (off by default; see PPOConfig.anneal_lr)",
    )
    parser.add_argument(
        "--continuous-actions", action="store_true",
        help="train ContinuousWarehouseEnv + ContinuousActorCriticNet -- (v, omega) velocity "
        "commands instead of discrete grid moves (see warehouse_env_continuous.py). Produces "
        "separate checkpoint/curve/history files (a '_continuous' suffix), architecturally "
        "incompatible with the discrete checkpoints -- a clean run, not a fine-tune.",
    )
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

    num_robots = args.robots if args.robots is not None else env_cfg["num_robots"]
    if args.continuous_actions:
        env = ContinuousWarehouseEnv.from_config(
            env_cfg, num_robots=num_robots, enable_veto=True, enable_deadlock_reversal=True,
        )
    else:
        env = WarehouseEnv.from_config(
            env_cfg, num_robots=num_robots, enable_veto=True, enable_deadlock_reversal=True, enable_slip=True,
        )

    config = PPOConfig.from_dict(train_cfg_dict)
    if args.anneal_lr:
        config.anneal_lr = True
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    # Cumulative since THIS run/resume started (not the checkpoint's lifetime total) -- lets a
    # long run's overall delivery rate be read off directly instead of hand-summing thousands of
    # per-update lines, which is exactly the kind of scanning this diagnostics work is meant to
    # replace.
    cumulative = {"deliveries": 0, "updates": 0}

    def log_fn(stats):
        lr_str = f" | lr {stats['learning_rate']:.2e}" if config.anneal_lr else ""
        cumulative["deliveries"] += stats["deliveries"]
        cumulative["updates"] += 1
        print(
            f"update {stats['update']:4d} | reward {stats['mean_episode_reward']:.2f} | "
            f"deliveries {stats['deliveries']:3d} | shield_interventions {stats['shield_interventions']:3d} | "
            f"collisions {stats['collisions']:3d} | policy_loss {stats['policy_loss']:.4f} | "
            f"value_loss {stats['value_loss']:.4f} | entropy {stats['entropy']:.4f}{lr_str} | "
            f"cum_deliveries {cumulative['deliveries']:5d} "
            f"({cumulative['deliveries'] / cumulative['updates']:.2f}/update since resume)"
        )
        # Continuous-action-only diagnostics (see PPOTrainer.collect_rollout): which reward term
        # is actually driving the total, how the robot fleet's time is split across the task
        # state machine, how long orders take end to end, and whether the action distribution is
        # saturating (all 3 useful for spotting entropy collapse/exploration loss directly,
        # rather than inferring it only from the scalar `entropy` above).
        if args.continuous_actions:
            print(
                f"    reward breakdown: time_cost {stats['reward_time_cost']:7.2f} | "
                f"collision {stats['reward_collision']:8.2f} | progress+ {stats['reward_progress_pos']:6.2f} | "
                f"progress- {stats['reward_progress_neg']:7.2f} | idle_wait {stats['reward_idle_wait']:7.2f} | "
                f"proximity {stats['reward_proximity']:7.2f} | pickup {stats['reward_pickup']:6.2f} | "
                f"delivery {stats['reward_delivery']:6.2f} | omega_penalty {stats['reward_omega_penalty']:7.2f} | "
                f"stuck_turn {stats['reward_stuck_turn']:7.2f} | unstuck_bonus {stats['reward_unstuck_bonus']:6.2f}"
            )
            print(
                f"    robot-ticks by status: IDLE {stats['status_IDLE']:4d} | TO_ITEM {stats['status_TO_ITEM']:4d} | "
                f"PICKING_UP {stats['status_PICKING_UP']:4d} | CARRYING {stats['status_CARRYING']:4d} | "
                f"pickup_latency~{stats['pickup_latency_mean']:.0f} steps | delivery_latency~{stats['delivery_latency_mean']:.0f} steps"
            )
            # Split of `shield_interventions` (their union) -- veto (penalized, a robot repeatedly
            # proposing a blocked move) vs. reversal (unpenalized, the shield breaking a deadlock)
            # vs. wall_block (penalized, driving into a shelf/map edge). A rollout where veto+
            # wall_block alone approach robots*rollout_steps is total gridlock for the whole
            # episode, not ordinary congestion -- worth searching the CSV for specifically.
            print(
                f"    shield events: veto {stats['shield_veto']:4d} | reversal {stats['shield_reversal']:4d} | "
                f"wall_block {stats['shield_wall_block']:4d} | stuck_events {stats['stuck_events']:4d}"
            )
            print(
                f"    action stats: mean|v| {stats['action_v_mean_abs']:.3f} | mean|omega| {stats['action_omega_mean_abs']:.3f} | "
                f"saturated(|a|>0.95) {stats['action_saturation_frac'] * 100:.1f}%"
            )

    suffix = "_continuous" if args.continuous_actions else ""
    checkpoint_path = os.path.join(args.output_dir, f"mappo_checkpoint{suffix}.pt")
    best_checkpoint_path = None if args.no_best_checkpoint else os.path.join(
        args.output_dir, f"mappo_checkpoint{suffix}_best.pt"
    )
    history_csv_path = os.path.join(args.output_dir, f"mappo_training_history{suffix}.csv")
    curve_path = os.path.join(args.output_dir, f"mappo_training_curve{suffix}.png")

    print(f"Training on device={device}, total_timesteps={config.total_timesteps}, robots={env.num_robots}")
    _, history = train_ppo(
        env,
        config,
        checkpoint_path=checkpoint_path,
        log_fn=log_fn,
        device=device,
        resume_path=args.resume,
        checkpoint_interval=args.checkpoint_interval,
        history_csv_path=history_csv_path,
        best_checkpoint_path=best_checkpoint_path,
    )

    plot_training_curve(
        history, curve_path,
        title="Milestone 3: MAPPO + Safety Shield" + (" (continuous actions)" if args.continuous_actions else ""),
    )
    print(f"\nSaved checkpoint to {checkpoint_path}")
    if best_checkpoint_path:
        print(f"Saved best-reward checkpoint to {best_checkpoint_path}")
    print(f"Saved training history to {history_csv_path}")
    print(f"Saved training curve to {curve_path}")


if __name__ == "__main__":
    main()
