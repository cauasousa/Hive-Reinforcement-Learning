"""Milestone 4: compare A* pure vs. A* + Shield(veto only) vs. the full hybrid across robot
densities, producing the throughput/collision/intervention plots used in the README.

Usage:
    python scripts/evaluate.py --densities 2 4 8 12 --eval-steps 400
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch
import yaml

from hiverl.agents.checkpoint import load_checkpoint
from hiverl.agents.networks import ActorCriticNet
from hiverl.agents.scripted_astar_agent import scripted_actions
from hiverl.env.warehouse_env import WarehouseEnv
from hiverl.utils.metrics import RunStats
from hiverl.utils.visualization import plot_density_comparison

ROOT = os.path.join(os.path.dirname(__file__), "..")


def _make_env(env_cfg: dict, num_robots: int, enable_veto: bool, enable_deadlock_reversal: bool) -> WarehouseEnv:
    return WarehouseEnv.from_config(
        env_cfg,
        num_robots=num_robots,
        max_episode_steps=10**9,
        enable_veto=enable_veto,
        enable_deadlock_reversal=enable_deadlock_reversal,
    )


def _load_mappo_policy(
    checkpoint_path: str, env: WarehouseEnv, hidden_dim: int, greedy: bool = False, temperature: float = 1.0
):
    """``greedy=False`` (default) samples from the policy's action distribution, matching how it
    was trained. Two independent findings say this default should stay, even for a well-trained
    policy -- not just an undertrained one:
      1. Early on, an undertrained policy's argmax got stuck oscillating between two near-tied
         actions even though the distribution put real probability mass on the correct move.
      2. Later, with a policy that had learned a genuinely good delivery rate, switching to greedy
         made things *worse* (deliveries 27->11, shield interventions 26->862 in one measured
         comparison), not better. Every robot shares one policy network, so a deterministic policy
         makes two robots in a symmetric situation (e.g. head-on in a corridor) pick the identical
         action every step -- they conflict, the shield blocks one, and next step they try the same
         losing move again, forever needing the shield to break the tie. Sampling lets identically-
         positioned robots randomly diverge and resolve it themselves.
    ``temperature`` < 1.0 sharpens the sampled distribution (more decisive, less back-and-forth
    between two similarly-good options) without going fully deterministic; push it too low and you
    approach the same symmetric-conflict problem as greedy.
    """
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
            if greedy:
                actions = torch.argmax(logits, dim=-1)
            else:
                logits = logits / temperature
                actions = torch.distributions.Categorical(logits=logits).sample()
        return {a: int(actions[i].item()) for i, a in enumerate(agents)}

    return policy


def run_scenario(env: WarehouseEnv, policy_fn, eval_steps: int) -> RunStats:
    obs, _ = env.reset()
    stats = RunStats()
    for _ in range(eval_steps):
        actions = policy_fn(obs) if policy_fn is not None else scripted_actions(env)
        obs, _, _, _, _ = env.step(actions)
        stats.record_step(env)
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-config", default=os.path.join(ROOT, "configs", "env_small.yaml"))
    parser.add_argument("--train-config", default=os.path.join(ROOT, "configs", "train_config.yaml"))
    parser.add_argument("--densities", type=int, nargs="+", default=[2, 4, 8, 12])
    parser.add_argument("--eval-steps", type=int, default=400)
    parser.add_argument("--mappo-checkpoint", default=os.path.join(ROOT, "outputs", "mappo_checkpoint.pt"))
    parser.add_argument("--greedy", action="store_true", help="argmax instead of sampling for the MAPPO policy")
    parser.add_argument(
        "--temperature", type=float, default=1.0,
        help="sharpen (<1.0) or flatten (>1.0) the sampled action distribution; ignored with --greedy",
    )
    parser.add_argument("--output-dir", default=os.path.join(ROOT, "outputs"))
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.env_config) as f:
        env_cfg = yaml.safe_load(f)
    with open(args.train_config) as f:
        train_cfg = yaml.safe_load(f)

    have_mappo = os.path.exists(args.mappo_checkpoint)
    if not have_mappo:
        print(f"[warn] no MAPPO checkpoint found at {args.mappo_checkpoint}; skipping the full-hybrid scenario.")

    scenario_defs = [
        ("A* pure", False, False, None),
        ("A* + Shield (veto only)", True, False, None),
    ]
    if have_mappo:
        scenario_defs.append(("Full hybrid (A* + MAPPO + Shield)", True, True, args.mappo_checkpoint))

    results = {name: {"throughput": [], "collisions": [], "interventions": []} for name, *_ in scenario_defs}
    csv_rows = []

    for density in args.densities:
        print(f"\n=== density = {density} robots ===")
        for name, enable_veto, enable_reversal, checkpoint in scenario_defs:
            env = _make_env(env_cfg, density, enable_veto, enable_reversal)
            policy_fn = None
            if checkpoint is not None:
                policy_fn = _load_mappo_policy(
                    checkpoint, env, train_cfg["hidden_dim"], greedy=args.greedy, temperature=args.temperature
                )
            stats = run_scenario(env, policy_fn, args.eval_steps)

            throughput = stats.throughput_per_100_steps()
            results[name]["throughput"].append(throughput)
            results[name]["collisions"].append(stats.collisions)
            results[name]["interventions"].append(stats.shield_interventions)
            csv_rows.append(
                {
                    "density": density,
                    "scenario": name,
                    "throughput_per_100_steps": throughput,
                    "collisions": stats.collisions,
                    "shield_interventions": stats.shield_interventions,
                    "deliveries": stats.deliveries,
                }
            )
            print(
                f"  {name:38s} throughput={throughput:6.2f}/100steps  "
                f"collisions={stats.collisions:4d}  interventions={stats.shield_interventions:4d}"
            )

    csv_path = os.path.join(args.output_dir, "evaluation_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)

    plot_density_comparison(args.densities, results, args.output_dir)
    print(f"\nSaved CSV to {csv_path}")
    print(f"Saved plots to {args.output_dir}/throughput_vs_density.png, collisions.png, shield_interventions.png")


if __name__ == "__main__":
    main()
