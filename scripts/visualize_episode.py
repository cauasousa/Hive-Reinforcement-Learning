"""Render an episode as an animated GIF: robots moving on the 2D warehouse grid, colored by
status (grey=idle, blue=to_item, green=carrying), with shelves/pickup/dropoff stations drawn.

Usage:
    # scripted A* baseline (no training needed)
    python scripts/visualize_episode.py --policy astar --steps 150

    # a trained MAPPO/PPO checkpoint
    python scripts/visualize_episode.py --policy checkpoint --checkpoint outputs/mappo_checkpoint.pt --robots 4 --steps 150
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.animation as animation
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from hiverl.agents.checkpoint import load_checkpoint
from hiverl.agents.networks import ActorCriticNet
from hiverl.agents.scripted_astar_agent import scripted_actions
from hiverl.env.robot import RobotStatus
from hiverl.env.warehouse_env import WarehouseEnv

ROOT = os.path.join(os.path.dirname(__file__), "..")
_ROBOT_COLORS = {
    RobotStatus.IDLE: "#888888",
    RobotStatus.TO_ITEM: "#1f77b4",
    RobotStatus.PICKING_UP: "#ff7f0e",
    RobotStatus.CARRYING: "#2ca02c",
}
_RELOCATING_COLOR = "#9467bd"
_CMAP = matplotlib.colors.ListedColormap(["white", "#4d4d4d", "#f4a261", "#2a9d8f"])


def _robot_color(robot) -> str:
    if robot.status == RobotStatus.IDLE and robot.path:
        return _RELOCATING_COLOR
    return _ROBOT_COLORS[robot.status]


def _load_checkpoint_policy(
    checkpoint_path: str, env: WarehouseEnv, hidden_dim: int, greedy: bool, temperature: float = 1.0
):
    """``greedy=True`` is deliberately not the recommended default for this multi-agent setup:
    every robot shares the exact same policy network, so when two robots face a symmetric local
    situation (e.g. head-on in a corridor), a deterministic policy makes them pick the *identical*
    action every step -- they conflict, the shield blocks one, and next step they try the same
    losing move again forever, needing the shield every single time. Sampling lets identically-
    positioned robots randomly diverge, breaking that standoff. ``temperature`` < 1.0 sharpens the
    sampled distribution (more decisive, less "hovering between two options") without going fully
    deterministic -- push it too low and you approach the same symmetric-conflict problem as
    greedy.
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
                actions = torch.distributions.Categorical(logits=logits / temperature).sample()
        return {a: int(actions[i].item()) for i, a in enumerate(agents)}

    return policy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-config", default=os.path.join(ROOT, "configs", "env_small.yaml"))
    parser.add_argument("--train-config", default=os.path.join(ROOT, "configs", "train_config.yaml"))
    parser.add_argument("--policy", choices=["astar", "checkpoint"], default="astar")
    parser.add_argument("--checkpoint", default=os.path.join(ROOT, "outputs", "mappo_checkpoint.pt"))
    parser.add_argument("--robots", type=int, default=None, help="override num_robots")
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--greedy", action="store_true", help="argmax instead of sampling for checkpoint policy")
    parser.add_argument(
        "--temperature", type=float, default=1.0,
        help="sharpen (<1.0) or flatten (>1.0) the sampled action distribution; ignored with --greedy",
    )
    parser.add_argument("--enable-veto", action="store_true", default=True)
    parser.add_argument("--no-veto", dest="enable_veto", action="store_false")
    parser.add_argument("--enable-reversal", action="store_true", default=True)
    parser.add_argument("--no-reversal", dest="enable_reversal", action="store_false")
    parser.add_argument("--output", default=os.path.join(ROOT, "outputs", "episode.gif"))
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    with open(args.env_config) as f:
        env_cfg = yaml.safe_load(f)
    with open(args.train_config) as f:
        train_cfg = yaml.safe_load(f)
    if args.robots is not None:
        env_cfg["num_robots"] = args.robots

    env = WarehouseEnv.from_config(
        env_cfg,
        max_episode_steps=10**9,
        enable_veto=args.enable_veto,
        enable_deadlock_reversal=args.enable_reversal,
    )

    if args.policy == "checkpoint":
        policy_fn = _load_checkpoint_policy(
            args.checkpoint, env, train_cfg["hidden_dim"], args.greedy, args.temperature
        )
    else:
        policy_fn = None

    obs, _ = env.reset()

    fig, ax = plt.subplots(figsize=(env.map.width / 2, env.map.height / 2))
    im = ax.imshow(env.map.grid, cmap=_CMAP, vmin=0, vmax=3)
    scat = ax.scatter([], [], s=250, edgecolors="black", zorder=3)
    target_scat = ax.scatter(
        [], [], s=60, marker="x", color=_RELOCATING_COLOR, zorder=2, label="relocation target"
    )
    texts = [ax.text(0, 0, "", ha="center", va="center", fontsize=8, color="white", zorder=4) for _ in env.robots]
    ax.set_xticks([])
    ax.set_yticks([])
    title = ax.set_title("")
    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=10, label=label)
        for label, c in [
            ("idle", _ROBOT_COLORS[RobotStatus.IDLE]),
            ("to pickup", _ROBOT_COLORS[RobotStatus.TO_ITEM]),
            ("picking up", _ROBOT_COLORS[RobotStatus.PICKING_UP]),
            ("carrying", _ROBOT_COLORS[RobotStatus.CARRYING]),
            ("relocating off station", _RELOCATING_COLOR),
        ]
    ]
    ax.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, -0.03), ncol=4, fontsize=7)

    stats = {"deliveries": 0, "collisions": 0, "interventions": 0}

    def update(frame):
        nonlocal obs
        actions = policy_fn(obs) if policy_fn is not None else scripted_actions(env)
        obs, _, _, _, _ = env.step(actions)
        stats["deliveries"] += len(env.last_delivery_events)
        stats["collisions"] += len(env.last_collision_events)
        stats["interventions"] += len(env.last_intervention_events)

        coords = np.array([[r.pos[1], r.pos[0]] for r in env.robots.values()])
        colors = [_robot_color(r) for r in env.robots.values()]
        scat.set_offsets(coords)
        scat.set_color(colors)

        relocating_targets = [r.path[-1] for r in env.robots.values() if r.status == RobotStatus.IDLE and r.path]
        if relocating_targets:
            target_scat.set_offsets(np.array([[c, rr] for rr, c in relocating_targets]))
        else:
            target_scat.set_offsets(np.empty((0, 2)))

        for text, r in zip(texts, env.robots.values()):
            text.set_position((r.pos[1], r.pos[0]))
            text.set_text(str(r.robot_id))
        title.set_text(
            f"step {frame + 1}/{args.steps} | deliveries={stats['deliveries']} "
            f"collisions={stats['collisions']} interventions={stats['interventions']}"
        )
        return [im, scat, target_scat, title, *texts]

    anim = animation.FuncAnimation(fig, update, frames=args.steps, blit=False)
    anim.save(args.output, writer=animation.PillowWriter(fps=args.fps))
    plt.close(fig)

    print(f"Saved {args.steps}-step episode GIF to {args.output}")
    print(f"deliveries={stats['deliveries']} collisions={stats['collisions']} interventions={stats['interventions']}")


if __name__ == "__main__":
    main()
