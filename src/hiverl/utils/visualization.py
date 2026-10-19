"""Matplotlib rendering of the warehouse and training plots (Agg backend, saved to disk)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np

from hiverl.env.robot import RobotStatus
from hiverl.env.warehouse_grid import AISLE, DROPOFF, PICKUP, SHELF

_CELL_CHARS = {AISLE: ".", SHELF: "#", PICKUP: "P", DROPOFF: "D"}
_ROBOT_COLORS = {
    RobotStatus.IDLE: "#888888",
    RobotStatus.TO_ITEM: "#1f77b4",
    RobotStatus.PICKING_UP: "#ff7f0e",
    RobotStatus.CARRYING: "#2ca02c",
}
_RELOCATING_COLOR = "#9467bd"


def _robot_color(robot) -> str:
    if robot.status == RobotStatus.IDLE and robot.path:
        return _RELOCATING_COLOR
    return _ROBOT_COLORS[robot.status]


def render_ascii(env) -> str:
    grid = env.map.grid
    chars = [[_CELL_CHARS[int(v)] for v in row] for row in grid]
    for robot in env.robots.values():
        r, c = robot.pos
        chars[r][c] = str(robot.robot_id % 10)
    return "\n".join("".join(row) for row in chars)


def render_matplotlib(env, ax: plt.Axes | None = None, title: str | None = None):
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(env.map.width / 2, env.map.height / 2))

    cmap = matplotlib.colors.ListedColormap(["white", "#4d4d4d", "#f4a261", "#2a9d8f"])
    ax.imshow(env.map.grid, cmap=cmap, vmin=0, vmax=3)

    for robot in env.robots.values():
        r, c = robot.pos
        ax.scatter([c], [r], color=_robot_color(robot), s=250, edgecolors="black", zorder=3)
        ax.text(c, r, str(robot.robot_id), ha="center", va="center", fontsize=8, color="white", zorder=4)
        if robot.status == RobotStatus.IDLE and robot.path:
            tr, tc = robot.path[-1]
            ax.scatter([tc], [tr], marker="x", color=_RELOCATING_COLOR, s=60, zorder=2)

    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title)

    if own_fig:
        return fig
    return ax


def save_frame(env, path: str, title: str | None = None):
    fig = render_matplotlib(env, title=title)
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)


def plot_training_curve(history: list[dict], path: str, title: str = "Training progress"):
    updates = [h["update"] for h in history]
    rewards = [h["mean_episode_reward"] for h in history]
    interventions = [h.get("shield_interventions", 0) for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(updates, rewards, color="#2a9d8f")
    ax1.set_xlabel("update")
    ax1.set_ylabel("mean episode reward")
    ax1.set_title("Reward")

    ax2.plot(updates, interventions, color="#e76f51")
    ax2.set_xlabel("update")
    ax2.set_ylabel("shield interventions / rollout")
    ax2.set_title("Safety Shield interventions")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_density_comparison(densities: list[int], results: dict[str, dict[str, list[float]]], out_dir: str):
    """Bar plots of throughput / interventions / collisions per scenario."""
    import os

    colors = {
        "A* pure": "#e76f51",
        "A* + Shield (veto only)": "#e9c46a",
        "Full hybrid (A* + MAPPO + Shield)": "#2a9d8f",
    }

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for name, data in results.items():
        ax.plot(densities, data["throughput"], marker="o", label=name, color=colors.get(name))
    ax.set_xlabel("number of robots")
    ax.set_ylabel("deliveries per 100 steps")
    ax.set_title("Throughput vs. robot density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "throughput_vs_density.png"), dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    width = 0.8 / max(1, len(results))
    x = np.arange(len(densities))
    for idx, (name, data) in enumerate(results.items()):
        ax.bar(x + idx * width, data["collisions"], width=width, label=name, color=colors.get(name))
    ax.set_xticks(x + width * (len(results) - 1) / 2)
    ax.set_xticklabels(densities)
    ax.set_xlabel("number of robots")
    ax.set_ylabel("collision events")
    ax.set_title("Collisions vs. robot density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "collisions.png"), dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for idx, (name, data) in enumerate(results.items()):
        ax.bar(x + idx * width, data["interventions"], width=width, label=name, color=colors.get(name))
    ax.set_xticks(x + width * (len(results) - 1) / 2)
    ax.set_xticklabels(densities)
    ax.set_xlabel("number of robots")
    ax.set_ylabel("shield interventions (log scale)")
    ax.set_yscale("log")
    ax.set_title("Safety Shield intervention rate vs. robot density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "shield_interventions.png"), dpi=120)
    plt.close(fig)
