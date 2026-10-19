#!/usr/bin/env python3
"""Plot the key metrics of a Webots fine-tuning lineage from all its per-round history CSVs into one
PNG.
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from hiverl.utils.history import load_history_csv

ROOT = os.path.join(os.path.dirname(__file__), "..")

# Plot colors.
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
SERIES_1 = "#2a78d6"
SERIES_2 = "#eb6834"
REFERENCE = "#8a8984"


def _to_float(row: dict, key: str, default: float = float("nan")) -> float:
    value = row.get(key)
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _load_rows(paths: list[str]) -> list[dict]:
    rows: list[dict] = []
    for path in paths:
        for row in load_history_csv(path):
            row["_source"] = os.path.basename(path)
            rows.append(row)
    rows.sort(key=lambda r: r.get("timestamp", ""))
    return rows


def _rolling_rate(counts: list[float], ticks: list[float], window: int) -> list[float]:
    """Tick-weighted rolling rate per 1000 ticks over the last window rounds."""
    out = []
    for i in range(len(counts)):
        c = [v for v in counts[max(0, i - window + 1): i + 1]]
        t = [v for v in ticks[max(0, i - window + 1): i + 1]]
        pairs = [(a, b) for a, b in zip(c, t) if not math.isnan(a) and not math.isnan(b)]
        total_ticks = sum(b for _, b in pairs)
        out.append(sum(a for a, _ in pairs) / total_ticks * 1000.0 if total_ticks > 0 else float("nan"))
    return out


def _rolling_mean(values: list[float], window: int, weights: list[float] | None = None) -> list[float]:
    """Rolling mean over the last window rounds that have a value (optionally tick-weighted)."""
    weights = weights or [1.0] * len(values)
    out = []
    for i in range(len(values)):
        pairs = [
            (v, w) for v, w in zip(values[max(0, i - window + 1): i + 1], weights[max(0, i - window + 1): i + 1])
            if not math.isnan(v) and not math.isnan(w)
        ]
        total_w = sum(w for _, w in pairs)
        out.append(sum(v * w for v, w in pairs) / total_w if total_w > 0 else float("nan"))
    return out


def _style(ax, title: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=11, color=TEXT, pad=8)
    ax.set_ylabel(ylabel, fontsize=9, color=TEXT_SECONDARY)
    ax.set_xlabel("round (chronological, all runs)", fontsize=9, color=TEXT_SECONDARY)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=8)


def _min_y_span(ax, values: list[float], span: float) -> None:
    """Keep the y-axis at least span tall around the data."""
    finite = [v for v in values if not math.isnan(v)]
    ax.ticklabel_format(axis="y", useOffset=False)
    if not finite:
        return
    lo, hi = min(finite), max(finite)
    if hi - lo < span:
        mid = (lo + hi) / 2.0
        ax.set_ylim(mid - span / 2.0, mid + span / 2.0)


def _raw_and_trend(ax, x, raw, trend, color, label=None) -> None:
    ax.plot(x, raw, color=color, linewidth=0.8, alpha=0.3)
    ax.plot(x, trend, color=color, linewidth=2.0, label=label)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--history-csv", nargs="+", default=None,
        help="one or more history CSVs to plot together, oldest to newest by their own "
        "timestamps regardless of the order given here. Defaults to every real training-lineage "
        "outputs/mappo_checkpoint_continuous_webots_finetuned_*_history.csv found (NOT the "
        "one-off exp_*_history.csv diagnostic runs -- pass those explicitly if wanted).",
    )
    parser.add_argument(
        "--output", default=os.path.join(ROOT, "outputs", "webots_finetune_progress.png"),
    )
    parser.add_argument(
        "--rolling", type=int, default=10,
        help="window (in rounds) of the trend line drawn over each panel's raw per-round values.",
    )
    args = parser.parse_args()

    paths = args.history_csv or sorted(glob.glob(
        os.path.join(ROOT, "outputs", "mappo_checkpoint_continuous_webots_finetuned_*_history.csv")
    ))
    paths = [p for p in paths if os.path.getsize(p) > 0]
    if not paths:
        raise SystemExit(
            "No history CSVs found (looked for outputs/mappo_checkpoint_continuous_webots_finetuned_"
            "*_history.csv). Run scripts/train_mappo_webots.py first -- it writes one every round."
        )
    print(f"Reading {len(paths)} history file(s):")
    for p in paths:
        print(f"  {p}")

    rows = _load_rows(paths)
    if not rows:
        raise SystemExit("Every history CSV found was empty (header only, zero rounds).")

    w = args.rolling
    x = list(range(1, len(rows) + 1))
    ticks = [_to_float(r, "ticks") for r in rows]
    per_1k = lambda key: [
        _to_float(r, key, 0.0) / t * 1000.0 if t and not math.isnan(t) else float("nan")
        for r, t in zip(rows, ticks)
    ]
    deliveries = [_to_float(r, "deliveries", 0.0) for r in rows]
    interventions = [_to_float(r, "interventions", 0.0) for r in rows]
    wall_block = [_to_float(r, "wall_block", 0.0) for r in rows]
    mean_reward = [_to_float(r, "mean_reward") for r in rows]
    approx_kl = [_to_float(r, "approx_kl") for r in rows]
    entropy = [_to_float(r, "entropy") for r in rows]
    explained_variance = [_to_float(r, "explained_variance") for r in rows]

    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, axes = plt.subplots(2, 3, figsize=(17, 8.5), facecolor=SURFACE)
    (ax_deliv, ax_reward, ax_safety), (ax_kl, ax_entropy, ax_ev) = axes

    _raw_and_trend(ax_deliv, x, per_1k("deliveries"), _rolling_rate(deliveries, ticks, w), SERIES_1)
    _style(ax_deliv, "Deliveries per 1000 ticks", "deliveries / 1k ticks")
    ax_deliv.set_ylim(bottom=0)

    _raw_and_trend(ax_reward, x, mean_reward, _rolling_mean(mean_reward, w, ticks), SERIES_1)
    ax_reward.axhline(0, color=REFERENCE, linewidth=0.8, linestyle="--")
    _style(ax_reward, "Mean reward per tick", "mean_reward")

    _raw_and_trend(
        ax_safety, x, per_1k("interventions"), _rolling_rate(interventions, ticks, w), SERIES_1,
        label="shield interventions",
    )
    _raw_and_trend(
        ax_safety, x, per_1k("wall_block"), _rolling_rate(wall_block, ticks, w), SERIES_2, label="wall_block",
    )
    _style(ax_safety, "Safety events per 1000 ticks (lower is better)", "events / 1k ticks")
    ax_safety.set_ylim(bottom=0)
    ax_safety.legend(fontsize=8, frameon=False, labelcolor=TEXT_SECONDARY, loc="upper left")

    ax_kl.axhspan(0.005, 0.02, color=SERIES_1, alpha=0.08, linewidth=0)
    ax_kl.plot(x, approx_kl, color=SERIES_1, linewidth=0, marker="o", markersize=3, alpha=0.4)
    ax_kl.plot(x, _rolling_mean(approx_kl, w), color=SERIES_1, linewidth=2.0)
    _style(ax_kl, "PPO step size (approx_kl)", "approx_kl")
    ax_kl.set_ylim(bottom=0)
    ax_kl.text(
        0.99, 0.97, "shaded = healthy 0.005-0.02", transform=ax_kl.transAxes,
        ha="right", va="top", fontsize=8, color=TEXT_SECONDARY,
    )

    ax_entropy.plot(x, entropy, color=SERIES_1, linewidth=0, marker="o", markersize=3, alpha=0.4)
    ax_entropy.plot(x, _rolling_mean(entropy, w), color=SERIES_1, linewidth=2.0)
    _style(ax_entropy, "Policy entropy (flat = exploration frozen)", "entropy")
    _min_y_span(ax_entropy, entropy, 0.2)

    ax_ev.plot(x, explained_variance, color=SERIES_1, linewidth=0, marker="o", markersize=3, alpha=0.4)
    ax_ev.plot(x, _rolling_mean(explained_variance, w), color=SERIES_1, linewidth=2.0)
    ax_ev.axhline(0, color=REFERENCE, linewidth=0.8, linestyle="--")
    _style(ax_ev, "Critic quality (explained_variance, 1 = perfect)", "explained_variance")

    seen_sources: list[str] = []
    for r in rows:
        if r["_source"] not in seen_sources:
            seen_sources.append(r["_source"])
    for source in seen_sources[1:]:
        boundary = next(i for i, r in enumerate(rows) if r["_source"] == source) + 1
        for ax in axes.flat:
            ax.axvline(boundary, color=GRID, linewidth=1.0, linestyle=":")

    updates = sum(1 for r in rows if _to_float(r, "update_skipped", 0.0) < 0.5)
    fig.suptitle(
        f"Webots fine-tuning -- {len(rows)} rounds, {len(paths)} run(s), {updates} PPO updates  "
        f"(faint = per round, solid = {w}-round trend)",
        x=0.01, ha="left", fontsize=12, color=TEXT,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(args.output, dpi=120, facecolor=SURFACE)
    plt.close(fig)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
