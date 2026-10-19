#!/usr/bin/env python3
"""Fine-tune a MAPPO policy on Webots over ROS2.

Each round collects real ticks through a lockstep coordinator (one or more Webots worlds), computes
GAE and runs one PPO update. Checkpoints, per-round history CSV, KL diagnostics and best checkpoints
are written next to --checkpoint-out.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from hiverl.agents.checkpoint import load_checkpoint, save_checkpoint
from hiverl.agents.networks import ActorCriticNet, ContinuousActorCriticNet, migrate_architecture_change
from hiverl.agents.ppo import PPOConfig, RunningMeanStd, compute_gae
from hiverl.bridge.coords import env_pose_to_world
from hiverl.bridge.lockstep_coordinator import (
    ContinuousLockstepCoordinator,
    LockstepCoordinator,
    TickResult,
    _LIDAR_STALL_SUSTAINED_STREAK_TICKS,
)
from hiverl.env.continuous_kinematics import OMEGA_MAX_RADPS, V_MAX_MPS
from hiverl.env.robot import RobotStatus
from hiverl.env.safety_shield_continuous import _min_wall_clearance
from hiverl.env.warehouse_env import WarehouseEnv
from hiverl.env.warehouse_env_continuous import ContinuousWarehouseEnv
from hiverl.utils.history import append_history_row

ROOT = os.path.join(os.path.dirname(__file__), "..")

_ACTION_LABELS = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT", 4: "WAIT"}

NEARBY_ROBOT_DISTANCE_M = 1.0
# Real chassis contact distance used by the collision metrics.
REAL_COLLISION_DISTANCE_M = 0.48

# How far below --actor-log-std-init the log-std floor starts.
LOG_STD_INIT_FLOOR_HEADROOM = 0.5


def _obs_to_tensors(
    obs_dict: dict, agents: list[str], device: str, is_continuous: bool = False, agent_state: bool = False,
):
    """Stack the per-agent observation dicts into (N, ...) tensors."""
    lidar = torch.tensor(np.stack([obs_dict[a]["lidar"] for a in agents]), dtype=torch.float32, device=device)
    vector = torch.tensor(np.stack([obs_dict[a]["vector"] for a in agents]), dtype=torch.float32, device=device)
    if is_continuous:
        shield = torch.tensor(
            np.stack([obs_dict[a]["shield_intervened"] for a in agents]), dtype=torch.float32, device=device
        )
        nearest_robot = torch.tensor(
            np.stack([obs_dict[a]["nearest_robot"] for a in agents]), dtype=torch.float32, device=device
        )
        shield = torch.cat([shield, nearest_robot], dim=-1)
        if agent_state:
            state = torch.tensor(
                np.stack([obs_dict[a]["agent_state"] for a in agents]), dtype=torch.float32, device=device
            )
            shield = torch.cat([shield, state], dim=-1)
        return lidar, vector, None, shield
    mask = torch.tensor(np.stack([obs_dict[a]["action_mask"] for a in agents]), dtype=torch.float32, device=device)
    return lidar, vector, mask, None


def _deliveries_per_1000_ticks(window: list[dict]) -> float:
    """Deliveries per 1000 ticks over the given rounds (round length varies)."""
    total_ticks = sum(s["ticks"] for s in window)
    if total_ticks == 0:
        return 0.0
    return sum(s["deliveries"] for s in window) / total_ticks * 1000.0


def _lidar_stall_summary(
    completed_streak_lengths: list[int], coordinator, ticks: int,
    streak_at_round_start: dict[int, int] | None = None,
) -> dict:
    """Split lidar-stall streaks into isolated vs sustained tick totals."""
    streak_at_round_start = streak_at_round_start or {}
    isolated_ticks = sum(n for n in completed_streak_lengths if n < _LIDAR_STALL_SUSTAINED_STREAK_TICKS)
    sustained_ticks = sum(n for n in completed_streak_lengths if n >= _LIDAR_STALL_SUSTAINED_STREAK_TICKS)
    all_lengths = list(completed_streak_lengths)
    for rid, current_len in getattr(coordinator, "_lidar_stall_streak", {}).items():
        if current_len <= 0:
            continue
        delta = current_len - streak_at_round_start.get(rid, 0)
        if delta <= 0:
            continue
        all_lengths.append(current_len)
        if current_len >= _LIDAR_STALL_SUSTAINED_STREAK_TICKS:
            sustained_ticks += delta
        else:
            isolated_ticks += delta
    num_robots = len(coordinator.env.robots)
    denom = max(1, ticks * num_robots)
    return {
        "lidar_stall_isolated_ticks": isolated_ticks,
        "lidar_stall_sustained_ticks": sustained_ticks,
        "lidar_stall_isolated_frac": isolated_ticks / denom,
        "lidar_stall_sustained_frac": sustained_ticks / denom,
        "max_lidar_stall_streak": max(all_lengths) if all_lengths else 0,
    }


def _log_std_floors(progress: float, omega_rate_multiplier: float, net) -> list[float]:
    """Per-dimension log-std floors for the anneal progress (omega can anneal faster)."""
    v_floor = net.LOG_STD_MIN + (net.LOG_STD_MIN_FINAL - net.LOG_STD_MIN) * progress
    omega_progress = min(1.0, progress * omega_rate_multiplier)
    omega_floor = net.LOG_STD_MIN + (net.LOG_STD_MIN_FINAL - net.LOG_STD_MIN) * omega_progress
    return [v_floor, omega_floor]


def _actor_critic_parameter_groups(net) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    """Split network parameters into actor and critic groups."""
    named_params = list(net.named_parameters())
    actor_params = [param for name, param in named_params if "critic" not in name]
    critic_params = [param for name, param in named_params if "critic" in name]
    if not actor_params or not critic_params:
        raise ValueError("Expected non-empty actor and critic parameter groups")
    if {id(param) for param in actor_params} & {id(param) for param in critic_params}:
        raise ValueError("Actor and critic optimizer parameter groups must be disjoint")
    return actor_params, critic_params


def _value_loss(v_pred: torch.Tensor, target: torch.Tensor, v_old: torch.Tensor | None, cfg: PPOConfig) -> torch.Tensor:
    """Huber or squared value loss, optionally clipped around the old prediction (pessimistic max).
    """
    def _base(pred: torch.Tensor) -> torch.Tensor:
        if cfg.value_loss == "huber":
            return F.huber_loss(pred, target, delta=cfg.huber_delta, reduction="none")
        return (pred - target).pow(2)

    loss = _base(v_pred)
    if cfg.value_clip > 0 and v_old is not None:
        v_clipped = v_old + torch.clamp(v_pred - v_old, -cfg.value_clip, cfg.value_clip)
        loss = torch.max(loss, _base(v_clipped))
    return loss.mean()


def _ppo_update(
    net, actor_optimizer, actor_params, critic_optimizer, critic_params, value_rms,
    segments: list[dict], cfg: PPOConfig, N: int, is_continuous: bool, device: str,
    round_label: str, round_idx: int, kl_diagnostics_csv_path: str | None, target_kl: float | None,
) -> dict:
    """One PPO update over one or more GAE-processed rollout segments: policy epochs (masked clipped
    loss, optional KL early stop), value epochs, adaptive lr and obs-normalizer update.
    """
    upd_T = sum(s["lidar"].shape[0] for s in segments)
    def _cat(key: str):
        return None if segments[0][key] is None else torch.cat([s[key] for s in segments])

    u_lidar_buf, u_vector_buf, u_shield_buf = _cat("lidar"), _cat("vector"), _cat("shield")
    u_flat_lidar = u_lidar_buf.reshape(upd_T * N, *u_lidar_buf.shape[2:])
    u_flat_vector = u_vector_buf.reshape(upd_T * N, *u_vector_buf.shape[2:])
    u_flat_shield = (
        u_shield_buf.reshape(upd_T * N, *u_shield_buf.shape[2:]) if u_shield_buf is not None else None
    )
    u_flat_logprobs = _cat("logprobs").reshape(upd_T * N)
    u_flat_active = (
        _cat("active").reshape(-1).to(torch.float32) if "active" in segments[0]
        else torch.ones(upd_T * N)
    )
    active_sel = u_flat_active > 0.5
    u_flat_adv = _cat("adv").reshape(-1)
    adv_ref = u_flat_adv[active_sel] if active_sel.sum() > 1 else u_flat_adv
    u_flat_adv = (u_flat_adv - adv_ref.mean()) / (adv_ref.std() + 1e-8)
    u_normalized_targets = value_rms.normalize(_cat("targets"))
    u_old_values_n = value_rms.normalize(_cat("old_values")) if "old_values" in segments[0] else None
    if is_continuous:
        u_flat_raw_actions = _cat("raw_actions").reshape(upd_T * N, net.action_dim)
    else:
        u_mask_buf = _cat("mask")
        u_flat_mask = u_mask_buf.reshape(upd_T * N, *u_mask_buf.shape[2:])
        u_flat_actions = _cat("actions").reshape(upd_T * N)

    sample_idx = np.arange(upd_T * N)
    time_idx = np.arange(upd_T)
    policy_losses, value_losses, entropies = [], [], []
    approx_kls, clip_fractions = [], []
    kl_rows: list[dict] = []
    policy_stopped_at_epoch = None
    # Policy epochs.
    for epoch_idx in range(cfg.update_epochs):
        np.random.shuffle(sample_idx)
        epoch_kl_start = len(approx_kls)
        for mb_idx, mb in enumerate(np.array_split(sample_idx, cfg.minibatches)):
            if len(mb) == 0 or policy_stopped_at_epoch is not None:
                continue
            mb_t = torch.as_tensor(mb, device=device)
            if is_continuous:
                new_logprobs, entropy = net.evaluate_actions(
                    u_flat_lidar[mb_t], u_flat_vector[mb_t], u_flat_raw_actions[mb_t], u_flat_shield[mb_t]
                )
            else:
                new_logprobs, entropy = net.evaluate_actions(
                    u_flat_lidar[mb_t], u_flat_vector[mb_t], u_flat_actions[mb_t], u_flat_mask[mb_t]
                )
            logprobs_before_this_update = new_logprobs.detach()
            mu_before_this_update = None
            if is_continuous and kl_diagnostics_csv_path is not None:
                with torch.no_grad():
                    mu_before_this_update = net._dist(
                        u_flat_lidar[mb_t], u_flat_vector[mb_t], u_flat_shield[mb_t]
                    ).mean
            logratio = new_logprobs - u_flat_logprobs[mb_t]
            ratio = torch.exp(logratio)
            mb_adv = u_flat_adv[mb_t]
            surr1 = ratio * mb_adv
            surr2 = torch.clamp(ratio, 1 - cfg.clip_range, 1 + cfg.clip_range) * mb_adv
            mb_w = u_flat_active[mb_t]
            mb_den = mb_w.sum().clamp(min=1.0)
            policy_loss = -(torch.min(surr1, surr2) * mb_w).sum() / mb_den
            entropy_bonus = (entropy * mb_w).sum() / mb_den
            loss = policy_loss - cfg.entropy_coef * entropy_bonus

            actor_optimizer.zero_grad(set_to_none=True)
            critic_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor_params, cfg.max_grad_norm)
            actor_optimizer.step()

            policy_losses.append(policy_loss.item())
            entropies.append(entropy_bonus.item())
            with torch.no_grad():
                mb_approx_kl = ((((ratio - 1) - logratio) * mb_w).sum() / mb_den).item()
                mb_clip_fraction = (
                    (((ratio - 1).abs() > cfg.clip_range).float() * mb_w).sum() / mb_den
                ).item()
                approx_kls.append(mb_approx_kl)
                clip_fractions.append(mb_clip_fraction)
                if kl_diagnostics_csv_path is not None:
                    if is_continuous:
                        logprobs_after_this_update, _ = net.evaluate_actions(
                            u_flat_lidar[mb_t], u_flat_vector[mb_t], u_flat_raw_actions[mb_t], u_flat_shield[mb_t]
                        )
                    else:
                        logprobs_after_this_update, _ = net.evaluate_actions(
                            u_flat_lidar[mb_t], u_flat_vector[mb_t], u_flat_actions[mb_t], u_flat_mask[mb_t]
                        )
                    local_logratio = logprobs_after_this_update - logprobs_before_this_update
                    local_ratio = torch.exp(local_logratio)
                    update_kl = ((local_ratio - 1) - local_logratio).mean().item()
                    mu_after_this_update = net._dist(
                        u_flat_lidar[mb_t], u_flat_vector[mb_t], u_flat_shield[mb_t]
                    ).mean
                    delta_mu = (mu_after_this_update - mu_before_this_update).abs().mean(dim=0)
                    kl_rows.append({
                        "round": round_idx,
                        "epoch": epoch_idx,
                        "minibatch": mb_idx,
                        "approx_kl": mb_approx_kl,
                        "clip_fraction": mb_clip_fraction,
                        "mean_ratio": ratio.mean().item(),
                        "max_ratio": ratio.max().item(),
                        "min_ratio": ratio.min().item(),
                        "update_kl": update_kl,
                        "delta_mu_v": delta_mu[0].item(),
                        "delta_mu_omega": delta_mu[1].item(),
                    })

        if (
            target_kl is not None and policy_stopped_at_epoch is None
            and len(approx_kls) > epoch_kl_start
            and float(np.mean(approx_kls[epoch_kl_start:])) > 1.5 * target_kl
        ):
            policy_stopped_at_epoch = epoch_idx
            print(
                f"[{round_label}] KL early stop: epoch {epoch_idx} mean approx_kl="
                f"{float(np.mean(approx_kls[epoch_kl_start:])):.4f} > 1.5 x --target-kl={target_kl} "
                f"-- no more policy epochs this round (value epochs continue).", flush=True,
            )

        np.random.shuffle(time_idx)
        # Value epochs over whole timesteps (centralized critic).
        for mb in np.array_split(time_idx, cfg.minibatches):
            if len(mb) == 0:
                continue
            mb_t = torch.as_tensor(mb, device=device)
            v_pred = net.centralized_value(
                u_lidar_buf[mb_t], u_vector_buf[mb_t], u_shield_buf[mb_t] if u_shield_buf is not None else None
            )
            value_loss = _value_loss(
                v_pred, u_normalized_targets[mb_t],
                u_old_values_n[mb_t] if u_old_values_n is not None else None, cfg,
            )

            actor_optimizer.zero_grad(set_to_none=True)
            critic_optimizer.zero_grad(set_to_none=True)
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor_params, cfg.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(critic_params, cfg.max_grad_norm)
            actor_optimizer.step()
            critic_optimizer.step()
            value_losses.append(value_loss.item())
    # Adaptive actor lr from the measured KL.
    if cfg.desired_kl > 0 and approx_kls:
        mean_kl = float(np.mean(approx_kls))
        for group in actor_optimizer.param_groups:
            if mean_kl > 2.0 * cfg.desired_kl:
                group["lr"] = max(cfg.lr_min, group["lr"] / 1.5)
            elif mean_kl < cfg.desired_kl / 2.0:
                group["lr"] = min(cfg.lr_max, group["lr"] * 1.5)
    if getattr(net, "obs_norm", False):
        net.update_obs_normalizer(net.raw_per_robot_features(u_lidar_buf, u_vector_buf, u_shield_buf))
    return {
        "policy_losses": policy_losses, "value_losses": value_losses, "entropies": entropies,
        "approx_kls": approx_kls, "clip_fractions": clip_fractions, "kl_rows": kl_rows,
        "actor_lr": actor_optimizer.param_groups[0]["lr"],
    }


def collect_and_update(
    coordinator: LockstepCoordinator | ContinuousLockstepCoordinator,
    net: ActorCriticNet | ContinuousActorCriticNet,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    value_rms: RunningMeanStd,
    agents: list[str],
    device: str,
    cfg: PPOConfig,
    ticks: int,
    round_label: str,
    reset: bool = True,
    is_continuous: bool = False,
    round_idx: int = 0,
    kl_diagnostics_csv_path: str | None = None,
    tick_trace_csv_path: str | None = None,
    min_update_ticks: int = 0,
    rollout_carry: list | None = None,
    target_kl: float | None = None,
    defer_update: bool = False,
) -> dict | None:
    """One round: collect ticks real transitions, compute GAE and update (or defer/accumulate). Returns
    the round summary, or None if too few ticks were collected.
    """
    actor_params, critic_params = _actor_critic_parameter_groups(net)
    pending: dict = {}
    buffer: dict[str, list] = {
        "lidar": [], "vector": [], "mask": [], "shield": [], "actions": [], "logprobs": [],
        "rewards": [], "dones": [], "values": [],
        "active": [],
        "raw_actions": [],
        "executed": [],
        "tick_omega_penalty": [],
        "tick_mean_abs_omega_active": [],
    }
    pending_fresh = [False]
    totals = {
        "deliveries": 0, "collisions": 0, "interventions": 0, "wall_block": 0, "stuck_events": 0,
        "wall_proximity_penalty": 0.0,
    }
    pickup_latency_ticks: list[int] = []
    delivery_latency_ticks: list[int] = []
    lidar_stall_event_ticks = 0
    lidar_stall_streak_lengths: list[int] = []
    lidar_stall_streak_at_round_start = dict(getattr(coordinator, "_lidar_stall_streak", {}))
    reward_breakdown_totals: dict[str, float] = {}
    map_span = coordinator.env.map.width + coordinator.env.map.height
    tick_trace_rows: list[dict] = []
    distance_first: dict[int, float] = {}
    distance_last: dict[int, float] = {}

    # Sample actions and buffer observations, actions, log-probs and values.
    def policy_fn(obs_dict: dict) -> dict:
        lidar, vector, mask, shield = _obs_to_tensors(
            obs_dict, agents, device, is_continuous, agent_state=getattr(net, "agent_state_dim", 0) > 0,
        )
        raw_action = None
        with torch.no_grad():
            if is_continuous:
                actions, logprobs, _, raw_action = net.act(lidar, vector, shield)
            else:
                actions, logprobs, _ = net.act(lidar, vector, mask)
            norm_value = net.centralized_value(
                lidar.unsqueeze(0), vector.unsqueeze(0), shield.unsqueeze(0) if shield is not None else None
            )[0]
            value = value_rms.denormalize(norm_value)
        pending["data"] = (lidar, vector, mask, shield, actions, logprobs, value, raw_action)
        pending_fresh[0] = True
        if is_continuous:
            for i, a in enumerate(agents):
                dist = float(obs_dict[a]["vector"][2]) * map_span
                distance_first.setdefault(i, dist)
                distance_last[i] = dist
            return {a: actions[i].cpu().numpy() for i, a in enumerate(agents)}
        return {a: int(actions[i].item()) for i, a in enumerate(agents)}

    # After every tick: buffer rewards/dones and accumulate round metrics.
    def on_tick(tick: int, result: TickResult) -> None:
        if not pending_fresh[0]:
            waiting = [
                f"robot_{i}: {coordinator.driver.arrival_diagnostic(i)}"
                for i, busy in coordinator._busy.items() if busy
            ]
            print(
                f"[{round_label} tick {tick}] WARNING: a robot is overdue past arrival_timeout_s "
                f"and hasn't reported arrival yet -- still waiting, not recording a transition.\n"
                f"  reset/arrival diagnostics: {'; '.join(waiting)}",
                flush=True,
            )
            return
        lidar, vector, mask, shield, actions, logprobs, value, raw_action = pending["data"]
        rewards = torch.tensor([coordinator.last_rewards[a] for a in agents], dtype=torch.float32, device=device)
        done = [
            float(coordinator.last_terminations.get(a, False) or coordinator.last_truncations.get(a, False))
            for a in agents
        ]
        buffer["lidar"].append(lidar)
        buffer["vector"].append(vector)
        buffer["mask"].append(mask)
        buffer["shield"].append(shield)
        buffer["actions"].append(actions)
        buffer["logprobs"].append(logprobs)
        if is_continuous:
            buffer["raw_actions"].append(raw_action)
        buffer["rewards"].append(rewards)
        buffer["dones"].append(done)
        controlled = getattr(coordinator.env, "last_policy_controlled_ids", None)
        overridden = getattr(coordinator, "last_overridden_ids", set())
        buffer["active"].append([
            1.0 if (controlled is None or i in controlled) and i not in overridden else 0.0
            for i in range(len(agents))
        ])
        buffer["values"].append(value)
        if is_continuous:
            final_v = getattr(coordinator.env, "last_final_velocities", {})
            executed = torch.stack([
                torch.tensor(
                    (final_v[i][0] / V_MAX_MPS, final_v[i][1] / OMEGA_MAX_RADPS), dtype=torch.float32,
                ) if i in final_v else actions[i].detach().cpu()
                for i in range(len(agents))
            ])
            buffer["executed"].append(executed)
        pending_fresh[0] = False
        env = coordinator.env
        totals["deliveries"] += result.deliveries
        totals["collisions"] += result.collisions
        totals["interventions"] += result.interventions
        totals["wall_block"] += len(getattr(env, "last_wall_block_events", ()))
        totals["stuck_events"] += len(getattr(env, "last_stuck_events", ()))
        pickup_latency_ticks.extend(getattr(env, "last_pickup_latency_steps", ()))
        delivery_latency_ticks.extend(getattr(env, "last_delivery_latency_steps", ()))
        nonlocal lidar_stall_event_ticks
        lidar_stall_event_ticks += len(getattr(coordinator, "last_lidar_stall_events", ()))
        completed_streaks = getattr(coordinator, "last_lidar_stall_streak_ends", ())
        lidar_stall_streak_lengths.extend(completed_streaks)
        totals["wall_proximity_penalty"] += getattr(env, "last_reward_breakdown", {}).get("wall_proximity", 0.0)
        for key, val in getattr(env, "last_reward_breakdown", {}).items():
            reward_breakdown_totals[key] = reward_breakdown_totals.get(key, 0.0) + val
        if is_continuous:
            active_omegas = [
                abs(float(actions[i][1])) for i in range(len(agents))
                if env.robots[i].status in (RobotStatus.TO_ITEM, RobotStatus.CARRYING)
            ]
            buffer["tick_omega_penalty"].append(
                float(getattr(env, "last_reward_breakdown", {}).get("omega_penalty", 0.0))
            )
            buffer["tick_mean_abs_omega_active"].append(
                sum(active_omegas) / len(active_omegas) if active_omegas else 0.0
            )
            if tick_trace_csv_path is not None:
                with torch.no_grad():
                    hidden_t = net.embed(lidar, vector, shield)
                    mu_t = net.actor_mean(hidden_t)
                noise_t = raw_action - mu_t
                for i in range(len(agents)):
                    r = env.robots[i]
                    nearest_obstacle = env._nearest_obstacle_distance(i, 3.0)
                    wall_clearance = _min_wall_clearance(env.map, r.x, r.y)
                    v_final, omega_final = final_v.get(i, (float("nan"), float("nan")))
                    tick_trace_rows.append({
                        "round": round_idx, "tick": tick, "robot_id": i,
                        "raw_v": float(actions[i][0]), "raw_omega": float(actions[i][1]),
                        "mu_v": float(mu_t[i][0]), "mu_omega": float(mu_t[i][1]),
                        "noise_v": float(noise_t[i][0]), "noise_omega": float(noise_t[i][1]),
                        "shield_v_norm": v_final / V_MAX_MPS if v_final == v_final else float("nan"),
                        "shield_omega_norm": omega_final / OMEGA_MAX_RADPS if omega_final == omega_final else float("nan"),
                        "delta_v": (v_final / V_MAX_MPS - float(actions[i][0])) if v_final == v_final else float("nan"),
                        "delta_omega": (omega_final / OMEGA_MAX_RADPS - float(actions[i][1])) if omega_final == omega_final else float("nan"),
                        "intervention": i in env.last_intervention_events,
                        "veto": i in getattr(env, "last_veto_events", ()),
                        "wall_block": i in env.last_wall_block_events,
                        "nearest_obstacle_m": nearest_obstacle,
                        "wall_clearance_m": wall_clearance,
                        "robot_status": r.status.name,
                        "distance_to_target": distance_last.get(i, float("nan")),
                    })
        if is_continuous:
            action_str = ",".join(
                f"{i}=(v={float(actions[i][0]):.2f},w={float(actions[i][1]):.2f})" for i in range(len(agents))
            )
        else:
            action_str = ",".join(
                f"{i}={_ACTION_LABELS.get(int(actions[i].item()), actions[i].item())}" for i in range(len(agents))
            )
        busy_str = ",".join(f"{i}:{'BUSY' if coordinator._busy[i] else 'free'}" for i in range(len(agents)))
        status_str = ",".join(f"{i}={env.robots[i].status.name}" for i in range(len(agents)))
        pos_str = ",".join(f"{i}={env.robots[i].pos}" for i in range(len(agents)))

        real_poses = {i: coordinator.driver.get_pose(i) for i in range(len(agents))}
        real_pos_str = ",".join(
            f"{i}=({p[0]:.2f},{p[1]:.2f})" if p is not None else f"{i}=?" for i, p in real_poses.items()
        )
        drift_parts = []
        for i, p in real_poses.items():
            if p is None:
                drift_parts.append(f"{i}=?")
                continue
            r = env.robots[i]
            wx, wy, _ = env_pose_to_world(r.x, r.y, r.theta)
            drift_parts.append(f"{i}={math.hypot(wx - p[0], wy - p[1]):.2f}m")
        drift_str = ",".join(drift_parts)
        nearby_pairs = []
        real_collision_pairs = []
        known = [(i, p) for i, p in real_poses.items() if p is not None]
        for idx_a in range(len(known)):
            for idx_b in range(idx_a + 1, len(known)):
                i, (xi, yi, _) = known[idx_a]
                j, (xj, yj, _) = known[idx_b]
                dist = math.hypot(xi - xj, yi - yj)
                if dist < NEARBY_ROBOT_DISTANCE_M:
                    nearby_pairs.append((i, j, round(dist, 3)))
                if dist < REAL_COLLISION_DISTANCE_M:
                    real_collision_pairs.append((i, j, round(dist, 3)))

        print(
            f"[{round_label} tick {tick}] reward_mean={rewards.mean().item():.3f} "
            f"deliveries={result.deliveries} collisions={result.collisions} "
            f"interventions={result.interventions} arrived_in_time={result.arrived_in_time}\n"
            f"  action={{{action_str}}}\n"
            f"  status={{{status_str}}}\n"
            f"  busy={{{busy_str}}} pos={{{pos_str}}}\n"
            f"  real_pos={{{real_pos_str}}} logical_vs_real_drift={{{drift_str}}} nearby_robot_pairs={nearby_pairs}\n"
            f"  collision_robot_ids={sorted(env.last_collision_events)} "
            f"intervention_robot_ids={sorted(env.last_intervention_events)} "
            f"wall_block_robot_ids={sorted(env.last_wall_block_events)} "
            f"physical_proximity_robot_ids={sorted(coordinator.last_physical_proximity_events)} "
            f"real_collision_pairs={real_collision_pairs}",
            flush=True,
        )

    coordinator.on_tick = on_tick
    print(f"[{round_label}] collecting {ticks} real ticks...", flush=True)
    t0 = time.time()
    # Collect the round on Webots.
    coordinator.run(ticks, policy_fn, reset=reset)
    elapsed = time.time() - t0
    n = len(buffer["rewards"])
    print(
        f"[{round_label}] collected {n} real ticks in {elapsed:.1f}s "
        f"({elapsed / max(1, n):.1f}s/tick average)",
        flush=True,
    )
    if n < 2:
        print(f"[{round_label}] too few real ticks to update on (need >=2) -- skipping this round.")
        return None

    agent_env = coordinator.env
    final_obs = {a: agent_env._get_obs(i) for i, a in enumerate(agents)}
    lidar_final, vector_final, _, shield_final = _obs_to_tensors(
        final_obs, agents, device, is_continuous, agent_state=getattr(net, "agent_state_dim", 0) > 0,
    )
    with torch.no_grad():
        norm_bootstrap = net.centralized_value(
            lidar_final.unsqueeze(0), vector_final.unsqueeze(0),
            shield_final.unsqueeze(0) if shield_final is not None else None,
        )[0]
    bootstrap_value = value_rms.denormalize(norm_bootstrap)

    T, N = n, len(agents)
    lidar_buf = torch.stack(buffer["lidar"])
    vector_buf = torch.stack(buffer["vector"])
    mask_buf = None if is_continuous else torch.stack(buffer["mask"])
    shield_buf = torch.stack(buffer["shield"]) if is_continuous else None
    action_buf = torch.stack(buffer["actions"])
    logprob_buf = torch.stack(buffer["logprobs"])
    reward_buf = torch.stack(buffer["rewards"])
    done_buf = torch.tensor(buffer["dones"], dtype=torch.float32, device=device)
    value_buf = torch.cat([torch.stack(buffer["values"]), bootstrap_value.unsqueeze(0)])

    # GAE with per-agent dones.
    adv_agents, value_targets = compute_gae(
        reward_buf, value_buf, done_buf, cfg.gamma, cfg.gae_lambda
    )
    with _VALUE_RMS_LOCK:
        value_rms.update(value_targets.detach())

    flat_adv = adv_agents.reshape(-1)
    flat_adv = (flat_adv - flat_adv.mean()) / (flat_adv.std() + 1e-8)
    flat_lidar = lidar_buf.reshape(T * N, *lidar_buf.shape[2:])
    flat_vector = vector_buf.reshape(T * N, *vector_buf.shape[2:])
    flat_shield = shield_buf.reshape(T * N, *shield_buf.shape[2:]) if shield_buf is not None else None
    flat_logprobs = logprob_buf.reshape(T * N)
    if is_continuous:
        flat_actions = action_buf.reshape(T * N, net.action_dim)
        raw_action_buf = torch.stack(buffer["raw_actions"])
        flat_raw_actions = raw_action_buf.reshape(T * N, net.action_dim)
    else:
        flat_mask = mask_buf.reshape(T * N, *mask_buf.shape[2:])
        flat_actions = action_buf.reshape(T * N)

    segment = {
        "lidar": lidar_buf, "vector": vector_buf, "shield": shield_buf, "mask": mask_buf,
        "actions": action_buf, "raw_actions": raw_action_buf if is_continuous else None,
        "logprobs": logprob_buf, "adv": adv_agents, "targets": value_targets,
        "active": torch.tensor(buffer["active"], dtype=torch.float32, device=device),
        "old_values": value_buf[:-1],
    }
    if rollout_carry is not None:
        rollout_carry.append(segment)
        segments = rollout_carry
    else:
        segments = [segment]
    upd_T = sum(s["lidar"].shape[0] for s in segments)
    # Update now, or keep the rollout until min_update_ticks is reached.
    do_update = (min_update_ticks <= 0 or upd_T >= min_update_ticks) and not defer_update
    # Action saturation / log-prob diagnostics (logged only).
    saturation_stats: dict[str, float] = {}
    if is_continuous:
        with torch.no_grad():
            logprob_current_pre_update, _ = net.evaluate_actions(
                flat_lidar, flat_vector, flat_raw_actions, flat_shield
            )
            abs_actions = flat_actions.abs()
            max_abs_log_prob_diff = float((flat_logprobs - logprob_current_pre_update).abs().max())
            hidden = net.embed(flat_lidar, flat_vector, flat_shield)
            mu = net.actor_mean(hidden)
            noise = flat_raw_actions - mu
            mu_v, mu_omega = mu[:, 0], mu[:, 1]
            noise_v, noise_omega = noise[:, 0], noise[:, 1]
            abs_mu_v, abs_mu_omega = mu_v.abs(), mu_omega.abs()
            mean_abs_hidden = float(hidden.norm(dim=-1).mean())
            growth_ratio_mu_v = float(abs_mu_v.mean()) / mean_abs_hidden if mean_abs_hidden > 0 else float("nan")
            growth_ratio_mu_omega = float(abs_mu_omega.mean()) / mean_abs_hidden if mean_abs_hidden > 0 else float("nan")

            def _percentiles(x: torch.Tensor) -> dict[str, float]:
                q = torch.quantile(x, torch.tensor([0.5, 0.9, 0.95, 0.99]))
                return {"p50": float(q[0]), "p90": float(q[1]), "p95": float(q[2]), "p99": float(q[3])}

            mu_v_pct, mu_omega_pct = _percentiles(abs_mu_v), _percentiles(abs_mu_omega)

            def _corr(x: torch.Tensor, y: torch.Tensor) -> float:
                xc, yc = x - x.mean(), y - y.mean()
                denom = xc.norm() * yc.norm()
                return float((xc @ yc) / denom) if denom > 0 else float("nan")

            corr_mu_v_adv = _corr(abs_mu_v, flat_adv)
            corr_mu_omega_adv = _corr(abs_mu_omega, flat_adv)

            saturation_stats = {
                "pre_tanh_abs_mean": float(flat_raw_actions.abs().mean()),
                "pre_tanh_abs_max": float(flat_raw_actions.abs().max()),
                "mean_abs_hidden": mean_abs_hidden,
                "growth_ratio_mu_v": growth_ratio_mu_v,
                "growth_ratio_mu_omega": growth_ratio_mu_omega,
                "mean_abs_mu_v": float(mu_v.abs().mean()),
                "mean_abs_mu_omega": float(mu_omega.abs().mean()),
                "max_abs_mu_v": float(mu_v.abs().max()),
                "max_abs_mu_omega": float(mu_omega.abs().max()),
                "mu_v_p50": mu_v_pct["p50"], "mu_v_p90": mu_v_pct["p90"],
                "mu_v_p95": mu_v_pct["p95"], "mu_v_p99": mu_v_pct["p99"],
                "mu_omega_p50": mu_omega_pct["p50"], "mu_omega_p90": mu_omega_pct["p90"],
                "mu_omega_p95": mu_omega_pct["p95"], "mu_omega_p99": mu_omega_pct["p99"],
                "fraction_mu_v_gt_1": float((abs_mu_v > 1).float().mean()),
                "fraction_mu_v_gt_2": float((abs_mu_v > 2).float().mean()),
                "fraction_mu_v_gt_4": float((abs_mu_v > 4).float().mean()),
                "fraction_mu_omega_gt_1": float((abs_mu_omega > 1).float().mean()),
                "fraction_mu_omega_gt_2": float((abs_mu_omega > 2).float().mean()),
                "fraction_mu_omega_gt_4": float((abs_mu_omega > 4).float().mean()),
                "corr_mu_v_advantage": corr_mu_v_adv,
                "corr_mu_omega_advantage": corr_mu_omega_adv,
                "mean_abs_noise_v": float(noise_v.abs().mean()),
                "mean_abs_noise_omega": float(noise_omega.abs().mean()),
                "max_abs_noise_v": float(noise_v.abs().max()),
                "max_abs_noise_omega": float(noise_omega.abs().max()),
                "log_prob_rollout_mean": float(flat_logprobs.mean()),
                "log_prob_rollout_min": float(flat_logprobs.min()),
                "log_prob_rollout_max": float(flat_logprobs.max()),
                "log_prob_current_mean": float(logprob_current_pre_update.mean()),
                "log_prob_current_min": float(logprob_current_pre_update.min()),
                "log_prob_current_max": float(logprob_current_pre_update.max()),
                "max_abs_log_prob_diff": max_abs_log_prob_diff,
                "fraction_action_gt_0.95": float((abs_actions > 0.95).float().mean()),
                "fraction_action_gt_0.98": float((abs_actions > 0.98).float().mean()),
                "fraction_action_gt_0.99": float((abs_actions > 0.99).float().mean()),
            }

            abs_action_omega = abs_actions[:, 1]
            action_omega_pct = _percentiles(abs_action_omega)
            saturation_stats.update({
                "mean_abs_action_omega": float(abs_action_omega.mean()),
                "action_omega_p50": action_omega_pct["p50"],
                "action_omega_p90": action_omega_pct["p90"],
                "action_omega_p95": action_omega_pct["p95"],
                "action_omega_p99": action_omega_pct["p99"],
                "fraction_abs_omega_gt_0.5": float((abs_action_omega > 0.5).float().mean()),
                "fraction_abs_omega_gt_0.8": float((abs_action_omega > 0.8).float().mean()),
                "omega_penalty_per_tick": reward_breakdown_totals.get("omega_penalty", 0.0) / T,
            })
            tick_omega_pen_t = torch.tensor(buffer["tick_omega_penalty"], dtype=torch.float32)
            tick_mean_omega_t = torch.tensor(buffer["tick_mean_abs_omega_active"], dtype=torch.float32)
            saturation_stats["corr_action_omega_omega_penalty"] = _corr(
                tick_mean_omega_t, tick_omega_pen_t.abs()
            )
            saturation_stats["fraction_ticks_omega_penalty_zero"] = float(
                (tick_omega_pen_t == 0.0).float().mean()
            )
            print(
                f"[{round_label}] omega-in-open-space audit: mean|action_omega|="
                f"{saturation_stats['mean_abs_action_omega']:.3f} "
                f"p50/p90/p95/p99={action_omega_pct['p50']:.3f}/{action_omega_pct['p90']:.3f}/"
                f"{action_omega_pct['p95']:.3f}/{action_omega_pct['p99']:.3f} "
                f"fraction>0.5={saturation_stats['fraction_abs_omega_gt_0.5']:.3f} "
                f">0.8={saturation_stats['fraction_abs_omega_gt_0.8']:.3f} | "
                f"omega_penalty/tick={saturation_stats['omega_penalty_per_tick']:.4f} "
                f"corr(action_omega,omega_penalty)={saturation_stats['corr_action_omega_omega_penalty']:.3f} "
                f"fraction_ticks_gated_out={saturation_stats['fraction_ticks_omega_penalty_zero']:.3f}",
                flush=True,
            )
            print(
                f"[{round_label}] log_prob/saturation audit (pre-update, same network for both, "
                f"using stored raw_action -- no atanh reconstruction): "
                f"pre_tanh|z| mean={saturation_stats['pre_tanh_abs_mean']:.3f} "
                f"max={saturation_stats['pre_tanh_abs_max']:.3f} | "
                f"log_prob rollout mean={saturation_stats['log_prob_rollout_mean']:.3f} "
                f"[{saturation_stats['log_prob_rollout_min']:.3f},{saturation_stats['log_prob_rollout_max']:.3f}] "
                f"vs current(same net) mean={saturation_stats['log_prob_current_mean']:.3f} "
                f"[{saturation_stats['log_prob_current_min']:.3f},{saturation_stats['log_prob_current_max']:.3f}] "
                f"max_abs_diff={max_abs_log_prob_diff:.6f} | "
                f"|action|>.95={saturation_stats['fraction_action_gt_0.95']:.3f} "
                f">.98={saturation_stats['fraction_action_gt_0.98']:.3f} "
                f">.99={saturation_stats['fraction_action_gt_0.99']:.3f}",
                flush=True,
            )
            print(
                f"[{round_label}] mu-vs-noise audit (z = mu + noise): "
                f"mean|mu| v={saturation_stats['mean_abs_mu_v']:.3f} omega={saturation_stats['mean_abs_mu_omega']:.3f} "
                f"| max|mu| v={saturation_stats['max_abs_mu_v']:.3f} omega={saturation_stats['max_abs_mu_omega']:.3f} "
                f"| mean|noise| v={saturation_stats['mean_abs_noise_v']:.3f} omega={saturation_stats['mean_abs_noise_omega']:.3f} "
                f"| max|noise| v={saturation_stats['max_abs_noise_v']:.3f} omega={saturation_stats['max_abs_noise_omega']:.3f}",
                flush=True,
            )
            print(
                f"[{round_label}] |mu| distribution: "
                f"v p50={mu_v_pct['p50']:.3f} p90={mu_v_pct['p90']:.3f} p95={mu_v_pct['p95']:.3f} p99={mu_v_pct['p99']:.3f} "
                f"| omega p50={mu_omega_pct['p50']:.3f} p90={mu_omega_pct['p90']:.3f} p95={mu_omega_pct['p95']:.3f} p99={mu_omega_pct['p99']:.3f}\n"
                f"  fraction|mu_v|>{{1,2,4}}={saturation_stats['fraction_mu_v_gt_1']:.3f},"
                f"{saturation_stats['fraction_mu_v_gt_2']:.3f},{saturation_stats['fraction_mu_v_gt_4']:.3f} "
                f"fraction|mu_omega|>{{1,2,4}}={saturation_stats['fraction_mu_omega_gt_1']:.3f},"
                f"{saturation_stats['fraction_mu_omega_gt_2']:.3f},{saturation_stats['fraction_mu_omega_gt_4']:.3f}\n"
                f"  corr(|mu|, advantage): v={corr_mu_v_adv:.3f} omega={corr_mu_omega_adv:.3f}\n"
                f"  mean|hidden|={mean_abs_hidden:.3f} growth_ratio_mu(mean|mu|/mean|hidden|): "
                f"v={growth_ratio_mu_v:.4f} omega={growth_ratio_mu_omega:.4f}",
                flush=True,
            )

    policy_losses, value_losses, entropies, approx_kls, clip_fractions = [], [], [], [], []
    kl_rows: list[dict] = []
    if do_update:
        update_stats = _ppo_update(
            net, actor_optimizer, actor_params, critic_optimizer, critic_params, value_rms,
            segments, cfg, N, is_continuous, device, round_label, round_idx,
            kl_diagnostics_csv_path, target_kl,
        )
        policy_losses, value_losses = update_stats["policy_losses"], update_stats["value_losses"]
        entropies, approx_kls = update_stats["entropies"], update_stats["approx_kls"]
        clip_fractions, kl_rows = update_stats["clip_fractions"], update_stats["kl_rows"]
    elif not defer_update:
        print(
            f"[{round_label}] skipping PPO update: only {upd_T} real ticks available "
            f"(< --min-update-ticks={min_update_ticks}) -- "
            + (
                f"kept for the next update ({len(segments)} round(s) accumulated)"
                if rollout_carry is not None else "discarded (no --accumulate-short-rounds)"
            )
            + "; round still counted for history/anneal-gate tracking.", flush=True,
        )
    if do_update and rollout_carry is not None:
        rollout_carry.clear()
    if kl_diagnostics_csv_path is not None:
        for row in kl_rows:
            append_history_row(kl_diagnostics_csv_path, row)
    if tick_trace_csv_path is not None:
        for row in tick_trace_rows:
            append_history_row(tick_trace_csv_path, row)
        print(
            f"[{round_label}] wrote {len(tick_trace_rows)} per-tick trace rows to "
            f"{tick_trace_csv_path}", flush=True,
        )

    with torch.no_grad():
        target_var = value_targets.var()
        explained_variance = (
            float("nan") if target_var.item() == 0
            else float((1 - (value_targets - value_buf[:-1]).var() / target_var).item())
        )
        value_pred_mean = float(value_buf[:-1].mean())
        value_target_mean = float(value_targets.mean())
        value_rms_mean = value_rms.mean
        value_rms_std = value_rms.var ** 0.5

    action_stats: dict[str, float] = {}
    mean_action_modification = float("nan")
    log_std_v = log_std_omega = float("nan")
    mean_distance_start = mean_distance_end = mean_progress_per_100_ticks = float("nan")
    if is_continuous:
        with torch.no_grad():
            v_actions = action_buf[..., 0].reshape(-1)
            omega_actions = action_buf[..., 1].reshape(-1)
            action_stats = {
                "mean_v": float(v_actions.mean()), "std_v": float(v_actions.std()),
                "mean_abs_v": float(v_actions.abs().mean()),
                "mean_omega": float(omega_actions.mean()), "mean_abs_omega": float(omega_actions.abs().mean()),
                "std_omega": float(omega_actions.std()), "max_abs_omega": float(omega_actions.abs().max()),
                "fraction_high_omega": float((omega_actions.abs() > 0.8).float().mean()),
            }
            executed_buf = torch.stack(buffer["executed"])
            mean_action_modification = float((action_buf - executed_buf).abs().sum(dim=-1).mean())
            clamped_log_std = net.effective_log_std().detach()
            log_std_v, log_std_omega = float(clamped_log_std[0]), float(clamped_log_std[1])
        if distance_first:
            mean_distance_start = sum(distance_first.values()) / len(distance_first)
            mean_distance_end = sum(distance_last.values()) / len(distance_last)
            mean_progress_per_100_ticks = (mean_distance_start - mean_distance_end) / max(1, T) * 100.0

    _safe_mean = lambda values: float(np.mean(values)) if values else float("nan")
    summary = {
        "ticks": T,
        "mean_reward": float(reward_buf.mean().item()),
        "update_skipped": int(not do_update),
        "update_ticks": upd_T if do_update else 0,
        "actor_lr": actor_optimizer.param_groups[0]["lr"],
        "policy_loss": _safe_mean(policy_losses),
        "value_loss": _safe_mean(value_losses),
        "entropy": _safe_mean(entropies),
        "approx_kl": _safe_mean(approx_kls),
        "clip_fraction": _safe_mean(clip_fractions),
        "explained_variance": explained_variance,
        "value_pred_mean": value_pred_mean,
        "value_target_mean": value_target_mean,
        "value_rms_mean": value_rms_mean,
        "value_rms_std": value_rms_std,
        "log_std_v": log_std_v,
        "log_std_omega": log_std_omega,
        "mean_action_modification": mean_action_modification,
        "mean_distance_start": mean_distance_start,
        "mean_distance_end": mean_distance_end,
        "mean_progress_per_100_ticks": mean_progress_per_100_ticks,
        "pickup_latency_mean_ticks": _safe_mean(pickup_latency_ticks),
        "pickup_latency_count": len(pickup_latency_ticks),
        "delivery_latency_mean_ticks": _safe_mean(delivery_latency_ticks),
        "delivery_latency_count": len(delivery_latency_ticks),
        "lidar_stall_event_ticks": lidar_stall_event_ticks,
        **_lidar_stall_summary(
            lidar_stall_streak_lengths, coordinator, T, lidar_stall_streak_at_round_start,
        ),
        **action_stats,
        **saturation_stats,
        **totals,
        **{f"reward_{key}": val for key, val in reward_breakdown_totals.items()},
    }
    update_status = (
        "collected (PPO update deferred to the multi-world step)" if defer_update
        else "PPO update done" if not summary["update_skipped"] else "PPO update skipped"
    )
    if defer_update:
        summary["_segment"] = segment
    print(
        f"[{round_label}] {update_status} on {upd_T if do_update else 0} real ticks "
        f"(this round: {T}, {N} robots): "
        f"policy_loss={summary['policy_loss']:.4f} value_loss={summary['value_loss']:.4f} "
        f"entropy={summary['entropy']:.4f} approx_kl={summary['approx_kl']:.5f} "
        f"clip_fraction={summary['clip_fraction']:.3f} explained_variance={summary['explained_variance']:.3f} "
        f"deliveries={totals['deliveries']} "
        f"collisions={totals['collisions']} interventions={totals['interventions']} "
        f"wall_block={totals['wall_block']} stuck_events={totals['stuck_events']} "
        f"wall_proximity_penalty={totals['wall_proximity_penalty']:.3f}\n"
        f"  pickup_latency_mean_ticks={summary['pickup_latency_mean_ticks']:.1f} "
        f"(n={summary['pickup_latency_count']}) "
        f"delivery_latency_mean_ticks={summary['delivery_latency_mean_ticks']:.1f} "
        f"(n={summary['delivery_latency_count']})\n"
        f"  lidar_stall (frac of robot-ticks): isolated={summary['lidar_stall_isolated_frac']:.3f} "
        f"sustained={summary['lidar_stall_sustained_frac']:.3f} "
        f"max_streak={summary['max_lidar_stall_streak']} "
        f"(raw_ticks={summary['lidar_stall_event_ticks']}/{T})\n"
        f"  action_modification={summary['mean_action_modification']:.4f} "
        f"log_std=(v={log_std_v:.3f},omega={log_std_omega:.3f}) "
        f"progress_per_100_ticks={summary['mean_progress_per_100_ticks']:.3f} "
        f"fraction_high_omega={action_stats.get('fraction_high_omega', float('nan')):.3f}\n"
        f"  value_pred_mean={value_pred_mean:.4f} value_target_mean={value_target_mean:.4f} "
        f"(gap={value_target_mean - value_pred_mean:+.4f}) value_rms(mean={value_rms_mean:.4f},"
        f"std={value_rms_std:.4f})",
        flush=True,
    )
    return summary


# Env seed offset between Webots worlds (seed + k * stride).
WORLD_SEED_STRIDE = 1009

# Summary keys that are summed across worlds (others are tick-weighted means).
_SUMMED_SUMMARY_KEYS = {
    "ticks", "deliveries", "collisions", "interventions", "wall_block", "stuck_events",
    "wall_proximity_penalty", "pickup_latency_count", "delivery_latency_count",
    "lidar_stall_event_ticks", "lidar_stall_isolated_ticks", "lidar_stall_sustained_ticks",
}
# Protects the shared value normalizer across world threads.
_VALUE_RMS_LOCK = threading.Lock()


def _merge_world_summaries(summaries: list[dict]) -> dict:
    """Combine per-world round summaries into one history row."""
    merged: dict = {}
    weights = [float(s["ticks"]) for s in summaries]
    for key in summaries[0]:
        values = [s.get(key) for s in summaries]
        if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
            merged[key] = values[0]
        elif key in _SUMMED_SUMMARY_KEYS or key.startswith("reward_"):
            merged[key] = sum(values)
        elif key == "max_lidar_stall_streak":
            merged[key] = max(values)
        else:
            pairs = [(v, w) for v, w in zip(values, weights) if not math.isnan(v) and w > 0]
            total_w = sum(w for _, w in pairs)
            merged[key] = sum(v * w for v, w in pairs) / total_w if total_w > 0 else float("nan")
    merged["worlds"] = len(summaries)
    for k, s in enumerate(summaries):
        merged[f"world_{k}_ticks"] = s["ticks"]
        merged[f"world_{k}_deliveries"] = s["deliveries"]
    return merged


def collect_and_update_multi_world(
    coordinators: list, net, actor_optimizer, critic_optimizer, value_rms: RunningMeanStd,
    agents: list[str], device: str, cfg: PPOConfig, ticks: int, round_label: str,
    resets: list[bool], is_continuous: bool, round_idx: int, kl_diagnostics_csv_path: str | None,
    min_update_ticks: int, rollout_carry: list | None, target_kl: float | None,
) -> tuple[dict | None, list[dict | None]]:
    """Collect one round in every Webots world in parallel, then run one shared PPO update."""
    results: list[dict | None] = [None] * len(coordinators)
    errors: list[BaseException | None] = [None] * len(coordinators)

    def _collect(k: int) -> None:
        try:
            results[k] = collect_and_update(
                coordinators[k], net, actor_optimizer, critic_optimizer, value_rms, agents, device,
                cfg, ticks, f"{round_label} w{k}", reset=resets[k], is_continuous=is_continuous,
                round_idx=round_idx, min_update_ticks=0, defer_update=True,
            )
        except BaseException as exc:
            errors[k] = exc

    threads = [threading.Thread(target=_collect, args=(k,), daemon=True) for k in range(len(coordinators))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for k, exc in enumerate(errors):
        if exc is not None:
            raise RuntimeError(f"world {k} failed during collection") from exc

    valid = [r for r in results if r is not None]
    if not valid:
        return None, results
    segments = [r.pop("_segment") for r in valid]
    if rollout_carry is not None:
        rollout_carry.extend(segments)
        segments = rollout_carry
    upd_T = sum(s["lidar"].shape[0] for s in segments)
    do_update = min_update_ticks <= 0 or upd_T >= min_update_ticks
    merged = _merge_world_summaries(valid)
    stats = {"policy_losses": [], "value_losses": [], "entropies": [], "approx_kls": [], "clip_fractions": []}
    if do_update:
        actor_params, critic_params = _actor_critic_parameter_groups(net)
        stats = _ppo_update(
            net, actor_optimizer, actor_params, critic_optimizer, critic_params, value_rms,
            segments, cfg, len(agents), is_continuous, device, round_label, round_idx,
            kl_diagnostics_csv_path, target_kl,
        )
        if kl_diagnostics_csv_path is not None:
            for row in stats["kl_rows"]:
                append_history_row(kl_diagnostics_csv_path, row)
        if rollout_carry is not None:
            rollout_carry.clear()
    _safe_mean = lambda values: float(np.mean(values)) if values else float("nan")
    merged.update({
        "update_skipped": int(not do_update),
        "update_ticks": upd_T if do_update else 0,
        "actor_lr": actor_optimizer.param_groups[0]["lr"],
        "policy_loss": _safe_mean(stats["policy_losses"]),
        "value_loss": _safe_mean(stats["value_losses"]),
        "entropy": _safe_mean(stats["entropies"]),
        "approx_kl": _safe_mean(stats["approx_kls"]),
        "clip_fraction": _safe_mean(stats["clip_fractions"]),
    })
    per_world = " ".join(
        f"w{k}={r['deliveries']}/{r['ticks']}" if r is not None else f"w{k}=FAILED"
        for k, r in enumerate(results)
    )
    print(
        f"[{round_label}] {'PPO update done' if do_update else 'PPO update skipped'} on "
        f"{upd_T if do_update else 0} real ticks from {len(valid)}/{len(coordinators)} world(s) "
        f"(deliveries/ticks per world: {per_world}): policy_loss={merged['policy_loss']:.4f} "
        f"value_loss={merged['value_loss']:.4f} entropy={merged['entropy']:.4f} "
        f"approx_kl={merged['approx_kl']:.5f} clip_fraction={merged['clip_fraction']:.3f} "
        f"deliveries={merged['deliveries']} interventions={merged['interventions']}",
        flush=True,
    )
    return merged, results


def main() -> None:
    """Parse arguments, build env/network/coordinators and run the training rounds."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-config", default=os.path.join(ROOT, "configs", "env_small.yaml"))
    parser.add_argument("--train-config", default=os.path.join(ROOT, "configs", "train_config.yaml"))
    parser.add_argument("--rounds", type=int, default=1, help="number of collect+update fine-tuning rounds")
    parser.add_argument("--ticks", type=int, default=30, help="real Webots ticks to collect per round")
    parser.add_argument(
        "--robots", type=int, default=None,
        help="MUST match --robots used when the Webots world was generated, or extra robots "
        "never receive a command and sit motionless (see run_lockstep_demo.py's matching note)",
    )
    parser.add_argument("--arrival-timeout", type=float, default=30.0)
    parser.add_argument(
        "--update-epochs", type=int, default=None,
        help="override train_config.yaml's PPOConfig.update_epochs for this invocation only "
        "(the yaml is shared with scripts/train_mappo.py's grid path). Added 2026-09-10: "
        "approx_kl/clip_fraction staying low across many rounds is the signal that there's room "
        "to reuse each expensive real-Webots round's data for more gradient steps before hitting "
        "the trust region -- see the CLI help text right above cfg.update_epochs' override site "
        "in main() for the full reasoning. Defaults to the yaml's value (4) if not passed.",
    )
    parser.add_argument(
        "--minibatches", type=int, default=None,
        help="override train_config.yaml's PPOConfig.minibatches for this invocation only, same "
        "reasoning/scoping as --update-epochs. Defaults to the yaml's value (4) if not passed.",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=None,
        help="override train_config.yaml's PPOConfig.learning_rate for this invocation only, same "
        "reasoning/scoping as --update-epochs. Defaults to the yaml's value (0.0003) if not passed.",
    )
    parser.add_argument(
        "--target-kl", type=float, default=0.02,
        help="stop further policy epochs within a round once an epoch's mean approx_kl exceeds "
        "1.5x this value (value epochs still run). Guard added 2026-09-28 so --learning-rate can be "
        "raised safely: run 60 sat at approx_kl~0.003 with lr=1e-4 and 15 epochs. Pass a negative "
        "value to disable.",
    )
    parser.add_argument(
        "--actor-hidden-dim", type=int, default=64,
        help="continuous-actions only (see ContinuousActorCriticNet's 2026-09-13 separate-"
        "encoders docstring): hidden width of the actor's OWN encoder (no longer shared with the "
        "critic). CleanRL/SB3-style separate networks default to a narrower actor than critic.",
    )
    parser.add_argument(
        "--critic-hidden-dim", type=int, default=256,
        help="continuous-actions only: hidden width of the critic's OWN, wider encoder (see "
        "--actor-hidden-dim's help) -- it processes every robot's concatenated raw features at "
        "once, a much wider input than the actor's single-robot one, hence the extra capacity.",
    )
    parser.add_argument(
        "--actor-log-std-init", type=float, default=None,
        help="continuous-actions only: force actor_log_std down to at most this value right "
        "after warm-starting (clamp MAX, not min -- the existing anneal-floor clamp a few lines "
        "below only ever clamps UP to the current floor, never down). User-requested (2026-09-13) "
        "after a fresh-architecture cold start produced near-zero actor_mean (gain=0.01 init) "
        "combined with wide exploration std (log_std pinned at the LOG_STD_MIN=-0.5 anneal floor "
        "for a brand-new lineage) -- confirmed live: 37/38 early rounds delivered zero, and the "
        "anti-pileup circuit breaker fired in effectively every round, consistent with mean~0 "
        "actor output making the fleet's motion noise-dominated (more erratic wandering, more "
        "mutual close encounters) rather than purposeful. Narrowing this (e.g. -1.0 to -1.2) "
        "trades exploration breadth for calmer, more mean-driven motion while the actor is still "
        "near its random init. Not applied at all if left at the default None -- purely additive, "
        "existing lineages that never needed this are unaffected.",
    )
    parser.add_argument(
        "--velocity-tick-sleep-s", type=float, default=None,
        help="continuous-actions only: real wall-clock seconds LockstepCoordinator.run sleeps "
        "after commanding cmd_vel each tick, before sampling the resulting physical pose -- "
        "distinct from continuous_kinematics.DT_SECONDS, which stays fixed at 1.0 for the "
        "reward/kinematics model's 'one RL tick = 1s of commanded motion' semantics regardless of "
        "this value. Added 2026-09-01: under --mode=fast --no-rendering, confirmed live (a "
        "KeyboardInterrupt during a run landed exactly on this sleep call) that this fixed "
        "DT_SECONDS=1.0s real sleep -- not physical robot speed -- was the dominant per-tick "
        "wall-clock cost, since Webots' own simulated clock already runs faster than real time in "
        "fast mode independent of how long this script waits before checking. Defaults to None, "
        "which falls back to DT_SECONDS (the prior, unchanged behavior) -- pass a smaller value "
        "(e.g. 0.3) to test faster real-tick collection. NOT YET validated against real training "
        "data for its effect on the per-tick reward balance (time_cost/progress/omega-penalty were "
        "all calibrated assuming ~1s of motion per tick) -- watch progress/time_cost reward "
        "magnitudes in the history CSV after changing this, same as any other constant here.",
    )
    parser.add_argument(
        "--disable-legacy-retreats", action="store_true",
        help="continuous-actions only: disable the old per-robot physical/lidar retreat "
        "overrides. The safety shield remains enabled, and a real pair closer than the global "
        "collision threshold still ends the round so the next round physically resets the fleet.",
    )
    parser.add_argument(
        "--global-collision-reset-m", type=float, default=0.40,
        help="continuous-actions only: end the current round when two reported real robot poses "
        "are closer than this distance, forcing the next round's physical reset. Default: 0.40m.",
    )
    parser.add_argument(
        "--gridlock-reset-streak", type=int, default=3,
        help="force a physical reset after this many CONSECUTIVE completed rounds (full "
        "--ticks collected, not the partial-round case above) with zero deliveries. Added "
        "2026-09-04 after a confirmed live incident on the checkpoint_39 lineage: two robots "
        "wedged against each other (not a wall -- the existing partial-round reset trigger never "
        "fired since every round still collected its full 1000/1000 ticks) and stayed wedged for "
        "40+ CONSECUTIVE rounds (~12+ real hours) -- interventions/round jumped from a ~110-140 "
        "healthy baseline to a near-constant ~430-545 (the shield vetoing the same conflict every "
        "tick) while deliveries stayed at 0 almost every round and mean_reward settled onto an "
        "unusually flat, repeating plateau (11 straight rounds within 0.004 of each other), with "
        "nothing in the existing reset logic able to detect or clear it. Calibrated against this "
        "lineage's own healthy history: rounds 1-160 never produced more than 2 consecutive "
        "zero-delivery rounds, so 3 has real margin against a false trigger during ordinary "
        "sparse-delivery stretches while still catching a real stall within a bounded number of "
        "rounds instead of potentially running for the rest of the whole training session. Set to "
        "0 to disable.",
    )
    parser.add_argument("--checkpoint-in", default=None, help="warm-start from an existing grid-trained checkpoint")
    parser.add_argument(
        "--checkpoint-out", default=os.path.join(ROOT, "outputs", "mappo_checkpoint_webots_finetuned.pt"),
        help="overwritten after EVERY round, so an interrupted run keeps its progress",
    )
    parser.add_argument(
        "--history-csv", default=None,
        help="append one row per successful round here (round, ticks, mean_reward, policy_loss, "
        "value_loss, entropy, timestamp) -- unlike scripts/train_mappo.py, this script previously "
        "wrote no persistent history at all, so a multi-hour real-Webots run's reward trend was "
        "only ever visible in live terminal output, impossible to review afterward. Defaults to "
        "<checkpoint-out without extension>_history.csv.",
    )
    parser.add_argument(
        "--kl-diagnostics-csv", default=None,
        help="per-PPO-minibatch KL/ratio diagnostics (round, epoch, minibatch, approx_kl, "
        "clip_fraction, mean/max/min_ratio, update_kl) -- see collect_and_update's own comment: "
        "added 2026-09-09 to distinguish cumulative drift across a round's update_epochs*"
        "minibatches sequential gradient steps (round-level approx_kl in --history-csv, which "
        "always compares against the fixed rollout policy) from the size of any single update "
        "(update_kl: KL(policy_before_this_update || policy_after_this_update), local to one "
        "optimizer.step()). Defaults to <checkpoint-out without extension>_kl_diagnostics.csv.",
    )
    parser.add_argument(
        "--no-kl-diagnostics", action="store_true",
        help="skip the per-minibatch KL diagnostics CSV above (costs one extra forward pass per "
        "minibatch -- update_epochs*minibatches per round -- to compute update_kl).",
    )
    parser.add_argument(
        "--tick-trace-csv", default=None,
        help="per-tick, per-robot navigation trace (raw action, mu/noise decomposition, shield's "
        "final velocity, nearest-obstacle/wall-clearance distance, intervention/veto/wall_block "
        "flags, robot status, distance to target) for ONLY the rounds listed in "
        "--tick-trace-rounds -- added 2026-09-11 to answer 'exactly where does raw action diverge "
        "from executed action, and why' for a specific round (e.g. a round with high "
        "mean_action_modification but zero interventions/wall_block) instead of speculating from "
        "the round-level aggregates alone. Defaults to <checkpoint-out without extension>"
        "_tick_trace.csv when --tick-trace-rounds is set.",
    )
    parser.add_argument(
        "--tick-trace-rounds", default=None,
        help="comma-separated round numbers (this invocation's own 1-based round_idx, e.g. "
        "'18,19,20') to write --tick-trace-csv rows for. No rows are written for any other round "
        "-- this is a targeted trace, not always-on logging, since it's one row per (tick, robot).",
    )
    parser.add_argument(
        "--min-update-ticks", type=int, default=300,
        help="skip the PPO gradient update (policy + value) entirely for a round that collected "
        "fewer than this many real ticks -- confirmed live: a 10-100-tick round (an early "
        "arrival_timeout_s stall, see MIN_TICKS_STALL_BREAK_FRAC's own comment) still ran a full "
        "update_epochs*minibatches update on a handful of samples, producing visibly degenerate "
        "steps (approx_kl/clip_fraction spiking well past every other round's range) that then had "
        "to be manually discounted from any trend reading. The round is still fully counted in "
        "history/the exploration-anneal gate (deliveries and ticks are real either way) -- only the "
        "gradient step is skipped; `update_skipped` in the history CSV marks which rounds this hit. "
        "0 disables (always update, the old behavior).",
    )
    parser.add_argument(
        "--team-reward-share", type=float, default=0.5,
        help="continuous-actions only: fraction of every pickup/delivery reward pooled and split "
        "equally across all robots (MAPPO cooperative reward), so yielding / clearing the way also "
        "pays. 0 = purely individual rewards (previous behavior).",
    )
    parser.add_argument(
        "--agent-state", action=argparse.BooleanOptionalAction, default=True,
        help="continuous-actions only: feed the env's agent_state field (status one-hot, last "
        "executed command, measured motion) to actor and critic. Warm-starts from older checkpoints "
        "function-preservingly (zero weights for the new inputs).",
    )
    parser.add_argument(
        "--obs-norm", action=argparse.BooleanOptionalAction, default=True,
        help="continuous-actions only: online running mean/std normalization of the per-robot "
        "inputs, updated between PPO updates and kept function-preserving.",
    )
    parser.add_argument(
        "--soft-log-std", action=argparse.BooleanOptionalAction, default=True,
        help="continuous-actions only: learnable exploration std with smooth bounds and a FIXED "
        "entropy coefficient (MAPPO standard) instead of the hard log-std floor + anneal/gate "
        "schedules (--log-std-anneal-*, --entropy-coef-anneal-*), which are ignored in this mode. "
        "--actor-log-std-init then SETS the starting log-std instead of capping it.",
    )
    parser.add_argument(
        "--layer-norm", action=argparse.BooleanOptionalAction, default=False,
        help="continuous-actions only: LayerNorm after every hidden layer (MAPPO default). Changes "
        "the network function, so it breaks warm starts -- use it for a NEW lineage only.",
    )
    parser.add_argument(
        "--value-loss", choices=["mse", "huber"], default="huber",
        help="critic loss on the normalized target: huber (MAPPO default) or mse (previous).",
    )
    parser.add_argument(
        "--value-clip", type=float, default=0.2,
        help="clip the new value prediction to old +- this (MAPPO value clipping); 0 disables.",
    )
    parser.add_argument(
        "--desired-kl", type=float, default=0.01,
        help="adaptive actor learning rate: /1.5 when approx_kl > 2x this, x1.5 when < half of it, "
        "within [1e-5, 3e-3]. 0 keeps --learning-rate fixed (previous behavior).",
    )
    parser.add_argument(
        "--shield-physical-walls-only", action=argparse.BooleanOptionalAction, default=False,
        help="continuous-actions only: the shield ignores the map's shelf blocks (they exist only "
        "logically; A* still routes around them) and checks only the perimeter walls and robots.",
    )
    parser.add_argument(
        "--shield-radius", type=float, default=None,
        help="continuous-actions only: shield robot-robot distance on open floor (default 0.8 m).",
    )
    parser.add_argument(
        "--shield-pickup-radius", type=float, default=None,
        help="continuous-actions only: shield distance around a robot picking up (default 1.3 m).",
    )
    parser.add_argument(
        "--shield-active-resolution", action=argparse.BooleanOptionalAction, default=False,
        help="continuous-actions only: shield conflict resolution for when its velocities are "
        "executed (stopped robots keep turning, the lower-priority robot backs away, working "
        "deadlock reversal).",
    )
    parser.add_argument(
        "--execute-shield", action=argparse.BooleanOptionalAction, default=False,
        help="continuous-actions only: send the env shield's corrected velocities to the real "
        "robots instead of the policy's action. Off by default -- the shield is too conservative "
        "to drive real robots (it froze most of them in run 67); rewards come from the real "
        "outcome either way.",
    )
    parser.add_argument(
        "--entropy-coef", type=float, default=None,
        help="override the train config's entropy coefficient (fixed for the whole run under "
        "--soft-log-std).",
    )
    parser.add_argument(
        "--worlds", type=int, default=1,
        help="number of Webots worlds to collect from in parallel (added 2026-09-29). Every round "
        "runs --ticks ticks in EACH world at the same time and does ONE PPO update on all of them "
        "(so N worlds = N x the data per update, same wall-clock per round). Each world needs its "
        "own Webots instance (its own --port) and its own set of extern controllers started with "
        "a distinct ROS_DOMAIN_ID -- see scripts/run_extern_controllers.sh's 4th argument. World "
        "k's env is seeded --seed + k*1009, so robots spawn in different free cells (always inside "
        "the walled map) and receive different orders in each world.",
    )
    parser.add_argument(
        "--world-domain-ids", default=None,
        help="comma-separated ROS_DOMAIN_ID of each world's controllers, e.g. '0,1'. Defaults to "
        "0..worlds-1 when --worlds > 1; with --worlds 1 the process's own ROS_DOMAIN_ID is used "
        "(the previous behavior).",
    )
    parser.add_argument(
        "--respawn-on-collision", action=argparse.BooleanOptionalAction, default=True,
        help="continuous-actions only: on a real collision / anti-pileup, teleport ONLY the robots "
        "involved to clear cells, penalize them (REWARD_PHYSICAL_COLLISION_CONTINUOUS) and end "
        "just their episode, while the round keeps going to its full --ticks -- instead of ending "
        "the round and physically resetting all robots (added 2026-09-28). Falls back to ending "
        "the round if no clear cell exists. --no-respawn-on-collision restores the old behavior.",
    )
    parser.add_argument(
        "--accumulate-short-rounds", action=argparse.BooleanOptionalAction, default=True,
        help="keep the rollout of a round shorter than --min-update-ticks and run the PPO update "
        "once the accumulated rounds reach it, instead of discarding that round's ticks (see "
        "collect_and_update's rollout_carry). Added 2026-09-28: in run 60, 65 of 80 rounds ended "
        "early (collision resets/anti-pileup) and ~50%% of all real ticks were thrown away. "
        "--no-accumulate-short-rounds restores the old discard behavior.",
    )
    parser.add_argument(
        "--no-best-checkpoint", action="store_true",
        help="skip tracking a separate best-reward checkpoint (matches scripts/train_mappo.py's "
        "flag of the same name). Rationale (2026-08-30): --checkpoint-out is overwritten every "
        "round unconditionally, so the final file on disk is whatever round the run happened to "
        "end on -- PPO has no monotonic-improvement guarantee (same reason train_mappo.py's "
        "best_checkpoint_path exists), and this was confirmed live on the outputs/"
        "mappo_checkpoint_continuous_webots_finetuned_32 lineage: deliveries/round fell 3x "
        "(0.113 -> 0.038) across the run's quartiles while the final round's raw mean_reward "
        "happened to look fine, i.e. --checkpoint-out's last save was not the best-performing one. "
        "Unlike train_mappo.py, this compares a TRAILING AVERAGE of mean_reward over "
        "--best-checkpoint-window rounds, not one round's raw value -- a single round here is only "
        "~50-60 real ticks across 5 robots, so one lucky delivery (REWARD_DELIVERY_CONTINUOUS=100 "
        "spread over the round) can spike a single round's mean_reward without reflecting durable "
        "behavior.",
    )
    parser.add_argument(
        "--best-checkpoint-window", type=int, default=20,
        help="rounds averaged for the --no-best-checkpoint comparison. Needs at least this many "
        "successful rounds collected in THIS run before a best checkpoint is saved at all (a "
        "partial window early in a run is too noisy to trust) -- same window size as "
        "--log-std-anneal-gate-window by default since both exist to smooth this run's own "
        "small-sample-size noise, not because the two need to move together.",
    )
    parser.add_argument(
        "--best-checkpoint-min-activity", type=float, default=10.0,
        help="minimum trailing-window average of (interventions + wall_block) per round required "
        "before a new trailing-mean_reward high is honored as a best checkpoint. Added 2026-09-02 "
        "after a confirmed live incident on the checkpoint_38 lineage: rounds 77-96 had the fleet "
        "effectively stop engaging with the map (3 of 5 robots sat IDLE the entire window with no "
        "order ever assigned to them -- root cause not yet fully isolated, suspected interaction "
        "between --velocity-tick-sleep-s and still-imperfect navigation -- while the other 2 "
        "wandered without completing their route), which mean_reward alone read as a steady "
        "IMPROVEMENT (fewer per-tick collision/omega/wall-proximity penalties fire when the fleet "
        "barely moves) and --best-checkpoint-out's own trailing-mean_reward check saved that exact "
        "degenerate state as 'best' on 10 consecutive rounds -- a worse regression than the one "
        "--no-best-checkpoint's docstring already documents (deliveries falling while raw reward "
        "looked fine): here reward went positive specifically BECAUSE the fleet went idle, so that "
        "check alone couldn't catch it. interventions+wall_block was ~3/round during the incident "
        "vs. tens-to-hundreds/round in every healthy stretch checked (both before and after) -- "
        "10.0 sits well below normal engagement with real margin above the incident's level. Set "
        "to 0 to disable and restore the old mean_reward-only check.",
    )
    parser.add_argument(
        "--best-checkpoint-out", default=None,
        help="path for the best-trailing-average checkpoint. Defaults to "
        "<checkpoint-out without extension>_best.pt.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--continuous-actions", action="store_true",
        help="fine-tune a ContinuousActorCriticNet checkpoint via ContinuousLockstepCoordinator "
        "((v, omega) velocity commands) instead of the discrete grid path.",
    )
    parser.add_argument(
        "--log-std-anneal-rounds", type=int, default=300,
        help="continuous-actions only: linearly lower the policy's exploration-noise floor "
        "(ContinuousActorCriticNet.LOG_STD_MIN -> LOG_STD_MIN_FINAL) over this many CUMULATIVE "
        "rounds (the checkpoint's own saved `update` counter plus this run's round_idx, so "
        "progress survives restarting the script from a checkpoint -- see train_ppo's identical "
        "anneal for why a fixed floor alone isn't enough). User-reported symptom this targets "
        "(2026-08-29): robots still spin/turn excessively while exploring even with the reward-"
        "side incentives (omega penalty, progress, stuck-turn) already in place -- unlike "
        "scripts/train_mappo.py, this script never called set_log_std_floor at all, so the "
        "exploration-noise floor stayed at its widest (LOG_STD_MIN) for the entire Webots fine-"
        "tuning history regardless of how many rounds ran; a wide floor means large *random* "
        "omega samples every tick, which reads as excessive turning independent of whatever the "
        "reward is teaching the policy to prefer on average. Set to 0 to disable (matches the old "
        "always-LOG_STD_MIN behavior). The 'CUMULATIVE rounds' counter this counts down over is "
        "gated by --log-std-anneal-gate-deliveries by default -- see its help.",
    )
    parser.add_argument(
        "--log-std-anneal-omega-rate-multiplier", type=float, default=1.0,
        help="continuous-actions only: anneal the omega action dim's log_std floor this many "
        "times FASTER than v's (v always follows --log-std-anneal-rounds unchanged). "
        "User-requested (2026-09-12), confirmed live: log_std_v and log_std_omega are independent "
        "learned parameters sharing one floor schedule -- omega tracked that shared floor exactly "
        "every round while v sat measurably above it, i.e. the two channels wanted different "
        "amounts of noise and only the shared (slower, v-paced) schedule was available to either. "
        "A value of 5.0-10.0 here lets omega's floor reach LOG_STD_MIN_FINAL in roughly "
        "--log-std-anneal-rounds/5 to /10 rounds while v keeps its original pace. 1.0 (default) "
        "keeps the old shared-floor behavior exactly.",
    )
    parser.add_argument(
        "--entropy-coef-anneal-rounds", type=int, default=None,
        help="linearly decay PPOConfig.entropy_coef (from train_config.yaml) toward ~0 over this "
        "many CUMULATIVE rounds (same start_update-relative counting as --log-std-anneal-rounds). "
        "User-reported (2026-08-29): entropy_coef is a fixed positive weight added to the loss as "
        "-entropy_coef*entropy, i.e. the optimizer is continuously REWARDED for keeping entropy "
        "high -- with nothing decaying it, that pressure never relents, which is a second, "
        "independent reason (besides log_std_floor never being lowered) that exploration noise "
        "doesn't shrink as training progresses. Defaults to the same value as "
        "--log-std-anneal-rounds if not given (so one knob controls both by default); pass "
        "explicitly to decouple them, or 0 to disable (keeps entropy_coef fixed, the old "
        "behavior).",
    )
    parser.add_argument(
        "--entropy-coef-anneal-floor-frac", type=float, default=0.25,
        help="entropy_coef never decays below this fraction of its starting (train_config.yaml) "
        "value -- a small residual exploration pressure, not a hard cliff to exactly 0. Raised from "
        "0.02 (2026-09-01 diagnosis, from outputs/mappo_checkpoint_continuous_webots_finetuned_36's "
        "347-round run/history CSV): `anneal_round` is cumulative across this checkpoint's ENTIRE "
        "warm-started lineage relative to its one-time unlock round, and that lineage is already "
        "~1000+ cumulative rounds deep, far past the default 300-round --entropy-coef-anneal-rounds "
        "budget -- so entropy_coef was pinned at floor_frac*base_entropy_coef for this run's entire "
        "duration before round 1 even started. At the old 0.02 (entropy_coef=0.0002 with the default "
        "0.01 base), that residual is too weak to oppose actor_log_std's policy-loss gradient at "
        "all: plot_webots_finetune_history.py's policy-entropy panel shows entropy declining "
        "smoothly and *without ever plateauing* across all 347 rounds (1.85 -> 1.10), i.e. an "
        "unopposed collapse, not a schedule converging to some equilibrium -- consistent with "
        "mean_reward and deliveries/round both staying flat over the same stretch despite continued "
        "training (the policy has nothing left to explore with). 0.25 keeps the anneal's early-run "
        "shape intact (progress=0 still gives full base entropy_coef) but stops the eventual floor "
        "from being negligible. Re-validate against real training data once tried, same as every "
        "other constant this project's fine-tuning history has tuned this way.",
    )
    parser.add_argument(
        "--log-std-anneal-gate-deliveries", type=float, default=0.15,
        help="continuous-actions only: don't let --log-std-anneal-rounds/--entropy-coef-anneal-"
        "rounds progress start counting down until the trailing average of `deliveries` per round "
        "(over the last --log-std-anneal-gate-window rounds that produced an update) reaches this "
        "value. Rationale (2026-08-30, from outputs/mappo_checkpoint_continuous_webots_"
        "finetuned_32_history.csv): --log-std-anneal-rounds counts CUMULATIVE rounds across this "
        "checkpoint's entire warm-started fine-tuning lineage (`update` in the checkpoint, same as "
        "anneal_lr), so by the time a 6th chained fine-tuning invocation started, cumulative_round "
        "was already ~270 -- past the (default 300-round) anneal budget within ~30 more rounds. The "
        "exploration-noise floor then sat at its narrowest for the rest of that ~430-round run while "
        "deliveries/round *fell* (0.113 -> 0.038 across quartiles) instead of rising -- the policy "
        "finished exploring before it had found reliable delivery behavior to exploit, then had too "
        "little noise left to find its way back out. Gating the countdown on demonstrated delivery "
        "competence (not just elapsed rounds) means a lineage that's still bad at deliveries keeps "
        "its exploration noise wide regardless of cumulative_round. The unlock round, once reached, "
        "is saved in the checkpoint (`extra.anneal_unlock_round`) so it -- like cumulative_round --"
        "survives restarting this script from that checkpoint. Set to 0 to disable gating and "
        "restore the old always-counts-from-cumulative_round behavior.",
    )
    parser.add_argument(
        "--log-std-anneal-gate-window", type=int, default=20,
        help="number of most recent successful rounds (this invocation's own round_summaries) "
        "averaged for the --log-std-anneal-gate-deliveries check. Requires at least this many "
        "rounds collected in THIS run before the gate can open -- a checkpoint's prior lineage "
        "performance isn't replayed, only round_summaries collected since this invocation started.",
    )
    parser.add_argument(
        "--log-std-anneal-relock-deliveries", type=float, default=None,
        help="continuous-actions only: once unlocked, RELOCK the exploration-anneal gate (reset "
        "anneal_unlock_round to None, restoring full log_std/entropy_coef exploration) if the "
        "trailing --log-std-anneal-gate-window average of deliveries/round falls back below this "
        "value. Added 2026-09-01: --log-std-anneal-gate-deliveries' own unlock mechanism is a "
        "one-way ratchet -- once a lineage briefly clears the unlock bar, anneal_unlock_round is "
        "permanent (saved in the checkpoint), and the anneal schedules keep counting down forever "
        "after that even if delivery performance later stalls or regresses for hundreds of rounds, "
        "exactly the 'finished exploring before it had found reliable delivery behavior' failure "
        "mode --log-std-anneal-gate-deliveries' own help text describes -- this mechanism just "
        "wasn't able to trigger a SECOND time to fix it once already unlocked. Defaults to half of "
        "--log-std-anneal-gate-deliveries (hysteresis around the unlock bar, so the gate can't flap "
        "open/closed within the same round-to-round noise) -- deliberately not the same value as "
        "the unlock bar itself, which real history (outputs/mappo_checkpoint_continuous_webots_"
        "finetuned_36_history.csv) shows a stalled-but-not-regressed lineage can sit right at or "
        "above without ever truly recovering. Set to 0 to disable relocking and restore the old "
        "permanent-unlock behavior.",
    )
    parser.add_argument(
        "--log-std-anneal-reunlock-deliveries", type=float, default=None,
        help="continuous-actions only: after the gate has RELOCKED at least once, require the "
        "trailing --log-std-anneal-gate-window average of deliveries/round to reach THIS value "
        "(not --log-std-anneal-gate-deliveries) before re-unlocking. Added 2026-09-09, confirmed "
        "live on the checkpoint_46 lineage: with only --log-std-anneal-relock-deliveries=4.0 as the "
        "single switching threshold, a trailing average sitting anywhere near 4.0 (e.g. oscillating "
        "3.6-4.1 round to round) relocked and re-unlocked on consecutive rounds over and over -- "
        "the checkpoint_38-incident hysteresis this file already has (relock below the unlock bar, "
        "by default half of it) only guards the gap between 'never learned to deliver at all' "
        "(0.05) and 'matches its own peak' (4.0), which does nothing when the noise is happening "
        "AT the 4.0 line itself. Re-unlocking off the same 0.05 floor that only makes sense for a "
        "checkpoint's very first, cold-start unlock let each relock's widened exploration get "
        "reversed again as soon as one lucky round nudged the trailing average back over 0.05 -- "
        "often the very next round. Defaults to relock-threshold + 1.0 (a real gap above the "
        "switching point the relock/unlock pair otherwise chatters around, not just a hair above "
        "it) -- a relock now has to earn a trailing average clearly past where it fell, not merely "
        "not-catastrophic, before exploration narrows again. Only applies to a re-unlock following "
        "an actual relock in THIS run; the very first unlock (anneal_unlock_round never set before) "
        "still uses --log-std-anneal-gate-deliveries, unchanged. Set to the same value as "
        "--log-std-anneal-relock-deliveries to restore the old chatter-prone behavior.",
    )
    args = parser.parse_args()
    is_continuous = args.continuous_actions

    with open(args.env_config) as f:
        env_cfg = yaml.safe_load(f)
    with open(args.train_config) as f:
        train_cfg = yaml.safe_load(f)
    if args.robots is not None:
        env_cfg["num_robots"] = args.robots

    env_cls = ContinuousWarehouseEnv if is_continuous else WarehouseEnv
    env_kwargs = dict(max_episode_steps=10**9, seed=args.seed)
    if not is_continuous:
        env_kwargs["enable_slip"] = False
    else:
        env_kwargs["team_reward_share"] = args.team_reward_share
        env_kwargs["physical_collision_distance_m"] = args.global_collision_reset_m
        env_kwargs["shield_physical_walls_only"] = args.shield_physical_walls_only
        env_kwargs["shield_active_resolution"] = args.shield_active_resolution
        env_kwargs["shield_safety_radius_m"] = args.shield_radius
        env_kwargs["shield_pickup_radius_m"] = args.shield_pickup_radius
    env = env_cls.from_config(env_cfg, **env_kwargs)
    agents = env.possible_agents
    device = "cpu"

    # PPO config with command-line overrides.
    cfg = PPOConfig.from_dict(train_cfg)
    if args.update_epochs is not None:
        cfg.update_epochs = args.update_epochs
    if args.minibatches is not None:
        cfg.minibatches = args.minibatches
    if args.learning_rate is not None:
        cfg.learning_rate = args.learning_rate
    if args.entropy_coef is not None:
        cfg.entropy_coef = args.entropy_coef
    cfg.value_loss = args.value_loss
    cfg.value_clip = args.value_clip
    cfg.desired_kl = args.desired_kl
    base_entropy_coef = cfg.entropy_coef
    entropy_coef_anneal_rounds = (
        args.entropy_coef_anneal_rounds
        if args.entropy_coef_anneal_rounds is not None
        else args.log_std_anneal_rounds
    )
    torch.manual_seed(args.seed)
    obs_space = env.observation_space(agents[0])
    if is_continuous:
        net = ContinuousActorCriticNet(
            lidar_dim=obs_space["lidar"].shape[-1],
            vector_dim=obs_space["vector"].shape[-1],
            actor_hidden_dim=args.actor_hidden_dim,
            critic_hidden_dim=args.critic_hidden_dim,
            action_dim=env.action_space(agents[0]).shape[0],
            num_robots=len(agents),
            shield_flag_dim=obs_space["shield_intervened"].shape[-1],
            nearest_robot_dim=obs_space["nearest_robot"].shape[-1],
            agent_state_dim=obs_space["agent_state"].shape[-1] if args.agent_state else 0,
            soft_log_std=args.soft_log_std,
            layer_norm=args.layer_norm,
            obs_norm=args.obs_norm,
        ).to(device)
    else:
        net = ActorCriticNet(
            lidar_dim=obs_space["lidar"].shape[-1],
            vector_dim=obs_space["vector"].shape[-1],
            hidden_dim=cfg.hidden_dim,
            num_actions=env.action_space(agents[0]).n,
            num_robots=len(agents),
        ).to(device)
    def _checkpoint_extra() -> dict:
        """Architecture options and current actor lr saved with every checkpoint."""
        arch = {}
        if is_continuous:
            arch = {
                "agent_state_dim": net.agent_state_dim, "soft_log_std": net.soft_log_std,
                "layer_norm": net.layer_norm, "obs_norm": net.obs_norm,
                "actor_hidden_dim": args.actor_hidden_dim, "critic_hidden_dim": args.critic_hidden_dim,
            }
        return {"arch": arch, "actor_lr": actor_optimizer.param_groups[0]["lr"]}

    actor_params, critic_params = _actor_critic_parameter_groups(net)
    # Separate optimizers for actor and critic.
    actor_optimizer = torch.optim.Adam(actor_params, lr=cfg.learning_rate)
    critic_optimizer = torch.optim.Adam(critic_params, lr=cfg.learning_rate)
    value_rms = RunningMeanStd()
    start_update = 0
    anneal_unlock_round: int | None = None
    has_relocked_this_run = False
    # Warm start from a checkpoint (partial load + migration if the architecture changed).
    if args.checkpoint_in:
        try:
            meta = load_checkpoint(
                net, args.checkpoint_in, device=device,
                actor_optimizer=actor_optimizer, critic_optimizer=critic_optimizer,
            )
        except RuntimeError as e:
            print(
                f"Strict checkpoint load failed ({e!r}) -- likely an architecture change since "
                f"this checkpoint was saved. Falling back to a partial (shape-filtered) load: "
                f"compatible layers warm-start, resized/new ones reinitialize fresh, optimizer "
                f"state does not carry over.",
                flush=True,
            )
            meta = load_checkpoint(net, args.checkpoint_in, device=device, strict=False)
            migrated = migrate_architecture_change(
                net, meta["skipped_tensors"], num_robots=len(agents)
            )
            print(
                f"  migrate_architecture_change: function-preserving migration applied to "
                f"{migrated} (vs. leaving them at a fresh random init)", flush=True,
            )
        if meta["value_rms"] is not None:
            value_rms.load_state_dict(meta["value_rms"])
        start_update = meta.get("update", 0) or 0
        anneal_unlock_round = (meta.get("extra") or {}).get("anneal_unlock_round")
        if args.learning_rate is not None:
            for optimizer in (actor_optimizer, critic_optimizer):
                for group in optimizer.param_groups:
                    group["lr"] = args.learning_rate
        elif cfg.desired_kl > 0 and (meta.get("extra") or {}).get("actor_lr"):
            for group in actor_optimizer.param_groups:
                group["lr"] = float(meta["extra"]["actor_lr"])
        print(
            f"Warm-started from {args.checkpoint_in} (actor lr {actor_optimizer.param_groups[0]['lr']:.2e}, "
            f"critic lr {critic_optimizer.param_groups[0]['lr']:.2e})", flush=True,
        )
        if anneal_unlock_round is not None:
            print(
                f"  exploration-anneal gate was already unlocked at cumulative_round "
                f"{anneal_unlock_round} in this checkpoint's lineage", flush=True,
            )
        if is_continuous:
            if not args.soft_log_std and args.log_std_anneal_rounds > 0:
                cumulative_round_0 = start_update + 1
                anneal_round_0 = (
                    cumulative_round_0 if args.log_std_anneal_gate_deliveries <= 0
                    else (0 if anneal_unlock_round is None else max(0, cumulative_round_0 - anneal_unlock_round))
                )
                progress_0 = min(1.0, anneal_round_0 / args.log_std_anneal_rounds)
                net.set_log_std_floor(
                    _log_std_floors(progress_0, args.log_std_anneal_omega_rate_multiplier, net)
                )
            if args.soft_log_std:
                if args.actor_log_std_init is not None:
                    with torch.no_grad():
                        net.actor_log_std.fill_(args.actor_log_std_init)
                    print(f"  actor_log_std set to {args.actor_log_std_init} (--soft-log-std)", flush=True)
            else:
                with torch.no_grad():
                    net.actor_log_std.clamp_(min=net.log_std_floor)
                    if args.actor_log_std_init is not None:
                        net.actor_log_std.clamp_(max=args.actor_log_std_init)
                        print(
                            f"  actor_log_std forced down to at most {args.actor_log_std_init} "
                            f"(now {net.actor_log_std.tolist()}) -- see --actor-log-std-init's help",
                            flush=True,
                        )

    from hiverl.bridge.ros2_driver import Ros2RobotDriver

    if args.worlds < 1:
        raise SystemExit("--worlds must be >= 1")
    if args.world_domain_ids is not None:
        domain_ids = [int(d) for d in args.world_domain_ids.split(",") if d.strip()]
    elif args.worlds > 1:
        domain_ids = list(range(args.worlds))
    else:
        domain_ids = [None]
    if len(domain_ids) != args.worlds or len(set(domain_ids)) != len(domain_ids):
        raise SystemExit(f"--world-domain-ids needs {args.worlds} distinct ids, got {domain_ids}")
    # One env, ROS2 driver and coordinator per Webots world.
    envs = [env] + [
        env_cls.from_config(env_cfg, **{**env_kwargs, "seed": args.seed + k * WORLD_SEED_STRIDE})
        for k in range(1, args.worlds)
    ]
    drivers = [
        Ros2RobotDriver(
            num_robots=env_cfg["num_robots"],
            node_name="hiverl_coordinator" if args.worlds == 1 else f"hiverl_coordinator_w{k}",
            domain_id=domain_ids[k],
        )
        for k in range(args.worlds)
    ]
    if args.worlds > 1:
        print(
            f"Multi-world collection: {args.worlds} Webots worlds on ROS_DOMAIN_IDs {domain_ids}, "
            f"env seeds {[args.seed + k * WORLD_SEED_STRIDE for k in range(args.worlds)]}",
            flush=True,
        )
    coordinator_cls = ContinuousLockstepCoordinator if is_continuous else LockstepCoordinator
    coordinator_kwargs = dict(arrival_timeout_s=args.arrival_timeout, synchronize_fleet=True)
    if is_continuous:
        coordinator_kwargs["velocity_tick_sleep_s"] = args.velocity_tick_sleep_s
        coordinator_kwargs["enable_legacy_retreats"] = not args.disable_legacy_retreats
        coordinator_kwargs["global_collision_reset_m"] = args.global_collision_reset_m
        coordinator_kwargs["respawn_on_collision"] = args.respawn_on_collision
        coordinator_kwargs["execute_shield_velocities"] = args.execute_shield
    coordinators = [coordinator_cls(e, d, **coordinator_kwargs) for e, d in zip(envs, drivers)]

    os.makedirs(os.path.dirname(args.checkpoint_out), exist_ok=True)
    # Output files.
    history_csv_path = args.history_csv or f"{os.path.splitext(args.checkpoint_out)[0]}_history.csv"
    print(f"Appending per-round history to {history_csv_path}", flush=True)
    kl_diagnostics_csv_path = (
        None if args.no_kl_diagnostics
        else args.kl_diagnostics_csv or f"{os.path.splitext(args.checkpoint_out)[0]}_kl_diagnostics.csv"
    )
    if kl_diagnostics_csv_path is not None:
        print(f"Appending per-minibatch KL diagnostics to {kl_diagnostics_csv_path}", flush=True)
    tick_trace_rounds = (
        {int(r.strip()) for r in args.tick_trace_rounds.split(",") if r.strip()}
        if args.tick_trace_rounds else set()
    )
    tick_trace_csv_path = None
    if tick_trace_rounds:
        tick_trace_csv_path = (
            args.tick_trace_csv or f"{os.path.splitext(args.checkpoint_out)[0]}_tick_trace.csv"
        )
        print(
            f"Writing per-tick navigation trace for rounds {sorted(tick_trace_rounds)} to "
            f"{tick_trace_csv_path}", flush=True,
        )
    best_checkpoint_path = args.best_checkpoint_out or f"{os.path.splitext(args.checkpoint_out)[0]}_best.pt"
    best_trailing_reward = float("-inf")
    best_deliveries_path = f"{os.path.splitext(args.checkpoint_out)[0]}_best_deliveries.pt"
    best_deliveries_seen = -1
    if os.path.exists(best_deliveries_path):
        try:
            existing_best = torch.load(best_deliveries_path, map_location="cpu", weights_only=False)
            best_deliveries_seen = existing_best.get("extra", {}).get("best_deliveries", -1)
        except Exception as exc:
            print(f"Could not read existing {best_deliveries_path}'s record ({exc}) -- starting from -1", flush=True)
    round_summaries = []
    rollout_carry: list | None = [] if args.accumulate_short_rounds else None
    needs_reset = [True] * len(coordinators)
    zero_delivery_streak = [0] * len(coordinators)
    try:
        # Training rounds.
        for round_idx in range(1, args.rounds + 1):
            round_label = f"round {round_idx}/{args.rounds}"
            cumulative_round = start_update + round_idx
            if args.log_std_anneal_gate_deliveries <= 0:
                anneal_round = cumulative_round
            else:
                if anneal_unlock_round is None:
                    unlock_bar = args.log_std_anneal_gate_deliveries
                    if has_relocked_this_run:
                        relock_threshold_for_reunlock = (
                            args.log_std_anneal_relock_deliveries
                            if args.log_std_anneal_relock_deliveries is not None
                            else args.log_std_anneal_gate_deliveries / 2.0
                        )
                        unlock_bar = (
                            args.log_std_anneal_reunlock_deliveries
                            if args.log_std_anneal_reunlock_deliveries is not None
                            else relock_threshold_for_reunlock + 1.0
                        )
                    window = round_summaries[-args.log_std_anneal_gate_window:]
                    if len(window) >= args.log_std_anneal_gate_window:
                        trailing_avg = _deliveries_per_1000_ticks(window)
                        if trailing_avg >= unlock_bar:
                            anneal_unlock_round = cumulative_round
                            print(
                                f"[{round_label}] exploration-anneal gate UNLOCKED: trailing "
                                f"{len(window)}-round deliveries/1000-ticks={trailing_avg:.3f} >= "
                                f"{unlock_bar} -- anneal schedules now "
                                f"start counting down from this round", flush=True,
                            )
                else:
                    relock_threshold = (
                        args.log_std_anneal_relock_deliveries
                        if args.log_std_anneal_relock_deliveries is not None
                        else args.log_std_anneal_gate_deliveries / 2.0
                    )
                    if relock_threshold > 0:
                        window = round_summaries[-args.log_std_anneal_gate_window:]
                        if len(window) >= args.log_std_anneal_gate_window:
                            trailing_avg = _deliveries_per_1000_ticks(window)
                            if trailing_avg < relock_threshold:
                                print(
                                    f"[{round_label}] exploration-anneal gate RELOCKED: trailing "
                                    f"{len(window)}-round deliveries/1000-ticks={trailing_avg:.3f} < "
                                    f"{relock_threshold} -- restoring full exploration", flush=True,
                                )
                                anneal_unlock_round = None
                                has_relocked_this_run = True
                anneal_round = (
                    0 if anneal_unlock_round is None else max(0, cumulative_round - anneal_unlock_round)
                )
            if is_continuous and not args.soft_log_std and args.log_std_anneal_rounds > 0:
                progress = min(1.0, anneal_round / args.log_std_anneal_rounds)
                net.set_log_std_floor(
                    _log_std_floors(progress, args.log_std_anneal_omega_rate_multiplier, net)
                )
                if args.actor_log_std_init is not None:
                    anneal_floor = net.log_std_floor
                    headroom_floor = torch.full_like(
                        anneal_floor, args.actor_log_std_init - LOG_STD_INIT_FLOOR_HEADROOM
                    )
                    net.set_log_std_floor(torch.minimum(anneal_floor, headroom_floor).tolist())
            if is_continuous and not args.soft_log_std and entropy_coef_anneal_rounds > 0:
                ec_progress = min(1.0, anneal_round / entropy_coef_anneal_rounds)
                cfg.entropy_coef = base_entropy_coef * (
                    1.0 - ec_progress * (1.0 - args.entropy_coef_anneal_floor_frac)
                )
            if len(coordinators) == 1:
                summary = collect_and_update(
                    coordinators[0], net, actor_optimizer, critic_optimizer, value_rms, agents, device, cfg,
                    args.ticks, round_label,
                    reset=needs_reset[0], is_continuous=is_continuous,
                    round_idx=round_idx, kl_diagnostics_csv_path=kl_diagnostics_csv_path,
                    tick_trace_csv_path=tick_trace_csv_path if round_idx in tick_trace_rounds else None,
                    min_update_ticks=args.min_update_ticks,
                    rollout_carry=rollout_carry,
                    target_kl=args.target_kl if args.target_kl > 0 else None,
                )
                world_summaries = [summary]
            else:
                summary, world_summaries = collect_and_update_multi_world(
                    coordinators, net, actor_optimizer, critic_optimizer, value_rms, agents, device,
                    cfg, args.ticks, round_label, needs_reset, is_continuous, round_idx,
                    kl_diagnostics_csv_path, args.min_update_ticks, rollout_carry,
                    args.target_kl if args.target_kl > 0 else None,
                )
            for k, world_summary in enumerate(world_summaries):
                world_label = round_label if len(coordinators) == 1 else f"{round_label} w{k}"
                needs_reset[k] = world_summary is None or world_summary["ticks"] < args.ticks
                if world_summary is None:
                    continue
                zero_delivery_streak[k] = 0 if world_summary["deliveries"] > 0 else zero_delivery_streak[k] + 1
                if args.gridlock_reset_streak > 0 and zero_delivery_streak[k] >= args.gridlock_reset_streak:
                    print(
                        f"[{world_label}] {zero_delivery_streak[k]} consecutive zero-delivery rounds "
                        f"-- forcing a physical reset next round (suspected gridlock)", flush=True,
                    )
                    needs_reset[k] = True
                    zero_delivery_streak[k] = 0
            if summary is not None:
                round_summaries.append(summary)
                save_checkpoint(
                    args.checkpoint_out, net, value_rms_state=value_rms.state_dict(),
                    update=cumulative_round, config=cfg,
                    extra={"anneal_unlock_round": anneal_unlock_round, **_checkpoint_extra()},
                    actor_optimizer=actor_optimizer, critic_optimizer=critic_optimizer,
                )
                print(f"[{round_label}] saved checkpoint to {args.checkpoint_out}", flush=True)
                if not args.no_best_checkpoint:
                    window = round_summaries[-args.best_checkpoint_window:]
                    if len(window) >= args.best_checkpoint_window:
                        trailing_reward = sum(s["mean_reward"] for s in window) / len(window)
                        trailing_activity = sum(
                            s["interventions"] + s["wall_block"] for s in window
                        ) / len(window)
                        if (
                            trailing_reward > best_trailing_reward
                            and trailing_activity >= args.best_checkpoint_min_activity
                        ):
                            best_trailing_reward = trailing_reward
                            save_checkpoint(
                                best_checkpoint_path, net, value_rms_state=value_rms.state_dict(),
                                update=cumulative_round, config=cfg,
                                extra={"anneal_unlock_round": anneal_unlock_round, **_checkpoint_extra()},
                                actor_optimizer=actor_optimizer, critic_optimizer=critic_optimizer,
                            )
                            print(
                                f"[{round_label}] new best trailing-{len(window)}-round mean_reward="
                                f"{trailing_reward:.3f} -- saved to {best_checkpoint_path}", flush=True,
                            )
                if summary["deliveries"] > best_deliveries_seen:
                    best_deliveries_seen = summary["deliveries"]
                    save_checkpoint(
                        best_deliveries_path, net, value_rms_state=value_rms.state_dict(),
                        update=cumulative_round, config=cfg,
                        extra={
                            "anneal_unlock_round": anneal_unlock_round,
                            **_checkpoint_extra(),
                            "best_deliveries": best_deliveries_seen,
                            "best_deliveries_round_ticks": summary["ticks"],
                        },
                        actor_optimizer=actor_optimizer, critic_optimizer=critic_optimizer,
                    )
                    print(
                        f"[{round_label}] new best RAW single-round deliveries={best_deliveries_seen} "
                        f"(on {summary['ticks']} ticks) -- saved to {best_deliveries_path}", flush=True,
                    )
                row = {
                    "round": round_idx,
                    "cumulative_round": cumulative_round,
                    "ticks": summary["ticks"],
                    "update_skipped": summary["update_skipped"],
                    "update_ticks": summary["update_ticks"],
                    "actor_lr": summary.get("actor_lr", float("nan")),
                    "mean_reward": summary["mean_reward"],
                    "policy_loss": summary["policy_loss"],
                    "value_loss": summary["value_loss"],
                    "entropy": summary["entropy"],
                    "approx_kl": summary["approx_kl"],
                    "clip_fraction": summary["clip_fraction"],
                    "explained_variance": summary["explained_variance"],
                    "value_pred_mean": summary["value_pred_mean"],
                    "value_target_mean": summary["value_target_mean"],
                    "value_rms_mean": summary["value_rms_mean"],
                    "value_rms_std": summary["value_rms_std"],
                    "log_std_v": summary["log_std_v"],
                    "log_std_omega": summary["log_std_omega"],
                    "mean_v": summary.get("mean_v", float("nan")),
                    "std_v": summary.get("std_v", float("nan")),
                    "mean_abs_v": summary.get("mean_abs_v", float("nan")),
                    "mean_omega": summary.get("mean_omega", float("nan")),
                    "mean_abs_omega": summary.get("mean_abs_omega", float("nan")),
                    "std_omega": summary.get("std_omega", float("nan")),
                    "max_abs_omega": summary.get("max_abs_omega", float("nan")),
                    "fraction_high_omega": summary.get("fraction_high_omega", float("nan")),
                    "mean_action_modification": summary["mean_action_modification"],
                    "mean_distance_start": summary["mean_distance_start"],
                    "mean_distance_end": summary["mean_distance_end"],
                    "mean_progress_per_100_ticks": summary["mean_progress_per_100_ticks"],
                    "pickup_latency_mean_ticks": summary["pickup_latency_mean_ticks"],
                    "pickup_latency_count": summary["pickup_latency_count"],
                    "delivery_latency_mean_ticks": summary["delivery_latency_mean_ticks"],
                    "delivery_latency_count": summary["delivery_latency_count"],
                    "lidar_stall_event_ticks": summary["lidar_stall_event_ticks"],
                    "lidar_stall_isolated_ticks": summary["lidar_stall_isolated_ticks"],
                    "lidar_stall_sustained_ticks": summary["lidar_stall_sustained_ticks"],
                    "lidar_stall_isolated_frac": summary["lidar_stall_isolated_frac"],
                    "lidar_stall_sustained_frac": summary["lidar_stall_sustained_frac"],
                    "max_lidar_stall_streak": summary["max_lidar_stall_streak"],
                    "deliveries": summary["deliveries"],
                    "collisions": summary["collisions"],
                    "interventions": summary["interventions"],
                    "wall_block": summary["wall_block"],
                    "stuck_events": summary["stuck_events"],
                    "wall_proximity_penalty": summary["wall_proximity_penalty"],
                    "anneal_round": anneal_round,
                    "anneal_gate_unlocked": int(anneal_unlock_round is not None),
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
                row.update({k: v for k, v in summary.items() if k.startswith("reward_")})
                row.update({k: v for k, v in summary.items() if k == "worlds" or k.startswith("world_")})
                row.update({
                    k: v for k, v in summary.items()
                    if k.startswith((
                        "pre_tanh_", "log_prob_", "fraction_action_", "max_abs_log_prob_diff",
                        "mean_abs_mu_", "max_abs_mu_", "mean_abs_noise_", "max_abs_noise_",
                        "mu_v_p", "mu_omega_p", "fraction_mu_", "corr_mu_",
                        "mean_abs_hidden", "growth_ratio_mu_",
                        "mean_abs_action_omega", "action_omega_p", "fraction_abs_omega_",
                        "omega_penalty_per_tick", "corr_action_omega_", "fraction_ticks_omega_",
                    ))
                })
                append_history_row(history_csv_path, row)
    finally:
        for driver in drivers:
            driver.shutdown()

    if round_summaries:
        rewards = [s["mean_reward"] for s in round_summaries]
        updated_rounds = sum(not s["update_skipped"] for s in round_summaries)
        print(
            f"\nFine-tuning complete: {updated_rounds}/{args.rounds} rounds produced a "
            f"PPO update. mean_reward per round: {[round(r, 3) for r in rewards]}",
            flush=True,
        )
    else:
        print("\nFine-tuning complete: no round collected enough real ticks to update on.", flush=True)


if __name__ == "__main__":
    main()
