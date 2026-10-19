#!/usr/bin/env python3
"""Behavior-cloning warm-start for a freshly-initialized ContinuousActorCriticNet's actor.

User-requested (2026-09-13), after checkpoint_50/51 (the 2026-09-13 separate-encoders
architecture, see ContinuousActorCriticNet's own docstring) spent 45+ real rounds delivering
essentially nothing (0.09-0.14 deliveries/1000-ticks, vs. 6.3-6.45 for the old shared-encoder
architecture and even a non-learned scripted controller) while the anti-pileup circuit breaker
fired in ~85% of rounds. Narrowing exploration noise (--actor-log-std-init) did NOT fix this --
confirmed live, the delivery rate drifted right back down after an initial 18-round blip.

Diagnosis: a policy with a NEAR-RANDOM actor (small-gain orthogonal init, as requested) has no
margin against this environment's known real issues (the lidar-forward-brake-with-no-escape bug,
the shield's idealized-vs-real pose lag) the way an already-competent policy does -- confirmed
live, checkpoint_49's already-trained actor delivered 4 in its very first round under the exact
same environment/shield/bugs (see exp_control_49). The fresh policy isn't failing to learn slowly;
it's spending its early rounds fighting the environment's physical failure modes before it ever
gets a chance to discover useful behavior at all.

This script breaks that deadlock without reverting the architecture change: it queries an already-
competent OLD checkpoint (the pre-2026-09-13 shared-encoder architecture, e.g. ..._49_best.pt) as
a behavior-cloning TEACHER, driving the real fleet deterministically (mean action, no exploration
noise -- this is imitation, not RL) to collect representative (observation -> action) pairs, then
trains the NEW architecture's actor (actor_encoder + actor_mean only, never the critic) via
supervised MSE regression against the teacher's pre-tanh mean action on those same observations.
The result is a NEW-architecture network whose actor already behaves roughly like the competent
old one, ready for further PPO fine-tuning instead of starting from near-random.

Usage (real Webots required for data collection, same two-terminal setup as
scripts/train_mappo_webots.py):
    python scripts/bc_warmstart_actor.py \\
        --teacher-checkpoint outputs/mappo_checkpoint_continuous_webots_finetuned_49_best.pt \\
        --student-checkpoint-in outputs/mappo_checkpoint_continuous_webots_finetuned_51.pt \\
        --checkpoint-out outputs/mappo_checkpoint_continuous_webots_finetuned_52_bcwarm.pt \\
        --robots 5 --collect-rounds 3 --collect-ticks 1000
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from hiverl.agents.checkpoint import load_checkpoint, save_checkpoint
from hiverl.agents.networks import ContinuousActorCriticNet, SharedEncoder, _BaseActorCriticNet
from hiverl.bridge.lockstep_coordinator import ContinuousLockstepCoordinator, _RL_CONTROLLED_STATUSES
from hiverl.env.warehouse_env_continuous import ContinuousWarehouseEnv

ROOT = os.path.join(os.path.dirname(__file__), "..")


class _LegacySharedEncoderActorNet(_BaseActorCriticNet):
    """Reconstructs the PRE-2026-09-13 ContinuousActorCriticNet (one encoder shared between actor
    and critic, see networks.py's ContinuousActorCriticNet docstring for why that changed) --
    used ONLY here, to load an old checkpoint faithfully as this script's behavior-cloning
    teacher. Not a general-purpose class; deliberately has no `act`/`evaluate_actions`/
    `centralized_value` beyond what `mean_action` below needs, since the teacher is only ever
    queried for its deterministic mean action."""

    def __init__(
        self, lidar_dim: int, vector_dim: int, hidden_dim: int, action_dim: int,
        num_robots: int | None = None, shield_flag_dim: int = 0, nearest_robot_dim: int = 0,
    ):
        super().__init__(lidar_dim, vector_dim, hidden_dim, num_robots=num_robots)
        extra_dim = shield_flag_dim + nearest_robot_dim
        if extra_dim > 0:
            self.encoder = SharedEncoder(lidar_dim, vector_dim, hidden_dim, extra_dim=extra_dim)
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))

    def mean_action(self, lidar: torch.Tensor, vector: torch.Tensor, extra: torch.Tensor | None = None) -> torch.Tensor:
        return self.actor_mean(self.embed(lidar, vector, extra))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-checkpoint", required=True)
    parser.add_argument(
        "--student-checkpoint-in", default=None,
        help="optional: warm-start the student's CRITIC (and anything else shape-compatible) "
        "from this NEW-architecture checkpoint (e.g. a prior checkpoint_5x.pt) -- the actor is "
        "always overwritten by behavior cloning regardless of what this file contains.",
    )
    parser.add_argument("--checkpoint-out", required=True)
    parser.add_argument("--env-config", default=os.path.join(ROOT, "configs", "env_webots.yaml"))
    parser.add_argument("--robots", type=int, default=None)
    parser.add_argument("--arrival-timeout", type=float, default=60.0)
    parser.add_argument("--collect-ticks", type=int, default=1000)
    parser.add_argument("--collect-rounds", type=int, default=3)
    parser.add_argument(
        "--velocity-tick-sleep-s", type=float, default=0.3,
        help="match whatever the PPO fine-tuning run afterward will use.",
    )
    parser.add_argument("--bc-epochs", type=int, default=300)
    parser.add_argument("--bc-lr", type=float, default=1e-3)
    parser.add_argument("--bc-batch-size", type=int, default=256)
    parser.add_argument("--actor-hidden-dim", type=int, default=64)
    parser.add_argument("--critic-hidden-dim", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    with open(args.env_config) as f:
        env_cfg = yaml.safe_load(f)
    if args.robots is not None:
        env_cfg["num_robots"] = args.robots

    env = ContinuousWarehouseEnv.from_config(env_cfg, max_episode_steps=10**9, seed=args.seed)
    agents = env.possible_agents
    obs_space = env.observation_space(agents[0])
    lidar_dim = obs_space["lidar"].shape[-1]
    vector_dim = obs_space["vector"].shape[-1]
    shield_flag_dim = obs_space["shield_intervened"].shape[-1]
    nearest_robot_dim = obs_space["nearest_robot"].shape[-1]
    action_dim = env.action_space(agents[0]).shape[0]

    teacher = _LegacySharedEncoderActorNet(
        lidar_dim, vector_dim, 128, action_dim, num_robots=len(agents),
        shield_flag_dim=shield_flag_dim, nearest_robot_dim=nearest_robot_dim,
    )
    load_checkpoint(teacher, args.teacher_checkpoint, optimizer=None, device="cpu")
    teacher.eval()
    print(f"Loaded teacher from {args.teacher_checkpoint} (strict load succeeded)", flush=True)

    from hiverl.bridge.ros2_driver import Ros2RobotDriver

    driver = Ros2RobotDriver(num_robots=env_cfg["num_robots"])
    coordinator = ContinuousLockstepCoordinator(
        env, driver, arrival_timeout_s=args.arrival_timeout, synchronize_fleet=True,
        velocity_tick_sleep_s=args.velocity_tick_sleep_s,
    )

    recorded_lidar: list[torch.Tensor] = []
    recorded_vector: list[torch.Tensor] = []
    recorded_extra: list[torch.Tensor] = []
    recorded_mu: list[torch.Tensor] = []

    def policy_fn(obs_dict: dict) -> dict:
        actions = {}
        for i, agent in enumerate(agents):
            lidar_t = torch.tensor(obs_dict[agent]["lidar"], dtype=torch.float32).unsqueeze(0)
            vector_t = torch.tensor(obs_dict[agent]["vector"], dtype=torch.float32).unsqueeze(0)
            shield_t = torch.tensor(obs_dict[agent]["shield_intervened"], dtype=torch.float32).unsqueeze(0)
            nearest_t = torch.tensor(obs_dict[agent]["nearest_robot"], dtype=torch.float32).unsqueeze(0)
            extra_t = torch.cat([shield_t, nearest_t], dim=-1)
            with torch.no_grad():
                mu = teacher.mean_action(lidar_t, vector_t, extra_t)
            # Only record samples for robots the network will actually be asked to act on for
            # real (TO_ITEM/CARRYING) -- an IDLE robot's `vector` is degenerate (no active order,
            # see warehouse_env_continuous.py's _get_obs: path is empty, vector stays (0,0,0)),
            # so recording the teacher's response there would train on a state the student will
            # never actually need to navigate from.
            if env.robots[i].status in _RL_CONTROLLED_STATUSES:
                recorded_lidar.append(lidar_t.squeeze(0))
                recorded_vector.append(vector_t.squeeze(0))
                recorded_extra.append(extra_t.squeeze(0))
                recorded_mu.append(mu.squeeze(0))
            # Deterministic action (mean only, tanh-squashed) -- this is imitation, not
            # exploration, so no sampling noise.
            actions[agent] = torch.tanh(mu).squeeze(0).numpy().astype(np.float32)
        return actions

    try:
        for r in range(args.collect_rounds):
            print(f"[bc-collect] round {r + 1}/{args.collect_rounds}: running {args.collect_ticks} ticks...", flush=True)
            stats = coordinator.run(args.collect_ticks, policy_fn, reset=True)
            print(f"[bc-collect] round {r + 1}/{args.collect_rounds}: collected {len(stats.ticks)} real ticks, "
                  f"{len(recorded_lidar)} samples so far", flush=True)
    finally:
        driver.shutdown()

    n = len(recorded_lidar)
    if n == 0:
        raise SystemExit(
            "Collected zero RL-controlled-robot samples -- the teacher never had a robot in "
            "TO_ITEM/CARRYING status the whole collection run. Check --collect-ticks/--collect-rounds."
        )
    print(f"Collected {n} (robot, tick) samples total", flush=True)

    lidar_batch = torch.stack(recorded_lidar)
    vector_batch = torch.stack(recorded_vector)
    extra_batch = torch.stack(recorded_extra)
    mu_batch = torch.stack(recorded_mu)

    student = ContinuousActorCriticNet(
        lidar_dim=lidar_dim, vector_dim=vector_dim, actor_hidden_dim=args.actor_hidden_dim,
        critic_hidden_dim=args.critic_hidden_dim, action_dim=action_dim, num_robots=len(agents),
        shield_flag_dim=shield_flag_dim, nearest_robot_dim=nearest_robot_dim,
    )
    if args.student_checkpoint_in:
        load_checkpoint(student, args.student_checkpoint_in, optimizer=None, device="cpu", strict=False)
        print(f"Warm-started student's critic (and any other shape-compatible layer) from "
              f"{args.student_checkpoint_in} -- actor will be overwritten by BC below regardless", flush=True)
    # Copy the teacher's already-tuned exploration scale too -- BC only fits the mean; leaving
    # actor_log_std at the student's fresh/forced-narrow init would pair a competent mean with an
    # arbitrary noise scale instead of the teacher's own considered one.
    with torch.no_grad():
        student.actor_log_std.copy_(teacher.actor_log_std)

    optimizer = torch.optim.Adam(
        list(student.actor_encoder.parameters()) + list(student.actor_mean.parameters()),
        lr=args.bc_lr,
    )
    for epoch in range(args.bc_epochs):
        perm = torch.randperm(n)
        total_loss = 0.0
        for start in range(0, n, args.bc_batch_size):
            idx = perm[start:start + args.bc_batch_size]
            pred_mu = student.actor_mean(student.embed(lidar_batch[idx], vector_batch[idx], extra_batch[idx]))
            loss = F.mse_loss(pred_mu, mu_batch[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(idx)
        if epoch % 20 == 0 or epoch == args.bc_epochs - 1:
            print(f"[bc-train] epoch {epoch}/{args.bc_epochs}: mse={total_loss / n:.6f}", flush=True)

    os.makedirs(os.path.dirname(args.checkpoint_out), exist_ok=True)
    save_checkpoint(
        args.checkpoint_out, student, update=0,
        extra={"bc_warmstart_teacher": args.teacher_checkpoint, "bc_warmstart_samples": n},
    )
    print(f"Saved BC-warmstarted checkpoint to {args.checkpoint_out}", flush=True)


if __name__ == "__main__":
    main()
