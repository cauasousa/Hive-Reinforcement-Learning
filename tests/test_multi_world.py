"""Parallel collection from several Webots worlds with one PPO update."""
import numpy as np
import torch

from scripts.train_mappo_webots import (
    WORLD_SEED_STRIDE,
    _actor_critic_parameter_groups,
    _merge_world_summaries,
    collect_and_update_multi_world,
)
from hiverl.agents.networks import ContinuousActorCriticNet
from hiverl.agents.ppo import PPOConfig, RunningMeanStd
from hiverl.bridge.lockstep_coordinator import ContinuousLockstepCoordinator
from hiverl.bridge.robot_driver import FakeRobotDriver
from hiverl.env.warehouse_env_continuous import ContinuousWarehouseEnv
from hiverl.env.warehouse_grid import AISLE

_ENV_KWARGS = dict(
    width=13, height=9, num_shelf_blocks_x=2, num_shelf_blocks_y=1, shelf_block_w=2,
    shelf_block_h=2, num_pickup_stations=2, num_dropoff_stations=2, num_robots=3,
    order_spawn_interval=2, max_queue_size=10, view_radius=3, max_episode_steps=10**9,
    deadlock_wait_threshold=1,
)


def _world(k: int, base_seed: int = 42):
    env = ContinuousWarehouseEnv(**_ENV_KWARGS, seed=base_seed + k * WORLD_SEED_STRIDE)
    coordinator = ContinuousLockstepCoordinator(
        env, FakeRobotDriver(ticks_to_arrive=0), arrival_timeout_s=5.0, synchronize_fleet=True,
        poll_interval_s=0.0, enable_legacy_retreats=False,
    )
    coordinator._sleep = lambda seconds: None
    return env, coordinator


def test_worlds_spawn_robots_in_different_cells_inside_the_walls():
    envs = [_world(k)[0] for k in range(2)]
    for reset in range(5):
        spawns = []
        for env in envs:
            env.reset()
            cells = {r.pos for r in env.robots.values()}
            h, w = env.map.grid.shape
            assert all(0 <= r < h and 0 <= c < w and env.map.grid[r, c] == AISLE for r, c in cells)
            spawns.append(cells)
        assert spawns[0] != spawns[1], f"worlds spawned identically on reset {reset}"


def test_multi_world_round_updates_once_on_every_worlds_ticks():
    worlds = [_world(k) for k in range(2)]
    env0 = worlds[0][0]
    env0.reset(seed=42)
    obs_space = env0.observation_space(env0.possible_agents[0])
    net = ContinuousActorCriticNet(
        lidar_dim=obs_space["lidar"].shape[-1], vector_dim=obs_space["vector"].shape[-1],
        actor_hidden_dim=16, critic_hidden_dim=16, action_dim=2, num_robots=3,
        shield_flag_dim=obs_space["shield_intervened"].shape[-1],
        nearest_robot_dim=obs_space["nearest_robot"].shape[-1],
    )
    actor_params, critic_params = _actor_critic_parameter_groups(net)
    actor_opt, critic_opt = torch.optim.Adam(actor_params, lr=3e-4), torch.optim.Adam(critic_params, lr=3e-4)
    before = [p.detach().clone() for p in net.parameters()]

    merged, per_world = collect_and_update_multi_world(
        [c for _, c in worlds], net, actor_opt, critic_opt, RunningMeanStd(), env0.possible_agents,
        "cpu", PPOConfig(update_epochs=1, minibatches=2), 20, "round 1/1", [True, True], True, 1,
        None, 0, None, None,
    )

    assert [w["ticks"] for w in per_world] == [20, 20]
    assert merged["ticks"] == 40 and merged["update_ticks"] == 40 and merged["update_skipped"] == 0
    assert merged["worlds"] == 2 and merged["world_0_ticks"] == 20 and merged["world_1_ticks"] == 20
    assert "_segment" not in merged and all("_segment" not in w for w in per_world)
    assert any(not torch.equal(b, p) for b, p in zip(before, net.parameters())), "no PPO step happened"


def test_merge_sums_totals_and_tick_weights_rates():
    a = {"ticks": 100, "deliveries": 3, "mean_reward": 1.0, "reward_pickup": 30.0, "max_lidar_stall_streak": 4,
         "entropy": float("nan"), "label": "x"}
    b = {"ticks": 300, "deliveries": 5, "mean_reward": 0.0, "reward_pickup": 60.0, "max_lidar_stall_streak": 9,
         "entropy": 0.5, "label": "y"}
    m = _merge_world_summaries([a, b])
    assert m["ticks"] == 400 and m["deliveries"] == 8 and m["reward_pickup"] == 90.0
    assert m["mean_reward"] == 0.25 and m["max_lidar_stall_streak"] == 9 and m["entropy"] == 0.5
    assert m["world_1_deliveries"] == 5 and m["label"] == "x"
