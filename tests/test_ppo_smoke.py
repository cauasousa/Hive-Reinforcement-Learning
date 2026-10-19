"""PPO smoke tests (no NaN rewards, history logging)."""
import math
import os

import torch

from hiverl.agents.ppo import PPOConfig, compute_gae, train_ppo
from hiverl.env.warehouse_env import WarehouseEnv
from hiverl.env.warehouse_env_continuous import ContinuousWarehouseEnv
from hiverl.utils.history import load_history_csv


def _make_env() -> WarehouseEnv:
    return WarehouseEnv(
        width=11,
        height=9,
        num_shelf_blocks_x=1,
        num_shelf_blocks_y=1,
        shelf_block_w=2,
        shelf_block_h=2,
        num_pickup_stations=2,
        num_dropoff_stations=2,
        num_robots=1,
        order_spawn_interval=3,
        max_queue_size=5,
        view_radius=3,
        max_episode_steps=25,
        deadlock_wait_threshold=1,
        enable_veto=False,
        enable_deadlock_reversal=False,
        seed=0,
    )


def _make_continuous_env() -> ContinuousWarehouseEnv:
    return ContinuousWarehouseEnv(
        width=11,
        height=9,
        num_shelf_blocks_x=1,
        num_shelf_blocks_y=1,
        shelf_block_w=2,
        shelf_block_h=2,
        num_pickup_stations=2,
        num_dropoff_stations=2,
        num_robots=2,
        order_spawn_interval=3,
        max_queue_size=5,
        view_radius=3,
        max_episode_steps=50,
        deadlock_wait_threshold=1,
        seed=0,
    )


def test_compute_gae_keeps_local_returns_aligned_with_local_values():
    rewards = torch.tensor([[0.0, 100.0], [1.0, 0.0]])
    values = torch.zeros(3, 2)
    dones = torch.zeros(2)

    advantages, returns = compute_gae(rewards, values, dones, gamma=1.0, lam=1.0)

    assert advantages.shape == returns.shape == rewards.shape
    assert torch.equal(advantages, torch.tensor([[1.0, 100.0], [1.0, 0.0]]))
    assert torch.equal(returns, advantages)


def test_continuous_action_space_trains_with_finite_losses():
    config = PPOConfig(
        total_timesteps=32 * 4, rollout_steps=32, update_epochs=2, minibatches=2, hidden_dim=16, seed=0,
    )
    trainer, history = train_ppo(_make_continuous_env(), config)
    assert trainer.is_continuous is True
    assert len(history) == 2
    for stats in history:
        assert math.isfinite(stats["mean_episode_reward"]), stats
        assert math.isfinite(stats["mean_step_reward"])
        assert math.isfinite(stats["policy_loss"])
        assert math.isfinite(stats["value_loss"])


def test_mean_episode_reward_is_always_finite():
    config = PPOConfig(
        total_timesteps=32 * 8,
        rollout_steps=32,
        update_epochs=1,
        minibatches=1,
        hidden_dim=16,
        seed=0,
    )
    _, history = train_ppo(_make_env(), config)
    assert len(history) == 8
    for stats in history:
        assert math.isfinite(stats["mean_episode_reward"]), stats
        assert math.isfinite(stats["mean_step_reward"])


def test_checkpoint_interval_and_history_csv(tmp_path):
    config = PPOConfig(
        total_timesteps=32 * 4, rollout_steps=32, update_epochs=1, minibatches=1, hidden_dim=16, seed=0,
    )
    checkpoint_path = str(tmp_path / "ckpt.pt")
    history_csv = str(tmp_path / "history.csv")
    best_path = str(tmp_path / "best.pt")

    _, history = train_ppo(
        _make_env(), config, checkpoint_path=checkpoint_path, checkpoint_interval=2,
        history_csv_path=history_csv, best_checkpoint_path=best_path,
    )

    assert os.path.exists(checkpoint_path)
    assert os.path.exists(best_path)
    rows = load_history_csv(history_csv)
    assert len(rows) == len(history) == 4


def test_anneal_lr_decays_monotonically_and_is_off_by_default():
    config = PPOConfig(
        total_timesteps=32 * 4, rollout_steps=32, update_epochs=1, minibatches=1, hidden_dim=16,
        seed=0, learning_rate=1e-3, anneal_lr=True,
    )
    _, history = train_ppo(_make_env(), config)
    lrs = [h["learning_rate"] for h in history]
    assert lrs[0] == config.learning_rate
    assert all(a >= b for a, b in zip(lrs, lrs[1:])), lrs

    config_off = PPOConfig(
        total_timesteps=32 * 4, rollout_steps=32, update_epochs=1, minibatches=1, hidden_dim=16,
        seed=0, learning_rate=1e-3,
    )
    assert config_off.anneal_lr is False
    _, history_off = train_ppo(_make_env(), config_off)
    assert all(h["learning_rate"] == config_off.learning_rate for h in history_off)


def test_resume_continues_update_count_and_restores_state():
    config = PPOConfig(
        total_timesteps=32 * 2, rollout_steps=32, update_epochs=1, minibatches=1, hidden_dim=16, seed=0,
    )
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        checkpoint_path = os.path.join(tmp, "ckpt.pt")
        trainer1, history1 = train_ppo(_make_env(), config, checkpoint_path=checkpoint_path)
        assert [s["update"] for s in history1] == [1, 2]

        trainer2, history2 = train_ppo(
            _make_env(), config, checkpoint_path=checkpoint_path, resume_path=checkpoint_path,
        )
        assert [s["update"] for s in history2] == [3, 4]
        for p1, p2 in zip(trainer1.net.parameters(), trainer2.net.parameters()):
            assert p1.shape == p2.shape
