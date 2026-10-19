"""Checkpoint save/load, including legacy state_dict files and partial loads."""
import torch

from hiverl.agents.checkpoint import load_checkpoint, save_checkpoint
from hiverl.agents.networks import ActorCriticNet, ContinuousActorCriticNet
from hiverl.agents.ppo import PPOConfig


def _make_net() -> ActorCriticNet:
    return ActorCriticNet(lidar_dim=8, vector_dim=3, hidden_dim=16, num_actions=5)


def test_round_trip_preserves_weights_optimizer_and_metadata(tmp_path):
    net = _make_net()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    loss = net.actor_head.weight.sum()
    loss.backward()
    optimizer.step()

    path = str(tmp_path / "ckpt.pt")
    save_checkpoint(
        path, net, optimizer, value_rms_state={"mean": 1.5, "var": 2.5, "count": 100.0},
        update=42, config=PPOConfig(seed=7), extra={"anneal_unlock_round": 17},
    )

    net2 = _make_net()
    optimizer2 = torch.optim.Adam(net2.parameters(), lr=1e-3)
    meta = load_checkpoint(net2, path, optimizer2, device="cpu")

    for p1, p2 in zip(net.parameters(), net2.parameters()):
        assert torch.equal(p1, p2)
    assert meta["update"] == 42
    assert meta["config"]["seed"] == 7
    assert meta["value_rms"] == {"mean": 1.5, "var": 2.5, "count": 100.0}
    assert meta["extra"] == {"anneal_unlock_round": 17}
    assert optimizer2.state_dict()["state"].keys() == optimizer.state_dict()["state"].keys()


def test_round_trip_preserves_separate_optimizer_states(tmp_path):
    net = _make_net()
    actor_optimizer = torch.optim.Adam(net.actor_head.parameters(), lr=1e-3)
    critic_optimizer = torch.optim.Adam(net.critic_head.parameters(), lr=1e-3)
    net.actor_head.weight.sum().backward()
    actor_optimizer.step()
    net.zero_grad(set_to_none=True)
    net.critic_head[-1].weight.sum().backward()
    critic_optimizer.step()

    path = str(tmp_path / "split_optimizers.pt")
    save_checkpoint(
        path, net, value_rms_state={"mean": 0.0, "var": 1.0, "count": 1.0},
        actor_optimizer=actor_optimizer, critic_optimizer=critic_optimizer,
    )
    net2 = _make_net()
    actor_optimizer2 = torch.optim.Adam(net2.actor_head.parameters(), lr=1e-3)
    critic_optimizer2 = torch.optim.Adam(net2.critic_head.parameters(), lr=1e-3)

    load_checkpoint(
        net2, path, actor_optimizer=actor_optimizer2, critic_optimizer=critic_optimizer2,
    )

    assert actor_optimizer2.state_dict()["state"].keys() == actor_optimizer.state_dict()["state"].keys()
    assert critic_optimizer2.state_dict()["state"].keys() == critic_optimizer.state_dict()["state"].keys()


def test_loading_single_optimizer_checkpoint_into_split_optimizers_starts_cold(tmp_path):
    net = _make_net()
    legacy_optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    net.actor_head.weight.sum().backward()
    legacy_optimizer.step()
    path = str(tmp_path / "single_optimizer.pt")
    save_checkpoint(path, net, legacy_optimizer)

    net2 = _make_net()
    actor_optimizer = torch.optim.Adam(net2.actor_head.parameters(), lr=1e-3)
    critic_optimizer = torch.optim.Adam(net2.critic_head.parameters(), lr=1e-3)
    load_checkpoint(net2, path, actor_optimizer=actor_optimizer, critic_optimizer=critic_optimizer)

    assert not actor_optimizer.state_dict()["state"]
    assert not critic_optimizer.state_dict()["state"]


def test_loads_legacy_raw_state_dict(tmp_path):
    net = _make_net()
    path = str(tmp_path / "legacy.pt")
    torch.save(net.state_dict(), path)

    net2 = _make_net()
    meta = load_checkpoint(net2, path, device="cpu")

    for p1, p2 in zip(net.parameters(), net2.parameters()):
        assert torch.equal(p1, p2)
    assert meta == {"update": 0, "config": None, "value_rms": None, "extra": {}, "skipped_tensors": {}}


def test_strict_load_raises_on_architecture_change(tmp_path):
    old_net = ContinuousActorCriticNet(lidar_dim=8, vector_dim=3, hidden_dim=16, action_dim=2)
    path = str(tmp_path / "old_arch.pt")
    save_checkpoint(path, old_net, update=5)

    new_net = ContinuousActorCriticNet(
        lidar_dim=8, vector_dim=3, hidden_dim=16, action_dim=2, num_robots=3, shield_flag_dim=1,
    )
    try:
        load_checkpoint(new_net, path, device="cpu")
        assert False, "expected a RuntimeError from the shape mismatch"
    except RuntimeError:
        pass


def test_strict_false_warm_starts_compatible_layers_and_reinits_the_rest(tmp_path):
    old_net = ContinuousActorCriticNet(lidar_dim=8, vector_dim=3, hidden_dim=16, action_dim=2)
    with torch.no_grad():
        old_net.actor_mean.weight.fill_(0.42)
        old_net.actor_log_std.fill_(-0.3)
    path = str(tmp_path / "old_arch.pt")
    save_checkpoint(path, old_net, update=5)

    new_net = ContinuousActorCriticNet(
        lidar_dim=8, vector_dim=3, hidden_dim=16, action_dim=2, num_robots=3, shield_flag_dim=1,
    )
    meta = load_checkpoint(new_net, path, device="cpu", strict=False)

    assert torch.equal(new_net.actor_mean.weight, old_net.actor_mean.weight)
    assert torch.equal(new_net.actor_log_std, old_net.actor_log_std)
    assert torch.equal(new_net.actor_encoder[2].weight, old_net.actor_encoder[2].weight)
    assert new_net.actor_encoder[0].weight.shape == (16, 8 + 3 + 1)
    assert new_net.critic_encoder[0].weight.shape == (16, 3 * (8 + 3 + 1))
    assert meta["update"] == 5
