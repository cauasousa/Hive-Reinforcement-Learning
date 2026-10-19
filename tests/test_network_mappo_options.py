"""ContinuousActorCriticNet options (agent state, soft log-std, LayerNorm, obs normalization)."""
import torch

from hiverl.agents.networks import (
    SOFT_LOG_STD_MAX,
    SOFT_LOG_STD_MIN,
    ContinuousActorCriticNet,
    _soft_log_std,
    migrate_architecture_change,
)


def _net(**kw):
    torch.manual_seed(0)
    return ContinuousActorCriticNet(
        lidar_dim=4, vector_dim=3, actor_hidden_dim=16, critic_hidden_dim=16, action_dim=2,
        num_robots=3, shield_flag_dim=1, nearest_robot_dim=2, **kw,
    )


def _inputs(b=6, n=3, extra_dim=3, scale=5.0):
    torch.manual_seed(1)
    return (
        torch.rand(b, n, 4) * scale, torch.randn(b, n, 3) * scale, torch.randn(b, n, extra_dim) * scale,
    )


def test_obs_normalizer_update_is_function_preserving():
    net = _net(obs_norm=True)
    lidar, vector, extra = _inputs(scale=1.0)
    before_mu = net._dist(lidar, vector, extra).mean.detach().clone()
    before_v = net.centralized_value(lidar, vector, extra).detach().clone()
    for _ in range(3):
        net.update_obs_normalizer(net.raw_per_robot_features(lidar, vector, extra) * 0.7 + 0.3)
    assert float(net.obs_count) > 0 and not torch.allclose(net.obs_mean, torch.zeros_like(net.obs_mean))
    assert torch.allclose(net._dist(lidar, vector, extra).mean, before_mu, atol=1e-4)
    assert torch.allclose(net.centralized_value(lidar, vector, extra), before_v, atol=1e-4)


def test_soft_log_std_keeps_values_inside_bounds_and_gradient_alive():
    inside = torch.tensor([-2.0, -1.4, -0.5, 0.0])
    assert torch.allclose(_soft_log_std(inside), inside, atol=0.02)
    raw = torch.tensor([-4.0, 3.0], requires_grad=True)
    out = _soft_log_std(raw)
    assert SOFT_LOG_STD_MIN - 1e-3 <= float(out[0].detach()) and float(out[1].detach()) <= SOFT_LOG_STD_MAX + 1e-3
    out.sum().backward()
    assert (raw.grad.abs() > 0).all(), "gradient must never be exactly zero at the bounds"


def test_migration_to_wider_agent_features_preserves_actor_and_critic():
    old = _net()
    new = _net(agent_state_dim=5)
    skipped = {}
    own = new.state_dict()
    for key, tensor in old.state_dict().items():
        if own[key].shape == tensor.shape:
            own[key] = tensor
        else:
            skipped[key] = tensor
    new.load_state_dict(own)
    migrate_architecture_change(new, skipped, num_robots=3)
    lidar, vector, extra = _inputs()
    extra_new = torch.cat([extra, torch.randn(*extra.shape[:2], 5)], dim=-1)
    assert torch.allclose(new._dist(lidar, vector, extra_new).mean, old._dist(lidar, vector, extra).mean, atol=1e-5)
    assert torch.allclose(
        new.centralized_value(lidar, vector, extra_new), old.centralized_value(lidar, vector, extra), atol=1e-5
    )
