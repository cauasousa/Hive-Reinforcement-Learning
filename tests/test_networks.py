"""Network shapes and the tanh-squashed log-prob math."""
import math

import torch

from hiverl.agents.networks import ActorCriticNet, ContinuousActorCriticNet, migrate_architecture_change


def test_discrete_net_action_and_logprob_shapes():
    net = ActorCriticNet(lidar_dim=8, vector_dim=3, hidden_dim=16, num_actions=5)
    lidar, vector = torch.rand(6, 8), torch.rand(6, 3)
    action, logprob, entropy = net.act(lidar, vector)
    assert action.shape == (6,)
    assert logprob.shape == (6,)
    assert entropy.shape == (6,)
    value = net.centralized_value(lidar.unsqueeze(0), vector.unsqueeze(0))
    assert value.shape == (1,)


def test_continuous_net_action_is_bounded_and_shapes_match():
    net = ContinuousActorCriticNet(lidar_dim=8, vector_dim=3, hidden_dim=16, action_dim=2)
    lidar, vector = torch.rand(6, 8), torch.rand(6, 3)
    action, logprob, entropy, raw_action = net.act(lidar, vector)
    assert action.shape == (6, 2)
    assert torch.all(action >= -1.0) and torch.all(action <= 1.0)
    assert logprob.shape == (6,)
    assert entropy.shape == (6,)
    assert raw_action.shape == (6, 2)
    assert torch.allclose(action, torch.tanh(raw_action))
    assert torch.isfinite(logprob).all()
    assert torch.isfinite(entropy).all()


def test_continuous_net_evaluate_actions_reproduces_sampled_logprob():
    torch.manual_seed(0)
    net = ContinuousActorCriticNet(lidar_dim=8, vector_dim=3, hidden_dim=16, action_dim=2)
    lidar, vector = torch.rand(10, 8), torch.rand(10, 3)
    action, logprob, _, raw_action = net.act(lidar, vector)
    logprob2, entropy2 = net.evaluate_actions(lidar, vector, raw_action)
    assert torch.allclose(logprob, logprob2, atol=1e-4)
    assert torch.isfinite(entropy2).all()


def test_evaluate_actions_from_raw_action_is_far_more_accurate_than_atanh_reconstruction():
    torch.manual_seed(0)
    net = ContinuousActorCriticNet(lidar_dim=8, vector_dim=3, hidden_dim=16, action_dim=2)
    with torch.no_grad():
        net.actor_mean.bias.fill_(12.0)
    lidar, vector = torch.rand(20, 8), torch.rand(20, 3)
    action, logprob, _, raw_action = net.act(lidar, vector)
    assert raw_action.abs().max() > 7.25, "test setup should actually reach past atanh's clamp ceiling"

    logprob_from_raw_action, _ = net.evaluate_actions(lidar, vector, raw_action)
    atanh_reconstructed = torch.atanh(action.clamp(-1 + 1e-6, 1 - 1e-6))
    logprob_from_atanh, _ = net.evaluate_actions(lidar, vector, atanh_reconstructed)

    error_from_raw_action = (logprob - logprob_from_raw_action).abs().max()
    error_from_atanh = (logprob - logprob_from_atanh).abs().max()
    assert error_from_raw_action < 1e-3
    assert error_from_atanh > 1.0, "saturated actions should make the old atanh path badly wrong"
    assert error_from_raw_action < error_from_atanh / 100


def test_evaluate_actions_handles_extreme_raw_actions_without_nan():
    net = ContinuousActorCriticNet(lidar_dim=8, vector_dim=3, hidden_dim=16, action_dim=2)
    lidar, vector = torch.rand(3, 8), torch.rand(3, 3)
    extreme_raw_action = torch.tensor([[10.0, -10.0], [15.0, 15.0], [-20.0, -20.0]])
    logprob, entropy = net.evaluate_actions(lidar, vector, extreme_raw_action)
    assert torch.isfinite(logprob).all()
    assert torch.isfinite(entropy).all()


def test_continuous_net_std_is_floored_even_if_stored_log_std_is_far_below_the_floor():
    net = ContinuousActorCriticNet(lidar_dim=8, vector_dim=3, hidden_dim=16, action_dim=2)
    with torch.no_grad():
        net.actor_log_std.fill_(-5.0)
    lidar, vector = torch.rand(4, 8), torch.rand(4, 3)
    dist = net._dist(lidar, vector)
    assert torch.all(dist.stddev >= math.exp(ContinuousActorCriticNet.LOG_STD_MIN) - 1e-6)


def test_continuous_net_std_is_capped_even_if_stored_log_std_is_far_above_the_cap():
    net = ContinuousActorCriticNet(lidar_dim=8, vector_dim=3, hidden_dim=16, action_dim=2)
    with torch.no_grad():
        net.actor_log_std.fill_(5.0)
    lidar, vector = torch.rand(4, 8), torch.rand(4, 3)
    dist = net._dist(lidar, vector)
    assert torch.all(dist.stddev <= math.exp(ContinuousActorCriticNet.LOG_STD_MAX) + 1e-6)


def test_set_log_std_floor_accepts_a_per_dimension_sequence():
    net = ContinuousActorCriticNet(lidar_dim=8, vector_dim=3, hidden_dim=16, action_dim=2)
    with torch.no_grad():
        net.actor_log_std.copy_(torch.tensor([-0.5, -5.0]))
    net.set_log_std_floor([net.LOG_STD_MIN, -1.9])
    lidar, vector = torch.rand(4, 8), torch.rand(4, 3)
    dist = net._dist(lidar, vector)
    assert torch.allclose(dist.stddev[:, 0], torch.full((4,), math.exp(net.LOG_STD_MIN)), atol=1e-6)
    assert torch.allclose(dist.stddev[:, 1], torch.full((4,), math.exp(-1.9)), atol=1e-6)

    with torch.no_grad():
        net.actor_log_std.fill_(-5.0)
    net.set_log_std_floor(net.LOG_STD_MIN_FINAL)
    dist = net._dist(lidar, vector)
    assert torch.allclose(
        dist.stddev, torch.full((4, 2), math.exp(net.LOG_STD_MIN_FINAL)), atol=1e-6
    )


def test_centralized_value_num_robots_none_keeps_old_mean_pooling_behavior():
    net = ContinuousActorCriticNet(lidar_dim=8, vector_dim=3, hidden_dim=16, action_dim=2)
    lidar, vector = torch.rand(4, 3, 8), torch.rand(4, 3, 3)
    value = net.centralized_value(lidar, vector)
    assert value.shape == (4,)


def test_centralized_value_flat_concat_distinguishes_spread_from_crushed_robots():
    torch.manual_seed(0)
    net = ContinuousActorCriticNet(lidar_dim=8, vector_dim=3, hidden_dim=16, action_dim=2, num_robots=3)
    lidar_a = torch.rand(1, 3, 8)
    vector_a = torch.rand(1, 3, 3)
    lidar_b = lidar_a[:, [1, 0, 2], :]
    vector_b = vector_a[:, [1, 0, 2], :]
    value_a = net.centralized_value(lidar_a, vector_a)
    value_b = net.centralized_value(lidar_b, vector_b)
    assert value_a.shape == (1, 3)
    assert not torch.allclose(value_a, value_b), (
        "flat-concat critic should generally be sensitive to which robot is which, unlike "
        "mean-pooling (this could theoretically tie by chance, but not with a random untrained "
        "network and non-identical per-robot readings)"
    )


def test_centralized_value_flat_concat_rejects_wrong_robot_count():
    net = ContinuousActorCriticNet(lidar_dim=8, vector_dim=3, hidden_dim=16, action_dim=2, num_robots=3)
    lidar, vector = torch.rand(2, 5, 8), torch.rand(2, 5, 3)
    try:
        net.centralized_value(lidar, vector)
        assert False, "expected an AssertionError for a robot-count mismatch"
    except AssertionError:
        pass


def test_shield_flag_dim_extends_encoder_input_and_act_accepts_the_extra_tensor():
    net = ContinuousActorCriticNet(
        lidar_dim=8, vector_dim=3, hidden_dim=16, action_dim=2, shield_flag_dim=1
    )
    lidar, vector = torch.rand(4, 8), torch.rand(4, 3)
    shield = torch.zeros(4, 1)
    action, logprob, entropy, raw_action = net.act(lidar, vector, shield)
    assert action.shape == (4, 2)
    assert torch.isfinite(logprob).all()
    logprob2, _ = net.evaluate_actions(lidar, vector, raw_action, shield)
    assert torch.allclose(logprob, logprob2, atol=1e-4)

    shield_on = torch.ones(4, 1)
    dist_off = net._dist(lidar, vector, shield)
    dist_on = net._dist(lidar, vector, shield_on)
    assert not torch.allclose(dist_off.mean, dist_on.mean), (
        "shield_intervened=1.0 should reach the encoder and generally change the actor's output "
        "for an untrained network with random weights -- if this ties, the flag isn't wired in"
    )


def test_nearest_robot_dim_combines_with_shield_flag_dim_into_one_extra_tensor():
    net = ContinuousActorCriticNet(
        lidar_dim=8, vector_dim=3, hidden_dim=16, action_dim=2, shield_flag_dim=1, nearest_robot_dim=4,
    )
    assert net.actor_encoder[0].in_features == 8 + 3 + 1 + 4
    lidar, vector = torch.rand(4, 8), torch.rand(4, 3)
    shield = torch.zeros(4, 1)
    nearest_robot_a = torch.zeros(4, 4)
    nearest_robot_b = torch.ones(4, 4)
    extra_a = torch.cat([shield, nearest_robot_a], dim=-1)
    extra_b = torch.cat([shield, nearest_robot_b], dim=-1)
    assert extra_a.shape == (4, 5)
    action, logprob, _, raw_action = net.act(lidar, vector, extra_a)
    assert action.shape == (4, 2)
    logprob2, _ = net.evaluate_actions(lidar, vector, raw_action, extra_a)
    assert torch.allclose(logprob, logprob2, atol=1e-4)

    dist_a = net._dist(lidar, vector, extra_a)
    dist_b = net._dist(lidar, vector, extra_b)
    assert not torch.allclose(dist_a.mean, dist_b.mean), (
        "nearest_robot must reach the encoder and generally change the actor's output for an "
        "untrained network with random weights -- if this ties, the feature isn't wired in"
    )


def test_migrate_architecture_change_encoder_matches_old_net_when_new_feature_is_zero():
    from hiverl.agents.checkpoint import _load_state_dict_partial

    torch.manual_seed(0)
    old_net = ContinuousActorCriticNet(lidar_dim=8, vector_dim=3, hidden_dim=16, action_dim=2)
    new_net = ContinuousActorCriticNet(lidar_dim=8, vector_dim=3, hidden_dim=16, action_dim=2, shield_flag_dim=1)

    _, skipped = _load_state_dict_partial(new_net, old_net.state_dict())
    migrated = migrate_architecture_change(new_net, skipped)
    assert migrated == ["actor_encoder.0.weight"]

    lidar, vector = torch.rand(5, 8), torch.rand(5, 3)
    zero_shield = torch.zeros(5, 1)
    old_embed = old_net.embed(lidar, vector)
    new_embed = new_net.embed(lidar, vector, zero_shield)
    assert torch.allclose(old_embed, new_embed, atol=1e-6)

    old_mean = old_net.actor_mean(old_embed)
    new_mean = new_net.actor_mean(new_embed)
    assert torch.allclose(old_mean, new_mean, atol=1e-6)


def test_migrate_architecture_change_critic_matches_old_mean_pooled_value():
    from hiverl.agents.checkpoint import _load_state_dict_partial

    torch.manual_seed(0)
    old_net = ContinuousActorCriticNet(lidar_dim=8, vector_dim=3, hidden_dim=16, action_dim=2)
    new_net = ContinuousActorCriticNet(
        lidar_dim=8, vector_dim=3, hidden_dim=16, action_dim=2, num_robots=4
    )
    _, skipped = _load_state_dict_partial(new_net, old_net.state_dict())
    migrated = migrate_architecture_change(new_net, skipped, num_robots=4)
    assert migrated == ["critic_encoder.0.weight", "critic_value.weight", "critic_value.bias"]

    lidar, vector = torch.rand(3, 4, 8), torch.rand(3, 4, 3)
    old_value = old_net.centralized_value(lidar, vector)
    new_value = new_net.centralized_value(lidar, vector)
    assert new_value.shape == (3, 4)
    assert torch.allclose(old_value.unsqueeze(-1).expand_as(new_value), new_value, atol=1e-5)


def test_discrete_local_critic_migrates_scalar_checkpoint_exactly():
    from hiverl.agents.checkpoint import _load_state_dict_partial

    torch.manual_seed(1)
    old_net = ActorCriticNet(lidar_dim=8, vector_dim=3, hidden_dim=16, num_actions=5)
    new_net = ActorCriticNet(lidar_dim=8, vector_dim=3, hidden_dim=16, num_actions=5, num_robots=3)
    _, skipped = _load_state_dict_partial(new_net, old_net.state_dict())
    migrated = migrate_architecture_change(new_net, skipped, num_robots=3)

    lidar, vector = torch.rand(2, 3, 8), torch.rand(2, 3, 3)
    old_value = old_net.centralized_value(lidar, vector)
    new_values = new_net.centralized_value(lidar, vector)
    assert migrated == ["critic_head.0.weight", "critic_head.2.weight", "critic_head.2.bias"]
    assert new_values.shape == (2, 3)
    assert torch.allclose(old_value[:, None].expand_as(new_values), new_values, atol=1e-6)


def test_migrate_architecture_change_is_a_noop_when_nothing_to_migrate():
    net = ContinuousActorCriticNet(lidar_dim=8, vector_dim=3, hidden_dim=16, action_dim=2)
    assert migrate_architecture_change(net, {}) == []


def test_discrete_and_continuous_nets_no_longer_share_an_architecture():
    discrete = ActorCriticNet(lidar_dim=8, vector_dim=3, hidden_dim=16, num_actions=5)
    assert hasattr(discrete, "encoder")
    assert hasattr(discrete, "critic_head")
    assert discrete.IS_CONTINUOUS is False

    continuous = ContinuousActorCriticNet(lidar_dim=8, vector_dim=3, hidden_dim=16, action_dim=2)
    assert not hasattr(continuous, "encoder")
    assert not hasattr(continuous, "critic_head")
    assert hasattr(continuous, "actor_encoder")
    assert hasattr(continuous, "actor_mean")
    assert hasattr(continuous, "critic_encoder")
    assert hasattr(continuous, "critic_value")
    assert continuous.IS_CONTINUOUS is True
