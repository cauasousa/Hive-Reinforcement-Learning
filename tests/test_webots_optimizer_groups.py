"""Actor/critic parameter groups of the Webots trainer."""
from scripts.train_mappo_webots import _actor_critic_parameter_groups
from hiverl.agents.networks import ActorCriticNet, ContinuousActorCriticNet


def _assert_complete_disjoint_groups(net):
    actor_params, critic_params = _actor_critic_parameter_groups(net)
    actor_ids = {id(param) for param in actor_params}
    critic_ids = {id(param) for param in critic_params}
    all_ids = {id(param) for param in net.parameters()}

    assert actor_ids.isdisjoint(critic_ids)
    assert actor_ids | critic_ids == all_ids
    return actor_ids, critic_ids


def test_discrete_optimizer_groups_assign_shared_encoder_once():
    net = ActorCriticNet(lidar_dim=8, vector_dim=3, hidden_dim=16, num_actions=5, num_robots=3)

    actor_ids, critic_ids = _assert_complete_disjoint_groups(net)

    assert all(id(param) in actor_ids for param in net.encoder.parameters())
    assert all(id(param) in critic_ids for param in net.critic_head.parameters())


def test_continuous_optimizer_groups_separate_actor_and_critic():
    net = ContinuousActorCriticNet(
        lidar_dim=8, vector_dim=3, actor_hidden_dim=16, critic_hidden_dim=32,
        action_dim=2, num_robots=3,
    )

    actor_ids, critic_ids = _assert_complete_disjoint_groups(net)

    assert all(id(param) in actor_ids for param in net.actor_encoder.parameters())
    assert all(id(param) in actor_ids for param in net.actor_mean.parameters())
    assert id(net.actor_log_std) in actor_ids
    assert all(id(param) in critic_ids for param in net.critic_encoder.parameters())
    assert all(id(param) in critic_ids for param in net.critic_value.parameters())
