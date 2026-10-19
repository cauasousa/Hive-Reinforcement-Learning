"""MAPPO alignment: real-pose rewards, inactive-agent masking, team reward, agent state, value loss,
adaptive lr.
"""
import math

import numpy as np
import pytest
import torch

from hiverl.agents.networks import ContinuousActorCriticNet
from hiverl.agents.ppo import PPOConfig, RunningMeanStd
from hiverl.bridge.lockstep_coordinator import REAL_OMEGA_DERATE, ContinuousLockstepCoordinator
from hiverl.bridge.robot_driver import FakeRobotDriver
from hiverl.env.continuous_kinematics import cell_to_world
from hiverl.env.robot import RobotStatus
from hiverl.env.task_manager import Order
from hiverl.env.warehouse_env_continuous import AGENT_STATE_DIM, ContinuousWarehouseEnv
from scripts.train_mappo_webots import _actor_critic_parameter_groups, _ppo_update, collect_and_update

_ENV = dict(
    width=13, height=9, num_shelf_blocks_x=2, num_shelf_blocks_y=1, shelf_block_w=2, shelf_block_h=2,
    num_pickup_stations=2, num_dropoff_stations=2, num_robots=3, order_spawn_interval=1000,
    max_queue_size=5, view_radius=3, max_episode_steps=10**9, deadlock_wait_threshold=1,
)


def _env(**kw):
    env = ContinuousWarehouseEnv(**{**_ENV, **kw}, seed=0)
    env.reset(seed=0)
    return env


def _to_item(env, rid=0):
    r = env.robots[rid]
    pickup, dropoff = env.map.pickup_stations[0], env.map.dropoff_stations[0]
    r.status, r.order = RobotStatus.TO_ITEM, Order(order_id=rid, pickup=pickup, dropoff=dropoff, created_step=0)
    r.path = env._route(r.pos, pickup)
    return r


def test_reward_uses_the_real_pose_from_the_pose_provider():
    env = _env()
    r = _to_item(env)
    frozen = {i: (rb.x, rb.y, rb.theta) for i, rb in env.robots.items()}
    received = {}

    def provider(final_velocities):
        received.update(final_velocities)
        return frozen

    env.pose_provider = provider
    actions = {a: np.array([1.0, 0.0], dtype=np.float32) for a in env.agents}
    env.step(actions)
    assert received, "the provider must receive the shield-corrected velocities to execute"
    assert (r.x, r.y) == frozen[0][:2]
    assert env.last_reward_breakdown["progress_pos"] == pytest.approx(0.0, abs=1e-9)


def test_team_reward_splits_the_delivery_across_the_fleet():
    rewards = {}
    for share in (0.0, 0.5):
        env = _env(team_reward_share=share)
        r = env.robots[0]
        dropoff = env.map.dropoff_stations[0]
        r.status, r.order = RobotStatus.CARRYING, Order(0, env.map.pickup_stations[0], dropoff, 0)
        r.x, r.y = cell_to_world(dropoff)
        r.sync_pos_from_continuous()
        r.path = []
        _, rew, _, _, _ = env.step({a: np.zeros(2, dtype=np.float32) for a in env.agents})
        rewards[share] = rew
    solo, team = rewards[0.0], rewards[0.5]
    assert team["robot_1"] > solo["robot_1"] and team["robot_0"] < solo["robot_0"]
    assert sum(team.values()) == pytest.approx(sum(solo.values()), abs=1e-6)


def test_agent_state_exposes_status_command_and_motion():
    env = _env()
    _to_item(env)
    obs, _, _, _, _ = env.step({a: np.array([0.5, 0.0], dtype=np.float32) for a in env.agents})
    state = obs["robot_0"]["agent_state"]
    assert state.shape == (AGENT_STATE_DIM,)
    assert list(state[:4]) == [0.0, 1.0, 0.0, 0.0]
    assert 0 in env.last_policy_controlled_ids and 1 not in env.last_policy_controlled_ids


def test_coordinator_executes_shield_velocities_and_stops_robots_at_round_end():
    env = ContinuousWarehouseEnv(**_ENV, seed=0)
    driver = FakeRobotDriver()
    co = ContinuousLockstepCoordinator(env, driver, arrival_timeout_s=5.0, synchronize_fleet=True,
                                       poll_interval_s=0.0, enable_legacy_retreats=False,
                                       execute_shield_velocities=True)
    co._sleep = lambda s: None
    co.reset()
    _to_item(env)
    pairs = []

    def on_tick(tick, result):
        sent_v, sent_w = [(v, w) for rid, v, w in driver.velocity_history if rid == 0][-1]
        pairs.append((env.last_final_velocities[0], (sent_v, sent_w)))

    co.on_tick = on_tick
    co.run(3, lambda obs: {a: np.array([0.8, 0.6], dtype=np.float32) for a in obs}, reset=False)
    assert pairs
    for (final_v, final_w), (sent_v, sent_w) in pairs:
        assert math.isclose(sent_v, final_v, abs_tol=1e-9)
        assert math.isclose(sent_w, final_w * REAL_OMEGA_DERATE, abs_tol=1e-9)
    last_sent = [(v, w) for rid, v, w in driver.velocity_history if rid == 0][-1]
    assert last_sent == (0.0, 0.0), "round end must stop velocity-controlled robots"


def _tiny_net(env, **kw):
    obs_space = env.observation_space(env.possible_agents[0])
    torch.manual_seed(0)
    return ContinuousActorCriticNet(
        lidar_dim=obs_space["lidar"].shape[-1], vector_dim=obs_space["vector"].shape[-1],
        actor_hidden_dim=16, critic_hidden_dim=16, action_dim=2, num_robots=3,
        shield_flag_dim=1, nearest_robot_dim=4, **kw,
    )


def test_inactive_robot_ticks_do_not_move_the_actor():
    env = _env()
    net = _tiny_net(env)
    T, N = 8, 3
    obs_d = 16
    seg = {
        "lidar": torch.rand(T, N, obs_d), "vector": torch.rand(T, N, 3), "shield": torch.rand(T, N, 5),
        "mask": None, "actions": torch.rand(T, N, 2), "raw_actions": torch.randn(T, N, 2),
        "logprobs": torch.randn(T, N), "adv": torch.randn(T, N), "targets": torch.randn(T, N),
        "active": torch.zeros(T, N), "old_values": torch.randn(T, N),
    }
    a_params, c_params = _actor_critic_parameter_groups(net)
    a_opt, c_opt = torch.optim.Adam(a_params, lr=1e-2), torch.optim.Adam(c_params, lr=1e-2)
    actor_before = [p.detach().clone() for p in a_params]
    critic_before = [p.detach().clone() for p in c_params]
    cfg = PPOConfig(update_epochs=2, minibatches=2, entropy_coef=0.0, value_loss="huber", value_clip=0.2)
    _ppo_update(net, a_opt, a_params, c_opt, c_params, RunningMeanStd(), [seg], cfg, N, True, "cpu",
                "t", 1, None, None)
    assert all(torch.equal(b, p) for b, p in zip(actor_before, a_params)), "masked ticks trained the actor"
    assert any(not torch.equal(b, p) for b, p in zip(critic_before, c_params)), "critic must still train"


def test_adaptive_lr_increases_when_kl_is_tiny():
    env = _env()
    net = _tiny_net(env)
    T, N = 6, 3
    seg = {
        "lidar": torch.rand(T, N, 16), "vector": torch.rand(T, N, 3), "shield": torch.rand(T, N, 5),
        "mask": None, "actions": torch.rand(T, N, 2), "raw_actions": torch.randn(T, N, 2),
        "adv": torch.randn(T, N), "targets": torch.randn(T, N), "active": torch.ones(T, N),
        "old_values": torch.randn(T, N),
    }
    with torch.no_grad():
        seg["logprobs"], _ = net.evaluate_actions(
            seg["lidar"].reshape(-1, 16), seg["vector"].reshape(-1, 3), seg["raw_actions"].reshape(-1, 2),
            seg["shield"].reshape(-1, 5),
        )
        seg["logprobs"] = seg["logprobs"].reshape(T, N)
    a_params, c_params = _actor_critic_parameter_groups(net)
    a_opt, c_opt = torch.optim.Adam(a_params, lr=1e-6), torch.optim.Adam(c_params, lr=1e-4)
    cfg = PPOConfig(update_epochs=1, minibatches=1, desired_kl=0.01)
    out = _ppo_update(net, a_opt, a_params, c_opt, c_params, RunningMeanStd(), [seg], cfg, N, True, "cpu",
                      "t", 1, None, None)
    assert out["actor_lr"] == pytest.approx(1.5e-6)


def test_full_round_with_all_mappo_options_on_the_fake_driver():
    env = ContinuousWarehouseEnv(**{**_ENV, "order_spawn_interval": 2}, seed=0, team_reward_share=0.5)
    agents = env.possible_agents
    env.reset(seed=0)
    net = _tiny_net(env, agent_state_dim=AGENT_STATE_DIM, soft_log_std=True, obs_norm=True)
    a_params, c_params = _actor_critic_parameter_groups(net)
    a_opt, c_opt = torch.optim.Adam(a_params, lr=3e-4), torch.optim.Adam(c_params, lr=3e-4)
    co = ContinuousLockstepCoordinator(env, FakeRobotDriver(), arrival_timeout_s=5.0, synchronize_fleet=True,
                                       poll_interval_s=0.0, enable_legacy_retreats=False)
    co._sleep = lambda s: None
    cfg = PPOConfig(update_epochs=2, minibatches=2, value_loss="huber", value_clip=0.2, desired_kl=0.01)
    summary = collect_and_update(co, net, a_opt, c_opt, RunningMeanStd(), agents, "cpu", cfg, 30, "r1",
                                 reset=True, is_continuous=True, round_idx=1, min_update_ticks=0)
    assert summary["update_skipped"] == 0 and summary["ticks"] == 30
    assert float(net.obs_count) == 30 * 3, "obs normalizer must fold in the round's robot-ticks"
    assert "actor_lr" in summary


def test_robot_gets_velocities_again_right_after_a_pickup():
    class _NeverAcksDriver(FakeRobotDriver):
        def poll_arrived(self, robot_id: int) -> bool:
            return False

    env = ContinuousWarehouseEnv(**_ENV, seed=0)
    driver = _NeverAcksDriver()
    co = ContinuousLockstepCoordinator(env, driver, arrival_timeout_s=5.0, synchronize_fleet=True,
                                       poll_interval_s=0.0, enable_legacy_retreats=False)
    co._sleep = lambda s: None
    co.reset()
    from hiverl.bridge.coords import cell_to_world as world_cell

    r = _to_item(env)
    driver.poses[0] = (*world_cell(r.order.pickup), 0.0)
    other = env.robots[1]
    other.status = RobotStatus.TO_ITEM
    other.order = Order(order_id=1, pickup=env.map.pickup_stations[1], dropoff=env.map.dropoff_stations[1], created_step=0)
    other.path = env._route(other.pos, other.order.pickup)
    statuses = []
    co.on_tick = lambda tick, result: statuses.append(env.robots[0].status)
    co.run(8, lambda obs: {a: np.array([0.6, 0.0], dtype=np.float32) for a in obs}, reset=False)
    assert RobotStatus.CARRYING in statuses, f"pickup never happened: {statuses}"
    after_pickup = statuses.index(RobotStatus.CARRYING)
    assert all(s == RobotStatus.CARRYING for s in statuses[after_pickup:]), statuses
    assert not co._real_pickup_in_progress, "PICKING_UP hold must be released once CARRYING"
    moving = [v for rid, v, w in driver.velocity_history if rid == 0 and abs(v) > 0]
    assert moving, "robot 0 must receive non-zero velocities after its pickup"


def _two_close_robots(env, gap_m=0.6):
    a, b = env.robots[0], env.robots[1]
    x0, y0 = cell_to_world((0, 8))
    a.x, a.y, a.theta = x0, y0, 0.0
    b.x, b.y, b.theta = x0 + gap_m, y0, 0.0
    for r in (a, b):
        r.sync_pos_from_continuous()
        r.blocked_steps, r.reversal_cooldown = 0, 0
    env.robots[2].x, env.robots[2].y = cell_to_world((8, 0))
    env.robots[2].sync_pos_from_continuous()
    return a, b


def test_shield_lets_a_robot_escape_an_existing_violation_but_not_close_in():
    from hiverl.env.safety_shield_continuous import apply_shield_continuous

    env = _env()
    a, b = _two_close_robots(env)
    robots = list(env.robots.values())
    common = dict(warehouse_map=env.map, dt=0.3, enable_veto=True, enable_deadlock_reversal=False, deadlock_wait_threshold=1)
    away = apply_shield_continuous(robots=robots, desired_velocities={0: (-0.5, 0.0), 1: (0.0, 0.0), 2: (0.0, 0.0)}, **common)
    assert away.final_velocities[0][0] < 0.0, "backing away from a too-close robot must be allowed"
    toward = apply_shield_continuous(robots=robots, desired_velocities={0: (0.5, 0.0), 1: (0.0, 0.0), 2: (0.0, 0.0)}, **common)
    assert toward.final_velocities[0] == (0.0, 0.0), "closing in further must still be vetoed"


def test_deadlock_reversal_is_a_real_backward_velocity():
    from hiverl.env.safety_shield_continuous import apply_shield_continuous

    env = _env()
    a, b = _two_close_robots(env)
    a.continuous_trail = [(a.x - 1.0, a.y), (a.x, a.y)]
    a.blocked_steps = 5
    res = apply_shield_continuous(
        robots=list(env.robots.values()), desired_velocities={0: (0.5, 0.0), 1: (0.0, 0.0), 2: (0.0, 0.0)},
        warehouse_map=env.map, dt=0.3, enable_veto=True, enable_deadlock_reversal=True, deadlock_wait_threshold=1,
    )
    assert 0 in res.reversal_events
    assert res.final_velocities[0][0] < 0.0, "a reversal must be executable by a real robot"
    assert res.final_velocities[0][1] == 0.0


def test_by_default_the_real_robot_executes_the_policy_action_and_obs_reports_it():
    env = ContinuousWarehouseEnv(**_ENV, seed=0)
    driver = FakeRobotDriver()
    co = ContinuousLockstepCoordinator(env, driver, arrival_timeout_s=5.0, synchronize_fleet=True,
                                       poll_interval_s=0.0, enable_legacy_retreats=False)
    co._sleep = lambda s: None
    co.reset()
    _to_item(env)
    co.run(2, lambda obs: {a: np.array([0.8, 0.6], dtype=np.float32) for a in obs}, reset=False)
    from hiverl.env.continuous_kinematics import OMEGA_MAX_RADPS, V_MAX_MPS

    first = [(v, w) for rid, v, w in driver.velocity_history if rid == 0][0]
    assert math.isclose(first[0], 0.8 * V_MAX_MPS, rel_tol=1e-6)
    assert math.isclose(first[1], 0.6 * OMEGA_MAX_RADPS * REAL_OMEGA_DERATE, rel_tol=1e-6)
    state = env._agent_state(0)
    assert math.isclose(float(state[4]), 0.8, rel_tol=1e-5), "agent_state must show the EXECUTED command"


def _place_pair(env, d, cell=(0, 8)):
    a, b = env.robots[0], env.robots[1]
    x0, y0 = cell_to_world(cell)
    a.x, a.y, a.theta = x0, y0, 0.0
    b.x, b.y, b.theta = x0 + d, y0, math.pi
    env.robots[2].x, env.robots[2].y = cell_to_world((8, 0))
    for r in env.robots.values():
        r.sync_pos_from_continuous()
    return a, b


def _step_frozen(env):
    frozen = {i: (r.x, r.y, r.theta) for i, r in env.robots.items()}
    env.pose_provider = lambda v: frozen
    try:
        return env.step({a: np.zeros(2, dtype=np.float32) for a in env.agents})
    finally:
        env.pose_provider = None


def test_contact_is_charged_every_tick_to_both_robots_and_only_inside_contact_distance():
    from hiverl.env.warehouse_env_continuous import CONTACT_DISTANCE_M, REWARD_CONTACT_CONTINUOUS

    env = _env()
    _place_pair(env, CONTACT_DISTANCE_M - 0.03)
    for _ in range(2):
        _step_frozen(env)
        assert env.last_contact_ids == {0, 1}
        assert env.last_reward_breakdown["contact"] == pytest.approx(2 * REWARD_CONTACT_CONTINUOUS)
        assert env.last_reward_breakdown["physical_collision"] == 0.0, "0.45 m is contact, not a deep overlap"
    env2 = _env()
    _place_pair(env2, CONTACT_DISTANCE_M + 0.05)
    _step_frozen(env2)
    assert env2.last_contact_ids == set() and env2.last_reward_breakdown["contact"] == 0.0


def test_graded_proximity_grows_as_robots_get_closer():
    from hiverl.env.warehouse_env_continuous import (
        CONTACT_DISTANCE_M,
        PROXIMITY_START_M,
        REWARD_PROXIMITY_MAX_CONTINUOUS,
        _graded_proximity_penalty,
    )

    distances = [PROXIMITY_START_M + 0.1, PROXIMITY_START_M, 1.0, 0.8, 0.6, CONTACT_DISTANCE_M, 0.3]
    values = [_graded_proximity_penalty(d) for d in distances]
    assert values[0] == 0.0 and values[1] == 0.0
    assert all(a >= b for a, b in zip(values, values[1:])), values
    assert values[-2] == pytest.approx(REWARD_PROXIMITY_MAX_CONTINUOUS)
    assert _graded_proximity_penalty(0.99) < -0.03 < 0.0, "0.99 m must cost far less than contact"


def test_deep_overlap_is_charged_once_on_the_causal_tick_and_respawn_does_not_double_charge():
    from hiverl.env.warehouse_env_continuous import REWARD_PHYSICAL_COLLISION_CONTINUOUS

    env = _env()
    _place_pair(env, 0.3)
    _step_frozen(env)
    assert env.last_reward_breakdown["physical_collision"] == pytest.approx(2 * REWARD_PHYSICAL_COLLISION_CONTINUOUS)
    _step_frozen(env)
    assert env.last_reward_breakdown["physical_collision"] == 0.0, "same overlap episode must not repeat"

    driver = FakeRobotDriver()
    env3 = ContinuousWarehouseEnv(**_ENV, seed=0)
    co = ContinuousLockstepCoordinator(env3, driver, arrival_timeout_s=5.0, synchronize_fleet=True, poll_interval_s=0.0,
                                       global_collision_reset_m=0.40, respawn_on_collision=True, enable_legacy_retreats=False)
    co._sleep = lambda s: None
    co.reset()
    x0, y0, _ = driver.poses[0]
    driver.poses[1] = (x0 + 0.3, y0, 0.0)
    env3._deep_pairs = {(0, 1)}
    totals = []
    co.on_tick = lambda t, r: totals.append(env3.last_reward_breakdown["physical_collision"])
    co.run(2, lambda obs: {a: np.zeros(2, dtype=np.float32) for a in obs}, reset=False)
    assert totals[0] == 0.0, "respawn must not charge an overlap the env already charged"
    assert co.last_terminations.get("robot_0") is not None


def test_relative_velocity_uses_the_measured_speed_of_the_other_robot():
    env = _env()
    a, b = _place_pair(env, 1.0)
    moved = {i: (r.x, r.y, r.theta) for i, r in env.robots.items()}
    bx, by, bt = moved[1]
    moved[1] = (bx - 0.3, by, bt)
    env.pose_provider = lambda v: moved
    env.step({ag: np.zeros(2, dtype=np.float32) for ag in env.agents})
    env.pose_provider = None
    _, _, _, rel_v = env._nearest_robot_features(0)
    assert rel_v > 0.0, "a robot really closing in must not look stopped"


def test_coordinator_flags_shield_blocked_robots_only_when_executing_the_shield():
    class Recorder(FakeRobotDriver):
        def __init__(self):
            super().__init__()
            self.flags = []

        def set_shield_blocked(self, robot_id, blocked):
            self.flags.append((robot_id, blocked))

    for execute in (True, False):
        env = ContinuousWarehouseEnv(**_ENV, seed=0, shield_active_resolution=execute)
        driver = Recorder()
        co = ContinuousLockstepCoordinator(env, driver, arrival_timeout_s=5.0, synchronize_fleet=True,
                                           poll_interval_s=0.0, enable_legacy_retreats=False,
                                           execute_shield_velocities=execute)
        co._sleep = lambda s: None
        co.reset()
        _to_item(env)
        co.run(2, lambda obs: {a: np.array([0.8, 0.0], dtype=np.float32) for a in obs}, reset=False)
        robot0 = [b for rid, b in driver.flags if rid == 0]
        assert robot0, "the flag must be refreshed every tick for velocity-controlled robots"
        if not execute:
            assert not any(robot0)
