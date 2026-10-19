"""Continuous warehouse environment (observations, rewards, tasks, respawn)."""
import math

import numpy as np

from hiverl.env.continuous_kinematics import ARRIVAL_TOLERANCE_M, OMEGA_MAX_RADPS, ROBOT_RADIUS_M, V_MAX_MPS, cell_to_world
from hiverl.env.warehouse_env_continuous import NUM_LIDAR_RAYS, ContinuousWarehouseEnv, _autopilot_velocity
from hiverl.env.warehouse_grid import SHELF


def _make_env(**overrides):
    kwargs = dict(
        width=13,
        height=9,
        num_shelf_blocks_x=2,
        num_shelf_blocks_y=1,
        shelf_block_w=2,
        shelf_block_h=2,
        num_pickup_stations=2,
        num_dropoff_stations=2,
        num_robots=3,
        order_spawn_interval=2,
        max_queue_size=10,
        view_radius=3,
        max_episode_steps=25,
        deadlock_wait_threshold=1,
        seed=0,
    )
    kwargs.update(overrides)
    return ContinuousWarehouseEnv(**kwargs)


def _autopilot_actions(env: ContinuousWarehouseEnv) -> dict:
    actions = {}
    for i, agent in enumerate(env.possible_agents):
        r = env.robots[i]
        target = cell_to_world(r.next_waypoint) if r.next_waypoint is not None else (r.x, r.y)
        v, omega = _autopilot_velocity(r, target)
        actions[agent] = np.array([v / V_MAX_MPS, omega / OMEGA_MAX_RADPS], dtype=np.float32)
    return actions


def test_reset_returns_obs_for_every_agent_with_no_action_mask():
    env = _make_env()
    obs, infos = env.reset(seed=0)
    assert set(obs.keys()) == set(env.possible_agents)
    for agent in env.possible_agents:
        assert obs[agent]["lidar"].shape == (NUM_LIDAR_RAYS,)
        assert obs[agent]["vector"].shape == (3,)
        assert obs[agent]["shield_intervened"].shape == (1,)
        assert obs[agent]["shield_intervened"][0] == 0.0
        assert obs[agent]["nearest_robot"].shape == (4,)
        assert "action_mask" not in obs[agent]


def test_shield_intervened_flag_reflects_a_veto_on_the_next_observation():
    from hiverl.env.robot import RobotStatus
    from hiverl.env.task_manager import Order

    env = _make_env(num_robots=2, order_spawn_interval=1000, max_queue_size=1, max_episode_steps=1000)
    env.reset(seed=0)
    a, b = env.robots[0], env.robots[1]
    for r in (a, b):
        r.status = RobotStatus.TO_ITEM
        r.order = Order(order_id=r.robot_id, pickup=(5, 5), dropoff=(0, 0), created_step=0)
        r.path = [(5, 5)]
    a.x, a.y, a.theta = 0.0, 0.0, 0.0
    b.x, b.y, b.theta = 1.0, 0.0, math.pi

    obs, _, _, _, _ = env.step({
        "robot_0": np.array([1.0, 0.0], dtype=np.float32),
        "robot_1": np.array([1.0, 0.0], dtype=np.float32),
    })
    vetoed = env.last_intervention_events | env.last_collision_events | env.last_wall_block_events
    assert vetoed, "expected the head-on approach to trigger at least one shield intervention"
    for i in vetoed:
        assert obs[f"robot_{i}"]["shield_intervened"][0] == 1.0
    for i in set(env.robots) - vetoed:
        assert obs[f"robot_{i}"]["shield_intervened"][0] == 0.0


def test_nearest_robot_features_are_distance_and_robot_frame_relative_position():
    env = _make_env(num_robots=3)
    env.reset(seed=0)
    a, b, c = env.robots[0], env.robots[1], env.robots[2]
    max_range = max(1, env.view_radius) * 1.0
    a.x, a.y, a.theta = 0.0, 0.0, 0.0
    b.x, b.y = 2.0, 0.0
    c.x, c.y = 10.0, 10.0

    dist, rel_x, rel_y, rel_v = env._nearest_robot_features(0)
    assert math.isclose(dist, min(2.0 / max_range, 1.0))
    assert math.isclose(rel_x, min(2.0 / max_range, 1.0))
    assert math.isclose(rel_y, 0.0, abs_tol=1e-6)
    assert math.isclose(rel_v, 0.0)

    env._last_speed = {0: 0.0, 1: 0.4}
    _, _, _, rel_v = env._nearest_robot_features(0)
    assert math.isclose(rel_v, 0.4 / (2.0 * V_MAX_MPS))

    a.theta = math.pi
    _, rel_x_facing_away, rel_y_facing_away, _ = env._nearest_robot_features(0)
    assert rel_x_facing_away < 0
    assert math.isclose(rel_y_facing_away, 0.0, abs_tol=1e-6)


def test_action_space_is_continuous_box():
    env = _make_env()
    space = env.action_space(env.possible_agents[0])
    assert space.shape == (2,)
    assert float(space.low[0]) == -1.0 and float(space.high[0]) == 1.0


def test_step_runs_and_robots_never_overlap_under_random_actions():
    env = _make_env(max_episode_steps=40)
    env.reset(seed=0)
    rng = np.random.default_rng(1)
    for step in range(40):
        actions = {a: rng.uniform(-1, 1, size=2).astype(np.float32) for a in env.possible_agents}
        obs, rewards, terms, truncs, infos = env.step(actions)
        positions = [(r.x, r.y) for r in env.robots.values()]
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                dist = math.hypot(positions[i][0] - positions[j][0], positions[i][1] - positions[j][1])
                assert dist > 1e-6, f"robots {i},{j} overlapped exactly at step {step}"
        for reward in rewards.values():
            assert math.isfinite(reward)
        if all(truncs.values()):
            break


def test_episode_truncates_at_max_steps():
    env = _make_env(max_episode_steps=5)
    env.reset(seed=0)
    truncs = {a: False for a in env.possible_agents}
    steps = 0
    while not all(truncs.values()) and steps < 100:
        actions = {a: np.zeros(2, dtype=np.float32) for a in env.possible_agents}
        _, _, _, truncs, _ = env.step(actions)
        steps += 1
    assert steps == 5


def test_deliveries_eventually_happen_when_following_waypoints():
    env = _make_env(max_episode_steps=500, order_spawn_interval=3)
    env.reset(seed=0)
    total_delivered = 0
    for _ in range(500):
        env.step(_autopilot_actions(env))
        total_delivered = env.task_manager.total_delivered
        if total_delivered > 0:
            break
    assert total_delivered > 0


def test_robot_cannot_drive_through_a_wall():
    env = _make_env(num_robots=1, max_episode_steps=1000)
    env.reset(seed=0)
    robot = env.robots[0]
    robot.x, robot.y = 0.5, robot.y
    robot.theta = math.pi
    for _ in range(10):
        env.step({"robot_0": np.array([1.0, 0.0], dtype=np.float32)})
    assert env.map.in_bounds(robot.pos)
    assert robot.x >= -1e-6


def test_waypoint_pops_once_within_arrival_tolerance():
    env = _make_env(num_robots=1, order_spawn_interval=1)
    env.reset(seed=0)
    robot = env.robots[0]
    for _ in range(5):
        env.step(_autopilot_actions(env))
        if robot.path:
            break
    assert robot.path, "expected robot to have picked up an assigned route"
    target = cell_to_world(robot.path[0])
    robot.x, robot.y = target[0], target[1] + ARRIVAL_TOLERANCE_M / 2
    remaining_before = len(robot.path)
    env.step(_autopilot_actions(env))
    assert len(robot.path) < remaining_before


def test_progress_reward_uses_the_pre_move_waypoint_not_the_popped_one():
    from hiverl.env.robot import RobotStatus
    from hiverl.env.task_manager import Order
    from hiverl.env.warehouse_env import REWARD_PROGRESS

    env = _make_env(num_robots=1, order_spawn_interval=1000, max_queue_size=1)
    env.reset(seed=0)
    robot = env.robots[0]

    waypoint_a = (5, 6)
    waypoint_b = (5, 10)
    robot.status = RobotStatus.TO_ITEM
    robot.order = Order(order_id=0, pickup=waypoint_b, dropoff=(0, 0), created_step=0)
    robot.path = [waypoint_a, waypoint_b]

    ax, ay = cell_to_world(waypoint_a)
    robot.x, robot.y, robot.theta = ax - 0.25, ay, 0.0

    _, rewards, _, _, _ = env.step({"robot_0": np.array([0.3, 0.0], dtype=np.float32)})

    assert robot.path == [waypoint_b], "expected waypoint_a to have popped"
    assert rewards["robot_0"] > -REWARD_PROGRESS / 2, (
        "reward should reflect real progress toward waypoint_a, not a penalty from comparing "
        "against the newly-popped, much farther waypoint_b"
    )


def test_pbrs_ignores_a_recompute_triggered_route_length_jump():
    from unittest.mock import patch

    from hiverl.env.robot import RobotStatus
    from hiverl.env.task_manager import Order

    env = _make_env(num_robots=2, order_spawn_interval=1000, max_queue_size=1, max_episode_steps=1000)
    env.reset(seed=0)
    a, b = env.robots[0], env.robots[1]
    a.status, b.status = RobotStatus.TO_ITEM, RobotStatus.TO_ITEM
    a.order = Order(order_id=0, pickup=(0, 3), dropoff=(0, 0), created_step=0)
    a.path = [(0, 3)]
    b.order = Order(order_id=1, pickup=(9, 9), dropoff=(0, 0), created_step=0)
    b.path = []
    a.x, a.y, a.theta = 0.0, 0.0, 0.0
    b.x, b.y, b.theta = 0.3, 0.0, math.pi

    long_detour = [(0, 0)] + [(0, k) for k in range(1, 15)]
    with patch("hiverl.env.warehouse_env_continuous.astar", return_value=long_detour):
        _, rewards, _, _, _ = env.step({
            "robot_0": np.array([1.0, 0.0], dtype=np.float32),
            "robot_1": np.array([0.0, 0.0], dtype=np.float32),
        })

    assert 0 in (env.last_veto_events | env.last_intervention_events), "expected robot_0 to be vetoed"
    assert len(a.path) > 3, "expected the mocked long detour to have actually been installed"
    from hiverl.env.warehouse_env_continuous import (
        REWARD_CONTACT_CONTINUOUS,
        REWARD_PHYSICAL_COLLISION_CONTINUOUS,
        REWARD_PROXIMITY_MAX_CONTINUOUS,
    )

    shaping_only = (
        rewards["robot_0"] - REWARD_CONTACT_CONTINUOUS - REWARD_PHYSICAL_COLLISION_CONTINUOUS
        - REWARD_PROXIMITY_MAX_CONTINUOUS
    )
    assert shaping_only > -0.5, (
        f"reward {shaping_only} (contact/collision terms removed) suggests the A*-recompute "
        f"route-length jump leaked into the PBRS progress term instead of being ignored for this tick"
    )


def test_vector_distance_reports_the_total_remaining_route_not_the_lookahead_distance():
    from hiverl.env.robot import RobotStatus
    from hiverl.env.task_manager import Order
    from hiverl.env.warehouse_env_continuous import _LOOKAHEAD_DISTANCE_M

    env = _make_env(num_robots=1, order_spawn_interval=1000, max_queue_size=1, max_episode_steps=1000)
    env.reset(seed=0)
    robot = env.robots[0]
    robot.status = RobotStatus.TO_ITEM
    robot.order = Order(order_id=0, pickup=(0, 10), dropoff=(0, 0), created_step=0)
    robot.path = [(0, k) for k in range(1, 11)]
    robot.x, robot.y, robot.theta = 0.0, 0.0, 0.0

    obs = env._get_obs(0)
    lookahead_frac = _LOOKAHEAD_DISTANCE_M / (env.map.width + env.map.height)
    assert obs["vector"][2] > lookahead_frac * 2, (
        "vector[2] should scale with the full ~10m remaining route, not sit near the constant "
        "~1.5m lookahead distance"
    )


def test_remaining_path_distance_sums_actual_leg_lengths():
    from hiverl.env.warehouse_env_continuous import _remaining_path_distance

    assert _remaining_path_distance((0.0, 0.0), []) == 0.0
    assert math.isclose(_remaining_path_distance((0.0, 0.0), [(0, 3)]), 3.0)
    assert math.isclose(_remaining_path_distance((0.0, 0.0), [(0, 3), (0, 10)]), 10.0)


def test_lookahead_point_interpolates_within_the_straddling_leg():
    from hiverl.env.warehouse_env_continuous import _LOOKAHEAD_DISTANCE_M, _lookahead_point

    path = [(0, 1), (0, 2), (0, 3)]
    assert _lookahead_point((0.0, 0.0), [], 1.5) == (0.0, 0.0)
    lx, ly = _lookahead_point((0.0, 0.0), path, _LOOKAHEAD_DISTANCE_M)
    assert math.isclose(lx, _LOOKAHEAD_DISTANCE_M) and math.isclose(ly, 0.0)
    lx, ly = _lookahead_point((0.0, 0.0), path, 100.0)
    assert (lx, ly) == (3.0, 0.0)


def test_wall_proximity_penalty_ramp():
    from hiverl.env.warehouse_env_continuous import _wall_proximity_penalty

    assert _wall_proximity_penalty(0.5, threshold=0.5, scale=-0.3) == 0.0
    assert _wall_proximity_penalty(1.0, threshold=0.5, scale=-0.3) == 0.0
    assert _wall_proximity_penalty(0.0, threshold=0.5, scale=-0.3) == -0.3
    assert math.isclose(_wall_proximity_penalty(0.25, threshold=0.5, scale=-0.3), -0.15)


def test_nearest_wall_distance_excludes_other_robots():
    env = _make_env(num_robots=2, max_episode_steps=1000)
    env.reset(seed=0)
    robot, other = env.robots[0], env.robots[1]
    other.x, other.y = robot.x, robot.y

    obstacle_dist = env._nearest_obstacle_distance(0, max_range=1.0)
    wall_dist = env._nearest_wall_distance(0, max_range=1.0)

    assert obstacle_dist <= 0.15, "the coincident other robot should dominate the obstacle-inclusive raycast"
    assert wall_dist > obstacle_dist, "the wall-only raycast must not be shortened by the other robot"


def test_wall_proximity_reward_breakdown_activates_near_a_wall():
    from hiverl.env.robot import RobotStatus
    from hiverl.env.task_manager import Order

    env = _make_env(num_robots=1, order_spawn_interval=1000, max_queue_size=1, max_episode_steps=1000)
    env.reset(seed=0)
    robot = env.robots[0]
    robot.status = RobotStatus.TO_ITEM
    robot.order = Order(order_id=0, pickup=(5, 5), dropoff=(0, 0), created_step=0)
    robot.path = [(5, 5)]

    robot.x, robot.y = -0.2, robot.y
    env.step({"robot_0": np.zeros(2, dtype=np.float32)})
    assert env.last_reward_breakdown["wall_proximity"] < 0.0


def test_omega_reversal_penalty_fires_on_a_real_sign_flip():
    from hiverl.env.robot import RobotStatus
    from hiverl.env.task_manager import Order

    env = _make_env(num_robots=1, order_spawn_interval=1000, max_queue_size=1, max_episode_steps=1000)
    env.reset(seed=0)
    robot = env.robots[0]
    robot.status = RobotStatus.TO_ITEM
    robot.order = Order(order_id=0, pickup=(5, 5), dropoff=(0, 0), created_step=0)
    robot.path = [(5, 5)]

    env.step({"robot_0": np.array([0.0, 1.0], dtype=np.float32)})
    assert env.last_reward_breakdown["omega_reversal"] == 0.0

    env.step({"robot_0": np.array([0.0, -1.0], dtype=np.float32)})
    assert env.last_reward_breakdown["omega_reversal"] < 0.0


def test_omega_rate_penalty_scales_with_the_squared_change_in_omega():
    import pytest

    from hiverl.env.robot import RobotStatus
    from hiverl.env.task_manager import Order
    from hiverl.env.warehouse_env_continuous import REWARD_OMEGA_RATE_CONTINUOUS

    def rate_penalty(omega_a, omega_b):
        env = _make_env(num_robots=1, order_spawn_interval=1000, max_queue_size=1, max_episode_steps=1000)
        env.reset(seed=0)
        robot = env.robots[0]
        robot.status = RobotStatus.TO_ITEM
        robot.order = Order(order_id=0, pickup=(5, 5), dropoff=(0, 0), created_step=0)
        robot.path = [(5, 5)]
        env.step({"robot_0": np.array([0.0, omega_a], dtype=np.float32)})
        env.step({"robot_0": np.array([0.0, omega_b], dtype=np.float32)})
        return env.last_reward_breakdown["omega_reversal"]

    assert rate_penalty(0.5, 0.5) == 0.0
    assert rate_penalty(0.8, 0.2) == pytest.approx(REWARD_OMEGA_RATE_CONTINUOUS * 0.36, abs=1e-6)
    assert rate_penalty(1.0, -1.0) == pytest.approx(REWARD_OMEGA_RATE_CONTINUOUS * 4.0, abs=1e-6)
    assert abs(rate_penalty(0.1, -0.1)) < abs(rate_penalty(1.0, -1.0)) / 50


def test_omega_reversal_does_not_compare_across_a_status_change():
    from hiverl.env.robot import RobotStatus
    from hiverl.env.task_manager import Order

    env = _make_env(num_robots=1, order_spawn_interval=1000, max_queue_size=1, max_episode_steps=1000)
    env.reset(seed=0)
    robot = env.robots[0]
    robot.status = RobotStatus.TO_ITEM
    robot.order = Order(order_id=0, pickup=(5, 5), dropoff=(0, 0), created_step=0)
    robot.path = [(5, 5)]
    env.step({"robot_0": np.array([0.0, 1.0], dtype=np.float32)})

    robot.status = RobotStatus.IDLE
    env.step({"robot_0": np.array([0.0, 0.0], dtype=np.float32)})

    robot.status = RobotStatus.TO_ITEM
    env.step({"robot_0": np.array([0.0, -1.0], dtype=np.float32)})
    assert env.last_reward_breakdown["omega_reversal"] == 0.0


def test_delivering_robot_does_not_reuse_another_robots_reserved_free_aisle_cell():
    from hiverl.env.robot import RobotStatus
    from hiverl.env.task_manager import Order

    env = _make_env(num_robots=2, order_spawn_interval=1000, max_queue_size=1, max_episode_steps=1000)
    env.reset(seed=0)
    assert env._nearest_free_aisle_cell((1, 12), set()) == (0, 12), (
        "test assumes this map's layout -- update the cells below if map generation changes"
    )

    robot0, robot1 = env.robots[0], env.robots[1]
    robot0.status = RobotStatus.IDLE
    robot0.pos = (5, 5)
    robot0.x, robot0.y = cell_to_world((5, 5))
    robot0.path = [(0, 12)]

    robot1.status = RobotStatus.CARRYING
    robot1.order = Order(order_id=0, pickup=(5, 5), dropoff=(1, 12), created_step=0)
    robot1.pos = (1, 12)
    robot1.x, robot1.y = cell_to_world((1, 12))
    robot1.path = []

    env.step({
        "robot_0": np.zeros(2, dtype=np.float32), "robot_1": np.zeros(2, dtype=np.float32),
    })

    assert robot1.status == RobotStatus.IDLE
    assert robot1.path, "robot 1 should have been routed to a free aisle cell after delivering"
    assert robot1.path[-1] != (0, 12), "must not reuse robot 0's already-reserved destination"
    assert robot1.path[-1] == (2, 12)


def test_delivering_robot_avoids_a_cell_merely_adjacent_to_another_robots_real_pose():
    from hiverl.env.robot import RobotStatus
    from hiverl.env.task_manager import Order

    env = _make_env(num_robots=2, order_spawn_interval=1000, max_queue_size=1, max_episode_steps=1000)
    env.reset(seed=0)
    assert env._nearest_free_aisle_cell((1, 12), set()) == (0, 12), (
        "test assumes this map's layout -- update the cells below if map generation changes"
    )

    robot0, robot1 = env.robots[0], env.robots[1]
    robot0.status = RobotStatus.IDLE
    robot0.pos = (5, 5)
    cx, cy = cell_to_world((0, 12))
    robot0.x, robot0.y = cx + 0.55, cy + 0.3
    robot0.path = []
    robot0.order = None

    robot1.status = RobotStatus.CARRYING
    robot1.order = Order(order_id=0, pickup=(5, 5), dropoff=(1, 12), created_step=0)
    robot1.pos = (1, 12)
    robot1.x, robot1.y = cell_to_world((1, 12))
    robot1.path = []

    env.step({
        "robot_0": np.zeros(2, dtype=np.float32), "robot_1": np.zeros(2, dtype=np.float32),
    })

    assert robot0.pos == (0, 13), "test setup assumption: robot 0's real pose resyncs to (0, 13), not (0, 12)"
    assert robot1.status == RobotStatus.IDLE
    assert robot1.path, "robot 1 should have been routed to a free aisle cell after delivering"
    assert robot1.path[-1] != (0, 12), "must clear robot 0's real pose by SAFETY_RADIUS_M, not just its logical cell"
    assert robot1.path[-1] == (2, 12)


def test_omega_penalty_is_a_quadratic_dead_zone():
    import pytest

    from hiverl.env.continuous_kinematics import cell_to_world
    from hiverl.env.robot import RobotStatus
    from hiverl.env.task_manager import Order
    from hiverl.env.warehouse_env_continuous import (
        _OMEGA_PENALTY_DEADZONE,
        REWARD_OMEGA_PENALTY_CONTINUOUS,
    )
    from hiverl.env.warehouse_grid import AISLE

    env = _make_env(num_robots=1, order_spawn_interval=1000, max_queue_size=1, max_episode_steps=1000)
    env.reset(seed=0)
    robot = env.robots[0]
    h, w = env.map.grid.shape
    open_cell = next(
        (r, c) for r in range(1, h - 1) for c in range(1, w - 1)
        if all(env.map.grid[r + dr, c + dc] == AISLE for dr in (-1, 0, 1) for dc in (-1, 0, 1))
    )

    def omega_penalty(omega):
        robot.status = RobotStatus.TO_ITEM
        robot.order = Order(order_id=0, pickup=env.map.pickup_stations[0], dropoff=env.map.dropoff_stations[0], created_step=0)
        robot.path = [env.map.pickup_stations[0]]
        robot.x, robot.y = cell_to_world(open_cell)
        robot.theta = 0.0
        robot.sync_pos_from_continuous()
        env.step({"robot_0": np.array([0.0, omega], dtype=np.float32)})
        return env.last_reward_breakdown["omega_penalty"]

    assert omega_penalty(_OMEGA_PENALTY_DEADZONE) == 0.0
    assert omega_penalty(-0.3) == 0.0
    assert omega_penalty(1.0) == pytest.approx(
        REWARD_OMEGA_PENALTY_CONTINUOUS * (1.0 - _OMEGA_PENALTY_DEADZONE) ** 2, abs=1e-6
    )
    assert omega_penalty(-1.0) == omega_penalty(1.0)


def test_respawn_robot_requeues_order_holds_robot_and_penalizes_once():
    import math

    from hiverl.env.robot import RobotStatus
    from hiverl.env.task_manager import Order
    from hiverl.env.warehouse_env_continuous import (
        _RESPAWN_CLEARANCE_M,
        REWARD_PHYSICAL_COLLISION_CONTINUOUS,
    )

    env = _make_env(num_robots=3, order_spawn_interval=1, max_queue_size=10, max_episode_steps=1000)
    env.reset(seed=0)
    robot = env.robots[0]
    order = Order(order_id=42, pickup=env.map.pickup_stations[0], dropoff=env.map.dropoff_stations[0], created_step=0)
    order.picked_up_step = 3
    robot.status, robot.order, robot.path = RobotStatus.CARRYING, order, [order.dropoff]

    cell = env.respawn_robot(0)

    assert cell is not None and robot.pos == cell and robot.status == RobotStatus.IDLE
    assert env.task_manager.pending_orders[0] is order and order.picked_up_step is None
    assert all(
        math.hypot(robot.x - o.x, robot.y - o.y) >= _RESPAWN_CLEARANCE_M
        for j, o in env.robots.items() if j != 0
    )

    zero = {a: np.zeros(2, dtype=np.float32) for a in env.agents}
    _, rewards, terms, _, _ = env.step(zero)
    assert terms["robot_0"] and not terms["robot_1"] and not terms["robot_2"]
    assert env.last_reward_breakdown["physical_collision"] == REWARD_PHYSICAL_COLLISION_CONTINUOUS

    _, _, terms, _, _ = env.step(zero)
    assert not terms["robot_0"] and env.last_reward_breakdown["physical_collision"] == 0.0


def test_held_robot_is_kept_out_of_the_auction_until_released():
    from hiverl.env.robot import RobotStatus
    from hiverl.env.task_manager import Order

    env = _make_env(num_robots=1, order_spawn_interval=1000, max_queue_size=10, max_episode_steps=1000)
    env.reset(seed=0)
    robot = env.robots[0]
    robot.status = RobotStatus.TO_ITEM
    robot.order = Order(order_id=7, pickup=env.map.pickup_stations[0], dropoff=env.map.dropoff_stations[0], created_step=0)
    robot.path = [robot.order.pickup]
    assert env.respawn_robot(0) is not None
    zero = {"robot_0": np.zeros(2, dtype=np.float32)}

    for _ in range(3):
        env.step(zero)
        assert robot.status == RobotStatus.IDLE and robot.order is None

    env.hold_robot_ids.discard(0)
    env.step(zero)
    assert robot.status == RobotStatus.TO_ITEM and robot.order.order_id == 7


def test_active_robot_is_replanned_when_its_cell_is_inside_a_shelf():
    from hiverl.env.continuous_kinematics import cell_to_world
    from hiverl.env.robot import RobotStatus
    from hiverl.env.task_manager import Order
    from hiverl.env.warehouse_grid import SHELF

    env = _make_env(num_robots=1, order_spawn_interval=1000, max_queue_size=1, max_episode_steps=1000)
    env.reset(seed=0)
    robot = env.robots[0]
    shelf_cell = next(tuple(int(v) for v in c) for c in np.argwhere(env.map.grid == SHELF))
    pickup = env.map.pickup_stations[0]

    route = env._route(shelf_cell, pickup)
    assert route and env.map.is_walkable(route[0]) and route[-1] == pickup

    robot.status = RobotStatus.TO_ITEM
    robot.order = Order(order_id=0, pickup=pickup, dropoff=env.map.dropoff_stations[0], created_step=0)
    robot.path = []
    robot.x, robot.y = cell_to_world(shelf_cell)
    robot.sync_pos_from_continuous()
    env.step({"robot_0": np.zeros(2, dtype=np.float32)})

    assert robot.path, "an RL-controlled robot must never be left without a route"
    assert env._get_obs(0)["vector"][2] > 0.0


def _toward_shelf(env):
    from hiverl.env.continuous_kinematics import V_MAX_MPS, cell_to_world
    from hiverl.env.robot import Robot, RobotStatus
    from hiverl.env.safety_shield_continuous import apply_shield_continuous

    shelf = next(tuple(int(v) for v in c) for c in np.argwhere(env.map.grid == SHELF))
    below = (shelf[0] - 1, shelf[1])
    assert env.map.is_walkable(below)
    x, y = cell_to_world(below)
    robot = Robot(0, pos=below, status=RobotStatus.TO_ITEM)
    robot.x, robot.y, robot.theta = x, y + 0.2, math.pi / 2  # heading +y, into the shelf block
    robot.continuous_trail = [(robot.x, robot.y)]
    return apply_shield_continuous(
        robots=[robot], desired_velocities={0: (V_MAX_MPS, 0.0)}, warehouse_map=env.shield_map, dt=0.3,
        enable_veto=True, enable_deadlock_reversal=True, deadlock_wait_threshold=2,
    )


def test_shield_treats_logical_shelves_as_walls_by_default():
    env = ContinuousWarehouseEnv(seed=0)
    env.reset(seed=0)
    assert env.shield_map is env.map
    assert 0 in _toward_shelf(env).wall_block_events


def test_shield_physical_walls_only_ignores_shelves_but_astar_still_avoids_them():
    env = ContinuousWarehouseEnv(seed=0, shield_physical_walls_only=True)
    env.reset(seed=0)
    assert not (env.shield_map.grid == SHELF).any()
    assert (env.map.grid == SHELF).any()
    assert 0 not in _toward_shelf(env).wall_block_events
    path = env._route((0, 0), (env.map.height - 1, env.map.width - 1))
    assert path and all(env.map.grid[c] != SHELF for c in path)
