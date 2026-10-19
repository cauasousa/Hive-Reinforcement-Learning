"""Grid warehouse environment."""
from hiverl.env.warehouse_env import REWARD_PROXIMITY_TIERS, WarehouseEnv, proximity_penalty


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
    return WarehouseEnv(**kwargs)


def test_from_config_matches_direct_construction():
    cfg = dict(
        width=13, height=9, num_shelf_blocks_x=2, num_shelf_blocks_y=1, shelf_block_w=2,
        shelf_block_h=2, num_pickup_stations=2, num_dropoff_stations=2, num_robots=3,
        order_spawn_interval=2, max_queue_size=10, view_radius=3, max_episode_steps=25,
        deadlock_wait_threshold=1, seed=0,
    )
    direct = WarehouseEnv(**cfg)
    from_cfg = WarehouseEnv.from_config(cfg)
    assert from_cfg.num_robots == direct.num_robots
    assert from_cfg.map.width == direct.map.width
    assert from_cfg.max_episode_steps == direct.max_episode_steps


def test_from_config_overrides_take_priority_and_ignore_unknown_keys():
    cfg = dict(
        width=13, height=9, num_shelf_blocks_x=2, num_shelf_blocks_y=1, shelf_block_w=2,
        shelf_block_h=2, num_pickup_stations=2, num_dropoff_stations=2, num_robots=3,
        order_spawn_interval=2, max_queue_size=10, view_radius=3, max_episode_steps=25,
        deadlock_wait_threshold=1, seed=0, not_a_real_param="ignore me",
    )
    env = WarehouseEnv.from_config(cfg, num_robots=5, max_episode_steps=10**9)
    assert env.num_robots == 5
    assert env.max_episode_steps == 10**9


def test_proximity_penalty_is_graduated_and_zero_beyond_threshold():
    assert proximity_penalty(5.0) == 0.0
    assert proximity_penalty(1.5) == -0.03
    assert proximity_penalty(1.0) == -0.08
    thresholds = sorted({t for t, _ in REWARD_PROXIMITY_TIERS} | {0.0, 2.0})
    penalties = [proximity_penalty(t) for t in thresholds]
    assert all(a <= b for a, b in zip(penalties, penalties[1:]))


def test_nearest_other_distance_tracks_closest_robot_and_is_inf_when_alone():
    env = _make_env(num_robots=3)
    env.reset(seed=0)
    env.robots[0].pos = (2, 2)
    env.robots[1].pos = (2, 3)
    env.robots[2].pos = (5, 5)
    assert env._nearest_other_distance(0) == 1.0

    solo = _make_env(num_robots=1)
    solo.reset(seed=0)
    assert solo._nearest_other_distance(0) == float("inf")


def test_reset_returns_obs_and_infos_for_every_agent():
    env = _make_env()
    obs, infos = env.reset(seed=0)
    assert set(obs.keys()) == set(env.possible_agents)
    for agent in env.possible_agents:
        assert obs[agent]["lidar"].shape == (8,)
        assert obs[agent]["vector"].shape == (3,)
        assert obs[agent]["action_mask"].shape == (5,)


def test_step_runs_and_never_lets_two_robots_share_a_cell():
    env = _make_env()
    env.reset(seed=0)
    rng_actions = [0, 1, 2, 3, 4]
    for step in range(25):
        actions = {agent: rng_actions[(step + i) % 5] for i, agent in enumerate(env.possible_agents)}
        obs, rewards, terms, truncs, infos = env.step(actions)
        positions = [r.pos for r in env.robots.values()]
        assert len(positions) == len(set(positions)), "two robots ended up on the same cell"
        for pos in positions:
            assert env.map.is_walkable(pos)
        if all(truncs.values()):
            break


def test_episode_truncates_at_max_steps():
    env = _make_env(max_episode_steps=5)
    env.reset(seed=0)
    truncs = {a: False for a in env.possible_agents}
    steps = 0
    while not all(truncs.values()) and steps < 100:
        actions = {a: 4 for a in env.possible_agents}
        _, _, _, truncs, _ = env.step(actions)
        steps += 1
    assert steps == 5


def test_veto_only_shield_never_lets_two_robots_share_a_cell():
    import numpy as np

    from hiverl.agents.scripted_astar_agent import scripted_actions

    env = _make_env(
        width=9,
        height=7,
        num_shelf_blocks_x=1,
        num_shelf_blocks_y=1,
        shelf_block_w=2,
        shelf_block_h=2,
        num_robots=5,
        order_spawn_interval=2,
        max_episode_steps=100,
    )
    env.enable_veto = True
    env.enable_deadlock_reversal = False
    env.reset(seed=1)
    rng = np.random.default_rng(1)
    any_intervention = False
    any_collision = False
    for _ in range(100):
        actions = scripted_actions(env)
        for agent in actions:
            if rng.random() < 0.5:
                actions[agent] = int(rng.integers(0, 5))
        env.step(actions)
        positions = [r.pos for r in env.robots.values()]
        assert len(positions) == len(set(positions)), "two robots ended up on the same cell"
        any_intervention = any_intervention or bool(env.last_intervention_events)
        any_collision = any_collision or bool(env.last_collision_events)
    assert any_intervention, "test setup didn't actually create any conflicts to veto"
    assert any_collision, "test setup didn't actually create any reactive vertex collisions"


def test_deliveries_eventually_happen_when_following_astar():
    from hiverl.agents.scripted_astar_agent import scripted_actions

    env = _make_env(max_episode_steps=200, order_spawn_interval=3)
    env.reset(seed=0)
    total_delivered = 0
    for _ in range(200):
        actions = scripted_actions(env)
        env.step(actions)
        total_delivered = env.task_manager.total_delivered
        if total_delivered > 0:
            break
    assert total_delivered > 0


def test_robot_relocates_off_station_after_delivering_there():
    from hiverl.env.robot import RobotStatus
    from hiverl.env.task_manager import Order
    from hiverl.env.warehouse_grid import DROPOFF

    env = _make_env(num_robots=1, order_spawn_interval=1000, max_queue_size=1)
    env.reset(seed=0)
    robot = env.robots[0]
    dropoff = env.map.dropoff_stations[0]
    robot.pos = dropoff
    robot.status = RobotStatus.CARRYING
    robot.order = Order(order_id=0, pickup=(0, 0), dropoff=dropoff, created_step=0)
    robot.path = []

    obs, rewards, terms, truncs, infos = env.step({"robot_0": 4})

    assert robot.status == RobotStatus.IDLE
    assert env.map.grid[robot.pos] == DROPOFF
    assert robot.path, "expected a pending relocation path off the station"

    env.step({"robot_0": 4})
    assert robot.pos != dropoff


def test_pickup_holds_picking_up_status_for_one_tick_before_carrying():
    from hiverl.env.robot import RobotStatus
    from hiverl.env.safety_shield import WAIT_ACTION
    from hiverl.env.task_manager import Order

    env = _make_env(num_robots=1, order_spawn_interval=1000, max_queue_size=1)
    env.reset(seed=0)
    robot = env.robots[0]
    pickup = env.map.pickup_stations[0]
    dropoff = env.map.dropoff_stations[0]
    robot.pos = pickup
    robot.status = RobotStatus.TO_ITEM
    robot.order = Order(order_id=0, pickup=pickup, dropoff=dropoff, created_step=0)
    robot.path = []

    _obs, rewards, _terms, _truncs, _infos = env.step({"robot_0": WAIT_ACTION})
    assert robot.status == RobotStatus.PICKING_UP
    assert rewards["robot_0"] > 0
    assert robot.pos == pickup

    _obs, _rewards, _terms, _truncs, _infos = env.step({"robot_0": 0})
    assert robot.status == RobotStatus.CARRYING
    assert robot.pos == pickup
    assert robot.path, "expected an A* path toward dropoff to already be set"


def test_auction_wont_send_a_second_robot_to_a_station_still_physically_occupied():
    from hiverl.env.robot import RobotStatus
    from hiverl.env.task_manager import Order

    env = _make_env(num_robots=2, order_spawn_interval=1000, max_queue_size=10)
    env.reset(seed=0)
    pickup = env.map.pickup_stations[0]
    dropoff = env.map.dropoff_stations[0]

    occupying_robot = env.robots[0]
    occupying_robot.pos = pickup
    occupying_robot.status = RobotStatus.CARRYING
    occupying_robot.order = Order(order_id=0, pickup=pickup, dropoff=dropoff, created_step=0)
    occupying_robot.path = [dropoff]

    idle_robot = env.robots[1]
    idle_robot.pos = (0, 0)
    idle_robot.status = RobotStatus.IDLE

    conflicting_order = Order(order_id=1, pickup=pickup, dropoff=dropoff, created_step=0)
    env.task_manager.pending_orders.append(conflicting_order)

    env.step({"robot_0": 4, "robot_1": 4})

    assert idle_robot.order is None
    assert conflicting_order in env.task_manager.pending_orders


def test_priority_key_treats_picking_up_the_same_as_carrying():
    from hiverl.env.robot import Robot, RobotStatus

    picking_up = Robot(robot_id=0, pos=(0, 0), status=RobotStatus.PICKING_UP)
    carrying = Robot(robot_id=1, pos=(0, 0), status=RobotStatus.CARRYING)
    idle = Robot(robot_id=2, pos=(0, 0), status=RobotStatus.IDLE)
    assert picking_up.priority_key()[0] == carrying.priority_key()[0] == 1
    assert idle.priority_key()[0] == 0
