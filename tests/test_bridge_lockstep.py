"""Lockstep coordinators (grid and continuous) with the fake robot driver."""
import pytest

from hiverl.bridge.lockstep_coordinator import STATUS_CODE, ContinuousLockstepCoordinator, LockstepCoordinator
from hiverl.bridge.robot_driver import FakeRobotDriver
from hiverl.env.robot import RobotStatus
from hiverl.env.warehouse_env import WarehouseEnv
from hiverl.env.warehouse_env_continuous import ContinuousWarehouseEnv


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
        max_episode_steps=10**9,
        deadlock_wait_threshold=1,
        seed=0,
    )
    kwargs.update(overrides)
    return WarehouseEnv(**kwargs)


def test_fake_driver_send_velocity_integrates_pose_and_negates_omega_in_world_frame():
    import math

    driver = FakeRobotDriver()
    driver.reset(0, (0, 0))
    x0, y0, theta0 = driver.get_pose(0)
    assert theta0 == 0.0

    driver.send_velocity(0, v=1.0, omega=0.5)
    x1, y1, theta1 = driver.get_pose(0)
    assert driver.velocity_history == [(0, 1.0, 0.5)]
    assert math.isclose(theta1, -0.5)
    assert math.isclose(x1, x0 + 1.0 * math.cos(theta0))
    assert math.isclose(y1, y0 + 1.0 * math.sin(theta0))


def test_reset_sends_every_robot_start_cell_to_the_driver():
    env = _make_env()
    driver = FakeRobotDriver(ticks_to_arrive=0)
    coordinator = LockstepCoordinator(env, driver)

    coordinator.reset(seed=0)

    assert set(driver.positions.keys()) == set(env.robots.keys())
    for robot_id, robot in env.robots.items():
        assert driver.positions[robot_id] == robot.pos


def test_run_scripted_astar_matches_env_final_positions_when_driver_is_instant():
    env = _make_env()
    driver = FakeRobotDriver(ticks_to_arrive=0)
    coordinator = LockstepCoordinator(env, driver, poll_interval_s=0.0)

    stats = coordinator.run(num_ticks=30, policy_fn=None)

    assert len(stats.ticks) == 30
    for robot_id, robot in env.robots.items():
        assert driver.positions[robot_id] == robot.pos


def test_lockstep_waits_out_a_simulated_travel_delay_without_desyncing():
    env = _make_env()
    driver = FakeRobotDriver(ticks_to_arrive=4)
    coordinator = LockstepCoordinator(env, driver, poll_interval_s=0.0)

    stats = coordinator.run(num_ticks=15, policy_fn=None)

    assert stats.timed_out_ticks == 0
    for robot_id, robot in env.robots.items():
        assert driver.positions[robot_id] == robot.pos


def test_one_slow_robot_does_not_block_the_others():
    env = _make_env()

    class OneSlowRobotDriver(FakeRobotDriver):
        SLOW_ROBOT_ID = 0
        SLOW_TICKS_TO_ARRIVE = 50

        def poll_arrived(self, robot_id: int) -> bool:
            if robot_id == self.SLOW_ROBOT_ID:
                self._polls_since_goal[robot_id] += 1
                if self._polls_since_goal[robot_id] >= self.SLOW_TICKS_TO_ARRIVE:
                    self.positions[robot_id] = self._goals[robot_id]
                    return True
                return False
            return super().poll_arrived(robot_id)

    driver = OneSlowRobotDriver(ticks_to_arrive=0)
    coordinator = LockstepCoordinator(env, driver, poll_interval_s=0.0)

    coordinator.run(num_ticks=10, policy_fn=None)

    slow_goal_count = sum(1 for rid, _ in driver.goal_history if rid == 0)
    fast_goal_counts = [
        sum(1 for rid, _ in driver.goal_history if rid == other)
        for other in env.robots
        if other != 0
    ]
    assert slow_goal_count <= 1
    assert all(count > slow_goal_count for count in fast_goal_counts)


def test_an_overdue_robot_is_reported_but_never_reassigned_mid_animation():
    env = _make_env()

    class DelayedArrivalDriver(FakeRobotDriver):
        SLOW_ROBOT_ID = 0
        SLOW_TICKS_TO_ARRIVE = 30

        def poll_arrived(self, robot_id: int) -> bool:
            if robot_id == self.SLOW_ROBOT_ID:
                self._polls_since_goal[robot_id] += 1
                if self._polls_since_goal[robot_id] >= self.SLOW_TICKS_TO_ARRIVE:
                    self.positions[robot_id] = self._goals[robot_id]
                    return True
                return False
            return super().poll_arrived(robot_id)

    driver = DelayedArrivalDriver(ticks_to_arrive=0)
    warnings = []
    coordinator = LockstepCoordinator(
        env, driver, poll_interval_s=0.0, arrival_timeout_s=0.0,
        on_tick=lambda tick, result: warnings.append((tick, result.arrived_in_time)),
    )

    coordinator.run(num_ticks=15, policy_fn=None)

    slow_goals = [cell for rid, cell in driver.goal_history if rid == 0]
    assert len(slow_goals) == 1
    overdue_reports = [w for w in warnings if not w[1]]
    assert 0 < len(overdue_reports) < 15


def test_arrival_timeout_is_reported_instead_of_hanging_forever():
    env = _make_env()

    class NeverArrivesDriver(FakeRobotDriver):
        def poll_arrived(self, robot_id: int) -> bool:
            return False

    driver = NeverArrivesDriver()
    coordinator = LockstepCoordinator(
        env, driver, poll_interval_s=0.0, arrival_timeout_s=0.05
    )

    stats = coordinator.run(num_ticks=3, policy_fn=None)

    assert len(stats.ticks) == 1
    assert stats.timed_out_ticks == 0


def test_status_reported_to_driver_matches_env_robot_status():
    env = _make_env()
    driver = FakeRobotDriver(ticks_to_arrive=0)
    coordinator = LockstepCoordinator(env, driver, poll_interval_s=0.0)

    coordinator.run(num_ticks=20, policy_fn=None)

    for robot_id, robot in env.robots.items():
        assert driver.statuses[robot_id] == STATUS_CODE[robot.status]


def test_status_starts_idle_at_reset():
    env = _make_env()
    driver = FakeRobotDriver(ticks_to_arrive=0)
    coordinator = LockstepCoordinator(env, driver)

    coordinator.reset(seed=0)

    assert all(status == STATUS_CODE[RobotStatus.IDLE] for status in driver.statuses.values())


def test_on_tick_callback_is_invoked_once_per_tick():
    env = _make_env()
    driver = FakeRobotDriver(ticks_to_arrive=0)
    seen = []
    coordinator = LockstepCoordinator(
        env, driver, poll_interval_s=0.0, on_tick=lambda tick, result: seen.append(tick)
    )

    coordinator.run(num_ticks=12, policy_fn=None)

    assert seen == list(range(12))


class _SlowPickupDriver(FakeRobotDriver):
    def __init__(self, slow_robot_id: int, ticks_to_arrive: int):
        super().__init__(ticks_to_arrive=0)
        self._slow_robot_id = slow_robot_id
        self._slow_ticks_to_arrive = ticks_to_arrive
        self._slow_polls = 0

    def send_goal(self, robot_id, cell):
        super().send_goal(robot_id, cell)
        if robot_id == self._slow_robot_id:
            self._slow_polls = 0

    def poll_arrived(self, robot_id):
        if robot_id != self._slow_robot_id:
            return super().poll_arrived(robot_id)
        self._slow_polls += 1
        if self._slow_polls >= self._slow_ticks_to_arrive:
            self.positions[robot_id] = self._goals[robot_id]
            return True
        return False


def test_picking_up_status_is_force_held_across_the_real_multi_tick_grab():
    from hiverl.env.task_manager import Order

    env = _make_env(num_robots=2, order_spawn_interval=1000, max_queue_size=1)
    pickup = env.map.pickup_stations[0]
    dropoff = env.map.dropoff_stations[0]

    real_reset = env.reset

    def _reset_with_robot0_mid_approach(seed=None):
        result = real_reset(seed=seed)
        robot = env.robots[0]
        robot.pos = pickup
        robot.status = RobotStatus.TO_ITEM
        robot.order = Order(order_id=0, pickup=pickup, dropoff=dropoff, created_step=0)
        robot.path = []
        return result

    env.reset = _reset_with_robot0_mid_approach

    driver = _SlowPickupDriver(slow_robot_id=0, ticks_to_arrive=4)
    coordinator = LockstepCoordinator(env, driver, poll_interval_s=0.0)

    status_at_step_call = []
    real_step = env.step

    def _spying_step(actions):
        status_at_step_call.append(env.robots[0].status)
        return real_step(actions)

    env.step = _spying_step

    coordinator.run(num_ticks=6, policy_fn=lambda obs: {name: 4 for name in obs})

    assert status_at_step_call[0] == RobotStatus.TO_ITEM
    assert status_at_step_call[1:4] == [RobotStatus.PICKING_UP] * 3
    assert status_at_step_call[4] == RobotStatus.CARRYING


class _PosedFakeDriver(FakeRobotDriver):
    def __init__(self, poses: dict[int, tuple[float, float, float]], **kwargs):
        super().__init__(**kwargs)
        self._poses = poses

    def get_pose(self, robot_id):
        return self._poses.get(robot_id)


def test_physical_proximity_override_forces_wait_regardless_of_grid_distance():
    env = _make_env(num_robots=3)
    poses = {0: (0.0, 0.0, 0.0), 1: (0.3, 0.0, 0.0), 2: (100.0, 100.0, 0.0)}
    driver = _PosedFakeDriver(poses, ticks_to_arrive=0)
    coordinator = LockstepCoordinator(env, driver, physical_proximity_stop_m=0.6)

    positions_before = {}

    def policy_fn(obs):
        positions_before.update({i: r.pos for i, r in env.robots.items()})
        return {name: 0 for name in obs}

    coordinator.run(num_ticks=1, policy_fn=policy_fn)

    assert coordinator.last_physical_proximity_events == {0, 1}
    assert env.robots[0].pos == positions_before[0]
    assert env.robots[1].pos == positions_before[1]


def _make_continuous_env(**overrides):
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
        max_episode_steps=10**9,
        deadlock_wait_threshold=1,
        seed=0,
    )
    kwargs.update(overrides)
    return ContinuousWarehouseEnv(**kwargs)


def _zero_continuous_policy(obs):
    import numpy as np

    return {a: np.zeros(2, dtype=np.float32) for a in obs}


def test_continuous_coordinator_rejects_a_discrete_env():
    env = _make_env()
    driver = FakeRobotDriver(ticks_to_arrive=0)
    coordinator = ContinuousLockstepCoordinator(env, driver)
    try:
        coordinator.run(1, _zero_continuous_policy)
        assert False, "expected an AssertionError for a non-continuous env"
    except AssertionError:
        pass


def test_continuous_coordinator_runs_full_tick_count_against_fake_driver():
    import numpy as np

    env = _make_continuous_env()
    driver = FakeRobotDriver(ticks_to_arrive=0)
    coordinator = ContinuousLockstepCoordinator(
        env, driver, arrival_timeout_s=5.0, synchronize_fleet=False, poll_interval_s=0.0
    )
    coordinator._sleep = lambda seconds: None

    rng = np.random.default_rng(0)

    def policy_fn(obs):
        return {a: rng.uniform(-1, 1, size=2).astype(np.float32) for a in obs}

    stats = coordinator.run(20, policy_fn)

    assert len(stats.ticks) == 20
    assert len(driver.goal_history) > 0, "expected at least one send_goal dispatch (IDLE robots)"
    assert len(driver.velocity_history) > 0, "expected at least one send_velocity dispatch (RL-controlled robots)"


def test_continuous_coordinator_resyncs_robot_pose_from_driver_after_each_tick():
    import numpy as np

    env = _make_continuous_env(num_robots=1)
    driver = FakeRobotDriver(ticks_to_arrive=0)
    coordinator = ContinuousLockstepCoordinator(
        env, driver, arrival_timeout_s=5.0, synchronize_fleet=False, poll_interval_s=0.0
    )
    coordinator._sleep = lambda seconds: None

    coordinator.run(1, lambda obs: {a: np.array([1.0, 0.0], dtype=np.float32) for a in obs})

    from hiverl.bridge.coords import world_pose_to_env

    driver_pose = driver.get_pose(0)
    assert driver_pose is not None
    expected = world_pose_to_env(*driver_pose)
    robot = env.robots[0]
    assert (robot.x, robot.y, robot.theta) == expected


def test_mutual_physical_stall_triggers_a_scripted_retreat_for_the_higher_id_robot():
    import numpy as np

    from hiverl.bridge.lockstep_coordinator import (
        PHYSICAL_STALL_TICKS_THRESHOLD,
    )
    from hiverl.env.robot import RobotStatus

    env = _make_continuous_env(num_robots=2)
    driver = FakeRobotDriver(ticks_to_arrive=0)
    coordinator = ContinuousLockstepCoordinator(
        env, driver, arrival_timeout_s=5.0, synchronize_fleet=False, poll_interval_s=0.0,
        physical_proximity_stop_m=0.6,
    )
    coordinator._sleep = lambda seconds: None
    coordinator.reset()

    for robot in env.robots.values():
        robot.status = RobotStatus.TO_ITEM
        robot.order = None
        robot.path = []
    env.robots[0].x, env.robots[0].y, env.robots[0].theta = 0.0, 0.0, 0.0
    env.robots[1].x, env.robots[1].y, env.robots[1].theta = 0.3, 0.0, 0.0
    driver.poses[0] = (0.0, 0.0, 0.0)
    driver.poses[1] = (0.3, 0.0, 0.0)

    def zero_policy(obs):
        return {a: np.array([0.0, 0.0], dtype=np.float32) for a in obs}

    for _ in range(PHYSICAL_STALL_TICKS_THRESHOLD):
        coordinator.run(1, zero_policy, reset=False)

    assert coordinator._physical_stall_retreat_elapsed.get(1, 0) > 0, (
        "the higher-id robot (1) should start a scripted retreat once the pair has been "
        "mutually stalled for PHYSICAL_STALL_TICKS_THRESHOLD ticks"
    )
    assert coordinator._physical_stall_retreat_elapsed.get(0, 0) == 0, (
        "the lower-id robot (0) should not retreat -- it's the one the pair yields to"
    )
    pose_before = driver.get_pose(1)
    coordinator.run(1, zero_policy, reset=False)
    pose_after = driver.get_pose(1)
    assert pose_after != pose_before, (
        "the yielding robot's physical pose should actually change during its scripted retreat, "
        "not just its logical/idealized one"
    )


def test_swept_physical_proximity_catches_a_fast_head_on_approach_a_snapshot_would_miss():
    import math

    import numpy as np

    from hiverl.env.robot import RobotStatus

    env = _make_continuous_env(num_robots=2)
    driver = FakeRobotDriver(ticks_to_arrive=0)
    coordinator = ContinuousLockstepCoordinator(
        env, driver, arrival_timeout_s=5.0, synchronize_fleet=False, poll_interval_s=0.0,
        physical_proximity_stop_m=0.6,
    )
    coordinator._sleep = lambda seconds: None
    coordinator.reset()

    for robot in env.robots.values():
        robot.status = RobotStatus.TO_ITEM
        robot.order = None
        robot.path = []
    driver.poses[0] = (0.0, 0.0, 0.0)
    driver.poses[1] = (1.5, 0.0, math.pi)

    def head_on_policy(obs):
        return {a: np.array([1.0, 0.0], dtype=np.float32) for a in obs}

    coordinator.run(1, head_on_policy, reset=False)

    assert coordinator.last_physical_proximity_events == {0, 1}, (
        "the swept prediction should have caught this closing pair on the first tick, before "
        "their current real distance (1.5m) ever came near physical_proximity_stop_m (0.6m)"
    )


def test_swept_physical_proximity_does_not_false_positive_on_robots_moving_apart():
    import math

    import numpy as np

    from hiverl.env.robot import RobotStatus

    env = _make_continuous_env(num_robots=2)
    driver = FakeRobotDriver(ticks_to_arrive=0)
    coordinator = ContinuousLockstepCoordinator(
        env, driver, arrival_timeout_s=5.0, synchronize_fleet=False, poll_interval_s=0.0,
        physical_proximity_stop_m=0.6,
    )
    coordinator._sleep = lambda seconds: None
    coordinator.reset()

    for robot in env.robots.values():
        robot.status = RobotStatus.TO_ITEM
        robot.order = None
        robot.path = []
    driver.poses[0] = (0.0, 0.0, math.pi)
    driver.poses[1] = (1.0, 0.0, 0.0)

    def away_policy(obs):
        return {a: np.array([1.0, 0.0], dtype=np.float32) for a in obs}

    coordinator.run(1, away_policy, reset=False)

    assert coordinator.last_physical_proximity_events == set()


@pytest.mark.parametrize("synchronize_fleet", [False, True])
def test_carrying_exit_refreshes_the_stale_pending_cell_before_set_status(synchronize_fleet):
    import numpy as np

    from hiverl.env.robot import RobotStatus
    from hiverl.env.task_manager import Order

    env = _make_continuous_env(num_robots=1)
    driver = FakeRobotDriver(ticks_to_arrive=0)
    coordinator = ContinuousLockstepCoordinator(
        env, driver, arrival_timeout_s=5.0, synchronize_fleet=synchronize_fleet, poll_interval_s=0.0
    )
    coordinator._sleep = lambda seconds: None
    coordinator.reset()

    from hiverl.env.continuous_kinematics import cell_to_world

    robot = env.robots[0]
    dropoff_cell = env.task_manager.dropoff_stations[0]
    robot.x, robot.y = cell_to_world(dropoff_cell)
    robot.sync_pos_from_continuous()
    driver.poses[0] = (robot.x, robot.y, robot.theta)
    stale_cell = env.task_manager.pickup_stations[0]
    driver.send_goal(0, stale_cell)
    robot.status = RobotStatus.CARRYING
    robot.order = Order(order_id=0, pickup=stale_cell, dropoff=dropoff_cell, created_step=0)
    robot.path = []

    coordinator.run(1, lambda obs: {a: np.array([0.0, 0.0], dtype=np.float32) for a in obs}, reset=False)

    assert robot.status != RobotStatus.CARRYING, "setup failed: delivery didn't happen this tick"
    assert driver.goal_history[-1] == (0, dropoff_cell), (
        f"expected the coordinator to refresh the driver's pending cell to the real dropoff cell "
        f"{dropoff_cell} right before set_status, not leave it at the stale {stale_cell} "
        f"-- got {driver.goal_history[-1]}"
    )


def test_physical_stall_yield_prefers_the_rl_controlled_robot_regardless_of_id():
    import numpy as np

    from hiverl.bridge.lockstep_coordinator import PHYSICAL_STALL_TICKS_THRESHOLD
    from hiverl.env.robot import RobotStatus
    from hiverl.env.task_manager import Order

    env = _make_continuous_env(num_robots=2)
    driver = FakeRobotDriver(ticks_to_arrive=0)
    coordinator = ContinuousLockstepCoordinator(
        env, driver, arrival_timeout_s=5.0, synchronize_fleet=False, poll_interval_s=0.0,
        physical_proximity_stop_m=0.6,
    )
    coordinator._sleep = lambda seconds: None
    coordinator.reset()

    env.robots[0].status = RobotStatus.TO_ITEM
    env.robots[0].order = None
    env.robots[0].path = []
    env.robots[1].status = RobotStatus.IDLE
    env.robots[1].order = None
    env.robots[1].path = []
    env.robots[0].x, env.robots[0].y, env.robots[0].theta = 0.0, 0.0, 0.0
    env.robots[1].x, env.robots[1].y, env.robots[1].theta = 0.3, 0.0, 0.0
    driver.poses[0] = (0.0, 0.0, 0.0)
    driver.poses[1] = (0.3, 0.0, 0.0)

    def zero_policy(obs):
        return {a: np.array([0.0, 0.0], dtype=np.float32) for a in obs}

    for _ in range(PHYSICAL_STALL_TICKS_THRESHOLD):
        env.robots[1].status = RobotStatus.IDLE
        env.robots[1].order = None
        env.robots[1].path = []
        coordinator.run(1, zero_policy, reset=False)

    assert coordinator._physical_stall_retreat_elapsed.get(0, 0) > 0, (
        "robot 0 is the only one of the pair actually capable of retreating (RL-controlled) -- "
        "it must be the one that yields even though it has the lower id"
    )


def test_physical_stall_retreat_flips_direction_when_reverse_moves_it_closer():
    import math

    import numpy as np

    from hiverl.bridge.lockstep_coordinator import (
        PHYSICAL_STALL_TICKS_THRESHOLD,
    )
    from hiverl.env.robot import RobotStatus

    env = _make_continuous_env(num_robots=2)
    driver = FakeRobotDriver(ticks_to_arrive=0)
    coordinator = ContinuousLockstepCoordinator(
        env, driver, arrival_timeout_s=5.0, synchronize_fleet=False, poll_interval_s=0.0,
        physical_proximity_stop_m=0.6,
    )
    coordinator._sleep = lambda seconds: None
    coordinator.reset()

    for robot in env.robots.values():
        robot.order = None
        robot.path = []
    env.robots[0].status = RobotStatus.CARRYING
    env.robots[1].status = RobotStatus.CARRYING
    env.robots[0].x, env.robots[0].y, env.robots[0].theta = 0.0, 0.0, 0.0
    env.robots[1].x, env.robots[1].y, env.robots[1].theta = 0.3, 0.0, 0.0
    driver.poses[0] = (0.0, 0.0, 0.0)
    driver.poses[1] = (0.3, 0.0, 0.0)

    def zero_policy(obs):
        return {a: np.array([0.0, 0.0], dtype=np.float32) for a in obs}

    for _ in range(PHYSICAL_STALL_TICKS_THRESHOLD):
        coordinator.run(1, zero_policy, reset=False)

    assert coordinator._physical_stall_retreat_sign.get(1) == -1.0, (
        "retreat should start by trying reverse (matches the original/wall-hugging-case default)"
    )
    dist_after_first_retreat_tick = math.hypot(
        driver.get_pose(1)[0] - driver.get_pose(0)[0], driver.get_pose(1)[1] - driver.get_pose(0)[1]
    )
    assert dist_after_first_retreat_tick < 0.3, (
        "setup check: reversing here must genuinely move robot 1 closer to robot 0, or this test "
        "isn't exercising the adversarial case it's meant to"
    )

    coordinator.run(1, zero_policy, reset=False)
    assert coordinator._physical_stall_retreat_sign.get(1) == -1.0, (
        "must not flip after a single non-improving tick -- requires "
        "PHYSICAL_STALL_SIGN_FLIP_PATIENCE_TICKS consecutive ones"
    )

    for _ in range(6):
        coordinator.run(1, zero_policy, reset=False)
    final_dist = math.hypot(
        driver.get_pose(1)[0] - driver.get_pose(0)[0], driver.get_pose(1)[1] - driver.get_pose(0)[1]
    )
    assert final_dist >= 0.6, (
        f"expected real separation to eventually clear physical_proximity_stop_m (0.6m) -- "
        f"got {final_dist:.3f}m"
    )


def test_pickup_in_progress_does_not_freeze_velocity_mode_robots():
    import numpy as np

    from hiverl.env.task_manager import Order

    class _PermanentlyBusyDriver(FakeRobotDriver):
        def __init__(self, stuck_robot_id: int):
            super().__init__(ticks_to_arrive=0)
            self._stuck_robot_id = stuck_robot_id

        def poll_arrived(self, robot_id: int) -> bool:
            if robot_id == self._stuck_robot_id:
                return False
            return super().poll_arrived(robot_id)

    env = _make_continuous_env(num_robots=2, order_spawn_interval=1000, max_queue_size=1)
    pickup = env.map.pickup_stations[0]
    dropoff = env.map.dropoff_stations[0]

    real_reset = env.reset

    def _reset_with_fixed_roles(seed=None):
        result = real_reset(seed=seed)
        robot0 = env.robots[0]
        robot0.status = RobotStatus.TO_ITEM
        robot0.order = Order(order_id=0, pickup=pickup, dropoff=dropoff, created_step=0)
        robot0.path = [pickup]

        robot1 = env.robots[1]
        robot1.status = RobotStatus.IDLE
        robot1.path = [robot1.pos]
        return result

    env.reset = _reset_with_fixed_roles

    driver = _PermanentlyBusyDriver(stuck_robot_id=1)
    coordinator = ContinuousLockstepCoordinator(
        env, driver, arrival_timeout_s=0.05, synchronize_fleet=True, poll_interval_s=0.0,
    )
    coordinator._sleep = lambda seconds: None

    policy_calls = 0

    def policy_fn(obs):
        nonlocal policy_calls
        policy_calls += 1
        return {a: np.zeros(2, dtype=np.float32) for a in obs}

    stats = coordinator.run(5, policy_fn)

    assert len(stats.ticks) == 5
    assert policy_calls == 5, (
        "the velocity-mode robot must get a fresh action every tick, even while another robot's "
        "goal-cell dispatch is stuck busy -- got only {} policy_fn calls".format(policy_calls)
    )


class _LidarStalledDriver(FakeRobotDriver):
    def send_velocity(self, robot_id: int, v: float, omega: float) -> None:
        if v > 0.0:
            return
        super().send_velocity(robot_id, v, omega)


def test_lidar_stall_retreat_overrides_action_after_sustained_freeze():
    import numpy as np

    from hiverl.env.task_manager import Order

    env = _make_continuous_env(num_robots=1, order_spawn_interval=1000, max_queue_size=1)
    pickup = env.map.pickup_stations[0]
    dropoff = env.map.dropoff_stations[0]

    real_reset = env.reset

    def _reset_with_robot0_to_item(seed=None):
        result = real_reset(seed=seed)
        robot0 = env.robots[0]
        robot0.status = RobotStatus.TO_ITEM
        robot0.order = Order(order_id=0, pickup=pickup, dropoff=dropoff, created_step=0)
        robot0.path = [pickup]
        return result

    env.reset = _reset_with_robot0_to_item

    driver = _LidarStalledDriver()
    coordinator = ContinuousLockstepCoordinator(
        env, driver, arrival_timeout_s=0.05, synchronize_fleet=True, poll_interval_s=0.0,
    )
    coordinator._sleep = lambda seconds: None

    def policy_fn(obs):
        return {a: np.array([1.0, 0.0], dtype=np.float32) for a in obs}

    num_ticks = 25
    stats = coordinator.run(num_ticks, policy_fn)

    assert len(stats.ticks) == num_ticks
    negative_sent = [v for (_rid, v, _omega) in driver.velocity_history if v < 0.0]
    assert negative_sent, (
        "expected the coordinator to override the policy's forward-only command with a reverse "
        "retreat at some point after LIDAR_STALL_RETREAT_TRIGGER_TICKS consecutive stalled ticks, "
        "but every velocity actually sent to the driver was >= 0"
    )
    assert env.robots[0].x != 0.0 or env.robots[0].y != 0.0, (
        "the reverse retreat should have actually moved the robot (this fake driver only "
        "integrates motion for v < 0), confirming escape is possible, not just attempted"
    )


def test_anti_pileup_circuit_breaker_ends_the_round_early():
    import numpy as np

    from hiverl.bridge.lockstep_coordinator import PILEUP_HARD_BREAK_TICKS

    env = _make_continuous_env(num_robots=2)
    poses = {0: (0.0, 0.0, 0.0), 1: (0.1, 0.0, 0.0)}
    driver = _PosedFakeDriver(poses, ticks_to_arrive=0)
    coordinator = ContinuousLockstepCoordinator(
        env, driver, arrival_timeout_s=1000.0, synchronize_fleet=True, poll_interval_s=0.0,
    )
    coordinator._sleep = lambda seconds: None

    def policy_fn(obs):
        return {a: np.zeros(2, dtype=np.float32) for a in obs}

    requested_ticks = PILEUP_HARD_BREAK_TICKS + 20
    stats = coordinator.run(requested_ticks, policy_fn)

    assert len(stats.ticks) < requested_ticks, (
        f"expected the anti-pileup breaker to end the round early (2 robots stayed "
        f"{poses[1][0] - poses[0][0]:.2f}m apart, under PILEUP_DISTANCE_M, for the whole run) -- "
        f"got the full {len(stats.ticks)} requested ticks instead"
    )


def test_global_collision_reset_ends_round_at_real_contact_distance():
    import numpy as np

    env = _make_continuous_env(num_robots=2)
    poses = {0: (0.0, 0.0, 0.0), 1: (0.3, 0.0, 0.0)}
    driver = _PosedFakeDriver(poses, ticks_to_arrive=0)
    coordinator = ContinuousLockstepCoordinator(
        env,
        driver,
        arrival_timeout_s=1000.0,
        synchronize_fleet=True,
        poll_interval_s=0.0,
        global_collision_reset_m=0.40,
    )
    coordinator._sleep = lambda seconds: None
    coordinator.reset()
    for robot in env.robots.values():
        robot.status = RobotStatus.TO_ITEM
        robot.order = None
        robot.path = []
    driver._poses = poses

    policy_calls = 0

    def policy_fn(obs):
        nonlocal policy_calls
        policy_calls += 1
        return {agent: np.zeros(2, dtype=np.float32) for agent in obs}

    stats = coordinator.run(20, policy_fn, reset=False)

    assert len(stats.ticks) == 0
    assert policy_calls == 1


class _CachedPoseDriver(FakeRobotDriver):
    def __init__(self):
        super().__init__()
        self._incoming: dict[int, tuple[float, float, float]] = {}

    def send_velocity(self, robot_id: int, v: float, omega: float) -> None:
        cached = self.poses[robot_id]
        super().send_velocity(robot_id, v, omega)
        self._incoming[robot_id] = self.poses[robot_id]
        self.poses[robot_id] = cached

    def spin(self) -> None:
        self.poses.update(self._incoming)
        self._incoming.clear()


def test_post_tick_pose_is_read_after_draining_the_driver_queue():
    import numpy as np

    from hiverl.env.task_manager import Order

    env = _make_continuous_env(num_robots=1, order_spawn_interval=1000, max_queue_size=1)
    pickup, dropoff = env.map.pickup_stations[0], env.map.dropoff_stations[0]
    real_reset = env.reset

    def _reset_with_robot0_to_item(seed=None):
        result = real_reset(seed=seed)
        env.robots[0].status = RobotStatus.TO_ITEM
        env.robots[0].order = Order(order_id=0, pickup=pickup, dropoff=dropoff, created_step=0)
        env.robots[0].path = [pickup]
        return result

    env.reset = _reset_with_robot0_to_item
    driver = _CachedPoseDriver()
    coordinator = ContinuousLockstepCoordinator(
        env, driver, arrival_timeout_s=0.05, synchronize_fleet=True, poll_interval_s=0.0,
        enable_legacy_retreats=False,
    )
    coordinator._sleep = lambda seconds: None
    stall_ticks = []
    coordinator.on_tick = lambda tick, result: stall_ticks.append(len(coordinator.last_lidar_stall_events))

    coordinator.run(5, lambda obs: {a: np.array([0.5, 0.0], dtype=np.float32) for a in obs})

    assert sum(stall_ticks) == 0, f"moving robot flagged as lidar-stalled: {stall_ticks}"


def test_collision_respawns_only_the_colliding_robots_and_the_round_keeps_going():
    import math

    import numpy as np

    from hiverl.env.task_manager import Order
    from hiverl.env.warehouse_env_continuous import REWARD_PHYSICAL_COLLISION_CONTINUOUS

    env = _make_continuous_env(num_robots=3, order_spawn_interval=1000, max_queue_size=5)
    driver = FakeRobotDriver(ticks_to_arrive=0)
    coordinator = ContinuousLockstepCoordinator(
        env, driver, arrival_timeout_s=1000.0, synchronize_fleet=True, poll_interval_s=0.0,
        global_collision_reset_m=0.40, respawn_on_collision=True, enable_legacy_retreats=False,
    )
    coordinator._sleep = lambda seconds: None
    coordinator.reset()
    pickup, dropoff = env.map.pickup_stations[0], env.map.dropoff_stations[0]
    order = Order(order_id=99, pickup=pickup, dropoff=dropoff, created_step=0)
    env.robots[0].status, env.robots[0].order, env.robots[0].path = RobotStatus.CARRYING, order, [dropoff]
    x0, y0, _ = driver.poses[0]
    driver.poses[1] = (x0 + 0.3, y0, 0.0)

    per_tick = []
    coordinator.on_tick = lambda tick, result: per_tick.append(
        (dict(coordinator.last_terminations), dict(coordinator.last_rewards))
    )
    stats = coordinator.run(10, lambda obs: {a: np.zeros(2, dtype=np.float32) for a in obs}, reset=False)

    assert len(stats.ticks) == 10, "round must not end early on a collision"
    first_terms, first_rewards = per_tick[0]
    assert first_terms["robot_0"] and first_terms["robot_1"] and not first_terms["robot_2"]
    assert first_rewards["robot_0"] <= REWARD_PHYSICAL_COLLISION_CONTINUOUS + 1.0
    assert first_rewards["robot_1"] <= REWARD_PHYSICAL_COLLISION_CONTINUOUS + 1.0
    assert not any(terms["robot_2"] for terms, _ in per_tick)
    assert sum(terms["robot_0"] for terms, _ in per_tick) == 1, "penalized once, not every tick"
    (xa, ya, _), (xb, yb, _) = driver.poses[0], driver.poses[1]
    assert math.hypot(xa - xb, ya - yb) >= 1.0, "colliding robots must be teleported apart"
    assert order in env.task_manager.pending_orders or env.robots[0].order is order or any(
        r.order is order for r in env.robots.values()
    ), "the carried order must go back to the queue (or be re-assigned), never dropped"
    assert not env.hold_robot_ids, "hold must be released once the teleport is acknowledged"


def test_collision_without_respawn_still_ends_the_round():
    import numpy as np

    env = _make_continuous_env(num_robots=2)
    driver = FakeRobotDriver(ticks_to_arrive=0)
    coordinator = ContinuousLockstepCoordinator(
        env, driver, arrival_timeout_s=1000.0, synchronize_fleet=True, poll_interval_s=0.0,
        global_collision_reset_m=0.40, respawn_on_collision=False,
    )
    coordinator._sleep = lambda seconds: None
    coordinator.reset()
    x0, y0, _ = driver.poses[0]
    driver.poses[1] = (x0 + 0.3, y0, 0.0)
    stats = coordinator.run(10, lambda obs: {a: np.zeros(2, dtype=np.float32) for a in obs}, reset=False)
    assert len(stats.ticks) == 0


def test_continuous_round_does_not_wait_for_a_busy_goal_cell_robot_at_the_end():
    import time as _time

    import numpy as np

    class _NeverArrivesDriver(FakeRobotDriver):
        def poll_arrived(self, robot_id: int) -> bool:
            return False

    from hiverl.env.task_manager import Order

    env = _make_continuous_env(num_robots=2, order_spawn_interval=1000, max_queue_size=1)
    driver = _NeverArrivesDriver()
    coordinator = ContinuousLockstepCoordinator(
        env, driver, arrival_timeout_s=30.0, synchronize_fleet=True, poll_interval_s=0.01,
    )
    coordinator._sleep = lambda seconds: None
    coordinator.reset()
    pickup, dropoff = env.map.pickup_stations[0], env.map.dropoff_stations[0]
    env.robots[1].status = RobotStatus.TO_ITEM
    env.robots[1].order = Order(order_id=0, pickup=pickup, dropoff=dropoff, created_step=0)
    env.robots[1].path = [pickup]
    coordinator._busy = {0: True, 1: False}
    started = _time.monotonic()
    stats = coordinator.run(3, lambda obs: {a: np.zeros(2, dtype=np.float32) for a in obs}, reset=False)
    assert len(stats.ticks) == 3
    assert _time.monotonic() - started < 10.0, "must not wait arrival_timeout_s at the end of the round"
