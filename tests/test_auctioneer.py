"""Order auction (nearest idle robot, fairness, claimed pickups)."""
import numpy as np

from hiverl.env.task_manager import Order, TaskManager
from hiverl.planners.auctioneer import run_auction


def _task_manager_with_order(order: Order) -> TaskManager:
    tm = TaskManager(
        pickup_stations=[order.pickup],
        dropoff_stations=[order.dropoff],
        order_spawn_interval=1000,
        max_queue_size=10,
        rng=np.random.default_rng(0),
    )
    tm.pending_orders.append(order)
    return tm


def test_nearest_idle_robot_wins_the_auction():
    order = Order(order_id=0, pickup=(5, 5), dropoff=(0, 0), created_step=0)
    tm = _task_manager_with_order(order)
    positions = {0: (0, 0), 1: (5, 6), 2: (9, 9)}
    assignments = run_auction(tm, idle_robot_ids=[0, 1, 2], robot_positions=positions)
    assert assignments == {1: order}
    assert order not in tm.pending_orders


def test_auction_assigns_at_most_one_order_per_robot():
    o1 = Order(0, pickup=(0, 0), dropoff=(1, 1), created_step=0)
    o2 = Order(1, pickup=(0, 1), dropoff=(1, 1), created_step=0)
    tm = TaskManager([(0, 0), (0, 1)], [(1, 1)], 1000, 10, np.random.default_rng(0))
    tm.pending_orders.extend([o1, o2])
    positions = {0: (0, 0), 1: (10, 10)}
    assignments = run_auction(tm, [0, 1], positions)
    assert len(assignments) == 2
    assert set(assignments.keys()) == {0, 1}


def test_auction_never_double_books_two_pending_orders_at_the_same_pickup():
    o1 = Order(0, pickup=(5, 5), dropoff=(0, 0), created_step=0)
    o2 = Order(1, pickup=(5, 5), dropoff=(0, 0), created_step=0)
    tm = TaskManager([(5, 5)], [(0, 0)], 1000, 10, np.random.default_rng(0))
    tm.pending_orders.extend([o1, o2])
    positions = {0: (0, 0), 1: (9, 9)}
    assignments = run_auction(tm, [0, 1], positions)
    assert len(assignments) == 1
    assert o2 in tm.pending_orders


def test_auction_skips_orders_at_an_already_claimed_pickup():
    order = Order(0, pickup=(5, 5), dropoff=(0, 0), created_step=0)
    tm = _task_manager_with_order(order)
    positions = {0: (0, 0)}
    assignments = run_auction(tm, [0], positions, claimed_pickups=frozenset({(5, 5)}))
    assert assignments == {}
    assert order in tm.pending_orders


def test_full_queue_still_spawns_an_order_for_a_station_with_none_pending():
    stations = [(1, 0), (6, 0), (11, 0)]
    tm = TaskManager(stations, [(1, 16)], order_spawn_interval=1, max_queue_size=4, rng=np.random.default_rng(0))
    for k in range(4):
        tm.pending_orders.append(Order(order_id=k, pickup=stations[k % 2], dropoff=(1, 16), created_step=0))
    tm.step(1)
    assert len(tm.pending_orders) == 5
    assert tm.pending_orders[-1].pickup == (11, 0)
    tm.step(2)
    assert len(tm.pending_orders) == 5


def test_new_orders_go_to_the_pickup_station_with_fewest_pending_orders():
    stations = [(1, 0), (6, 0), (11, 0)]
    tm = TaskManager(stations, [(1, 16)], order_spawn_interval=1, max_queue_size=100, rng=np.random.default_rng(0))
    for step in range(1, 31):
        tm.step(step)
    counts = [sum(o.pickup == s for o in tm.pending_orders) for s in stations]
    assert counts == [10, 10, 10]
