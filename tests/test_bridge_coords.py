"""Grid <-> Webots coordinate conversion."""
import math

from hiverl.bridge.coords import CELL_SIZE_M, arrived_at, cell_to_world, env_pose_to_world, world_pose_to_env, world_to_cell


def test_cell_to_world_is_cell_center():
    x, y = cell_to_world((0, 0))
    assert x == CELL_SIZE_M / 2
    assert y == -CELL_SIZE_M / 2


def test_world_to_cell_round_trips_through_cell_center():
    for cell in [(0, 0), (3, 4), (12, 16)]:
        x, y = cell_to_world(cell)
        assert world_to_cell(x, y) == cell


def test_arrived_at_true_within_tolerance_false_outside():
    x, y = cell_to_world((5, 5))
    assert arrived_at(x + 0.05, y - 0.05, (5, 5), tolerance_m=0.08)
    assert not arrived_at(x + 0.5, y, (5, 5), tolerance_m=0.08)


def test_env_pose_to_world_round_trips():
    for x, y, theta in [(0.0, 0.0, 0.0), (3.2, 4.7, 1.1), (-1.0, 5.0, -2.5)]:
        wx, wy, wtheta = env_pose_to_world(x, y, theta)
        ex, ey, etheta = world_pose_to_env(wx, wy, wtheta)
        assert math.isclose(ex, x, abs_tol=1e-9)
        assert math.isclose(ey, y, abs_tol=1e-9)
        assert math.isclose(etheta, theta, abs_tol=1e-9)


def test_env_pose_to_world_matches_cell_to_world_at_cell_centers():
    for cell in [(0, 0), (2, 3), (5, 1)]:
        row, col = cell
        wx, wy, _ = env_pose_to_world(float(col), float(row), 0.0)
        assert (wx, wy) == cell_to_world(cell)


def test_env_pose_to_world_negates_heading():
    _, _, wtheta = env_pose_to_world(0.0, 0.0, 0.7)
    assert wtheta == -0.7
