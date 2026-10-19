"""Unicycle integration and grid rounding."""
import math

from hiverl.env.continuous_kinematics import integrate_unicycle


def test_integrate_unicycle_matches_the_exact_arc_not_forward_euler():
    v, omega, dt = 1.0, math.pi / 2, 1.0
    x, y, theta = integrate_unicycle(0.0, 0.0, 0.0, v, omega, dt)
    radius = v / omega
    assert math.isclose(x, radius, rel_tol=1e-9)
    assert math.isclose(y, radius, rel_tol=1e-9)
    assert math.isclose(theta, math.pi / 2, rel_tol=1e-9)
    assert not math.isclose(x, 1.0, rel_tol=1e-3)
    assert not math.isclose(y, 0.0, abs_tol=1e-3)


def test_integrate_unicycle_zero_omega_is_a_straight_line():
    x, y, theta = integrate_unicycle(1.0, 2.0, math.pi / 4, v=2.0, omega=0.0, dt=1.0)
    assert math.isclose(x, 1.0 + 2.0 * math.cos(math.pi / 4))
    assert math.isclose(y, 2.0 + 2.0 * math.sin(math.pi / 4))
    assert math.isclose(theta, math.pi / 4)


def test_integrate_unicycle_full_circle_returns_to_start():
    v, omega = 1.0, 1.0
    dt = 2 * math.pi / omega
    x, y, theta = integrate_unicycle(3.0, -1.0, 0.5, v, omega, dt)
    assert math.isclose(x, 3.0, abs_tol=1e-9)
    assert math.isclose(y, -1.0, abs_tol=1e-9)
