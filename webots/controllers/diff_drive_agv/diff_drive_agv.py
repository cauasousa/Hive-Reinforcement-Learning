"""Webots controller for one warehouse-AGV robot, bridged to ROS2.

Run *by Webots itself* (one process per robot, spawned automatically per the world file's
``controller "diff_drive_agv"`` field) -- not meant to be launched by hand. The robot body itself
is authored directly in ``webots_world_generator.py`` (a plain ``Robot``/``HingeJoint``/``Solid``
tree, not an external PROTO -- see docs/webots_ros2_integration.md section 8), which is what lets
every device name and geometry constant here be *guaranteed* consistent with the world file rather
than guessed against an unfamiliar PROTO's internals.

Each instance:

  1. Reads its own ``--robot-id``/``--start-row``/``--start-col`` from ``controllerArgs``.
  2. Owns the actual differential-drive control loop: dead-reckoning odometry from the wheel
     position sensors, and simple proportional turn-then-drive control toward a target grid cell
     -- this is exactly idea.md's "Low-Level (abstração de controle)" layer, kept out of the
     ROS2 coordinator entirely, which only ever thinks in grid cells.
  3. Subscribes to ``/hiverl/robot_<id>/command`` (``[row, col, status]`` -- the next logical
     cell the Python-side ``WarehouseEnv`` has already committed the robot to, plus its
     0=IDLE/1=TO_ITEM/2=PICKING_UP/3=CARRYING status, published together as one atomic message -- see
     ``hiverl.bridge.ros2_driver``'s module docstring for why goal and status can't be separate
     topics) and publishes ``/hiverl/robot_<id>/arrived`` (True once within tolerance and
     stopped) plus ``/hiverl/robot_<id>/pose`` for monitoring.
  4. Lights the chassis's two status LEDs from that status: both off while idle,
     ``led_to_item`` (blue) while heading to a pickup, ``led_carrying`` (green) while carrying a
     package -- the same grey/blue/green convention ``scripts/visualize_episode.py`` already
     uses for the 2D GIF renderer.

Caveats (see docs/webots_ros2_integration.md "Known gaps"): the wheel radius/axle-length
constants below must match ``webots_world_generator.py``'s ``WHEEL_RADIUS_M``/``HALF_AXLE_M``
exactly (duplicated, not shared -- this controller runs in Webots' own bundled Python
interpreter and does not import the ``hiverl`` package). This entire robot body (chassis, wheel
joints, caster, LEDs) is new and has **not** been run inside a real Webots session -- unlike the
earlier e-puck-based controller, whose geometry (though not this exact device/control logic) was
at least confirmed to render. Expect to retune ``KP_HEADING``/``BASE_SPEED`` and possibly the
joint anchor points once this runs against real physics.
"""
from __future__ import annotations

import argparse
import math
import sys

from controller import Robot

TIME_STEP_MS = 32

WHEEL_RADIUS_M = 0.09
AXLE_LENGTH_M = 0.42
MAX_WHEEL_SPEED = 10.0
BASE_SPEED = 0.6 * MAX_WHEEL_SPEED
KP_HEADING = 4.0
ARRIVAL_TOLERANCE_M = 0.08
TURN_IN_PLACE_THRESHOLD_RAD = 0.35

STATUS_IDLE, STATUS_TO_ITEM, STATUS_PICKING_UP, STATUS_CARRYING = 0, 1, 2, 3


def _normalize_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot-id", type=int, required=True)
    parser.add_argument("--start-row", type=int, required=True)
    parser.add_argument("--start-col", type=int, required=True)
    args, _unknown = parser.parse_known_args(sys.argv[1:])
    return args


class DiffDriveOdometry:
    """Dead-reckoning pose (x, y, theta) integrated from wheel-encoder deltas."""

    def __init__(self, start_x: float, start_y: float, start_theta: float = 0.0):
        self.x = start_x
        self.y = start_y
        self.theta = start_theta
        self._last_left = None
        self._last_right = None

    def update(self, left_pos: float, right_pos: float) -> None:
        if self._last_left is None:
            self._last_left, self._last_right = left_pos, right_pos
            return
        d_left = (left_pos - self._last_left) * WHEEL_RADIUS_M
        d_right = (right_pos - self._last_right) * WHEEL_RADIUS_M
        self._last_left, self._last_right = left_pos, right_pos

        d_center = (d_left + d_right) / 2.0
        d_theta = (d_right - d_left) / AXLE_LENGTH_M

        self.x += d_center * math.cos(self.theta + d_theta / 2.0)
        self.y += d_center * math.sin(self.theta + d_theta / 2.0)
        self.theta = _normalize_angle(self.theta + d_theta)


def _cell_to_world(cell: tuple[int, int]) -> tuple[float, float]:
    row, col = cell
    return (col + 0.5) * 1.0, -(row + 0.5) * 1.0


def main() -> None:
    args = _parse_args()
    robot_id = args.robot_id

    robot = Robot()
    left_motor = robot.getDevice("left wheel motor")
    right_motor = robot.getDevice("right wheel motor")
    left_motor.setPosition(float("inf"))
    right_motor.setPosition(float("inf"))
    left_motor.setVelocity(0.0)
    right_motor.setVelocity(0.0)

    left_sensor = robot.getDevice("left wheel sensor")
    right_sensor = robot.getDevice("right wheel sensor")
    left_sensor.enable(TIME_STEP_MS)
    right_sensor.enable(TIME_STEP_MS)

    led_to_item = robot.getDevice("led_to_item")
    led_carrying = robot.getDevice("led_carrying")
    status = [STATUS_IDLE]

    def _apply_led_pattern() -> None:
        led_to_item.set(1 if status[0] == STATUS_TO_ITEM else 0)
        led_carrying.set(1 if status[0] == STATUS_CARRYING else 0)

    start_x, start_y = _cell_to_world((args.start_row, args.start_col))
    odom = DiffDriveOdometry(start_x, start_y)

    goal_cell = [args.start_row, args.start_col]

    import rclpy
    from geometry_msgs.msg import Pose2D
    from std_msgs.msg import Bool, Int32MultiArray

    rclpy.init(args=None)
    node = rclpy.create_node(f"robot_{robot_id}_controller")
    topic_prefix = f"/hiverl/robot_{robot_id}"

    def _on_command(msg: Int32MultiArray) -> None:
        goal_cell[0], goal_cell[1] = int(msg.data[0]), int(msg.data[1])
        status[0] = int(msg.data[2])
        _apply_led_pattern()

    node.create_subscription(Int32MultiArray, f"{topic_prefix}/command", _on_command, 10)
    arrived_pub = node.create_publisher(Bool, f"{topic_prefix}/arrived", 10)
    pose_pub = node.create_publisher(Pose2D, f"{topic_prefix}/pose", 10)
    _apply_led_pattern()

    while robot.step(TIME_STEP_MS) != -1:
        rclpy.spin_once(node, timeout_sec=0.0)

        odom.update(left_sensor.getValue(), right_sensor.getValue())

        target_x, target_y = _cell_to_world(tuple(goal_cell))
        dx, dy = target_x - odom.x, target_y - odom.y
        dist = math.hypot(dx, dy)

        arrived = dist <= ARRIVAL_TOLERANCE_M
        if arrived:
            left_motor.setVelocity(0.0)
            right_motor.setVelocity(0.0)
        else:
            heading_error = _normalize_angle(math.atan2(dy, dx) - odom.theta)
            if abs(heading_error) > TURN_IN_PLACE_THRESHOLD_RAD:
                turn = KP_HEADING * heading_error
                left_speed = -turn
                right_speed = turn
            else:
                turn = KP_HEADING * heading_error
                left_speed = BASE_SPEED - turn
                right_speed = BASE_SPEED + turn
            clamp = lambda v: max(-MAX_WHEEL_SPEED, min(MAX_WHEEL_SPEED, v))
            left_motor.setVelocity(clamp(left_speed))
            right_motor.setVelocity(clamp(right_speed))

        arrived_pub.publish(Bool(data=arrived))
        pose_pub.publish(Pose2D(x=odom.x, y=odom.y, theta=odom.theta))

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
