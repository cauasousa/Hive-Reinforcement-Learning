"""RobotDriver over ROS2 for the Webots youBot controllers.

Per robot it publishes command / cmd_vel / reset and subscribes to arrived / pose. Each driver can
use its own ROS_DOMAIN_ID (one per Webots world).
"""
from __future__ import annotations

from hiverl.bridge.robot_driver import RobotDriver
from hiverl.bridge.coords import arrived_at

_ROS2_IMPORT_ERROR = (
    "hiverl.bridge.ros2_driver requires a sourced ROS2 environment (rclpy, std_msgs, "
    "geometry_msgs). See docs/run_with_webots_ros2.md for setup -- this module is not usable "
    "on a plain pip install."
)

# Max spin_once calls per spin() to drain queued messages.
_SPIN_DRAIN_ITERATIONS = 128


class Ros2RobotDriver(RobotDriver):
    """ROS2 publishers/subscribers for num_robots robots, on a private context and executor."""
    def __init__(self, num_robots: int, node_name: str = "hiverl_coordinator", domain_id: int | None = None):
        try:
            import rclpy
            from geometry_msgs.msg import Pose2D, Twist
            from rclpy.node import Node
            from std_msgs.msg import Bool, Int32MultiArray
        except ImportError as exc:
            raise ImportError(_ROS2_IMPORT_ERROR) from exc

        self._rclpy = rclpy
        self._Int32MultiArray = Int32MultiArray
        self._Twist = Twist

        self._context = None
        self._executor = None
        if domain_id is None:
            if not rclpy.ok():
                rclpy.init()
            self.node: Node = rclpy.create_node(node_name)
        else:
            from rclpy.executors import SingleThreadedExecutor

            self._context = rclpy.Context()
            rclpy.init(context=self._context, domain_id=int(domain_id))
            self.node = rclpy.create_node(node_name, context=self._context)
            self._executor = SingleThreadedExecutor(context=self._context)
            self._executor.add_node(self.node)

        self._arrived: dict[int, bool] = {i: True for i in range(num_robots)}
        self._poses: dict[int, tuple[float, float, float]] = {}
        self._shield_blocked: dict[int, bool] = {}
        self._pending_cell: dict[int, tuple[int, int]] = {}
        self._pending_reset: dict[int, bool] = {i: False for i in range(num_robots)}
        self._reset_poll_count: dict[int, int] = {i: 0 for i in range(num_robots)}
        self._status: dict[int, int] = {i: 0 for i in range(num_robots)}
        self._command_publishers = {}
        self._cmd_vel_publishers = {}
        for robot_id in range(num_robots):
            topic_prefix = f"/hiverl/robot_{robot_id}"
            self._command_publishers[robot_id] = self.node.create_publisher(
                Int32MultiArray, f"{topic_prefix}/command", 10
            )
            self._cmd_vel_publishers[robot_id] = self.node.create_publisher(
                Twist, f"{topic_prefix}/cmd_vel", 10
            )

            def _make_cb(rid: int):
                def _cb(msg: Bool):
                    self._arrived[rid] = bool(msg.data)

                return _cb

            self.node.create_subscription(Bool, f"{topic_prefix}/arrived", _make_cb(robot_id), 10)

            def _make_pose_cb(rid: int):
                def _cb(msg: Pose2D):
                    self._poses[rid] = (msg.x, msg.y, msg.theta)

                return _cb

            self.node.create_subscription(Pose2D, f"{topic_prefix}/pose", _make_pose_cb(robot_id), 10)

    def reset(self, robot_id: int, cell: tuple[int, int]) -> None:
        self._arrived[robot_id] = False
        row, col = int(cell[0]), int(cell[1])
        self._pending_cell[robot_id] = (row, col)
        self._pending_reset[robot_id] = True
        self._reset_poll_count[robot_id] = 0
        self._publish_reset(robot_id)

    def _publish_reset(self, robot_id: int) -> None:
        """Publish the reset (teleport) command for the pending cell."""
        row, col = self._pending_cell[robot_id]
        msg = self._Int32MultiArray()
        msg.data = [row, col, -1]
        self._command_publishers[robot_id].publish(msg)

    def send_goal(self, robot_id: int, cell: tuple[int, int]) -> None:
        self._arrived[robot_id] = False
        self._pending_cell[robot_id] = (int(cell[0]), int(cell[1]))

    def spin(self) -> None:
        for _ in range(_SPIN_DRAIN_ITERATIONS):
            if self._executor is not None:
                self._executor.spin_once(timeout_sec=0.0)
            else:
                self._rclpy.spin_once(self.node, timeout_sec=0.0)

    def poll_arrived(self, robot_id: int) -> bool:
        self.spin()
        pose = self._poses.get(robot_id)
        cell = self._pending_cell.get(robot_id)
        pickup_complete = self._status[robot_id] == 2 and self._arrived[robot_id]
        arrived = bool(
            pickup_complete
            or (
                self._arrived[robot_id]
                and pose is not None
                and cell is not None
                and arrived_at(pose[0], pose[1], cell, tolerance_m=0.30)
            )
        )
        if arrived:
            self._pending_reset[robot_id] = False
            return True

        if self._pending_reset[robot_id]:
            self._reset_poll_count[robot_id] += 1
            if self._reset_poll_count[robot_id] % 10 == 0:
                self._publish_reset(robot_id)
        return False

    def set_shield_blocked(self, robot_id: int, blocked: bool) -> None:
        self._shield_blocked[robot_id] = bool(blocked)

    def send_velocity(self, robot_id: int, v: float, omega: float) -> None:
        """Publish (v, omega) in the env frame; the controller converts to the world frame.
        linear.z = 1 tells the controller the shield blocked this robot (triggers its yield nudge)."""
        msg = self._Twist()
        msg.linear.x = float(v)
        msg.linear.z = 1.0 if self._shield_blocked.get(robot_id, False) else 0.0
        msg.angular.z = float(omega)
        self._cmd_vel_publishers[robot_id].publish(msg)

    def get_pose(self, robot_id: int) -> tuple[float, float, float] | None:
        return self._poses.get(robot_id)

    def arrival_diagnostic(self, robot_id: int) -> str:
        pose = self._poses.get(robot_id)
        cell = self._pending_cell.get(robot_id)
        if pose is None:
            pose_text = "none"
        else:
            pose_text = f"({pose[0]:.2f},{pose[1]:.2f})"
        return f"arrived_flag={self._arrived[robot_id]} pose={pose_text} expected_cell={cell}"

    def set_status(self, robot_id: int, status: int) -> None:
        self._status[robot_id] = int(status)
        row, col = self._pending_cell[robot_id]
        msg = self._Int32MultiArray()
        msg.data = [row, col, int(status)]
        self._command_publishers[robot_id].publish(msg)

    def shutdown(self) -> None:
        if self._executor is not None:
            self._executor.shutdown()
        self.node.destroy_node()
        if self._context is not None:
            self._context.try_shutdown()
