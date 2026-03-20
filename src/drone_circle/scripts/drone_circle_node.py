#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import JointState, Imu
from std_msgs.msg import Float64MultiArray
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, Quaternion, PoseStamped
from rclpy.qos import ReliabilityPolicy, QoSProfile

import numpy as np

from drone_circle.orientation_funcs import euler_from_quaternion, quaternion_from_euler

from mavros.base import SENSOR_QOS
from mavros_msgs.msg import State

# Action stuff
from rclpy.action import ActionServer
# from mission.mission_manager import MissionManager # TODO: See this again - might want to link to the action file
# from mission.mission_manager.action import mission


class DroneCircle(Node):
    def __init__(self):
        super().__init__('drone_circle')
        self.get_logger().info("drone circler node started")

        self.drone_pos_sub = self.create_subscription(
            PoseStamped,
            '/mavros/local_position/pose',
            self.drone_pos_callback,
            QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        )

        self.sub_pos_angle = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )

        self.end_effector_sub = self.create_subscription(
            Odometry,
            "/end_effector/pose",
            self.end_effector_callback,
            10
        )
        self.end_effector_pose = Odometry().pose.pose
        self.end_effector_twist = Twist()

        self.imu_sub = self.create_subscription(
            Imu,
            "/mavros/imu/data",
            self.imu_callback,
            qos_profile=SENSOR_QOS
        )
        self.orientation = Quaternion()

        self.state_sub = self.create_subscription(
            State,
            "/mavros/state",
            self.state_callback,
            10
        )

        self.drone_pos_pub = self.create_publisher(
            PoseStamped,
            "/mavros/setpoint_position/local",
            10
        )

        self.drone_vel_pub = self.create_publisher(
            Twist,
            "/mavros/setpoint_velocity/cmd_vel_unstamped",
            10
        )

        self.arm_pub = self.create_publisher(
            Float64MultiArray,
            '/arm_controller/commands',
            10
        )

        self.current_state = State()
        self.drone_pose = np.zeros(3, dtype=float)
        self.setpoint_pose = None
        self.yaw = 0.0
        self.setpoint_yaw = 0.0
        self.joint_pose = [0.0, 0.0]
        self.local_pose_received = False

        self.circle_height = 2.0
        self.max_height = 2.0
        self.circle_center = np.array([2.0, 0.0, self.circle_height], dtype=float)
        self.circle_radius = 2.0
        self.circle_angular_speed = 0.35  # rad/s
        self.position_gain = 0.8
        self.max_linear_speed = 0.8
        self.goto_tolerance = 0.15
        self.flight_phase = "goto"
        self.circle_angle = 0.0
        self.goto_target = self.circle_center + np.array([self.circle_radius, 0.0, 0.0], dtype=float)

        # null space
        self.pose_des = [0, 0, 0, 0, 0, np.pi / 2]
        self.pose_k = [0, 0, 0, 0, 1, 1]

        self.last_time = self.get_clock().now()
        self.offboard_ready_logged = False
        self.last_status_log_time = self.get_clock().now()

        # PX4 offboard mode requires a continuous stream of setpoints.
        self.timer = self.create_timer(0.05, self.timer_callback)

    def joint_state_callback(self, msg: JointState):
        t0 = msg.position[0]
        t1 = msg.position[1]
        self.joint_pose = [t0, t1]

    def end_effector_callback(self, msg: Odometry):
        self.end_effector_pose = msg.pose.pose
        self.end_effector_twist = msg.twist.twist

    def imu_callback(self, msg: Imu):
        quat = [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]
        self.roll, self.pitch, self.yaw = euler_from_quaternion(quat)

    def state_callback(self, msg: State):
        self.current_state = msg

    def drone_pos_callback(self, msg: PoseStamped):
        self.local_pose_received = True
        self.drone_pose = np.array(
            [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z],
            dtype=float
        )
        if self.setpoint_pose is None:
            self.setpoint_pose = self.drone_pose.copy()
            self.setpoint_yaw = self.yaw

    def publish_setpoint(self):
        if self.setpoint_pose is None:
            return

        cmd = PoseStamped()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.pose.position.x = float(self.setpoint_pose[0])
        cmd.pose.position.y = float(self.setpoint_pose[1])
        cmd.pose.position.z = float(self.setpoint_pose[2])

        q = quaternion_from_euler(0, 0, self.setpoint_yaw)
        cmd.pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
        self.drone_pos_pub.publish(cmd)

    def log_manual_offboard_status(self):
        if self.setpoint_pose is None:
            return

        now = self.get_clock().now()
        if (now - self.last_status_log_time).nanoseconds < 2_000_000_000:
            return
        self.last_status_log_time = now

        if not self.current_state.connected:
            self.get_logger().info("Waiting for MAVROS to connect to PX4.")
            return

        if not self.local_pose_received:
            self.get_logger().info("Waiting for /mavros/local_position/pose before starting trajectory.")
            return

        if self.current_state.mode != "OFFBOARD":
            self.offboard_ready_logged = False
            self.get_logger().info("Streaming setpoints. Switch vehicle mode to OFFBOARD in QGroundControl, then arm.")
            return

        if not self.current_state.armed:
            self.offboard_ready_logged = False
            self.get_logger().info("OFFBOARD mode is active, but the vehicle is not armed. Arm it in QGroundControl.")
            return

        if not self.offboard_ready_logged:
            self.get_logger().info("OFFBOARD is active and the vehicle is armed. The trajectory should now be running.")
            self.offboard_ready_logged = True

    def get_v(self):
        """Desired null space"""
        (t0, t1) = self.joint_pose

        v = np.array([
            [self.pose_k[0] * 0],
            [self.pose_k[1] * 0],
            [self.pose_k[2] * 0],
            [self.pose_k[3] * 0],
            [self.pose_k[4] * (self.pose_des[4] - t0)],
            [self.pose_k[5] * (self.pose_des[5] - t1)]
        ])
        return v

    def limit_velocity(self, velocity: np.ndarray) -> np.ndarray:
        speed = np.linalg.norm(velocity)
        if speed > self.max_linear_speed:
            return velocity * (self.max_linear_speed / speed)
        return velocity

    def update_drone_trajectory(self, dt: float):
        if self.flight_phase == "goto":
            position_error = self.goto_target - self.drone_pose
            velocity_cmd = self.limit_velocity(self.position_gain * position_error)
            self.setpoint_pose = self.drone_pose + velocity_cmd * dt

            if np.linalg.norm(position_error) < self.goto_tolerance:
                self.flight_phase = "circle"
                self.circle_angle = 0.0
                self.setpoint_pose = self.goto_target.copy()
                self.get_logger().info(
                    "Reached the circle start point. Starting a horizontal circle at 2 m height with a 2 m radius."
                )
            return

        self.circle_angle += self.circle_angular_speed * dt
        self.setpoint_pose = np.array([
            self.circle_center[0] + self.circle_radius * np.cos(self.circle_angle),
            self.circle_center[1] + self.circle_radius * np.sin(self.circle_angle),
            self.circle_center[2]
        ], dtype=float)

        tangent_yaw = self.circle_angle + (np.pi / 2.0)
        self.setpoint_yaw = tangent_yaw

    def timer_callback(self):
        """Move to the entry point, then keep flying a circle."""
        now = self.get_clock().now()
        dt = (now - self.last_time).nanoseconds / 1e9
        self.last_time = now

        if self.setpoint_pose is None:
            return

        dt = max(dt, 1e-3)
        self.update_drone_trajectory(dt)
        self.setpoint_pose[2] = np.clip(self.setpoint_pose[2], 0.1, self.max_height)

        self.publish_setpoint()
        self.log_manual_offboard_status()

        v = self.get_v()
        t0 = self.joint_pose[0] + v[4][0] * dt
        t1 = self.joint_pose[1] + v[5][0] * dt

        cmd_arm = Float64MultiArray()
        cmd_arm.data = [t0, t1]
        self.arm_pub.publish(cmd_arm)


def main(args=None):
    rclpy.init(args=args)
    node = DroneCircle()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


# # Potential action server main
# def main(args=None):
#     rclpy.init(args=args)
#     action_server = inverse_jacobian()
#     rclpy.spin(action_server)


if __name__ == '__main__':
    main()
