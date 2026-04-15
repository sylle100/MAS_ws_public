#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import JointState, Imu
from std_msgs.msg import Float64MultiArray
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, Quaternion, PoseStamped
from rclpy.qos import ReliabilityPolicy, QoSProfile

import math

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
        self.drone_pose = [0.0, 0.0, 0.0]
        self.setpoint_pose = None
        self.yaw = 0.0
        self.setpoint_yaw = 0.0
        self.joint_pose = [0.0, 0.0]
        self.local_pose_received = False

        self.circle_height = 2.0
        self.min_goto_height = 1.5
        self.max_height = 2.0
        self.circle_center = [0.0, 0.0, self.circle_height]
        self.circle_radius = 2.0
        self.circle_angular_speed = 0.20  # rad/s
        self.transition_duration = 3.0
        self.max_yaw_rate = 0.8  # rad/s
        self.position_gain = 0.8
        self.max_linear_speed = 0.8
        self.goto_tolerance = 0.15
        self.flight_phase = "goto"
        self.circle_angle = 0.0
        self.transition_time = 0.0
        self.transition_radius = 0.0
        self.goto_target = self.circle_center.copy()

        # null space
        self.pose_des = [0, 0, 0, 0, 0, math.pi / 2]
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
        self.drone_pose = [
            float(msg.pose.position.x),
            float(msg.pose.position.y),
            float(msg.pose.position.z),
        ]
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

        v = [
            0.0,
            0.0,
            0.0,
            0.0,
            self.pose_k[4] * (self.pose_des[4] - t0),
            self.pose_k[5] * (self.pose_des[5] - t1),
        ]
        return v

    def limit_velocity(self, velocity):
        speed = math.sqrt(sum(component * component for component in velocity))
        if speed > self.max_linear_speed:
            scale = self.max_linear_speed / speed
            return [component * scale for component in velocity]
        return velocity

    def wrap_angle(self, angle: float) -> float:
        return math.atan2(math.sin(angle), math.cos(angle))

    def update_setpoint_yaw(self, desired_yaw: float, dt: float):
        yaw_error = self.wrap_angle(desired_yaw - self.setpoint_yaw)
        max_step = self.max_yaw_rate * dt
        yaw_step = max(-max_step, min(max_step, yaw_error))
        self.setpoint_yaw = self.wrap_angle(self.setpoint_yaw + yaw_step)

    def update_drone_trajectory(self, dt: float):
        if self.flight_phase == "goto":
            position_error = [
                self.goto_target[i] - self.drone_pose[i]
                for i in range(3)
            ]
            velocity_cmd = self.limit_velocity([
                self.position_gain * component for component in position_error
            ])
            self.setpoint_pose = [
                self.drone_pose[i] + velocity_cmd[i] * dt
                for i in range(3)
            ]
            self.setpoint_pose[2] = max(self.setpoint_pose[2], self.min_goto_height)

            if math.sqrt(sum(component * component for component in position_error)) < self.goto_tolerance:
                self.flight_phase = "transition"
                self.circle_angle = 0.0
                self.transition_time = 0.0
                self.transition_radius = 0.0
                self.setpoint_pose = self.circle_center.copy()
                self.get_logger().info(
                    "Reached (2, 0, 2). Blending smoothly into the horizontal circle."
                )
            return

        if self.flight_phase == "transition":
            self.transition_time += dt
            transition_alpha = min(self.transition_time / self.transition_duration, 1.0)
            smooth_alpha = 0.5 - 0.5 * math.cos(math.pi * transition_alpha)
            angular_speed = self.circle_angular_speed * smooth_alpha
            self.circle_angle += angular_speed * dt
            self.transition_radius = self.circle_radius * smooth_alpha
            self.setpoint_pose = [
                self.circle_center[0] + self.transition_radius * math.cos(self.circle_angle),
                self.circle_center[1] + self.transition_radius * math.sin(self.circle_angle),
                self.circle_center[2],
            ]
            tangent_yaw = self.circle_angle + (math.pi / 2.0)
            self.update_setpoint_yaw(tangent_yaw, dt)

            if transition_alpha >= 1.0:
                self.flight_phase = "circle"
                self.transition_radius = self.circle_radius
                self.get_logger().info(
                    "Transition complete. Continuing the horizontal circle."
                )
            return

        self.circle_angle += self.circle_angular_speed * dt
        self.setpoint_pose = [
            self.circle_center[0] + self.circle_radius * math.cos(self.circle_angle),
            self.circle_center[1] + self.circle_radius * math.sin(self.circle_angle),
            self.circle_center[2],
        ]

        tangent_yaw = self.circle_angle + (math.pi / 2.0)
        self.update_setpoint_yaw(tangent_yaw, dt)

    def timer_callback(self):
        """Move to the circle center, then blend smoothly into the circle."""
        now = self.get_clock().now()
        dt = (now - self.last_time).nanoseconds / 1e9
        self.last_time = now

        if self.setpoint_pose is None:
            return

        dt = max(dt, 1e-3)
        self.update_drone_trajectory(dt)
        self.setpoint_pose[2] = min(max(self.setpoint_pose[2], 0.1), self.max_height)

        self.publish_setpoint()
        self.log_manual_offboard_status()

        v = self.get_v()
        t0 = self.joint_pose[0] + v[4] * dt
        t1 = self.joint_pose[1] + v[5] * dt

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
