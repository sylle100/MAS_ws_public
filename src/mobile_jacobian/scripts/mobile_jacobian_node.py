#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer
from sensor_msgs.msg import JointState, Imu
from std_msgs.msg import Float64MultiArray
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, Quaternion, Vector3
from geometry_msgs.msg import PoseStamped
from rclpy.qos import ReliabilityPolicy, QoSProfile

import numpy as np
import functools

from mobile_jacobian.orientation_funcs import euler_from_quaternion, quaternion_from_euler
from mobile_jacobian.jacobian_math import JMoore, J_mobile, Jinv

from mavros.base import SENSOR_QOS
import time


# Action stuff
from rclpy.action import ActionServer
# from mission.mission_manager import MissionManager # TODO: See this again - might want to link to the action file
# from mission.mission_manager.action import mission

class mobilejacobian(Node):
    def __init__(self):
        super().__init__('mobile_jacobian')
        self.get_logger().info("mobile_jacobian node started")

        self.drone_pos_sub = self.create_subscription(
            PoseStamped,                    # Topic type
            '/mavros/local_position/pose',  # Topic name
            self.drone_pos_callback,        # Runs the function
            QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)                               # QoS depth
        )
        
        self.sub_pos_angle = self.create_subscription(
            JointState,                 # Topic type
            '/joint_states',            # Topic name
            self.joint_state_callback,  # Runs the function
            10                          # QoS depth
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

        # Targets for the end-effector to reach (3D points).
        # Each subscription updates one of these 4 targets.
        self.targets = [np.zeros(3) for _ in range(4)]
        self.targets_received = [False] * 4

        self.create_subscription(
            Odometry,
            "/target1/pose",
            lambda msg, idx=0: self.target_callback(msg, idx),
            10
        )

        self.create_subscription(
            Odometry,
            "/target2/pose",
            lambda msg, idx=1: self.target_callback(msg, idx),
            10
        )

        self.create_subscription(
            Odometry,
            "/target3/pose",
            lambda msg, idx=2: self.target_callback(msg, idx),
            10
        )

        self.create_subscription(
            Odometry,
            "/target4/pose",
            lambda msg, idx=3: self.target_callback(msg, idx),
            10
        )

        self.drone_pos_pub = self.create_publisher(
            PoseStamped,
            "/mavros/setpoint_position/local",
            10)

        self.drone_vel_pub = self.create_publisher(
            Twist,
            "/mavros/setpoint_velocity/cmd_vel_unstamped",
            10)
        
        self.arm_pub = self.create_publisher(
            Float64MultiArray,
            '/arm_controller/commands',
            10)
        


        self.drone_pose = [0,0,0]
        self.setpoint_pose = None
        self.yaw = 0
        self.setpoint_yaw = 0
        self.joint_pose = [0,0]

        # record the starting drone position so we can return to it
        self.start_pose = None

        # Control parameters
        self.links = np.array([0.15, 0.15])
        self.k = 1

        # Safety limits
        self.min_z = 0.1  # minimum allowed altitude for commanded setpoint (meters)

        # null space
        self.pose_des = [0,0,0,0,0,np.pi/2]
        self.pose_k   = [0,0,0,0,1,1]

        # mission phases: 'goto', 'hover', 'return', 'done'
        self.phase = 'goto'
        self.phase_start_time = time.time()
        self.hover_start_time = None

        # Target sequencing
        self.target_index = 0

        self.hover_height = 0.2  # meters above target
        self.hover_duration = 5.0  # seconds
        self.arrival_threshold = 0.05  # meters, to consider "arrived" at a point (for phase switching)

        # For integration of velocity commands
        self.last_time = self.get_clock().now()

        # Limits for commanded velocities
        self.max_linear_vel = 0.5  # m/s
        self.max_angular_vel = 1.0  # rad/s
        self.max_joint_vel = 1.0  # rad/s

        self.timer = self.create_timer(.1, self.timer_callback)

    def target_callback(self, msg, index):
        p = msg.pose.pose
        self.targets[index] = np.array([p.position.x, p.position.y, p.position.z])
        if not self.targets_received[index]:
            self.targets_received[index] = True
            self.get_logger().info(f"Received target {index + 1}: {self.targets[index]}")
    
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

    def drone_pos_callback(self, msg: PoseStamped):
        self.drone_pose = [msg.pose.position.x,msg.pose.position.y,msg.pose.position.z]
        if self.setpoint_pose is None:
            self.setpoint_pose = np.array(self.drone_pose)
            self.setpoint_yaw = self.yaw

        if self.start_pose is None:
            self.start_pose = np.array(self.drone_pose)
            self.get_logger().info(f"Recorded start pose: {self.start_pose}")

    def get_p_dot_des(self, desired_pos: np.ndarray):
        """Desired change in position (3x1 vector).

        Args:
            desired_pos: 3-element desired XYZ position in world frame.
        """

        # current end-effector position
        x = self.end_effector_pose.position.x
        y = self.end_effector_pose.position.y
        z = self.end_effector_pose.position.z

        error = np.array([
            [desired_pos[0] - x],
            [desired_pos[1] - y],
            [desired_pos[2] - z],
        ])

        # k is a velocity scalar
        return error * self.k

    def _compute_desired_position(self) -> np.ndarray:
        """Compute the desired drone position for the current mission phase."""

        # Default to start pose until we have a true starting pose.
        if self.start_pose is None:
            return np.array(self.drone_pose)

        if self.phase in ('goto', 'hover'):
            # Skip targets that haven't been received yet.
            while self.target_index < len(self.targets) and not self.targets_received[self.target_index]:
                self.get_logger().info(f"Skipping target {self.target_index + 1}: no data received")
                self.target_index += 1

            if self.target_index >= len(self.targets):
                return self.start_pose

            target = self.targets[self.target_index].copy()
            target[2] += self.hover_height
            # Safety: never command below minimum altitude
            target[2] = max(target[2], self.min_z)
            return target

        if self.phase == 'return':
            return self.start_pose

        # done
        return self.start_pose

    def get_v(self):
        """Desired null space"""
        (t0,t1) = self.joint_pose
        
        v = np.array([
            [self.pose_k[0] * 0],
            [self.pose_k[1] * 0],
            [self.pose_k[2] * 0],
            [self.pose_k[3] * 0],
            [self.pose_k[4] * (self.pose_des[4] - t0)],
            [self.pose_k[5] * (self.pose_des[5] - t1)]
            ])
        return v

    def timer_callback(self):
        """Calculates, q, the desired velocity to reach target and executes mission phases."""

        # Wait until we have a defined start pose.
        if self.start_pose is None:
            return

        # If we haven't received any targets yet, hold position until the first target arrives.
        if not any(self.targets_received):
            return

        # Determine desired position based on mission phase.
        desired_pos = self._compute_desired_position()

        # Phase transitions (based on end-effector, not drone base)
        current_pos = np.array([self.end_effector_pose.position.x,
                                self.end_effector_pose.position.y,
                                self.end_effector_pose.position.z])
        dist = np.linalg.norm(current_pos - desired_pos)
        now = self.get_clock().now()

        if self.phase == 'goto' and dist < self.arrival_threshold:
            self.phase = 'hover'
            self.hover_start_time = time.time()
            self.get_logger().info('Reached target; hovering for %.1fs' % self.hover_duration)

        if self.phase == 'hover':
            elapsed = time.time() - self.hover_start_time
            if elapsed >= self.hover_duration:
                # Move to the next target, or return home if done.
                self.target_index += 1
                while self.target_index < len(self.targets) and not self.targets_received[self.target_index]:
                    self.get_logger().info(f"Skipping target {self.target_index + 1}: no data received")
                    self.target_index += 1

                if self.target_index < len(self.targets):
                    self.phase = 'goto'
                    self.phase_start_time = time.time()
                    self.get_logger().info(f"Hover complete; going to target {self.target_index + 1}")
                else:
                    self.phase = 'return'
                    self.phase_start_time = time.time()
                    self.get_logger().info('Hover complete; returning to start')

        if self.phase == 'return' and dist < self.arrival_threshold:
            self.phase = 'done'
            self.get_logger().info('Returned to start; mission done')

        # Compute Jacobian and inverse once per cycle.
        j_mobile = J_mobile(self.links, self.joint_pose[0], self.joint_pose[1], self.yaw)
        j_mobile_inv = JMoore(j_mobile)

        # q task
        pd_des = self.get_p_dot_des(desired_pos)
        q_task = j_mobile_inv @ pd_des

        # q Null
        v = self.get_v()
        q_null = (np.eye(6) - j_mobile_inv @ j_mobile) @ v

        # q dot
        if self.phase == 'done':
            q_dot = np.zeros((6, 1))
        else:
            q_dot = q_task + q_null

        # Clamp commanded velocities to reasonable limits
        q_dot[0:3, 0] = np.clip(q_dot[0:3, 0], -self.max_linear_vel, self.max_linear_vel)
        q_dot[3, 0] = np.clip(q_dot[3, 0], -self.max_angular_vel, self.max_angular_vel)
        q_dot[4:6, 0] = np.clip(q_dot[4:6, 0], -self.max_joint_vel, self.max_joint_vel)

        # Integrate velocity to get next setpoint using actual elapsed time
        now = self.get_clock().now()
        dt = (now - self.last_time).nanoseconds * 1e-9
        if dt <= 0:
            dt = 1e-3
        self.last_time = now

        # update commanded setpoint based on computed velocity
        if self.setpoint_pose is None:
            self.setpoint_pose = np.array(self.drone_pose)
            self.setpoint_yaw = self.yaw

        self.setpoint_pose[0] += q_dot[0][0] * dt
        self.setpoint_pose[1] += q_dot[1][0] * dt
        self.setpoint_pose[2] += q_dot[2][0] * dt
        self.setpoint_yaw += q_dot[3][0] * dt

        # Safety: don't command below the minimum allowed altitude
        self.setpoint_pose[2] = max(self.setpoint_pose[2], self.min_z)


        # velocity bases
        # cmd_lin = Vector3(x=q_dot[0][0],y=q_dot[1][0], z=q_dot[2][0])
        # cmd_ang = Vector3(x=0.0, y=0.0, z=q_dot[3][0])
        # cmd = Twist()
        # cmd.linear = cmd_lin
        # cmd.angular = cmd_ang
        # self.drone_vel_pub.publish(cmd)

        cmd = PoseStamped()
        cmd.pose.position.x = self.setpoint_pose[0]
        cmd.pose.position.y = self.setpoint_pose[1]
        cmd.pose.position.z = max(self.setpoint_pose[2], self.min_z)


        Q = quaternion_from_euler(0,0,self.setpoint_yaw)
        
        cmd.pose.orientation = Quaternion(x = Q[0], y = Q[1], z = Q[2], w = Q[3])
        self.drone_pos_pub.publish(cmd)




        # update arm joint position based on velosity
        t0 = self.joint_pose[0] + q_dot[4][0]
        t1 = self.joint_pose[1] + q_dot[5][0]
        
        cmd_arm = Float64MultiArray()

        # transmit new target positions
        cmd_arm.data = [t0, t1]
        self.arm_pub.publish(cmd_arm)

    def execute_callback(self, goal_handle):
        """Execute the switch target action."""
        target_index = goal_handle.request.target_index

        if target_index < 0 or target_index >= len(self.targets):
            goal_handle.abort()
            result = SwitchTarget.Result()
            result.success = False
            result.message = f"Invalid target index {target_index}. Must be 0-3."
            return result

        if not self.targets_received[target_index]:
            goal_handle.abort()
            result = SwitchTarget.Result()
            result.success = False
            result.message = f"Target {target_index} not received yet."
            return result

        # Set the target
        self.target_index = target_index
        self.phase = 'goto'
        self.phase_start_time = time.time()
        self.get_logger().info(f"Switching to target {target_index + 1}")

        # Wait until reached
        while rclpy.ok():
            if self.phase == 'hover':
                break

            # Publish feedback
            feedback = SwitchTarget.Feedback()
            feedback.current_target = self.target_index
            current_pos = np.array([self.end_effector_pose.position.x,
                                    self.end_effector_pose.position.y,
                                    self.end_effector_pose.position.z])
            desired_pos = self._compute_desired_position()
            feedback.distance_to_target = float(np.linalg.norm(current_pos - desired_pos))
            goal_handle.publish_feedback(feedback)

            time.sleep(0.1)  # Small delay to not hog CPU

        if goal_handle.is_cancel_requested:
            goal_handle.canceled()
            result = SwitchTarget.Result()
            result.success = False
            result.message = "Goal canceled"
            return result

        goal_handle.succeed()
        result = SwitchTarget.Result()
        result.success = True
        result.message = f"Successfully switched to target {target_index + 1}"
        return result


def main(args=None):
    rclpy.init(args=args)
    node = mobilejacobian()
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


