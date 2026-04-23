#!/usr/bin/env python3

import math
import threading
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Quaternion, Twist
from nav_msgs.msg import Odometry
from rclpy.action import ActionServer
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Imu, JointState
from std_msgs.msg import Float64MultiArray

from mobile_jacobian.action import SwitchTarget
from mobile_jacobian.jacobian_math import J_mobile, JMoore
from mobile_jacobian.orientation_funcs import euler_from_quaternion, quaternion_from_euler

from mavros.base import SENSOR_QOS


class MobileJacobian(Node):
    def __init__(self):
        super().__init__('mobile_jacobian')
        self.get_logger().info('mobile_jacobian node started')

        self.action_group = ReentrantCallbackGroup()
        self.state_lock = threading.Lock()

        self.expected_joint_names = ['joint1', 'joint2']
        self.use_degrees_for_gimbal = True
        self.command_frame = None
        self.last_hover_log_second = -1
        self.frame_warnings_issued = set()

        self.have_pose = False
        self.have_imu = False
        self.have_joint_state = False
        self.have_ee_pose = False

        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        self.drone_pose = np.zeros(3, dtype=float)
        self.setpoint_pose = None
        self.setpoint_yaw = 0.0
        self.joint_pose = [0.0, 0.0]
        self.gimbal_pitch = 0.0
        self.gimbal_roll = 0.0
        self.start_pose = None
        self.start_ee_pose = None
        self.end_effector_pose = Odometry().pose.pose
        self.end_effector_twist = Twist()
        self.orientation = Quaternion()

        self.current_goal_handle = None
        self.current_goal_target_index = None

        self.drone_pos_sub = self.create_subscription(
            PoseStamped,
            '/mavros/local_position/pose',
            self.drone_pos_callback,
            QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT),
        )

        self.sub_pos_angle = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10,
        )

        self.end_effector_sub = self.create_subscription(
            Odometry,
            '/end_effector/pose',
            self.end_effector_callback,
            10,
        )

        self.imu_sub = self.create_subscription(
            Imu,
            '/mavros/imu/data',
            self.imu_callback,
            qos_profile=SENSOR_QOS,
        )

        self.targets = [np.zeros(3) for _ in range(4)]
        self.targets_received = [False] * 4

        self.create_subscription(Odometry, '/target1/pose', lambda msg, idx=0: self.target_callback(msg, idx), 10)
        self.create_subscription(Odometry, '/target2/pose', lambda msg, idx=1: self.target_callback(msg, idx), 10)
        self.create_subscription(Odometry, '/target3/pose', lambda msg, idx=2: self.target_callback(msg, idx), 10)
        self.create_subscription(Odometry, '/target4/pose', lambda msg, idx=3: self.target_callback(msg, idx), 10)

        self.drone_pos_pub = self.create_publisher(PoseStamped, '/mavros/setpoint_position/local', 10)
        self.drone_vel_pub = self.create_publisher(Twist, '/mavros/setpoint_velocity/cmd_vel_unstamped', 10)
        self.arm_pub = self.create_publisher(Float64MultiArray, '/arm_controller/commands', 10)
        self.gimbal_pub = self.create_publisher(Float64MultiArray, '/gimbal_controller/commands', 10)

        self.links = np.array([0.15, 0.15])
        self.k = 0.3
        self.min_z = 0.1

        self.pose_des_goto = [0, 0, 0, 0, np.pi - (8 * np.pi) / 9, np.pi / 3]
        self.pose_des_hover = [0, 0, 0, 0, (8 * np.pi) / 9, np.pi / 3]
        self.pose_des_tap = [0, 0, 0, 0, -np.pi / 3, 0]
        self.pose_k = [0, 0, 0, 0, 1, 1]

        self.phase = 'goto'
        self.phase_start_time = time.time()
        self.hover_start_time = None
        self.tap_start_time = None
        self.target_index = 0

        self.goto_timeout = 25.0
        self.hover_height = 0.2
        self.approach_height = 0.2
        self.descend_radius_xy = 0.12
        self.target_avoid_radius = 0.30  # increased from 0.18
        self.target_avoid_influence = 0.60  # increased from 0.35  
        self.target_avoid_gain = 0.40  # increased from 0.25
        self.hover_duration = 2.0
        self.tap_depth = 0.22
        self.tap_duration = 1.0
        self.arrival_threshold = 0.10

        self.optitrack_x_min = -5.8
        self.optitrack_x_max = 5.8
        self.optitrack_y_min = -5.8
        self.optitrack_y_max = 5.8
        self.optitrack_z_min = 0.1
        self.optitrack_z_max = 5.5

        self.last_time = self.get_clock().now()

        self.max_linear_vel = 0.3
        self.max_angular_vel = 0.6
        self.max_joint_vel = 0.6
        self.max_linear_acc = 0.5
        self.max_angular_acc = 0.8
        self.max_joint_acc = 1.0
        self.approach_smoothing = 0.25
        self.last_q_dot = np.zeros((6, 1))

        self.joint_min = np.array([-np.pi, -np.pi])
        self.joint_max = np.array([np.pi, np.pi])

        self.switch_target_action_server = ActionServer(
            self,
            SwitchTarget,
            'switch_target',
            self.execute_callback,
            callback_group=self.action_group,
        )

        self.timer = self.create_timer(0.1, self.timer_callback)

    def _warn_frame_once(self, label: str, frame_id: str):
        key = (label, frame_id)
        if key in self.frame_warnings_issued:
            return
        self.frame_warnings_issued.add(key)
        self.get_logger().warn(
            f'{label} frame_id={frame_id!r}. Verify all pose topics share a compatible world frame.'
        )

    def _validate_target(self, target: np.ndarray) -> bool:
        if not np.all(np.isfinite(target)):
            self.get_logger().warn(f'Rejecting invalid target (non-finite values): {target}')
            return False

        if not (
            self.optitrack_x_min <= target[0] <= self.optitrack_x_max
            and self.optitrack_y_min <= target[1] <= self.optitrack_y_max
            and self.optitrack_z_min <= target[2] <= self.optitrack_z_max
        ):
            self.get_logger().warn(
                'Rejecting out-of-bounds target '
                f'{target}; bounds x=[{self.optitrack_x_min}, {self.optitrack_x_max}], '
                f'y=[{self.optitrack_y_min}, {self.optitrack_y_max}], '
                f'z=[{self.optitrack_z_min}, {self.optitrack_z_max}]'
            )
            return False

        return True

    def _all_required_state_ready(self) -> bool:
        return self.have_pose and self.have_imu and self.have_joint_state and self.have_ee_pose

    def _wrap_angle(self, angle: float) -> float:
        return math.atan2(math.sin(angle), math.cos(angle))

    def target_callback(self, msg, index):
        target = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z,
        ], dtype=float)

        if msg.header.frame_id:
            self._warn_frame_once(f'target{index + 1}', msg.header.frame_id)

        if not self._validate_target(target):
            return

        with self.state_lock:
            self.targets[index] = target
            first_time = not self.targets_received[index]
            self.targets_received[index] = True

        if first_time:
            self.get_logger().info(f'Received target {index + 1}: {target}')

    def joint_state_callback(self, msg: JointState):
        if len(msg.position) < 2:
            self.get_logger().warn('Received JointState with fewer than 2 positions; ignoring message')
            return

        if msg.name and all(name in msg.name for name in self.expected_joint_names):
            name_to_idx = {name: idx for idx, name in enumerate(msg.name)}
            t0 = msg.position[name_to_idx[self.expected_joint_names[0]]]
            t1 = msg.position[name_to_idx[self.expected_joint_names[1]]]
        else:
            if msg.name and set(self.expected_joint_names) - set(msg.name):
                self.get_logger().warn(
                    f'Expected joints {self.expected_joint_names}, got {list(msg.name)}; falling back to first two positions'
                )
            t0 = msg.position[0]
            t1 = msg.position[1]

        if not (math.isfinite(t0) and math.isfinite(t1)):
            self.get_logger().warn('Received non-finite joint state values; ignoring message')
            return

        with self.state_lock:
            self.joint_pose = [float(t0), float(t1)]
            self.have_joint_state = True

    def end_effector_callback(self, msg: Odometry):
        if msg.header.frame_id:
            self._warn_frame_once('end_effector', msg.header.frame_id)

        self.end_effector_pose = msg.pose.pose
        self.end_effector_twist = msg.twist.twist
        self.have_ee_pose = True

        if self.start_ee_pose is None:
            self.start_ee_pose = np.array([
                self.end_effector_pose.position.x,
                self.end_effector_pose.position.y,
                self.end_effector_pose.position.z,
            ], dtype=float)
            self.get_logger().info(f'Recorded start end-effector pose: {self.start_ee_pose}')

    def imu_callback(self, msg: Imu):
        quat = [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]
        if not all(math.isfinite(v) for v in quat):
            self.get_logger().warn('Received non-finite IMU quaternion; ignoring message')
            return
        self.roll, self.pitch, self.yaw = euler_from_quaternion(quat)
        self.have_imu = True

    def drone_pos_callback(self, msg: PoseStamped):
        if msg.header.frame_id:
            self._warn_frame_once('drone_pose', msg.header.frame_id)
            if self.command_frame is None:
                self.command_frame = msg.header.frame_id

        pose = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
        ], dtype=float)
        if not np.all(np.isfinite(pose)):
            self.get_logger().warn('Received non-finite drone pose; ignoring message')
            return

        self.drone_pose = pose
        self.have_pose = True

        if self.setpoint_pose is None and self.have_imu:
            self.setpoint_pose = pose.copy()
            self.setpoint_yaw = self.yaw

        if self.start_pose is None:
            self.start_pose = pose.copy()
            self.get_logger().info(f'Recorded start pose: {self.start_pose}')

    def get_p_dot_des(self, desired_pos: np.ndarray):
        x = self.end_effector_pose.position.x
        y = self.end_effector_pose.position.y
        z = self.end_effector_pose.position.z

        error = np.array([
            [desired_pos[0] - x],
            [desired_pos[1] - y],
            [desired_pos[2] - z],
        ])

        error_norm = np.linalg.norm(error)
        if error_norm < 1e-6:
            return np.zeros((3, 1))

        speed_scale = np.tanh(error_norm / self.approach_smoothing)
        return error * (self.k * speed_scale)

    def _rate_limit(self, q_dot: np.ndarray, dt: float) -> np.ndarray:
        if dt <= 0.0:
            return q_dot

        max_delta = np.array([
            self.max_linear_acc * dt,
            self.max_linear_acc * dt,
            self.max_linear_acc * dt,
            self.max_angular_acc * dt,
            self.max_joint_acc * dt,
            self.max_joint_acc * dt,
        ]).reshape(6, 1)

        delta = q_dot - self.last_q_dot
        return self.last_q_dot + np.clip(delta, -max_delta, max_delta)

    def _clamp_to_optitrack_bounds(self, pos: np.ndarray) -> np.ndarray:
        pos_clamped = pos.copy()
        pos_clamped[0] = np.clip(pos_clamped[0], self.optitrack_x_min, self.optitrack_x_max)
        pos_clamped[1] = np.clip(pos_clamped[1], self.optitrack_y_min, self.optitrack_y_max)
        pos_clamped[2] = np.clip(pos_clamped[2], self.optitrack_z_min, self.optitrack_z_max)
        return pos_clamped

    def _compute_desired_position(self) -> np.ndarray:
        if self.start_pose is None:
            return np.array(self.drone_pose)

        if self.phase in ('goto', 'hover', 'tap', 'clear'):
            while self.target_index < len(self.targets) and not self.targets_received[self.target_index]:
                self.get_logger().info(f'Skipping target {self.target_index + 1}: no data received')
                self.target_index += 1

            if self.target_index >= len(self.targets):
                return self.start_pose

            target = self.targets[self.target_index].copy()
            hover_pos = target.copy()
            hover_pos[2] = max(target[2] + self.hover_height, self.min_z)

            if self.phase == 'clear':
                # Clear phase: go to high altitude above current position
                clear_pos = np.array(self.drone_pose).copy()
                clear_pos[2] = min(clear_pos[2] + 2.0, self.optitrack_z_max)  # 2m above current, within bounds
                return clear_pos

            if self.phase == 'hover':
                return hover_pos

            if self.phase == 'tap':
                tap_pos = hover_pos.copy()
                tap_pos[2] = max(hover_pos[2] - self.tap_depth, self.min_z)
                return tap_pos

            current_pos = np.array([
                self.end_effector_pose.position.x,
                self.end_effector_pose.position.y,
                self.end_effector_pose.position.z,
            ])
            overhead_pos = hover_pos.copy()
            overhead_pos[2] += self.approach_height

            horizontal_dist = np.linalg.norm(current_pos[:2] - target[:2])
            overhead_ready = current_pos[2] >= (overhead_pos[2] - self.arrival_threshold)

            if horizontal_dist > self.descend_radius_xy or not overhead_ready:
                return overhead_pos
            return hover_pos

        if self.phase == 'return':
            return self.start_pose

        if self.start_ee_pose is not None:
            return self.start_ee_pose.copy()
        return self.start_pose

    def _compute_target_avoidance_velocity(self, current_pos: np.ndarray) -> np.ndarray:
        avoid = np.zeros((3, 1))

        for idx, target_received in enumerate(self.targets_received):
            if not target_received:
                continue

            target = self.targets[idx].copy()
            safe_hover_z = max(target[2] + self.hover_height, self.min_z)
            safe_overhead_z = safe_hover_z + self.approach_height

            xy_delta = current_pos[:2] - target[:2]
            xy_dist = np.linalg.norm(xy_delta)

            if xy_dist >= self.target_avoid_influence:
                continue

            active_target = idx == self.target_index and self.phase in ('goto', 'hover', 'tap', 'clear')
            if active_target:
                # Don't avoid the target we're actively trying to reach
                continue

            if xy_dist < 1e-6:
                xy_dir = np.array([1.0, 0.0])
            else:
                xy_dir = xy_delta / xy_dist

            if xy_dist <= self.target_avoid_radius:
                horizontal_strength = self.target_avoid_gain
            else:
                ratio = (self.target_avoid_influence - xy_dist) / (
                    self.target_avoid_influence - self.target_avoid_radius
                )
                horizontal_strength = self.target_avoid_gain * ratio

            upward_ratio = max(0.0, safe_overhead_z - current_pos[2]) / max(self.approach_height, 1e-6)
            upward_strength = self.target_avoid_gain * min(1.0, upward_ratio)

            avoid += np.array([
                [xy_dir[0] * horizontal_strength],
                [xy_dir[1] * horizontal_strength],
                [upward_strength],
            ])

        return avoid

    def _compute_gimbal_angles(self):
        # Gimbal compensates for drone pitch to keep link1 pointing down in world frame
        # When drone pitches up, gimbal pitches down to maintain vertical arm orientation
        self.gimbal_pitch = -self.pitch
        self.gimbal_roll = -self.roll

    def get_v(self):
        t0, t1 = self.joint_pose

        if self.phase == 'hover':
            pose_des = self.pose_des_hover
        elif self.phase == 'tap':
            pose_des = self.pose_des_tap
        else:
            pose_des = self.pose_des_goto

        return np.array([
            [self.pose_k[0] * 0],
            [self.pose_k[1] * 0],
            [self.pose_k[2] * 0],
            [self.pose_k[3] * 0],
            [self.pose_k[4] * (pose_des[4] - t0)],
            [self.pose_k[5] * (pose_des[5] - t1)],
        ])

    def _maybe_finish_goal(self):
        goal_handle = self.current_goal_handle
        if goal_handle is None:
            return

        if goal_handle.is_cancel_requested:
            goal_handle.canceled()
            self.current_goal_handle = None
            self.current_goal_target_index = None
            return

        feedback = SwitchTarget.Feedback()
        feedback.current_target = self.target_index
        current_pos = np.array([
            self.end_effector_pose.position.x,
            self.end_effector_pose.position.y,
            self.end_effector_pose.position.z,
        ])
        desired_pos = self._compute_desired_position()
        feedback.distance_to_target = float(np.linalg.norm(current_pos - desired_pos))
        goal_handle.publish_feedback(feedback)

        if self.phase == 'hover' and self.current_goal_target_index == self.target_index:
            goal_handle.succeed()
            self.current_goal_handle = None
            self.current_goal_target_index = None
            self.get_logger().info('SwitchTarget goal completed')

    def timer_callback(self):
        if not self._all_required_state_ready():
            return

        if not any(self.targets_received):
            return

        desired_pos = self._compute_desired_position()

        current_pos = np.array([
            self.end_effector_pose.position.x,
            self.end_effector_pose.position.y,
            self.end_effector_pose.position.z,
        ])
        dist = np.linalg.norm(current_pos - desired_pos)

        if self.phase == 'goto' and dist < self.arrival_threshold:
            self.phase = 'hover'
            self.phase_start_time = time.time()
            self.hover_start_time = self.phase_start_time
            self.last_hover_log_second = -1
            self.get_logger().info(
                f'Reached target; EE_pos={current_pos}, desired_pos={desired_pos}, '
                f'dist={dist:.3f}; hovering for {self.hover_duration:.1f}s'
            )

        if self.phase == 'goto':
            goto_elapsed = time.time() - self.phase_start_time
            if goto_elapsed >= self.goto_timeout:
                self.get_logger().warn(f'Goto timeout ({goto_elapsed:.1f}s) reached! Forcing hover phase.')
                self.phase = 'hover'
                self.phase_start_time = time.time()
                self.hover_start_time = self.phase_start_time
                self.last_hover_log_second = -1

        if self.phase == 'hover':
            if self.hover_start_time is not None:
                elapsed = time.time() - self.hover_start_time
                if elapsed >= self.hover_duration:
                    self.get_logger().warn(f'HOVER COMPLETE! Transitioning to TAP phase. (elapsed={elapsed:.2f}s)')
                    self.phase = 'tap'
                    self.phase_start_time = time.time()
                    self.tap_start_time = self.phase_start_time
                else:
                    elapsed_whole = int(elapsed)
                    if elapsed_whole != self.last_hover_log_second:
                        self.last_hover_log_second = elapsed_whole
                        self.get_logger().info(
                            f'HOVER phase: elapsed={elapsed:.2f}s / {self.hover_duration}s, '
                            f'dist_to_target={dist:.3f}m'
                        )
            else:
                self.get_logger().warn('HOVER phase but hover_start_time is None!')

        if self.phase == 'tap':
            if self.tap_start_time is not None:
                elapsed = time.time() - self.tap_start_time
                if elapsed >= self.tap_duration:
                    self.get_logger().warn('TAP COMPLETE! Moving to next target.')
                    self.target_index += 1
                    while self.target_index < len(self.targets) and not self.targets_received[self.target_index]:
                        self.get_logger().info(f'Skipping target {self.target_index + 1}: no data received')
                        self.target_index += 1

                    if self.target_index < len(self.targets):
                        self.phase = 'clear'  # Go to clear phase first
                        self.phase_start_time = time.time()
                        self.get_logger().info(f'Clearing obstacles before going to target {self.target_index + 1}')
                    else:
                        self.phase = 'return'
                        self.phase_start_time = time.time()
                        self.get_logger().info('All targets tapped; returning to start')
            else:
                self.get_logger().warn('TAP phase but tap_start_time is None!')

        if self.phase == 'clear':
            # Clear phase: go to high altitude to avoid obstacles
            clear_height = 2.0  # meters above current position
            if current_pos[2] >= (self.drone_pose[2] + clear_height - self.arrival_threshold):
                self.phase = 'goto'
                self.phase_start_time = time.time()
                self.get_logger().info(f'Cleared obstacles; going to target {self.target_index + 1}')
            else:
                elapsed = time.time() - self.phase_start_time
                if int(elapsed) % 2 == 0:  # Log every 2 seconds
                    self.get_logger().info(f'CLEAR phase: ascending to {self.drone_pose[2] + clear_height:.2f}m, current={current_pos[2]:.2f}m')

        if self.phase == 'return' and dist < self.arrival_threshold:
            self.phase = 'done'
            self.phase_start_time = time.time()
            self.get_logger().info('Returned to start; mission done')

        j_mobile = J_mobile(self.links, self.joint_pose[0], self.joint_pose[1], self.yaw)
        j_mobile_inv = JMoore(j_mobile)

        pd_des = self.get_p_dot_des(desired_pos)
        pd_des += self._compute_target_avoidance_velocity(current_pos)
        q_task = j_mobile_inv @ pd_des

        v = self.get_v()
        q_null = (np.eye(6) - j_mobile_inv @ j_mobile) @ v

        if self.phase == 'done':
            q_dot = np.zeros((6, 1))
            if self.start_pose is not None:
                self.setpoint_pose = self.start_pose.copy()
        else:
            q_dot = q_task + q_null

        now = self.get_clock().now()
        dt = (now - self.last_time).nanoseconds * 1e-9
        if dt <= 0:
            dt = 1e-3
        self.last_time = now

        q_dot = self._rate_limit(q_dot, dt)
        q_dot[0:3, 0] = np.clip(q_dot[0:3, 0], -self.max_linear_vel, self.max_linear_vel)
        q_dot[3, 0] = np.clip(q_dot[3, 0], -self.max_angular_vel, self.max_angular_vel)
        q_dot[4:6, 0] = np.clip(q_dot[4:6, 0], -self.max_joint_vel, self.max_joint_vel)
        self.last_q_dot = q_dot.copy()

        if self.setpoint_pose is None:
            self.setpoint_pose = np.array(self.drone_pose)
            self.setpoint_yaw = self.yaw

        self.setpoint_pose[0] += q_dot[0][0] * dt
        self.setpoint_pose[1] += q_dot[1][0] * dt
        self.setpoint_pose[2] += q_dot[2][0] * dt
        self.setpoint_yaw = self._wrap_angle(self.setpoint_yaw + q_dot[3][0] * dt)

        self.setpoint_pose = self._clamp_to_optitrack_bounds(self.setpoint_pose)

        cmd = PoseStamped()
        cmd.header.stamp = self.get_clock().now().to_msg()
        if self.command_frame is not None:
            cmd.header.frame_id = self.command_frame
        cmd.pose.position.x = float(self.setpoint_pose[0])
        cmd.pose.position.y = float(self.setpoint_pose[1])
        cmd.pose.position.z = float(max(self.setpoint_pose[2], self.min_z))

        q = quaternion_from_euler(0, 0, self.setpoint_yaw)
        cmd.pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
        self.drone_pos_pub.publish(cmd)

        self._compute_gimbal_angles()
        cmd_gimbal = Float64MultiArray()
        if self.use_degrees_for_gimbal:
            cmd_gimbal.data = [math.degrees(self.gimbal_pitch), math.degrees(self.gimbal_roll)]
        else:
            cmd_gimbal.data = [self.gimbal_pitch, self.gimbal_roll]
        self.gimbal_pub.publish(cmd_gimbal)

        t0 = np.clip(self.joint_pose[0] + q_dot[4][0] * dt, self.joint_min[0], self.joint_max[0])
        t1 = np.clip(self.joint_pose[1] + q_dot[5][0] * dt, self.joint_min[1], self.joint_max[1])
        cmd_arm = Float64MultiArray()
        cmd_arm.data = [float(t0), float(t1)]
        self.arm_pub.publish(cmd_arm)

        self._maybe_finish_goal()

    def execute_callback(self, goal_handle):
        target_index = goal_handle.request.target_index

        if target_index < 0 or target_index >= len(self.targets):
            goal_handle.abort()
            result = SwitchTarget.Result()
            result.success = False
            result.message = f'Invalid target index {target_index}. Must be 0-3.'
            return result

        if not self.targets_received[target_index]:
            goal_handle.abort()
            result = SwitchTarget.Result()
            result.success = False
            result.message = f'Target {target_index} not received yet.'
            return result

        if not self._all_required_state_ready():
            goal_handle.abort()
            result = SwitchTarget.Result()
            result.success = False
            result.message = 'Robot state not ready yet.'
            return result

        if self.current_goal_handle is not None:
            goal_handle.abort()
            result = SwitchTarget.Result()
            result.success = False
            result.message = 'Another SwitchTarget goal is already active.'
            return result

        self.current_goal_handle = goal_handle
        self.current_goal_target_index = target_index
        self.target_index = target_index
        self.phase = 'goto'
        self.phase_start_time = time.time()
        self.hover_start_time = None
        self.tap_start_time = None
        self.last_hover_log_second = -1
        self.get_logger().info(f'Switching to target {target_index + 1}')

        while rclpy.ok():
            if self.current_goal_handle is None:
                result = SwitchTarget.Result()
                if goal_handle.status == goal_handle.STATUS_SUCCEEDED:
                    result.success = True
                    result.message = f'Successfully switched to target {target_index + 1}'
                elif goal_handle.status == goal_handle.STATUS_CANCELED:
                    result.success = False
                    result.message = 'Goal canceled'
                else:
                    result.success = False
                    result.message = 'Goal ended without success'
                return result
            time.sleep(0.05)

        result = SwitchTarget.Result()
        result.success = False
        result.message = 'ROS shutdown before goal completion'
        return result


def main(args=None):
    rclpy.init(args=args)
    node = MobileJacobian()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()