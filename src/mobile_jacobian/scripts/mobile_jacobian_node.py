#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer
from sensor_msgs.msg import JointState, Imu
from std_msgs.msg import Float64MultiArray
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, Quaternion, Vector3
from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import State
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

        self.mavros_state_sub = self.create_subscription(
            State,
            "/mavros/state",
            self.mavros_state_callback,
            10,
        )
        self.mavros_state = State()

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

        self.gimbal_pub = self.create_publisher(
            Float64MultiArray,
            '/gimbal_controller/commands',
            10)
        


        self.drone_pose = [0.0, 0.0, 0.0]
        self.setpoint_pose = None
        self.yaw = 0
        self.setpoint_yaw = 0
        self.joint_pose = [0.0, 0.0]
        self.gimbal_pitch = 0.0  # radians, positive = tilt up
        self.gimbal_roll = 0.0   # radians, positive = tilt right

        # record the starting drone position so we can return to it
        self.start_pose = None
        self.start_ee_pose = None

        # Control parameters
        self.links = np.array([0.15, 0.15])
        self.k = 0.3

        # Safety limits
        self.min_z = 0.1  # minimum allowed altitude for commanded setpoint (meters)

        # null space - separate positions for start/goto, hover, and tap
        self.pose_des_goto = [0,0,0,0,np.pi - (8*np.pi)/9, np.pi/3]  # start: link1 down, link2 150 deg up
        self.pose_des_hover = [0,0,0,0,(8*np.pi)/9,np.pi/3]  # hover: link1 30 deg, link2 30 deg
        self.pose_des_tap = [0,0,0,0,-np.pi/3,0]  # tap: link1 down further, link2 straight (90 deg down when link1 is vertical)
        self.pose_k   = [0,0,0,0,1,1]

        # mission phases: 'takeoff', 'goto', 'hover', 'tap', 'return', 'hold', 'done'
        self.phase = 'takeoff'
        self.phase_start_time = time.time()
        self.hover_start_time = None
        self.tap_start_time = None
        self.tap_descent_start_time = None
        self.hold_position = None

        # Target sequencing
        self.target_index = 0

        self.goto_timeout = 0.0  # disabled; keep pursuing target until truly above it
        self.hover_height = 0.05  # meters above target
        self.approach_height = 0.2  # meters above hover height for overhead approach
        self.descend_radius_xy = 0.12  # only descend when nearly centered above the target
        self.target_avoid_radius = 0.20  # meters; keep-out bubble around non-active targets
        self.target_avoid_influence = 0.50  # meters; start bending around non-active targets
        self.target_avoid_gain = 0.20  # m/s repulsive task-space velocity at keep-out bubble edge
        self.hover_duration = 2.0  # seconds
        self.takeoff_altitude = 1.5  # meters in world frame before lateral motion
        self.takeoff_z_threshold = 0.10  # meters
        self.target_top_offset = -0.05  # meters from target pose to the top of the pole
        self.tap_duration = 2.0  # seconds to spend tapping
        self.tap_max_duration = 6.0  # upper bound for total tap phase before skipping target
        self.arrival_threshold = 0.10  # meters, to consider "arrived" at a point (for phase switching)
        self.hover_xy_threshold = 0.20  # XY proximity to count as directly over target for hover/tap
        self.tap_xy_threshold = 0.25  # reserved threshold for tap diagnostics
        self.hover_z_threshold = 0.20  # Z proximity to hover plane before starting hover timer

        # Optitrack area boundaries (from optitrack.world) with safety margins
        self.optitrack_x_min = -5.8
        self.optitrack_x_max = 5.8
        self.optitrack_y_min = -5.8
        self.optitrack_y_max = 5.8
        self.optitrack_z_min = 0.1   # min altitude
        self.optitrack_z_max = 5.5   # max altitude (below ceiling)

        # For integration of velocity commands
        self.last_time = self.get_clock().now()

        # Limits for commanded velocities
        self.max_linear_vel = 0.3  # m/s
        self.max_angular_vel = 0.6  # rad/s
        self.max_joint_vel = 0.6  # rad/s
        self.max_linear_acc = 0.5  # m/s^2
        self.max_angular_acc = 0.8  # rad/s^2
        self.max_joint_acc = 1.0  # rad/s^2
        self.approach_smoothing = 0.25  # meters; larger values soften long moves
        self.last_q_dot = np.zeros((6, 1))
        self.last_goto_log_time = 0.0
        self.last_wait_offboard_log_time = 0.0

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
        self.joint_pose = [float(t0), float(t1)]

    def end_effector_callback(self, msg: Odometry):
        self.end_effector_pose = msg.pose.pose
        self.end_effector_twist = msg.twist.twist
        if self.start_ee_pose is None:
            self.start_ee_pose = np.array([
                self.end_effector_pose.position.x,
                self.end_effector_pose.position.y,
                self.end_effector_pose.position.z,
            ])
            self.get_logger().info(f"Recorded start end-effector pose: {self.start_ee_pose}")

    def imu_callback(self, msg: Imu):
        quat = [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]
        self.roll, self.pitch, self.yaw = euler_from_quaternion(quat)

    def mavros_state_callback(self, msg: State):
        self.mavros_state = msg

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

        # Use drone world position for guidance; end-effector odometry can be in a different frame.
        x = self.drone_pose[0]
        y = self.drone_pose[1]
        z = self.drone_pose[2]

        error = np.array([
            [desired_pos[0] - x],
            [desired_pos[1] - y],
            [desired_pos[2] - z],
        ])

        error_norm = np.linalg.norm(error)
        if error_norm < 1e-6:
            return np.zeros((3, 1))

        # Use a smooth saturating profile so large target jumps do not create a lunge.
        speed_scale = np.tanh(error_norm / self.approach_smoothing)
        return error * (self.k * speed_scale)

    def _rate_limit(self, q_dot: np.ndarray, dt: float) -> np.ndarray:
        """Limit acceleration so command changes ramp in smoothly."""
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
        """Clamp position to stay within optitrack area boundaries."""
        pos_clamped = pos.copy()
        pos_clamped[0] = np.clip(pos_clamped[0], self.optitrack_x_min, self.optitrack_x_max)
        pos_clamped[1] = np.clip(pos_clamped[1], self.optitrack_y_min, self.optitrack_y_max)
        pos_clamped[2] = np.clip(pos_clamped[2], self.optitrack_z_min, self.optitrack_z_max)
        return pos_clamped

    def _target_top_z(self, target: np.ndarray) -> float:
        """Return the Z coordinate of the top of the pole target."""
        return target[2] + self.target_top_offset

    def _compute_desired_position(self) -> np.ndarray:
        """Compute the desired drone position for the current mission phase."""

        # Default to start pose until we have a true starting pose.
        if self.start_pose is None:
            return np.array(self.drone_pose)

        if self.phase == 'takeoff':
            takeoff_pos = self.start_pose.copy()
            takeoff_pos[2] = max(self.takeoff_altitude, self.min_z)
            return takeoff_pos

        if self.phase in ('goto', 'hover', 'tap'):
            # Skip targets that haven't been received yet.
            while self.target_index < len(self.targets) and not self.targets_received[self.target_index]:
                self.get_logger().info(f"Skipping target {self.target_index + 1}: no data received")
                self.target_index += 1

            if self.target_index >= len(self.targets):
                return self.start_pose

            target = self.targets[self.target_index].copy()
            hover_pos = target.copy()
            hover_pos[2] = max(self._target_top_z(target) + self.hover_height, self.min_z)

            if self.phase == 'hover':
                self.get_logger().info(f"HOVER: Target={target}, hover_pos_Z={hover_pos[2]:.3f}, EE_Z={self.end_effector_pose.position.z:.3f}")
                return hover_pos
            
            if self.phase == 'tap':
                # During tap, lower the end effector smoothly from hover to the pole top.
                tap_pos = hover_pos.copy()
                tap_pos[2] = max(self._target_top_z(target), self.min_z)
                if self.tap_start_time is None:
                    return hover_pos

                current_pos = np.array([
                    self.end_effector_pose.position.x,
                    self.end_effector_pose.position.y,
                    self.end_effector_pose.position.z,
                ])
                xy_error = np.linalg.norm(current_pos[:2] - target[:2])

                # Do not descend until we are centered over the active target.
                if xy_error > self.tap_xy_threshold:
                    self.get_logger().info(
                        f"TAP HOLD: waiting for XY centering before descent (xy_error={xy_error:.3f})"
                    )
                    return hover_pos

                if self.tap_descent_start_time is None:
                    self.tap_descent_start_time = time.time()

                elapsed = time.time() - self.tap_descent_start_time
                fraction = min(1.0, max(0.0, elapsed / self.tap_duration))
                desired_pos = hover_pos.copy()
                desired_pos[2] = hover_pos[2] + (tap_pos[2] - hover_pos[2]) * fraction
                self.get_logger().warn(
                    f"TAP DESIRED: fraction={fraction:.2f}, hover_Z={hover_pos[2]:.3f}, tap_pos_Z={tap_pos[2]:.3f}, target_top_Z={self._target_top_z(target):.3f}, EE_Z={self.end_effector_pose.position.z:.3f}"
                )
                return desired_pos

            current_pos = np.array(self.drone_pose)
            overhead_pos = hover_pos.copy()
            overhead_pos[2] += self.approach_height

            horizontal_dist = np.linalg.norm(current_pos[:2] - target[:2])
            overhead_ready = current_pos[2] >= (overhead_pos[2] - self.arrival_threshold)

            # First move above the target, then descend vertically once centered.
            if horizontal_dist > self.descend_radius_xy or not overhead_ready:
                return overhead_pos
            return hover_pos

        if self.phase == 'return':
            if self.start_ee_pose is not None:
                return self.start_ee_pose.copy()
            return self.start_pose

        # done
        if self.start_ee_pose is not None:
            return self.start_ee_pose.copy()
        return self.start_pose

    def _compute_target_avoidance_velocity(self, current_pos: np.ndarray) -> np.ndarray:
        """Generate a repulsive task-space velocity around targets and upward near them."""
        # Keep precision phases strictly locked to the active target position.
        if self.phase in ('takeoff', 'hover', 'tap'):
            return np.zeros((3, 1))

        avoid = np.zeros((3, 1))

        for idx, target_received in enumerate(self.targets_received):
            if not target_received:
                continue

            target = self.targets[idx].copy()
            safe_hover_z = max(self._target_top_z(target) + self.hover_height, self.min_z)
            safe_overhead_z = safe_hover_z + self.approach_height

            xy_delta = current_pos[:2] - target[:2]
            xy_dist = np.linalg.norm(xy_delta)

            active_target = idx == self.target_index and self.phase in ('goto', 'hover', 'tap')

            # For the active target: only enforce vertical clearance while approaching.
            # Horizontal repulsion here can fight the approach and cause long detours/timeouts.
            if active_target:
                if (
                    self.phase == 'goto'
                    and xy_dist < self.descend_radius_xy
                    and current_pos[2] < safe_overhead_z - self.arrival_threshold
                ):
                    upward_ratio = max(0.0, safe_overhead_z - current_pos[2]) / max(self.approach_height, 1e-6)
                    avoid[2, 0] += self.target_avoid_gain * min(1.0, upward_ratio)
                continue

            if xy_dist >= self.target_avoid_influence:
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
        """Compute gimbal pitch and roll based on mission phase.
        
        During tap phase, point gimbal down at the target.
        Otherwise, keep gimbal level.
        """
        if self.phase == 'tap' and self.target_index < len(self.targets):
            # Point gimbal down at the target during tap
            current_pos = np.array([
                self.end_effector_pose.position.x,
                self.end_effector_pose.position.y,
                self.end_effector_pose.position.z,
            ])
            target = self.targets[self.target_index]
            target_top = target.copy()
            target_top[2] = self._target_top_z(target)
            
            # Compute angle to target
            delta = target_top - current_pos
            distance = np.linalg.norm(delta)
            
            if distance > 0.01:
                # Pitch: look down at target (negative pitch = look down)
                self.gimbal_pitch = -np.arctan2(delta[2], np.sqrt(delta[0]**2 + delta[1]**2))
                # Roll: keep level
                self.gimbal_roll = 0.0
            else:
                # Very close to target, point straight down
                self.gimbal_pitch = -np.pi / 2  # -90 degrees
                self.gimbal_roll = 0.0
        else:
            # Keep gimbal level during other phases
            self.gimbal_pitch = 0.0
            self.gimbal_roll = 0.0

    def get_v(self):
        """Desired null space"""
        (t0,t1) = self.joint_pose
        
        # Select arm position based on mission phase
        if self.phase == 'hover':
            pose_des = self.pose_des_hover
        elif self.phase == 'tap':
            pose_des = self.pose_des_tap
        else:
            pose_des = self.pose_des_goto
        
        v = np.array([
            [self.pose_k[0] * 0],
            [self.pose_k[1] * 0],
            [self.pose_k[2] * 0],
            [self.pose_k[3] * 0],
            [self.pose_k[4] * (pose_des[4] - t0)],
            [self.pose_k[5] * (pose_des[5] - t1)]
            ])
        return v

    def _publish_hold_setpoints(self):
        """Publish neutral hold commands while waiting for OFFBOARD engagement."""
        if self.setpoint_pose is None:
            self.setpoint_pose = np.array(self.drone_pose, dtype=float)
            self.setpoint_yaw = float(self.yaw)

        cmd = PoseStamped()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.header.frame_id = 'world'
        cmd.pose.position.x = float(self.setpoint_pose[0])
        cmd.pose.position.y = float(self.setpoint_pose[1])
        cmd.pose.position.z = float(max(self.setpoint_pose[2], self.min_z))
        q = quaternion_from_euler(0, 0, self.setpoint_yaw)
        cmd.pose.orientation = Quaternion(x=float(q[0]), y=float(q[1]), z=float(q[2]), w=float(q[3]))
        self.drone_pos_pub.publish(cmd)

        cmd_arm = Float64MultiArray()
        cmd_arm.data = [float(self.joint_pose[0]), float(self.joint_pose[1])]
        self.arm_pub.publish(cmd_arm)

    def timer_callback(self):
        """Calculates, q, the desired velocity to reach target and executes mission phases."""

        # Wait until we have a defined start pose.
        if self.start_pose is None:
            self._publish_hold_setpoints()
            return

        # Only start mission motion after operator explicitly engages OFFBOARD and arms.
        if not (self.mavros_state.mode == 'OFFBOARD' and self.mavros_state.armed):
            self._publish_hold_setpoints()
            now_wall = time.time()
            if now_wall - self.last_wait_offboard_log_time >= 1.0:
                self.last_wait_offboard_log_time = now_wall
                self.get_logger().info(
                    f"Waiting for OFFBOARD+ARMED (mode={self.mavros_state.mode}, armed={self.mavros_state.armed})"
                )
            return

        # If we haven't received any targets yet, hold position until the first target arrives.
        if not any(self.targets_received):
            self._publish_hold_setpoints()
            return

        # Determine desired position based on mission phase.
        desired_pos = self._compute_desired_position()

        # Phase transitions in world frame based on drone base position.
        current_pos = np.array(self.drone_pose)
        dist = np.linalg.norm(current_pos - desired_pos)

        target_xy_dist = float('inf')
        hover_z_error = float('inf')
        if self.target_index < len(self.targets) and self.targets_received[self.target_index]:
            active_target = self.targets[self.target_index].copy()
            hover_target_z = max(self._target_top_z(active_target) + self.hover_height, self.min_z)
            target_xy_dist = np.linalg.norm(current_pos[:2] - active_target[:2])
            hover_z_error = abs(current_pos[2] - hover_target_z)

        if (
            self.phase == 'goto'
            and target_xy_dist <= self.hover_xy_threshold
            and hover_z_error <= self.hover_z_threshold
        ):
            self.phase = 'hover'
            self.hover_start_time = time.time()
            self.get_logger().info(
                f'Reached hover-over-target; EE_pos={current_pos}, target_xy_dist={target_xy_dist:.3f}, '
                f'hover_z_error={hover_z_error:.3f}; hovering for {self.hover_duration:.1f}s'
            )

        if self.phase == 'takeoff':
            takeoff_error = abs(current_pos[2] - max(self.takeoff_altitude, self.min_z))
            if takeoff_error <= self.takeoff_z_threshold:
                self.phase = 'goto'
                self.phase_start_time = time.time()
                self.get_logger().info(
                    f"Takeoff complete (z_error={takeoff_error:.3f}); starting target navigation"
                )
        
        # Optional timeout notice only; controller stays in goto until centered over target.
        if self.phase == 'goto' and self.goto_timeout > 0.0:
            goto_elapsed = time.time() - self.phase_start_time
            if goto_elapsed >= self.goto_timeout:
                self.get_logger().warn(
                    f"Goto timeout ({goto_elapsed:.1f}s) reached; staying in goto. "
                    f"target_xy_dist={target_xy_dist:.3f}, hover_z_error={hover_z_error:.3f}"
                )
                self.phase_start_time = time.time()

        if self.phase == 'goto':
            now_wall = time.time()
            if now_wall - self.last_goto_log_time >= 1.0:
                self.last_goto_log_time = now_wall
                self.get_logger().info(
                    f"GOTO tracking target {self.target_index + 1}: target_xy_dist={target_xy_dist:.3f}, "
                    f"hover_z_error={hover_z_error:.3f}"
                )

        if self.phase == 'hover':
            if self.hover_start_time is not None:
                if target_xy_dist > self.hover_xy_threshold or hover_z_error > self.hover_z_threshold:
                    # Only count hover time while directly over the active target at the hover height.
                    self.hover_start_time = time.time()
                    self.get_logger().info(
                        f"HOVER realigning: target_xy_dist={target_xy_dist:.3f}, hover_z_error={hover_z_error:.3f}"
                    )
                    return

                elapsed = time.time() - self.hover_start_time
                if elapsed >= self.hover_duration:
                    self.get_logger().warn(f"HOVER COMPLETE! Transitioning to TAP phase. (elapsed={elapsed:.2f}s)")
                    self.phase = 'tap'
                    self.tap_start_time = time.time()
                    self.tap_descent_start_time = None
                elif int(elapsed) % 1 == 0:  # Log every 1 second
                    self.get_logger().info(f"HOVER phase: elapsed={elapsed:.2f}s / {self.hover_duration}s, dist_to_target={dist:.3f}m")
            else:
                self.get_logger().warn("HOVER phase but hover_start_time is None!")

        if self.phase == 'tap':
            if self.tap_start_time is not None:
                elapsed_total = time.time() - self.tap_start_time
                elapsed_descent = 0.0
                if self.tap_descent_start_time is not None:
                    elapsed_descent = time.time() - self.tap_descent_start_time

                self.get_logger().warn(
                    f"TAP phase: total_elapsed={elapsed_total:.2f}s, descent_elapsed={elapsed_descent:.2f}s / {self.tap_duration}s, "
                    f"EE_Z={current_pos[2]:.3f}m, desired_Z={desired_pos[2]:.3f}m"
                )

                tap_done = (
                    self.tap_descent_start_time is not None
                    and elapsed_descent >= self.tap_duration
                )
                tap_timed_out = elapsed_total >= self.tap_max_duration

                if tap_done or tap_timed_out:
                    if tap_timed_out and not tap_done:
                        self.get_logger().warn(
                            f"TAP timeout ({elapsed_total:.2f}s) before full descent; advancing to next target."
                        )
                    self.get_logger().warn(f"TAP COMPLETE! Moving to next target.")
                    # Move to the next target, or return home if done.
                    self.target_index += 1
                    self.tap_start_time = None
                    self.tap_descent_start_time = None
                    while self.target_index < len(self.targets) and not self.targets_received[self.target_index]:
                        self.get_logger().info(f"Skipping target {self.target_index + 1}: no data received")
                        self.target_index += 1

                    if self.target_index < len(self.targets):
                        self.phase = 'goto'
                        self.phase_start_time = time.time()
                        self.get_logger().info(f"Going to target {self.target_index + 1}")
                    else:
                        self.phase = 'hold'
                        self.hold_position = np.array(self.drone_pose)
                        self.phase_start_time = time.time()
                        self.get_logger().info('All targets tapped; holding a safe hover over target 4')
            else:
                self.get_logger().warn("TAP phase but tap_start_time is None!")

        if self.phase == 'hold' and self.hold_position is not None:
            desired_pos = self.hold_position.copy()

        if self.phase == 'return' and dist < self.arrival_threshold:
            self.phase = 'done'
            self.get_logger().info('Returned to start; mission done')

        # Task-space control using measured end-effector pose in world frame.
        # This avoids model mismatch between analytic Jacobian geometry and the actual arm model.
        pd_goal = self.get_p_dot_des(desired_pos)
        pd_avoid = self._compute_target_avoidance_velocity(current_pos)

        # Keep avoidance from flipping motion away from the active goal in XY.
        goal_xy = pd_goal[0:2, 0]
        avoid_xy = pd_avoid[0:2, 0]
        goal_xy_norm = np.linalg.norm(goal_xy)
        if goal_xy_norm > 1e-6:
            goal_xy_dir = goal_xy / goal_xy_norm
            avoid_along_goal = float(np.dot(avoid_xy, goal_xy_dir))
            if avoid_along_goal < 0.0:
                # Remove only the component that fights goal progress.
                pd_avoid[0:2, 0] = avoid_xy - avoid_along_goal * goal_xy_dir

        # Also cap avoidance magnitude so it cannot dominate attraction.
        avoid_norm = np.linalg.norm(pd_avoid)
        goal_norm = np.linalg.norm(pd_goal)
        max_avoid = max(0.05, 0.6 * goal_norm)
        if avoid_norm > max_avoid:
            pd_avoid *= max_avoid / avoid_norm

        pd_des = pd_goal + pd_avoid
        q_task = np.zeros((6, 1))
        q_task[0:3, 0] = pd_des[:, 0]

        # q Null (joint posture shaping)
        v = self.get_v()
        q_null = v

        # q dot
        if self.phase == 'done':
            q_dot = np.zeros((6, 1))
        else:
            q_dot = q_task + q_null

        # Integrate velocity to get next setpoint using actual elapsed time
        now = self.get_clock().now()
        dt = (now - self.last_time).nanoseconds * 1e-9
        if dt <= 0:
            dt = 1e-3
        self.last_time = now

        q_dot = self._rate_limit(q_dot, dt)

        # Clamp commanded velocities to reasonable limits
        q_dot[0:3, 0] = np.clip(q_dot[0:3, 0], -self.max_linear_vel, self.max_linear_vel)
        q_dot[3, 0] = np.clip(q_dot[3, 0], -self.max_angular_vel, self.max_angular_vel)
        q_dot[4:6, 0] = np.clip(q_dot[4:6, 0], -self.max_joint_vel, self.max_joint_vel)
        self.last_q_dot = q_dot.copy()

        # update commanded setpoint based on computed velocity
        if self.setpoint_pose is None:
            self.setpoint_pose = np.array(self.drone_pose, dtype=float)
            self.setpoint_yaw = float(self.yaw)

        self.setpoint_pose[0] += q_dot[0][0] * dt
        self.setpoint_pose[1] += q_dot[1][0] * dt
        self.setpoint_pose[2] += q_dot[2][0] * dt
        self.setpoint_yaw += q_dot[3][0] * dt

        # Safety: clamp to optitrack bounds and don't command below the minimum allowed altitude
        self.setpoint_pose = self._clamp_to_optitrack_bounds(self.setpoint_pose)


        # velocity bases
        # cmd_lin = Vector3(x=q_dot[0][0],y=q_dot[1][0], z=q_dot[2][0])
        # cmd_ang = Vector3(x=0.0, y=0.0, z=q_dot[3][0])
        # cmd = Twist()
        # cmd.linear = cmd_lin
        # cmd.angular = cmd_ang
        # self.drone_vel_pub.publish(cmd)

        cmd = PoseStamped()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.header.frame_id = 'world'
        cmd.pose.position.x = float(self.setpoint_pose[0])
        cmd.pose.position.y = float(self.setpoint_pose[1])
        cmd.pose.position.z = float(max(self.setpoint_pose[2], self.min_z))


        Q = quaternion_from_euler(0,0,self.setpoint_yaw)
        
        cmd.pose.orientation = Quaternion(x=float(Q[0]), y=float(Q[1]), z=float(Q[2]), w=float(Q[3]))
        self.drone_pos_pub.publish(cmd)

        # Compute and publish gimbal commands
        self._compute_gimbal_angles()
        cmd_gimbal = Float64MultiArray()
        cmd_gimbal.data = [float(-self.gimbal_pitch), float(-self.gimbal_roll)]
        self.gimbal_pub.publish(cmd_gimbal)

        # update arm joint position based on velosity
        t0 = self.joint_pose[0] + q_dot[4][0] * dt
        t1 = self.joint_pose[1] + q_dot[5][0] * dt
        
        cmd_arm = Float64MultiArray()

        # transmit new target positions
        cmd_arm.data = [float(t0), float(t1)]
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
