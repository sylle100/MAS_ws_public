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

        self.gimbal_pub = self.create_publisher(
            Float64MultiArray,
            '/gimbal_controller/commands',
            10)
        


        self.drone_pose = [0,0,0]
        self.setpoint_pose = None
        self.yaw = 0
        self.setpoint_yaw = 0
        self.joint_pose = [0,0]
        self.gimbal_pitch = 0.0  # radians, positive = tilt up
        self.gimbal_roll = 0.0   # radians, positive = tilt right

        # record the starting drone position so we can return to it
        self.start_pose = None

        # Control parameters
        self.links = np.array([0.15, 0.15])
        self.k = 0.3

        # Safety limits
        self.min_z = 0.1  # minimum allowed altitude for commanded setpoint (meters)

        # null space - separate positions for start/goto, hover, and tap
        self.pose_des_goto = [0,0,0,0,np.pi - (8*np.pi)/9, np.pi/3]  # start: link1 down, link2 150 deg up
        self.pose_des_hover = [0,0,0,0,(8*np.pi)/9,np.pi/3]  # hover: link1 30 deg, link2 30 deg
        #self.pose_des_tap = [0,0,0,0,-np.pi/3,0]  # tap: link1 down further, link2 straight (90 deg down when link1 is vertical)
        self.pose_k   = [0,0,0,0,1,1]

        # mission phases: 'goto', 'hover', 'tap', 'return', 'done'
        self.phase = 'goto'
        self.phase_start_time = time.time()
        self.hover_start_time = None
        self.tap_start_time = None

        # Target sequencing
        self.target_index = 0

        self.goto_timeout = 60.0  # max time to spend in goto phase before force-switching (seconds)
        self.hover_height = 0.2  # meters above target
        self.approach_height = 0.2  # meters above hover height for overhead approach
        self.descend_radius_xy = 0.08  # only descend when nearly centered above the target
        self.target_avoid_radius = 0.18  # meters; protected bubble around each target
        self.target_avoid_influence = 0.35  # meters; start bending away before reaching the bubble
        self.target_avoid_gain = 0.25  # m/s of repulsive task-space velocity at the bubble edge
        self.hover_duration = 2.0  # seconds
        self.tap_depth = 0.15  # meters to lower end effector for tapping
        self.tap_duration = 1.0  # seconds to spend tapping
        self.arrival_threshold = 0.05  # meters, to consider "arrived" at a point (for phase switching)

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

    def _compute_desired_position(self) -> np.ndarray:
        """Compute the desired drone position for the current mission phase."""

        # Default to start pose until we have a true starting pose.
        if self.start_pose is None:
            return np.array(self.drone_pose)

        if self.phase in ('goto', 'hover', 'tap'):
            # Skip targets that haven't been received yet.
            while self.target_index < len(self.targets) and not self.targets_received[self.target_index]:
                self.get_logger().info(f"Skipping target {self.target_index + 1}: no data received")
                self.target_index += 1

            if self.target_index >= len(self.targets):
                return self.start_pose

            target = self.targets[self.target_index].copy()
            hover_pos = target.copy()
            hover_pos[2] = max(target[2] + self.hover_height, self.min_z)

            if self.phase == 'hover':
                self.get_logger().info(f"HOVER: Target={target}, hover_pos_Z={hover_pos[2]:.3f}, EE_Z={self.end_effector_pose.position.z:.3f}")
                return hover_pos
            
            if self.phase == 'tap':
                # During tap, lower the end effector to actually contact the target
                # tap_depth is how much to lower BELOW hover height
                tap_pos = hover_pos.copy()
                tap_pos[2] = hover_pos[2] - self.tap_depth  # Lower from hover height
                tap_pos[2] = max(tap_pos[2], self.min_z)  # But don't go below minimum altitude
                self.get_logger().warn(f"TAP DESIRED: hover_Z={hover_pos[2]:.3f}, tap_pos_Z={tap_pos[2]:.3f}, EE_Z={self.end_effector_pose.position.z:.3f}")
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

            # First move above the target, then descend vertically once centered.
            if horizontal_dist > self.descend_radius_xy or not overhead_ready:
                return overhead_pos
            return hover_pos

        if self.phase == 'return':
            return self.start_pose

        # done
        return self.start_pose

    def _compute_target_avoidance_velocity(self, current_pos: np.ndarray) -> np.ndarray:
        """Generate a repulsive task-space velocity around targets and upward near them."""
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

            active_target = idx == self.target_index and self.phase in ('goto', 'hover', 'tap')
            if active_target and current_pos[2] >= safe_overhead_z - self.arrival_threshold:
                # Once we are safely overhead, allow the vertical approach.
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
            
            # Compute angle to target
            delta = target - current_pos
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

        if self.phase == 'goto' and dist < self.arrival_threshold:
            self.phase = 'hover'
            self.hover_start_time = time.time()
            self.get_logger().info(f'Reached target; EE_pos={current_pos}, desired_pos={desired_pos}, dist={dist:.3f}; hovering for %.1fs' % self.hover_duration)
        
        # Fallback: if in goto phase too long, force go to hover anyway
        if self.phase == 'goto':
            goto_elapsed = time.time() - self.phase_start_time
            if goto_elapsed >= self.goto_timeout:
                self.get_logger().warn(f"Goto timeout ({goto_elapsed:.1f}s) reached! Forcing hover phase.")
                self.phase = 'hover'
                self.hover_start_time = time.time()

        if self.phase == 'hover':
            if self.hover_start_time is not None:
                elapsed = time.time() - self.hover_start_time
                if elapsed >= self.hover_duration:
                    self.get_logger().warn(f"HOVER COMPLETE! Transitioning to TAP phase. (elapsed={elapsed:.2f}s)")
                    self.phase = 'tap'
                    self.tap_start_time = time.time()
                elif int(elapsed) % 1 == 0:  # Log every 1 second
                    self.get_logger().info(f"HOVER phase: elapsed={elapsed:.2f}s / {self.hover_duration}s, dist_to_target={dist:.3f}m")
            else:
                self.get_logger().warn("HOVER phase but hover_start_time is None!")

        if self.phase == 'tap':
            if self.tap_start_time is not None:
                elapsed = time.time() - self.tap_start_time
                self.get_logger().warn(f"TAP phase: elapsed={elapsed:.2f}s / {self.tap_duration}s, EE_Z={current_pos[2]:.3f}m, desired_Z={desired_pos[2]:.3f}m")
                if elapsed >= self.tap_duration:
                    self.get_logger().warn(f"TAP COMPLETE! Moving to next target.")
                    # Move to the next target, or return home if done.
                    self.target_index += 1
                    while self.target_index < len(self.targets) and not self.targets_received[self.target_index]:
                        self.get_logger().info(f"Skipping target {self.target_index + 1}: no data received")
                        self.target_index += 1

                    if self.target_index < len(self.targets):
                        self.phase = 'goto'
                        self.phase_start_time = time.time()
                        self.get_logger().info(f"Going to target {self.target_index + 1}")
                    else:
                        self.phase = 'return'
                        self.phase_start_time = time.time()
                        self.get_logger().info('All targets tapped; returning to start')
            else:
                self.get_logger().warn("TAP phase but tap_start_time is None!")

        if self.phase == 'return' and dist < self.arrival_threshold:
            self.phase = 'done'
            self.get_logger().info('Returned to start; mission done')

        # Compute Jacobian and inverse once per cycle.
        j_mobile = J_mobile(self.links, self.joint_pose[0], self.joint_pose[1], self.yaw)
        j_mobile_inv = JMoore(j_mobile)

        # q task
        pd_des = self.get_p_dot_des(desired_pos)
        pd_des += self._compute_target_avoidance_velocity(current_pos)
        q_task = j_mobile_inv @ pd_des

        # q Null
        v = self.get_v()
        q_null = (np.eye(6) - j_mobile_inv @ j_mobile) @ v

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
            self.setpoint_pose = np.array(self.drone_pose)
            self.setpoint_yaw = self.yaw

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
        cmd.pose.position.x = self.setpoint_pose[0]
        cmd.pose.position.y = self.setpoint_pose[1]
        cmd.pose.position.z = max(self.setpoint_pose[2], self.min_z)


        Q = quaternion_from_euler(0,0,self.setpoint_yaw)
        
        cmd.pose.orientation = Quaternion(x = Q[0], y = Q[1], z = Q[2], w = Q[3])
        self.drone_pos_pub.publish(cmd)

        # Compute and publish gimbal commands
        self._compute_gimbal_angles()
        cmd_gimbal = Float64MultiArray()
        cmd_gimbal.data = [-self.gimbal_pitch, -self.gimbal_roll]
        self.gimbal_pub.publish(cmd_gimbal)

        # update arm joint position based on velosity
        t0 = self.joint_pose[0] + q_dot[4][0] * dt
        t1 = self.joint_pose[1] + q_dot[5][0] * dt
        
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
