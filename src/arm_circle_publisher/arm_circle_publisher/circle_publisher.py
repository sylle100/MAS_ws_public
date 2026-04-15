#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray


class ArmCirclePublisher(Node):

    def __init__(self):
        super().__init__('arm_circle_publisher')

        # Parameters (motion)
        self.declare_parameter('radius', 0.5)         # radians amplitude
        self.declare_parameter('frequency', 0.2)      # Hz
        self.declare_parameter('center', [0.0, 0.15])  # joint offsets [θ1_0, θ2_0]
        self.declare_parameter('rate', 50.0)

        # Parameters (forward kinematics)
        self.declare_parameter('l1', 0.15)  # link1 length
        self.declare_parameter('l2', 0.1383)  # link2 length
        self.declare_parameter('fk_log_hz', 5.0)  # how often to log FK

        self.radius = float(self.get_parameter('radius').value)
        self.frequency = float(self.get_parameter('frequency').value)
        self.center = self.get_parameter('center').value
        self.rate = float(self.get_parameter('rate').value)

        self.l1 = float(self.get_parameter('l1').value)
        self.l2 = float(self.get_parameter('l2').value)
        self.fk_log_hz = float(self.get_parameter('fk_log_hz').value)

        # Command publisher
        self.publisher_ = self.create_publisher(
            Float64MultiArray,
            '/arm_controller/commands',
            10
        )

        # Optional FK publisher (x,y)
        self.fk_pub_ = self.create_publisher(
            Float64MultiArray,
            '/arm_fk',
            10
        )

        self.t = 0.0
        self.dt = 1.0 / self.rate

        self._last_fk_log_t = 0.0

        self.timer = self.create_timer(self.dt, self.timer_callback)

        self.get_logger().info("Publishing circular joint motion to /arm_controller/commands")
        self.get_logger().info(f"FK enabled (2-link planar): l1={self.l1}, l2={self.l2}, publishing /arm_fk as [x,y]")

    def forward_kinematics(self, theta1: float, theta2: float):
        """
        2-link planar FK:
          x = l1*cos(t1) + l2*cos(t1 + t2)
          y = l1*sin(t1) + l2*sin(t1 + t2)
        theta2 is assumed to be relative elbow angle.
        """
        x = self.l1 * math.cos(theta1) + self.l2 * math.cos(theta1 + theta2)
        y = self.l1 * math.sin(theta1) + self.l2 * math.sin(theta1 + theta2)
        return x, y

    def timer_callback(self):
        w = 2.0 * math.pi * self.frequency

        joint1 = float(self.center[0]) + self.radius * math.cos(w * self.t)
        joint2 = float(self.center[1]) + self.radius * math.sin(w * self.t)

        # Publish joint commands
        cmd = Float64MultiArray()
        cmd.data = [joint1, joint2]
        self.publisher_.publish(cmd)

        # Compute + publish FK
        x, y = self.forward_kinematics(joint1, joint2)

        fk_msg = Float64MultiArray()
        fk_msg.data = [x, y]
        self.fk_pub_.publish(fk_msg)

        # Throttled FK logging
        if self.fk_log_hz > 0.0:
            log_period = 1.0 / self.fk_log_hz
            if (self.t - self._last_fk_log_t) >= log_period:
                self.get_logger().info(
                    f"t={self.t:.2f}  joints=[{joint1:.3f}, {joint2:.3f}]  FK(x,y)=[{x:.3f}, {y:.3f}]"
                )
                self._last_fk_log_t = self.t

        self.t += self.dt


def main():
    rclpy.init()
    node = ArmCirclePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()