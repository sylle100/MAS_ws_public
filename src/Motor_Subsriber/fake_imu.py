#! /usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu

import math

class FakeImuPublisher(Node):

    def __init__(self):
        super().__init__('fake_imu_publisher')

        self.publisher_ = self.create_publisher(
            Imu,
            '/mavros/imu/data',
            10
        )

        self.timer = self.create_timer(0.05, self.timer_callback)  # 20 Hz

        self.t = 0.0

        # Parameters
        self.amplitude_deg = 30.0      # ±30 degrees
        self.frequency = 0.05          # Hz (slow motion)

    def timer_callback(self):
        msg = Imu()

        # Time-based oscillations
        pitch_deg = self.amplitude_deg * math.sin(2 * math.pi * self.frequency * self.t)
        roll_deg = 10.0 * math.sin(2 * math.pi * self.frequency * self.t)  # ±10°

        pitch = math.radians(pitch_deg)
        roll = math.radians(roll_deg)
        yaw = 0.0  # keep yaw constant

        # Convert roll, pitch, yaw → quaternion
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)

        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy

        msg.orientation.x = qx
        msg.orientation.y = qy
        msg.orientation.z = qz
        msg.orientation.w = qw

        self.publisher_.publish(msg)

        self.t += 0.05

def main(args=None):
    rclpy.init(args=args)

    node = FakeImuPublisher()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
