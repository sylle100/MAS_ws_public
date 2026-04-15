#! /usr/bin/env python3
import rclpy
from rclpy.node import Node

from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import Imu
# from tf_transformations import euler_from_quaternion

from rclpy.qos import qos_profile_sensor_data

import numpy as np
import math

class GimbalPublisher(Node):

    def __init__(self):
        super().__init__('gimbal_publisher')
        self.publisher_ = self.create_publisher(
            Float64MultiArray, 
            '/gimbal_controller/commands', 
            10
        )

        self.subscriber_ = self.create_subscription(
            Imu,
            '/mavros/imu/data',
            self.imu_callback,
            qos_profile_sensor_data
        )

        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        #msg = Float64MultiArray()
        #msg.data = [5.0, 7.2]
        #self.publisher_.publish(msg)
        # self.get_logger().info('Publishing: "%s"' % msg.data)
        self.i += 1
    
    def imu_callback(self, msg):
        # Extract quaternion
        qx = msg.orientation.x
        qy = msg.orientation.y
        qz = msg.orientation.z
        qw = msg.orientation.w

        # Roll (x-axis rotation)
        sinr_cosp = 2.0 * (qw * qx + qy * qz)
        cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2.0 * (qw * qy - qz * qx)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)  # use 90 degrees if out of range
        else:
            pitch = math.asin(sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        # Convert to degrees (optional)
        roll_deg = math.degrees(roll)
        pitch_deg = math.degrees(pitch)
        yaw_deg = math.degrees(yaw)

        # self.get_logger().info(f"Roll: {roll_deg:.2f}°")
        # self.get_logger().info(f"Pitch: {pitch_deg:.2f}°")
        # self.get_logger().info(f"Yaw: {yaw_deg:.2f}°")
        msg = Float64MultiArray()
        msg.data = [-pitch_deg, -roll_deg]
        print(f"Publishing: {msg.data}")
        self.publisher_.publish(msg)
        


def main(args=None):
    rclpy.init(args=args)

    gimbal_publisher = GimbalPublisher()

    rclpy.spin(gimbal_publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    gimbal_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()


# self.get_logger().info('Done wiggling.')