#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from dynamixel_sdk import *
import math

# ---------------- DYNAMIXEL SETUP ----------------
TORQUE_ADDR = 64
GOAL_POSITION_ADDR = 116

# Motor mapping
GIMBAL_IDS = [1, 2]   # motor 1,2
ARM_IDS    = [3, 4]   # motor 3,4
ALL_IDS    = GIMBAL_IDS + ARM_IDS

portHandler = PortHandler("/dev/ttyUSB0")
packetHandler = PacketHandler(2.0)


# ---------------- HELPERS ----------------
def rad_to_deg(rad):
    return rad * 180.0 / math.pi


def deg_to_dxl(deg):
    return int(deg * 4095.0 / 360.0)


# ---------------- NODE ----------------
class CombinedDynamixelController(Node):

    def __init__(self):
        super().__init__('combined_dynamixel_controller')

        # Store positions
        self.gimbal_positions = [315.0, 315.0]
        self.arm_positions = [180.0, 180.0]

        # -------- SUBSCRIBERS --------

        # Gimbal (IMU)
        self.create_subscription(
            Float64MultiArray,
            '/gimbal_controller/commands',
            self.gimbal_callback,
            10
        )

        # Arm (IK)
        self.create_subscription(
            Float64MultiArray,
            '/arm_controller/commands',
            self.arm_callback,
            10
        )

        # Control loop
        self.timer = self.create_timer(0.02, self.control_loop)

        self.get_logger().info("Combined Dynamixel Controller Started")
        self.setup_dynamixel()

    # ---------------- SETUP ----------------
    def setup_dynamixel(self):
        if not portHandler.openPort():
            self.get_logger().error("Failed to open port")
            quit()

        if not portHandler.setBaudRate(57600):
            self.get_logger().error("Failed baudrate")
            quit()

        self.get_logger().info("Port OK")

        for dxl_id in ALL_IDS:
            packetHandler.write1ByteTxRx(portHandler, dxl_id, TORQUE_ADDR, 1)
            self.get_logger().info(f"Torque ON: {dxl_id}")

    # ---------------- CALLBACKS ----------------
    def gimbal_callback(self, msg):
        if len(msg.data) >= 2:
            pitch = msg.data[0]
            roll  = msg.data[1]


            # Map to motors 1,2
            m1 = 315 + pitch
            m2 = 315 + roll

            # Clamp
            m1 = max(275, min(345, m1))
            m2 = max(275, min(345, m2))

            self.gimbal_positions = [m1, m2]

        else:
            self.get_logger().warn("Gimbal needs 2 values")

    def arm_callback(self, msg):
        if len(msg.data) >= 2:
            joint1 = rad_to_deg(msg.data[0])
            joint2 = rad_to_deg(msg.data[1])

            # Map to motors 3,4
            m3 = 180#joint1
            m4 = 180#joint2

            # Clamp
            m3 = max(90, min(270, m3))
            m4 = max(0, min(360, m4))

            self.arm_positions = [m3, m4]

        else:
            self.get_logger().warn("Arm needs 2 joints")

    # ---------------- CONTROL LOOP ----------------
    def control_loop(self):

        # Combine all motor targets
        targets = self.gimbal_positions + self.arm_positions

        for i, dxl_id in enumerate(ALL_IDS):
            goal_pos = deg_to_dxl(targets[i])

            dxl_comm_result, dxl_error = packetHandler.write4ByteTxRx(
                portHandler, dxl_id, GOAL_POSITION_ADDR, goal_pos
            )

            if dxl_comm_result != COMM_SUCCESS:
                self.get_logger().error(packetHandler.getTxRxResult(dxl_comm_result))
            elif dxl_error != 0:
                self.get_logger().error(packetHandler.getRxPacketError(dxl_error))

    # ---------------- SHUTDOWN ----------------
    def shutdown(self):
        self.get_logger().info("Shutting down")

        for dxl_id in ALL_IDS:
            packetHandler.write1ByteTxRx(portHandler, dxl_id, TORQUE_ADDR, 0)

        portHandler.closePort()


# ---------------- MAIN ----------------
def main(args=None):
    rclpy.init(args=args)

    node = CombinedDynamixelController()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.shutdown()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
