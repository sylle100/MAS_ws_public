import math
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

class InverseKinematics(Node):
    def __init__(self):
        super().__init__('inverse_kinematics')

        self.publisher = self.create_publisher(
            Float64MultiArray, 
            "arm_controller/commands", 
            10
        )

        self.timer_period = 0.5

        self.timer = self.create_timer(self.timer_period, self.timer_callback)

    def timer_callback(self):
        # Desired pos
        x = -0.1383
        y = -(0.15)  # Straight down, at max reach

        # Length of links
        l1 = 0.15
        l2 = 0.1383

        # IK
        cos_angles2 = (x**2 + y**2 - l1**2 - l2**2) / (2 * l1 * l2)

        if abs(cos_angles2) > 1.0:
            self.get_logger().error('Target position is out of reach')
            return

        joint_2_angle = math.acos(cos_angles2)

        k1 = l1 + l2 * math.cos(joint_2_angle)
        k2 = l2 * math.sin(joint_2_angle)

        joint_1_angle = math.atan2(y, x) - math.atan2(k2, k1)

        # Convert + remap
        joint_1_deg = math.degrees(joint_1_angle)+180+90
        joint_2_deg = math.degrees(joint_2_angle)+180

        msg = Float64MultiArray()
        msg.data = [joint_1_deg, joint_2_deg]

        self.get_logger().info(
            f'Publishing: joint1 = {joint_1_deg:.2f}°, '
            f'joint2 = {joint_2_deg:.2f}°'
        )


def main(args=None):
    rclpy.init(args=args)
    node = InverseKinematics()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()