from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='mobile_jacobian',
            executable='mobile_jacobian_node.py',
            name='mobile_jacobian',
            output='screen'
        )
    ])