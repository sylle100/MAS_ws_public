from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='drone_circle',
            executable='drone_circle_node.py',
            name='drone_circle',
            output='screen'
        )
    ])