import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    package_share = get_package_share_directory("dynamixel_interface")
    config_path = os.path.join(package_share, "config", "dynamixel_topic_controller.yaml")

    return LaunchDescription(
        [
            Node(
                package="dynamixel_interface",
                executable="dynamixel_topic_controller",
                name="dynamixel_topic_controller",
                output="screen",
                parameters=[config_path],
            )
        ]
    )
