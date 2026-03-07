from launch import LaunchDescription
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    return LaunchDescription(
        [
            Node(
                package="spacemouse_interface",
                executable="spacemouse_twist",
                name="spacemouse_twist",
                output="screen",
                parameters=[
                    PathJoinSubstitution(
                        [
                            FindPackageShare("spacemouse_interface"),
                            "config",
                            "spacemouse_interface.yaml",
                        ]
                    )
                ],
            ),
        ]
    )
