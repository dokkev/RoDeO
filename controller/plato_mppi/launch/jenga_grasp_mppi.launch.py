#!/usr/bin/env python3

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    package_share = Path(get_package_share_directory("plato_mppi"))
    default_params = package_share / "config" / "jenga_grasp_mppi.yaml"

    params_file = LaunchConfiguration("params_file")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "params_file",
                default_value=str(default_params),
                description="YAML parameters for the Jenga MPPI impedance adapter.",
            ),
            Node(
                package="plato_mppi",
                executable="jenga_grasp_mppi_node",
                name="jenga_grasp_mppi_node",
                output="screen",
                parameters=[params_file],
            ),
        ]
    )
