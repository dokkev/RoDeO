#!/usr/bin/env python3

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_dir = get_package_share_directory('joint_impedance_controller')

    default_controller_params = os.path.join(pkg_dir, 'config', 'impedance_preset.yaml')

    controller_params_arg = DeclareLaunchArgument(
        'controller_params_file',
        default_value=default_controller_params,
        description='Path to impedance preset YAML'
    )

    hand_namespace_arg = DeclareLaunchArgument(
        'hand_namespace',
        default_value='plato2',
        description='Namespace used to derive default hand topics for trajectory control.',
    )

    controller_node = Node(
        package='joint_impedance_controller',
        executable='impedance_trajectory_controller_node',
        name='impedance_trajectory_controller_node',
        output='screen',
        parameters=[{
            'hand_namespace': LaunchConfiguration('hand_namespace'),
            'impedance_preset_yaml_path': LaunchConfiguration('controller_params_file'),
        }],
    )

    return LaunchDescription([
        controller_params_arg,
        hand_namespace_arg,
        controller_node,
    ])
