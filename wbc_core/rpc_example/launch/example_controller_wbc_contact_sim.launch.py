from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    dt = LaunchConfiguration('dt')
    duration = LaunchConfiguration('duration')
    real_time = LaunchConfiguration('real_time')
    zero_torque = LaunchConfiguration('zero_torque')
    rviz = LaunchConfiguration('rviz')
    urdf_file = LaunchConfiguration('urdf')
    sim_yaml = LaunchConfiguration('yaml')
    rviz_config = LaunchConfiguration('rviz_config')
    base_height = LaunchConfiguration('base_height')
    contact_frames = LaunchConfiguration('contact_frames')

    default_urdf = PathJoinSubstitution(
        [FindPackageShare('draco_description'), 'urdf', 'draco_modified_rviz.urdf']
    )
    default_rviz_config = PathJoinSubstitution(
        [FindPackageShare('draco_description'), 'rviz', 'draco.rviz']
    )
    default_sim_yaml = PathJoinSubstitution(
        [FindPackageShare('rpc_example'), 'config', 'example_wbic_draco_floating.yaml']
    )

    robot_description = ParameterValue(
        Command([FindExecutable(name='cat'), ' ', urdf_file]),
        value_type=str,
    )

    sim_node = Node(
        package='rpc_example',
        executable='example_controller_wbc_contact_sim',
        output='screen',
        arguments=[
            '--dt', dt,
            '--duration', duration,
            '--base-height', base_height,
            '--real-time', real_time,
            '--zero-torque', zero_torque,
            '--publish-joint-states', 'true',
            '--contact-frames', contact_frames,
            '--urdf', urdf_file,
            '--yaml', sim_yaml,
        ],
    )

    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_description}],
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        output='screen',
        arguments=['-d', rviz_config],
        condition=IfCondition(rviz),
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument('dt', default_value='0.001'),
            DeclareLaunchArgument('duration', default_value='10.0'),
            DeclareLaunchArgument('base_height', default_value='0.95'),
            DeclareLaunchArgument('real_time', default_value='true'),
            DeclareLaunchArgument('zero_torque', default_value='false'),
            DeclareLaunchArgument('contact_frames', default_value='l_foot_contact,r_foot_contact'),
            DeclareLaunchArgument('urdf', default_value=default_urdf),
            DeclareLaunchArgument('yaml', default_value=default_sim_yaml),
            DeclareLaunchArgument('rviz_config', default_value=default_rviz_config),
            DeclareLaunchArgument('rviz', default_value='true'),
            sim_node,
            robot_state_publisher_node,
            rviz_node,
        ]
    )
