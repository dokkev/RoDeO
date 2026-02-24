from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    default_urdf = PathJoinSubstitution(
        [FindPackageShare("draco_description"), "urdf", "draco_modified_rviz.urdf"]
    )
    default_rviz = PathJoinSubstitution(
        [FindPackageShare("draco_description"), "rviz", "draco.rviz"]
    )

    urdf_file = LaunchConfiguration("urdf_file")
    rviz_config = LaunchConfiguration("rviz_config")
    robot_description = ParameterValue(Command(["cat ", urdf_file]), value_type=str)

    return LaunchDescription(
        [
            DeclareLaunchArgument("urdf_file", default_value=default_urdf),
            DeclareLaunchArgument("rviz_config", default_value=default_rviz),
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                name="robot_state_publisher",
                output="screen",
                parameters=[{"robot_description": robot_description}],
            ),
            Node(
                package="joint_state_publisher",
                executable="joint_state_publisher",
                name="joint_state_publisher",
                output="screen",
            ),
            Node(
                package="rviz2",
                executable="rviz2",
                name="rviz2",
                output="screen",
                arguments=["-d", rviz_config],
            ),
        ]
    )
