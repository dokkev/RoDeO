from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    headless = LaunchConfiguration("headless")
    rviz = LaunchConfiguration("rviz")
    prefix = LaunchConfiguration("prefix")
    robot_index = LaunchConfiguration("robot_index")
    ns = LaunchConfiguration("ns")
    mujoco_model = LaunchConfiguration("mujoco_model")

    optimo_wbc_mujoco_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("optimo_bringup"), "launch", "optimo_wbc_mujoco.launch.py"]
            )
        ),
        launch_arguments={
            "headless": headless,
            "rviz": rviz,
            "prefix": prefix,
            "robot_index": robot_index,
            "ns": ns,
            "mujoco_model": mujoco_model,
        }.items(),
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("headless", default_value="false"),
            DeclareLaunchArgument("rviz", default_value="true"),
            DeclareLaunchArgument("prefix", default_value=""),
            DeclareLaunchArgument("robot_index", default_value="0"),
            DeclareLaunchArgument("ns", default_value="optimo"),
            DeclareLaunchArgument(
                "mujoco_model",
                default_value=PathJoinSubstitution(
                    [FindPackageShare("optimo_description"), "mjcf", "optimo.xml"]
                ),
            ),
            optimo_wbc_mujoco_launch,
        ]
    )
