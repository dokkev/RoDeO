import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    headless = LaunchConfiguration("headless")
    rviz = LaunchConfiguration("rviz")
    prefix = LaunchConfiguration("prefix")
    robot_index = LaunchConfiguration("robot_index")
    ns = LaunchConfiguration("ns")
    mujoco_model = LaunchConfiguration("mujoco_model")

    xacro_file = PathJoinSubstitution(
        [FindPackageShare("optimo_description"), "urdf", "optimo.urdf.xacro"]
    )
    controller_file = PathJoinSubstitution(
        [FindPackageShare("optimo_bringup"), "config", "optimo_controller.yaml"]
    )
    rviz_config = PathJoinSubstitution(
        [FindPackageShare("optimo_description"), "rviz", "optimo.rviz"]
    )

    robot_description_content = Command(
        [
            FindExecutable(name="xacro"),
            " ",
            xacro_file,
            " ",
            "use_sim_hardware:=false",
            " ",
            "use_mock_hardware:=false",
            " ",
            "use_mukoco_sim:=true",
            " ",
            "headless:=",
            headless,
            " ",
            "mujoco_model:=",
            mujoco_model,
            " ",
            "prefix:=",
            prefix,
            " ",
            "robot_index:=",
            robot_index,
            " ",
            "ns:=",
            ns,
        ]
    )

    robot_description = {
        "robot_description": ParameterValue(robot_description_content, value_type=str)
    }

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="both",
        parameters=[robot_description, {"use_sim_time": True}],
    )

    control_node = Node(
        package="mujoco_ros2_control",
        executable="ros2_control_node",
        output="both",
        parameters=[
            {"use_sim_time": True},
            controller_file,
        ],
        remappings=(
            [("~/robot_description", "/robot_description")]
            if os.environ.get("ROS_DISTRO") == "humble"
            else []
        ),
    )

    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster", "--controller-manager", "/controller_manager"],
        output="both",
    )

    wbc_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["wbc_controller", "--controller-manager", "/controller_manager"],
        output="both",
    )

    delayed_wbc_spawner = RegisterEventHandler(
        OnProcessExit(
            target_action=joint_state_broadcaster_spawner,
            on_exit=[wbc_controller_spawner],
        )
    )

    rviz_node = Node(
        condition=IfCondition(rviz),
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="log",
        arguments=["-d", rviz_config],
        parameters=[{"use_sim_time": True}],
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
                    [FindPackageShare("mujoco_sim"), "mjcf", "scene.xml"]
                ),
            ),
            robot_state_publisher,
            control_node,
            joint_state_broadcaster_spawner,
            delayed_wbc_spawner,
            rviz_node,
        ]
    )
