from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler, TimerAction
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit, OnProcessStart
from launch.substitutions import (
    Command,
    FindExecutable,
    LaunchConfiguration,
    PathJoinSubstitution,
    PythonExpression,
)
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterFile, ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    headless = LaunchConfiguration("headless")
    rviz = LaunchConfiguration("rviz")
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
    robot_description_topic = PathJoinSubstitution(["/", ns, "robot_description"])
    pal_stats_log_level = PythonExpression(
        ["'", ns, ".controller_manager.pal_statistics:=fatal'"]
    )

    robot_description_content = Command(
        [
            FindExecutable(name="xacro"),
            " ",
            xacro_file,
            " ",
            "use_sim_hardware:=true",
            " ",
            "use_mock_hardware:=false",
            " ",
            "use_mujoco_sim:=true",
            " ",
            "headless:=",
            headless,
            " ",
            "mujoco_model:=",
            mujoco_model,
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
        namespace=ns,
        output="both",
        parameters=[robot_description, {"use_sim_time": True}],
        remappings=[
            ("joint_states", "joint_state_broadcaster/joint_states"),
        ],
    )

    controller_manager_name = PathJoinSubstitution(["/", ns, "controller_manager"])

    control_node = Node(
        package="mujoco_ros2_control",
        executable="ros2_control_node",
        namespace=ns,
        output="both",
        arguments=[
            "--ros-args",
            "--log-level",
            pal_stats_log_level,
        ],
        parameters=[
            {"use_sim_time": True},
            ParameterFile(controller_file),
        ],
        remappings=[
            ("~/robot_description", robot_description_topic),
            ("/robot_description", robot_description_topic),
        ],
    )

    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        namespace=ns,
        arguments=[
            "joint_state_broadcaster",
            "--controller-manager", controller_manager_name,
            "--controller-manager-timeout", "30",
        ],
        output="both",
    )

    wbc_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        namespace=ns,
        arguments=[
            "wbc_controller",
            "--controller-manager", controller_manager_name,
            "--controller-manager-timeout", "30",
        ],
        output="both",
    )

    # Delay spawners until hardware has had time to initialize
    delayed_jsb_spawner = RegisterEventHandler(
        OnProcessStart(
            target_action=control_node,
            on_start=[
                TimerAction(
                    period=5.0,
                    actions=[joint_state_broadcaster_spawner],
                ),
            ],
        )
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
            DeclareLaunchArgument("robot_index", default_value="0"),
            DeclareLaunchArgument("ns", default_value="optimo"),
            DeclareLaunchArgument(
                "mujoco_model",
                default_value=PathJoinSubstitution(
                    [FindPackageShare("optimo_description"), "mjcf", "optimo.xml"]
                ),
            ),
            robot_state_publisher,
            control_node,
            delayed_jsb_spawner,
            delayed_wbc_spawner,
            rviz_node,
        ]
    )
