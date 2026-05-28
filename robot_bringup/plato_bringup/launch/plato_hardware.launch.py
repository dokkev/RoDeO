import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def package_file(package_name, *relative_parts):
    relative_path = Path(*relative_parts)
    prefixes = os.environ.get("AMENT_PREFIX_PATH", "").split(":")

    for prefix in prefixes:
        if not prefix:
            continue
        share_dir = Path(prefix) / "share" / package_name
        marker = (
            Path(prefix)
            / "share"
            / "ament_index"
            / "resource_index"
            / "packages"
            / package_name
        )
        candidate = share_dir / relative_path
        if marker.exists() and candidate.exists():
            return str(candidate)

    candidate = Path(get_package_share_directory(package_name)) / relative_path
    if candidate.exists():
        return str(candidate)

    raise FileNotFoundError(
        f"Could not find {relative_path} in package '{package_name}'"
    )


def generate_launch_description():
    rviz = LaunchConfiguration("rviz")
    plato_ns = LaunchConfiguration("plato_ns")
    use_sim_time = LaunchConfiguration("use_sim_time")
    fake_hardware = LaunchConfiguration("fake_hardware")
    zeroing = LaunchConfiguration("zeroing")
    robot_description_xacro_path = LaunchConfiguration("robot_description_xacro_path")
    controller_config_path = LaunchConfiguration("controller_config_path")
    rviz_config_path = LaunchConfiguration("rviz_config_path")
    controller_manager_name = LaunchConfiguration("controller_manager_name")
    joint_state_broadcaster_name = LaunchConfiguration("joint_state_broadcaster_name")
    joint_impedance_controller_name = LaunchConfiguration("joint_impedance_controller_name")

    declared_arguments = [
        DeclareLaunchArgument("rviz", default_value="true", description="Start RViz2 automatically."),
        DeclareLaunchArgument("plato_ns", default_value="plato2", description="Namespace for Plato hand."),
        DeclareLaunchArgument("use_sim_time", default_value="false", description="Use simulated clock if true."),
        DeclareLaunchArgument(
            "fake_hardware",
            default_value="false",
            description="Use ros2_control mock hardware instead of Plato CAN hardware.",
        ),
        DeclareLaunchArgument(
            "zeroing",
            default_value="false",
            description="Set current actuator positions as software zero after the first full feedback snapshot.",
        ),
        DeclareLaunchArgument(
            "robot_description_xacro_path",
            default_value=package_file("plato_description", "urdf", "plato_naritouch.urdf.xacro"),
            description="Absolute path to the Plato NariTouch robot description xacro.",
        ),
        DeclareLaunchArgument(
            "controller_config_path",
            default_value=package_file("plato_bringup", "config", "ros2_controllers.yaml"),
            description="Absolute path to the ros2_control controller manager parameters YAML.",
        ),
        DeclareLaunchArgument(
            "rviz_config_path",
            default_value=package_file("plato_description", "rviz", "plato2.rviz"),
            description="Absolute path to the RViz config file.",
        ),
        DeclareLaunchArgument(
            "controller_manager_name",
            default_value="controller_manager",
            description="Controller manager node name inside the namespace.",
        ),
        DeclareLaunchArgument(
            "joint_state_broadcaster_name",
            default_value="plato2_joint_state_broadcaster",
            description="Joint state broadcaster controller name.",
        ),
        DeclareLaunchArgument(
            "joint_impedance_controller_name",
            default_value="joint_impedance_controller",
            description="Joint impedance controller name.",
        ),
    ]

    robot_description_content = Command(
        [
            FindExecutable(name="xacro"),
            " ",
            robot_description_xacro_path,
            " ",
            "fake_hardware:=",
            fake_hardware,
            " ",
            "zeroing:=",
            zeroing,
        ]
    )
    robot_description = {
        "robot_description": ParameterValue(robot_description_content, value_type=str)
    }
    controller_manager_path = PathJoinSubstitution(["/", plato_ns, controller_manager_name])
    robot_description_topic = PathJoinSubstitution(["/", plato_ns, "robot_description"])

    control_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[robot_description, controller_config_path, {"use_sim_time": use_sim_time}],
        output="both",
        namespace=plato_ns,
        remappings=[
            ("~/robot_description", robot_description_topic),
            ("/robot_description", robot_description_topic),
        ],
    )

    robot_state_pub = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[robot_description, {"use_sim_time": use_sim_time}, {"publish_rate": 100.0}],
        output="both",
        namespace=plato_ns,
    )

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="log",
        arguments=["-d", rviz_config_path],
        condition=IfCondition(rviz),
    )

    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[joint_state_broadcaster_name, "--controller-manager", controller_manager_path],
        namespace=plato_ns,
        output="screen",
    )

    joint_impedance_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[joint_impedance_controller_name, "--controller-manager", controller_manager_path],
        namespace=plato_ns,
        output="screen",
    )

    start_rviz_after_jsb = RegisterEventHandler(
        OnProcessExit(target_action=joint_state_broadcaster_spawner, on_exit=[rviz])
    )
    start_impedance_after_jsb = RegisterEventHandler(
        OnProcessExit(
            target_action=joint_state_broadcaster_spawner,
            on_exit=[joint_impedance_controller_spawner],
        )
    )

    return LaunchDescription(
        declared_arguments
        + [
            control_node,
            robot_state_pub,
            joint_state_broadcaster_spawner,
            start_impedance_after_jsb,
            start_rviz_after_jsb,
        ]
    )
