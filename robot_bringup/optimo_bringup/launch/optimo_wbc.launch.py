from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    use_sim_hardware = LaunchConfiguration("use_sim_hardware")
    use_mock_hardware = LaunchConfiguration("use_mock_hardware")
    prefix = LaunchConfiguration("prefix")
    robot_index = LaunchConfiguration("robot_index")
    ns = LaunchConfiguration("ns")

    xacro_file = PathJoinSubstitution(
        [FindPackageShare("optimo_description"), "urdf", "optimo.urdf.xacro"]
    )
    controller_file = PathJoinSubstitution(
        [FindPackageShare("optimo_bringup"), "config", "optimo_controller.yaml"]
    )
    controller_manager_name = PathJoinSubstitution(["/", ns, "controller_manager"])
    robot_description_topic = PathJoinSubstitution(["/", ns, "robot_description"])

    robot_description_content = Command(
        [
            FindExecutable(name="xacro"),
            " ",
            xacro_file,
            " ",
            "use_sim_hardware:=",
            use_sim_hardware,
            " ",
            "use_mock_hardware:=",
            use_mock_hardware,
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
    wbc_controller_robot_description = {
        "wbc_controller.robot_description": ParameterValue(
            robot_description_content, value_type=str
        )
    }

    control_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        namespace=ns,
        output="screen",
        parameters=[
            robot_description,
            wbc_controller_robot_description,
            controller_file,
        ],
        remappings=[
            ("~/robot_description", robot_description_topic),
            ("/robot_description", robot_description_topic),
        ],
    )

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        namespace=ns,
        output="screen",
        parameters=[robot_description],
        remappings=[
            ("joint_states", "joint_state_broadcaster/joint_states"),
        ],
    )

    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        namespace=ns,
        arguments=["joint_state_broadcaster", "--controller-manager", controller_manager_name],
        output="screen",
    )

    wbc_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        namespace=ns,
        arguments=["wbc_controller", "--controller-manager", controller_manager_name],
        output="screen",
    )

    delayed_wbc_spawner = RegisterEventHandler(
        OnProcessExit(
            target_action=joint_state_broadcaster_spawner,
            on_exit=[wbc_controller_spawner],
        )
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("use_sim_hardware", default_value="false"),
            DeclareLaunchArgument("use_mock_hardware", default_value="true"),
            DeclareLaunchArgument("prefix", default_value=""),
            DeclareLaunchArgument("robot_index", default_value="0"),
            DeclareLaunchArgument("ns", default_value="optimo"),
            robot_state_publisher,
            control_node,
            joint_state_broadcaster_spawner,
            delayed_wbc_spawner,
        ]
    )
