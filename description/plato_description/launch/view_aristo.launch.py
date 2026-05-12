from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    use_sim_time = LaunchConfiguration("use_sim_time")
    use_gui = LaunchConfiguration("use_gui")
    namespace = LaunchConfiguration("namespace")
    urdf_file = LaunchConfiguration("urdf_file")
    rviz_config = LaunchConfiguration("rviz_config")

    default_urdf = PathJoinSubstitution(
        [FindPackageShare("plato_description"), "urdf", "aristo.urdf.xacro"]
    )
    default_rviz = PathJoinSubstitution(
        [FindPackageShare("plato_description"), "rviz", "plato2.rviz"]
    )

    robot_description = ParameterValue(
        Command(["xacro ", urdf_file]),
        value_type=str,
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("use_sim_time", default_value="false"),
            DeclareLaunchArgument("use_gui", default_value="true"),
            DeclareLaunchArgument("namespace", default_value="aristo"),
            DeclareLaunchArgument("urdf_file", default_value=default_urdf),
            DeclareLaunchArgument("rviz_config", default_value=default_rviz),
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                name="robot_state_publisher",
                output="screen",
                namespace=namespace,
                parameters=[
                    {
                        "robot_description": robot_description,
                        "use_sim_time": use_sim_time,
                    }
                ],
            ),
            Node(
                package="joint_state_publisher_gui",
                executable="joint_state_publisher_gui",
                name="joint_state_publisher_gui",
                output="screen",
                namespace=namespace,
                condition=IfCondition(use_gui),
            ),
            Node(
                package="joint_state_publisher",
                executable="joint_state_publisher",
                name="joint_state_publisher",
                output="screen",
                namespace=namespace,
                condition=UnlessCondition(use_gui),
            ),
            Node(
                package="rviz2",
                executable="rviz2",
                name="rviz2",
                output="screen",
                namespace=namespace,
                arguments=["-d", rviz_config],
            ),
        ]
    )
