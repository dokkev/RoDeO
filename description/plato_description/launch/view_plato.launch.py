from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    use_aristo_hand = LaunchConfiguration("use_aristo_hand")
    joint1 = LaunchConfiguration("joint1")
    joint2 = LaunchConfiguration("joint2")
    joint3 = LaunchConfiguration("joint3")
    joint4 = LaunchConfiguration("joint4")
    joint5 = LaunchConfiguration("joint5")
    joint6 = LaunchConfiguration("joint6")
    joint7 = LaunchConfiguration("joint7")

    xacro_file = PathJoinSubstitution(
        [FindPackageShare("plato_description"), "urdf", "plato.urdf.xacro"]
    )
    rviz_config = PathJoinSubstitution(
        [FindPackageShare("plato_description"), "rviz", "plato_world.rviz"]
    )

    robot_description = ParameterValue(
        Command(
            [
                "xacro ",
                xacro_file,
                " use_aristo_hand:=",
                use_aristo_hand,
            ]
        ),
        value_type=str,
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("use_aristo_hand", default_value="true"),
            # Set a meaningful initial arm pose instead of midpoint-of-limits.
            DeclareLaunchArgument("joint1", default_value="0.0"),
            DeclareLaunchArgument("joint2", default_value="3.14159"),
            DeclareLaunchArgument("joint3", default_value="0.0"),
            DeclareLaunchArgument("joint4", default_value="-1.5708"),
            DeclareLaunchArgument("joint5", default_value="0.0"),
            DeclareLaunchArgument("joint6", default_value="-1.5708"),
            DeclareLaunchArgument("joint7", default_value="0.0"),
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                output="screen",
                parameters=[{"robot_description": robot_description}],
            ),
            Node(
                package="joint_state_publisher",
                executable="joint_state_publisher",
                output="screen",
                parameters=[
                    {
                        "zeros": {
                            "joint1": joint1,
                            "joint2": joint2,
                            "joint3": joint3,
                            "joint4": joint4,
                            "joint5": joint5,
                            "joint6": joint6,
                            "joint7": joint7,
                            "aristo_joint1": 0.0,
                            "aristo_joint2": 0.0,
                            "aristo_joint3": 0.0,
                            "aristo_joint4": 0.0,
                            "aristo_joint5": 0.0,
                            "aristo_joint6": 0.0,
                            "aristo_joint7": 0.0,
                            "aristo_joint8": 0.0,
                        }
                    }
                ],
            ),
            Node(
                package="rviz2",
                executable="rviz2",
                output="screen",
                arguments=["-d", rviz_config],
            ),
        ]
    )
