#!/usr/bin/env python3
"""SpaceMouse -> TwistStamped publisher."""

import math

import rclpy
from geometry_msgs.msg import TwistStamped
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException

from spacemouse_interface.spacemouse_hardware import (
    SpaceMouseHardware,
    SpaceMouseState,
)


class SpaceMouseTwistPublisher(Node):
    """Publish TwistStamped commands from SpaceMouse input."""

    def __init__(self) -> None:
        super().__init__("spacemouse_twist_publisher")

        self.declare_parameter("translation_scale", 0.1)
        self.declare_parameter("rotation_scale", 0.1)
        self.declare_parameter("translation_threshold", 0.001)
        self.declare_parameter("rotation_threshold", 0.001)
        self.declare_parameter("publish_rate", 100.0)
        self.declare_parameter("twist_topic", "/optimo/wbc_controller/ee_vel_cmd")
        self.declare_parameter("frame_id", "world")

        self.translation_scale = (
            self.get_parameter("translation_scale").get_parameter_value().double_value
        )
        self.rotation_scale = (
            self.get_parameter("rotation_scale").get_parameter_value().double_value
        )
        self.translation_threshold = (
            self.get_parameter("translation_threshold")
            .get_parameter_value()
            .double_value
        )
        self.rotation_threshold = (
            self.get_parameter("rotation_threshold").get_parameter_value().double_value
        )
        self.publish_rate = (
            self.get_parameter("publish_rate").get_parameter_value().double_value
        )
        self.twist_topic = (
            self.get_parameter("twist_topic").get_parameter_value().string_value
        )
        self.frame_id = self.get_parameter("frame_id").get_parameter_value().string_value

        self.spacemouse = SpaceMouseHardware()
        if not self.spacemouse.open():
            self.get_logger().error(
                "Failed to open SpaceMouse device. Install pyspacemouse and check USB connection."
            )
            raise RuntimeError("SpaceMouse initialization failed")

        self.twist_pub = self.create_publisher(TwistStamped, self.twist_topic, 10)
        timer_period = 1.0 / max(self.publish_rate, 1.0)
        self.timer = self.create_timer(timer_period, self.timer_callback)

        self.get_logger().info(
            f"SpaceMouse ready: topic={self.twist_topic} "
            f"rate={self.publish_rate:.1f}Hz frame={self.frame_id}"
            f" translation_scale={self.translation_scale} rotation_scale={self.rotation_scale}"
        )

    @staticmethod
    def translation_magnitude(state: SpaceMouseState) -> float:
        return math.sqrt(state.x * state.x + state.y * state.y + state.z * state.z)

    @staticmethod
    def rotation_magnitude(state: SpaceMouseState) -> float:
        return math.sqrt(
            state.roll * state.roll
            + state.pitch * state.pitch
            + state.yaw * state.yaw
        )

    def decide_movement(self, state: SpaceMouseState) -> str:
        trans = self.translation_magnitude(state)
        rot = self.rotation_magnitude(state)

        if trans > self.translation_threshold and trans > rot:
            return "translation"
        if rot > self.rotation_threshold:
            return "rotation"
        return "none"

    def publish_twist(
        self,
        linear_x: float = 0.0,
        linear_y: float = 0.0,
        linear_z: float = 0.0,
        angular_x: float = 0.0,
        angular_y: float = 0.0,
        angular_z: float = 0.0,
    ) -> None:
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        msg.twist.linear.x = linear_x
        msg.twist.linear.y = linear_y
        msg.twist.linear.z = linear_z
        msg.twist.angular.x = angular_x
        msg.twist.angular.y = angular_y
        msg.twist.angular.z = angular_z
        self.twist_pub.publish(msg)

    def timer_callback(self) -> None:
        state = self.spacemouse.read()
        if state is None:
            self.get_logger().warn(
                "Failed to read SpaceMouse state",
                throttle_duration_sec=1.0,
            )
            return

        # Mapping from legacy SpaceMouse publisher behavior.
        mapped_state = SpaceMouseState(
            x=state.y,
            y=-state.x,
            z=state.z,
            roll=state.roll,
            pitch=state.pitch,
            yaw=-state.yaw,
            buttons=state.buttons,
        )

        action = self.decide_movement(mapped_state)
        if action == "translation":
            self.publish_twist(
                linear_x=mapped_state.x * self.translation_scale,
                linear_y=mapped_state.y * self.translation_scale,
                linear_z=mapped_state.z * self.translation_scale,
            )
        elif action == "rotation":
            self.publish_twist(
                angular_x=mapped_state.roll * self.rotation_scale,
                angular_y=mapped_state.pitch * self.rotation_scale,
                angular_z=mapped_state.yaw * self.rotation_scale,
            )
        else:
            self.publish_twist()

        pressed, released = self.spacemouse.get_button_transitions(state.buttons)
        if pressed:
            self.get_logger().debug(f"buttons pressed: {pressed}")
        if released:
            self.get_logger().debug(f"buttons released: {released}")

    def destroy_node(self) -> bool:
        self.spacemouse.close()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = SpaceMouseTwistPublisher()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except ExternalShutdownException:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
