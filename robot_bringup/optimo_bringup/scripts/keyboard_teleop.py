#!/usr/bin/env python3
"""Keyboard teleop for Optimo WBC controller.

Two modes (toggle with Tab):
  Joint mode     – move one joint at a time
  Cartesian mode – move end-effector in task space

Launch the MuJoCo simulation first:
  ros2 launch mujoco_sim mujoco_sim.launch.py

Then run this script:
  ros2 run optimo_bringup keyboard_teleop.py
"""
import curses
import sys

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped
from std_msgs.msg import Float64MultiArray
from wbc_msgs.srv import TransitionState

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
NS = "/optimo/wbc_controller"
NUM_JOINTS = 7
JOINT_VEL = 0.3          # rad/s per key-press
LINEAR_VEL = 0.05        # m/s
ANGULAR_VEL = 0.3        # rad/s
PUBLISH_HZ = 50          # command publish rate


class KeyboardTeleop(Node):
    def __init__(self):
        super().__init__("keyboard_teleop")

        # Publishers
        self.joint_pub = self.create_publisher(
            Float64MultiArray, f"{NS}/joint_vel_cmd", 10
        )
        self.ee_pub = self.create_publisher(
            TwistStamped, f"{NS}/ee_vel_cmd", 10
        )

        # Service client for state transitions
        self.state_client = self.create_client(
            TransitionState, f"{NS}/set_state"
        )

        # State
        self.mode = "joint"  # "joint" or "cartesian"
        self.active_joint = 0  # 0-indexed
        self.joint_vel = [0.0] * NUM_JOINTS
        self.lin = [0.0, 0.0, 0.0]  # x, y, z
        self.ang = [0.0, 0.0, 0.0]  # roll, pitch, yaw

    # -- service call (non-blocking, fire-and-forget) ----------------------
    def request_state(self, name: str):
        if not self.state_client.service_is_ready():
            self.get_logger().warn("set_state service not ready")
            return
        req = TransitionState.Request()
        req.state_name = name
        self.state_client.call_async(req)

    # -- publish commands --------------------------------------------------
    def publish_joint(self):
        msg = Float64MultiArray()
        msg.data = list(self.joint_vel)
        self.joint_pub.publish(msg)

    def publish_ee(self):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.twist.linear.x = self.lin[0]
        msg.twist.linear.y = self.lin[1]
        msg.twist.linear.z = self.lin[2]
        msg.twist.angular.x = self.ang[0]
        msg.twist.angular.y = self.ang[1]
        msg.twist.angular.z = self.ang[2]
        self.ee_pub.publish(msg)

    def publish(self):
        if self.mode == "joint":
            self.publish_joint()
        else:
            self.publish_ee()

    def stop_all(self):
        self.joint_vel = [0.0] * NUM_JOINTS
        self.lin = [0.0, 0.0, 0.0]
        self.ang = [0.0, 0.0, 0.0]

    # -- key handling ------------------------------------------------------
    def handle_key(self, key: int) -> bool:
        """Process one key press. Returns False to quit."""
        # ESC or q
        if key in (27, ord("q")):
            self.stop_all()
            self.publish()
            return False

        # Tab – toggle mode
        if key == ord("\t"):
            self.stop_all()
            self.publish()
            if self.mode == "joint":
                self.mode = "cartesian"
                self.request_state("cartesian_teleop")
            else:
                self.mode = "joint"
                self.request_state("joint_teleop")
            return True

        # Space – stop
        if key == ord(" "):
            self.stop_all()
            return True

        if self.mode == "joint":
            return self._handle_joint_key(key)
        else:
            return self._handle_cartesian_key(key)

    def _handle_joint_key(self, key: int) -> bool:
        # 1-7: select joint
        if ord("1") <= key <= ord("7"):
            self.joint_vel = [0.0] * NUM_JOINTS  # stop previous
            self.active_joint = key - ord("1")
            return True
        j = self.active_joint
        if key in (ord("w"), curses.KEY_UP):
            self.joint_vel = [0.0] * NUM_JOINTS
            self.joint_vel[j] = JOINT_VEL
        elif key in (ord("s"), curses.KEY_DOWN):
            self.joint_vel = [0.0] * NUM_JOINTS
            self.joint_vel[j] = -JOINT_VEL
        return True

    def _handle_cartesian_key(self, key: int) -> bool:
        self.lin = [0.0, 0.0, 0.0]
        self.ang = [0.0, 0.0, 0.0]
        # Linear: WASD + QE
        if key == ord("w"):
            self.lin[0] = LINEAR_VEL
        elif key == ord("s"):
            self.lin[0] = -LINEAR_VEL
        elif key == ord("a"):
            self.lin[1] = LINEAR_VEL
        elif key == ord("d"):
            self.lin[1] = -LINEAR_VEL
        elif key == ord("q"):
            return False  # q is quit, handled above
        elif key == ord("e"):
            self.lin[2] = LINEAR_VEL
        elif key == ord("c"):
            self.lin[2] = -LINEAR_VEL
        # Angular: IJKL + UO
        elif key == ord("i"):
            self.ang[1] = ANGULAR_VEL
        elif key == ord("k"):
            self.ang[1] = -ANGULAR_VEL
        elif key == ord("j"):
            self.ang[2] = ANGULAR_VEL
        elif key == ord("l"):
            self.ang[2] = -ANGULAR_VEL
        elif key == ord("u"):
            self.ang[0] = ANGULAR_VEL
        elif key == ord("o"):
            self.ang[0] = -ANGULAR_VEL
        return True


# ---------------------------------------------------------------------------
# curses UI
# ---------------------------------------------------------------------------
HELP_JOINT = """
  JOINT TELEOP
  ────────────────────────────
  1-7 : select joint (current: {joint})
  w/s : +/- velocity
  space : stop
  Tab : switch to Cartesian
  q/ESC : quit
"""

HELP_CART = """
  CARTESIAN TELEOP
  ────────────────────────────
  w/s : +/- X    a/d : +/- Y
  e/c : +/- Z
  i/k : +/- pitch   j/l : +/- yaw
  u/o : +/- roll
  space : stop
  Tab : switch to Joint
  ESC : quit
"""


def draw(stdscr, teleop: KeyboardTeleop):
    stdscr.clear()
    if teleop.mode == "joint":
        stdscr.addstr(0, 0, HELP_JOINT.format(joint=teleop.active_joint + 1))
        stdscr.addstr(10, 2, f"vel: {teleop.joint_vel}")
    else:
        stdscr.addstr(0, 0, HELP_CART)
        stdscr.addstr(12, 2, f"lin: [{teleop.lin[0]:+.3f}, {teleop.lin[1]:+.3f}, {teleop.lin[2]:+.3f}]")
        stdscr.addstr(13, 2, f"ang: [{teleop.ang[0]:+.3f}, {teleop.ang[1]:+.3f}, {teleop.ang[2]:+.3f}]")
    stdscr.refresh()


def main(stdscr):
    curses.curs_set(0)
    stdscr.nodelay(True)  # non-blocking getch
    stdscr.timeout(1000 // PUBLISH_HZ)

    rclpy.init()
    teleop = KeyboardTeleop()

    # Request initial state
    teleop.request_state("joint_teleop")

    try:
        while rclpy.ok():
            key = stdscr.getch()
            if key != -1:
                if not teleop.handle_key(key):
                    break
            teleop.publish()
            draw(stdscr, teleop)
            rclpy.spin_once(teleop, timeout_sec=0)
    finally:
        teleop.stop_all()
        teleop.publish()
        teleop.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    curses.wrapper(main)
