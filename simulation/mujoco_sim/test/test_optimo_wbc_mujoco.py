#!/usr/bin/env python3
"""
launch_testing integration test: Optimo WBC + MuJoCo headless.

Launches the full ROS2 stack (mujoco_ros2_control + optimo_controller) in
headless mode, then verifies:
  1. Controller manager and WBC controller come up
  2. Joint states are published with finite values
  3. State transition service works (initialize → joint_teleop → cartesian_teleop)
  4. Joint teleop velocity commands produce motion
  5. WBC state topic publishes torque data
"""
import os
import time
import unittest

from ament_index_python.packages import get_package_share_directory
from controller_manager.test_utils import (
    check_controllers_running,
    check_if_js_published,
    check_node_running,
)
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_testing.actions import ReadyToTest
from launch_testing.util import KeepAliveProc
from launch_testing_ros import WaitForTopics
import pytest
import rclpy
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import TwistStamped
from wbc_msgs.srv import TransitionState
from wbc_msgs.msg import WbcState

# Match SensorDataQoS used by the controller (best_effort, keep_last 5).
SENSOR_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=5,
)


OPTIMO_JOINTS = [
    "joint1", "joint2", "joint3", "joint4",
    "joint5", "joint6", "joint7",
]

NS = "optimo"
CONTROLLER_MANAGER = f"/{NS}/controller_manager"
WBC_CONTROLLER = "wbc_controller"
JSB = "joint_state_broadcaster"
JOINT_STATES_TOPIC = f"/{NS}/{JSB}/joint_states"
WBC_STATE_TOPIC = f"/{NS}/{WBC_CONTROLLER}/wbc_state"
SET_STATE_SRV = f"/{NS}/{WBC_CONTROLLER}/set_state"
JOINT_VEL_TOPIC = f"/{NS}/{WBC_CONTROLLER}/joint_vel_cmd"


# ---------------------------------------------------------------------------
# Launch description — headless MuJoCo + optimo WBC
# ---------------------------------------------------------------------------
@pytest.mark.rostest
def generate_test_description():
    proc_env = os.environ.copy()
    proc_env["PYTHONUNBUFFERED"] = "1"

    launch_include = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("mujoco_sim"),
                "launch", "mujoco_sim.launch.py",
            )
        ),
        launch_arguments={
            "headless": "true",
            "rviz": "false",
            "ns": NS,
        }.items(),
    )

    return LaunchDescription([
        launch_include,
        KeepAliveProc(),
        ReadyToTest(),
    ])


# ---------------------------------------------------------------------------
# Test fixture
# ---------------------------------------------------------------------------
class OptimoWbcMujocoTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        self.node = rclpy.create_node("optimo_wbc_test_node")

    def tearDown(self):
        self.node.destroy_node()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _wait_for_service(self, srv_type, srv_name, timeout=15.0):
        client = self.node.create_client(srv_type, srv_name)
        ok = client.wait_for_service(timeout_sec=timeout)
        self.assertTrue(ok, f"Service {srv_name} not available after {timeout}s")
        return client

    def _call_service(self, client, request, timeout=10.0):
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self.node, future, timeout_sec=timeout)
        self.assertIsNotNone(future.result(), "Service call timed out")
        return future.result()

    def _transition_state(self, state_name):
        """Call set_state service and assert success."""
        client = self._wait_for_service(TransitionState, SET_STATE_SRV)
        req = TransitionState.Request()
        req.state_name = state_name
        res = self._call_service(client, req)
        self.assertTrue(
            res.success,
            f"Transition to '{state_name}' failed: {res.message}",
        )
        return res

    def _collect_joint_states(self, duration=1.0):
        """Collect joint state messages for `duration` seconds."""
        msgs = []

        def cb(msg):
            msgs.append(msg)

        sub = self.node.create_subscription(JointState, JOINT_STATES_TOPIC, cb, 10)
        end = time.time() + duration
        while time.time() < end:
            rclpy.spin_once(self.node, timeout_sec=0.05)
        self.node.destroy_subscription(sub)
        return msgs

    # ------------------------------------------------------------------
    # Tests (run in alphabetical order — prefix with number for ordering)
    # ------------------------------------------------------------------
    def test_01_node_startup(self, proc_output):
        """Verify robot_state_publisher is running."""
        check_node_running(self.node, "robot_state_publisher", timeout=20.0)

    def test_02_controllers_running(self):
        """Verify joint_state_broadcaster and wbc_controller are active."""
        # Give controllers time to spawn (delayed spawner waits for JSB exit).
        check_controllers_running(
            self.node, [JSB, WBC_CONTROLLER],
            namespace=f"/{NS}", timeout=30.0,
        )

    def test_03_joint_states_published(self):
        """Verify /joint_states has all 7 Optimo joints with finite values."""
        check_if_js_published(JOINT_STATES_TOPIC, OPTIMO_JOINTS)

        # Also verify values are finite (not NaN/Inf).
        msgs = self._collect_joint_states(duration=0.5)
        self.assertGreater(len(msgs), 0, "No joint state messages received")
        msg = msgs[-1]
        for i, name in enumerate(msg.name):
            if name in OPTIMO_JOINTS:
                self.assertTrue(
                    abs(msg.position[i]) < 100.0,
                    f"Joint {name} position looks invalid: {msg.position[i]}",
                )

    def test_04_wbc_state_published(self):
        """Verify WBC state topic publishes with non-zero torque."""
        wbc_msgs = []

        def cb(msg):
            wbc_msgs.append(msg)

        sub = self.node.create_subscription(WbcState, WBC_STATE_TOPIC, cb, SENSOR_QOS)

        # Wait up to 10s for at least one message.
        end = time.time() + 10.0
        while time.time() < end and not wbc_msgs:
            rclpy.spin_once(self.node, timeout_sec=0.1)
        self.node.destroy_subscription(sub)

        self.assertGreater(len(wbc_msgs), 0, "No WbcState messages received")
        msg = wbc_msgs[-1]
        self.assertEqual(len(msg.tau), 7, f"Expected 7 tau values, got {len(msg.tau)}")
        # After initialize state, torque should be non-zero (at least gravity).
        tau_norm = sum(t * t for t in msg.tau) ** 0.5
        self.assertGreater(tau_norm, 0.01, f"Torque norm too small: {tau_norm}")

    def test_05_state_transition_service(self):
        """Test FSM transitions: initialize → joint_teleop → cartesian_teleop."""
        # Wait for initialize state to finish (duration ~3s in default config).
        time.sleep(5.0)

        # Transition to joint_teleop.
        res = self._transition_state("joint_teleop")
        self.assertIn("joint_teleop", res.message)

        # Transition to cartesian_teleop.
        res = self._transition_state("cartesian_teleop")
        self.assertIn("cartesian_teleop", res.message)

        # Back to joint_teleop.
        res = self._transition_state("joint_teleop")
        self.assertIn("joint_teleop", res.message)

    def test_06_joint_teleop_velocity_command(self):
        """Send joint velocity commands and verify joints move."""
        # Ensure we're in joint_teleop.
        time.sleep(5.0)
        self._transition_state("joint_teleop")
        time.sleep(0.5)

        # Record initial joint positions.
        before = self._collect_joint_states(duration=0.3)
        self.assertGreater(len(before), 0)
        q_before = dict(zip(before[-1].name, before[-1].position))

        # Publish velocity commands: move joint2 at 0.3 rad/s for 2s.
        pub = self.node.create_publisher(
            Float64MultiArray, JOINT_VEL_TOPIC, SENSOR_QOS
        )
        # Wait for subscriber
        end = time.time() + 5.0
        while time.time() < end and pub.get_subscription_count() == 0:
            rclpy.spin_once(self.node, timeout_sec=0.1)
        self.assertGreater(
            pub.get_subscription_count(), 0,
            "No subscriber on joint velocity topic",
        )

        cmd = Float64MultiArray()
        cmd.data = [0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0]
        end = time.time() + 2.0
        while time.time() < end:
            pub.publish(cmd)
            rclpy.spin_once(self.node, timeout_sec=0.01)

        # Stop command.
        cmd.data = [0.0] * 7
        for _ in range(10):
            pub.publish(cmd)
            rclpy.spin_once(self.node, timeout_sec=0.01)

        time.sleep(0.5)

        # Check joint2 moved.
        after = self._collect_joint_states(duration=0.3)
        self.assertGreater(len(after), 0)
        q_after = dict(zip(after[-1].name, after[-1].position))

        delta = abs(q_after.get("joint2", 0.0) - q_before.get("joint2", 0.0))
        self.assertGreater(
            delta, 0.05,
            f"joint2 should have moved significantly (delta={delta:.4f} rad)",
        )
        self.node.get_logger().info(f"joint2 moved {delta:.4f} rad — OK")
