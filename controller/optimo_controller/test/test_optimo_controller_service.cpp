/**
 * @file controller/optimo_controller/test/test_optimo_controller_service.cpp
 * @brief Integration test for the ~/set_state ROS2 service in OptimoController.
 *
 * Uses a self-contained 2-DOF URDF and minimal WBC config (temp files) so the
 * test runs without requiring any installed robot description packages.
 */
#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>

#include <rclcpp/rclcpp.hpp>

#include "optimo_controller/optimo_controller.hpp"
#include "wbc_msgs/srv/transition_state.hpp"

namespace {

// ---------------------------------------------------------------------------
// Minimal URDF: base_link → joint1 → link1 → joint2 → end_effector
// ---------------------------------------------------------------------------
std::string TwoDofUrdf() {
  return R"(
<robot name="test_two_dof">
  <link name="base_link">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="1.0"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
  </link>
  <joint name="joint1" type="revolute">
    <parent link="base_link"/>
    <child link="link1"/>
    <origin xyz="0 0 0.1" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="100" velocity="10"/>
  </joint>
  <link name="link1">
    <inertial>
      <origin xyz="0 0 0.05" rpy="0 0 0"/>
      <mass value="1.0"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
  </link>
  <joint name="joint2" type="revolute">
    <parent link="link1"/>
    <child link="end_effector"/>
    <origin xyz="0 0 0.1" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-3.14" upper="3.14" effort="100" velocity="10"/>
  </joint>
  <link name="end_effector">
    <inertial>
      <origin xyz="0 0 0.05" rpy="0 0 0"/>
      <mass value="0.5"/>
      <inertia ixx="0.005" ixy="0" ixz="0" iyy="0.005" iyz="0" izz="0.005"/>
    </inertial>
  </link>
</robot>
)";
}

// ---------------------------------------------------------------------------
// Minimal WBC YAML configs
// ---------------------------------------------------------------------------
std::string WbcYaml(const std::string& urdf_path) {
  return
    "robot_model:\n"
    "  urdf_path: \"" + urdf_path + "\"\n"
    "  is_floating_base: false\n"
    "  base_frame: \"base_link\"\n"
    "\n"
    "controller:\n"
    "  enable_gravity_compensation: true\n"
    "  enable_coriolis_compensation: false\n"
    "  enable_inertia_compensation: false\n"
    "  joint_pid:\n"
    "    enabled: false\n"
    "\n"
    "regularization:\n"
    "  w_qddot: 1.0e-6\n"
    "  w_rf: 1.0e-4\n"
    "  w_tau: 1.0e-3\n"
    "  w_tau_dot: 0.0\n"
    "  w_xc_ddot: 1.0e-3\n"
    "  w_f_dot: 1.0e-3\n"
    "\n"
    "  JointPosLimitConstraint:\n"
    "    enabled: true\n"
    "    scale: 1.0\n"
    "    is_soft: true\n"
    "    soft_weight: 1.0e+5\n"
    "\n"
    "  JointVelLimitConstraint:\n"
    "    enabled: true\n"
    "    scale: 1.0\n"
    "    is_soft: true\n"
    "    soft_weight: 1.0e+5\n"
    "\n"
    "  JointTrqLimitConstraint:\n"
    "    enabled: false\n"
    "\n"
    "task_pool_yaml: \"task_list.yaml\"\n"
    "state_machine_yaml: \"state_machine.yaml\"\n";
}

std::string TaskListYaml() {
  return R"(
task_pool:
  - name: "jpos_task"
    type: "JointTask"
    kp: [10.0, 10.0]
    kd: [1.0, 1.0]
    kp_ik: [1.0, 1.0]

  - name: "ee_pos_task"
    type: "LinkPosTask"
    target_frame: "end_effector"
    reference_frame: "base_link"
    kp: [10.0, 10.0, 10.0]
    kd: [1.0, 1.0, 1.0]
    kp_ik: [1.0, 1.0, 1.0]

  - name: "ee_ori_task"
    type: "LinkOriTask"
    target_frame: "end_effector"
    reference_frame: "base_link"
    kp: [10.0, 10.0, 10.0]
    kd: [1.0, 1.0, 1.0]
    kp_ik: [1.0, 1.0, 1.0]
)";
}

std::string StateMachineYaml() {
  return R"(
state_machine:
  - id: 0
    name: "initialize"
    params:
      duration: 0.1
      wait_time: 0.0
      stay_here: false
      target_jpos: [0.0, 0.0]
    task_hierarchy:
      - name: "jpos_task"

  - id: 1
    name: "joint_teleop"
    params:
      stay_here: true
    task_hierarchy:
      - name: "jpos_task"

  - id: 2
    name: "cartesian_teleop"
    params:
      stay_here: true
      linear_vel_max: 0.1
      angular_vel_max: 0.5
      manipulability:
        step_size: 0.5
        w_threshold: 0.01
    task_hierarchy:
      - name: "ee_pos_task"
        priority: 0
      - name: "ee_ori_task"
        priority: 1
      - name: "jpos_task"
        priority: 2
)";
}

void WriteFile(const std::filesystem::path& path, const std::string& content) {
  std::ofstream ofs(path);
  ofs << content;
}

// ---------------------------------------------------------------------------
// Test fixture
// ---------------------------------------------------------------------------
class OptimoServiceTest : public ::testing::Test {
protected:
  void SetUp() override {
    temp_dir_ = std::filesystem::temp_directory_path() /
                ("optimo_svc_test_" + std::to_string(::getpid()));
    std::filesystem::create_directories(temp_dir_);

    const auto urdf_path = temp_dir_ / "test_robot.urdf";
    WriteFile(urdf_path, TwoDofUrdf());
    WriteFile(temp_dir_ / "task_list.yaml", TaskListYaml());
    WriteFile(temp_dir_ / "state_machine.yaml", StateMachineYaml());
    WriteFile(temp_dir_ / "optimo_wbc.yaml", WbcYaml(urdf_path.string()));

    // Create controller and initialize with parameter overrides.
    controller_ = std::make_shared<optimo_controller::OptimoController>();

    rclcpp::NodeOptions node_options;
    node_options.parameter_overrides({
      rclcpp::Parameter("joints",
                        std::vector<std::string>{"joint1", "joint2"}),
      rclcpp::Parameter("wbc_yaml_path",
                        (temp_dir_ / "optimo_wbc.yaml").string()),
      rclcpp::Parameter("control_frequency", 1000.0),
    });

    // Deprecated init() — creates the lifecycle node and calls on_init().
    const auto init_ret =
        controller_->init("optimo_controller", "", 1000, "/test", node_options);
    ASSERT_EQ(init_ret, controller_interface::return_type::OK)
        << "Controller init() failed";

    // configure() triggers the lifecycle transition → on_configure().
    const auto& state = controller_->configure();
    ASSERT_EQ(state.label(), "inactive")
        << "on_configure() failed — got state: " << state.label();

    // Client node for service calls.
    client_node_ = rclcpp::Node::make_shared("test_client", "/test");
    client_ = client_node_->create_client<wbc_msgs::srv::TransitionState>(
        "/test/optimo_controller/set_state");

    executor_ = std::make_shared<rclcpp::executors::SingleThreadedExecutor>();
    executor_->add_node(controller_->get_node()->get_node_base_interface());
    executor_->add_node(client_node_);

    ASSERT_TRUE(client_->wait_for_service(std::chrono::seconds(2)))
        << "~/set_state service not available";
  }

  void TearDown() override {
    executor_.reset();
    client_.reset();
    client_node_.reset();
    controller_.reset();
    std::error_code ec;
    std::filesystem::remove_all(temp_dir_, ec);
  }

  /// Send a TransitionState request and spin until the response arrives.
  wbc_msgs::srv::TransitionState::Response::SharedPtr
  CallService(const wbc_msgs::srv::TransitionState::Request::SharedPtr& req) {
    auto future = client_->async_send_request(req);
    const auto status =
        executor_->spin_until_future_complete(future, std::chrono::seconds(5));
    EXPECT_EQ(status, rclcpp::FutureReturnCode::SUCCESS)
        << "Service call timed out";
    return future.get();
  }

  std::filesystem::path temp_dir_;
  std::shared_ptr<optimo_controller::OptimoController> controller_;
  rclcpp::Node::SharedPtr client_node_;
  rclcpp::Client<wbc_msgs::srv::TransitionState>::SharedPtr client_;
  std::shared_ptr<rclcpp::executors::SingleThreadedExecutor> executor_;
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

TEST_F(OptimoServiceTest, TransitionByNameValid) {
  auto req = std::make_shared<wbc_msgs::srv::TransitionState::Request>();
  req->state_name = "joint_teleop";

  auto res = CallService(req);
  EXPECT_TRUE(res->success);
  EXPECT_NE(res->message.find("joint_teleop"), std::string::npos);
}

TEST_F(OptimoServiceTest, TransitionByNameInvalid) {
  auto req = std::make_shared<wbc_msgs::srv::TransitionState::Request>();
  req->state_name = "nonexistent_state";

  auto res = CallService(req);
  EXPECT_FALSE(res->success);
  EXPECT_NE(res->message.find("Unknown"), std::string::npos);
}

TEST_F(OptimoServiceTest, TransitionById) {
  auto req = std::make_shared<wbc_msgs::srv::TransitionState::Request>();
  req->state_id = 2;  // cartesian_teleop

  auto res = CallService(req);
  EXPECT_TRUE(res->success);
  EXPECT_NE(res->message.find("id=2"), std::string::npos);
}

TEST_F(OptimoServiceTest, NameTakesPriorityOverId) {
  auto req = std::make_shared<wbc_msgs::srv::TransitionState::Request>();
  req->state_name = "joint_teleop";
  req->state_id = 999;  // invalid id — should be ignored because name is set

  auto res = CallService(req);
  EXPECT_TRUE(res->success);
  EXPECT_NE(res->message.find("joint_teleop"), std::string::npos);
}

}  // namespace

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  ::testing::InitGoogleTest(&argc, argv);
  int result = RUN_ALL_TESTS();
  rclcpp::shutdown();
  return result;
}
