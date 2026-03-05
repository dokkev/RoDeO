/**
 * @file wbc_core/wbc_architecture/test/test_wbc_config_compiler.cpp
 * @brief Doxygen documentation for test_wbc_config_compiler module.
 */
#include <sys/types.h>
#include <unistd.h>

#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>

#include <gtest/gtest.h>

#include "wbc_architecture/config_compiler.hpp"
#include "wbc_architecture/runtime_config.hpp"
#include "wbc_formulation/kinematic_constraint.hpp"
#include "wbc_robot_system/pinocchio_robot_system.hpp"

namespace wbc {
namespace {

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
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="100" velocity="10"/>
  </joint>
  <link name="link1">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="1.0"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
  </link>
  <joint name="joint2" type="revolute">
    <parent link="link1"/>
    <child link="link2"/>
    <origin xyz="0 0 0.1" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-3.14" upper="3.14" effort="100" velocity="10"/>
  </joint>
  <link name="link2">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="1.0"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
  </link>
</robot>
)";
}

class WbcConfigTest : public ::testing::Test {
protected:
  void SetUp() override {
    const std::string suffix =
        std::to_string(static_cast<long long>(::getpid())) + "_" +
        std::to_string(counter_++);
    temp_dir_ = std::filesystem::temp_directory_path() /
                ("wbc_cfg_compiler_test_" + suffix);
    std::filesystem::create_directories(temp_dir_);

    urdf_path_ = temp_dir_ / "test_robot.urdf";
    WriteFile(urdf_path_, TwoDofUrdf());

    robot_ = std::make_unique<PinocchioRobotSystem>(
        urdf_path_.string(), "", true, false, nullptr);
  }

  void TearDown() override {
    robot_.reset();
    std::error_code ec;
    std::filesystem::remove_all(temp_dir_, ec);
  }

  static void WriteFile(const std::filesystem::path& path,
                        const std::string& content) {
    std::ofstream ofs(path);
    if (!ofs.is_open()) {
      throw std::runtime_error("Failed to open file: " + path.string());
    }
    ofs << content;
  }

  std::filesystem::path WriteYaml(const std::string& filename,
                                  const std::string& yaml) {
    const std::filesystem::path path = temp_dir_ / filename;
    WriteFile(path, yaml);
    return path;
  }

  std::filesystem::path temp_dir_;
  std::filesystem::path urdf_path_;
  std::unique_ptr<PinocchioRobotSystem> robot_;

  static inline int counter_ = 0;
};

TEST_F(WbcConfigTest, MotionTaskOrderFollowsPriorityAndYamlOrderForTies) {
  const auto yaml_path = WriteYaml(
      "priority.yaml",
      R"(
task_pool:
  - name: t1
    type: JointTask
  - name: t2
    type: JointTask
  - name: t3
    type: JointTask
state_machine:
  - id: 1
    name: s1
    task_hierarchy:
      - name: t2
        priority: 0
      - name: t1
        priority: 0
      - name: t3
        priority: 1
)");

  const auto compiler = ConfigCompiler::Compile(robot_.get(), yaml_path.string())->TakeConfig();
  const StateConfig& state = compiler->State(1);
  TaskRegistry* registry = compiler->taskRegistry();
  ASSERT_NE(registry, nullptr);
  ASSERT_EQ(state.motion.size(), 3u);

  EXPECT_EQ(state.motion[0], registry->GetMotionTask("t2"));
  EXPECT_EQ(state.motion[1], registry->GetMotionTask("t1"));
  EXPECT_EQ(state.motion[2], registry->GetMotionTask("t3"));
}

TEST_F(WbcConfigTest, ThrowsOnMissingContactForForceTask) {
  const auto yaml_path = WriteYaml(
      "missing_contact.yaml",
      R"(
task_pool:
  - name: force_task
    type: ForceTask
    contact_name: missing_contact
state_machine:
  - id: 1
    name: s1
    task_hierarchy:
      - name: any
)");

  EXPECT_THROW(
      {
        const auto compiler =
            ConfigCompiler::Compile(robot_.get(), yaml_path.string())->TakeConfig();
        (void)compiler;
      },
      std::runtime_error);
}

TEST_F(WbcConfigTest, ThrowsOnMissingMotionTaskReferenceInState) {
  const auto yaml_path = WriteYaml(
      "missing_motion_ref.yaml",
      R"(
task_pool:
  - name: existing_task
    type: JointTask
state_machine:
  - id: 1
    name: s1
    task_hierarchy:
      - name: missing_task
)");

  EXPECT_THROW(
      {
        const auto compiler =
            ConfigCompiler::Compile(robot_.get(), yaml_path.string())->TakeConfig();
        (void)compiler;
      },
      std::runtime_error);
}

TEST_F(WbcConfigTest, ScalarGainBroadcastAppliesAcrossTaskDimension) {
  const auto yaml_path = WriteYaml(
      "scalar_broadcast.yaml",
      R"(
task_pool:
  - name: joint_task
    type: JointTask
    kp: 3.0
    kd: [1.0, 2.0]
state_machine:
  - id: 1
    name: s1
    task_hierarchy:
      - name: joint_task
)");

  const auto compiler = ConfigCompiler::Compile(robot_.get(), yaml_path.string())->TakeConfig();
  TaskRegistry* registry = compiler->taskRegistry();
  ASSERT_NE(registry, nullptr);
  Task* task = registry->GetMotionTask("joint_task");
  ASSERT_NE(task, nullptr);
  ASSERT_EQ(task->Dim(), 2);

  const Eigen::VectorXd kp = task->Kp();
  const Eigen::VectorXd kd = task->Kd();
  ASSERT_EQ(kp.size(), 2);
  ASSERT_EQ(kd.size(), 2);
  EXPECT_NEAR(kp[0], 3.0, 1.0e-12);
  EXPECT_NEAR(kp[1], 3.0, 1.0e-12);
  EXPECT_NEAR(kd[0], 1.0, 1.0e-12);
  EXPECT_NEAR(kd[1], 2.0, 1.0e-12);
}

TEST_F(WbcConfigTest, ThrowsOnGainDimensionMismatch) {
  const auto yaml_path = WriteYaml(
      "gain_dim_mismatch.yaml",
      R"(
task_pool:
  - name: joint_task
    type: JointTask
    kp: [1.0, 2.0, 3.0]
state_machine:
  - id: 1
    name: s1
    task_hierarchy:
      - name: joint_task
)");

  EXPECT_THROW(
      {
        const auto compiler =
            ConfigCompiler::Compile(robot_.get(), yaml_path.string())->TakeConfig();
        (void)compiler;
      },
      std::runtime_error);
}

TEST_F(WbcConfigTest, StartStateIdFallbackPriority) {
  const auto top_level_start_yaml = WriteYaml(
      "start_top_level.yaml",
      R"(
start_state_id: 3
solver_params:
  start_state_id: 2
task_pool:
  - name: j
    type: JointTask
state_machine:
  - id: 1
    name: s1
    task_hierarchy: [{name: j}]
  - id: 2
    name: s2
    task_hierarchy: [{name: j}]
  - id: 3
    name: s3
    task_hierarchy: [{name: j}]
)");

  const auto solver_start_yaml = WriteYaml(
      "start_solver.yaml",
      R"(
solver_params:
  start_state_id: 2
task_pool:
  - name: j
    type: JointTask
state_machine:
  - id: 1
    name: s1
    task_hierarchy: [{name: j}]
  - id: 2
    name: s2
    task_hierarchy: [{name: j}]
)");

  const auto first_state_yaml = WriteYaml(
      "start_first.yaml",
      R"(
task_pool:
  - name: j
    type: JointTask
state_machine:
  - id: 7
    name: s7
    task_hierarchy: [{name: j}]
  - id: 8
    name: s8
    task_hierarchy: [{name: j}]
)");

  const auto c1 =
      ConfigCompiler::Compile(robot_.get(), top_level_start_yaml.string())->TakeConfig();
  const auto c2 =
      ConfigCompiler::Compile(robot_.get(), solver_start_yaml.string())->TakeConfig();
  const auto c3 =
      ConfigCompiler::Compile(robot_.get(), first_state_yaml.string())->TakeConfig();

  EXPECT_EQ(c1->StartStateId(), 3);
  EXPECT_EQ(c2->StartStateId(), 1);
  EXPECT_EQ(c3->StartStateId(), 7);
}

TEST_F(WbcConfigTest, StateTypeDefaultsToNameAndSupportsExplicitType) {
  const auto yaml_path = WriteYaml(
      "state_type.yaml",
      R"(
task_pool:
  - name: j
    type: JointTask
state_machine:
  - id: 1
    name: swing_left
    type: swing
    task_hierarchy: [{name: j}]
  - id: 2
    name: swing_right
    task_hierarchy: [{name: j}]
)");

  const auto config = ConfigCompiler::Compile(robot_.get(), yaml_path.string())->TakeConfig();
  ASSERT_NE(config, nullptr);

  const StateConfig& state1 = config->State(1);
  const StateConfig& state2 = config->State(2);
  EXPECT_EQ(state1.name, "swing_left");
  EXPECT_EQ(state1.type, "swing");
  EXPECT_EQ(state2.name, "swing_right");
  EXPECT_EQ(state2.type, "swing_right");
}

// ---------------------------------------------------------------------------
// Joint constraint scaling tests
// ---------------------------------------------------------------------------

TEST_F(WbcConfigTest, GlobalScalingAppliesCorrectly) {
  // URDF: pos [-3.14, 3.14], vel 10, trq 100  (2 joints)
  const auto yaml_path = WriteYaml("scaling_test.yaml", R"(
task_pool:
  - name: j
    type: JointTask
global_constraints:
  JointPosLimitConstraint:
    enabled: true
    scale: 0.5
  JointVelLimitConstraint:
    enabled: true
    scale: 0.6
  JointTrqLimitConstraint:
    enabled: true
    scale: 0.7
state_machine:
  - id: 0
    name: s0
    task_hierarchy: [{name: j}]
)");

  auto compiler = ConfigCompiler::Compile(robot_.get(), yaml_path.string());
  auto config = compiler->TakeConfig();

  // Verify constraints were created.
  ConstraintRegistry* reg = config->constraintRegistry();
  ASSERT_NE(reg, nullptr);

  auto* pos_c = dynamic_cast<JointPosLimitConstraint*>(
      reg->GetConstraint("JointPosLimitConstraint"));
  auto* vel_c = dynamic_cast<JointVelLimitConstraint*>(
      reg->GetConstraint("JointVelLimitConstraint"));
  auto* trq_c = dynamic_cast<JointTrqLimitConstraint*>(
      reg->GetConstraint("JointTrqLimitConstraint"));
  ASSERT_NE(pos_c, nullptr);
  ASSERT_NE(vel_c, nullptr);
  ASSERT_NE(trq_c, nullptr);

  // q_ and qdot_ are zero-initialized by the constructor.
  pos_c->UpdateConstraint();
  vel_c->UpdateConstraint();
  trq_c->UpdateConstraint();

  // pos_scale=0.5 on [-3.14, 3.14]: mid=0, half=3.14
  //   → new limits = [-1.57, 1.57]
  // Look-ahead damper (kTauPos=0.05, kTauVel=0.02):
  //   At q=0, qdot=0: qdot_safe = pos_lim / kTauPos
  //                    qddot     = qdot_safe / kTauVel = pos_lim / (kTauPos * kTauVel)
  // Constraint vector: [-qddot_min, qddot_max] for each joint
  const Eigen::VectorXd& pos_vec = pos_c->ConstraintVector();
  ASSERT_EQ(pos_vec.size(), 4);  // 2 joints * 2 bounds
  constexpr double kTauPos = 0.05;
  constexpr double kTauVel_pos = 0.02;
  const double expected_pos_limit = 3.14 * 0.5;
  const double expected_qddot = expected_pos_limit / (kTauPos * kTauVel_pos);
  EXPECT_NEAR(pos_vec(0), expected_qddot, 1.0);   // -qddot_min = -(-val) = val
  EXPECT_NEAR(pos_vec(2), expected_qddot, 1.0);   // qddot_max

  // vel_scale=0.6 on [-10, 10] → [-6, 6]
  // Look-ahead damper (kTauVel=0.02):
  //   At qdot=0: qddot = vel_lim / kTauVel
  constexpr double kTauVel_vel = 0.02;
  const Eigen::VectorXd& vel_vec = vel_c->ConstraintVector();
  ASSERT_EQ(vel_vec.size(), 4);
  const double expected_vel = 10.0 * 0.6 / kTauVel_vel;
  EXPECT_NEAR(vel_vec(0), expected_vel, 1.0);   // -qddot_min
  EXPECT_NEAR(vel_vec(2), expected_vel, 1.0);   // qddot_max

  // trq_scale=0.7 on [-100, 100] → [-70, 70]
  const Eigen::VectorXd& trq_vec = trq_c->ConstraintVector();
  ASSERT_EQ(trq_vec.size(), 4);
  EXPECT_NEAR(trq_vec(0), 100.0 * 0.7, 1e-6);  // -trq_min = 70
  EXPECT_NEAR(trq_vec(2), 100.0 * 0.7, 1e-6);  // trq_max = 70
}

TEST_F(WbcConfigTest, DisabledConstraintIsNotRegistered) {
  const auto yaml_path = WriteYaml("disabled_test.yaml", R"(
task_pool:
  - name: j
    type: JointTask
global_constraints:
  JointPosLimitConstraint:
    enabled: true
  JointVelLimitConstraint:
    enabled: false
  JointTrqLimitConstraint:
    enabled: false
state_machine:
  - id: 0
    name: s0
    task_hierarchy: [{name: j}]
)");

  auto compiler = ConfigCompiler::Compile(robot_.get(), yaml_path.string());
  auto config = compiler->TakeConfig();
  ConstraintRegistry* reg = config->constraintRegistry();

  EXPECT_NE(reg->GetConstraint("JointPosLimitConstraint"), nullptr);
  EXPECT_EQ(reg->GetConstraint("JointVelLimitConstraint"), nullptr);
  EXPECT_EQ(reg->GetConstraint("JointTrqLimitConstraint"), nullptr);

  // Only 1 global constraint should be registered.
  EXPECT_EQ(config->GlobalConstraints().size(), 1u);
}

TEST_F(WbcConfigTest, NoScalingUsesUrdfDefaults) {
  const auto yaml_path = WriteYaml("no_scaling.yaml", R"(
task_pool:
  - name: j
    type: JointTask
global_constraints:
  JointPosLimitConstraint:
    enabled: true
state_machine:
  - id: 0
    name: s0
    task_hierarchy: [{name: j}]
)");

  auto compiler = ConfigCompiler::Compile(robot_.get(), yaml_path.string());
  auto config = compiler->TakeConfig();
  ConstraintRegistry* reg = config->constraintRegistry();

  auto* pos_c = dynamic_cast<JointPosLimitConstraint*>(
      reg->GetConstraint("JointPosLimitConstraint"));
  ASSERT_NE(pos_c, nullptr);

  pos_c->UpdateConstraint();

  // Without scaling section, should use full URDF range [-3.14, 3.14]
  // Look-ahead damper: qddot_max = pos_lim / (kTauPos * kTauVel) = 3.14 / 0.001
  const Eigen::VectorXd& pos_vec = pos_c->ConstraintVector();
  const double expected = 3.14 / (0.05 * 0.02);  // kTauPos=0.05, kTauVel=0.02
  EXPECT_NEAR(pos_vec(0), expected, 1.0);
  EXPECT_NEAR(pos_vec(2), expected, 1.0);
}

} // namespace
} // namespace wbc
