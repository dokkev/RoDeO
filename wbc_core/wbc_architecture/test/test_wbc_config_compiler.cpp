#include <sys/types.h>
#include <unistd.h>

#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>

#include <gtest/gtest.h>

#include "wbc_architecture/wbc_compiled_config.hpp"
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

class WbcConfigCompilerTest : public ::testing::Test {
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

TEST_F(WbcConfigCompilerTest, MotionTaskOrderFollowsPriorityAndYamlOrderForTies) {
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
    motion_tasks:
      - name: t2
        priority: 0
      - name: t1
        priority: 0
      - name: t3
        priority: 1
)");

  const auto compiler = WbcConfigCompiler::Compile(robot_.get(), yaml_path.string());
  const CompiledState& state = compiler->State(1);
  TaskRegistry* registry = compiler->TaskRegistryPtr();
  ASSERT_NE(registry, nullptr);
  ASSERT_EQ(state.motion.size(), 3u);

  EXPECT_EQ(state.motion[0], registry->GetMotionTask("t2"));
  EXPECT_EQ(state.motion[1], registry->GetMotionTask("t1"));
  EXPECT_EQ(state.motion[2], registry->GetMotionTask("t3"));
}

TEST_F(WbcConfigCompilerTest, ThrowsOnMissingContactForForceTask) {
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
    motion_tasks:
      - name: any
)");

  EXPECT_THROW(
      {
        const auto compiler =
            WbcConfigCompiler::Compile(robot_.get(), yaml_path.string());
        (void)compiler;
      },
      std::runtime_error);
}

TEST_F(WbcConfigCompilerTest, ThrowsOnMissingMotionTaskReferenceInState) {
  const auto yaml_path = WriteYaml(
      "missing_motion_ref.yaml",
      R"(
task_pool:
  - name: existing_task
    type: JointTask
state_machine:
  - id: 1
    name: s1
    motion_tasks:
      - name: missing_task
)");

  EXPECT_THROW(
      {
        const auto compiler =
            WbcConfigCompiler::Compile(robot_.get(), yaml_path.string());
        (void)compiler;
      },
      std::runtime_error);
}

TEST_F(WbcConfigCompilerTest, ScalarGainBroadcastAppliesAcrossTaskDimension) {
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
    motion_tasks:
      - name: joint_task
)");

  const auto compiler = WbcConfigCompiler::Compile(robot_.get(), yaml_path.string());
  TaskRegistry* registry = compiler->TaskRegistryPtr();
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

TEST_F(WbcConfigCompilerTest, ThrowsOnGainDimensionMismatch) {
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
    motion_tasks:
      - name: joint_task
)");

  EXPECT_THROW(
      {
        const auto compiler =
            WbcConfigCompiler::Compile(robot_.get(), yaml_path.string());
        (void)compiler;
      },
      std::runtime_error);
}

TEST_F(WbcConfigCompilerTest, StartStateIdFallbackPriority) {
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
    motion_tasks: [{name: j}]
  - id: 2
    name: s2
    motion_tasks: [{name: j}]
  - id: 3
    name: s3
    motion_tasks: [{name: j}]
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
    motion_tasks: [{name: j}]
  - id: 2
    name: s2
    motion_tasks: [{name: j}]
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
    motion_tasks: [{name: j}]
  - id: 8
    name: s8
    motion_tasks: [{name: j}]
)");

  const auto c1 =
      WbcConfigCompiler::Compile(robot_.get(), top_level_start_yaml.string());
  const auto c2 =
      WbcConfigCompiler::Compile(robot_.get(), solver_start_yaml.string());
  const auto c3 =
      WbcConfigCompiler::Compile(robot_.get(), first_state_yaml.string());

  EXPECT_EQ(c1->StartStateId(), 3);
  EXPECT_EQ(c2->StartStateId(), 2);
  EXPECT_EQ(c3->StartStateId(), 7);
}

} // namespace
} // namespace wbc

