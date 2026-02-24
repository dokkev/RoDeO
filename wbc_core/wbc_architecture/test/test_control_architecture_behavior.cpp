#include <sys/types.h>
#include <unistd.h>

#include <filesystem>
#include <fstream>
#include <memory>
#include <limits>
#include <stdexcept>
#include <string>

#include <gtest/gtest.h>

#include "wbc_architecture/interface/control_architecture.hpp"
#include "wbc_fsm/state_factory.hpp"
#include "wbc_robot_system/pinocchio_robot_system.hpp"

namespace wbc {
namespace {

class UnitStayState final : public StateMachine {
public:
  UnitStayState(StateId id, const std::string& name, PinocchioRobotSystem* robot,
                TaskRegistry* task_reg, ConstraintRegistry* const_reg,
                StateProvider* state_provider)
      : StateMachine(id, name, robot, task_reg, const_reg, state_provider) {}

  void FirstVisit() override {}
  void OneStep() override {}
  void LastVisit() override {}
  bool EndOfState() override { return false; }
  StateId GetNextState() override { return id(); }
};

class UnitLatchState final : public StateMachine {
public:
  UnitLatchState(StateId id, const std::string& name, PinocchioRobotSystem* robot,
                 TaskRegistry* task_reg, ConstraintRegistry* const_reg,
                 StateProvider* state_provider)
      : StateMachine(id, name, robot, task_reg, const_reg, state_provider) {}

  void FirstVisit() override {}
  void OneStep() override {}
  void LastVisit() override {}
  bool EndOfState() override { return true; }
  StateId GetNextState() override { return 999; }
};

WBC_REGISTER_STATE(
    "ut_home_state",
    [](StateId id, const std::string& state_name,
       const StateBuildContext& context) -> std::unique_ptr<StateMachine> {
      return std::make_unique<UnitStayState>(
          id, state_name, context.robot, context.task_registry,
          context.constraint_registry, context.state_provider);
    });

WBC_REGISTER_STATE(
    "ut_teleop_state",
    [](StateId id, const std::string& state_name,
       const StateBuildContext& context) -> std::unique_ptr<StateMachine> {
      return std::make_unique<UnitStayState>(
          id, state_name, context.robot, context.task_registry,
          context.constraint_registry, context.state_provider);
    });

WBC_REGISTER_STATE(
    "ut_latch_state",
    [](StateId id, const std::string& state_name,
       const StateBuildContext& context) -> std::unique_ptr<StateMachine> {
      return std::make_unique<UnitLatchState>(
          id, state_name, context.robot, context.task_registry,
          context.constraint_registry, context.state_provider);
    });

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

class ControlArchitectureBehaviorTest : public ::testing::Test {
protected:
  void SetUp() override {
    const std::string suffix =
        std::to_string(static_cast<long long>(::getpid())) + "_" +
        std::to_string(counter_++);
    temp_dir_ = std::filesystem::temp_directory_path() /
                ("control_arch_behavior_test_" + suffix);
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

  std::unique_ptr<ControlArchitecture> MakeArchitecture(
      const std::filesystem::path& yaml_path) {
    return BuildControlArchitecture(
        robot_.get(), yaml_path.string(), 0.001,
        std::make_unique<StateProvider>(0.001), std::make_unique<FSMHandler>());
  }

  Eigen::VectorXd ActuatedGravity() const {
    const Eigen::VectorXd& grav = robot_->GetGravityRef();
    const int n_act = robot_->NumActiveDof();
    if (grav.size() == n_act) {
      return grav;
    }
    return grav.tail(n_act);
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

TEST_F(ControlArchitectureBehaviorTest, RequestStateByNameSwitchesState) {
  const auto yaml_path = WriteYaml(
      "state_switch.yaml",
      R"(
start_state_id: 1
task_pool:
  - name: ee_pos
    type: LinkPosTask
    link_name: link2
  - name: ee_ori
    type: LinkOriTask
    link_name: link2
  - name: jpos_task
    type: JointTask
state_machine:
  - id: 1
    name: ut_home_state
    motion_tasks: [{name: ee_pos}, {name: ee_ori}, {name: jpos_task}]
  - id: 2
    name: ut_teleop_state
    motion_tasks: [{name: ee_pos}, {name: ee_ori}, {name: jpos_task}]
)");

  auto arch = MakeArchitecture(yaml_path);
  ASSERT_NE(arch, nullptr);

  const Eigen::VectorXd q = Eigen::Vector2d::Zero();
  const Eigen::VectorXd qdot = Eigen::Vector2d::Zero();

  arch->Update(q, qdot, 0.0, 0.001);
  EXPECT_EQ(arch->CurrentStateId(), 1);

  EXPECT_TRUE(arch->RequestStateByName("ut_teleop_state"));
  EXPECT_FALSE(arch->RequestStateByName("no_such_state"));
  arch->Update(q, qdot, 0.001, 0.001);
  EXPECT_EQ(arch->CurrentStateId(), 2);
}

TEST_F(ControlArchitectureBehaviorTest,
       TeleopCommandAppliesOnlyInTeleopState) {
  const auto yaml_path = WriteYaml(
      "teleop_gate.yaml",
      R"(
start_state_id: 1
task_pool:
  - name: ee_pos
    type: LinkPosTask
    link_name: link2
  - name: ee_ori
    type: LinkOriTask
    link_name: link2
  - name: jpos_task
    type: JointTask
state_machine:
  - id: 1
    name: ut_home_state
    motion_tasks: [{name: ee_pos}, {name: ee_ori}, {name: jpos_task}]
  - id: 2
    name: ut_teleop_state
    motion_tasks: [{name: ee_pos}, {name: ee_ori}, {name: jpos_task}]
)");

  auto arch = MakeArchitecture(yaml_path);
  ASSERT_NE(arch, nullptr);
  ASSERT_NE(arch->compiled_config(), nullptr);

  TaskReference teleop;
  teleop.x_des = Eigen::Vector3d(0.2, -0.3, 0.4);
  teleop.quat_des = Eigen::Quaterniond(0.7, 0.2, 0.1, -0.3); // not unit
  teleop.joint_pos = Eigen::Vector2d(0.5, -0.25);
  arch->SetTeleopCommand(teleop);

  const Eigen::VectorXd q = Eigen::Vector2d::Zero();
  const Eigen::VectorXd qdot = Eigen::Vector2d::Zero();
  arch->Update(q, qdot, 0.0, 0.001);

  const CompiledState* s_home = arch->compiled_config()->FindState(1);
  ASSERT_NE(s_home, nullptr);
  ASSERT_NE(s_home->ee_pos, nullptr);
  ASSERT_NE(s_home->ee_ori, nullptr);
  ASSERT_NE(s_home->joint, nullptr);
  EXPECT_TRUE(s_home->ee_pos->DesiredPos().isApprox(Eigen::Vector3d::Zero()));
  EXPECT_TRUE(s_home->joint->DesiredPos().isApprox(Eigen::Vector2d::Zero()));

  arch->RequestStateByName("ut_teleop_state");
  arch->Update(q, qdot, 0.001, 0.001);

  const CompiledState* s_teleop = arch->compiled_config()->FindState(2);
  ASSERT_NE(s_teleop, nullptr);
  ASSERT_NE(s_teleop->ee_pos, nullptr);
  ASSERT_NE(s_teleop->ee_ori, nullptr);
  ASSERT_NE(s_teleop->joint, nullptr);

  EXPECT_TRUE(s_teleop->ee_pos->DesiredPos().isApprox(*teleop.x_des, 1.0e-12));
  EXPECT_TRUE(
      s_teleop->joint->DesiredPos().isApprox(*teleop.joint_pos, 1.0e-12));

  const Eigen::Quaterniond qn = teleop.quat_des->normalized();
  Eigen::Vector4d quat_xyzw;
  quat_xyzw << qn.x(), qn.y(), qn.z(), qn.w();
  EXPECT_TRUE(s_teleop->ee_ori->DesiredPos().isApprox(quat_xyzw, 1.0e-12));
}

TEST_F(ControlArchitectureBehaviorTest,
       TeleopReferenceFrameDefaultsToBaseFrameAndSupportsOverride) {
  const auto yaml_path = WriteYaml(
      "teleop_reference_frame.yaml",
      R"(
start_state_id: 1
robot_model:
  base_frame: link1
task_pool:
  - name: ee_pos
    type: LinkPosTask
    link_name: link2
  - name: ee_ori
    type: LinkOriTask
    link_name: link2
state_machine:
  - id: 1
    name: ut_teleop_state
    motion_tasks: [{name: ee_pos}, {name: ee_ori}]
)");

  auto arch = MakeArchitecture(yaml_path);
  ASSERT_NE(arch, nullptr);
  ASSERT_NE(arch->compiled_config(), nullptr);

  Eigen::Vector2d q;
  q << 1.5707963267948966, 0.0;
  const Eigen::Vector2d qdot = Eigen::Vector2d::Zero();

  TaskReference local_ref;
  local_ref.x_des = Eigen::Vector3d(1.0, 0.0, 0.0);
  local_ref.quat_des =
      Eigen::Quaterniond(Eigen::AngleAxisd(0.3, Eigen::Vector3d::UnitX()));
  arch->SetTeleopCommand(local_ref);
  arch->Update(q, qdot, 0.0, 0.001);

  const CompiledState* state = arch->compiled_config()->FindState(1);
  ASSERT_NE(state, nullptr);
  ASSERT_NE(state->ee_pos, nullptr);
  ASSERT_NE(state->ee_ori, nullptr);

  const Eigen::Isometry3d world_iso_base = robot_->GetLinkIsometry("link1");
  const Eigen::Vector3d expected_pos =
      world_iso_base.linear() * (*local_ref.x_des) + world_iso_base.translation();
  EXPECT_TRUE(state->ee_pos->DesiredPos().isApprox(expected_pos, 1.0e-12));

  const Eigen::Quaterniond expected_quat =
      (Eigen::Quaterniond(world_iso_base.linear()) *
       local_ref.quat_des->normalized())
          .normalized();
  Eigen::Vector4d expected_quat_xyzw;
  expected_quat_xyzw << expected_quat.x(), expected_quat.y(), expected_quat.z(),
      expected_quat.w();
  EXPECT_TRUE(
      state->ee_ori->DesiredPos().isApprox(expected_quat_xyzw, 1.0e-12));

  TaskReference world_ref;
  world_ref.reference_frame = "world";
  world_ref.x_des = Eigen::Vector3d(0.1, -0.2, 0.3);
  world_ref.quat_des = Eigen::Quaterniond(0.9, 0.1, -0.2, 0.3);
  arch->SetTeleopCommand(world_ref);
  arch->Update(q, qdot, 0.001, 0.001);

  EXPECT_TRUE(state->ee_pos->DesiredPos().isApprox(*world_ref.x_des, 1.0e-12));
  const Eigen::Quaterniond qn = world_ref.quat_des->normalized();
  Eigen::Vector4d world_quat_xyzw;
  world_quat_xyzw << qn.x(), qn.y(), qn.z(), qn.w();
  EXPECT_TRUE(state->ee_ori->DesiredPos().isApprox(world_quat_xyzw, 1.0e-12));
}

TEST_F(ControlArchitectureBehaviorTest,
       SafeCommandIsLatchedWhenTransitionTargetMissing) {
  const auto yaml_path = WriteYaml(
      "safe_on_missing_next.yaml",
      R"(
start_state_id: 1
task_pool:
  - name: jpos_task
    type: JointTask
state_machine:
  - id: 1
    name: ut_latch_state
    motion_tasks: [{name: jpos_task}]
)");

  auto arch = MakeArchitecture(yaml_path);
  ASSERT_NE(arch, nullptr);

  Eigen::Vector2d q;
  q << 0.15, -0.35;
  Eigen::Vector2d qdot = Eigen::Vector2d::Zero();
  arch->Update(q, qdot, 0.0, 0.001);

  EXPECT_EQ(arch->CurrentStateId(), -1);
  const RobotCommand cmd = arch->GetCommand();
  ASSERT_EQ(cmd.q.size(), 2);
  ASSERT_EQ(cmd.qdot.size(), 2);
  ASSERT_EQ(cmd.tau.size(), 2);
  EXPECT_TRUE(cmd.q.isApprox(q, 1.0e-12));
  EXPECT_TRUE(cmd.qdot.isApprox(Eigen::Vector2d::Zero(), 1.0e-12));
  EXPECT_TRUE(cmd.tau.isApprox(Eigen::Vector2d::Zero(), 1.0e-12));
}

TEST_F(ControlArchitectureBehaviorTest, GravityComp_JointHold) {
  const auto yaml_path = WriteYaml(
      "gravity_joint_hold.yaml",
      R"(
start_state_id: 1
task_pool:
  - name: jpos_task
    type: JointTask
state_machine:
  - id: 1
    name: ut_teleop_state
    motion_tasks: [{name: jpos_task}]
)");

  auto arch = MakeArchitecture(yaml_path);
  ASSERT_NE(arch, nullptr);

  Eigen::Vector2d q;
  q << 0.4, -0.6;
  const Eigen::Vector2d qdot = Eigen::Vector2d::Zero();

  TaskReference teleop;
  teleop.joint_pos = q;
  arch->SetTeleopCommand(teleop);
  arch->Update(q, qdot, 0.0, 0.001);

  const RobotCommand cmd = arch->GetCommand();
  ASSERT_EQ(cmd.tau.size(), 2);
  const Eigen::VectorXd expected_grav = ActuatedGravity();
  ASSERT_EQ(expected_grav.size(), 2);
  EXPECT_TRUE(cmd.tau.isApprox(expected_grav, 1.0e-5));
}

TEST_F(ControlArchitectureBehaviorTest, GravityComp_EEHold) {
  const auto yaml_path = WriteYaml(
      "gravity_ee_hold.yaml",
      R"(
start_state_id: 1
task_pool:
  - name: ee_pos
    type: LinkPosTask
    link_name: link2
  - name: ee_ori
    type: LinkOriTask
    link_name: link2
  - name: jpos_task
    type: JointTask
state_machine:
  - id: 1
    name: ut_teleop_state
    motion_tasks:
      - {name: ee_pos, priority: 0}
      - {name: ee_ori, priority: 1}
      - {name: jpos_task, priority: 2}
)");

  auto arch = MakeArchitecture(yaml_path);
  ASSERT_NE(arch, nullptr);

  Eigen::Vector2d q;
  q << -0.5, 0.7;
  const Eigen::Vector2d qdot = Eigen::Vector2d::Zero();

  robot_->UpdateRobotModel(Eigen::Vector3d::Zero(), Eigen::Quaterniond::Identity(),
                           Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), q,
                           qdot, false);
  const Eigen::Isometry3d ee_iso = robot_->GetLinkIsometry("link2");

  TaskReference teleop;
  teleop.x_des = ee_iso.translation();
  teleop.quat_des = Eigen::Quaterniond(ee_iso.linear());
  teleop.joint_pos = q;
  arch->SetTeleopCommand(teleop);

  arch->Update(q, qdot, 0.0, 0.001);
  const RobotCommand cmd = arch->GetCommand();

  ASSERT_EQ(cmd.tau.size(), 2);
  const Eigen::VectorXd expected_grav = ActuatedGravity();
  ASSERT_EQ(expected_grav.size(), 2);
  EXPECT_TRUE(cmd.tau.isApprox(expected_grav, 5.0e-4));
}

TEST_F(ControlArchitectureBehaviorTest, FailureFallback_HoldPrevTorque) {
  const auto yaml_path = WriteYaml(
      "fallback_hold_prev.yaml",
      R"(
start_state_id: 1
task_pool:
  - name: jpos_task
    type: JointTask
state_machine:
  - id: 1
    name: ut_teleop_state
    motion_tasks: [{name: jpos_task}]
)");

  auto arch = MakeArchitecture(yaml_path);
  ASSERT_NE(arch, nullptr);

  Eigen::Vector2d q;
  q << 0.3, -0.2;
  const Eigen::Vector2d qdot = Eigen::Vector2d::Zero();

  TaskReference valid;
  valid.joint_pos = q;
  arch->SetTeleopCommand(valid);
  arch->Update(q, qdot, 0.0, 0.001);
  const RobotCommand cmd_ok = arch->GetCommand();
  ASSERT_EQ(cmd_ok.tau.size(), 2);

  TaskReference invalid;
  invalid.joint_pos =
      Eigen::Vector2d(std::numeric_limits<double>::quiet_NaN(), 0.0);
  arch->SetTeleopCommand(invalid);
  arch->Update(q, qdot, 0.001, 0.001);
  const RobotCommand cmd_hold = arch->GetCommand();
  EXPECT_TRUE(cmd_hold.tau.isApprox(cmd_ok.tau, 1.0e-12));
  EXPECT_TRUE(cmd_hold.q.isApprox(q, 1.0e-12));
  EXPECT_TRUE(cmd_hold.qdot.isApprox(Eigen::Vector2d::Zero(), 1.0e-12));

  arch->SetHoldPreviousTorqueOnFailure(false);
  arch->Update(q, qdot, 0.002, 0.001);
  const RobotCommand cmd_zero = arch->GetCommand();
  EXPECT_TRUE(cmd_zero.tau.isApprox(Eigen::Vector2d::Zero(), 1.0e-12));
  EXPECT_TRUE(cmd_zero.q.isApprox(q, 1.0e-12));
  EXPECT_TRUE(cmd_zero.qdot.isApprox(Eigen::Vector2d::Zero(), 1.0e-12));
}

TEST_F(ControlArchitectureBehaviorTest, GravityComp_JointHold_FloatingBase) {
  const auto yaml_path = WriteYaml(
      "gravity_joint_hold_floating.yaml",
      R"(
start_state_id: 1
task_pool:
  - name: jpos_task
    type: JointTask
state_machine:
  - id: 1
    name: ut_teleop_state
    motion_tasks: [{name: jpos_task}]
)");

  auto floating_robot = std::make_unique<PinocchioRobotSystem>(
      urdf_path_.string(), "", false, false, nullptr);
  auto arch = BuildControlArchitecture(
      floating_robot.get(), yaml_path.string(), 0.001,
      std::make_unique<StateProvider>(0.001), std::make_unique<FSMHandler>());
  ASSERT_NE(arch, nullptr);

  Eigen::Vector2d q;
  q << -0.2, 0.55;
  const Eigen::Vector2d qdot = Eigen::Vector2d::Zero();

  TaskReference teleop;
  teleop.joint_pos = q;
  arch->SetTeleopCommand(teleop);
  arch->Update(q, qdot, 0.0, 0.001);

  const RobotCommand cmd = arch->GetCommand();
  ASSERT_EQ(cmd.tau.size(), floating_robot->NumActiveDof());
  const Eigen::VectorXd expected_grav =
      floating_robot->GetGravityRef().tail(floating_robot->NumActiveDof());
  EXPECT_TRUE(cmd.tau.isApprox(expected_grav, 1.0e-5));
}

TEST_F(ControlArchitectureBehaviorTest, EmptyMotionStateIsRejectedByCompiler) {
  const auto yaml_path = WriteYaml(
      "empty_motion_state.yaml",
      R"(
start_state_id: 1
state_machine:
  - id: 1
    name: ut_home_state
)");

  EXPECT_THROW((void)MakeArchitecture(yaml_path), std::runtime_error);
}

TEST_F(ControlArchitectureBehaviorTest, UpdateThrowsOnActuatedDimensionMismatch) {
  const auto yaml_path = WriteYaml(
      "update_dim_guard.yaml",
      R"(
start_state_id: 1
task_pool:
  - name: jpos_task
    type: JointTask
state_machine:
  - id: 1
    name: ut_home_state
    motion_tasks: [{name: jpos_task}]
)");

  auto arch = MakeArchitecture(yaml_path);
  ASSERT_NE(arch, nullptr);

  const Eigen::VectorXd q_wrong = Eigen::VectorXd::Zero(1);
  const Eigen::VectorXd qdot_wrong = Eigen::VectorXd::Zero(1);
  EXPECT_THROW(arch->Update(q_wrong, qdot_wrong, 0.0, 0.001),
               std::runtime_error);
}

} // namespace
} // namespace wbc
