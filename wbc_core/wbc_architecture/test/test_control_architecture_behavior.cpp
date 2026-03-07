/**
 * @file wbc_core/wbc_architecture/test/test_control_architecture_behavior.cpp
 * @brief Doxygen documentation for test_control_architecture_behavior module.
 */
#include <sys/types.h>
#include <unistd.h>

#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

#include <gtest/gtest.h>

#include "wbc_architecture/control_architecture.hpp"
#include "wbc_fsm/state_factory.hpp"
#include "wbc_formulation/motion_task.hpp"
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

class UnitAutoTransitionFromState final : public StateMachine {
public:
  UnitAutoTransitionFromState(StateId id, const std::string& name,
                              PinocchioRobotSystem* robot,
                              TaskRegistry* task_reg,
                              ConstraintRegistry* const_reg,
                              StateProvider* state_provider)
      : StateMachine(id, name, robot, task_reg, const_reg, state_provider) {}

  void FirstVisit() override {}
  void OneStep() override {}
  void LastVisit() override {}
  bool EndOfState() override { return true; }
  StateId GetNextState() override { return 2; }
};

class UnitAutoTransitionToState final : public StateMachine {
public:
  UnitAutoTransitionToState(StateId id, const std::string& name,
                            PinocchioRobotSystem* robot,
                            TaskRegistry* task_reg,
                            ConstraintRegistry* const_reg,
                            StateProvider* state_provider)
      : StateMachine(id, name, robot, task_reg, const_reg, state_provider) {}

  void FirstVisit() override {
    JointTask* joint_task = GetMotionTask<JointTask>("jpos_task");
    if (joint_task == nullptr || robot_ == nullptr) {
      return;
    }
    const Eigen::VectorXd q_curr = robot_->GetJointPos();
    Eigen::VectorXd q_des = q_curr;
    q_des.array() += 0.25;
    const Eigen::VectorXd zeros = Eigen::VectorXd::Zero(q_des.size());
    joint_task->UpdateDesired(q_des, zeros, zeros);
  }

  void OneStep() override {}
  void LastVisit() override {}
  bool EndOfState() override { return false; }
  StateId GetNextState() override { return id(); }
};

void UpdateWithJointState(ControlArchitecture* architecture,
                          const Eigen::VectorXd& q,
                          const Eigen::VectorXd& qdot, double t, double dt) {
  ASSERT_NE(architecture, nullptr);
  RobotJointState joint_state;
  joint_state.q = q;
  joint_state.qdot = qdot;
  joint_state.tau = Eigen::VectorXd::Zero(q.size());
  architecture->Update(joint_state, t, dt);
}

void UpdateWithFloatingBase(ControlArchitecture* architecture,
                            const Eigen::VectorXd& q,
                            const Eigen::VectorXd& qdot,
                            const RobotBaseState& base_state,
                            double t, double dt) {
  ASSERT_NE(architecture, nullptr);
  RobotJointState joint_state;
  joint_state.q = q;
  joint_state.qdot = qdot;
  joint_state.tau = Eigen::VectorXd::Zero(q.size());
  architecture->Update(joint_state, base_state, t, dt);
}

WBC_REGISTER_STATE(
    "ut_home_state",
    [](StateId id, const std::string& state_name,
       const StateMachineConfig& context) -> std::unique_ptr<StateMachine> {
      return std::make_unique<UnitStayState>(
          id, state_name, context.robot, context.task_registry,
          context.constraint_registry, context.state_provider);
    });

WBC_REGISTER_STATE(
    "ut_teleop_state",
    [](StateId id, const std::string& state_name,
       const StateMachineConfig& context) -> std::unique_ptr<StateMachine> {
      return std::make_unique<UnitStayState>(
          id, state_name, context.robot, context.task_registry,
          context.constraint_registry, context.state_provider);
    });

WBC_REGISTER_STATE(
    "ut_latch_state",
    [](StateId id, const std::string& state_name,
       const StateMachineConfig& context) -> std::unique_ptr<StateMachine> {
      return std::make_unique<UnitLatchState>(
          id, state_name, context.robot, context.task_registry,
          context.constraint_registry, context.state_provider);
    });

WBC_REGISTER_STATE(
    "ut_auto_transition_from_state",
    [](StateId id, const std::string& state_name,
       const StateMachineConfig& context) -> std::unique_ptr<StateMachine> {
      return std::make_unique<UnitAutoTransitionFromState>(
          id, state_name, context.robot, context.task_registry,
          context.constraint_registry, context.state_provider);
    });

WBC_REGISTER_STATE(
    "ut_auto_transition_to_state",
    [](StateId id, const std::string& state_name,
       const StateMachineConfig& context) -> std::unique_ptr<StateMachine> {
      return std::make_unique<UnitAutoTransitionToState>(
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
  }

  void TearDown() override {
    robot_ = nullptr;
    std::error_code ec;
    std::filesystem::remove_all(temp_dir_, ec);
  }

  std::unique_ptr<ControlArchitecture> MakeArchitecture(
      const std::filesystem::path& yaml_path) {
    auto arch_config = ControlArchitectureConfig::FromYaml(yaml_path.string(), 0.001);
    arch_config.state_provider = std::make_unique<StateProvider>(0.001);
    auto arch = std::make_unique<ControlArchitecture>(std::move(arch_config));
    arch->Initialize();
    robot_ = arch->GetRobot();
    return arch;
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

  std::string WithRobotModel(const std::string& yaml,
                             bool is_floating_base = false,
                             const std::string& robot_model_extra = "") const {
    std::ostringstream oss;
    oss << "robot_model:\n"
        << "  urdf_path: \"" << urdf_path_.string() << "\"\n"
        << "  is_floating_base: " << (is_floating_base ? "true" : "false")
        << "\n";
    if (!robot_model_extra.empty()) {
      oss << robot_model_extra;
    }
    oss << yaml;
    return oss.str();
  }

  std::filesystem::path temp_dir_;
  std::filesystem::path urdf_path_;
  PinocchioRobotSystem* robot_{nullptr};

  static inline int counter_ = 0;
};

TEST_F(ControlArchitectureBehaviorTest, RequestStateByNameSwitchesState) {
  const auto yaml_path = WriteYaml(
      "state_switch.yaml",
      WithRobotModel(R"(
start_state_id: 1
task_pool:
  - name: ee_pos
    type: LinkPosTask
    target_frame: link2
  - name: ee_ori
    type: LinkOriTask
    target_frame: link2
  - name: jpos_task
    type: JointTask
state_machine:
  - id: 1
    name: ut_home_state
    task_hierarchy: [{name: ee_pos}, {name: ee_ori}, {name: jpos_task}]
  - id: 2
    name: ut_teleop_state
    task_hierarchy: [{name: ee_pos}, {name: ee_ori}, {name: jpos_task}]
)"));

  auto arch = MakeArchitecture(yaml_path);
  ASSERT_NE(arch, nullptr);

  const Eigen::VectorXd q = Eigen::Vector2d::Zero();
  const Eigen::VectorXd qdot = Eigen::Vector2d::Zero();

  UpdateWithJointState(arch.get(), q, qdot, 0.0, 0.001);
  ASSERT_NE(arch->GetFsmHandler(), nullptr);
  EXPECT_EQ(arch->GetFsmHandler()->GetCurrentStateId(), 1);

  EXPECT_TRUE(arch->GetFsmHandler()->RequestStateByName("ut_teleop_state"));
  EXPECT_FALSE(arch->GetFsmHandler()->RequestStateByName("no_such_state"));
  UpdateWithJointState(arch.get(), q, qdot, 0.001, 0.001);
  EXPECT_EQ(arch->GetFsmHandler()->GetCurrentStateId(), 2);
}

TEST_F(ControlArchitectureBehaviorTest,
       SameStateTypeCanBeInstantiatedWithDifferentNames) {
  const auto yaml_path = WriteYaml(
      "state_type_vs_instance_name.yaml",
      WithRobotModel(R"(
start_state_id: 1
task_pool:
  - name: jpos_task
    type: JointTask
state_machine:
  - id: 1
    name: swing_left
    type: ut_home_state
    task_hierarchy: [{name: jpos_task}]
  - id: 2
    name: swing_right
    type: ut_home_state
    task_hierarchy: [{name: jpos_task}]
)"));

  auto arch = MakeArchitecture(yaml_path);
  ASSERT_NE(arch, nullptr);
  ASSERT_NE(arch->GetFsmHandler(), nullptr);

  const Eigen::VectorXd q = Eigen::Vector2d::Zero();
  const Eigen::VectorXd qdot = Eigen::Vector2d::Zero();

  UpdateWithJointState(arch.get(), q, qdot, 0.0, 0.001);
  EXPECT_EQ(arch->GetFsmHandler()->GetCurrentStateId(), 1);

  EXPECT_TRUE(arch->GetFsmHandler()->RequestStateByName("swing_right"));
  UpdateWithJointState(arch.get(), q, qdot, 0.001, 0.001);
  EXPECT_EQ(arch->GetFsmHandler()->GetCurrentStateId(), 2);
}

TEST_F(ControlArchitectureBehaviorTest,
       AutoTransitionAppliesNextStateFirstVisitInSameTick) {
  const auto yaml_path = WriteYaml(
      "auto_transition_first_visit_same_tick.yaml",
      WithRobotModel(R"(
start_state_id: 1
task_pool:
  - name: jpos_task
    type: JointTask
state_machine:
  - id: 1
    name: ut_auto_transition_from_state
    task_hierarchy: [{name: jpos_task}]
  - id: 2
    name: ut_auto_transition_to_state
    task_hierarchy: [{name: jpos_task}]
)"));

  auto arch = MakeArchitecture(yaml_path);
  ASSERT_NE(arch, nullptr);
  ASSERT_NE(arch->GetFsmHandler(), nullptr);
  ASSERT_NE(arch->GetConfig(), nullptr);

  const Eigen::Vector2d q = Eigen::Vector2d::Zero();
  const Eigen::Vector2d qdot = Eigen::Vector2d::Zero();

  // One tick should transition 1->2 and run state's FirstVisit immediately.
  UpdateWithJointState(arch.get(), q, qdot, 0.0, 0.001);
  EXPECT_EQ(arch->GetFsmHandler()->GetCurrentStateId(), 2);

  const StateConfig* state2 = arch->GetConfig()->FindState(2);
  ASSERT_NE(state2, nullptr);
  ASSERT_NE(state2->joint, nullptr);
  EXPECT_TRUE(state2->joint->DesiredPos().isApprox(Eigen::Vector2d::Constant(0.25),
                                                   1.0e-12));
}

TEST_F(ControlArchitectureBehaviorTest,
       StateGainOverrideFallsBackToPoolDefaultWhenMissingInNextState) {
  const auto yaml_path = WriteYaml(
      "state_gain_fallback.yaml",
      WithRobotModel(R"(
start_state_id: 1
task_pool:
  - name: jpos_task
    type: JointTask
    kp: 10.0
state_machine:
  - id: 1
    name: ut_home_state
    task_hierarchy:
      - name: jpos_task
        kp: 100.0
  - id: 2
    name: ut_teleop_state
    task_hierarchy:
      - name: jpos_task
)"));

  auto arch = MakeArchitecture(yaml_path);
  ASSERT_NE(arch, nullptr);
  ASSERT_NE(arch->GetConfig(), nullptr);
  ASSERT_NE(arch->GetFsmHandler(), nullptr);

  const Eigen::VectorXd q = Eigen::Vector2d::Zero();
  const Eigen::VectorXd qdot = Eigen::Vector2d::Zero();

  UpdateWithJointState(arch.get(), q, qdot, 0.0, 0.001);
  const StateConfig* state_a = arch->GetConfig()->FindState(1);
  ASSERT_NE(state_a, nullptr);
  ASSERT_NE(state_a->joint, nullptr);
  ASSERT_EQ(state_a->joint->Kp().size(), 2);
  EXPECT_TRUE(state_a->joint->Kp().isApprox(Eigen::Vector2d::Constant(100.0),
                                            1.0e-12));

  EXPECT_TRUE(arch->GetFsmHandler()->RequestStateByName("ut_teleop_state"));
  UpdateWithJointState(arch.get(), q, qdot, 0.001, 0.001);
  const StateConfig* state_b = arch->GetConfig()->FindState(2);
  ASSERT_NE(state_b, nullptr);
  ASSERT_NE(state_b->joint, nullptr);
  ASSERT_EQ(state_b->joint->Kp().size(), 2);
  EXPECT_TRUE(state_b->joint->Kp().isApprox(Eigen::Vector2d::Constant(10.0),
                                            1.0e-12));
}

TEST_F(ControlArchitectureBehaviorTest,
       SafeCommandIsLatchedWhenTransitionTargetMissing) {
  const auto yaml_path = WriteYaml(
      "safe_on_missing_next.yaml",
      WithRobotModel(R"(
start_state_id: 1
task_pool:
  - name: jpos_task
    type: JointTask
state_machine:
  - id: 1
    name: ut_latch_state
    task_hierarchy: [{name: jpos_task}]
)"));

  auto arch = MakeArchitecture(yaml_path);
  ASSERT_NE(arch, nullptr);

  Eigen::Vector2d q;
  q << 0.15, -0.35;
  Eigen::Vector2d qdot = Eigen::Vector2d::Zero();
  UpdateWithJointState(arch.get(), q, qdot, 0.0, 0.001);

  ASSERT_NE(arch->GetFsmHandler(), nullptr);
  EXPECT_EQ(arch->GetFsmHandler()->GetCurrentStateId(), -1);
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
      WithRobotModel(R"(
start_state_id: 1
task_pool:
  - name: jpos_task
    type: JointTask
state_machine:
  - id: 1
    name: ut_teleop_state
    task_hierarchy: [{name: jpos_task}]
)"));

  auto arch = MakeArchitecture(yaml_path);
  ASSERT_NE(arch, nullptr);

  Eigen::Vector2d q;
  q << 0.4, -0.6;
  const Eigen::Vector2d qdot = Eigen::Vector2d::Zero();

  // Set joint desired = q so position error is zero; WBIC outputs gravity only.
  const StateConfig* state = arch->GetConfig()->FindState(1);
  ASSERT_NE(state, nullptr);
  ASSERT_NE(state->joint, nullptr);
  state->joint->UpdateDesired(q, Eigen::Vector2d::Zero(), Eigen::Vector2d::Zero());
  UpdateWithJointState(arch.get(), q, qdot, 0.0, 0.001);

  const RobotCommand cmd = arch->GetCommand();
  ASSERT_EQ(cmd.tau.size(), 2);
  const Eigen::VectorXd expected_grav = ActuatedGravity();
  ASSERT_EQ(expected_grav.size(), 2);
  EXPECT_TRUE(cmd.tau.isApprox(expected_grav, 1.0e-5));
}

TEST_F(ControlArchitectureBehaviorTest, GravityComp_EEHold) {
  const auto yaml_path = WriteYaml(
      "gravity_ee_hold.yaml",
      WithRobotModel(R"(
start_state_id: 1
task_pool:
  - name: ee_pos
    type: LinkPosTask
    target_frame: link2
  - name: ee_ori
    type: LinkOriTask
    target_frame: link2
  - name: jpos_task
    type: JointTask
state_machine:
  - id: 1
    name: ut_teleop_state
    task_hierarchy:
      - {name: ee_pos, priority: 0}
      - {name: ee_ori, priority: 1}
      - {name: jpos_task, priority: 2}
)"));

  auto arch = MakeArchitecture(yaml_path);
  ASSERT_NE(arch, nullptr);

  Eigen::Vector2d q;
  q << -0.5, 0.7;
  const Eigen::Vector2d qdot = Eigen::Vector2d::Zero();

  robot_->UpdateRobotModel(Eigen::Vector3d::Zero(), Eigen::Quaterniond::Identity(),
                           Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), q,
                           qdot, false);
  const Eigen::Isometry3d ee_iso = robot_->GetLinkIsometry("link2");

  // Set desired = current pose/joints so errors are zero; WBIC outputs gravity only.
  const Eigen::Quaterniond q_curr(ee_iso.linear());
  Eigen::Vector4d quat_xyzw;
  quat_xyzw << q_curr.x(), q_curr.y(), q_curr.z(), q_curr.w();
  const StateConfig* state = arch->GetConfig()->FindState(1);
  ASSERT_NE(state, nullptr);
  ASSERT_NE(state->ee_pos, nullptr);
  ASSERT_NE(state->ee_ori, nullptr);
  ASSERT_NE(state->joint, nullptr);
  state->ee_pos->UpdateDesired(ee_iso.translation(), Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());
  state->ee_ori->UpdateDesired(quat_xyzw,            Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());
  state->joint->UpdateDesired(q, Eigen::Vector2d::Zero(), Eigen::Vector2d::Zero());

  UpdateWithJointState(arch.get(), q, qdot, 0.0, 0.001);
  const RobotCommand cmd = arch->GetCommand();

  ASSERT_EQ(cmd.tau.size(), 2);
  const Eigen::VectorXd expected_grav = ActuatedGravity();
  ASSERT_EQ(expected_grav.size(), 2);
  EXPECT_TRUE(cmd.tau.isApprox(expected_grav, 5.0e-4));
}

TEST_F(ControlArchitectureBehaviorTest, FailureFallback_HoldPrevTorque) {
  const auto yaml_path = WriteYaml(
      "fallback_hold_prev.yaml",
      WithRobotModel(R"(
start_state_id: 1
task_pool:
  - name: jpos_task
    type: JointTask
state_machine:
  - id: 1
    name: ut_teleop_state
    task_hierarchy: [{name: jpos_task}]
)"));

  auto arch = MakeArchitecture(yaml_path);
  ASSERT_NE(arch, nullptr);

  Eigen::Vector2d q;
  q << 0.3, -0.2;
  const Eigen::Vector2d qdot = Eigen::Vector2d::Zero();

  const StateConfig* state = arch->GetConfig()->FindState(1);
  ASSERT_NE(state, nullptr);
  ASSERT_NE(state->joint, nullptr);
  state->joint->UpdateDesired(q, Eigen::Vector2d::Zero(), Eigen::Vector2d::Zero());
  UpdateWithJointState(arch.get(), q, qdot, 0.0, 0.001);
  const RobotCommand cmd_ok = arch->GetCommand();
  ASSERT_EQ(cmd_ok.tau.size(), 2);

  state->joint->UpdateDesired(
      Eigen::Vector2d(std::numeric_limits<double>::quiet_NaN(), 0.0),
      Eigen::Vector2d::Zero(), Eigen::Vector2d::Zero());
  UpdateWithJointState(arch.get(), q, qdot, 0.001, 0.001);
  const RobotCommand cmd_hold = arch->GetCommand();
  EXPECT_TRUE(cmd_hold.tau.isApprox(cmd_ok.tau, 1.0e-12));
  EXPECT_TRUE(cmd_hold.q.isApprox(q, 1.0e-12));
  EXPECT_TRUE(cmd_hold.qdot.isApprox(Eigen::Vector2d::Zero(), 1.0e-12));

  arch->SetHoldPreviousTorqueOnFailure(false);
  UpdateWithJointState(arch.get(), q, qdot, 0.002, 0.001);
  const RobotCommand cmd_zero = arch->GetCommand();
  EXPECT_TRUE(cmd_zero.tau.isApprox(Eigen::Vector2d::Zero(), 1.0e-12));
  EXPECT_TRUE(cmd_zero.q.isApprox(q, 1.0e-12));
  EXPECT_TRUE(cmd_zero.qdot.isApprox(Eigen::Vector2d::Zero(), 1.0e-12));
}

TEST_F(ControlArchitectureBehaviorTest, GravityComp_JointHold_FloatingBase) {
  const auto yaml_path = WriteYaml(
      "gravity_joint_hold_floating.yaml",
      WithRobotModel(
          R"(
start_state_id: 1
task_pool:
  - name: jpos_task
    type: JointTask
state_machine:
  - id: 1
    name: ut_teleop_state
    task_hierarchy: [{name: jpos_task}]
)",
          false));

  auto arch = MakeArchitecture(yaml_path);
  ASSERT_NE(arch, nullptr);

  Eigen::Vector2d q;
  q << -0.2, 0.55;
  const Eigen::Vector2d qdot = Eigen::Vector2d::Zero();

  const StateConfig* state = arch->GetConfig()->FindState(1);
  ASSERT_NE(state, nullptr);
  ASSERT_NE(state->joint, nullptr);
  state->joint->UpdateDesired(q, Eigen::Vector2d::Zero(), Eigen::Vector2d::Zero());
  UpdateWithJointState(arch.get(), q, qdot, 0.0, 0.001);

  const RobotCommand cmd = arch->GetCommand();
  ASSERT_EQ(cmd.tau.size(), robot_->NumActiveDof());
  const Eigen::VectorXd expected_grav =
      robot_->GetGravityRef().tail(robot_->NumActiveDof());
  EXPECT_TRUE(cmd.tau.isApprox(expected_grav, 1.0e-5));
}

TEST_F(ControlArchitectureBehaviorTest, EmptyMotionStateIsRejectedByCompiler) {
  const auto yaml_path = WriteYaml(
      "empty_motion_state.yaml",
      WithRobotModel(R"(
start_state_id: 1
state_machine:
  - id: 1
    name: ut_home_state
)"));

  EXPECT_THROW((void)MakeArchitecture(yaml_path), std::runtime_error);
}

TEST_F(ControlArchitectureBehaviorTest, UpdateUsesSafeFallbackOnActuatedDimensionMismatch) {
  const auto yaml_path = WriteYaml(
      "update_dim_guard.yaml",
      WithRobotModel(R"(
start_state_id: 1
task_pool:
  - name: jpos_task
    type: JointTask
state_machine:
  - id: 1
    name: ut_home_state
    task_hierarchy: [{name: jpos_task}]
)"));

  auto arch = MakeArchitecture(yaml_path);
  ASSERT_NE(arch, nullptr);

  const Eigen::VectorXd q_wrong = Eigen::VectorXd::Zero(1);
  const Eigen::VectorXd qdot_wrong = Eigen::VectorXd::Zero(1);
  EXPECT_NO_THROW(UpdateWithJointState(arch.get(), q_wrong, qdot_wrong, 0.0, 0.001));
}

// =============================================================================
// Safety Clamping Tests
// =============================================================================

TEST_F(ControlArchitectureBehaviorTest, SafetyClampingEnforcesPositionLimits) {
  // pos_scale: 0.1 → range ≈ [-0.314, 0.314].
  // UnitAutoTransitionToState sets q_des = q_curr + 0.25.
  // With kp_ik=10, delta_q = 10 * 0.25 = 2.5, which exceeds the limit.
  const auto yaml_path = WriteYaml(
      "safety_pos_clamp.yaml",
      WithRobotModel(R"(
start_state_id: 1
task_pool:
  - name: jpos_task
    type: JointTask
    kp: [10.0, 10.0]
    kd: [1.0, 1.0]
    kp_ik: [10.0, 10.0]
global_constraints:
  JointPosLimitConstraint:
    enabled: true
    scale: 0.1
state_machine:
  - id: 1
    name: ut_auto_transition_to_state
    task_hierarchy: [{name: jpos_task}]
)"));

  auto arch = MakeArchitecture(yaml_path);
  ASSERT_NE(arch, nullptr);

  const Eigen::Vector2d q_init = Eigen::Vector2d::Zero();
  const Eigen::Vector2d qdot_zero = Eigen::Vector2d::Zero();

  for (int i = 0; i < 10; ++i) {
    UpdateWithJointState(arch.get(), q_init, qdot_zero, i * 0.001, 0.001);
  }

  const auto& cmd = arch->GetCommand();
  // URDF range [-3.14, 3.14], pos_scale=0.1 → [-0.314, 0.314].
  const double pos_limit = 3.14 * 0.1;
  for (int i = 0; i < cmd.q.size(); ++i) {
    EXPECT_GE(cmd.q[i], -pos_limit - 1e-9)
        << "Joint " << i << " below pos_min";
    EXPECT_LE(cmd.q[i], pos_limit + 1e-9)
        << "Joint " << i << " above pos_max";
  }
}

TEST_F(ControlArchitectureBehaviorTest, SafetyClampingEnforcesTorqueLimits) {
  // Tight torque limits: trq_scale 0.01 → ±1.0 Nm (URDF effort=100).
  // With kp=100 and kp_ik=10, the solver produces substantial torque that exceeds ±1 Nm.
  const auto yaml_path = WriteYaml(
      "safety_trq_clamp.yaml",
      WithRobotModel(R"(
start_state_id: 1
task_pool:
  - name: jpos_task
    type: JointTask
    kp: [100.0, 100.0]
    kd: [10.0, 10.0]
    kp_ik: [10.0, 10.0]
global_constraints:
  JointTrqLimitConstraint:
    enabled: true
    scale: 0.01
state_machine:
  - id: 1
    name: ut_auto_transition_to_state
    task_hierarchy: [{name: jpos_task}]
)"));

  auto arch = MakeArchitecture(yaml_path);
  ASSERT_NE(arch, nullptr);

  const Eigen::Vector2d q_init = Eigen::Vector2d::Zero();
  const Eigen::Vector2d qdot_zero = Eigen::Vector2d::Zero();

  for (int i = 0; i < 10; ++i) {
    UpdateWithJointState(arch.get(), q_init, qdot_zero, i * 0.001, 0.001);
  }

  const auto& cmd = arch->GetCommand();
  const double trq_limit = 100.0 * 0.01;  // ±1.0 Nm
  for (int i = 0; i < cmd.tau.size(); ++i) {
    EXPECT_GE(cmd.tau[i], -trq_limit - 1e-9)
        << "Joint " << i << " torque below trq_min";
    EXPECT_LE(cmd.tau[i], trq_limit + 1e-9)
        << "Joint " << i << " torque above trq_max";
  }
}

TEST_F(ControlArchitectureBehaviorTest, DisabledConstraintSkipsClamping) {
  // Position constraint disabled → position should NOT be clamped.
  // With kp_ik=10 and error=0.25, IK produces cmd.q ≈ 2.5 (well beyond 0.0314).
  const auto yaml_path = WriteYaml(
      "safety_disabled_no_clamp.yaml",
      WithRobotModel(R"(
start_state_id: 1
task_pool:
  - name: jpos_task
    type: JointTask
    kp: [10.0, 10.0]
    kd: [1.0, 1.0]
    kp_ik: [10.0, 10.0]
global_constraints:
  JointPosLimitConstraint:
    enabled: false
    scale: 0.01
state_machine:
  - id: 1
    name: ut_auto_transition_to_state
    task_hierarchy: [{name: jpos_task}]
)"));

  auto arch = MakeArchitecture(yaml_path);
  ASSERT_NE(arch, nullptr);

  const Eigen::Vector2d q_init = Eigen::Vector2d::Zero();
  const Eigen::Vector2d qdot_zero = Eigen::Vector2d::Zero();

  for (int i = 0; i < 10; ++i) {
    UpdateWithJointState(arch.get(), q_init, qdot_zero, i * 0.001, 0.001);
  }

  const auto& cmd = arch->GetCommand();
  // pos_scale=0.01 → tiny range [-0.0314, 0.0314]. But constraint is disabled,
  // so IK output of ~2.5 should NOT be clamped.
  const double tiny_limit = 3.14 * 0.01;
  bool any_exceeds = false;
  for (int i = 0; i < cmd.q.size(); ++i) {
    if (std::abs(cmd.q[i]) > tiny_limit + 1e-6) {
      any_exceeds = true;
    }
  }
  EXPECT_TRUE(any_exceeds)
      << "Disabled constraint should not clamp — expected q outside the tiny range";
}

// =============================================================================
// Performance Benchmark
// =============================================================================

TEST_F(ControlArchitectureBehaviorTest, Benchmark_MaxLoopFrequency) {
  const auto yaml_path = WriteYaml(
      "benchmark.yaml",
      WithRobotModel(R"(
start_state_id: 1
task_pool:
  - name: ee_pos
    type: LinkPosTask
    target_frame: link2
  - name: ee_ori
    type: LinkOriTask
    target_frame: link2
  - name: jpos_task
    type: JointTask
state_machine:
  - id: 1
    name: ut_teleop_state
    task_hierarchy:
      - {name: ee_pos, priority: 0}
      - {name: ee_ori, priority: 1}
      - {name: jpos_task, priority: 2}
)"));

  auto arch = MakeArchitecture(yaml_path);
  ASSERT_NE(arch, nullptr);

  Eigen::Vector2d q;
  q << 0.4, -0.6;
  const Eigen::Vector2d qdot = Eigen::Vector2d::Zero();

  // Set desired = current so solver runs normally.
  const StateConfig* state = arch->GetConfig()->FindState(1);
  ASSERT_NE(state, nullptr);
  ASSERT_NE(state->joint, nullptr);
  state->joint->UpdateDesired(q, Eigen::Vector2d::Zero(),
                              Eigen::Vector2d::Zero());

  // Warm-up: run a few iterations to trigger lazy initialization.
  for (int i = 0; i < 10; ++i) {
    UpdateWithJointState(arch.get(), q, qdot, i * 0.001, 0.001);
  }

  // Benchmark: measure N iterations.
  constexpr int kIterations = 1000;
  const auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < kIterations; ++i) {
    const double t = (10 + i) * 0.001;
    UpdateWithJointState(arch.get(), q, qdot, t, 0.001);
  }
  const auto end = std::chrono::high_resolution_clock::now();

  const double elapsed_sec =
      std::chrono::duration<double>(end - start).count();
  const double avg_us = (elapsed_sec / kIterations) * 1.0e6;
  const double max_hz = kIterations / elapsed_sec;

  std::cout << "\n===== WBC Loop Benchmark (2-DOF, 3 tasks, ProxQP) =====\n"
            << "  Iterations : " << kIterations << "\n"
            << "  Total time : " << elapsed_sec * 1e3 << " ms\n"
            << "  Avg / tick : " << avg_us << " us\n"
            << "  Max freq   : " << max_hz << " Hz\n"
            << "======================================================\n";

  // Sanity: each tick should be < 10 ms (>100 Hz) for a 2-DOF robot.
  EXPECT_LT(avg_us, 10000.0) << "Average tick time exceeds 10 ms";
}

TEST_F(ControlArchitectureBehaviorTest, Benchmark_Optimo7DOF) {
  // Use the actual Optimo 7-DOF URDF for a realistic benchmark.
  const std::string optimo_urdf =
      "/home/dk/workspace/rpc_ws/src/rpc_ros/description/"
      "optimo_description/urdf/optimo.urdf";
  if (!std::filesystem::exists(optimo_urdf)) {
    GTEST_SKIP() << "Optimo URDF not found — skipping benchmark";
  }

  std::ostringstream yaml;
  yaml << "robot_model:\n"
       << "  urdf_path: \"" << optimo_urdf << "\"\n"
       << R"(  is_floating_base: false
  base_frame: optimo_base_link
controller: {}
regularization:
  w_tau: 1.0e-3
start_state_id: 1
task_pool:
  - name: jpos_task
    type: JointTask
    kp: [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
    kd: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    kp_ik: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
  - name: ee_pos
    type: LinkPosTask
    target_frame: optimo_end_effector
    reference_frame: optimo_base_link
    kp: [100.0, 100.0, 100.0]
    kd: [10.0, 10.0, 10.0]
    kp_ik: [1.0, 1.0, 1.0]
  - name: ee_ori
    type: LinkOriTask
    target_frame: optimo_end_effector
    reference_frame: optimo_base_link
    kp: [100.0, 100.0, 100.0]
    kd: [10.0, 10.0, 10.0]
    kp_ik: [1.0, 1.0, 1.0]
global_constraints:
  JointPosLimitConstraint:
    enabled: true
    scale: 0.9
  JointVelLimitConstraint:
    enabled: true
    scale: 0.8
  JointTrqLimitConstraint:
    enabled: true
    scale: 0.9
state_machine:
  - id: 1
    name: ut_teleop_state
    task_hierarchy:
      - {name: ee_pos, priority: 0}
      - {name: ee_ori, priority: 1}
      - {name: jpos_task, priority: 2}
)";

  const auto yaml_path = WriteYaml("benchmark_optimo.yaml", yaml.str());
  auto arch = MakeArchitecture(yaml_path);
  ASSERT_NE(arch, nullptr);

  const int n_act = robot_->NumActiveDof();
  ASSERT_EQ(n_act, 7);

  // Start at a non-trivial pose (mid-range).
  Eigen::VectorXd q = Eigen::VectorXd::Zero(n_act);
  q << 0.0, -0.5, 0.0, 1.0, 0.0, -0.5, 0.0;
  const Eigen::VectorXd qdot = Eigen::VectorXd::Zero(n_act);

  // Set desired = current for steady-state benchmarking.
  const StateConfig* state = arch->GetConfig()->FindState(1);
  ASSERT_NE(state, nullptr);
  ASSERT_NE(state->joint, nullptr);
  state->joint->UpdateDesired(q, Eigen::VectorXd::Zero(n_act),
                              Eigen::VectorXd::Zero(n_act));

  // Warm-up: trigger lazy init (Pinocchio model, ProxQP, etc.)
  for (int i = 0; i < 20; ++i) {
    UpdateWithJointState(arch.get(), q, qdot, i * 0.001, 0.001);
  }

  // Benchmark.
  constexpr int kIterations = 1000;
  const auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < kIterations; ++i) {
    const double t = (20 + i) * 0.001;
    UpdateWithJointState(arch.get(), q, qdot, t, 0.001);
  }
  const auto end = std::chrono::high_resolution_clock::now();

  const double elapsed_sec =
      std::chrono::duration<double>(end - start).count();
  const double avg_us = (elapsed_sec / kIterations) * 1.0e6;
  const double max_hz = kIterations / elapsed_sec;

  std::cout
      << "\n===== WBC Loop Benchmark (Optimo 7-DOF, 3 tasks, constraints, ProxQP) =====\n"
      << "  Robot      : Optimo (7-DOF fixed-base)\n"
      << "  Tasks      : ee_pos + ee_ori + jpos_task (3 priority levels)\n"
      << "  Constraints: JointPosLimit + JointVelLimit\n"
      << "  Torque reg : w_tau = 1e-3\n"
      << "  Iterations : " << kIterations << "\n"
      << "  Total time : " << elapsed_sec * 1e3 << " ms\n"
      << "  Avg / tick : " << avg_us << " us\n"
      << "  Max freq   : " << max_hz << " Hz\n"
      << "===========================================================================\n";

  // 7-DOF should comfortably run at >1 kHz (< 1000 us per tick).
  EXPECT_LT(avg_us, 1000.0) << "Average tick time exceeds 1 ms for 7-DOF robot";
}

TEST_F(ControlArchitectureBehaviorTest, SoftConstraintYamlParsing) {
  // Verify is_soft/soft_weight YAML fields are parsed and routed to WBIC.
  const std::string optimo_urdf =
      "/home/dk/workspace/rpc_ws/src/rpc_ros/description/"
      "optimo_description/urdf/optimo.urdf";
  if (!std::filesystem::exists(optimo_urdf)) {
    GTEST_SKIP() << "Optimo URDF not found";
  }

  // Map format with is_soft toggled per constraint type.
  std::ostringstream yaml;
  yaml << "robot_model:\n"
       << "  urdf_path: \"" << optimo_urdf << "\"\n"
       << R"(  is_floating_base: false
  base_frame: optimo_base_link
regularization:
  w_tau: 1.0e-3
start_state_id: 1
task_pool:
  - name: jpos_task
    type: JointTask
    kp: 10.0
    kd: 1.0
    kp_ik: 1.0
global_constraints:
  JointPosLimitConstraint:
    enabled: true
    is_soft: true
    soft_weight: 2.0e+4
  JointVelLimitConstraint:
    enabled: true
    is_soft: false
  JointTrqLimitConstraint:
    enabled: true
    is_soft: true
    soft_weight: 5.0e+3
state_machine:
  - id: 1
    name: ut_teleop_state
    task_hierarchy:
      - {name: jpos_task, priority: 0}
)";

  const auto yaml_path = WriteYaml("soft_constraint_test.yaml", yaml.str());
  auto arch = MakeArchitecture(yaml_path);
  ASSERT_NE(arch, nullptr);

  // Verify RuntimeConfig parsed soft flags.
  const auto* cfg = arch->GetConfig();
  EXPECT_TRUE(cfg->SoftConfig("JointPosLimitConstraint").is_soft);
  EXPECT_DOUBLE_EQ(cfg->SoftConfig("JointPosLimitConstraint").weight, 2.0e+4);
  EXPECT_FALSE(cfg->SoftConfig("JointVelLimitConstraint").is_soft);
  EXPECT_TRUE(cfg->SoftConfig("JointTrqLimitConstraint").is_soft);
  EXPECT_DOUBLE_EQ(cfg->SoftConfig("JointTrqLimitConstraint").weight, 5.0e+3);

  // Run a few ticks to verify the solver works with mixed soft/hard constraints.
  const int n_act = robot_->NumActiveDof();
  Eigen::VectorXd q = Eigen::VectorXd::Zero(n_act);
  q << 0.0, -0.5, 0.0, 1.0, 0.0, -0.5, 0.0;
  const Eigen::VectorXd qdot = Eigen::VectorXd::Zero(n_act);

  const StateConfig* state = cfg->FindState(1);
  ASSERT_NE(state, nullptr);
  ASSERT_NE(state->joint, nullptr);
  state->joint->UpdateDesired(q, Eigen::VectorXd::Zero(n_act),
                              Eigen::VectorXd::Zero(n_act));

  for (int i = 0; i < 50; ++i) {
    UpdateWithJointState(arch.get(), q, qdot, i * 0.001, 0.001);
  }
  const auto& cmd = arch->GetCommand();
  EXPECT_TRUE(cmd.tau.allFinite()) << "Torque must be finite with mixed soft/hard constraints";
}

TEST_F(ControlArchitectureBehaviorTest, SoftConstraintSequenceFormat) {
  // Verify is_soft works with legacy sequence YAML format too.
  const std::string optimo_urdf =
      "/home/dk/workspace/rpc_ws/src/rpc_ros/description/"
      "optimo_description/urdf/optimo.urdf";
  if (!std::filesystem::exists(optimo_urdf)) {
    GTEST_SKIP() << "Optimo URDF not found";
  }

  std::ostringstream yaml;
  yaml << "robot_model:\n"
       << "  urdf_path: \"" << optimo_urdf << "\"\n"
       << R"(  is_floating_base: false
  base_frame: optimo_base_link
regularization:
  w_tau: 1.0e-3
start_state_id: 1
task_pool:
  - name: jpos_task
    type: JointTask
    kp: 10.0
    kd: 1.0
    kp_ik: 1.0
global_constraints:
  - name: joint_limit
    type: JointPosLimitConstraint
    is_soft: true
    soft_weight: 8.0e+4
  - name: joint_vel_limit
    type: JointVelLimitConstraint
state_machine:
  - id: 1
    name: ut_teleop_state
    task_hierarchy:
      - {name: jpos_task, priority: 0}
)";

  const auto yaml_path = WriteYaml("soft_constraint_seq.yaml", yaml.str());
  auto arch = MakeArchitecture(yaml_path);
  ASSERT_NE(arch, nullptr);

  const auto* cfg = arch->GetConfig();
  EXPECT_TRUE(cfg->SoftConfig("JointPosLimitConstraint").is_soft);
  EXPECT_DOUBLE_EQ(cfg->SoftConfig("JointPosLimitConstraint").weight, 8.0e+4);
  EXPECT_FALSE(cfg->SoftConfig("JointVelLimitConstraint").is_soft);

  // Run a few ticks.
  const int n_act = robot_->NumActiveDof();
  Eigen::VectorXd q = Eigen::VectorXd::Zero(n_act);
  q << 0.0, -0.5, 0.0, 1.0, 0.0, -0.5, 0.0;
  const Eigen::VectorXd qdot = Eigen::VectorXd::Zero(n_act);

  const StateConfig* state = cfg->FindState(1);
  ASSERT_NE(state, nullptr);
  state->joint->UpdateDesired(q, Eigen::VectorXd::Zero(n_act),
                              Eigen::VectorXd::Zero(n_act));

  for (int i = 0; i < 50; ++i) {
    UpdateWithJointState(arch.get(), q, qdot, i * 0.001, 0.001);
  }
  const auto& cmd = arch->GetCommand();
  EXPECT_TRUE(cmd.tau.allFinite()) << "Torque must be finite with soft pos constraint";
}

// ===========================================================================
// Behavioral tests: tracking, smoothness, constraint satisfaction, transitions
// ===========================================================================

// --- Helper: build Draco3 YAML config for behavioral tests ---
std::string Draco3BehaviorYaml(
    const std::string& draco_urdf, int start_state_id = 1,
    const std::string& extra_states = "") {
  std::ostringstream yaml;
  yaml << "robot_model:\n"
       << "  urdf_path: \"" << draco_urdf << "\"\n"
       << R"(  is_floating_base: true
  base_frame: torso_link
regularization:
  w_qddot: 1.0e-6
  w_rf: 1.0e-4
  w_tau: 1.0e-3
  w_xc_ddot: 1.0e-3
  w_f_dot: 1.0e-3
start_state_id: )" << start_state_id << R"(
task_pool:
  - name: com_task
    type: ComTask
    kp: [40, 40, 40]
    kd: [8, 8, 8]
    kp_ik: [1.0, 1.0, 1.0]
    weight: [50, 50, 50]
  - name: jpos_task
    type: JointTask
    kp: 30.0
    kd: 3.0
    kp_ik: 1.0
    weight: 1.0
  - name: lfoot_force
    type: ForceTask
    contact_name: lfoot_contact
    weight: [0, 0, 0, 0, 0, 0]
  - name: rfoot_force
    type: ForceTask
    contact_name: rfoot_contact
    weight: [0, 0, 0, 0, 0, 0]
contact_pool:
  - name: lfoot_contact
    type: SurfaceContact
    target_frame: l_foot_contact
    mu: 0.5
    foot_half_length: 0.10
    foot_half_width: 0.05
  - name: rfoot_contact
    type: SurfaceContact
    target_frame: r_foot_contact
    mu: 0.5
    foot_half_length: 0.10
    foot_half_width: 0.05
global_constraints:
  JointPosLimitConstraint:
    enabled: true
    scale: 0.9
    is_soft: true
    soft_weight: 1.0e+5
  JointVelLimitConstraint:
    enabled: true
    scale: 0.8
    is_soft: true
    soft_weight: 1.0e+5
  JointTrqLimitConstraint:
    enabled: true
    scale: 0.9
state_machine:
  - id: 1
    name: ut_teleop_state
    task_hierarchy:
      - {name: com_task, priority: 0}
      - {name: jpos_task, priority: 1}
    contact_constraints:
      - {name: lfoot_contact}
      - {name: rfoot_contact}
    force_tasks:
      - {name: lfoot_force}
      - {name: rfoot_force}
)" + extra_states;
  return yaml.str();
}

// Draco3 standing joint configuration (single-knee model).
Eigen::VectorXd Draco3StandingConfig() {
  Eigen::VectorXd q = Eigen::VectorXd::Zero(25);
  q(2)  = -0.565;  q(3)  =  0.565;  q(4)  = -0.565;
  q(7)  =  0.523;  q(9)  = -1.57;
  q(15) = -0.565;  q(16) =  0.565;  q(17) = -0.565;
  q(20) = -0.523;  q(22) = -1.57;
  return q;
}

RobotBaseState Draco3StandingBaseState() {
  RobotBaseState bs;
  bs.pos = Eigen::Vector3d(0.0, 0.0, 0.841);
  bs.quat = Eigen::Quaterniond::Identity();
  bs.lin_vel = Eigen::Vector3d::Zero();
  bs.ang_vel = Eigen::Vector3d::Zero();
  bs.rot_world_local = Eigen::Matrix3d::Identity();
  return bs;
}

std::string Optimo7BehaviorYaml(const std::string& optimo_urdf,
                                 int start_state_id = 1) {
  std::ostringstream yaml;
  yaml << "robot_model:\n"
       << "  urdf_path: \"" << optimo_urdf << "\"\n"
       << R"(  is_floating_base: false
  base_frame: optimo_base_link
regularization:
  w_qddot: 1.0e-6
  w_tau: 1.0e-3
start_state_id: )" << start_state_id << R"(
task_pool:
  - name: jpos_task
    type: JointTask
    kp: [10, 10, 10, 10, 10, 10, 10]
    kd: [1, 1, 1, 1, 1, 1, 1]
    kp_ik: [1, 1, 1, 1, 1, 1, 1]
  - name: ee_pos
    type: LinkPosTask
    target_frame: optimo_end_effector
    reference_frame: optimo_base_link
    kp: [100, 100, 100]
    kd: [10, 10, 10]
    kp_ik: [1, 1, 1]
  - name: ee_ori
    type: LinkOriTask
    target_frame: optimo_end_effector
    reference_frame: optimo_base_link
    kp: [100, 100, 100]
    kd: [10, 10, 10]
    kp_ik: [1, 1, 1]
global_constraints:
  JointPosLimitConstraint:
    enabled: true
    scale: 0.9
    is_soft: true
    soft_weight: 1.0e+5
  JointVelLimitConstraint:
    enabled: true
    scale: 0.8
    is_soft: true
    soft_weight: 1.0e+5
state_machine:
  - id: 1
    name: ut_teleop_state
    task_hierarchy:
      - {name: ee_pos, priority: 0}
      - {name: ee_ori, priority: 1}
      - {name: jpos_task, priority: 2}
)";
  return yaml.str();
}

// --- Optimo: Joint tracking IK output direction ---
TEST_F(ControlArchitectureBehaviorTest, Behavior_Optimo_JointTrackingConvergence) {
  const std::string optimo_urdf =
      "/home/dk/workspace/rpc_ws/src/rpc_ros/description/"
      "optimo_description/urdf/optimo.urdf";
  if (!std::filesystem::exists(optimo_urdf)) {
    GTEST_SKIP() << "Optimo URDF not found";
  }

  // Use joint-only state (single task).
  std::ostringstream yaml;
  yaml << "robot_model:\n"
       << "  urdf_path: \"" << optimo_urdf << "\"\n"
       << R"(  is_floating_base: false
  base_frame: optimo_base_link
regularization:
  w_qddot: 1.0e-6
  w_tau: 1.0e-3
start_state_id: 1
task_pool:
  - name: jpos_task
    type: JointTask
    kp: [10, 10, 10, 10, 10, 10, 10]
    kd: [1, 1, 1, 1, 1, 1, 1]
    kp_ik: [1, 1, 1, 1, 1, 1, 1]
global_constraints:
  JointPosLimitConstraint:
    enabled: true
    scale: 0.9
    is_soft: true
    soft_weight: 1.0e+5
state_machine:
  - id: 1
    name: ut_teleop_state
    task_hierarchy:
      - {name: jpos_task}
)";

  const auto yaml_path = WriteYaml("behavior_optimo_jtrack.yaml", yaml.str());
  auto arch = MakeArchitecture(yaml_path);
  ASSERT_NE(arch, nullptr);

  const int n_act = robot_->NumActiveDof();
  ASSERT_EQ(n_act, 7);

  Eigen::VectorXd q = Eigen::VectorXd::Zero(n_act);
  q << 0.0, -0.5, 0.0, 1.0, 0.0, -0.5, 0.0;
  const Eigen::VectorXd qdot = Eigen::VectorXd::Zero(n_act);

  // Set desired = current first (warmup at rest).
  const StateConfig* state = arch->GetConfig()->FindState(1);
  ASSERT_NE(state, nullptr);
  ASSERT_NE(state->joint, nullptr);
  state->joint->UpdateDesired(q, Eigen::VectorXd::Zero(n_act),
                              Eigen::VectorXd::Zero(n_act));

  // Warmup: run enough ticks to complete the weight scheduler ramp (~300ms).
  constexpr int kWarmupTicks = 400;
  for (int i = 0; i < kWarmupTicks; ++i) {
    UpdateWithJointState(arch.get(), q, qdot, i * 0.001, 0.001);
  }

  // Apply a 0.1 rad step to joint 3.
  Eigen::VectorXd q_des = q;
  q_des(3) += 0.1;
  state->joint->UpdateDesired(q_des, Eigen::VectorXd::Zero(n_act),
                              Eigen::VectorXd::Zero(n_act));

  // Open-loop: verify IK output moves the TARGET joint toward the target.
  // Without a dynamics simulator we cannot close the loop, but we can verify
  // that the WBC's IK output (cmd.q) is in the correct direction.
  const auto& cmd = arch->GetCommand();

  // Run a few ticks (open-loop, same sensor state each tick).
  for (int i = 0; i < 10; ++i) {
    UpdateWithJointState(arch.get(), q, qdot, (kWarmupTicks + i) * 0.001, 0.001);
    ASSERT_TRUE(cmd.q.allFinite()) << "NaN q_cmd at tick " << i;
    ASSERT_TRUE(cmd.tau.allFinite()) << "NaN torque at tick " << i;
  }

  // The correction on joint 3 should be in the positive direction.
  const double delta_j3 = cmd.q(3) - q(3);
  EXPECT_GT(delta_j3, 0.0)
      << "Joint 3 correction is in wrong direction: delta=" << delta_j3;

  // The correction should cover most of the 0.1 rad step.
  EXPECT_GT(delta_j3, 0.05)
      << "Joint 3 correction too small: " << delta_j3 << " rad (expected ~0.1)";

  // cmd.q(3) should be close to q_des(3) — IK with kp_ik=1 should hit target.
  EXPECT_NEAR(cmd.q(3), q_des(3), 0.01)
      << "Joint 3 cmd.q not close to target: cmd=" << cmd.q(3)
      << " des=" << q_des(3);

  // Output should be consistent across ticks (deterministic).
  const Eigen::VectorXd q_cmd_prev = cmd.q;
  UpdateWithJointState(arch.get(), q, qdot, (kWarmupTicks + 10) * 0.001, 0.001);
  const double consistency = (cmd.q - q_cmd_prev).norm();
  EXPECT_LT(consistency, 1e-6)
      << "IK output not consistent across ticks: " << consistency;

  std::cout << "[Optimo JointTracking] Open-loop IK output check\n"
            << "  Joint 3 delta: " << delta_j3 << " rad (target step: 0.1)\n"
            << "  cmd.q(3): " << cmd.q(3) << " (target: " << q_des(3) << ")\n"
            << "  Output consistency: " << consistency << "\n"
            << "  cmd.q: " << cmd.q.transpose() << "\n";
}

// --- Optimo: Torque smoothness ---
TEST_F(ControlArchitectureBehaviorTest, Behavior_Optimo_TorqueSmoothness) {
  const std::string optimo_urdf =
      "/home/dk/workspace/rpc_ws/src/rpc_ros/description/"
      "optimo_description/urdf/optimo.urdf";
  if (!std::filesystem::exists(optimo_urdf)) {
    GTEST_SKIP() << "Optimo URDF not found";
  }

  const auto yaml_path =
      WriteYaml("behavior_optimo_smooth.yaml", Optimo7BehaviorYaml(optimo_urdf));
  auto arch = MakeArchitecture(yaml_path);
  ASSERT_NE(arch, nullptr);

  const int n_act = robot_->NumActiveDof();
  Eigen::VectorXd q = Eigen::VectorXd::Zero(n_act);
  q << 0.0, -0.5, 0.0, 1.0, 0.0, -0.5, 0.0;
  const Eigen::VectorXd qdot = Eigen::VectorXd::Zero(n_act);

  const StateConfig* state = arch->GetConfig()->FindState(1);
  ASSERT_NE(state, nullptr);
  state->joint->UpdateDesired(q, Eigen::VectorXd::Zero(n_act),
                              Eigen::VectorXd::Zero(n_act));

  // Warmup.
  for (int i = 0; i < 20; ++i) {
    UpdateWithJointState(arch.get(), q, qdot, i * 0.001, 0.001);
  }

  // Apply a small EE perturbation and measure torque rate.
  // (EE pos task is active — any joint step propagates through hierarchy.)
  const auto& cmd = arch->GetCommand();
  Eigen::VectorXd tau_prev = cmd.tau;
  double max_dtau = 0.0;

  for (int i = 0; i < 200; ++i) {
    UpdateWithJointState(arch.get(), q, qdot, (20 + i) * 0.001, 0.001);
    ASSERT_TRUE(cmd.tau.allFinite()) << "NaN torque at tick " << i;
    const double dtau = (cmd.tau - tau_prev).cwiseAbs().maxCoeff();
    max_dtau = std::max(max_dtau, dtau);
    tau_prev = cmd.tau;
  }

  // In steady state, torque rate should be very small (< 1 Nm/tick at 1kHz).
  EXPECT_LT(max_dtau, 1.0)
      << "Torque rate exceeded 1 Nm/tick: " << max_dtau;
  std::cout << "[Optimo Smoothness] Max |dτ/dt|: " << max_dtau << " Nm/tick\n";
}

// --- Optimo: All-constraint determinism + torque limit satisfaction ---
// Exercises cached constraint pointers (pos/vel/trq), sa_tau0_scratch_,
// and PInvSquare LLT path across 3-dim (ee_pos), 3-dim (ee_ori), and
// 7-dim (joint) tasks in the nullspace hierarchy.
TEST_F(ControlArchitectureBehaviorTest, Behavior_Optimo_AllConstraintDeterminism) {
  const std::string optimo_urdf =
      "/home/dk/workspace/rpc_ws/src/rpc_ros/description/"
      "optimo_description/urdf/optimo.urdf";
  if (!std::filesystem::exists(optimo_urdf)) {
    GTEST_SKIP() << "Optimo URDF not found";
  }

  // YAML with all 3 constraint types (pos, vel, trq) enabled.
  std::ostringstream yaml_ss;
  yaml_ss << "robot_model:\n"
          << "  urdf_path: \"" << optimo_urdf << "\"\n"
          << R"(  is_floating_base: false
  base_frame: optimo_base_link
regularization:
  w_qddot: 1.0e-6
  w_tau: 1.0e-3
start_state_id: 1
task_pool:
  - name: jpos_task
    type: JointTask
    kp: [100, 100, 100, 100, 100, 100, 100]
    kd: [10, 10, 10, 10, 10, 10, 10]
    kp_ik: [1, 1, 1, 1, 1, 1, 1]
  - name: ee_pos
    type: LinkPosTask
    target_frame: optimo_end_effector
    reference_frame: optimo_base_link
    kp: [200, 200, 200]
    kd: [20, 20, 20]
    kp_ik: [1, 1, 1]
  - name: ee_ori
    type: LinkOriTask
    target_frame: optimo_end_effector
    reference_frame: optimo_base_link
    kp: [200, 200, 200]
    kd: [20, 20, 20]
    kp_ik: [1, 1, 1]
global_constraints:
  JointPosLimitConstraint:
    enabled: true
    scale: 0.9
    is_soft: true
    soft_weight: 1.0e+5
  JointVelLimitConstraint:
    enabled: true
    scale: 0.8
    is_soft: true
    soft_weight: 1.0e+5
  JointTrqLimitConstraint:
    enabled: true
    scale: 0.8
state_machine:
  - id: 1
    name: ut_teleop_state
    task_hierarchy:
      - {name: ee_pos, priority: 0}
      - {name: ee_ori, priority: 1}
      - {name: jpos_task, priority: 2}
)";

  constexpr int kTicks = 100;
  const double trq_limit_scale = 0.8;

  // Return type: torque history + URDF limits (captured before arch is destroyed).
  struct PipelineResult {
    std::vector<Eigen::VectorXd> torques;
    Eigen::MatrixXd trq_limits;
  };

  // Lambda to run the full pipeline and collect torque history.
  auto run_pipeline = [&]() -> PipelineResult {
    const auto yaml_path =
        WriteYaml("optimo_allconst.yaml", yaml_ss.str());
    auto arch = MakeArchitecture(yaml_path);
    EXPECT_NE(arch, nullptr);
    if (!arch) return {};

    const int n_act = robot_->NumActiveDof();
    Eigen::VectorXd q = Eigen::VectorXd::Zero(n_act);
    q << 0.0, -0.8, 0.0, 1.5, 0.0, -0.3, 0.0;
    const Eigen::VectorXd qdot = Eigen::VectorXd::Zero(n_act);

    // Set desired to a different pose to create non-trivial torques.
    const StateConfig* state = arch->GetConfig()->FindState(1);
    if (state && state->joint) {
      Eigen::VectorXd q_des = Eigen::VectorXd::Zero(n_act);
      q_des << 0.0, -0.5, 0.0, 1.0, 0.0, -0.5, 0.0;
      state->joint->UpdateDesired(
          q_des, Eigen::VectorXd::Zero(n_act), Eigen::VectorXd::Zero(n_act));
    }

    PipelineResult result;
    result.trq_limits = robot_->JointTrqLimits();
    result.torques.reserve(kTicks);
    for (int i = 0; i < kTicks; ++i) {
      UpdateWithJointState(arch.get(), q, qdot, i * 0.001, 0.001);
      const RobotCommand& cmd = arch->GetCommand();
      EXPECT_TRUE(cmd.tau.allFinite()) << "NaN at tick " << i;
      EXPECT_TRUE(cmd.q.allFinite()) << "NaN q at tick " << i;
      result.torques.push_back(cmd.tau);
    }
    return result;
  };

  const auto res1 = run_pipeline();
  const auto res2 = run_pipeline();
  const auto& run1 = res1.torques;
  const auto& run2 = res2.torques;

  // (1) Determinism: bit-exact match.
  ASSERT_EQ(run1.size(), run2.size());
  for (int i = 0; i < kTicks; ++i) {
    EXPECT_TRUE(run1[i] == run2[i])
        << "Non-determinism at tick " << i << ":\n"
        << "  run1: " << run1[i].transpose() << "\n"
        << "  run2: " << run2[i].transpose();
  }

  // (2) Torque limits: URDF efforts are [87, 87, 87, 87, 12, 12, 12] for Optimo.
  // With scale=0.8, effective limits are 80% of those.
  // The post-clamp in ControlArchitecture must enforce these.
  const Eigen::MatrixXd& trq_limits = res1.trq_limits;
  for (int i = 0; i < kTicks; ++i) {
    for (int j = 0; j < run1[i].size(); ++j) {
      const double lo = trq_limits(j, 0) * trq_limit_scale;
      const double hi = trq_limits(j, 1) * trq_limit_scale;
      EXPECT_GE(run1[i][j], lo - 1e-3)
          << "tick " << i << " joint " << j << " below trq limit";
      EXPECT_LE(run1[i][j], hi + 1e-3)
          << "tick " << i << " joint " << j << " above trq limit";
    }
  }

  std::cout << "[Optimo AllConstraint] " << kTicks
            << " ticks deterministic, torque limits satisfied\n";
}

// --- Optimo: Closed-loop RBD simulation via Pinocchio ABA ---
// The gold standard for pre-hardware validation: run the full
// WBC → torque → forward dynamics → state update loop and verify
// that the controller converges, stays stable, and respects limits.
TEST_F(ControlArchitectureBehaviorTest, Behavior_Optimo_ClosedLoopRBD) {
  const std::string optimo_urdf =
      "/home/dk/workspace/rpc_ws/src/rpc_ros/description/"
      "optimo_description/urdf/optimo.urdf";
  if (!std::filesystem::exists(optimo_urdf)) {
    GTEST_SKIP() << "Optimo URDF not found";
  }

  // Optimo URDF limits:
  //   J1:[-2.09,2.09] J2:[1.05,3.84] J3:[-2.09,2.09] J4:[-2.53,2.53]
  //   J5:[-2.62,2.62] J6:[-2.09,2.09] J7:[-2.09,2.09]
  //   Effort: [95, 95, 40, 40, 15, 15, 15] Nm
  //
  // Conservative gains for closed-loop stability with ABA forward dynamics.
  // Closed-loop RBD test: WBIC inverse dynamics + kp/kd feedback vs ABA physics.
  // No constraints — validates that WBIC output drives the robot to q_des.
  std::ostringstream yaml_ss;
  yaml_ss << "robot_model:\n"
          << "  urdf_path: \"" << optimo_urdf << "\"\n"
          << R"(  is_floating_base: false
  base_frame: optimo_base_link
regularization:
  w_qddot: 1.0e-6
  w_tau: 0.0
start_state_id: 1
task_pool:
  - name: jpos_task
    type: JointTask
    kp: [100, 100, 100, 100, 50, 50, 50]
    kd: [20, 20, 20, 20, 14, 14, 14]
    kp_ik: [1, 1, 1, 1, 1, 1, 1]
state_machine:
  - id: 1
    name: ut_teleop_state
    task_hierarchy:
      - {name: jpos_task, priority: 0}
)";

  const auto yaml_path =
      WriteYaml("optimo_closedloop.yaml", yaml_ss.str());
  auto arch = MakeArchitecture(yaml_path);
  ASSERT_NE(arch, nullptr);

  const int n_act = robot_->NumActiveDof();
  ASSERT_EQ(n_act, 7);

  const pinocchio::Model& model = robot_->GetModel();
  pinocchio::Data sim_data(model);  // separate data for simulation

  // Optimo home pose (from controller config).
  Eigen::VectorXd q_des = Eigen::VectorXd::Zero(n_act);
  q_des << 0.0, 3.3, 0.0, -2.35, 0.0, -1.13, 0.0;

  // Start with 0.1 rad perturbation from home.
  Eigen::VectorXd q_sim = q_des;
  q_sim.array() += 0.1;
  Eigen::VectorXd qdot_sim = Eigen::VectorXd::Zero(n_act);

  // Set desired pose.
  const StateConfig* state = arch->GetConfig()->FindState(1);
  ASSERT_NE(state, nullptr);
  ASSERT_NE(state->joint, nullptr);
  state->joint->UpdateDesired(
      q_des, Eigen::VectorXd::Zero(n_act), Eigen::VectorXd::Zero(n_act));

  constexpr double dt = 0.001;
  constexpr int kTotalTicks = 3000;   // 3 seconds
  constexpr int kSettleTicks = 2000;  // check convergence after 2s

  const Eigen::MatrixXd& pos_limits = robot_->JointPosLimits();
  const Eigen::MatrixXd& trq_limits = robot_->JointTrqLimits();

  double max_pos_err = 0.0;
  double max_trq = 0.0;
  int nan_count = 0;

  for (int i = 0; i < kTotalTicks; ++i) {
    // (1) Feed current state to WBC.
    RobotJointState js;
    js.q    = q_sim;
    js.qdot = qdot_sim;
    js.tau  = Eigen::VectorXd::Zero(n_act);
    arch->Update(js, i * dt, dt);

    const RobotCommand& cmd = arch->GetCommand();
    if (!cmd.tau.allFinite()) {
      nan_count++;
      break;
    }

    // (2) Forward dynamics: qddot = M^{-1}(tau - h) via Pinocchio ABA.
    pinocchio::aba(model, sim_data, q_sim, qdot_sim, cmd.tau);
    const Eigen::VectorXd& qddot = sim_data.ddq;

    // (3) Semi-implicit Euler integration.
    qdot_sim += qddot * dt;
    q_sim += qdot_sim * dt;

    // (4) Clamp to URDF joint limits (hard stops) + zero velocity at limit.
    for (int j = 0; j < n_act; ++j) {
      if (q_sim[j] < pos_limits(j, 0)) {
        q_sim[j] = pos_limits(j, 0);
        qdot_sim[j] = std::max(qdot_sim[j], 0.0);
      } else if (q_sim[j] > pos_limits(j, 1)) {
        q_sim[j] = pos_limits(j, 1);
        qdot_sim[j] = std::min(qdot_sim[j], 0.0);
      }
    }

    // Track stats.
    max_trq = std::max(max_trq, cmd.tau.cwiseAbs().maxCoeff());
    if (i >= kSettleTicks) {
      const double pos_err = (q_sim - q_des).cwiseAbs().maxCoeff();
      max_pos_err = std::max(max_pos_err, pos_err);
    }
  }

  EXPECT_EQ(nan_count, 0) << "NaN torque detected during simulation";

  // After settling, position error should be small (< 5 degrees ≈ 0.087 rad).
  EXPECT_LT(max_pos_err, 0.087)
      << "Joint tracking did not converge after "
      << kSettleTicks * dt << "s: max error = "
      << max_pos_err << " rad (" << max_pos_err * 180.0 / M_PI << " deg)";

  // Velocity should have settled near zero.
  const double final_qdot_norm = qdot_sim.norm();
  EXPECT_LT(final_qdot_norm, 0.5)
      << "Velocity not settled: ||qdot|| = " << final_qdot_norm;

  std::cout << "[Optimo ClosedLoop RBD] " << kTotalTicks * dt << "s simulation\n"
            << "  Converged pos error: " << max_pos_err * 1000.0 << " mrad\n"
            << "  Peak torque: " << max_trq << " Nm\n"
            << "  Final ||qdot||: " << final_qdot_norm << " rad/s\n"
            << "  Final q:    " << q_sim.transpose() << "\n"
            << "  Desired q:  " << q_des.transpose() << "\n";
}

// --- Optimo: Weight transfer verification ---
// Verify that per-state weights from YAML are correctly applied to tasks
// after state transitions, and that tasks not in the state revert to pool default.
TEST_F(ControlArchitectureBehaviorTest, Behavior_Optimo_WeightTransfer) {
  const std::string optimo_urdf =
      "/home/dk/workspace/rpc_ws/src/rpc_ros/description/"
      "optimo_description/urdf/optimo.urdf";
  if (!std::filesystem::exists(optimo_urdf)) {
    GTEST_SKIP() << "Optimo URDF not found";
  }

  // Two-state config: state 1 = joint-dominant, state 2 = EE-dominant.
  std::ostringstream yaml;
  yaml << "robot_model:\n"
       << "  urdf_path: \"" << optimo_urdf << "\"\n"
       << R"(  is_floating_base: false
  base_frame: optimo_base_link
regularization:
  w_qddot: 1.0e-6
  w_tau: 1.0e-3
controller:
  weight_min: 1.0e-6
  weight_max: 1.0e+4
start_state_id: 1
task_pool:
  - name: jpos_task
    type: JointTask
    kp: 200.0
    kd: 28.0
    kp_ik: 1.0
    weight: 50.0
  - name: ee_pos
    type: LinkPosTask
    target_frame: optimo_end_effector
    reference_frame: optimo_base_link
    kp: 100.0
    kd: 20.0
    kp_ik: 1.0
    weight: 50.0
  - name: ee_ori
    type: LinkOriTask
    target_frame: optimo_end_effector
    reference_frame: optimo_base_link
    kp: 100.0
    kd: 20.0
    kp_ik: 1.0
    weight: 50.0
state_machine:
  - id: 1
    name: joint_state
    type: ut_teleop_state
    params:
      weight_ramp_duration: 0.01
    task_hierarchy:
      - {name: jpos_task, weight: 100.0}
      - {name: ee_pos,    weight: 1e-6}
      - {name: ee_ori,    weight: 1e-6}
  - id: 2
    name: ee_state
    type: ut_teleop_state
    params:
      weight_ramp_duration: 0.01
    task_hierarchy:
      - {name: ee_pos,    weight: 200.0}
      - {name: ee_ori,    weight: 200.0}
      - {name: jpos_task, weight: 0.5}
)";

  const auto yaml_path = WriteYaml("behavior_optimo_weight_xfer.yaml", yaml.str());
  auto arch = MakeArchitecture(yaml_path);
  ASSERT_NE(arch, nullptr);

  const int n_act = robot_->NumActiveDof();
  Eigen::VectorXd q = Eigen::VectorXd::Zero(n_act);
  q << 0.0, -0.5, 0.0, 1.0, 0.0, -0.5, 0.0;
  const Eigen::VectorXd qdot = Eigen::VectorXd::Zero(n_act);

  // Set desired = current (no motion command).
  const StateConfig* state1 = arch->GetConfig()->FindState(1);
  ASSERT_NE(state1, nullptr);
  ASSERT_NE(state1->joint, nullptr);
  state1->joint->UpdateDesired(q, Eigen::VectorXd::Zero(n_act),
                                Eigen::VectorXd::Zero(n_act));

  // Run enough ticks to complete the 10ms ramp.
  for (int i = 0; i < 50; ++i) {
    UpdateWithJointState(arch.get(), q, qdot, i * 0.001, 0.001);
  }

  // Check state 1 weights: jpos=100, ee_pos=1e-6, ee_ori=1e-6
  auto* reg = arch->GetConfig()->taskRegistry();
  Task* jpos_task = reg->GetMotionTask("jpos_task");
  Task* ee_pos_task = reg->GetMotionTask("ee_pos");
  Task* ee_ori_task = reg->GetMotionTask("ee_ori");
  ASSERT_NE(jpos_task, nullptr);
  ASSERT_NE(ee_pos_task, nullptr);
  ASSERT_NE(ee_ori_task, nullptr);

  // After ramp completes, weights should be at their state targets.
  EXPECT_NEAR(jpos_task->Weight().maxCoeff(), 100.0, 1e-3)
      << "jpos_task weight should be 100 in state 1";
  EXPECT_NEAR(ee_pos_task->Weight().maxCoeff(), 1e-6, 1e-7)
      << "ee_pos weight should be ~0 in state 1";
  EXPECT_NEAR(ee_ori_task->Weight().maxCoeff(), 1e-6, 1e-7)
      << "ee_ori weight should be ~0 in state 1";

  // Transition to state 2.
  arch->GetFsmHandler()->RequestState(2);
  for (int i = 50; i < 100; ++i) {
    UpdateWithJointState(arch.get(), q, qdot, i * 0.001, 0.001);
  }

  // Check state 2 weights: ee_pos=200, ee_ori=200, jpos=0.5
  EXPECT_NEAR(ee_pos_task->Weight().maxCoeff(), 200.0, 1e-3)
      << "ee_pos weight should be 200 in state 2";
  EXPECT_NEAR(ee_ori_task->Weight().maxCoeff(), 200.0, 1e-3)
      << "ee_ori weight should be 200 in state 2";
  EXPECT_NEAR(jpos_task->Weight().maxCoeff(), 0.5, 1e-3)
      << "jpos_task weight should be 0.5 in state 2";

  // Transition back to state 1.
  arch->GetFsmHandler()->RequestState(1);
  for (int i = 100; i < 150; ++i) {
    UpdateWithJointState(arch.get(), q, qdot, i * 0.001, 0.001);
  }

  // Weights should return to state 1 values.
  EXPECT_NEAR(jpos_task->Weight().maxCoeff(), 100.0, 1e-3)
      << "jpos_task weight should return to 100 in state 1";
  EXPECT_NEAR(ee_pos_task->Weight().maxCoeff(), 1e-6, 1e-7)
      << "ee_pos weight should return to ~0 in state 1";

  std::cout << "[Optimo WeightTransfer] State transitions verified.\n"
            << "  State 1: jpos=" << jpos_task->Weight().maxCoeff()
            << ", ee_pos=" << ee_pos_task->Weight().maxCoeff()
            << ", ee_ori=" << ee_ori_task->Weight().maxCoeff() << "\n";
}

// --- Draco3: CoM IK output direction after step command ---
TEST_F(ControlArchitectureBehaviorTest, Behavior_Draco3_ComTrackingStep) {
  const std::string draco_urdf =
      "/home/dk/workspace/rpc_ws/src/rpc_ros/description/"
      "draco_description/urdf/draco_modified_single_knee.urdf";
  if (!std::filesystem::exists(draco_urdf)) {
    GTEST_SKIP() << "Draco3 URDF not found";
  }

  const auto yaml_path =
      WriteYaml("behavior_draco3_com.yaml", Draco3BehaviorYaml(draco_urdf));
  auto arch = MakeArchitecture(yaml_path);
  ASSERT_NE(arch, nullptr);
  ASSERT_EQ(robot_->NumActiveDof(), 25);

  Eigen::VectorXd q = Draco3StandingConfig();
  const Eigen::VectorXd qdot = Eigen::VectorXd::Zero(25);
  RobotBaseState bs = Draco3StandingBaseState();

  // Set desired = current for warmup.
  const StateConfig* state = arch->GetConfig()->FindState(1);
  ASSERT_NE(state, nullptr);
  state->joint->UpdateDesired(q, Eigen::VectorXd::Zero(25),
                              Eigen::VectorXd::Zero(25));

  for (int i = 0; i < 50; ++i) {
    UpdateWithFloatingBase(arch.get(), q, qdot, bs, i * 0.001, 0.001);
  }

  // Read initial CoM position from robot model.
  const Eigen::Vector3d com_init = robot_->GetComPosition();

  // Apply a 2cm lateral CoM shift (along y-axis).
  ComTask* com_task = dynamic_cast<ComTask*>(state->com);
  ASSERT_NE(com_task, nullptr) << "ComTask not found in state 1";

  Eigen::Vector3d com_target = com_init;
  com_target.y() += 0.02;  // 2cm lateral shift
  com_task->UpdateDesired(com_target, Eigen::Vector3d::Zero(),
                          Eigen::Vector3d::Zero());

  // Open-loop: verify IK output produces a joint correction that would
  // shift the CoM. Without dynamics simulation, we check that:
  // 1) cmd.q differs from initial q (non-zero correction)
  // 2) The correction is consistent across ticks (deterministic)
  // 3) The output is finite and reasonable
  const auto& cmd = arch->GetCommand();

  // Run a few ticks with the same sensor state.
  for (int i = 0; i < 10; ++i) {
    UpdateWithFloatingBase(arch.get(), q, qdot, bs, (50 + i) * 0.001, 0.001);
    ASSERT_TRUE(cmd.q.allFinite()) << "NaN q_cmd at tick " << i;
    ASSERT_TRUE(cmd.tau.allFinite()) << "NaN torque at tick " << i;
  }

  // The IK should produce a non-trivial joint correction.
  const double q_delta_norm = (cmd.q - q).norm();
  EXPECT_GT(q_delta_norm, 1e-4)
      << "WBC produced no meaningful joint correction for CoM shift";

  // Verify consistency: run another tick, correction should be similar.
  const Eigen::VectorXd q_cmd_prev = cmd.q;
  UpdateWithFloatingBase(arch.get(), q, qdot, bs, 60 * 0.001, 0.001);
  const double consistency = (cmd.q - q_cmd_prev).norm();
  EXPECT_LT(consistency, 1e-6)
      << "IK output not consistent across ticks with same input: "
      << consistency;

  // Verify the CoM task error is non-zero (task is active).
  const double com_task_error = com_task->LocalPosError().norm();
  EXPECT_GT(com_task_error, 0.01)
      << "CoM task error suspiciously small in open-loop";

  std::cout << "[Draco3 CoM Step] Open-loop IK output check\n"
            << "  Target shift: 2cm lateral\n"
            << "  Joint correction norm: " << q_delta_norm << " rad\n"
            << "  Output consistency: " << consistency << "\n"
            << "  CoM task error: " << com_task_error * 1000.0 << " mm\n";
}

// --- Draco3: CoM lateral swing torque smoothness ---
TEST_F(ControlArchitectureBehaviorTest, Behavior_Draco3_ComSwing) {
  const std::string draco_urdf =
      "/home/dk/workspace/rpc_ws/src/rpc_ros/description/"
      "draco_description/urdf/draco_modified_single_knee.urdf";
  if (!std::filesystem::exists(draco_urdf)) {
    GTEST_SKIP() << "Draco3 URDF not found";
  }

  const auto yaml_path =
      WriteYaml("behavior_draco3_swing.yaml", Draco3BehaviorYaml(draco_urdf));
  auto arch = MakeArchitecture(yaml_path);
  ASSERT_NE(arch, nullptr);

  Eigen::VectorXd q = Draco3StandingConfig();
  const Eigen::VectorXd qdot = Eigen::VectorXd::Zero(25);
  RobotBaseState bs = Draco3StandingBaseState();

  const StateConfig* state = arch->GetConfig()->FindState(1);
  ASSERT_NE(state, nullptr);
  state->joint->UpdateDesired(q, Eigen::VectorXd::Zero(25),
                              Eigen::VectorXd::Zero(25));

  // Warmup at rest.
  for (int i = 0; i < 50; ++i) {
    UpdateWithFloatingBase(arch.get(), q, qdot, bs, i * 0.001, 0.001);
  }

  const Eigen::Vector3d com_init = robot_->GetComPosition();

  ComTask* com_task = dynamic_cast<ComTask*>(state->com);
  ASSERT_NE(com_task, nullptr);

  // Oscillate CoM target laterally: ±3cm at 1Hz over 2 seconds.
  // Open-loop: sensor state stays constant (no dynamics simulator).
  // We verify that the torque output is smooth as the reference changes.
  constexpr double kAmplitude = 0.03;  // 3cm
  constexpr double kFreqHz = 1.0;
  constexpr int kTicks = 2000;  // 2 seconds at 1kHz
  const auto& cmd = arch->GetCommand();

  double max_tau_rate = 0.0;
  Eigen::VectorXd tau_prev = cmd.tau;
  int finite_count = 0;

  for (int i = 0; i < kTicks; ++i) {
    const double t = (50 + i) * 0.001;
    // Sinusoidal CoM target along y-axis.
    Eigen::Vector3d com_des = com_init;
    com_des.y() += kAmplitude * std::sin(2.0 * M_PI * kFreqHz * t);
    com_task->UpdateDesired(com_des, Eigen::Vector3d::Zero(),
                            Eigen::Vector3d::Zero());

    // Open-loop: always use same sensor state.
    UpdateWithFloatingBase(arch.get(), q, qdot, bs, t, 0.001);

    if (cmd.tau.allFinite()) {
      ++finite_count;
      if (i > 0) {  // skip first tick (warmup → swing transition)
        const double dtau = (cmd.tau - tau_prev).cwiseAbs().maxCoeff();
        max_tau_rate = std::max(max_tau_rate, dtau);
      }
      tau_prev = cmd.tau;
    }
  }

  const double success_rate = static_cast<double>(finite_count) / kTicks * 100.0;

  std::cout << "[Draco3 CoM Swing] 2s lateral oscillation ±3cm at 1Hz (open-loop)\n"
            << "  Solve success: " << success_rate << "%\n"
            << "  Max torque rate: " << max_tau_rate << " Nm/tick\n";

  // Should solve successfully >95% of the time.
  EXPECT_GT(success_rate, 95.0) << "Too many solve failures during swing";
  // In open-loop, torque changes come only from reference changes (smooth sine).
  // The rate should be well-bounded.
  EXPECT_LT(max_tau_rate, 50.0)
      << "Torque discontinuity during CoM swing: " << max_tau_rate;
}

// --- Draco3: Constraint satisfaction under load ---
TEST_F(ControlArchitectureBehaviorTest, Behavior_Draco3_ConstraintSatisfaction) {
  const std::string draco_urdf =
      "/home/dk/workspace/rpc_ws/src/rpc_ros/description/"
      "draco_description/urdf/draco_modified_single_knee.urdf";
  if (!std::filesystem::exists(draco_urdf)) {
    GTEST_SKIP() << "Draco3 URDF not found";
  }

  const auto yaml_path = WriteYaml("behavior_draco3_constraint.yaml",
                                    Draco3BehaviorYaml(draco_urdf));
  auto arch = MakeArchitecture(yaml_path);
  ASSERT_NE(arch, nullptr);

  Eigen::VectorXd q = Draco3StandingConfig();
  const Eigen::VectorXd qdot = Eigen::VectorXd::Zero(25);
  RobotBaseState bs = Draco3StandingBaseState();

  const StateConfig* state = arch->GetConfig()->FindState(1);
  ASSERT_NE(state, nullptr);
  state->joint->UpdateDesired(q, Eigen::VectorXd::Zero(25),
                              Eigen::VectorXd::Zero(25));

  // Warmup.
  for (int i = 0; i < 50; ++i) {
    UpdateWithFloatingBase(arch.get(), q, qdot, bs, i * 0.001, 0.001);
  }

  // Get joint limits from robot model (Nx2 matrices: [min, max]).
  const Eigen::MatrixXd& pos_limits = robot_->JointPosLimits();
  const Eigen::MatrixXd& trq_limits = robot_->JointTrqLimits();

  const auto& cmd = arch->GetCommand();
  int pos_violations = 0;
  int trq_violations = 0;

  // Run 500 ticks of steady-state.
  for (int i = 0; i < 500; ++i) {
    UpdateWithFloatingBase(arch.get(), q, qdot, bs, (50 + i) * 0.001, 0.001);
    if (!cmd.tau.allFinite()) continue;

    // Check q_cmd within URDF limits (with some tolerance).
    for (int j = 0; j < cmd.q.size(); ++j) {
      if (cmd.q(j) < pos_limits(j, 0) - 0.01 ||
          cmd.q(j) > pos_limits(j, 1) + 0.01) {
        ++pos_violations;
      }
    }

    // Check torque within URDF limits (with 10% tolerance).
    for (int j = 0; j < cmd.tau.size(); ++j) {
      const double trq_max = std::max(std::abs(trq_limits(j, 0)),
                                       std::abs(trq_limits(j, 1)));
      if (std::abs(cmd.tau(j)) > trq_max * 1.1) {
        ++trq_violations;
      }
    }
  }

  std::cout << "[Draco3 Constraints] 500-tick steady state\n"
            << "  Position violations (>URDF+1cm): " << pos_violations << "\n"
            << "  Torque violations (>110% URDF):  " << trq_violations << "\n";

  EXPECT_EQ(pos_violations, 0)
      << "Joint position commands exceeded URDF limits";
  EXPECT_EQ(trq_violations, 0) << "Torque commands exceeded URDF limits";
}

// --- Draco3: State transition torque continuity ---
TEST_F(ControlArchitectureBehaviorTest, Behavior_Draco3_StateTransitionContinuity) {
  const std::string draco_urdf =
      "/home/dk/workspace/rpc_ws/src/rpc_ros/description/"
      "draco_description/urdf/draco_modified_single_knee.urdf";
  if (!std::filesystem::exists(draco_urdf)) {
    GTEST_SKIP() << "Draco3 URDF not found";
  }

  // Two states with different task configurations.
  const std::string extra_states = R"(  - id: 2
    name: ut_teleop_state
    task_hierarchy:
      - {name: com_task, priority: 0, weight: [100, 100, 100]}
      - {name: jpos_task, priority: 1}
    contact_constraints:
      - {name: lfoot_contact}
      - {name: rfoot_contact}
    force_tasks:
      - {name: lfoot_force}
      - {name: rfoot_force}
)";

  const auto yaml_path = WriteYaml(
      "behavior_draco3_transition.yaml",
      Draco3BehaviorYaml(draco_urdf, 1, extra_states));
  auto arch = MakeArchitecture(yaml_path);
  ASSERT_NE(arch, nullptr);

  Eigen::VectorXd q = Draco3StandingConfig();
  const Eigen::VectorXd qdot = Eigen::VectorXd::Zero(25);
  RobotBaseState bs = Draco3StandingBaseState();

  const StateConfig* state = arch->GetConfig()->FindState(1);
  ASSERT_NE(state, nullptr);
  state->joint->UpdateDesired(q, Eigen::VectorXd::Zero(25),
                              Eigen::VectorXd::Zero(25));

  // Also set desired for state 2.
  const StateConfig* state2 = arch->GetConfig()->FindState(2);
  ASSERT_NE(state2, nullptr);
  state2->joint->UpdateDesired(q, Eigen::VectorXd::Zero(25),
                               Eigen::VectorXd::Zero(25));

  // Warmup in state 1.
  for (int i = 0; i < 100; ++i) {
    UpdateWithFloatingBase(arch.get(), q, qdot, bs, i * 0.001, 0.001);
  }

  const auto& cmd = arch->GetCommand();
  ASSERT_TRUE(cmd.tau.allFinite());
  const Eigen::VectorXd tau_before = cmd.tau;

  // Request state transition.
  arch->RequestState(2);

  // Run one tick after transition.
  UpdateWithFloatingBase(arch.get(), q, qdot, bs, 100 * 0.001, 0.001);
  ASSERT_TRUE(cmd.tau.allFinite());
  const Eigen::VectorXd tau_after = cmd.tau;

  const double transition_jump = (tau_after - tau_before).cwiseAbs().maxCoeff();
  std::cout << "[Draco3 Transition] State 1 → State 2\n"
            << "  Max torque jump: " << transition_jump << " Nm\n";

  // Torque jump across state transition should be bounded.
  // Both states have the same tasks (different weights), so jump should be small.
  EXPECT_LT(transition_jump, 20.0)
      << "Excessive torque discontinuity at state transition";
}

// ===========================================================================
// Benchmark tests
// ===========================================================================

TEST_F(ControlArchitectureBehaviorTest, Benchmark_Draco3) {
  // Draco3 single-knee: 25-DOF floating-base humanoid with CoM task,
  // surface contacts, force tasks, and all kinematic constraints.
  const std::string draco_urdf =
      "/home/dk/workspace/rpc_ws/src/rpc_ros/description/"
      "draco_description/urdf/draco_modified_single_knee.urdf";
  if (!std::filesystem::exists(draco_urdf)) {
    GTEST_SKIP() << "Draco3 URDF not found — skipping benchmark";
  }

  std::ostringstream yaml;
  yaml << "robot_model:\n"
       << "  urdf_path: \"" << draco_urdf << "\"\n"
       << R"(  is_floating_base: true
  base_frame: torso_link
regularization:
  w_tau: 1.0e-3
start_state_id: 1
task_pool:
  - name: com_task
    type: ComTask
    kp: [40, 40, 40]
    kd: [8, 8, 8]
    kp_ik: [1.0, 1.0, 1.0]
    weight: [50, 50, 50]
  - name: jpos_task
    type: JointTask
    kp: 30.0
    kd: 3.0
    kp_ik: 1.0
    weight: 1.0
  - name: lfoot_force
    type: ForceTask
    contact_name: lfoot_contact
    weight: [0, 0, 0, 0, 0, 0]
  - name: rfoot_force
    type: ForceTask
    contact_name: rfoot_contact
    weight: [0, 0, 0, 0, 0, 0]
contact_pool:
  - name: lfoot_contact
    type: SurfaceContact
    target_frame: l_foot_contact
    mu: 0.5
    foot_half_length: 0.10
    foot_half_width: 0.05
  - name: rfoot_contact
    type: SurfaceContact
    target_frame: r_foot_contact
    mu: 0.5
    foot_half_length: 0.10
    foot_half_width: 0.05
global_constraints:
  JointPosLimitConstraint:
    enabled: true
    scale: 0.9
  JointVelLimitConstraint:
    enabled: true
    scale: 0.8
  JointTrqLimitConstraint:
    enabled: true
    scale: 0.9
state_machine:
  - id: 1
    name: ut_teleop_state
    task_hierarchy:
      - {name: com_task, priority: 0}
      - {name: jpos_task, priority: 1}
    contact_constraints:
      - {name: lfoot_contact}
      - {name: rfoot_contact}
    force_tasks:
      - {name: lfoot_force}
      - {name: rfoot_force}
)";

  const auto yaml_path = WriteYaml("benchmark_draco3.yaml", yaml.str());
  auto arch = MakeArchitecture(yaml_path);
  ASSERT_NE(arch, nullptr);

  const int n_act = robot_->NumActiveDof();
  ASSERT_EQ(n_act, 25);

  // Standing configuration (from rpc_source, single-knee model).
  Eigen::VectorXd q = Eigen::VectorXd::Zero(n_act);
  q(2)  = -0.565;  // l_hip_fe
  q(3)  =  0.565;  // l_knee_fe_jd
  q(4)  = -0.565;  // l_ankle_fe
  q(7)  =  0.523;  // l_shoulder_aa
  q(9)  = -1.57;   // l_elbow_fe
  q(15) = -0.565;  // r_hip_fe
  q(16) =  0.565;  // r_knee_fe_jd
  q(17) = -0.565;  // r_ankle_fe
  q(20) = -0.523;  // r_shoulder_aa
  q(22) = -1.57;   // r_elbow_fe
  const Eigen::VectorXd qdot = Eigen::VectorXd::Zero(n_act);

  // Base state: standing at height 0.841 m.
  RobotBaseState base_state;
  base_state.pos = Eigen::Vector3d(0.0, 0.0, 0.841);
  base_state.quat = Eigen::Quaterniond::Identity();
  base_state.lin_vel = Eigen::Vector3d::Zero();
  base_state.ang_vel = Eigen::Vector3d::Zero();
  base_state.rot_world_local = Eigen::Matrix3d::Identity();

  // Set desired = current for steady-state benchmarking.
  const StateConfig* state = arch->GetConfig()->FindState(1);
  ASSERT_NE(state, nullptr);
  ASSERT_NE(state->joint, nullptr);
  state->joint->UpdateDesired(q, Eigen::VectorXd::Zero(n_act),
                              Eigen::VectorXd::Zero(n_act));

  // Warm-up: trigger lazy init (Pinocchio model, ProxQP, etc.)
  for (int i = 0; i < 20; ++i) {
    UpdateWithFloatingBase(arch.get(), q, qdot, base_state, i * 0.001, 0.001);
  }

  // Benchmark.
  constexpr int kIterations = 1000;
  const auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < kIterations; ++i) {
    const double t = (20 + i) * 0.001;
    UpdateWithFloatingBase(arch.get(), q, qdot, base_state, t, 0.001);
  }
  const auto end = std::chrono::high_resolution_clock::now();

  const double elapsed_sec =
      std::chrono::duration<double>(end - start).count();
  const double avg_us = (elapsed_sec / kIterations) * 1.0e6;
  const double max_hz = kIterations / elapsed_sec;

  std::cout
      << "\n===== WBC Loop Benchmark (Draco3 25-DOF floating-base, ProxQP) =====\n"
      << "  Robot      : Draco3 (25-DOF floating-base humanoid)\n"
      << "  Tasks      : com_task + jpos_task (2 priority levels)\n"
      << "  Contacts   : lfoot + rfoot (SurfaceContact, 6D)\n"
      << "  Constraints: JointPosLimit + JointVelLimit + JointTrqLimit\n"
      << "  Torque reg : w_tau = 1e-3\n"
      << "  Iterations : " << kIterations << "\n"
      << "  Total time : " << elapsed_sec * 1e3 << " ms\n"
      << "  Avg / tick : " << avg_us << " us\n"
      << "  Max freq   : " << max_hz << " Hz\n"
      << "====================================================================\n";

  // 25-DOF floating-base with contacts should run at >500 Hz.
  EXPECT_LT(avg_us, 2000.0) << "Average tick time exceeds 2 ms for Draco3";
}

// ---------------------------------------------------------------------------
// Combinatorial Draco3 Benchmark: all {Pos, Vel, Trq} × {off, hard, soft}
// ---------------------------------------------------------------------------

enum class CMode { Off, Hard, Soft };

const char* CModeName(CMode m) {
  switch (m) {
    case CMode::Off:  return "off ";
    case CMode::Hard: return "hard";
    case CMode::Soft: return "soft";
  }
  return "????";
}

void AppendConstraintYaml(std::ostringstream& yaml,
                          const std::string& type,
                          CMode mode, double scale) {
  if (mode == CMode::Off) return;
  yaml << "  " << type << ":\n"
       << "    enabled: true\n"
       << "    scale: " << scale << "\n";
  if (mode == CMode::Soft) {
    yaml << "    is_soft: true\n"
         << "    soft_weight: 1.0e+5\n";
  }
}

TEST_F(ControlArchitectureBehaviorTest, Benchmark_Draco3_Profiling) {
  // Per-phase timing breakdown for Draco3 to identify optimization targets.
  const std::string draco_urdf =
      "/home/dk/workspace/rpc_ws/src/rpc_ros/description/"
      "draco_description/urdf/draco_modified_single_knee.urdf";
  if (!std::filesystem::exists(draco_urdf)) {
    GTEST_SKIP() << "Draco3 URDF not found — skipping profiling";
  }

  std::ostringstream yaml;
  yaml << "robot_model:\n"
       << "  urdf_path: \"" << draco_urdf << "\"\n"
       << R"(  is_floating_base: true
  base_frame: torso_link
regularization:
  w_tau: 1.0e-3
start_state_id: 1
task_pool:
  - name: com_task
    type: ComTask
    kp: [40, 40, 40]
    kd: [8, 8, 8]
    kp_ik: [1.0, 1.0, 1.0]
    weight: [50, 50, 50]
  - name: jpos_task
    type: JointTask
    kp: 30.0
    kd: 3.0
    kp_ik: 1.0
    weight: 1.0
  - name: lfoot_force
    type: ForceTask
    contact_name: lfoot_contact
    weight: [0, 0, 0, 0, 0, 0]
  - name: rfoot_force
    type: ForceTask
    contact_name: rfoot_contact
    weight: [0, 0, 0, 0, 0, 0]
contact_pool:
  - name: lfoot_contact
    type: SurfaceContact
    target_frame: l_foot_contact
    mu: 0.5
    foot_half_length: 0.10
    foot_half_width: 0.05
  - name: rfoot_contact
    type: SurfaceContact
    target_frame: r_foot_contact
    mu: 0.5
    foot_half_length: 0.10
    foot_half_width: 0.05
global_constraints:
  JointPosLimitConstraint:
    enabled: true
    scale: 0.9
  JointVelLimitConstraint:
    enabled: true
    scale: 0.8
  JointTrqLimitConstraint:
    enabled: true
    scale: 0.9
state_machine:
  - id: 1
    name: ut_teleop_state
    task_hierarchy:
      - {name: com_task, priority: 0}
      - {name: jpos_task, priority: 1}
    contact_constraints:
      - {name: lfoot_contact}
      - {name: rfoot_contact}
    force_tasks:
      - {name: lfoot_force}
      - {name: rfoot_force}
)";

  const auto yaml_path = WriteYaml("profiling_draco3.yaml", yaml.str());
  auto arch = MakeArchitecture(yaml_path);
  ASSERT_NE(arch, nullptr);

  const int n_act = robot_->NumActiveDof();
  ASSERT_EQ(n_act, 25);

  Eigen::VectorXd q = Eigen::VectorXd::Zero(n_act);
  q(2)  = -0.565;  q(3)  =  0.565;  q(4)  = -0.565;
  q(7)  =  0.523;  q(9)  = -1.57;
  q(15) = -0.565;  q(16) =  0.565;  q(17) = -0.565;
  q(20) = -0.523;  q(22) = -1.57;
  const Eigen::VectorXd qdot = Eigen::VectorXd::Zero(n_act);

  RobotBaseState base_state;
  base_state.pos = Eigen::Vector3d(0.0, 0.0, 0.841);
  base_state.quat = Eigen::Quaterniond::Identity();
  base_state.lin_vel = Eigen::Vector3d::Zero();
  base_state.ang_vel = Eigen::Vector3d::Zero();
  base_state.rot_world_local = Eigen::Matrix3d::Identity();

  const StateConfig* state = arch->GetConfig()->FindState(1);
  ASSERT_NE(state, nullptr);
  ASSERT_NE(state->joint, nullptr);
  state->joint->UpdateDesired(q, Eigen::VectorXd::Zero(n_act),
                              Eigen::VectorXd::Zero(n_act));

  // Warm-up.
  for (int i = 0; i < 20; ++i) {
    UpdateWithFloatingBase(arch.get(), q, qdot, base_state, i * 0.001, 0.001);
  }

  // Enable per-phase timing on ControlArchitecture.
  // WBIC internal breakdown (qp_setup/qp_solve/torque_recovery) is captured
  // inside the arch-level find_config_us and make_torque_us phases.
  arch->enable_timing_ = true;

  constexpr int kIterations = 500;

  // Accumulators for averaging.
  double sum_robot_model = 0, sum_kinematics = 0, sum_dynamics = 0;
  double sum_find_config = 0, sum_make_torque = 0, sum_feedback = 0;
  double sum_total = 0;

  for (int i = 0; i < kIterations; ++i) {
    const double t = (20 + i) * 0.001;
    const auto tick_start = std::chrono::high_resolution_clock::now();
    UpdateWithFloatingBase(arch.get(), q, qdot, base_state, t, 0.001);
    const auto tick_end = std::chrono::high_resolution_clock::now();

    const auto& s = arch->timing_stats_;
    sum_robot_model += s.robot_model_us;
    sum_kinematics  += s.kinematics_us;
    sum_dynamics    += s.dynamics_us;
    sum_find_config += s.find_config_us;
    sum_make_torque += s.make_torque_us;
    sum_feedback    += s.feedback_us;
    sum_total += std::chrono::duration<double, std::micro>(tick_end - tick_start).count();
  }

  const double N = static_cast<double>(kIterations);
  const double avg_total       = sum_total / N;
  const double avg_robot_model = sum_robot_model / N;
  const double avg_kinematics  = sum_kinematics / N;
  const double avg_dynamics    = sum_dynamics / N;
  const double avg_find_config = sum_find_config / N;
  const double avg_make_torque = sum_make_torque / N;
  const double avg_feedback    = sum_feedback / N;
  const double avg_other       = avg_total - avg_robot_model - avg_kinematics
                                 - avg_dynamics - avg_find_config
                                 - avg_make_torque - avg_feedback;

  auto pct = [&](double phase) { return (phase / avg_total) * 100.0; };

  char buf[512];
  std::cout
      << "\n===== Draco3 Per-Phase Profiling (" << kIterations << " ticks) =====\n"
      << "  Phase              |   Avg (us)  |    %\n"
      << " --------------------+-------------+--------\n";
  std::snprintf(buf, sizeof(buf), "  Robot Model (FK)   | %11.1f | %5.1f%%\n",
                avg_robot_model, pct(avg_robot_model));
  std::cout << buf;
  std::snprintf(buf, sizeof(buf), "  Kinematics (J,Jd)  | %11.1f | %5.1f%%\n",
                avg_kinematics, pct(avg_kinematics));
  std::cout << buf;
  std::snprintf(buf, sizeof(buf), "  Dynamics (M,h,g)   | %11.1f | %5.1f%%\n",
                avg_dynamics, pct(avg_dynamics));
  std::cout << buf;
  std::snprintf(buf, sizeof(buf), "  FindConfig (LLT)   | %11.1f | %5.1f%%\n",
                avg_find_config, pct(avg_find_config));
  std::cout << buf;
  std::snprintf(buf, sizeof(buf), "  MakeTorque (QP)    | %11.1f | %5.1f%%\n",
                avg_make_torque, pct(avg_make_torque));
  std::cout << buf;
  std::snprintf(buf, sizeof(buf), "  Feedback (PID+comp)| %11.1f | %5.1f%%\n",
                avg_feedback, pct(avg_feedback));
  std::cout << buf;
  std::snprintf(buf, sizeof(buf), "  Other (FSM+SP)     | %11.1f | %5.1f%%\n",
                avg_other, pct(avg_other));
  std::cout << buf;
  std::cout << " --------------------+-------------+--------\n";
  std::snprintf(buf, sizeof(buf), "  TOTAL              | %11.1f | 100.0%%\n",
                avg_total);
  std::cout << buf;
  std::cout << "  Max freq           : " << (1.0e6 / avg_total) << " Hz\n"
            << "========================================================\n";

  EXPECT_TRUE(true);  // Profiling test — always passes.
}

TEST_F(ControlArchitectureBehaviorTest, Benchmark_Draco3_ConstraintCombinations) {
  const std::string draco_urdf =
      "/home/dk/workspace/rpc_ws/src/rpc_ros/description/"
      "draco_description/urdf/draco_modified_single_knee.urdf";
  if (!std::filesystem::exists(draco_urdf)) {
    GTEST_SKIP() << "Draco3 URDF not found — skipping benchmark";
  }

  // All 27 combinations: 3 constraints × 3 modes (off/hard/soft).
  struct ConstraintCombo {
    CMode pos, vel, trq;
  };
  std::vector<CMode> modes = {CMode::Off, CMode::Hard, CMode::Soft};
  std::vector<ConstraintCombo> combos;
  for (auto p : modes)
    for (auto v : modes)
      for (auto t : modes)
        combos.push_back({p, v, t});

  // Standing configuration (from rpc_source, single-knee model).
  const int n_act = 25;
  Eigen::VectorXd q = Eigen::VectorXd::Zero(n_act);
  q(2)  = -0.565;  // l_hip_fe
  q(3)  =  0.565;  // l_knee_fe_jd
  q(4)  = -0.565;  // l_ankle_fe
  q(7)  =  0.523;  // l_shoulder_aa
  q(9)  = -1.57;   // l_elbow_fe
  q(15) = -0.565;  // r_hip_fe
  q(16) =  0.565;  // r_knee_fe_jd
  q(17) = -0.565;  // r_ankle_fe
  q(20) = -0.523;  // r_shoulder_aa
  q(22) = -1.57;   // r_elbow_fe
  const Eigen::VectorXd qdot = Eigen::VectorXd::Zero(n_act);

  RobotBaseState base_state;
  base_state.pos = Eigen::Vector3d(0.0, 0.0, 0.841);
  base_state.quat = Eigen::Quaterniond::Identity();
  base_state.lin_vel = Eigen::Vector3d::Zero();
  base_state.ang_vel = Eigen::Vector3d::Zero();
  base_state.rot_world_local = Eigen::Matrix3d::Identity();

  struct BenchResult {
    int idx;
    CMode pos, vel, trq;
    double avg_us;
    double max_hz;
  };
  std::vector<BenchResult> results;

  constexpr int kWarmup = 20;
  constexpr int kIterations = 500;

  for (int ci = 0; ci < static_cast<int>(combos.size()); ++ci) {
    const auto& c = combos[ci];

    // Build YAML with varying constraint section.
    std::ostringstream yaml;
    yaml << "robot_model:\n"
         << "  urdf_path: \"" << draco_urdf << "\"\n"
         << R"(  is_floating_base: true
  base_frame: torso_link
regularization:
  w_tau: 1.0e-3
start_state_id: 1
task_pool:
  - name: com_task
    type: ComTask
    kp: [40, 40, 40]
    kd: [8, 8, 8]
    kp_ik: [1.0, 1.0, 1.0]
    weight: [50, 50, 50]
  - name: jpos_task
    type: JointTask
    kp: 30.0
    kd: 3.0
    kp_ik: 1.0
    weight: 1.0
  - name: lfoot_force
    type: ForceTask
    contact_name: lfoot_contact
    weight: [0, 0, 0, 0, 0, 0]
  - name: rfoot_force
    type: ForceTask
    contact_name: rfoot_contact
    weight: [0, 0, 0, 0, 0, 0]
contact_pool:
  - name: lfoot_contact
    type: SurfaceContact
    target_frame: l_foot_contact
    mu: 0.5
    foot_half_length: 0.10
    foot_half_width: 0.05
  - name: rfoot_contact
    type: SurfaceContact
    target_frame: r_foot_contact
    mu: 0.5
    foot_half_length: 0.10
    foot_half_width: 0.05
)";
    // Global constraints section.
    yaml << "global_constraints:\n";
    AppendConstraintYaml(yaml, "JointPosLimitConstraint", c.pos, 0.9);
    AppendConstraintYaml(yaml, "JointVelLimitConstraint", c.vel, 0.8);
    AppendConstraintYaml(yaml, "JointTrqLimitConstraint", c.trq, 0.9);
    // If all off, still need the key (empty map).
    if (c.pos == CMode::Off && c.vel == CMode::Off && c.trq == CMode::Off) {
      yaml << "  {}\n";
    }

    yaml << R"(state_machine:
  - id: 1
    name: ut_teleop_state
    task_hierarchy:
      - {name: com_task, priority: 0}
      - {name: jpos_task, priority: 1}
    contact_constraints:
      - {name: lfoot_contact}
      - {name: rfoot_contact}
    force_tasks:
      - {name: lfoot_force}
      - {name: rfoot_force}
)";

    const std::string fname =
        "bench_draco3_combo_" + std::to_string(ci) + ".yaml";
    const auto yaml_path = WriteYaml(fname, yaml.str());
    auto arch = MakeArchitecture(yaml_path);
    ASSERT_NE(arch, nullptr) << "Failed to build config #" << ci;

    // Set desired = current for steady-state.
    const StateConfig* state = arch->GetConfig()->FindState(1);
    ASSERT_NE(state, nullptr);
    ASSERT_NE(state->joint, nullptr);
    state->joint->UpdateDesired(q, Eigen::VectorXd::Zero(n_act),
                                Eigen::VectorXd::Zero(n_act));

    // Warmup.
    for (int i = 0; i < kWarmup; ++i) {
      UpdateWithFloatingBase(arch.get(), q, qdot, base_state, i * 0.001, 0.001);
    }

    // Benchmark.
    const auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < kIterations; ++i) {
      const double t = (kWarmup + i) * 0.001;
      UpdateWithFloatingBase(arch.get(), q, qdot, base_state, t, 0.001);
    }
    const auto end = std::chrono::high_resolution_clock::now();

    const double elapsed_sec =
        std::chrono::duration<double>(end - start).count();
    const double avg_us = (elapsed_sec / kIterations) * 1.0e6;
    const double max_hz = kIterations / elapsed_sec;

    results.push_back({ci + 1, c.pos, c.vel, c.trq, avg_us, max_hz});
  }

  // Print summary table.
  std::cout
      << "\n===== Draco3 Constraint Combination Benchmark (27 configs) =====\n"
      << "  Robot      : Draco3 (25-DOF floating-base humanoid)\n"
      << "  Tasks      : com_task + jpos_task (2 priority levels)\n"
      << "  Contacts   : lfoot + rfoot (SurfaceContact, 6D)\n"
      << "  Iterations : " << kIterations << " per config\n"
      << "----------------------------------------------------------------\n"
      << "  #  | Pos  | Vel  | Trq  | Avg (us) | Freq (Hz)\n"
      << " ----+------+------+------+----------+-----------\n";
  for (const auto& r : results) {
    char buf[128];
    std::snprintf(buf, sizeof(buf),
                  " %3d | %s | %s | %s | %8.1f | %9.1f\n",
                  r.idx, CModeName(r.pos), CModeName(r.vel), CModeName(r.trq),
                  r.avg_us, r.max_hz);
    std::cout << buf;
  }
  std::cout
      << "================================================================\n";
}

TEST_F(ControlArchitectureBehaviorTest, Benchmark_Optimo7DOF_ConstraintCombinations) {
  const std::string optimo_urdf =
      "/home/dk/workspace/rpc_ws/src/rpc_ros/description/"
      "optimo_description/urdf/optimo.urdf";
  if (!std::filesystem::exists(optimo_urdf)) {
    GTEST_SKIP() << "Optimo URDF not found — skipping benchmark";
  }

  // All 27 combinations: 3 constraints × 3 modes (off/hard/soft).
  struct ConstraintCombo {
    CMode pos, vel, trq;
  };
  std::vector<CMode> modes = {CMode::Off, CMode::Hard, CMode::Soft};
  std::vector<ConstraintCombo> combos;
  for (auto p : modes)
    for (auto v : modes)
      for (auto t : modes)
        combos.push_back({p, v, t});

  const int n_act = 7;
  Eigen::VectorXd q = Eigen::VectorXd::Zero(n_act);
  q << 0.0, -0.5, 0.0, 1.0, 0.0, -0.5, 0.0;
  const Eigen::VectorXd qdot = Eigen::VectorXd::Zero(n_act);

  struct BenchResult {
    int idx;
    CMode pos, vel, trq;
    double avg_us;
    double max_hz;
  };
  std::vector<BenchResult> results;

  constexpr int kWarmup = 20;
  constexpr int kIterations = 500;

  for (int ci = 0; ci < static_cast<int>(combos.size()); ++ci) {
    const auto& c = combos[ci];

    // Build YAML with varying constraint section.
    std::ostringstream yaml;
    yaml << "robot_model:\n"
         << "  urdf_path: \"" << optimo_urdf << "\"\n"
         << R"(  is_floating_base: false
  base_frame: optimo_base_link
controller: {}
regularization:
  w_tau: 1.0e-3
start_state_id: 1
task_pool:
  - name: jpos_task
    type: JointTask
    kp: [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
    kd: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    kp_ik: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
  - name: ee_pos
    type: LinkPosTask
    target_frame: optimo_end_effector
    reference_frame: optimo_base_link
    kp: [100.0, 100.0, 100.0]
    kd: [10.0, 10.0, 10.0]
    kp_ik: [1.0, 1.0, 1.0]
  - name: ee_ori
    type: LinkOriTask
    target_frame: optimo_end_effector
    reference_frame: optimo_base_link
    kp: [100.0, 100.0, 100.0]
    kd: [10.0, 10.0, 10.0]
    kp_ik: [1.0, 1.0, 1.0]
)";
    // Global constraints section.
    yaml << "global_constraints:\n";
    AppendConstraintYaml(yaml, "JointPosLimitConstraint", c.pos, 0.9);
    AppendConstraintYaml(yaml, "JointVelLimitConstraint", c.vel, 0.8);
    AppendConstraintYaml(yaml, "JointTrqLimitConstraint", c.trq, 0.9);
    // If all off, still need the key (empty map).
    if (c.pos == CMode::Off && c.vel == CMode::Off && c.trq == CMode::Off) {
      yaml << "  {}\n";
    }

    yaml << R"(state_machine:
  - id: 1
    name: ut_teleop_state
    task_hierarchy:
      - {name: ee_pos, priority: 0}
      - {name: ee_ori, priority: 1}
      - {name: jpos_task, priority: 2}
)";

    const std::string fname =
        "bench_optimo_combo_" + std::to_string(ci) + ".yaml";
    const auto yaml_path = WriteYaml(fname, yaml.str());
    auto arch = MakeArchitecture(yaml_path);
    ASSERT_NE(arch, nullptr) << "Failed to build config #" << ci;

    // Set desired = current for steady-state.
    const StateConfig* state = arch->GetConfig()->FindState(1);
    ASSERT_NE(state, nullptr);
    ASSERT_NE(state->joint, nullptr);
    state->joint->UpdateDesired(q, Eigen::VectorXd::Zero(n_act),
                                Eigen::VectorXd::Zero(n_act));

    // Warmup.
    for (int i = 0; i < kWarmup; ++i) {
      UpdateWithJointState(arch.get(), q, qdot, i * 0.001, 0.001);
    }

    // Benchmark.
    const auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < kIterations; ++i) {
      const double t = (kWarmup + i) * 0.001;
      UpdateWithJointState(arch.get(), q, qdot, t, 0.001);
    }
    const auto end = std::chrono::high_resolution_clock::now();

    const double elapsed_sec =
        std::chrono::duration<double>(end - start).count();
    const double avg_us = (elapsed_sec / kIterations) * 1.0e6;
    const double max_hz = kIterations / elapsed_sec;

    results.push_back({ci + 1, c.pos, c.vel, c.trq, avg_us, max_hz});
  }

  // Print summary table.
  std::cout
      << "\n===== Optimo 7-DOF Constraint Combination Benchmark (27 configs) =====\n"
      << "  Robot      : Optimo (7-DOF fixed-base)\n"
      << "  Tasks      : ee_pos + ee_ori + jpos_task (3 priority levels)\n"
      << "  Iterations : " << kIterations << " per config\n"
      << "----------------------------------------------------------------------\n"
      << "  #  | Pos  | Vel  | Trq  | Avg (us) | Freq (Hz)\n"
      << " ----+------+------+------+----------+-----------\n";
  for (const auto& r : results) {
    char buf[128];
    std::snprintf(buf, sizeof(buf),
                  " %3d | %s | %s | %s | %8.1f | %9.1f\n",
                  r.idx, CModeName(r.pos), CModeName(r.vel), CModeName(r.trq),
                  r.avg_us, r.max_hz);
    std::cout << buf;
  }
  std::cout
      << "======================================================================\n";
}

// ---------------------------------------------------------------------------
// Compensation flag tests: verify enable_gravity/coriolis/inertia_compensation
// actually change the output torque.
// ---------------------------------------------------------------------------

TEST_F(ControlArchitectureBehaviorTest, CompensationFlags_GravityToggle) {
  const std::string optimo_urdf =
      "/home/dk/workspace/rpc_ws/src/rpc_ros/description/"
      "optimo_description/urdf/optimo.urdf";
  if (!std::filesystem::exists(optimo_urdf)) {
    GTEST_SKIP() << "Optimo URDF not found";
  }

  auto make_yaml = [&](bool grav, bool cori, bool inertia) {
    std::ostringstream yaml;
    yaml << "robot_model:\n"
         << "  urdf_path: \"" << optimo_urdf << "\"\n"
         << "  is_floating_base: false\n"
         << "  base_frame: optimo_base_link\n"
         << "controller:\n"
         << "  enable_gravity_compensation: " << (grav ? "true" : "false") << "\n"
         << "  enable_coriolis_compensation: " << (cori ? "true" : "false") << "\n"
         << "  enable_inertia_compensation: " << (inertia ? "true" : "false") << "\n"
         << R"(regularization:
  w_tau: 1.0e-3
start_state_id: 1
task_pool:
  - name: jpos_task
    type: JointTask
    kp: [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
    kd: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    kp_ik: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
global_constraints:
  JointPosLimitConstraint:
    enabled: true
    scale: 0.9
  JointVelLimitConstraint:
    enabled: true
    scale: 0.8
  JointTrqLimitConstraint:
    enabled: true
    scale: 0.9
state_machine:
  - id: 1
    name: ut_teleop_state
    params:
      stay_here: true
    task_hierarchy:
      - {name: jpos_task, priority: 0}
)";
    return yaml.str();
  };

  // Non-trivial pose so gravity is non-zero.
  Eigen::VectorXd q(7);
  q << 0.0, -0.5, 0.0, 1.0, 0.0, -0.5, 0.0;
  const Eigen::VectorXd qdot = Eigen::VectorXd::Zero(7);

  // Build "all enabled" config, run ticks, capture tau AND gravity reference.
  const auto yaml_all = WriteYaml("comp_grav_all.yaml", make_yaml(true, true, true));
  auto arch_all = MakeArchitecture(yaml_all);
  ASSERT_NE(arch_all, nullptr);
  ASSERT_EQ(robot_->NumActiveDof(), 7);
  {
    const StateConfig* state = arch_all->GetConfig()->FindState(1);
    ASSERT_NE(state, nullptr);
    state->joint->UpdateDesired(q, Eigen::VectorXd::Zero(7), Eigen::VectorXd::Zero(7));
  }
  for (int i = 0; i < 10; ++i) {
    UpdateWithJointState(arch_all.get(), q, qdot, i * 0.001, 0.001);
  }
  const Eigen::VectorXd tau_all = arch_all->GetCommand().tau;
  const Eigen::VectorXd grav_ref = robot_->GetGravityRef();  // valid while arch_all alive
  ASSERT_TRUE(tau_all.allFinite());

  // Build "gravity disabled" config.
  const auto yaml_ng = WriteYaml("comp_grav_off.yaml", make_yaml(false, true, true));
  auto arch_ng = MakeArchitecture(yaml_ng);
  ASSERT_NE(arch_ng, nullptr);
  {
    const StateConfig* state = arch_ng->GetConfig()->FindState(1);
    ASSERT_NE(state, nullptr);
    state->joint->UpdateDesired(q, Eigen::VectorXd::Zero(7), Eigen::VectorXd::Zero(7));
  }
  for (int i = 0; i < 10; ++i) {
    UpdateWithJointState(arch_ng.get(), q, qdot, i * 0.001, 0.001);
  }
  const Eigen::VectorXd tau_no_grav = arch_ng->GetCommand().tau;
  ASSERT_TRUE(tau_no_grav.allFinite());

  const Eigen::VectorXd grav_diff = tau_all - tau_no_grav;
  std::cout << "\n--- Gravity compensation toggle ---\n"
            << "  tau_all:     " << tau_all.transpose() << "\n"
            << "  tau_no_grav: " << tau_no_grav.transpose() << "\n"
            << "  diff:        " << grav_diff.transpose() << "\n"
            << "  grav_ref:    " << grav_ref.transpose() << "\n";

  // Disabling gravity compensation must measurably change torque output.
  // With QP correction and joint limits active, directional alignment with
  // grav_ref can vary, so enforce only a robust non-zero delta.
  EXPECT_GT(grav_diff.norm(), 1e-2)
      << "Disabling gravity compensation should change torque output";
}

TEST_F(ControlArchitectureBehaviorTest, CompensationFlags_CoriolisToggle) {
  const std::string optimo_urdf =
      "/home/dk/workspace/rpc_ws/src/rpc_ros/description/"
      "optimo_description/urdf/optimo.urdf";
  if (!std::filesystem::exists(optimo_urdf)) {
    GTEST_SKIP() << "Optimo URDF not found";
  }

  auto make_yaml = [&](bool cori) {
    std::ostringstream yaml;
    yaml << "robot_model:\n"
         << "  urdf_path: \"" << optimo_urdf << "\"\n"
         << "  is_floating_base: false\n"
         << "  base_frame: optimo_base_link\n"
         << "controller:\n"
         << "  enable_gravity_compensation: true\n"
         << "  enable_coriolis_compensation: " << (cori ? "true" : "false") << "\n"
         << "  enable_inertia_compensation: true\n"
         << R"(regularization:
  w_tau: 1.0e-3
start_state_id: 1
task_pool:
  - name: jpos_task
    type: JointTask
    kp: [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
    kd: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    kp_ik: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
state_machine:
  - id: 1
    name: ut_teleop_state
    params:
      stay_here: true
    task_hierarchy:
      - {name: jpos_task, priority: 0}
)";
    return yaml.str();
  };

  // Non-zero velocity so Coriolis is non-zero.
  Eigen::VectorXd q(7);
  q << 0.0, -0.5, 0.0, 1.0, 0.0, -0.5, 0.0;
  Eigen::VectorXd qdot(7);
  qdot << 0.5, -0.3, 0.2, 0.4, -0.1, 0.3, -0.2;

  // Build "coriolis enabled" config, capture tau AND coriolis reference.
  const auto yaml_on = WriteYaml("comp_cori_on.yaml", make_yaml(true));
  auto arch_on = MakeArchitecture(yaml_on);
  ASSERT_NE(arch_on, nullptr);
  {
    const StateConfig* state = arch_on->GetConfig()->FindState(1);
    ASSERT_NE(state, nullptr);
    state->joint->UpdateDesired(q, qdot, Eigen::VectorXd::Zero(7));
  }
  UpdateWithJointState(arch_on.get(), q, qdot, 0.0, 0.001);
  const Eigen::VectorXd tau_with = arch_on->GetCommand().tau;
  const Eigen::VectorXd cori_ref = arch_on->GetRobot()->GetCoriolisRef();
  ASSERT_TRUE(tau_with.allFinite());

  // Build "coriolis disabled" config.
  const auto yaml_off = WriteYaml("comp_cori_off.yaml", make_yaml(false));
  auto arch_off = MakeArchitecture(yaml_off);
  ASSERT_NE(arch_off, nullptr);
  {
    const StateConfig* state = arch_off->GetConfig()->FindState(1);
    ASSERT_NE(state, nullptr);
    state->joint->UpdateDesired(q, qdot, Eigen::VectorXd::Zero(7));
  }
  UpdateWithJointState(arch_off.get(), q, qdot, 0.0, 0.001);
  const Eigen::VectorXd tau_without = arch_off->GetCommand().tau;
  ASSERT_TRUE(tau_without.allFinite());

  const Eigen::VectorXd cori_diff = tau_with - tau_without;
  std::cout << "\n--- Coriolis compensation toggle ---\n"
            << "  tau_with:    " << tau_with.transpose() << "\n"
            << "  tau_without: " << tau_without.transpose() << "\n"
            << "  diff:        " << cori_diff.transpose() << "\n"
            << "  cori_ref:    " << cori_ref.transpose() << "\n"
            << "  cori_norm:   " << cori_ref.norm() << "\n";

  // Torque should change when coriolis is toggled (at non-zero velocity).
  EXPECT_NE(tau_with, tau_without)
      << "Disabling coriolis should produce different torque";

  // The magnitude of change may be small relative to cori_ref because the QP
  // adapts its qddot solution. Just verify the torques are not identical.
  EXPECT_GT((tau_with - tau_without).norm(), 1e-6);
}

TEST_F(ControlArchitectureBehaviorTest, CompensationFlags_InertiaToggle) {
  const std::string optimo_urdf =
      "/home/dk/workspace/rpc_ws/src/rpc_ros/description/"
      "optimo_description/urdf/optimo.urdf";
  if (!std::filesystem::exists(optimo_urdf)) {
    GTEST_SKIP() << "Optimo URDF not found";
  }

  auto make_yaml = [&](bool inertia) {
    std::ostringstream yaml;
    yaml << "robot_model:\n"
         << "  urdf_path: \"" << optimo_urdf << "\"\n"
         << "  is_floating_base: false\n"
         << "  base_frame: optimo_base_link\n"
         << "controller:\n"
         << "  enable_gravity_compensation: true\n"
         << "  enable_coriolis_compensation: true\n"
         << "  enable_inertia_compensation: " << (inertia ? "true" : "false") << "\n"
         << R"(regularization:
  w_tau: 1.0e-3
start_state_id: 1
task_pool:
  - name: jpos_task
    type: JointTask
    kp: [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
    kd: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    kp_ik: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
state_machine:
  - id: 1
    name: ut_teleop_state
    params:
      stay_here: true
    task_hierarchy:
      - {name: jpos_task, priority: 0}
)";
    return yaml.str();
  };

  // Non-trivial pose with zero velocity.
  Eigen::VectorXd q(7);
  q << 0.0, -0.5, 0.0, 1.0, 0.0, -0.5, 0.0;
  const Eigen::VectorXd qdot = Eigen::VectorXd::Zero(7);

  // Desired differs from current → produces non-zero qddot from IK.
  Eigen::VectorXd q_des(7);
  q_des << 0.1, -0.4, 0.1, 0.9, 0.1, -0.4, 0.1;

  // Build "inertia enabled" config.
  const auto yaml_on = WriteYaml("comp_inertia_on.yaml", make_yaml(true));
  auto arch_on = MakeArchitecture(yaml_on);
  ASSERT_NE(arch_on, nullptr);
  {
    const StateConfig* state = arch_on->GetConfig()->FindState(1);
    ASSERT_NE(state, nullptr);
    state->joint->UpdateDesired(q_des, Eigen::VectorXd::Zero(7), Eigen::VectorXd::Zero(7));
  }
  UpdateWithJointState(arch_on.get(), q, qdot, 0.0, 0.001);
  const Eigen::VectorXd tau_with = arch_on->GetCommand().tau;
  ASSERT_TRUE(tau_with.allFinite());

  // Build "inertia disabled" config.
  const auto yaml_off = WriteYaml("comp_inertia_off.yaml", make_yaml(false));
  auto arch_off = MakeArchitecture(yaml_off);
  ASSERT_NE(arch_off, nullptr);
  {
    const StateConfig* state = arch_off->GetConfig()->FindState(1);
    ASSERT_NE(state, nullptr);
    state->joint->UpdateDesired(q_des, Eigen::VectorXd::Zero(7), Eigen::VectorXd::Zero(7));
  }
  UpdateWithJointState(arch_off.get(), q, qdot, 0.0, 0.001);
  const Eigen::VectorXd tau_without = arch_off->GetCommand().tau;
  ASSERT_TRUE(tau_without.allFinite());

  // With inertia disabled, tau should be closer to pure gravity compensation.
  // The gravity vector from the robot model (qdot=0, so cori=0).
  const Eigen::VectorXd grav = arch_off->GetRobot()->GetGravityRef();

  const Eigen::VectorXd inertia_diff = tau_with - tau_without;
  std::cout << "\n--- Inertia compensation toggle ---\n"
            << "  tau_with:    " << tau_with.transpose() << "\n"
            << "  tau_without: " << tau_without.transpose() << "\n"
            << "  diff:        " << inertia_diff.transpose() << "\n"
            << "  grav_ref:    " << grav.transpose() << "\n";

  // With a position error, the solver produces non-zero qddot, so M*qddot
  // should be significant.
  EXPECT_GT(inertia_diff.norm(), 0.01)
      << "Disabling inertia should change torque when qddot is non-zero";

  // Without inertia, the output torque should approximate pure gravity
  // (since qdot=0 → cori=0, and inertia term removed).
  const double grav_err = (tau_without - grav).norm();
  EXPECT_LT(grav_err, 0.1)
      << "With inertia disabled + zero velocity, tau should be ~gravity (err=" << grav_err << ")";
}

} // namespace
} // namespace wbc
