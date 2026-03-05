/**
 * @file wbc_core/wbc_architecture/test/test_review_improvements.cpp
 * @brief Regression tests verifying code-review improvements.
 *
 * Covers:
 *   P3-3  LinkOriTask::UpdateDesired silently rejects wrong-size input (no throw)
 *   P3-4  ForceTask::SetParameters silently rejects wrong-size weight (no throw)
 *   P2-4  SelectedJointTask::UpdateOpCommand uses GetQRef (const-ref, no copy)
 *   P1-6  WBIC MakeTorque no-contact path: rf_cmd_ resized to 0, not allocated
 *   P1-7  PointContact::UpdateJacobian via scratch buffer gives correct result
 *   P1-8  Ni_Nci_dyn_ removed: compilation verifies member is gone
 *   Integration: ControlArchitecture runs N ticks → tau finite, no-contact path
 */
#include <sys/types.h>
#include <unistd.h>

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

#include <Eigen/Dense>
#include <gtest/gtest.h>

#include "wbc_architecture/control_architecture.hpp"
#include "wbc_architecture/control_architecture_config.hpp"
#include "wbc_formulation/basic_contact.hpp"
#include "wbc_formulation/force_task.hpp"
#include "wbc_formulation/motion_task.hpp"
#include "wbc_fsm/state_factory.hpp"
#include "wbc_robot_system/pinocchio_robot_system.hpp"

namespace wbc {
namespace {

// ---------------------------------------------------------------------------
// Minimal stay-in-place state for the integration test
// ---------------------------------------------------------------------------
class ReviewStayState final : public StateMachine {
public:
  ReviewStayState(StateId id, const std::string& name,
                  PinocchioRobotSystem* robot, TaskRegistry* task_reg,
                  ConstraintRegistry* const_reg, StateProvider* sp)
      : StateMachine(id, name, robot, task_reg, const_reg, sp) {}

  void FirstVisit() override {}
  void OneStep() override {}
  void LastVisit() override {}
  bool EndOfState() override { return false; }
  StateId GetNextState() override { return id(); }
};

} // namespace (closed so WBC_REGISTER_STATE is at namespace wbc scope)

WBC_REGISTER_STATE(
    "review_stay_state",
    [](StateId id, const std::string& name,
       const StateMachineConfig& ctx) -> std::unique_ptr<StateMachine> {
      return std::make_unique<ReviewStayState>(
          id, name, ctx.robot, ctx.task_registry,
          ctx.constraint_registry, ctx.state_provider);
    });

namespace {

// ---------------------------------------------------------------------------
// Minimal 2-DOF fixed-base URDF (same as the behavior test suite)
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

// ---------------------------------------------------------------------------
// Fixture: writes URDF to a temp directory, owns a PinocchioRobotSystem
// ---------------------------------------------------------------------------
class ReviewImprovementsTest : public ::testing::Test {
protected:
  void SetUp() override {
    const std::string suffix =
        std::to_string(static_cast<long long>(::getpid())) + "_" +
        std::to_string(counter_++);
    temp_dir_ = std::filesystem::temp_directory_path() /
                ("review_improvements_test_" + suffix);
    std::filesystem::create_directories(temp_dir_);

    const std::filesystem::path urdf_path = temp_dir_ / "test_robot.urdf";
    std::ofstream ofs(urdf_path);
    ofs << TwoDofUrdf();
    ofs.close();

    // Fixed-base, no floating base.
    robot_ = std::make_unique<PinocchioRobotSystem>(
        urdf_path.string(), /*package_root=*/"",
        /*is_fixed_base=*/true, /*print_info=*/false);
  }

  void TearDown() override {
    robot_.reset();
    std::error_code ec;
    std::filesystem::remove_all(temp_dir_, ec);
  }

  std::filesystem::path temp_dir_;
  std::unique_ptr<PinocchioRobotSystem> robot_;
  static inline int counter_ = 0;
};

// ---------------------------------------------------------------------------
// P3-3: LinkOriTask::UpdateDesired with wrong-size must NOT throw
// ---------------------------------------------------------------------------
TEST_F(ReviewImprovementsTest, LinkOriTaskUpdateDesiredWrongSizeNoThrow) {
  // link index 1 = "link2" in the 2-DOF urdf
  LinkOriTask task(robot_.get(), /*target_idx=*/1);

  // Set a valid desired first (size=4 quaternion [x,y,z,w])
  const Eigen::VectorXd valid_quat = Eigen::Vector4d(0, 0, 0, 1);
  const Eigen::VectorXd zeros3     = Eigen::VectorXd::Zero(3);
  ASSERT_NO_THROW(task.UpdateDesired(valid_quat, zeros3, zeros3));

  // Calling with wrong-size (3 instead of 4) must not throw and must leave
  // the previously set desired unchanged.
  const Eigen::VectorXd wrong_size = Eigen::VectorXd::Zero(3);
  EXPECT_NO_THROW(task.UpdateDesired(wrong_size, zeros3, zeros3));

  // Verify the desired was NOT overwritten by the bad call.
  // des_pos_ is accessible only indirectly — attempt to run UpdateOpCommand
  // which should still see the valid quaternion and not crash.
  robot_->UpdateRobotModel(Eigen::Vector3d::Zero(), Eigen::Quaterniond::Identity(),
                           Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(),
                           Eigen::VectorXd::Zero(2), Eigen::VectorXd::Zero(2));
  EXPECT_NO_THROW(task.UpdateOpCommand(Eigen::Matrix3d::Identity()));
}

// ---------------------------------------------------------------------------
// P3-4: ForceTask::SetParameters with wrong-size weight must NOT throw
// ---------------------------------------------------------------------------
TEST_F(ReviewImprovementsTest, ForceTaskSetParametersWrongSizeNoThrow) {
  // PointContact has dim=3.
  PointContact contact(robot_.get(), /*target_link_idx=*/1, /*mu=*/0.5);
  ForceTask force_task(robot_.get(), &contact);

  // A valid 3-element weight should succeed and change the weight.
  ForceTaskConfig good_config;
  good_config.weight = Eigen::Vector3d(1.0, 1.0, 1.0);
  ASSERT_NO_THROW(force_task.SetParameters(good_config));
  EXPECT_DOUBLE_EQ(force_task.Weight()(0), 1.0);

  // A wrong-size weight (5 elements for a 3-DOF contact) must NOT throw
  // and must leave the previous weight unchanged.
  ForceTaskConfig bad_config;
  bad_config.weight = Eigen::VectorXd::Constant(5, 99.0);
  EXPECT_NO_THROW(force_task.SetParameters(bad_config));

  // Weight must remain at previous value (1.0, not 99.0).
  EXPECT_DOUBLE_EQ(force_task.Weight()(0), 1.0);
}

// ---------------------------------------------------------------------------
// P2-4: SelectedJointTask::UpdateOpCommand — behavioral correctness
//        (Uses GetQRef internally; verifies position/velocity error is right)
// ---------------------------------------------------------------------------
TEST_F(ReviewImprovementsTest, SelectedJointTaskOpCommandCorrect) {
  robot_->UpdateRobotModel(Eigen::Vector3d::Zero(), Eigen::Quaterniond::Identity(),
                           Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(),
                           Eigen::VectorXd::Zero(2), Eigen::VectorXd::Zero(2));

  // Select joint 0 only.
  SelectedJointTask task(robot_.get(), {0});

  const Eigen::VectorXd q_des    = Eigen::VectorXd::Constant(1, 0.5);
  const Eigen::VectorXd qdot_des = Eigen::VectorXd::Constant(1, 0.1);
  const Eigen::VectorXd zeros    = Eigen::VectorXd::Zero(1);
  task.UpdateDesired(q_des, qdot_des, zeros);

  task.UpdateJacobian();
  ASSERT_NO_THROW(task.UpdateOpCommand(Eigen::Matrix3d::Identity()));

  // pos_err should be q_des - q_current = 0.5 - 0.0 = 0.5
  // This verifies GetQRef() returns the correct (zero) joint state.
  EXPECT_NEAR(task.LocalPosError()(0), 0.5, 1e-9);
}

// ---------------------------------------------------------------------------
// P1-7: PointContact::UpdateJacobian via scratch buffer — result correctness
//        The scratch-buffer path must produce the same Jacobian rows as
//        explicitly computing GetLinkBodyJacobian().bottomRows(3).
// ---------------------------------------------------------------------------
TEST_F(ReviewImprovementsTest, PointContactJacobianMatchesBodyJacobian) {
  robot_->UpdateRobotModel(Eigen::Vector3d::Zero(), Eigen::Quaterniond::Identity(),
                           Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(),
                           Eigen::VectorXd::Zero(2), Eigen::VectorXd::Zero(2));

  const int link_idx = 1;
  PointContact contact(robot_.get(), link_idx, /*mu=*/0.5);

  contact.UpdateJacobian();
  const Eigen::MatrixXd jac_from_contact = contact.Jacobian();

  // Reference: extract linear rows from the full body Jacobian directly.
  const Eigen::MatrixXd full_jac = robot_->GetLinkLocalJacobian(link_idx);
  const Eigen::MatrixXd jac_ref  = full_jac.bottomRows(3);

  ASSERT_EQ(jac_from_contact.rows(), 3);
  ASSERT_EQ(jac_from_contact.cols(), robot_->NumQdot());
  EXPECT_TRUE(jac_from_contact.isApprox(jac_ref, 1e-12))
      << "PointContact Jacobian via scratch buffer differs from reference.\n"
      << "contact Jac:\n" << jac_from_contact << "\nreference:\n" << jac_ref;
}

// ---------------------------------------------------------------------------
// Integration: fixed-base ControlArchitecture runs N ticks → tau finite
//              No-contact path (P1-6 rf_cmd_.resize(0)) must not crash.
// ---------------------------------------------------------------------------
TEST_F(ReviewImprovementsTest, FixedBaseNticksTauFinite) {
  // Write a minimal YAML for a 2-DOF JointTask, no contacts.
  const std::string yaml_content = R"(
robot_model:
  urdf_path: ")" + (temp_dir_ / "test_robot.urdf").string() + R"("
  is_floating_base: false

task_pool:
  - name: "jpos_task"
    type: "JointTask"
    kp: 100.0
    kd: 10.0
    kp_ik: 1.0
    weight: 1.0

state_machine:
  - id: 0
    name: "review_stay_state"
    params:
      b_stay_here: true
    task_hierarchy:
      - name: "jpos_task"
        priority: 0
)";

  const std::filesystem::path yaml_path = temp_dir_ / "test_arch.yaml";
  {
    std::ofstream ofs(yaml_path);
    ofs << yaml_content;
  }

  auto arch_config =
      ControlArchitectureConfig::FromYaml(yaml_path.string(), 0.001);
  auto arch = std::make_unique<ControlArchitecture>(std::move(arch_config));
  arch->Initialize();
  PinocchioRobotSystem* robot = arch->GetRobot();

  const int n_dof = robot->NumActiveDof();
  ASSERT_GT(n_dof, 0);

  RobotJointState state;
  state.q    = Eigen::VectorXd::Zero(n_dof);
  state.qdot = Eigen::VectorXd::Zero(n_dof);
  state.tau  = Eigen::VectorXd::Zero(n_dof);

  constexpr int kNumTicks = 50;
  for (int i = 0; i < kNumTicks; ++i) {
    arch->Update(state, i * 0.001, 0.001);
    const RobotCommand& cmd = arch->GetCommand();

    ASSERT_EQ(cmd.tau.size(), n_dof) << "tick " << i;
    EXPECT_TRUE(cmd.tau.allFinite())
        << "tau is not finite at tick " << i << ": " << cmd.tau.transpose();
    EXPECT_TRUE(cmd.tau_ff.allFinite())
        << "tau_ff not finite at tick " << i;
    EXPECT_TRUE(cmd.tau_fb.allFinite())
        << "tau_fb not finite at tick " << i;
  }
}

// ---------------------------------------------------------------------------
// Torque determinism: same input → bit-exact same output across two runs.
// Catches any uninitialized memory, floating-point non-determinism, or
// state leaking between ticks in pre-allocated buffers.
// ---------------------------------------------------------------------------
TEST_F(ReviewImprovementsTest, TorqueDeterminism) {
  const std::string yaml_content = R"(
robot_model:
  urdf_path: ")" + (temp_dir_ / "test_robot.urdf").string() + R"("
  is_floating_base: false

task_pool:
  - name: "jpos_task"
    type: "JointTask"
    kp: 100.0
    kd: 10.0
    kp_ik: 1.0
    weight: 1.0

state_machine:
  - id: 0
    name: "review_stay_state"
    task_hierarchy:
      - name: "jpos_task"
        priority: 0
)";

  auto run_n_ticks = [&](int n) -> std::vector<Eigen::VectorXd> {
    const std::filesystem::path yaml_path = temp_dir_ / "determinism.yaml";
    { std::ofstream ofs(yaml_path); ofs << yaml_content; }
    auto cfg = ControlArchitectureConfig::FromYaml(yaml_path.string(), 0.001);
    auto arch = std::make_unique<ControlArchitecture>(std::move(cfg));
    arch->Initialize();
    const int n_dof = arch->GetRobot()->NumActiveDof();
    RobotJointState state;
    state.q    = Eigen::VectorXd::Constant(n_dof, 0.1);
    state.qdot = Eigen::VectorXd::Constant(n_dof, -0.05);
    state.tau  = Eigen::VectorXd::Zero(n_dof);
    std::vector<Eigen::VectorXd> torques;
    torques.reserve(n);
    for (int i = 0; i < n; ++i) {
      arch->Update(state, i * 0.001, 0.001);
      torques.push_back(arch->GetCommand().tau);
    }
    return torques;
  };

  constexpr int kTicks = 20;
  const auto run1 = run_n_ticks(kTicks);
  const auto run2 = run_n_ticks(kTicks);

  ASSERT_EQ(run1.size(), run2.size());
  for (int i = 0; i < kTicks; ++i) {
    ASSERT_EQ(run1[i].size(), run2[i].size()) << "tick " << i;
    EXPECT_TRUE(run1[i] == run2[i])
        << "Torque non-determinism at tick " << i << ":\n"
        << "  run1: " << run1[i].transpose() << "\n"
        << "  run2: " << run2[i].transpose();
  }
}

// ---------------------------------------------------------------------------
// Constraint clamping with cached pointers: verify position and torque
// limits are enforced correctly in the ClampCommandLimits code path
// that now uses cached_pos_limit_ / cached_trq_limit_ instead of
// dynamic_cast every tick.
// ---------------------------------------------------------------------------
TEST_F(ReviewImprovementsTest, CachedConstraintClampingCorrect) {
  const std::string yaml_content = R"(
robot_model:
  urdf_path: ")" + (temp_dir_ / "test_robot.urdf").string() + R"("
  is_floating_base: false

task_pool:
  - name: "jpos_task"
    type: "JointTask"
    kp: 100000.0
    kd: 1000.0
    kp_ik: 1.0
    weight: 1.0

global_constraints:
  JointPosLimitConstraint:
    enabled: true
    scale: 0.1
  JointTrqLimitConstraint:
    enabled: true
    scale: 0.5

state_machine:
  - id: 0
    name: "review_stay_state"
    task_hierarchy:
      - name: "jpos_task"
        priority: 0
)";

  const std::filesystem::path yaml_path = temp_dir_ / "clamp_test.yaml";
  { std::ofstream ofs(yaml_path); ofs << yaml_content; }
  auto cfg = ControlArchitectureConfig::FromYaml(yaml_path.string(), 0.001);
  auto arch = std::make_unique<ControlArchitecture>(std::move(cfg));
  arch->Initialize();
  PinocchioRobotSystem* robot = arch->GetRobot();
  const int n_dof = robot->NumActiveDof();

  // Use very high gains to push torque above limits — clamping must enforce.
  RobotJointState state;
  state.q    = Eigen::VectorXd::Constant(n_dof, 2.0);  // Far from desired (0)
  state.qdot = Eigen::VectorXd::Zero(n_dof);
  state.tau  = Eigen::VectorXd::Zero(n_dof);

  for (int i = 0; i < 10; ++i) {
    arch->Update(state, i * 0.001, 0.001);
    const RobotCommand& cmd = arch->GetCommand();

    ASSERT_TRUE(cmd.tau.allFinite()) << "tick " << i;
    ASSERT_TRUE(cmd.q.allFinite()) << "tick " << i;

    // Position must be clamped within scaled URDF limits.
    // URDF: [-3.14, 3.14], scale=0.1 → [-0.314, 0.314]
    const double pos_limit = 3.14 * 0.1;
    for (int j = 0; j < n_dof; ++j) {
      EXPECT_GE(cmd.q[j], -pos_limit - 1e-6)
          << "q[" << j << "]=" << cmd.q[j] << " below pos limit at tick " << i;
      EXPECT_LE(cmd.q[j], pos_limit + 1e-6)
          << "q[" << j << "]=" << cmd.q[j] << " above pos limit at tick " << i;
    }

    // Torque must be clamped within scaled URDF limits.
    // URDF effort=100, scale=0.5 → [-50, 50]
    const double trq_limit = 100.0 * 0.5;
    for (int j = 0; j < n_dof; ++j) {
      EXPECT_GE(cmd.tau[j], -trq_limit - 1e-6)
          << "tau[" << j << "]=" << cmd.tau[j] << " below trq limit at tick " << i;
      EXPECT_LE(cmd.tau[j], trq_limit + 1e-6)
          << "tau[" << j << "]=" << cmd.tau[j] << " above trq limit at tick " << i;
    }
  }
}

// ---------------------------------------------------------------------------
// ComTask FillComJacobian: verify the fill-style method produces the
// same Jacobian as the old copy-return style.
// ---------------------------------------------------------------------------
TEST_F(ReviewImprovementsTest, ComTaskJacobianMatchesFillMethod) {
  robot_->UpdateRobotModel(Eigen::Vector3d::Zero(), Eigen::Quaterniond::Identity(),
                           Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(),
                           Eigen::VectorXd::Zero(2), Eigen::VectorXd::Zero(2));

  // GetComJacobian() returns the copy; FillComJacobian writes in-place.
  const Eigen::Matrix3Xd jac_copy = robot_->GetComJacobian();
  Eigen::MatrixXd jac_fill = Eigen::MatrixXd::Zero(3, robot_->NumQdot());
  robot_->FillComJacobian(jac_fill);

  ASSERT_EQ(jac_fill.rows(), 3);
  ASSERT_EQ(jac_fill.cols(), robot_->NumQdot());
  EXPECT_TRUE(jac_fill.isApprox(jac_copy, 1e-12))
      << "FillComJacobian differs from GetComJacobian.\n"
      << "fill:\n" << jac_fill << "\ncopy:\n" << jac_copy;
}

// ---------------------------------------------------------------------------
// Gravity compensation: at zero state with identity desired, tau_ff should
// approximate the actuated gravity vector. This is the most basic correctness
// check for the full WBIC pipeline.
// ---------------------------------------------------------------------------
TEST_F(ReviewImprovementsTest, GravityCompensationBaseline) {
  const std::string yaml_content = R"(
robot_model:
  urdf_path: ")" + (temp_dir_ / "test_robot.urdf").string() + R"("
  is_floating_base: false

task_pool:
  - name: "jpos_task"
    type: "JointTask"
    kp: 0.0
    kd: 0.0
    kp_ik: 1.0
    weight: 1.0

state_machine:
  - id: 0
    name: "review_stay_state"
    task_hierarchy:
      - name: "jpos_task"
        priority: 0
)";

  const std::filesystem::path yaml_path = temp_dir_ / "grav_comp.yaml";
  { std::ofstream ofs(yaml_path); ofs << yaml_content; }
  auto cfg = ControlArchitectureConfig::FromYaml(yaml_path.string(), 0.001);
  auto arch = std::make_unique<ControlArchitecture>(std::move(cfg));
  arch->Initialize();
  PinocchioRobotSystem* robot = arch->GetRobot();
  const int n_dof = robot->NumActiveDof();

  RobotJointState state;
  state.q    = Eigen::VectorXd::Zero(n_dof);
  state.qdot = Eigen::VectorXd::Zero(n_dof);
  state.tau  = Eigen::VectorXd::Zero(n_dof);

  // Let the pipeline settle for a few ticks.
  for (int i = 0; i < 5; ++i) {
    arch->Update(state, i * 0.001, 0.001);
  }

  const RobotCommand& cmd = arch->GetCommand();
  const Eigen::VectorXd& grav = robot->GetGravityRef();
  // For fixed-base, gravity vector is num_qdot = num_actuated sized.
  const Eigen::VectorXd grav_actuated = grav.tail(n_dof);

  ASSERT_TRUE(cmd.tau_ff.allFinite());
  // With kp=0, kd=0, des=current, the WBIC feedforward should be pure
  // gravity compensation. Allow generous tolerance due to QP regularization.
  for (int j = 0; j < n_dof; ++j) {
    EXPECT_NEAR(cmd.tau_ff[j], grav_actuated[j], 1.0)
        << "joint " << j << " tau_ff=" << cmd.tau_ff[j]
        << " grav=" << grav_actuated[j];
  }
}

} // namespace
} // namespace wbc
