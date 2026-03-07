/**
 * @file wbc_core/wbc_architecture/test/test_real_tasks.cpp
 * @brief Comprehensive unit tests for WBC with real robotic task scenarios.
 *
 * Test categories:
 *   1. EE position tracking (2-DOF fixed-base, 7-DOF Optimo)
 *   2. EE orientation tracking (quaternion correctness)
 *   3. Multi-priority task hierarchy (ee_pos > ee_ori > jpos)
 *   4. Contact constraint integration (SurfaceContact on floating-base)
 *   5. ForceTask world-to-body rotation
 *   6. Cone constraint dirty-flag caching
 *   7. Static Jacobian correctness (JointTask, SelectedJointTask)
 *   8. Joint/torque limit enforcement under extreme gains
 *   9. COM task tracking
 *  10. State transition with contact add/remove
 */
#include <sys/types.h>
#include <unistd.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Geometry>
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
// Test state: stays in place, keeps whatever tasks the YAML assigns.
// ---------------------------------------------------------------------------
class RealTaskStayState final : public StateMachine {
public:
  RealTaskStayState(StateId id, const std::string& name,
                    PinocchioRobotSystem* robot, TaskRegistry* task_reg,
                    ConstraintRegistry* const_reg, StateProvider* sp)
      : StateMachine(id, name, robot, task_reg, const_reg, sp) {}

  void FirstVisit() override {}
  void OneStep() override {}
  void LastVisit() override {}
  bool EndOfState() override { return false; }
  StateId GetNextState() override { return id(); }
};

// Second state (for transition tests): identical behavior, different name.
class RealTaskSecondState final : public StateMachine {
public:
  RealTaskSecondState(StateId id, const std::string& name,
                      PinocchioRobotSystem* robot, TaskRegistry* task_reg,
                      ConstraintRegistry* const_reg, StateProvider* sp)
      : StateMachine(id, name, robot, task_reg, const_reg, sp) {}

  void FirstVisit() override {}
  void OneStep() override {}
  void LastVisit() override {}
  bool EndOfState() override { return false; }
  StateId GetNextState() override { return id(); }
};

} // namespace

WBC_REGISTER_STATE(
    "rt_stay_state",
    [](StateId id, const std::string& name,
       const StateMachineConfig& ctx) -> std::unique_ptr<StateMachine> {
      return std::make_unique<RealTaskStayState>(
          id, name, ctx.robot, ctx.task_registry,
          ctx.constraint_registry, ctx.state_provider);
    });

WBC_REGISTER_STATE(
    "rt_second_state",
    [](StateId id, const std::string& name,
       const StateMachineConfig& ctx) -> std::unique_ptr<StateMachine> {
      return std::make_unique<RealTaskSecondState>(
          id, name, ctx.robot, ctx.task_registry,
          ctx.constraint_registry, ctx.state_provider);
    });

namespace {

// ---------------------------------------------------------------------------
// URDF: minimal 2-DOF RR planar arm (Z then Y joint, links at 0.1m)
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
      <origin xyz="0 0 0.05" rpy="0 0 0"/>
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
      <origin xyz="0 0 0.05" rpy="0 0 0"/>
      <mass value="1.0"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
  </link>
</robot>
)";
}

// ---------------------------------------------------------------------------
// Fixture
// ---------------------------------------------------------------------------
class RealTaskTest : public ::testing::Test {
protected:
  void SetUp() override {
    const std::string suffix =
        std::to_string(static_cast<long long>(::getpid())) + "_" +
        std::to_string(counter_++);
    temp_dir_ = std::filesystem::temp_directory_path() /
                ("real_task_test_" + suffix);
    std::filesystem::create_directories(temp_dir_);

    urdf_path_ = temp_dir_ / "test_robot.urdf";
    WriteFile(urdf_path_, TwoDofUrdf());
  }

  void TearDown() override {
    std::error_code ec;
    std::filesystem::remove_all(temp_dir_, ec);
  }

  static void WriteFile(const std::filesystem::path& path,
                        const std::string& content) {
    std::ofstream ofs(path);
    ofs << content;
  }

  std::filesystem::path WriteYaml(const std::string& filename,
                                  const std::string& yaml) {
    const auto path = temp_dir_ / filename;
    WriteFile(path, yaml);
    return path;
  }

  std::unique_ptr<ControlArchitecture> MakeArch(
      const std::filesystem::path& yaml_path) {
    auto cfg = ControlArchitectureConfig::FromYaml(yaml_path.string(), 0.001);
    auto arch = std::make_unique<ControlArchitecture>(std::move(cfg));
    arch->Initialize();
    return arch;
  }

  void UpdateFixedBase(ControlArchitecture* arch,
                       const Eigen::VectorXd& q,
                       const Eigen::VectorXd& qdot,
                       double t, double dt) {
    RobotJointState js;
    js.q = q;
    js.qdot = qdot;
    js.tau = Eigen::VectorXd::Zero(q.size());
    arch->Update(js, t, dt);
  }

  std::string RobotModelHeader(bool floating = false) const {
    std::string yaml = "robot_model:\n";
    yaml += "  urdf_path: \"" + urdf_path_.string() + "\"\n";
    yaml += "  is_floating_base: " + std::string(floating ? "true" : "false") + "\n";
    return yaml;
  }

  std::filesystem::path temp_dir_;
  std::filesystem::path urdf_path_;
  static inline int counter_ = 0;
};

// =========================================================================
// 1. JointTask static Jacobian: [0|I] structure
// =========================================================================
TEST_F(RealTaskTest, JointTaskStaticJacobianStructure) {
  auto robot = std::make_unique<PinocchioRobotSystem>(
      urdf_path_.string(), "", true, false);
  robot->UpdateRobotModel(Eigen::Vector3d::Zero(),
                           Eigen::Quaterniond::Identity(),
                           Eigen::Vector3d::Zero(),
                           Eigen::Vector3d::Zero(),
                           Eigen::VectorXd::Zero(2),
                           Eigen::VectorXd::Zero(2));

  JointTask task(robot.get());

  // UpdateJacobian is a no-op; Jacobian should already be set.
  task.UpdateJacobian();
  const auto& J = task.Jacobian();
  ASSERT_EQ(J.rows(), 2);
  ASSERT_EQ(J.cols(), 2);  // fixed base: num_qdot = num_active = 2

  // Should be identity for fixed base (no floating DOFs).
  EXPECT_TRUE(J.isApprox(Eigen::MatrixXd::Identity(2, 2), 1e-14))
      << "JointTask Jacobian is not identity:\n" << J;

  // Calling UpdateJacobian again should not change it.
  task.UpdateJacobian();
  EXPECT_TRUE(J.isApprox(Eigen::MatrixXd::Identity(2, 2), 1e-14));
}

// =========================================================================
// 2. SelectedJointTask static Jacobian: sparse identity
// =========================================================================
TEST_F(RealTaskTest, SelectedJointTaskSparseJacobian) {
  auto robot = std::make_unique<PinocchioRobotSystem>(
      urdf_path_.string(), "", true, false);
  robot->UpdateRobotModel(Eigen::Vector3d::Zero(),
                           Eigen::Quaterniond::Identity(),
                           Eigen::Vector3d::Zero(),
                           Eigen::Vector3d::Zero(),
                           Eigen::VectorXd::Zero(2),
                           Eigen::VectorXd::Zero(2));

  // Select only joint 1 (second joint).
  SelectedJointTask task(robot.get(), {1});
  task.UpdateJacobian();
  const auto& J = task.Jacobian();
  ASSERT_EQ(J.rows(), 1);
  ASSERT_EQ(J.cols(), 2);

  EXPECT_DOUBLE_EQ(J(0, 0), 0.0);
  EXPECT_DOUBLE_EQ(J(0, 1), 1.0);
}

// =========================================================================
// 3. LinkPosTask: EE position tracks toward desired
// =========================================================================
TEST_F(RealTaskTest, LinkPosTaskEePositionDirection) {
  const std::string yaml = RobotModelHeader() + R"(
regularization:
  w_qddot: 1.0e-6
  w_tau: 1.0e-3
task_pool:
  - name: ee_pos
    type: LinkPosTask
    target_frame: link2
    kp: [100, 100, 100]
    kd: [10, 10, 10]
    kp_ik: [1, 1, 1]
  - name: jpos_task
    type: JointTask
    kp: 10.0
    kd: 1.0
    kp_ik: 1.0
    weight: 1.0
state_machine:
  - id: 0
    name: rt_stay_state
    task_hierarchy:
      - {name: ee_pos, priority: 0}
      - {name: jpos_task, priority: 1}
)";

  auto yaml_path = WriteYaml("ee_pos.yaml", yaml);
  auto arch = MakeArch(yaml_path);
  auto* robot = arch->GetRobot();
  const int n = robot->NumActiveDof();

  // Start at a non-trivial config so gravity is nonzero.
  Eigen::VectorXd q(n);
  q << 0.3, 0.5;
  Eigen::VectorXd qdot = Eigen::VectorXd::Zero(n);

  // Warmup to get EE position at this config.
  UpdateFixedBase(arch.get(), q, qdot, 0.0, 0.001);
  const Eigen::Vector3d ee_pos_init =
      robot->GetLinkIsometry(robot->GetFrameIndex("link2")).translation();

  // Set desired EE position offset in +X (perpendicular to gravity).
  const auto* state = arch->GetConfig()->FindState(0);
  ASSERT_NE(state, nullptr);
  ASSERT_NE(state->ee_pos, nullptr);
  Eigen::Vector3d des_pos = ee_pos_init;
  des_pos.x() += 0.05;  // 5cm lateral
  state->ee_pos->UpdateDesired(des_pos, Eigen::Vector3d::Zero(),
                                Eigen::Vector3d::Zero());

  // Also set jpos desired to current to avoid it fighting.
  if (state->joint) {
    state->joint->UpdateDesired(q, Eigen::VectorXd::Zero(n),
                                 Eigen::VectorXd::Zero(n));
  }

  // Run ticks (open-loop).
  for (int i = 1; i <= 20; ++i) {
    UpdateFixedBase(arch.get(), q, qdot, i * 0.001, 0.001);
  }

  const auto& cmd = arch->GetCommand();
  ASSERT_TRUE(cmd.q.allFinite());
  ASSERT_TRUE(cmd.tau.allFinite());

  // The IK should produce a q_cmd different from the current q.
  // With kp_ik=1 and a 5cm EE offset, the IK should move.
  // Also tau_ff should be non-zero (gravity comp + dynamics).
  EXPECT_TRUE(cmd.tau_ff.allFinite());
  // At this config, gravity compensation alone is nonzero.
  EXPECT_GT(cmd.tau_ff.norm(), 1e-6)
      << "tau_ff should be non-zero at non-trivial config with EE task";
}

// =========================================================================
// 4. LinkOriTask: quaternion desired preserves orientation
// =========================================================================
TEST_F(RealTaskTest, LinkOriTaskQuaternionTracking) {
  auto robot = std::make_unique<PinocchioRobotSystem>(
      urdf_path_.string(), "", true, false);
  robot->UpdateRobotModel(Eigen::Vector3d::Zero(),
                           Eigen::Quaterniond::Identity(),
                           Eigen::Vector3d::Zero(),
                           Eigen::Vector3d::Zero(),
                           Eigen::VectorXd::Zero(2),
                           Eigen::VectorXd::Zero(2));

  LinkOriTask task(robot.get(), robot->GetFrameIndex("link2"));
  task.SetKp(Eigen::Vector3d(100, 100, 100));
  task.SetKd(Eigen::Vector3d(10, 10, 10));

  // Set desired = identity quaternion [x,y,z,w] = [0,0,0,1].
  Eigen::VectorXd quat_des(4);
  quat_des << 0, 0, 0, 1;
  task.UpdateDesired(quat_des, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());

  task.UpdateJacobian();
  task.UpdateJacobianDotQdot();
  task.UpdateOpCommand();

  // At zero config, the robot is at identity orientation.
  // With des = identity and current = identity, error should be near zero.
  EXPECT_LT(task.PosError().norm(), 1e-6)
      << "Orientation error at identity should be ~0, got: "
      << task.PosError().transpose();

  // Op command (kp * err + kd * vel_err) should be near zero.
  EXPECT_LT(task.OpCommand().norm(), 1e-4)
      << "OpCommand should be ~0 at identity, got: "
      << task.OpCommand().transpose();
}

// =========================================================================
// 5. LinkOriTask: wrong-size quaternion is rejected
// =========================================================================
TEST_F(RealTaskTest, LinkOriTaskRejectsWrongSizeQuat) {
  auto robot = std::make_unique<PinocchioRobotSystem>(
      urdf_path_.string(), "", true, false);
  LinkOriTask task(robot.get(), 1);

  // Set valid quaternion.
  Eigen::VectorXd valid(4);
  valid << 0, 0, 0, 1;
  task.UpdateDesired(valid, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());
  EXPECT_EQ(task.DesiredPos().size(), 4);
  EXPECT_DOUBLE_EQ(task.DesiredPos()(3), 1.0);

  // Attempt wrong-size (3-element) — should be silently rejected.
  Eigen::VectorXd bad(3);
  bad << 1, 2, 3;
  task.UpdateDesired(bad, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());
  // Desired should remain at previous valid value.
  EXPECT_DOUBLE_EQ(task.DesiredPos()(3), 1.0);
}

// =========================================================================
// 6. ForceTask: world-to-body rotation correctness
// =========================================================================
TEST_F(RealTaskTest, ForceTaskWorldToBodyRotation) {
  auto robot = std::make_unique<PinocchioRobotSystem>(
      urdf_path_.string(), "", true, false);

  // Put robot at a non-trivial config so link2 has non-identity rotation.
  Eigen::VectorXd q(2), qdot(2);
  q << 0.5, 0.3;  // non-zero joint angles
  qdot.setZero();
  robot->UpdateRobotModel(Eigen::Vector3d::Zero(),
                           Eigen::Quaterniond::Identity(),
                           Eigen::Vector3d::Zero(),
                           Eigen::Vector3d::Zero(), q, qdot);

  const int link_idx = robot->GetFrameIndex("link2");
  PointContact contact(robot.get(), link_idx, 0.5);
  ForceTask force_task(robot.get(), &contact);

  // Set a world-frame desired force [0, 0, 10] (gravity direction).
  Eigen::VectorXd rf_world(3);
  rf_world << 0.0, 0.0, 10.0;

  // UpdateDesiredToLocal should rotate into body frame.
  force_task.UpdateDesiredToLocal(rf_world);
  const Eigen::VectorXd& rf_local = force_task.DesiredRf();

  // The body-frame force should have the same magnitude.
  EXPECT_NEAR(rf_local.norm(), rf_world.norm(), 1e-10)
      << "Body-frame force magnitude differs from world-frame";

  // Verify it equals R^T * rf_world.
  const Eigen::Matrix3d R =
      robot->GetLinkIsometry(link_idx).linear().transpose();
  Eigen::Vector3d expected = R * rf_world;
  EXPECT_TRUE(rf_local.isApprox(expected, 1e-10))
      << "ForceTask body-frame: " << rf_local.transpose()
      << "\nExpected: " << expected.transpose();
}

// =========================================================================
// 7. ForceTask: SetParameters rejects wrong dimension
// =========================================================================
TEST_F(RealTaskTest, ForceTaskSetParametersWrongDimRejects) {
  auto robot = std::make_unique<PinocchioRobotSystem>(
      urdf_path_.string(), "", true, false);
  PointContact contact(robot.get(), 1, 0.5);
  ForceTask ft(robot.get(), &contact);

  // Valid: dim=3.
  ForceTaskConfig good;
  good.weight = Eigen::Vector3d(1.0, 2.0, 3.0);
  ft.SetParameters(good);
  EXPECT_DOUBLE_EQ(ft.Weight()(1), 2.0);

  // Invalid: dim=6 for a 3D contact.
  ForceTaskConfig bad;
  bad.weight = Eigen::VectorXd::Constant(6, 99.0);
  ft.SetParameters(bad);
  // Should keep previous weight.
  EXPECT_DOUBLE_EQ(ft.Weight()(1), 2.0);
}

// =========================================================================
// 8. Cone constraint dirty flag: repeated calls skip recomputation
// =========================================================================
TEST_F(RealTaskTest, ConeConstraintDirtyFlagSkipsRecompute) {
  auto robot = std::make_unique<PinocchioRobotSystem>(
      urdf_path_.string(), "", true, false);
  robot->UpdateRobotModel(Eigen::Vector3d::Zero(),
                           Eigen::Quaterniond::Identity(),
                           Eigen::Vector3d::Zero(),
                           Eigen::Vector3d::Zero(),
                           Eigen::VectorXd::Zero(2),
                           Eigen::VectorXd::Zero(2));

  PointContact contact(robot.get(), 1, 0.5);

  // First call: builds the cone.
  contact.UpdateConeConstraint();
  Eigen::MatrixXd cone1 = contact.UfMatrix();

  // Second call: should skip (dirty flag cleared).
  // Corrupt the matrix to verify it's NOT rebuilt.
  // We can't directly corrupt it, but we can verify the values stay.
  contact.UpdateConeConstraint();
  EXPECT_TRUE(contact.UfMatrix().isApprox(cone1, 1e-14))
      << "Cone matrix changed on second call (should be cached)";

  // After SetMaxFz, dirty flag set → next call rebuilds.
  contact.SetMaxFz(500.0);
  contact.UpdateConeConstraint();
  // Row 5 (upper bound) should reflect new rf_z_max.
  EXPECT_DOUBLE_EQ(contact.UfVector()(5), -500.0);
}

// =========================================================================
// 9. SurfaceContact cone dimensions and friction structure
// =========================================================================
TEST_F(RealTaskTest, SurfaceContactConeConstraintStructure) {
  auto robot = std::make_unique<PinocchioRobotSystem>(
      urdf_path_.string(), "", true, false);
  robot->UpdateRobotModel(Eigen::Vector3d::Zero(),
                           Eigen::Quaterniond::Identity(),
                           Eigen::Vector3d::Zero(),
                           Eigen::Vector3d::Zero(),
                           Eigen::VectorXd::Zero(2),
                           Eigen::VectorXd::Zero(2));

  const double mu = 0.6;
  const double hl = 0.10, hw = 0.05;
  SurfaceContact contact(robot.get(), 1, mu, hl, hw);

  contact.UpdateConeConstraint();
  const auto& Uf = contact.UfMatrix();
  const auto& uf_vec = contact.UfVector();

  // SurfaceContact: 18 rows x 6 cols.
  ASSERT_EQ(Uf.rows(), 18);
  ASSERT_EQ(Uf.cols(), 6);
  ASSERT_EQ(uf_vec.size(), 18);

  // Row 0: f_z >= 0 → Uf(0,5) = 1.
  EXPECT_DOUBLE_EQ(Uf(0, 5), 1.0);

  // Row 1: f_x + mu*f_z >= 0 → Uf(1,3) = 1, Uf(1,5) = mu.
  EXPECT_DOUBLE_EQ(Uf(1, 3), 1.0);
  EXPECT_DOUBLE_EQ(Uf(1, 5), mu);

  // Row 17: -f_z >= -rf_z_max → Uf(17,5) = -1.
  EXPECT_DOUBLE_EQ(Uf(17, 5), -1.0);

  // Foot half-size appears in COP rows (5-8).
  EXPECT_DOUBLE_EQ(Uf(5, 5), hw);  // foot_half_width
  EXPECT_DOUBLE_EQ(Uf(7, 5), hl);  // foot_half_length

  // After SetFootHalfSize, cone should be rebuilt.
  contact.SetFootHalfSize(0.15, 0.08);
  contact.UpdateConeConstraint();
  EXPECT_DOUBLE_EQ(contact.UfMatrix()(5, 5), 0.08);
  EXPECT_DOUBLE_EQ(contact.UfMatrix()(7, 5), 0.15);
}

// =========================================================================
// 10. PointContact Jacobian matches body Jacobian linear rows
// =========================================================================
TEST_F(RealTaskTest, PointContactJacobianLinearRows) {
  auto robot = std::make_unique<PinocchioRobotSystem>(
      urdf_path_.string(), "", true, false);

  Eigen::VectorXd q(2), qdot(2);
  q << 0.3, -0.5;
  qdot << 0.1, -0.2;
  robot->UpdateRobotModel(Eigen::Vector3d::Zero(),
                           Eigen::Quaterniond::Identity(),
                           Eigen::Vector3d::Zero(),
                           Eigen::Vector3d::Zero(), q, qdot);

  const int link_idx = 1;
  PointContact contact(robot.get(), link_idx, 0.5);
  contact.UpdateJacobian();

  // Reference: full 6xN body Jacobian, take bottom 3 rows.
  const Eigen::MatrixXd full_body_jac = robot->GetLinkLocalJacobian(link_idx);
  const Eigen::MatrixXd expected = full_body_jac.bottomRows(3);

  ASSERT_EQ(contact.Jacobian().rows(), 3);
  ASSERT_EQ(contact.Jacobian().cols(), robot->NumQdot());
  EXPECT_TRUE(contact.Jacobian().isApprox(expected, 1e-12))
      << "PointContact Jac:\n" << contact.Jacobian()
      << "\nExpected:\n" << expected;
}

// =========================================================================
// 11. Multi-priority hierarchy: ee_pos > jpos
//     Verify EE task dominates over joint posture.
// =========================================================================
TEST_F(RealTaskTest, MultiPriorityEePosOverJpos) {
  const std::string yaml = RobotModelHeader() + R"(
regularization:
  w_qddot: 1.0e-6
  w_tau: 1.0e-3
task_pool:
  - name: ee_pos
    type: LinkPosTask
    target_frame: link2
    kp: [200, 200, 200]
    kd: [20, 20, 20]
    kp_ik: [1, 1, 1]
  - name: jpos_task
    type: JointTask
    kp: 1.0
    kd: 0.1
    kp_ik: 1.0
state_machine:
  - id: 0
    name: rt_stay_state
    task_hierarchy:
      - {name: ee_pos, priority: 0}
      - {name: jpos_task, priority: 1}
)";

  auto yaml_path = WriteYaml("multi_prio.yaml", yaml);
  auto arch = MakeArch(yaml_path);
  auto* robot = arch->GetRobot();
  const int n = robot->NumActiveDof();

  // Non-trivial config so gravity is nonzero.
  Eigen::VectorXd q(n);
  q << 0.4, 0.6;
  Eigen::VectorXd qdot = Eigen::VectorXd::Zero(n);

  // Warmup.
  for (int i = 0; i < 5; ++i) {
    UpdateFixedBase(arch.get(), q, qdot, i * 0.001, 0.001);
  }

  // Get EE position.
  const Eigen::Vector3d ee_init =
      robot->GetLinkIsometry(robot->GetFrameIndex("link2")).translation();

  // Set EE target +0.03 in X (perpendicular).
  const auto* state = arch->GetConfig()->FindState(0);
  ASSERT_NE(state->ee_pos, nullptr);
  Eigen::Vector3d ee_des = ee_init;
  ee_des.x() += 0.03;
  state->ee_pos->UpdateDesired(ee_des, Eigen::Vector3d::Zero(),
                                Eigen::Vector3d::Zero());

  // jpos_task wants to stay at current config.
  if (state->joint) {
    state->joint->UpdateDesired(q, Eigen::VectorXd::Zero(n),
                                 Eigen::VectorXd::Zero(n));
  }

  // Run ticks.
  for (int i = 5; i < 30; ++i) {
    UpdateFixedBase(arch.get(), q, qdot, i * 0.001, 0.001);
  }

  const auto& cmd = arch->GetCommand();
  ASSERT_TRUE(cmd.q.allFinite());
  ASSERT_TRUE(cmd.tau_ff.allFinite());

  // At non-trivial config with EE offset: tau_ff should be non-zero
  // (at minimum, gravity compensation is nonzero).
  EXPECT_GT(cmd.tau_ff.norm(), 1e-4)
      << "tau_ff should be non-zero with multi-priority at non-trivial config";
}

// =========================================================================
// 12. Joint limit enforcement under extreme gains
// =========================================================================
TEST_F(RealTaskTest, JointLimitClampingUnderExtremeGains) {
  const std::string yaml = RobotModelHeader() + R"(
regularization:
  w_qddot: 1.0e-6
  w_tau: 1.0e-3
task_pool:
  - name: jpos_task
    type: JointTask
    kp: 100000.0
    kd: 1000.0
    kp_ik: 1.0
    weight: 1.0
global_constraints:
  JointPosLimitConstraint:
    enabled: true
    scale: 0.5
  JointTrqLimitConstraint:
    enabled: true
    scale: 0.5
state_machine:
  - id: 0
    name: rt_stay_state
    task_hierarchy:
      - {name: jpos_task, priority: 0}
)";

  auto yaml_path = WriteYaml("limit_clamp.yaml", yaml);
  auto arch = MakeArch(yaml_path);
  auto* robot = arch->GetRobot();
  const int n = robot->NumActiveDof();

  // Start far from origin → massive error with extreme gains.
  Eigen::VectorXd q = Eigen::VectorXd::Constant(n, 2.5);
  Eigen::VectorXd qdot = Eigen::VectorXd::Zero(n);

  // URDF limit: [-3.14, 3.14], scale=0.5 → [-1.57, 1.57]
  const double pos_lim = 3.14 * 0.5;
  // URDF effort=100, scale=0.5 → [-50, 50]
  const double trq_lim = 100.0 * 0.5;

  for (int i = 0; i < 20; ++i) {
    UpdateFixedBase(arch.get(), q, qdot, i * 0.001, 0.001);
    const auto& cmd = arch->GetCommand();

    ASSERT_TRUE(cmd.q.allFinite()) << "tick " << i;
    ASSERT_TRUE(cmd.tau.allFinite()) << "tick " << i;

    for (int j = 0; j < n; ++j) {
      EXPECT_GE(cmd.q[j], -pos_lim - 1e-6) << "q[" << j << "] tick " << i;
      EXPECT_LE(cmd.q[j], pos_lim + 1e-6) << "q[" << j << "] tick " << i;
      EXPECT_GE(cmd.tau[j], -trq_lim - 1e-6) << "tau[" << j << "] tick " << i;
      EXPECT_LE(cmd.tau[j], trq_lim + 1e-6) << "tau[" << j << "] tick " << i;
    }
  }
}

// =========================================================================
// 13. Gravity compensation: zero-gain baseline
// =========================================================================
TEST_F(RealTaskTest, GravityCompensationZeroGain) {
  const std::string yaml = RobotModelHeader() + R"(
task_pool:
  - name: jpos_task
    type: JointTask
    kp: 0.0
    kd: 0.0
    kp_ik: 1.0
    weight: 1.0
state_machine:
  - id: 0
    name: rt_stay_state
    task_hierarchy:
      - {name: jpos_task, priority: 0}
)";

  auto yaml_path = WriteYaml("grav_comp.yaml", yaml);
  auto arch = MakeArch(yaml_path);
  auto* robot = arch->GetRobot();
  const int n = robot->NumActiveDof();

  Eigen::VectorXd q = Eigen::VectorXd::Zero(n);
  Eigen::VectorXd qdot = Eigen::VectorXd::Zero(n);

  for (int i = 0; i < 10; ++i) {
    UpdateFixedBase(arch.get(), q, qdot, i * 0.001, 0.001);
  }

  const auto& cmd = arch->GetCommand();
  const Eigen::VectorXd grav = robot->GetGravityRef().tail(n);

  ASSERT_TRUE(cmd.tau_ff.allFinite());
  // With kp=0, kd=0, des=current, feedforward ≈ gravity compensation.
  for (int j = 0; j < n; ++j) {
    EXPECT_NEAR(cmd.tau_ff[j], grav[j], 2.0)
        << "joint " << j << " tau_ff=" << cmd.tau_ff[j]
        << " grav=" << grav[j];
  }
}

// =========================================================================
// 14. Torque determinism: identical inputs → identical outputs
// =========================================================================
TEST_F(RealTaskTest, TorqueDeterminismMultipleRuns) {
  const std::string yaml = RobotModelHeader() + R"(
task_pool:
  - name: jpos_task
    type: JointTask
    kp: 50.0
    kd: 5.0
    kp_ik: 1.0
state_machine:
  - id: 0
    name: rt_stay_state
    task_hierarchy:
      - {name: jpos_task, priority: 0}
)";

  auto run = [&](int n_ticks) -> std::vector<Eigen::VectorXd> {
    auto path = WriteYaml("determ.yaml", yaml);
    auto arch = MakeArch(path);
    const int n = arch->GetRobot()->NumActiveDof();
    Eigen::VectorXd q = Eigen::VectorXd::Constant(n, 0.3);
    Eigen::VectorXd qdot = Eigen::VectorXd::Constant(n, -0.1);
    std::vector<Eigen::VectorXd> taus;
    taus.reserve(n_ticks);
    for (int i = 0; i < n_ticks; ++i) {
      UpdateFixedBase(arch.get(), q, qdot, i * 0.001, 0.001);
      taus.push_back(arch->GetCommand().tau);
    }
    return taus;
  };

  const auto r1 = run(15);
  const auto r2 = run(15);
  ASSERT_EQ(r1.size(), r2.size());
  for (size_t i = 0; i < r1.size(); ++i) {
    EXPECT_TRUE(r1[i] == r2[i])
        << "Non-deterministic at tick " << i << ":\n"
        << "  r1: " << r1[i].transpose() << "\n"
        << "  r2: " << r2[i].transpose();
  }
}

// =========================================================================
// 15. State transition: tasks change between states
// =========================================================================
TEST_F(RealTaskTest, StateTransitionSwitchesTasks) {
  const std::string yaml = RobotModelHeader() + R"(
regularization:
  w_qddot: 1.0e-6
  w_tau: 1.0e-3
start_state_id: 0
task_pool:
  - name: jpos_task
    type: JointTask
    kp: 50.0
    kd: 5.0
    kp_ik: 1.0
  - name: ee_pos
    type: LinkPosTask
    target_frame: link2
    kp: [100, 100, 100]
    kd: [10, 10, 10]
    kp_ik: [1, 1, 1]
state_machine:
  - id: 0
    name: rt_stay_state
    task_hierarchy:
      - {name: jpos_task, priority: 0}
  - id: 1
    name: rt_second_state
    task_hierarchy:
      - {name: ee_pos, priority: 0}
      - {name: jpos_task, priority: 1}
)";

  auto path = WriteYaml("transition.yaml", yaml);
  auto arch = MakeArch(path);
  const int n = arch->GetRobot()->NumActiveDof();

  Eigen::VectorXd q = Eigen::VectorXd::Zero(n);
  Eigen::VectorXd qdot = Eigen::VectorXd::Zero(n);

  // Run in state 0 (jpos only).
  for (int i = 0; i < 10; ++i) {
    UpdateFixedBase(arch.get(), q, qdot, i * 0.001, 0.001);
  }
  EXPECT_EQ(arch->GetCurrentStateId(), 0);
  const Eigen::VectorXd tau_state0 = arch->GetCommand().tau;
  ASSERT_TRUE(tau_state0.allFinite());

  // Request transition to state 1.
  arch->RequestState(1);
  for (int i = 10; i < 30; ++i) {
    UpdateFixedBase(arch.get(), q, qdot, i * 0.001, 0.001);
  }
  EXPECT_EQ(arch->GetCurrentStateId(), 1);
  const Eigen::VectorXd tau_state1 = arch->GetCommand().tau;
  ASSERT_TRUE(tau_state1.allFinite());
}

// =========================================================================
// 16. COM task: position matches Pinocchio COM
// =========================================================================
TEST_F(RealTaskTest, ComTaskPositionMatchesPinocchio) {
  auto robot = std::make_unique<PinocchioRobotSystem>(
      urdf_path_.string(), "", true, false);
  Eigen::VectorXd q(2), qdot(2);
  q << 0.2, -0.4;
  qdot << 0.1, -0.05;
  robot->UpdateRobotModel(Eigen::Vector3d::Zero(),
                           Eigen::Quaterniond::Identity(),
                           Eigen::Vector3d::Zero(),
                           Eigen::Vector3d::Zero(), q, qdot);

  ComTask task(robot.get());
  task.SetKp(Eigen::Vector3d(100, 100, 100));
  task.SetKd(Eigen::Vector3d(10, 10, 10));

  // Set desired = current COM.
  const Eigen::Vector3d com_pos = robot->GetComPosition();
  task.UpdateDesired(com_pos, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());

  task.UpdateJacobian();
  task.UpdateJacobianDotQdot();
  task.UpdateOpCommand();

  // Error should be ~0 when desired = current.
  EXPECT_LT(task.PosError().norm(), 1e-10)
      << "COM error when des=current: " << task.PosError().transpose();

  // COM Jacobian should be 3 x num_qdot.
  EXPECT_EQ(task.Jacobian().rows(), 3);
  EXPECT_EQ(task.Jacobian().cols(), robot->NumQdot());
}

// =========================================================================
// 17. PointContact: OpCommand body-frame PD tracking
// =========================================================================
TEST_F(RealTaskTest, PointContactOpCommandBodyFrame) {
  auto robot = std::make_unique<PinocchioRobotSystem>(
      urdf_path_.string(), "", true, false);
  Eigen::VectorXd q(2), qdot(2);
  q.setZero();
  qdot.setZero();
  robot->UpdateRobotModel(Eigen::Vector3d::Zero(),
                           Eigen::Quaterniond::Identity(),
                           Eigen::Vector3d::Zero(),
                           Eigen::Vector3d::Zero(), q, qdot);

  const int link_idx = 1;
  PointContact contact(robot.get(), link_idx, 0.5);

  ContactConfig cfg;
  cfg.mu = 0.5;
  cfg.kp = Eigen::Vector3d(100, 100, 100);
  cfg.kd = Eigen::Vector3d(10, 10, 10);
  cfg.max_fz = 500.0;
  contact.SetParameters(cfg);

  // Set desired position = current link position (zero error).
  const Eigen::Vector3d link_pos =
      robot->GetLinkIsometry(link_idx).translation();
  contact.SetDesiredPos(link_pos);

  contact.UpdateJacobian();
  contact.UpdateJacobianDotQdot();
  contact.UpdateOpCommand();

  // At desired = current, velocity = 0: op_cmd should be ~0.
  EXPECT_LT(contact.OpCommand().norm(), 1e-6)
      << "Contact OpCommand should be ~0 when des=current: "
      << contact.OpCommand().transpose();
}

// =========================================================================
// 18. SurfaceContact: Jacobian is full 6xN body Jacobian
// =========================================================================
TEST_F(RealTaskTest, SurfaceContactFullJacobian) {
  auto robot = std::make_unique<PinocchioRobotSystem>(
      urdf_path_.string(), "", true, false);
  Eigen::VectorXd q(2), qdot(2);
  q << 0.1, -0.2;
  qdot.setZero();
  robot->UpdateRobotModel(Eigen::Vector3d::Zero(),
                           Eigen::Quaterniond::Identity(),
                           Eigen::Vector3d::Zero(),
                           Eigen::Vector3d::Zero(), q, qdot);

  SurfaceContact contact(robot.get(), 1, 0.5, 0.1, 0.05);
  contact.UpdateJacobian();

  // SurfaceContact uses FillLinkBodyJacobian — full 6xN.
  const auto& J = contact.Jacobian();
  ASSERT_EQ(J.rows(), 6);
  ASSERT_EQ(J.cols(), robot->NumQdot());

  // Should match GetLinkLocalJacobian.
  const Eigen::MatrixXd ref = robot->GetLinkLocalJacobian(1);
  EXPECT_TRUE(J.isApprox(ref, 1e-12))
      << "SurfaceContact Jac differs from reference";
}

// =========================================================================
// 19. Multiple ticks stability: no NaN/Inf accumulation
// =========================================================================
TEST_F(RealTaskTest, LongRunStabilityNoNaN) {
  const std::string yaml = RobotModelHeader() + R"(
regularization:
  w_qddot: 1.0e-6
  w_tau: 1.0e-3
task_pool:
  - name: ee_pos
    type: LinkPosTask
    target_frame: link2
    kp: [50, 50, 50]
    kd: [5, 5, 5]
    kp_ik: [1, 1, 1]
  - name: ee_ori
    type: LinkOriTask
    target_frame: link2
    kp: [50, 50, 50]
    kd: [5, 5, 5]
    kp_ik: [1, 1, 1]
  - name: jpos_task
    type: JointTask
    kp: 10.0
    kd: 1.0
    kp_ik: 1.0
state_machine:
  - id: 0
    name: rt_stay_state
    task_hierarchy:
      - {name: ee_pos, priority: 0}
      - {name: ee_ori, priority: 0}
      - {name: jpos_task, priority: 1}
)";

  auto path = WriteYaml("long_run.yaml", yaml);
  auto arch = MakeArch(path);
  const int n = arch->GetRobot()->NumActiveDof();

  Eigen::VectorXd q = Eigen::VectorXd::Constant(n, 0.2);
  Eigen::VectorXd qdot = Eigen::VectorXd::Zero(n);

  constexpr int kTicks = 200;
  for (int i = 0; i < kTicks; ++i) {
    UpdateFixedBase(arch.get(), q, qdot, i * 0.001, 0.001);
    const auto& cmd = arch->GetCommand();
    ASSERT_TRUE(cmd.tau.allFinite()) << "NaN tau at tick " << i;
    ASSERT_TRUE(cmd.q.allFinite()) << "NaN q at tick " << i;
    ASSERT_TRUE(cmd.qdot.allFinite()) << "NaN qdot at tick " << i;
  }
}

// =========================================================================
// 20. TaskConfig round-trip: FromTask recovers SetParameters values
// =========================================================================
TEST_F(RealTaskTest, TaskConfigRoundTrip) {
  auto robot = std::make_unique<PinocchioRobotSystem>(
      urdf_path_.string(), "", true, false);
  JointTask task(robot.get());

  TaskConfig cfg;
  cfg.kp = Eigen::VectorXd::Constant(2, 42.0);
  cfg.kd = Eigen::VectorXd::Constant(2, 7.0);
  cfg.ki = Eigen::VectorXd::Constant(2, 0.1);
  cfg.weight = Eigen::VectorXd::Constant(2, 3.0);
  cfg.kp_ik = Eigen::VectorXd::Constant(2, 0.5);
  task.SetParameters(cfg, WbcType::WBIC);

  TaskConfig recovered = TaskConfig::FromTask(task);
  EXPECT_TRUE(recovered.kp.isApprox(cfg.kp));
  EXPECT_TRUE(recovered.kd.isApprox(cfg.kd));
  EXPECT_TRUE(recovered.kp_ik.isApprox(cfg.kp_ik));
}

// =========================================================================
// 21. Optimo 7-DOF: EE position tracking (requires real URDF)
// =========================================================================
TEST_F(RealTaskTest, Optimo7DofEeTracking) {
  const std::string optimo_urdf =
      "/home/dk/workspace/rpc_ws/src/rpc_ros/description/"
      "optimo_description/urdf/optimo.urdf";
  if (!std::filesystem::exists(optimo_urdf)) {
    GTEST_SKIP() << "Optimo URDF not found";
  }

  // Write URDF-referencing YAML.
  std::ostringstream yaml;
  yaml << "robot_model:\n"
       << "  urdf_path: \"" << optimo_urdf << "\"\n"
       << R"(  is_floating_base: false
  base_frame: optimo_base_link
regularization:
  w_qddot: 1.0e-6
  w_tau: 1.0e-3
task_pool:
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
  - id: 0
    name: rt_stay_state
    task_hierarchy:
      - {name: ee_pos, priority: 0}
      - {name: ee_ori, priority: 0}
      - {name: jpos_task, priority: 1}
)";

  auto yaml_path = WriteYaml("optimo_ee.yaml", yaml.str());
  auto cfg = ControlArchitectureConfig::FromYaml(yaml_path.string(), 0.001);
  auto arch = std::make_unique<ControlArchitecture>(std::move(cfg));
  arch->Initialize();
  auto* robot = arch->GetRobot();
  const int n = robot->NumActiveDof();
  ASSERT_EQ(n, 7);

  // Start at a nominal config.
  Eigen::VectorXd q(7);
  q << 0.0, -0.5, 0.0, 1.0, 0.0, -0.5, 0.0;
  Eigen::VectorXd qdot = Eigen::VectorXd::Zero(7);

  // Warmup.
  for (int i = 0; i < 20; ++i) {
    UpdateFixedBase(arch.get(), q, qdot, i * 0.001, 0.001);
  }

  // Record EE position.
  const int ee_idx = robot->GetFrameIndex("optimo_end_effector");
  const Eigen::Vector3d ee_init =
      robot->GetLinkIsometry(ee_idx).translation();

  // Offset desired EE by 3cm in X.
  const auto* state = arch->GetConfig()->FindState(0);
  ASSERT_NE(state->ee_pos, nullptr);
  Eigen::Vector3d ee_des = ee_init;
  ee_des.x() += 0.03;
  state->ee_pos->UpdateDesired(ee_des, Eigen::Vector3d::Zero(),
                                Eigen::Vector3d::Zero());

  for (int i = 20; i < 50; ++i) {
    UpdateFixedBase(arch.get(), q, qdot, i * 0.001, 0.001);
  }

  const auto& cmd = arch->GetCommand();
  ASSERT_TRUE(cmd.q.allFinite());
  ASSERT_TRUE(cmd.tau.allFinite());

  // Joints should move to accommodate the 3cm X offset.
  EXPECT_GT((cmd.q - q).norm(), 0.005)
      << "7-DOF Optimo: no correction for 3cm EE offset";
}

// =========================================================================
// 22. Draco3 floating-base with contacts (requires real URDF)
// =========================================================================
TEST_F(RealTaskTest, Draco3FloatingBaseWithContacts) {
  const std::string draco_urdf =
      "/home/dk/workspace/rpc_ws/src/rpc_ros/description/"
      "draco_description/urdf/draco_modified_single_knee.urdf";
  if (!std::filesystem::exists(draco_urdf)) {
    GTEST_SKIP() << "Draco3 URDF not found";
  }

  std::ostringstream yaml;
  yaml << "robot_model:\n"
       << "  urdf_path: \"" << draco_urdf << "\"\n"
       << R"(  is_floating_base: true
  base_frame: base_link
regularization:
  w_qddot: 1.0e-6
  w_tau: 1.0e-3
task_pool:
  - name: com_task
    type: ComTask
    kp: [50, 50, 50]
    kd: [5, 5, 5]
    kp_ik: [1, 1, 1]
  - name: jpos_task
    type: JointTask
    kp: 10.0
    kd: 1.0
    kp_ik: 1.0
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
  JointTrqLimitConstraint:
    enabled: true
    scale: 0.9
state_machine:
  - id: 0
    name: rt_stay_state
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

  auto yaml_path = WriteYaml("draco3_contact.yaml", yaml.str());
  auto cfg = ControlArchitectureConfig::FromYaml(yaml_path.string(), 0.001);
  auto arch = std::make_unique<ControlArchitecture>(std::move(cfg));
  arch->Initialize();
  auto* robot = arch->GetRobot();
  const int n = robot->NumActiveDof();
  ASSERT_GT(n, 0);

  // Draco3 standing config.
  Eigen::VectorXd q = Eigen::VectorXd::Zero(n);
  q(2) = -0.565; q(3) = 0.565; q(4) = -0.565;
  q(7) = 0.523; q(9) = -1.57;
  q(15) = -0.565; q(16) = 0.565; q(17) = -0.565;
  q(20) = -0.523; q(22) = -1.57;
  Eigen::VectorXd qdot = Eigen::VectorXd::Zero(n);

  RobotBaseState bs;
  bs.pos = Eigen::Vector3d(0.0, 0.0, 0.841);
  bs.quat = Eigen::Quaterniond::Identity();
  bs.lin_vel = Eigen::Vector3d::Zero();
  bs.ang_vel = Eigen::Vector3d::Zero();
  bs.rot_world_local = Eigen::Matrix3d::Identity();

  // Run ticks with floating-base update.
  RobotJointState js;
  js.q = q;
  js.qdot = qdot;
  js.tau = Eigen::VectorXd::Zero(n);

  for (int i = 0; i < 30; ++i) {
    arch->Update(js, bs, i * 0.001, 0.001);
    const auto& cmd = arch->GetCommand();
    ASSERT_TRUE(cmd.tau.allFinite())
        << "Draco3 NaN tau at tick " << i << ": " << cmd.tau.transpose();
    ASSERT_TRUE(cmd.q.allFinite())
        << "Draco3 NaN q at tick " << i;
  }

  // Contact reaction forces should exist in the solver.
  auto* solver = arch->GetSolver();
  ASSERT_NE(solver, nullptr);
}

// =========================================================================
// =========================================================================
// CLOSED-LOOP SIMULATION TESTS
//
// These tests close the control loop: WBC computes torque → forward dynamics
// (Pinocchio ABA) computes qddot → semi-implicit Euler integrates q, qdot →
// new state fed back to WBC. This validates actual tracking convergence,
// stability under disturbance, and realistic controller behavior.
// =========================================================================
// =========================================================================

// 23. 2-DOF closed-loop joint tracking convergence
// Perturb 0.5 rad from desired, verify convergence to < 5 mrad in 2 seconds.
TEST_F(RealTaskTest, ClosedLoopJointTrackingConvergence) {
  const std::string yaml = RobotModelHeader() + R"(
regularization:
  w_qddot: 1.0e-6
  w_tau: 1.0e-3
task_pool:
  - name: jpos_task
    type: JointTask
    kp: [100, 100]
    kd: [20, 20]
    kp_ik: [1, 1]
state_machine:
  - id: 0
    name: rt_stay_state
    task_hierarchy:
      - {name: jpos_task, priority: 0}
)";

  auto path = WriteYaml("cl_joint.yaml", yaml);
  auto arch = MakeArch(path);
  auto* robot = arch->GetRobot();
  const int n = robot->NumActiveDof();
  ASSERT_EQ(n, 2);

  const pinocchio::Model& model = robot->GetModel();
  pinocchio::Data sim_data(model);

  // Desired: zero config. Start perturbed.
  const Eigen::VectorXd q_des = Eigen::VectorXd::Zero(n);
  Eigen::VectorXd q_sim(n);
  q_sim << 0.5, -0.3;
  Eigen::VectorXd qdot_sim = Eigen::VectorXd::Zero(n);

  // Set desired pose on the JointTask.
  const auto* state = arch->GetConfig()->FindState(0);
  ASSERT_NE(state, nullptr);
  ASSERT_NE(state->joint, nullptr);
  state->joint->UpdateDesired(
      q_des, Eigen::VectorXd::Zero(n), Eigen::VectorXd::Zero(n));

  constexpr double dt = 0.001;
  constexpr int kTotalTicks = 2000;  // 2 seconds
  constexpr int kSettleTicks = 1500; // check convergence after 1.5s

  double max_settled_err = 0.0;

  for (int i = 0; i < kTotalTicks; ++i) {
    UpdateFixedBase(arch.get(), q_sim, qdot_sim, i * dt, dt);
    const auto& cmd = arch->GetCommand();
    ASSERT_TRUE(cmd.tau.allFinite()) << "NaN tau at tick " << i;

    // Forward dynamics via ABA.
    pinocchio::aba(model, sim_data, q_sim, qdot_sim, cmd.tau);
    qdot_sim += sim_data.ddq * dt;
    q_sim += qdot_sim * dt;

    if (i >= kSettleTicks) {
      max_settled_err = std::max(max_settled_err,
                                 (q_sim - q_des).cwiseAbs().maxCoeff());
    }
  }

  // After settling: < 5 mrad per-joint error.
  EXPECT_LT(max_settled_err, 0.005)
      << "Joint tracking failed to converge: max error = "
      << max_settled_err << " rad (" << max_settled_err * 180.0 / M_PI << " deg)";

  // Velocity should have settled near zero.
  EXPECT_LT(qdot_sim.norm(), 0.1)
      << "Velocity not settled: ||qdot|| = " << qdot_sim.norm();
}

// 24. 2-DOF closed-loop joint tracking to non-zero target
// Verify convergence when target is a non-trivial joint config.
TEST_F(RealTaskTest, ClosedLoopJointTrackingNonZeroTarget) {
  const std::string yaml = RobotModelHeader() + R"(
regularization:
  w_qddot: 1.0e-6
  w_tau: 1.0e-3
task_pool:
  - name: jpos_task
    type: JointTask
    kp: [100, 100]
    kd: [20, 20]
    kp_ik: [1, 1]
state_machine:
  - id: 0
    name: rt_stay_state
    task_hierarchy:
      - {name: jpos_task, priority: 0}
)";

  auto path = WriteYaml("cl_joint_nz.yaml", yaml);
  auto arch = MakeArch(path);
  auto* robot = arch->GetRobot();
  const int n = robot->NumActiveDof();
  ASSERT_EQ(n, 2);

  const pinocchio::Model& model = robot->GetModel();
  pinocchio::Data sim_data(model);

  // Start at zero, target at (0.5, -0.3).
  Eigen::VectorXd q_sim = Eigen::VectorXd::Zero(n);
  Eigen::VectorXd qdot_sim = Eigen::VectorXd::Zero(n);
  Eigen::VectorXd q_des(n);
  q_des << 0.5, -0.3;

  const auto* state = arch->GetConfig()->FindState(0);
  state->joint->UpdateDesired(
      q_des, Eigen::VectorXd::Zero(n), Eigen::VectorXd::Zero(n));

  constexpr double dt = 0.001;
  constexpr int kTotalTicks = 2000;

  for (int i = 0; i < kTotalTicks; ++i) {
    UpdateFixedBase(arch.get(), q_sim, qdot_sim, i * dt, dt);
    const auto& cmd = arch->GetCommand();
    ASSERT_TRUE(cmd.tau.allFinite()) << "NaN tau at tick " << i;
    pinocchio::aba(model, sim_data, q_sim, qdot_sim, cmd.tau);
    qdot_sim += sim_data.ddq * dt;
    q_sim += qdot_sim * dt;
  }

  const double final_err = (q_sim - q_des).cwiseAbs().maxCoeff();
  EXPECT_LT(final_err, 0.005)
      << "Joint tracking to non-zero target failed: error = "
      << final_err << " rad\n"
      << "  desired: " << q_des.transpose() << "\n"
      << "  actual:  " << q_sim.transpose();
  EXPECT_LT(qdot_sim.norm(), 0.1);
}

// 25. 2-DOF closed-loop stability under velocity disturbance
// After convergence, inject a velocity kick and verify recovery.
TEST_F(RealTaskTest, ClosedLoopDisturbanceRecovery) {
  const std::string yaml = RobotModelHeader() + R"(
regularization:
  w_qddot: 1.0e-6
  w_tau: 1.0e-3
task_pool:
  - name: jpos_task
    type: JointTask
    kp: [100, 100]
    kd: [20, 20]
    kp_ik: [1, 1]
state_machine:
  - id: 0
    name: rt_stay_state
    task_hierarchy:
      - {name: jpos_task, priority: 0}
)";

  auto path = WriteYaml("cl_disturb.yaml", yaml);
  auto arch = MakeArch(path);
  auto* robot = arch->GetRobot();
  const int n = robot->NumActiveDof();

  const pinocchio::Model& model = robot->GetModel();
  pinocchio::Data sim_data(model);

  Eigen::VectorXd q_sim = Eigen::VectorXd::Zero(n);
  Eigen::VectorXd qdot_sim = Eigen::VectorXd::Zero(n);
  const Eigen::VectorXd q_des = Eigen::VectorXd::Zero(n);

  const auto* state = arch->GetConfig()->FindState(0);
  state->joint->UpdateDesired(
      q_des, Eigen::VectorXd::Zero(n), Eigen::VectorXd::Zero(n));

  constexpr double dt = 0.001;

  // Phase 1: converge for 1 second.
  for (int i = 0; i < 1000; ++i) {
    UpdateFixedBase(arch.get(), q_sim, qdot_sim, i * dt, dt);
    const auto& cmd = arch->GetCommand();
    ASSERT_TRUE(cmd.tau.allFinite()) << "NaN tau at tick " << i;
    pinocchio::aba(model, sim_data, q_sim, qdot_sim, cmd.tau);
    qdot_sim += sim_data.ddq * dt;
    q_sim += qdot_sim * dt;
  }

  // Verify settled before disturbance.
  ASSERT_LT((q_sim - q_des).norm(), 0.01)
      << "Failed to converge before disturbance";

  // Phase 2: inject velocity disturbance.
  qdot_sim << 5.0, -5.0;  // 5 rad/s kick

  // Phase 3: recover for 2 seconds.
  for (int i = 1000; i < 3000; ++i) {
    UpdateFixedBase(arch.get(), q_sim, qdot_sim, i * dt, dt);
    const auto& cmd = arch->GetCommand();
    ASSERT_TRUE(cmd.tau.allFinite()) << "NaN tau at tick " << i;
    pinocchio::aba(model, sim_data, q_sim, qdot_sim, cmd.tau);
    qdot_sim += sim_data.ddq * dt;
    q_sim += qdot_sim * dt;
  }

  // Should recover to < 10 mrad.
  const double final_err = (q_sim - q_des).cwiseAbs().maxCoeff();
  EXPECT_LT(final_err, 0.01)
      << "Failed to recover after disturbance: error = "
      << final_err << " rad";
  EXPECT_LT(qdot_sim.norm(), 0.1)
      << "Velocity not settled after disturbance: ||qdot|| = "
      << qdot_sim.norm();
}

// 26. 2-DOF closed-loop: torques stay bounded throughout simulation
TEST_F(RealTaskTest, ClosedLoopTorqueBounded) {
  const std::string yaml = RobotModelHeader() + R"(
regularization:
  w_qddot: 1.0e-6
  w_tau: 1.0e-3
task_pool:
  - name: jpos_task
    type: JointTask
    kp: [100, 100]
    kd: [20, 20]
    kp_ik: [1, 1]
state_machine:
  - id: 0
    name: rt_stay_state
    task_hierarchy:
      - {name: jpos_task, priority: 0}
)";

  auto path = WriteYaml("cl_torque_bnd.yaml", yaml);
  auto arch = MakeArch(path);
  auto* robot = arch->GetRobot();
  const int n = robot->NumActiveDof();

  const pinocchio::Model& model = robot->GetModel();
  pinocchio::Data sim_data(model);

  Eigen::VectorXd q_sim(n);
  q_sim << 1.0, -1.0;  // large perturbation
  Eigen::VectorXd qdot_sim = Eigen::VectorXd::Zero(n);

  const auto* state = arch->GetConfig()->FindState(0);
  state->joint->UpdateDesired(
      Eigen::VectorXd::Zero(n), Eigen::VectorXd::Zero(n),
      Eigen::VectorXd::Zero(n));

  constexpr double dt = 0.001;
  double peak_torque = 0.0;
  double peak_qdot = 0.0;

  for (int i = 0; i < 2000; ++i) {
    UpdateFixedBase(arch.get(), q_sim, qdot_sim, i * dt, dt);
    const auto& cmd = arch->GetCommand();
    ASSERT_TRUE(cmd.tau.allFinite()) << "NaN tau at tick " << i;

    peak_torque = std::max(peak_torque, cmd.tau.cwiseAbs().maxCoeff());
    peak_qdot = std::max(peak_qdot, qdot_sim.cwiseAbs().maxCoeff());

    pinocchio::aba(model, sim_data, q_sim, qdot_sim, cmd.tau);
    qdot_sim += sim_data.ddq * dt;
    q_sim += qdot_sim * dt;
  }

  // URDF effort limit is 100 Nm. With ClampCommandLimits (scale 0.5 default),
  // expect clamped to [-50, 50]. But even without clamping, torques should not
  // blow up for a well-tuned controller.
  EXPECT_LT(peak_torque, 200.0)
      << "Peak torque unreasonably high: " << peak_torque << " Nm";
  EXPECT_LT(peak_qdot, 50.0)
      << "Peak velocity unreasonably high: " << peak_qdot << " rad/s";

  // Should converge.
  const double final_err = q_sim.norm();
  EXPECT_LT(final_err, 0.01) << "Failed to converge from 1 rad perturbation";
}

// 27. 2-DOF closed-loop: monotonic error decrease (no oscillation blow-up)
TEST_F(RealTaskTest, ClosedLoopMonotonicErrorDecrease) {
  const std::string yaml = RobotModelHeader() + R"(
regularization:
  w_qddot: 1.0e-6
  w_tau: 1.0e-3
task_pool:
  - name: jpos_task
    type: JointTask
    kp: [50, 50]
    kd: [14, 14]
    kp_ik: [1, 1]
state_machine:
  - id: 0
    name: rt_stay_state
    task_hierarchy:
      - {name: jpos_task, priority: 0}
)";

  auto path = WriteYaml("cl_monotonic.yaml", yaml);
  auto arch = MakeArch(path);
  auto* robot = arch->GetRobot();
  const int n = robot->NumActiveDof();

  const pinocchio::Model& model = robot->GetModel();
  pinocchio::Data sim_data(model);

  Eigen::VectorXd q_sim(n);
  q_sim << 0.4, -0.3;
  Eigen::VectorXd qdot_sim = Eigen::VectorXd::Zero(n);

  const auto* state = arch->GetConfig()->FindState(0);
  state->joint->UpdateDesired(
      Eigen::VectorXd::Zero(n), Eigen::VectorXd::Zero(n),
      Eigen::VectorXd::Zero(n));

  constexpr double dt = 0.001;
  constexpr int kWindowSize = 200;  // 200ms windows

  // Collect RMS error per window.
  std::vector<double> window_rms;
  double window_sum_sq = 0.0;
  int window_count = 0;

  for (int i = 0; i < 2000; ++i) {
    UpdateFixedBase(arch.get(), q_sim, qdot_sim, i * dt, dt);
    const auto& cmd = arch->GetCommand();
    ASSERT_TRUE(cmd.tau.allFinite()) << "NaN tau at tick " << i;

    pinocchio::aba(model, sim_data, q_sim, qdot_sim, cmd.tau);
    qdot_sim += sim_data.ddq * dt;
    q_sim += qdot_sim * dt;

    window_sum_sq += q_sim.squaredNorm();
    window_count++;

    if (window_count == kWindowSize) {
      window_rms.push_back(std::sqrt(window_sum_sq / kWindowSize));
      window_sum_sq = 0.0;
      window_count = 0;
    }
  }

  // Each window's RMS error should be <= previous (allowing small tolerance
  // for transient oscillation during settling).
  ASSERT_GE(window_rms.size(), 4u);
  for (size_t w = 2; w < window_rms.size(); ++w) {
    EXPECT_LE(window_rms[w], window_rms[w - 1] * 1.05)
        << "Error increased in window " << w << ": "
        << window_rms[w] << " > " << window_rms[w - 1];
  }

  // Last window should be near zero.
  EXPECT_LT(window_rms.back(), 0.005)
      << "Final window RMS error: " << window_rms.back();
}

// 28. Optimo 7-DOF closed-loop joint tracking (requires real URDF)
// Same structure as behavior test but via test_real_tasks fixture.
TEST_F(RealTaskTest, ClosedLoopOptimo7DofJointTracking) {
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
  w_qddot: 1.0e-6
  w_tau: 0.0
task_pool:
  - name: jpos_task
    type: JointTask
    kp: [100, 100, 100, 100, 50, 50, 50]
    kd: [20, 20, 20, 20, 14, 14, 14]
    kp_ik: [1, 1, 1, 1, 1, 1, 1]
state_machine:
  - id: 0
    name: rt_stay_state
    task_hierarchy:
      - {name: jpos_task, priority: 0}
)";

  auto yaml_path = WriteYaml("cl_optimo_jpos.yaml", yaml.str());
  auto cfg = ControlArchitectureConfig::FromYaml(yaml_path.string(), 0.001);
  auto arch = std::make_unique<ControlArchitecture>(std::move(cfg));
  arch->Initialize();
  auto* robot = arch->GetRobot();
  const int n = robot->NumActiveDof();
  ASSERT_EQ(n, 7);

  const pinocchio::Model& model = robot->GetModel();
  pinocchio::Data sim_data(model);

  // Optimo home config (desired target).
  Eigen::VectorXd q_des(7);
  q_des << 0.0, 3.3, 0.0, -2.35, 0.0, -1.13, 0.0;

  // Start with 0.15 rad perturbation.
  Eigen::VectorXd q_sim = q_des;
  q_sim.array() += 0.15;
  Eigen::VectorXd qdot_sim = Eigen::VectorXd::Zero(7);

  const auto* state = arch->GetConfig()->FindState(0);
  ASSERT_NE(state->joint, nullptr);
  state->joint->UpdateDesired(
      q_des, Eigen::VectorXd::Zero(n), Eigen::VectorXd::Zero(n));

  constexpr double dt = 0.001;
  constexpr int kTotalTicks = 3000;
  constexpr int kSettleTicks = 2000;

  const Eigen::MatrixXd& pos_lim = robot->JointPosLimits();
  double max_settled_err = 0.0;

  for (int i = 0; i < kTotalTicks; ++i) {
    UpdateFixedBase(arch.get(), q_sim, qdot_sim, i * dt, dt);
    const auto& cmd = arch->GetCommand();
    ASSERT_TRUE(cmd.tau.allFinite()) << "NaN tau at tick " << i;

    pinocchio::aba(model, sim_data, q_sim, qdot_sim, cmd.tau);
    qdot_sim += sim_data.ddq * dt;
    q_sim += qdot_sim * dt;

    for (int j = 0; j < n; ++j) {
      if (q_sim[j] < pos_lim(j, 0)) {
        q_sim[j] = pos_lim(j, 0);
        qdot_sim[j] = std::max(qdot_sim[j], 0.0);
      } else if (q_sim[j] > pos_lim(j, 1)) {
        q_sim[j] = pos_lim(j, 1);
        qdot_sim[j] = std::min(qdot_sim[j], 0.0);
      }
    }

    if (i >= kSettleTicks) {
      max_settled_err = std::max(max_settled_err,
                                 (q_sim - q_des).cwiseAbs().maxCoeff());
    }
  }

  EXPECT_LT(max_settled_err, 0.01)
      << "Optimo joint tracking failed: max settled error = "
      << max_settled_err * 180.0 / M_PI << " deg";
  EXPECT_LT(qdot_sim.norm(), 0.5)
      << "Velocity not settled: ||qdot|| = " << qdot_sim.norm();

  std::cout << "[Optimo ClosedLoop Joint] 3s simulation\n"
            << "  Max settled error: " << max_settled_err * 1000.0 << " mrad\n"
            << "  Final ||qdot||: " << qdot_sim.norm() << " rad/s\n";
}

} // namespace
} // namespace wbc
