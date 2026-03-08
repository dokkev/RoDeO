/**
 * @file draco_controller/test/test_draco_controller_unit.cpp
 * @brief Unit tests for draco_controller state machines.
 */
#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>

#include <gtest/gtest.h>

#include "wbc_architecture/control_architecture.hpp"
#include "wbc_architecture/control_architecture_config.hpp"
#include "wbc_fsm/state_factory.hpp"
#include "wbc_robot_system/pinocchio_robot_system.hpp"
#include "wbc_robot_system/state_provider.hpp"

// Force-link the draco state machine registrations.
#include "draco_controller/state_machines/initialize.hpp"
#include "draco_controller/state_machines/balance.hpp"

namespace wbc {
namespace {

static const char* kDracoUrdf =
    "/home/dk/workspace/rpc_ws/src/rpc_ros/description/"
    "draco_description/urdf/draco_modified_single_knee.urdf";

// Standing joint config for Draco3 single-knee model (25 active DOFs).
Eigen::VectorXd Draco3Standing() {
  Eigen::VectorXd q = Eigen::VectorXd::Zero(25);
  q(2)  = -0.565;  q(3)  =  0.565;  q(4)  = -0.565;
  q(7)  =  0.523;  q(9)  = -1.57;
  q(15) = -0.565;  q(16) =  0.565;  q(17) = -0.565;
  q(20) = -0.523;  q(22) = -1.57;
  return q;
}

RobotBaseState Draco3StandingBase() {
  RobotBaseState bs;
  bs.pos             = Eigen::Vector3d(0.0, 0.0, 0.841);
  bs.quat            = Eigen::Quaterniond::Identity();
  bs.lin_vel         = Eigen::Vector3d::Zero();
  bs.ang_vel         = Eigen::Vector3d::Zero();
  bs.rot_world_local = Eigen::Matrix3d::Identity();
  return bs;
}

// Inline WBC config for Draco3 balance (mirrors draco_wbc.yaml + task_list.yaml).
std::string MakeDracoBalanceYaml(const std::string& urdf_path) {
  return
    "robot_model:\n"
    "  urdf_path: \"" + urdf_path + "\"\n"
    "  is_floating_base: true\n"
    "  base_frame: torso_link\n"
    "controller:\n"
    "  enable_gravity_compensation: true\n"
    "  enable_coriolis_compensation: false\n"
    "  enable_inertia_compensation: false\n"
    "  kp_acc: 100.0\n"
    "  kd_acc: 20.0\n"
    "regularization:\n"
    "  w_qddot: 1.0e-6\n"
    "  w_rf: 1.0e-4\n"
    "  w_tau: 1.0e-3\n"
    "  w_tau_dot: 0.0\n"
    "  w_xc_ddot: 1.0e-3\n"
    "  w_f_dot: 1.0e-3\n"
    "global_constraints:\n"
    "  JointPosLimitConstraint:\n"
    "    enabled: true\n"
    "    scale: 1.0\n"
    "    is_soft: true\n"
    "    soft_weight: 1.0e+5\n"
    "  JointVelLimitConstraint:\n"
    "    enabled: true\n"
    "    scale: 1.0\n"
    "    is_soft: true\n"
    "    soft_weight: 1.0e+5\n"
    "  JointTrqLimitConstraint:\n"
    "    enabled: true\n"
    "    scale: 1.0\n"
    "    is_soft: false\n"
    "start_state_id: 1\n"
    "task_pool:\n"
    "  - name: com_task\n"
    "    type: ComTask\n"
    "    kp: [40, 40, 40]\n"
    "    kd: [8, 8, 8]\n"
    "    kp_ik: [1.0, 1.0, 1.0]\n"
    "    weight: [50, 50, 50]\n"
    "  - name: jpos_task\n"
    "    type: JointTask\n"
    "    kp: 30.0\n"
    "    kd: 3.0\n"
    "    kp_ik: 1.0\n"
    "    weight: 1.0\n"
    "  - name: lfoot_force\n"
    "    type: ForceTask\n"
    "    contact_name: lfoot_contact\n"
    "    weight: [0, 0, 0, 0, 0, 0]\n"
    "  - name: rfoot_force\n"
    "    type: ForceTask\n"
    "    contact_name: rfoot_contact\n"
    "    weight: [0, 0, 0, 0, 0, 0]\n"
    "contact_pool:\n"
    "  - name: lfoot_contact\n"
    "    type: SurfaceContact\n"
    "    target_frame: l_foot_contact\n"
    "    mu: 0.5\n"
    "    foot_half_length: 0.10\n"
    "    foot_half_width: 0.05\n"
    "  - name: rfoot_contact\n"
    "    type: SurfaceContact\n"
    "    target_frame: r_foot_contact\n"
    "    mu: 0.5\n"
    "    foot_half_length: 0.10\n"
    "    foot_half_width: 0.05\n"
    "state_machine:\n"
    "  - id: 1\n"
    "    name: draco_balance\n"
    "    task_hierarchy:\n"
    "      - {name: com_task, priority: 0}\n"
    "      - {name: jpos_task, priority: 1}\n"
    "    contact_constraints:\n"
    "      - {name: lfoot_contact}\n"
    "      - {name: rfoot_contact}\n"
    "    force_tasks:\n"
    "      - {name: lfoot_force}\n"
    "      - {name: rfoot_force}\n";
}

std::unique_ptr<ControlArchitecture> MakeDracoArch(const std::string& yaml_content,
                                                   const std::string& tmp_name) {
  const auto tmp = std::filesystem::temp_directory_path() / tmp_name;
  { std::ofstream f(tmp); f << yaml_content; }
  auto cfg = ControlArchitectureConfig::FromYaml(tmp.string(), 0.001);
  cfg.state_provider = std::make_unique<StateProvider>(0.001);
  auto arch = std::make_unique<ControlArchitecture>(std::move(cfg));
  arch->Initialize();
  return arch;
}

// -----------------------------------------------------------------------
// 1. State factory: verify all draco state types are registered.
// -----------------------------------------------------------------------
TEST(DracoControllerUnit, StateFactoryRegistration) {
  EXPECT_TRUE(StateFactory::Instance().Has("draco_initialize"));
  EXPECT_TRUE(StateFactory::Instance().Has("draco_balance"));
  EXPECT_TRUE(StateFactory::Instance().Has("draco_home"));
}

// -----------------------------------------------------------------------
// 2. Functional: WBC produces finite, non-zero torques at standing config.
// -----------------------------------------------------------------------
TEST(DracoControllerUnit, BalanceState_ProducesValidTorques) {
  if (!std::filesystem::exists(kDracoUrdf)) {
    GTEST_SKIP() << "Draco3 URDF not found";
  }

  auto arch = MakeDracoArch(MakeDracoBalanceYaml(kDracoUrdf), "draco_balance_test.yaml");
  ASSERT_NE(arch, nullptr);

  const Eigen::VectorXd q    = Draco3Standing();
  const Eigen::VectorXd qdot = Eigen::VectorXd::Zero(25);
  const RobotBaseState  bs   = Draco3StandingBase();

  // Set jpos desired = current (hold standing pose).
  const StateConfig* state = arch->GetConfig()->FindState(1);
  ASSERT_NE(state, nullptr);
  state->joint->UpdateDesired(q, Eigen::VectorXd::Zero(25), Eigen::VectorXd::Zero(25));

  // Warm up.
  for (int i = 0; i < 50; ++i) {
    RobotJointState js;
    js.q   = q;
    js.qdot = qdot;
    js.tau  = Eigen::VectorXd::Zero(25);
    arch->Update(js, bs, i * 0.001, 0.001);
  }

  const auto& cmd = arch->GetCommand();
  ASSERT_TRUE(cmd.tau.allFinite()) << "Torque command contains NaN/Inf";
  ASSERT_EQ(cmd.tau.size(), 25) << "Expected 25 actuated DOFs";

  // Gravity compensation should produce non-trivial torques for a standing humanoid.
  const double tau_norm = cmd.tau.norm();
  EXPECT_GT(tau_norm, 1.0) << "Torque norm too small — gravity comp may be off";

  std::cout << "\n[Draco3 Balance] Torque at standing config (gravity comp only):\n"
            << "  tau norm: " << tau_norm << " Nm\n";
}

// -----------------------------------------------------------------------
// 3. Performance: measure WBC solve rate (max control frequency).
// -----------------------------------------------------------------------
TEST(DracoControllerUnit, BalanceState_MaxControlFrequency) {
  if (!std::filesystem::exists(kDracoUrdf)) {
    GTEST_SKIP() << "Draco3 URDF not found";
  }

  auto arch = MakeDracoArch(MakeDracoBalanceYaml(kDracoUrdf), "draco_perf_test.yaml");
  ASSERT_NE(arch, nullptr);

  const Eigen::VectorXd q    = Draco3Standing();
  const Eigen::VectorXd qdot = Eigen::VectorXd::Zero(25);
  const RobotBaseState  bs   = Draco3StandingBase();

  const StateConfig* state = arch->GetConfig()->FindState(1);
  ASSERT_NE(state, nullptr);
  state->joint->UpdateDesired(q, Eigen::VectorXd::Zero(25), Eigen::VectorXd::Zero(25));

  // Warm up (not timed).
  for (int i = 0; i < 200; ++i) {
    RobotJointState js;
    js.q   = q;
    js.qdot = qdot;
    js.tau  = Eigen::VectorXd::Zero(25);
    arch->Update(js, bs, i * 0.001, 0.001);
  }

  // Timed run.
  constexpr int kTicks = 1000;
  const auto& cmd = arch->GetCommand();
  int finite_count = 0;

  const auto t_start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < kTicks; ++i) {
    RobotJointState js;
    js.q   = q;
    js.qdot = qdot;
    js.tau  = Eigen::VectorXd::Zero(25);
    arch->Update(js, bs, (200 + i) * 0.001, 0.001);
    if (cmd.tau.allFinite()) ++finite_count;
  }
  const auto t_end = std::chrono::high_resolution_clock::now();

  const double elapsed_s = std::chrono::duration<double>(t_end - t_start).count();
  const double avg_us    = elapsed_s / kTicks * 1e6;
  const double max_hz    = 1.0 / (elapsed_s / kTicks);

  std::cout << "\n[Draco3 Balance] Control frequency (Release build, open-loop):\n"
            << "  Ticks:        " << kTicks << "\n"
            << "  Total time:   " << elapsed_s * 1000.0 << " ms\n"
            << "  Avg per tick: " << avg_us << " us\n"
            << "  Max Hz:       " << static_cast<int>(max_hz) << " Hz\n"
            << "  Solve success: " << finite_count << "/" << kTicks << "\n";

  EXPECT_EQ(finite_count, kTicks) << "Some ticks produced non-finite torques";
  // Expect at least 500 Hz. In Release build with ProxQP should be >1 kHz.
  EXPECT_LT(avg_us, 2000.0) << "WBC too slow: " << avg_us << " us/tick";
}

}  // namespace
}  // namespace wbc
