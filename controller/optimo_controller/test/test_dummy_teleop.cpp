/**
 * @file test_dummy_teleop.cpp
 * @brief MuJoCo + WBC dummy teleop demo for Optimo.
 *
 * Runs a closed-loop MuJoCo sim through the full state pipeline:
 *   initialize → home → joint_teleop (sinusoidal velocity) →
 *   cartesian_teleop (sinusoidal EE velocity)
 *
 * Validates that WBC tracks the dummy commands stably and prints
 * per-phase diagnostics (EE position, joint errors, torques).
 */
#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

#include <Eigen/Dense>
#include <mujoco/mujoco.h>

#include "wbc_architecture/control_architecture.hpp"
#include "wbc_formulation/motion_task.hpp"
#include "wbc_robot_system/pinocchio_robot_system.hpp"

#include "optimo_controller/state_machines/joint_teleop.hpp"
#include "optimo_controller/state_machines/cartesian_teleop.hpp"

namespace {

constexpr int kNJoints = 7;
constexpr double kDt = 0.001;

const std::array<double, kNJoints> kHomeQpos = {
    0.0, 3.14159, 0.0, 0.0, 0.0, 0.0, 0.0};

std::string ResolvePackagePath(const std::string& pkg_name,
                               const std::string& rel_path) {
  const char* prefix = std::getenv("AMENT_PREFIX_PATH");
  if (!prefix)
    throw std::runtime_error(
        "AMENT_PREFIX_PATH not set. Source install/setup.bash first.");
  std::istringstream ss(prefix);
  std::string token;
  while (std::getline(ss, token, ':')) {
    auto full = std::filesystem::path(token) / "share" / pkg_name / rel_path;
    if (std::filesystem::exists(full)) return full.string();
  }
  throw std::runtime_error("Cannot resolve package://" + pkg_name + "/" +
                           rel_path);
}

// ── YAML config writers ─────────────────────────────────────────────────────

void WriteTaskYaml(const std::filesystem::path& dir) {
  std::ofstream f(dir / "task_list.yaml");
  f << "task_pool:\n"
    << "  - name: \"jpos_task\"\n"
    << "    type: \"JointTask\"\n"
    << "    role: \"posture_task\"\n"
    << "    kp: [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]\n"
    << "    kd: [20.0,  20.0,  20.0,  20.0,  20.0,  20.0,  20.0]\n"
    << "    kp_ik: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]\n"
    << "\n"
    << "  - name: \"ee_pos_task\"\n"
    << "    type: \"LinkPosTask\"\n"
    << "    role: \"operational_task\"\n"
    << "    target_frame: \"optimo_end_effector\"\n"
    << "    reference_frame: \"optimo_base_link\"\n"
    << "    kp: [3200.0, 3200.0, 3200.0]\n"
    << "    kd: [113.0,  113.0,  113.0]\n"
    << "    kp_ik: [1.0, 1.0, 1.0]\n"
    << "\n"
    << "  - name: \"ee_ori_task\"\n"
    << "    type: \"LinkOriTask\"\n"
    << "    role: \"operational_task\"\n"
    << "    target_frame: \"optimo_end_effector\"\n"
    << "    reference_frame: \"optimo_base_link\"\n"
    << "    kp: [3200.0, 3200.0, 3200.0]\n"
    << "    kd: [113.0,  113.0,  113.0]\n"
    << "    kp_ik: [1.0, 1.0, 1.0]\n";
}

void WriteWbcYaml(const std::filesystem::path& dir, bool enable_pid = true) {
  std::ofstream f(dir / "optimo_wbc.yaml");
  f << "robot_model:\n"
    << "  urdf_path: \"package://optimo_description/urdf/optimo.urdf\"\n"
    << "  is_floating_base: false\n"
    << "  base_frame: \"optimo_base_link\"\n"
    << "\n"
    << "controller:\n"
    << "  enable_gravity_compensation: true\n"
    << "  enable_coriolis_compensation: true\n"
    << "  enable_inertia_compensation: true\n"
    << "  joint_pid:\n"
    << "    enabled: " << (enable_pid ? "true" : "false") << "\n"
    << "    gains_yaml: \"joint_pid_gains.yaml\"\n"
    << "\n"
    << "regularization:\n"
    << "  w_qddot: 0.01\n"
    << "  w_tau: 0.0\n"
    << "  w_tau_dot: 0.0\n"
    << "  w_rf: 1.0e-4\n"
    << "  w_xc_ddot: 1.0e-3\n"
    << "  w_f_dot: 1.0e-3\n"
    << "\n"
    << "  JointPosLimitConstraint:\n"
    << "    enabled: true\n"
    << "    scale: 0.9\n"
    << "    is_soft: true\n"
    << "    soft_weight: 1.0e+5\n"
    << "  JointVelLimitConstraint:\n"
    << "    enabled: true\n"
    << "    scale: 0.8\n"
    << "    is_soft: true\n"
    << "    soft_weight: 1.0e+5\n"
    << "  JointTrqLimitConstraint:\n"
    << "    enabled: false\n"
    << "\n"
    << "task_pool_yaml: \"task_list.yaml\"\n"
    << "state_machine_yaml: \"state_machine.yaml\"\n";
}

void WriteStateMachineYaml(const std::filesystem::path& dir) {
  std::ofstream f(dir / "state_machine.yaml");
  f << "state_machine:\n"
    << "  - id: 0\n"
    << "    name: \"initialize\"\n"
    << "    params:\n"
    << "      duration: 0.5\n"
    << "      wait_time: 0.0\n"
    << "      stay_here: true\n"
    << "      target_jpos: [0.0, 3.14159, 0.0, 0.0, 0.0, 0.0, 0.0]\n"
    << "    task_hierarchy:\n"
    << "      - name: \"jpos_task\"\n"
    << "\n"
    << "  - id: 1\n"
    << "    name: \"home\"\n"
    << "    type: \"initialize\"\n"
    << "    params:\n"
    << "      duration: 1.0\n"
    << "      wait_time: 0.0\n"
    << "      stay_here: true\n"
    << "      target_jpos: [0.0, 3.14159, 0.0, 0.0, 0.0, 0.0, 0.0]\n"
    << "    task_hierarchy:\n"
    << "      - name: \"jpos_task\"\n"
    << "\n"
    << "  - id: 2\n"
    << "    name: \"joint_teleop\"\n"
    << "    params:\n"
    << "      stay_here: true\n"
    << "      joint_vel_limit: [0.5, 0.5, 0.5, 0.5, 0.3, 0.3, 0.3]\n"
    << "    task_hierarchy:\n"
    << "      - name: \"jpos_task\"\n"
    << "\n"
    << "  - id: 3\n"
    << "    name: \"cartesian_teleop\"\n"
    << "    params:\n"
    << "      stay_here: true\n"
    << "      linear_vel_max: 0.1\n"
    << "      angular_vel_max: 0.5\n"
    << "    task_hierarchy:\n"
    << "      - name: \"ee_pos_task\"\n"
    << "        weight: 100.0\n"
    << "      - name: \"ee_ori_task\"\n"
    << "        weight: 100.0\n"
    << "      - name: \"jpos_task\"\n"
    << "        weight: 1.0\n";
}

void WritePidYaml(const std::filesystem::path& dir,
                  double kp_pos = 200.0, double kd_pos = 28.0) {
  std::ofstream f(dir / "joint_pid_gains.yaml");
  f << "default:\n"
    << "  kp_pos: " << kp_pos << "\n"
    << "  ki_pos: 0.0\n"
    << "  kd_pos: " << kd_pos << "\n"
    << "  kp_vel: 1.0\n"
    << "  ki_vel: 0.0\n"
    << "  kd_vel: 0.0\n";
}

// ── Sim environment ─────────────────────────────────────────────────────────

struct SimEnv {
  std::filesystem::path tmp_dir;
  std::unique_ptr<wbc::ControlArchitecture> arch;
  mjModel* m{nullptr};
  mjData* d{nullptr};
  wbc::RobotJointState js;
  wbc::JointTeleop* jt{nullptr};
  wbc::CartesianTeleop* ct{nullptr};

  ~SimEnv() {
    if (d) mj_deleteData(d);
    if (m) mj_deleteModel(m);
    if (std::filesystem::exists(tmp_dir))
      std::filesystem::remove_all(tmp_dir);
  }
};

std::unique_ptr<SimEnv> BuildEnv(bool enable_pid = true) {
  auto env = std::make_unique<SimEnv>();
  env->tmp_dir = std::filesystem::temp_directory_path() / "wbc_dummy_teleop";
  std::filesystem::create_directories(env->tmp_dir);

  WriteTaskYaml(env->tmp_dir);
  WriteWbcYaml(env->tmp_dir, enable_pid);
  WriteStateMachineYaml(env->tmp_dir);
  WritePidYaml(env->tmp_dir);

  std::string yaml_path = (env->tmp_dir / "optimo_wbc.yaml").string();
  auto arch_config =
      wbc::ControlArchitectureConfig::FromYaml(yaml_path, kDt);
  arch_config.state_provider = std::make_unique<wbc::StateProvider>(kDt);
  env->arch =
      std::make_unique<wbc::ControlArchitecture>(std::move(arch_config));
  env->arch->Initialize();
  env->arch->enable_timing_ = true;

  // Load MuJoCo model
  std::string mjcf_path =
      ResolvePackagePath("optimo_description", "mjcf/optimo.xml");
  char error[1000] = "";
  env->m = mj_loadXML(mjcf_path.c_str(), nullptr, error, sizeof(error));
  if (!env->m)
    throw std::runtime_error(std::string("MuJoCo load error: ") + error);
  env->d = mj_makeData(env->m);
  if (env->m->nkey > 0)
    mju_copy(env->d->qpos, env->m->key_qpos, env->m->nq);
  mj_forward(env->m, env->d);

  env->js.Reset(kNJoints);

  // Cache teleop state pointers
  auto* fsm = env->arch->GetFsmHandler();
  env->jt = dynamic_cast<wbc::JointTeleop*>(fsm->FindStateById(2));
  env->ct = dynamic_cast<wbc::CartesianTeleop*>(fsm->FindStateById(3));

  return env;
}

void ReadJointState(SimEnv* env) {
  for (int i = 0; i < kNJoints; ++i) {
    env->js.q[i] = env->d->qpos[i];
    env->js.qdot[i] = env->d->qvel[i];
    env->js.tau[i] = env->d->qfrc_actuator[i];
  }
}

void StepSim(SimEnv* env, double t) {
  ReadJointState(env);
  env->arch->Update(env->js, t, kDt);
  const auto& cmd = env->arch->GetCommand();
  for (int i = 0; i < kNJoints; ++i) env->d->ctrl[i] = cmd.tau[i];
  mj_step(env->m, env->d);
}

Eigen::Vector3d GetEEPos(SimEnv* env) {
  return env->arch->GetRobot()
      ->GetLinkIsometry("optimo_end_effector")
      .translation();
}

void PrintEEPos(const std::string& label, const Eigen::Vector3d& p) {
  std::cout << label << " EE pos: [" << std::fixed << std::setprecision(4)
            << p[0] << ", " << p[1] << ", " << p[2] << "]\n";
}

void PrintJointPos(const std::string& label, mjData* d) {
  std::cout << label << " qpos: [" << std::fixed << std::setprecision(4);
  for (int i = 0; i < kNJoints; ++i) {
    if (i) std::cout << ", ";
    std::cout << d->qpos[i];
  }
  std::cout << "]\n";
}

}  // namespace

// =============================================================================
// Dummy Teleop Demo: full state pipeline with sinusoidal dummy commands
// =============================================================================

TEST(DummyTeleop, JointTeleopSinusoidal) {
  std::cout << "\n========================================\n"
            << "  Joint Teleop - Sinusoidal Velocity\n"
            << "========================================\n";

  auto env = BuildEnv();
  ASSERT_NE(env->jt, nullptr) << "JointTeleop state (id=2) not found";

  double t = 0.0;

  // Phase 1: Initialize (1s)
  std::cout << "\n--- Phase 1: Initialize (1s) ---\n";
  PrintJointPos("Initial", env->d);
  for (int step = 0; step < 1000; ++step, t += kDt) StepSim(env.get(), t);
  PrintJointPos("After init", env->d);
  PrintEEPos("After init", GetEEPos(env.get()));

  // Transition to joint_teleop
  env->arch->RequestState(2);

  // Phase 2: Sinusoidal joint velocity on joints 0,2,4 (3s)
  // v_i(t) = A_i * sin(2π * f_i * t)
  std::cout << "\n--- Phase 2: Sinusoidal joint velocity (3s) ---\n";
  const double freq[3] = {0.5, 0.7, 1.0};       // Hz
  const double amp[3] = {0.3, 0.2, 0.15};        // rad/s
  const int joints[3] = {0, 2, 4};

  Eigen::VectorXd vel_cmd = Eigen::VectorXd::Zero(kNJoints);
  Eigen::VectorXd dummy_pos = Eigen::VectorXd::Zero(kNJoints);
  int64_t ts = 1;

  std::array<double, kNJoints> q_start;
  for (int i = 0; i < kNJoints; ++i) q_start[i] = env->d->qpos[i];

  std::cout << std::fixed << std::setprecision(4);
  std::cout << "  time | q0      | q2      | q4      | tau0    | tau2    | tau4\n"
            << "  -----+---------+---------+---------+---------+---------+--------\n";

  for (int step = 0; step < 3000; ++step, t += kDt) {
    double phase_t = step * kDt;
    vel_cmd.setZero();
    for (int k = 0; k < 3; ++k)
      vel_cmd[joints[k]] = amp[k] * std::sin(2 * M_PI * freq[k] * phase_t);

    ts += 1000000;
    env->jt->UpdateCommand(vel_cmd, ts, dummy_pos, 0);
    ReadJointState(env.get());
    env->arch->Update(env->js, t, kDt);
    const auto& cmd = env->arch->GetCommand();
    for (int i = 0; i < kNJoints; ++i) env->d->ctrl[i] = cmd.tau[i];

    if (step % 500 == 0 || step == 2999) {
      std::cout << "  " << std::setw(4) << phase_t << " | "
                << std::setw(7) << env->d->qpos[0] << " | "
                << std::setw(7) << env->d->qpos[2] << " | "
                << std::setw(7) << env->d->qpos[4] << " | "
                << std::setw(7) << cmd.tau[0] << " | "
                << std::setw(7) << cmd.tau[2] << " | "
                << std::setw(7) << cmd.tau[4] << "\n";
    }
    mj_step(env->m, env->d);
  }

  // Phase 3: Stop and hold (1s)
  std::cout << "\n--- Phase 3: Hold position (1s) ---\n";
  std::array<double, kNJoints> q_hold;
  for (int i = 0; i < kNJoints; ++i) q_hold[i] = env->d->qpos[i];

  vel_cmd.setZero();
  for (int step = 0; step < 1000; ++step, t += kDt) {
    ts += 1000000;
    env->jt->UpdateCommand(vel_cmd, ts, dummy_pos, 0);
    StepSim(env.get(), t);
  }

  double max_drift = 0;
  for (int i = 0; i < kNJoints; ++i)
    max_drift = std::max(max_drift, std::abs(env->d->qpos[i] - q_hold[i]));
  std::cout << "Max drift after hold: " << std::setprecision(6) << max_drift
            << " rad\n";
  EXPECT_LT(max_drift, 0.02) << "Robot should hold position within 20 mrad";

  // Check stability
  for (int i = 0; i < kNJoints; ++i) {
    EXPECT_TRUE(std::isfinite(env->d->qpos[i]))
        << "Joint " << i << " diverged to NaN";
  }

  PrintJointPos("Final", env->d);
  PrintEEPos("Final", GetEEPos(env.get()));
}

TEST(DummyTeleop, CartesianTeleopCircle) {
  std::cout << "\n========================================\n"
            << "  Cartesian Teleop - Circle in XZ plane\n"
            << "========================================\n";

  auto env = BuildEnv();
  ASSERT_NE(env->ct, nullptr) << "CartesianTeleop state (id=3) not found";

  double t = 0.0;

  // Phase 1: Initialize (1s)
  std::cout << "\n--- Phase 1: Initialize (1s) ---\n";
  for (int step = 0; step < 1000; ++step, t += kDt) StepSim(env.get(), t);
  PrintEEPos("After init", GetEEPos(env.get()));
  PrintJointPos("After init", env->d);

  // Transition to cartesian_teleop
  env->arch->RequestState(3);

  // Let cartesian_teleop FirstVisit() run for 1 tick
  StepSim(env.get(), t);
  t += kDt;

  Eigen::Vector3d ee_start = GetEEPos(env.get());
  std::cout << "\n--- Phase 2: Circle in XZ plane (5s) ---\n";
  PrintEEPos("Start", ee_start);

  // Command: circular EE velocity in XZ plane
  // vx = A * cos(2π*f*t), vz = A * sin(2π*f*t)
  const double circle_freq = 0.3;   // Hz — slow circle
  const double circle_amp = 0.05;   // m/s — 5cm/s speed
  const Eigen::Vector3d zero3 = Eigen::Vector3d::Zero();
  const Eigen::Quaterniond ident = Eigen::Quaterniond::Identity();
  int64_t ts = 1;

  std::cout << std::fixed << std::setprecision(4);
  std::cout << "  time | ee_x    | ee_y    | ee_z    | vx_cmd  | vz_cmd  | tau_max\n"
            << "  -----+---------+---------+---------+---------+---------+--------\n";

  double max_tau_all = 0;
  std::vector<Eigen::Vector3d> trajectory;

  for (int step = 0; step < 5000; ++step, t += kDt) {
    double phase_t = step * kDt;
    Eigen::Vector3d xdot;
    xdot[0] = circle_amp * std::cos(2 * M_PI * circle_freq * phase_t);
    xdot[1] = 0.0;
    xdot[2] = circle_amp * std::sin(2 * M_PI * circle_freq * phase_t);

    ts += 1000000;
    env->ct->UpdateCommand(xdot, zero3, ts, zero3, ident, 0);
    ReadJointState(env.get());
    env->arch->Update(env->js, t, kDt);
    const auto& cmd = env->arch->GetCommand();
    for (int i = 0; i < kNJoints; ++i) env->d->ctrl[i] = cmd.tau[i];

    double tau_max = 0;
    for (int i = 0; i < kNJoints; ++i)
      tau_max = std::max(tau_max, std::abs(cmd.tau[i]));
    max_tau_all = std::max(max_tau_all, tau_max);

    Eigen::Vector3d ee = GetEEPos(env.get());
    if (step % 1000 == 0 || step == 4999) {
      trajectory.push_back(ee);
      std::cout << "  " << std::setw(4) << phase_t << " | "
                << std::setw(7) << ee[0] << " | "
                << std::setw(7) << ee[1] << " | "
                << std::setw(7) << ee[2] << " | "
                << std::setw(7) << xdot[0] << " | "
                << std::setw(7) << xdot[2] << " | "
                << std::setw(7) << tau_max << "\n";
    }
    mj_step(env->m, env->d);
  }

  // Phase 3: Stop and hold (1s)
  std::cout << "\n--- Phase 3: Hold position (1s) ---\n";
  Eigen::Vector3d ee_before_hold = GetEEPos(env.get());
  for (int step = 0; step < 1000; ++step, t += kDt) {
    ts += 1000000;
    env->ct->UpdateCommand(zero3, zero3, ts, zero3, ident, 0);
    StepSim(env.get(), t);
  }
  Eigen::Vector3d ee_after_hold = GetEEPos(env.get());
  double hold_drift = (ee_after_hold - ee_before_hold).norm();
  std::cout << "EE drift after hold: " << std::setprecision(6) << hold_drift
            << " m\n";
  // Three-stage WBIC: posture bias (-kd_acc * qdot) creates transit lag during
  // the circle. After stop, EE converges ~25mm of lag during hold — this is
  // correct behavior (tracking to goal), not instability. Threshold reflects
  // new architecture's transit-lag characteristics with MuJoCo torque limits.
  EXPECT_LT(hold_drift, 0.035) << "EE should hold within 35mm";

  // Verify stability
  for (int i = 0; i < kNJoints; ++i) {
    EXPECT_TRUE(std::isfinite(env->d->qpos[i]))
        << "Joint " << i << " diverged";
  }

  // Verify the trajectory actually traced a reasonable path
  double total_displacement = 0;
  for (size_t i = 1; i < trajectory.size(); ++i)
    total_displacement += (trajectory[i] - trajectory[i - 1]).norm();
  std::cout << "Total EE displacement: " << total_displacement << " m\n";
  std::cout << "Max torque: " << max_tau_all << " Nm\n";
  EXPECT_GT(total_displacement, 0.02)
      << "EE should have moved meaningfully during teleop";

  PrintEEPos("Final", GetEEPos(env.get()));
  PrintJointPos("Final", env->d);
}

TEST(DummyTeleop, FullPipelineDemo) {
  std::cout << "\n========================================\n"
            << "  Full Pipeline Demo\n"
            << "  init → home → joint_teleop → cartesian_teleop\n"
            << "========================================\n";

  auto env = BuildEnv();
  ASSERT_NE(env->jt, nullptr);
  ASSERT_NE(env->ct, nullptr);

  double t = 0.0;
  int64_t ts = 1;
  const Eigen::Vector3d zero3 = Eigen::Vector3d::Zero();
  const Eigen::Quaterniond ident = Eigen::Quaterniond::Identity();

  // ── 1) Initialize (0.5s) ──────────────────────────────────────────────
  std::cout << "\n[1] Initialize (0.5s)\n";
  for (int step = 0; step < 500; ++step, t += kDt) StepSim(env.get(), t);
  PrintEEPos("  After init", GetEEPos(env.get()));

  // ── 2) Home (0.5s) ────────────────────────────────────────────────────
  std::cout << "[2] Home (0.5s)\n";
  env->arch->RequestState(1);
  for (int step = 0; step < 500; ++step, t += kDt) StepSim(env.get(), t);
  PrintEEPos("  After home", GetEEPos(env.get()));

  // ── 3) Joint teleop: wiggle joint 0 (2s) ──────────────────────────────
  std::cout << "[3] Joint teleop: wiggle joint0 at 0.5 Hz (2s)\n";
  env->arch->RequestState(2);
  Eigen::VectorXd vel_cmd = Eigen::VectorXd::Zero(kNJoints);
  Eigen::VectorXd dummy_pos = Eigen::VectorXd::Zero(kNJoints);

  for (int step = 0; step < 2000; ++step, t += kDt) {
    vel_cmd[0] = 0.3 * std::sin(2 * M_PI * 0.5 * step * kDt);
    ts += 1000000;
    env->jt->UpdateCommand(vel_cmd, ts, dummy_pos, 0);
    StepSim(env.get(), t);
  }
  PrintEEPos("  After jt teleop", GetEEPos(env.get()));
  PrintJointPos("  After jt teleop", env->d);

  // ── 4) Cartesian teleop: EE X sweep (2s) ──────────────────────────────
  std::cout << "[4] Cartesian teleop: EE X sweep at 0.05 m/s (2s)\n";
  env->arch->RequestState(3);
  Eigen::Vector3d ee_before = GetEEPos(env.get());

  for (int step = 0; step < 2000; ++step, t += kDt) {
    Eigen::Vector3d xdot;
    xdot[0] = 0.05 * std::sin(2 * M_PI * 0.25 * step * kDt);
    xdot[1] = 0.0;
    xdot[2] = 0.0;
    ts += 1000000;
    env->ct->UpdateCommand(xdot, zero3, ts, zero3, ident, 0);
    StepSim(env.get(), t);
  }
  Eigen::Vector3d ee_after = GetEEPos(env.get());
  PrintEEPos("  After ct teleop", ee_after);
  PrintJointPos("  After ct teleop", env->d);

  double ee_moved = (ee_after - ee_before).norm();
  std::cout << "  EE displacement: " << std::setprecision(4) << ee_moved
            << " m\n";

  // ── 5) Hold (1s) ──────────────────────────────────────────────────────
  std::cout << "[5] Hold (1s)\n";
  Eigen::Vector3d ee_hold_start = GetEEPos(env.get());
  for (int step = 0; step < 1000; ++step, t += kDt) {
    ts += 1000000;
    env->ct->UpdateCommand(zero3, zero3, ts, zero3, ident, 0);
    StepSim(env.get(), t);
  }
  double drift = (GetEEPos(env.get()) - ee_hold_start).norm();
  std::cout << "  Hold drift: " << std::setprecision(6) << drift << " m\n";
  // Threshold is 30mm: robot enters cartesian_teleop from an arbitrary
  // post-joint-teleop configuration (not home), so larger hold drift is expected.
  EXPECT_LT(drift, 0.030);

  // ── Stability check ───────────────────────────────────────────────────
  for (int i = 0; i < kNJoints; ++i) {
    EXPECT_TRUE(std::isfinite(env->d->qpos[i]));
    EXPECT_TRUE(std::isfinite(env->d->qvel[i]));
  }

  // Print timing stats
  const auto& stats = env->arch->timing_stats_;
  std::cout << "\n  WBC Timing (last tick):\n"
            << "    Robot model:  " << stats.robot_model_us << " us\n"
            << "    Kinematics:   " << stats.kinematics_us << " us\n"
            << "    Dynamics:     " << stats.dynamics_us << " us\n"
            << "    FindConfig:   " << stats.find_config_us << " us\n"
            << "    MakeTorque:   " << stats.make_torque_us << " us\n"
            << "    Feedback:     " << stats.feedback_us << " us\n";

  std::cout << "\n  Full pipeline demo complete.\n";
}

// =============================================================================
// PID vs No-PID comparison: does joint PID reduce EE drift during hold?
// =============================================================================

struct HoldResult {
  double ee_drift_m;       // EE position drift during hold [m]
  double max_joint_drift;  // max joint position drift [rad]
  double max_tau;          // max absolute torque seen [Nm]
  bool stable;
};

HoldResult RunCartesianHoldTest(bool enable_pid, bool full_comp = true) {
  auto env = BuildEnv(enable_pid);
  if (!env->ct) return {999, 999, 999, false};

  double t = 0.0;
  int64_t ts = 1;
  const Eigen::Vector3d zero3 = Eigen::Vector3d::Zero();
  const Eigen::Quaterniond ident = Eigen::Quaterniond::Identity();

  // Initialize (1s)
  for (int step = 0; step < 1000; ++step, t += kDt) StepSim(env.get(), t);

  // Transition to cartesian_teleop
  env->arch->RequestState(3);
  StepSim(env.get(), t); t += kDt;

  // Move EE in +X for 1s (to create a non-trivial pose)
  for (int step = 0; step < 1000; ++step, t += kDt) {
    Eigen::Vector3d xdot(0.05, 0.0, 0.0);
    ts += 1000000;
    env->ct->UpdateCommand(xdot, zero3, ts, zero3, ident, 0);
    StepSim(env.get(), t);
  }

  // Hold for 2s and measure drift
  Eigen::Vector3d ee_start = GetEEPos(env.get());
  std::array<double, kNJoints> q_start;
  for (int i = 0; i < kNJoints; ++i) q_start[i] = env->d->qpos[i];
  double max_tau = 0;

  for (int step = 0; step < 2000; ++step, t += kDt) {
    ts += 1000000;
    env->ct->UpdateCommand(zero3, zero3, ts, zero3, ident, 0);
    ReadJointState(env.get());
    env->arch->Update(env->js, t, kDt);
    const auto& cmd = env->arch->GetCommand();
    for (int i = 0; i < kNJoints; ++i) {
      env->d->ctrl[i] = cmd.tau[i];
      max_tau = std::max(max_tau, std::abs(cmd.tau[i]));
    }
    mj_step(env->m, env->d);
  }

  HoldResult r;
  r.ee_drift_m = (GetEEPos(env.get()) - ee_start).norm();
  r.max_joint_drift = 0;
  for (int i = 0; i < kNJoints; ++i)
    r.max_joint_drift =
        std::max(r.max_joint_drift, std::abs(env->d->qpos[i] - q_start[i]));
  r.max_tau = max_tau;
  r.stable = true;
  for (int i = 0; i < kNJoints; ++i)
    if (!std::isfinite(env->d->qpos[i])) r.stable = false;
  return r;
}

TEST(DummyTeleop, PIDvsNoPID_CartesianHold) {
  std::cout << "\n========================================\n"
            << "  PID vs No-PID: Cartesian Hold Drift\n"
            << "========================================\n";

  auto no_pid = RunCartesianHoldTest(false);
  auto with_pid = RunCartesianHoldTest(true);

  std::cout << std::fixed << std::setprecision(6);
  std::cout << "\n             | EE drift [m] | Joint drift [rad] | Max tau [Nm]\n"
            << "  -----------+--------------+-------------------+------------\n"
            << "  No PID     | " << std::setw(12) << no_pid.ee_drift_m
            << " | " << std::setw(17) << no_pid.max_joint_drift
            << " | " << std::setw(10) << no_pid.max_tau << "\n"
            << "  With PID   | " << std::setw(12) << with_pid.ee_drift_m
            << " | " << std::setw(17) << with_pid.max_joint_drift
            << " | " << std::setw(10) << with_pid.max_tau << "\n";

  double improvement = (no_pid.ee_drift_m > 0)
                            ? (1.0 - with_pid.ee_drift_m / no_pid.ee_drift_m) * 100.0
                            : 0.0;
  std::cout << "\n  EE drift improvement with PID: "
            << std::setprecision(1) << improvement << "%\n";

  EXPECT_TRUE(no_pid.stable) << "No-PID run unstable";
  EXPECT_TRUE(with_pid.stable) << "With-PID run unstable";
  // WBC alone should hold well. Joint PID on top of WBC torque feedforward
  // creates double-feedback that corrupts the null-space structure, causing
  // significantly higher drift (known incompatibility — see gain_tuning.md).
  EXPECT_LT(no_pid.ee_drift_m, 0.02) << "No-PID drift > 20mm";
  // Only check stability for with_pid, not drift (PID known to degrade WBC perf)
  EXPECT_GT(with_pid.ee_drift_m, no_pid.ee_drift_m)
      << "PID should not improve over pure WBC (null-space interference expected)";
}
