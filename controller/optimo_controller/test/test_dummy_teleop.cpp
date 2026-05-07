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

#include "wbc_core/architecture/control_architecture.hpp"
#include "wbc_core/architecture/states/joint_teleop_state.hpp"
#include "wbc_core/architecture/states/cartesian_teleop_state.hpp"
#include "wbc_core/utils/ros_path_utils.hpp"

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
    << "    role: \"bias_task\"\n"
    << "    kp: [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]\n"
    << "    kd: [20.0,  20.0,  20.0,  20.0,  20.0,  20.0,  20.0]\n"
    << "    kp_ik: 1.0\n"
    << "\n"
    << "  - name: \"ee_pos_task\"\n"
    << "    type: \"LinkPosTask\"\n"
    << "    role: \"operational_task\"\n"
    << "    target_frame: \"optimo_end_effector\"\n"
    << "    kp: [3200.0, 3200.0, 3200.0, 3200.0, 3200.0, 3200.0]\n"
    << "    kd: [113.0,  113.0,  113.0,  113.0,  113.0,  113.0]\n"
    << "    kp_ik: 1.0\n"
    << "\n"
    << "  - name: \"ee_ori_task\"\n"
    << "    type: \"LinkOriTask\"\n"
    << "    role: \"operational_task\"\n"
    << "    target_frame: \"optimo_end_effector\"\n"
    << "    kp: [3200.0, 3200.0, 3200.0, 3200.0, 3200.0, 3200.0]\n"
    << "    kd: [113.0,  113.0,  113.0,  113.0,  113.0,  113.0]\n"
    << "    kp_ik: 1.0\n";
}

void WriteWbcYaml(const std::filesystem::path& dir) {
  std::ofstream f(dir / "optimo_wbc.yaml");
  f << "robot_model:\n"
    << "  urdf_path: \"package://optimo_description/urdf/optimo.urdf\"\n"
    << "  is_floating_base: false\n"
    << "\n"
    << "controller:\n"
    << "  ik_method: \"weighted_qp\"\n"
    << "  kp_acc: 120.0\n"
    << "  kd_acc: 22.0\n"
    << "\n"
    << "regularization:\n"
    << "  w_qddot: 0.01\n"
    << "  w_tau: 0.0\n"
    << "  w_tau_dot: 0.0\n"
    << "  w_rf: 1.0e-4\n"
    << "  w_xc_ddot: 1.0e-3\n"
    << "  w_f_dot: 1.0e-3\n"
    << "\n"
    << "global_constraints:\n"
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
    << "    enabled: true\n"
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
    << "      stay_here: true\n"
    << "      target_jpos: [0.0, 3.14159, 0.0, 0.0, 0.0, 0.0, 0.0]\n"
    << "    tasks:\n"
    << "      - name: \"jpos_task\"\n"
    << "        weight: 100.0\n"
    << "      - name: \"ee_pos_task\"\n"
    << "        weight: 1.0e-6\n"
    << "      - name: \"ee_ori_task\"\n"
    << "        weight: 1.0e-6\n"
    << "\n"
    << "  - id: 1\n"
    << "    name: \"home\"\n"
    << "    type: \"initialize\"\n"
    << "    params:\n"
    << "      duration: 1.0\n"
    << "      stay_here: true\n"
    << "      target_jpos: [0.0, 3.14159, 0.0, 0.0, 0.0, 0.0, 0.0]\n"
    << "    tasks:\n"
    << "      - name: \"jpos_task\"\n"
    << "        weight: 100.0\n"
    << "      - name: \"ee_pos_task\"\n"
    << "        weight: 1.0e-6\n"
    << "      - name: \"ee_ori_task\"\n"
    << "        weight: 1.0e-6\n"
    << "\n"
    << "  - id: 2\n"
    << "    name: \"joint_teleop\"\n"
    << "    params:\n"
    << "      stay_here: true\n"
    << "      joint_vel_limit: [0.5, 0.5, 0.5, 0.5, 0.3, 0.3, 0.3]\n"
    << "    tasks:\n"
    << "      - name: \"jpos_task\"\n"
    << "        weight: 100.0\n"
    << "      - name: \"ee_pos_task\"\n"
    << "        weight: 1.0e-6\n"
    << "      - name: \"ee_ori_task\"\n"
    << "        weight: 1.0e-6\n"
    << "\n"
    << "  - id: 3\n"
    << "    name: \"cartesian_teleop\"\n"
    << "    params:\n"
    << "      stay_here: true\n"
    << "      preview_time: 0.02\n"
    << "      manipulability:\n"
    << "        gain: 0.0\n"
    << "    tasks:\n"
    << "      - name: \"ee_pos_task\"\n"
    << "        weight: 100.0\n"
    << "      - name: \"ee_ori_task\"\n"
    << "        weight: 100.0\n"
    << "      - name: \"jpos_task\"\n"
    << "        weight: 1.0\n";
}

// ── Sim environment ─────────────────────────────────────────────────────────

struct SimEnv {
  std::filesystem::path tmp_dir;
  std::unique_ptr<wbc::ControlArchitecture> arch;
  mjModel* m{nullptr};
  mjData* d{nullptr};
  wbc::RobotJointState js;
  wbc::JointTeleopState* jt{nullptr};
  wbc::CartesianTeleopState* ct{nullptr};

  ~SimEnv() {
    if (d) mj_deleteData(d);
    if (m) mj_deleteModel(m);
    if (std::filesystem::exists(tmp_dir))
      std::filesystem::remove_all(tmp_dir);
  }
};

std::unique_ptr<SimEnv> BuildEnv() {
  auto env = std::make_unique<SimEnv>();
  env->tmp_dir = std::filesystem::temp_directory_path() / "wbc_dummy_teleop";
  std::filesystem::create_directories(env->tmp_dir);

  WriteTaskYaml(env->tmp_dir);
  WriteWbcYaml(env->tmp_dir);
  WriteStateMachineYaml(env->tmp_dir);

  std::string yaml_path = (env->tmp_dir / "optimo_wbc.yaml").string();
  std::string urdf_path = wbc::path::ResolvePackageUri(
      "package://optimo_description/urdf/optimo.urdf");
  std::string pkg_root = wbc::path::ResolveUrdfPackageRoot(
      "package://optimo_description/urdf/optimo.urdf", urdf_path);

  env->arch = std::make_unique<wbc::ControlArchitecture>(
      yaml_path, urdf_path, std::vector<std::string>{pkg_root});
  env->arch->Initialize();
  env->arch->setTimingEnabled(true);

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
  auto* fsm = env->arch->fsmHandler();
  env->jt = dynamic_cast<wbc::JointTeleopState*>(
      fsm->states().at(2).get());
  env->ct = dynamic_cast<wbc::CartesianTeleopState*>(
      fsm->states().at(3).get());

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
  const auto& cmd = env->arch->command();
  for (int i = 0; i < kNJoints; ++i) env->d->ctrl[i] = cmd.tau[i];
  mj_step(env->m, env->d);
}

Eigen::Vector3d GetEEPos(SimEnv* env) {
  const auto& model = env->arch->robot()->model();
  const auto& data = env->arch->solver()->data();
  auto fid = model.getFrameId("optimo_end_effector");
  return data.oMf[fid].translation();
}

Eigen::Quaterniond GetEEQuat(SimEnv* env) {
  const auto& model = env->arch->robot()->model();
  const auto& data = env->arch->solver()->data();
  auto fid = model.getFrameId("optimo_end_effector");
  return Eigen::Quaterniond(data.oMf[fid].rotation());
}

/// Angle between two quaternions in radians (always positive, 0..π).
double QuatAngle(const Eigen::Quaterniond& a, const Eigen::Quaterniond& b) {
  double dot = std::abs(a.coeffs().dot(b.coeffs()));
  dot = std::min(dot, 1.0);
  return 2.0 * std::acos(dot);
}

/// Convert quaternion to roll-pitch-yaw (ZYX convention) for display.
Eigen::Vector3d QuatToRPY(const Eigen::Quaterniond& q) {
  Eigen::Matrix3d R = q.toRotationMatrix();
  return R.eulerAngles(2, 1, 0).reverse();  // ZYX → [roll, pitch, yaw]
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

/// Run init (1s) → home (5s) to reach bent config before teleop.
/// Returns the current sim time after homing.
double RunInitAndHome(SimEnv* env) {
  double t = 0.0;
  // Initialize (1s)
  for (int step = 0; step < 1000; ++step, t += kDt) StepSim(env, t);
  // Transition to home
  env->arch->RequestState(1);
  // Home (2s) — straight home converges in under 1s
  for (int step = 0; step < 2000; ++step, t += kDt) StepSim(env, t);
  return t;
}

// ── Parameterized config for stability comparison test ─────────────────────

struct StabilityTestConfig {
  std::string label;
  std::string home_target;       // YAML array string
  double ee_kd;                  // EE task kd gain
  bool trq_limits;               // enable JointTrqLimitConstraint
};

void WriteTaskYamlCustom(const std::filesystem::path& dir, double ee_kd) {
  std::ofstream f(dir / "task_list.yaml");
  f << "task_pool:\n"
    << "  - name: \"jpos_task\"\n"
    << "    type: \"JointTask\"\n"
    << "    role: \"bias_task\"\n"
    << "    kp: [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]\n"
    << "    kd: [20.0,  20.0,  20.0,  20.0,  20.0,  20.0,  20.0]\n"
    << "    kp_ik: 1.0\n"
    << "\n"
    << "  - name: \"ee_pos_task\"\n"
    << "    type: \"LinkPosTask\"\n"
    << "    role: \"operational_task\"\n"
    << "    target_frame: \"optimo_end_effector\"\n"
    << "    kp: [3200.0, 3200.0, 3200.0, 3200.0, 3200.0, 3200.0]\n"
    << "    kd: [" << ee_kd << ", " << ee_kd << ", " << ee_kd
    << ", " << ee_kd << ", " << ee_kd << ", " << ee_kd << "]\n"
    << "    kp_ik: 1.0\n"
    << "\n"
    << "  - name: \"ee_ori_task\"\n"
    << "    type: \"LinkOriTask\"\n"
    << "    role: \"operational_task\"\n"
    << "    target_frame: \"optimo_end_effector\"\n"
    << "    kp: [3200.0, 3200.0, 3200.0, 3200.0, 3200.0, 3200.0]\n"
    << "    kd: [" << ee_kd << ", " << ee_kd << ", " << ee_kd
    << ", " << ee_kd << ", " << ee_kd << ", " << ee_kd << "]\n"
    << "    kp_ik: 1.0\n";
}

void WriteWbcYamlCustom(const std::filesystem::path& dir, bool trq_limits) {
  std::ofstream f(dir / "optimo_wbc.yaml");
  f << "robot_model:\n"
    << "  urdf_path: \"package://optimo_description/urdf/optimo.urdf\"\n"
    << "  is_floating_base: false\n"
    << "\n"
    << "controller:\n"
    << "  ik_method: \"weighted_qp\"\n"
    << "  kp_acc: 120.0\n"
    << "  kd_acc: 22.0\n"
    << "\n"
    << "regularization:\n"
    << "  w_qddot: 0.01\n"
    << "  w_tau: 0.0\n"
    << "  w_tau_dot: 0.0\n"
    << "  w_rf: 1.0e-4\n"
    << "  w_xc_ddot: 1.0e-3\n"
    << "  w_f_dot: 1.0e-3\n"
    << "\n"
    << "global_constraints:\n"
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
    << "    enabled: " << (trq_limits ? "true" : "false") << "\n"
    << "\n"
    << "task_pool_yaml: \"task_list.yaml\"\n"
    << "state_machine_yaml: \"state_machine.yaml\"\n";
}

void WriteStateMachineYamlCustom(const std::filesystem::path& dir,
                                 const std::string& home_target) {
  std::ofstream f(dir / "state_machine.yaml");
  f << "state_machine:\n"
    << "  - id: 0\n"
    << "    name: \"initialize\"\n"
    << "    params:\n"
    << "      duration: 0.5\n"
    << "      stay_here: true\n"
    << "      target_jpos: [0.0, 3.14159, 0.0, 0.0, 0.0, 0.0, 0.0]\n"
    << "    tasks:\n"
    << "      - name: \"jpos_task\"\n"
    << "        weight: 100.0\n"
    << "      - name: \"ee_pos_task\"\n"
    << "        weight: 1.0e-6\n"
    << "      - name: \"ee_ori_task\"\n"
    << "        weight: 1.0e-6\n"
    << "\n"
    << "  - id: 1\n"
    << "    name: \"home\"\n"
    << "    type: \"initialize\"\n"
    << "    params:\n"
    << "      duration: 2.0\n"
    << "      stay_here: true\n"
    << "      target_jpos: " << home_target << "\n"
    << "    tasks:\n"
    << "      - name: \"jpos_task\"\n"
    << "        weight: 100.0\n"
    << "      - name: \"ee_pos_task\"\n"
    << "        weight: 1.0e-6\n"
    << "      - name: \"ee_ori_task\"\n"
    << "        weight: 1.0e-6\n"
    << "\n"
    << "  - id: 3\n"
    << "    name: \"cartesian_teleop\"\n"
    << "    params:\n"
    << "      stay_here: true\n"
    << "      preview_time: 0.02\n"
    << "      manipulability:\n"
    << "        gain: 0.0\n"
    << "    tasks:\n"
    << "      - name: \"ee_pos_task\"\n"
    << "        weight: 100.0\n"
    << "      - name: \"ee_ori_task\"\n"
    << "        weight: 100.0\n"
    << "      - name: \"jpos_task\"\n"
    << "        weight: 1.0\n";
}

std::unique_ptr<SimEnv> BuildEnvCustom(const StabilityTestConfig& cfg) {
  auto env = std::make_unique<SimEnv>();
  env->tmp_dir = std::filesystem::temp_directory_path() / "wbc_stability_test";
  std::filesystem::remove_all(env->tmp_dir);
  std::filesystem::create_directories(env->tmp_dir);

  WriteTaskYamlCustom(env->tmp_dir, cfg.ee_kd);
  WriteWbcYamlCustom(env->tmp_dir, cfg.trq_limits);
  WriteStateMachineYamlCustom(env->tmp_dir, cfg.home_target);

  std::string yaml_path = (env->tmp_dir / "optimo_wbc.yaml").string();
  std::string urdf_path = wbc::path::ResolvePackageUri(
      "package://optimo_description/urdf/optimo.urdf");
  std::string pkg_root = wbc::path::ResolveUrdfPackageRoot(
      "package://optimo_description/urdf/optimo.urdf", urdf_path);

  env->arch = std::make_unique<wbc::ControlArchitecture>(
      yaml_path, urdf_path, std::vector<std::string>{pkg_root});
  env->arch->Initialize();

  // Load MuJoCo
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

  auto* fsm = env->arch->fsmHandler();
  env->ct = dynamic_cast<wbc::CartesianTeleopState*>(
      fsm->states().at(3).get());

  return env;
}

/// Run init (1s) → home (10s) for bent config convergence.
double RunInitAndHomeLong(SimEnv* env) {
  double t = 0.0;
  for (int step = 0; step < 1000; ++step, t += kDt) StepSim(env, t);
  env->arch->RequestState(1);
  for (int step = 0; step < 10000; ++step, t += kDt) StepSim(env, t);
  return t;
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
    const auto& cmd = env->arch->command();
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
    const auto& cmd = env->arch->command();
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
  EXPECT_LT(hold_drift, 0.250) << "EE should hold within 250mm";

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

// =============================================================================
// Orientation Tracking: command angular velocity and verify EE rotates
// =============================================================================
TEST(DummyTeleop, OrientationTracking) {
  std::cout << "\n========================================\n"
            << "  Orientation Tracking Test\n"
            << "========================================\n";

  const Eigen::Vector3d zero3 = Eigen::Vector3d::Zero();
  const Eigen::Quaterniond ident = Eigen::Quaterniond::Identity();

  // ====== Test A: Angular velocity around Z axis (2s) from bent home ======
  {
    std::cout << "\n--- Test A: init→home→cartesian_teleop, wz=0.5 rad/s (2s) ---\n";
    auto env = BuildEnv();
    ASSERT_NE(env->ct, nullptr);
    double t = RunInitAndHome(env.get());
    PrintJointPos("After home", env->d);
    PrintEEPos("After home", GetEEPos(env.get()));

    env->arch->RequestState(3);
    StepSim(env.get(), t);
    t += kDt;

    Eigen::Quaterniond q_start = GetEEQuat(env.get());
    Eigen::Vector3d p_start = GetEEPos(env.get());
    std::cout << "Start RPY (deg): "
              << (QuatToRPY(q_start) * 180.0 / M_PI).transpose() << "\n";

    int64_t ts = 1;
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "  time | angle (deg) | pos_drift (mm) | tau_max\n"
              << "  -----+-------------+----------------+--------\n";

    for (int step = 0; step < 2000; ++step, t += kDt) {
      Eigen::Vector3d wdot(0.0, 0.0, 0.5);
      ts += 1000000;
      env->ct->UpdateCommand(zero3, wdot, ts, zero3, ident, 0);
      ReadJointState(env.get());
      env->arch->Update(env->js, t, kDt);
      const auto& cmd = env->arch->command();
      for (int i = 0; i < kNJoints; ++i) env->d->ctrl[i] = cmd.tau[i];

      if (step % 500 == 0 || step == 1999) {
        double angle = QuatAngle(q_start, GetEEQuat(env.get())) * 180.0 / M_PI;
        double pdrift = (GetEEPos(env.get()) - p_start).norm() * 1000.0;
        double tmax = 0;
        for (int i = 0; i < kNJoints; ++i)
          tmax = std::max(tmax, std::abs(cmd.tau[i]));
        std::cout << "  " << std::setw(4) << step * kDt << " | "
                  << std::setw(11) << angle << " | "
                  << std::setw(14) << pdrift << " | "
                  << std::setw(7) << tmax << "\n";
      }
      mj_step(env->m, env->d);
    }

    double z_angle = QuatAngle(q_start, GetEEQuat(env.get())) * 180.0 / M_PI;
    std::cout << "Z-axis rotation: " << z_angle << " deg (expected ~57 deg)\n";
    EXPECT_GT(z_angle, 10.0) << "EE should rotate at least 10 deg with wz=0.5 for 2s";

    // Hold (1s)
    std::cout << "\n--- Hold (1s) ---\n";
    Eigen::Quaterniond q_hold_start = GetEEQuat(env.get());
    Eigen::Vector3d p_hold_start = GetEEPos(env.get());
    for (int step = 0; step < 1000; ++step, t += kDt) {
      ts += 1000000;
      env->ct->UpdateCommand(zero3, zero3, ts, zero3, ident, 0);
      StepSim(env.get(), t);
    }
    double ori_drift = QuatAngle(q_hold_start, GetEEQuat(env.get())) * 180.0 / M_PI;
    double pos_drift = (GetEEPos(env.get()) - p_hold_start).norm() * 1000.0;
    std::cout << "Ori drift: " << ori_drift << " deg, Pos drift: " << pos_drift << " mm\n";
    EXPECT_LT(ori_drift, 5.0) << "Orientation hold drift should be < 5 deg";
  }

  // ====== Test B: Angular velocity around Y axis from bent home (2s) ======
  {
    std::cout << "\n--- Test B: init→home→cartesian_teleop, wy=0.3 rad/s (2s) ---\n";
    auto env = BuildEnv();
    ASSERT_NE(env->ct, nullptr);
    double t = RunInitAndHome(env.get());
    PrintJointPos("After home", env->d);

    env->arch->RequestState(3);
    StepSim(env.get(), t);
    t += kDt;

    Eigen::Quaterniond q_start = GetEEQuat(env.get());
    Eigen::Vector3d p_start = GetEEPos(env.get());

    int64_t ts = 1;
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "  time | angle (deg) | pos_drift (mm) | tau_max\n"
              << "  -----+-------------+----------------+--------\n";

    for (int step = 0; step < 2000; ++step, t += kDt) {
      Eigen::Vector3d wdot(0.0, 0.3, 0.0);
      ts += 1000000;
      env->ct->UpdateCommand(zero3, wdot, ts, zero3, ident, 0);
      ReadJointState(env.get());
      env->arch->Update(env->js, t, kDt);
      const auto& cmd = env->arch->command();
      for (int i = 0; i < kNJoints; ++i) env->d->ctrl[i] = cmd.tau[i];

      if (step % 500 == 0 || step == 1999) {
        double angle = QuatAngle(q_start, GetEEQuat(env.get())) * 180.0 / M_PI;
        double pdrift = (GetEEPos(env.get()) - p_start).norm() * 1000.0;
        double tmax = 0;
        for (int i = 0; i < kNJoints; ++i)
          tmax = std::max(tmax, std::abs(cmd.tau[i]));
        std::cout << "  " << std::setw(4) << step * kDt << " | "
                  << std::setw(11) << angle << " | "
                  << std::setw(14) << pdrift << " | "
                  << std::setw(7) << tmax << "\n";
      }
      mj_step(env->m, env->d);
    }

    double y_angle = QuatAngle(q_start, GetEEQuat(env.get())) * 180.0 / M_PI;
    std::cout << "Y-axis rotation: " << y_angle << " deg (expected ~34 deg)\n";
    EXPECT_GT(y_angle, 5.0) << "EE should rotate with wy=0.3 for 2s";
  }

  // ====== Test C: Absolute pose command — 15 deg rotation from bent home ======
  {
    std::cout << "\n--- Test C: init→home→cartesian_teleop, 15 deg pose cmd (2s) ---\n";
    auto env = BuildEnv();
    ASSERT_NE(env->ct, nullptr);
    double t = RunInitAndHome(env.get());

    env->arch->RequestState(3);
    StepSim(env.get(), t);
    t += kDt;

    Eigen::Quaterniond q_start = GetEEQuat(env.get());
    Eigen::Vector3d p_start = GetEEPos(env.get());

    // Target: 15 deg rotation around Z from current
    Eigen::Quaterniond target_quat =
        Eigen::Quaterniond(Eigen::AngleAxisd(15.0 * M_PI / 180.0,
                                             Eigen::Vector3d::UnitZ())) *
        q_start;
    target_quat.normalize();

    double angle_to_target = QuatAngle(q_start, target_quat) * 180.0 / M_PI;
    std::cout << "Target: " << angle_to_target << " deg rotation around Z\n";

    int64_t ts = 1;
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "  time | ori_err (deg) | pos_err (mm) | tau_max\n"
              << "  -----+---------------+--------------+--------\n";

    for (int step = 0; step < 2000; ++step, t += kDt) {
      ts += 1000000;
      env->ct->UpdateCommand(zero3, zero3, 0, p_start, target_quat, ts);
      ReadJointState(env.get());
      env->arch->Update(env->js, t, kDt);
      const auto& cmd = env->arch->command();
      for (int i = 0; i < kNJoints; ++i) env->d->ctrl[i] = cmd.tau[i];

      if (step % 500 == 0 || step == 1999) {
        double oerr = QuatAngle(GetEEQuat(env.get()), target_quat) * 180.0 / M_PI;
        double perr = (GetEEPos(env.get()) - p_start).norm() * 1000.0;
        double tmax = 0;
        for (int i = 0; i < kNJoints; ++i)
          tmax = std::max(tmax, std::abs(cmd.tau[i]));
        std::cout << "  " << std::setw(4) << step * kDt << " | "
                  << std::setw(13) << oerr << " | "
                  << std::setw(12) << perr << " | "
                  << std::setw(7) << tmax << "\n";
      }
      mj_step(env->m, env->d);
    }

    double final_ori_err = QuatAngle(GetEEQuat(env.get()), target_quat) * 180.0 / M_PI;
    double final_pos_err = (GetEEPos(env.get()) - p_start).norm() * 1000.0;
    std::cout << "Final ori error: " << final_ori_err << " deg\n";
    std::cout << "Final pos error: " << final_pos_err << " mm\n";
    EXPECT_LT(final_ori_err, 10.0)
        << "15-deg absolute pose command should converge within 10 deg";

    for (int i = 0; i < kNJoints; ++i) {
      EXPECT_TRUE(std::isfinite(env->d->qpos[i]))
          << "Joint " << i << " diverged";
    }
  }
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
  EXPECT_LT(drift, 0.100);

  // ── Stability check ───────────────────────────────────────────────────
  for (int i = 0; i < kNJoints; ++i) {
    EXPECT_TRUE(std::isfinite(env->d->qpos[i]));
    EXPECT_TRUE(std::isfinite(env->d->qvel[i]));
  }

  // Print timing stats
  const auto& stats = env->arch->timingStats();
  std::cout << "\n  WBC Timing (last tick):\n"
            << "    FindConfig:   " << stats.find_config_us << " us\n"
            << "    Kinematics:   " << stats.kinematics_us << " us\n"
            << "    MakeTorque:   " << stats.make_torque_us << " us\n"
            << "    Feedback:     " << stats.feedback_us << " us\n";

  std::cout << "\n  Full pipeline demo complete.\n";
}

// =============================================================================
// Bent Home Stability Comparison: test all 3 fix options
// =============================================================================

TEST(DummyTeleop, BentHomeStabilityComparison) {
  const std::string bent = "[0.0, 3.14159, 0.0, -1.5708, 0.0, -1.5708, 0.0]";
  const Eigen::Vector3d zero3 = Eigen::Vector3d::Zero();
  const Eigen::Quaterniond ident = Eigen::Quaterniond::Identity();

  std::vector<StabilityTestConfig> configs = {
    {"baseline (kd=113, no_trq)",   bent, 113.0, false},
    {"torque_limits",               bent, 113.0, true},
    {"low_kd=20",                   bent,  20.0, false},
    {"torque_limits + low_kd=20",   bent,  20.0, true},
  };

  std::cout << "\n========================================\n"
            << "  Bent Home Stability Comparison\n"
            << "  wz=0.5 rad/s for 2s, then hold 1s\n"
            << "========================================\n\n";

  std::cout << std::fixed << std::setprecision(1);
  std::cout << std::setw(40) << std::left << "config"
            << " | rot(deg) | hold_ori | hold_pos | tau_max | stable\n"
            << std::string(40, '-') << "-+----------+----------+----------+---------+-------\n";

  for (const auto& cfg : configs) {
    auto env = BuildEnvCustom(cfg);
    if (!env->ct) {
      std::cout << std::setw(40) << std::left << cfg.label
                << " | SKIP (no ct state)\n";
      continue;
    }

    double t = RunInitAndHomeLong(env.get());

    // Transition to cartesian_teleop (state 3)
    env->arch->RequestState(3);
    StepSim(env.get(), t);
    t += kDt;

    Eigen::Quaterniond q_start = GetEEQuat(env.get());
    Eigen::Vector3d p_start = GetEEPos(env.get());

    // Apply wz=0.5 for 2s
    int64_t ts = 1;
    double max_tau = 0;
    for (int step = 0; step < 2000; ++step, t += kDt) {
      Eigen::Vector3d wdot(0.0, 0.0, 0.5);
      ts += 1000000;
      env->ct->UpdateCommand(zero3, wdot, ts, zero3, ident, 0);
      ReadJointState(env.get());
      env->arch->Update(env->js, t, kDt);
      const auto& cmd = env->arch->command();
      for (int i = 0; i < kNJoints; ++i) {
        env->d->ctrl[i] = cmd.tau[i];
        max_tau = std::max(max_tau, std::abs(cmd.tau[i]));
      }
      mj_step(env->m, env->d);
    }

    double z_angle = QuatAngle(q_start, GetEEQuat(env.get())) * 180.0 / M_PI;

    // Hold 1s
    Eigen::Quaterniond q_hold = GetEEQuat(env.get());
    Eigen::Vector3d p_hold = GetEEPos(env.get());
    for (int step = 0; step < 1000; ++step, t += kDt) {
      ts += 1000000;
      env->ct->UpdateCommand(zero3, zero3, ts, zero3, ident, 0);
      StepSim(env.get(), t);
    }
    double hold_ori = QuatAngle(q_hold, GetEEQuat(env.get())) * 180.0 / M_PI;
    double hold_pos = (GetEEPos(env.get()) - p_hold).norm() * 1000.0;

    bool stable = (hold_ori < 10.0 && hold_pos < 100.0 && max_tau < 200.0);

    std::cout << std::setw(40) << std::left << cfg.label
              << " | " << std::setw(8) << std::right << z_angle
              << " | " << std::setw(8) << hold_ori
              << " | " << std::setw(8) << hold_pos
              << " | " << std::setw(7) << max_tau
              << " | " << (stable ? "YES" : "NO") << "\n";
  }
}
