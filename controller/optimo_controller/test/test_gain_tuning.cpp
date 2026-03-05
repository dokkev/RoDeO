/**
 * @file test_gain_tuning.cpp
 * @brief Headless MuJoCo + WBC gain tuning test.
 *
 * Loads the Optimo MuJoCo model, instantiates the WBC ControlArchitecture,
 * and runs a closed-loop sim to measure steady-state tracking error for
 * different task kp/kd gain configurations.
 */
#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <mujoco/mujoco.h>

#include "wbc_architecture/control_architecture.hpp"
#include "wbc_formulation/interface/task.hpp"
#include "wbc_formulation/wbc_formulation.hpp"
#include "wbc_robot_system/pinocchio_robot_system.hpp"

namespace {

// Optimo has 7 actuated joints, no floating base.
constexpr int kNJoints = 7;
constexpr double kDt = 0.001;  // Must match MuJoCo timestep and WBC dt.

// Home position from keyframe.
const std::array<double, kNJoints> kHomeQpos = {
  0.0, 3.14159, 0.0, -1.5708, 0.0, -1.5708, 0.0
};

// Resolve package:// path to absolute filesystem path.
std::string ResolvePackagePath(const std::string& pkg_name,
                               const std::string& rel_path) {
  // Walk up from the test binary to find the install dir.
  // We rely on ament_index at runtime, but for this test we use env.
  const char* prefix = std::getenv("AMENT_PREFIX_PATH");
  if (!prefix) {
    throw std::runtime_error("AMENT_PREFIX_PATH not set. Source install/setup.bash first.");
  }
  // Search each path in AMENT_PREFIX_PATH
  std::string paths(prefix);
  std::istringstream ss(paths);
  std::string token;
  while (std::getline(ss, token, ':')) {
    auto candidate = std::filesystem::path(token) / "share" / pkg_name;
    if (std::filesystem::exists(candidate)) {
      // For description packages, files are relative to share/pkg/
      auto full = candidate / rel_path;
      if (std::filesystem::exists(full)) return full.string();
    }
  }
  throw std::runtime_error("Cannot resolve package://" + pkg_name + "/" + rel_path);
}

struct SimResult {
  double max_abs_error;
  double sum_abs_error;
  std::array<double, kNJoints> per_joint_error;
  std::array<double, kNJoints> final_tau{};
  bool stable;  // no NaN or joint limit violation
};

// Write a temporary task_list.yaml with the given gains.
std::string WriteTaskYaml(const std::filesystem::path& dir,
                          const std::array<double, kNJoints>& kp,
                          const std::array<double, kNJoints>& kd) {
  auto path = dir / "task_list.yaml";
  std::ofstream f(path);

  auto arr = [](const std::array<double, kNJoints>& v) {
    std::ostringstream os;
    os << "[";
    for (int i = 0; i < kNJoints; ++i) {
      if (i) os << ", ";
      os << std::fixed << std::setprecision(1) << v[i];
    }
    os << "]";
    return os.str();
  };

  f << "task_pool:\n";
  f << "  - name: \"jpos_task\"\n";
  f << "    type: \"JointTask\"\n";
  f << "    kp: " << arr(kp) << "\n";
  f << "    kd: " << arr(kd) << "\n";
  f << "    kp_ik: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]\n";
  f << "\n";
  f << "  - name: \"ee_pos_task\"\n";
  f << "    type: \"LinkPosTask\"\n";
  f << "    target_frame: \"end_effector\"\n";
  f << "    reference_frame: \"base_link\"\n";
  f << "    kp: [100.0, 100.0, 100.0]\n";
  f << "    kd: [20.0, 20.0, 20.0]\n";
  f << "    kp_ik: [1.0, 1.0, 1.0]\n";
  f << "\n";
  f << "  - name: \"ee_ori_task\"\n";
  f << "    type: \"LinkOriTask\"\n";
  f << "    target_frame: \"end_effector\"\n";
  f << "    reference_frame: \"base_link\"\n";
  f << "    kp: [100.0, 100.0, 100.0]\n";
  f << "    kd: [20.0, 20.0, 20.0]\n";
  f << "    kp_ik: [1.0, 1.0, 1.0]\n";

  f.close();
  return path.string();
}

struct ControllerFlags {
  bool gravity{true};
  bool coriolis{false};
  bool inertia{false};
  bool pid{false};
};

// Write the main WBC yaml referencing the temp task file.
std::string WriteWbcYaml(const std::filesystem::path& dir,
                         const ControllerFlags& flags = {}) {
  auto path = dir / "optimo_wbc.yaml";
  std::ofstream f(path);

  auto b = [](bool v) { return v ? "true" : "false"; };

  f << "robot_model:\n";
  f << "  urdf_path: \"package://optimo_description/urdf/optimo.urdf\"\n";
  f << "  is_floating_base: false\n";
  f << "  base_frame: \"base_link\"\n";
  f << "\n";
  f << "controller:\n";
  f << "  enable_gravity_compensation: " << b(flags.gravity) << "\n";
  f << "  enable_coriolis_compensation: " << b(flags.coriolis) << "\n";
  f << "  enable_inertia_compensation: " << b(flags.inertia) << "\n";
  f << "  joint_pid:\n";
  f << "    enabled: " << b(flags.pid) << "\n";
  f << "    gains_yaml: \"joint_pid_gains.yaml\"\n";
  f << "\n";
  f << "regularization:\n";
  f << "  w_qddot: 1.0e-6\n";
  f << "  w_tau: 0.0\n";
  f << "  w_tau_dot: 0.0\n";
  f << "  w_rf: 1.0e-4\n";
  f << "  w_xc_ddot: 1.0e-3\n";
  f << "  w_f_dot: 1.0e-3\n";
  f << "\n";
  f << "  JointPosLimitConstraint:\n";
  f << "    enabled: false\n";
  f << "  JointVelLimitConstraint:\n";
  f << "    enabled: false\n";
  f << "  JointTrqLimitConstraint:\n";
  f << "    enabled: false\n";
  f << "\n";
  f << "task_pool_yaml: \"task_list.yaml\"\n";
  f << "state_machine_yaml: \"state_machine.yaml\"\n";

  f.close();
  return path.string();
}

// Write a minimal state machine that goes to the target pose.
void WriteStateMachineYaml(const std::filesystem::path& dir,
                           const std::array<double, kNJoints>& target,
                           double duration = 0.5) {
  auto path = dir / "state_machine.yaml";
  std::ofstream f(path);

  f << "state_machine:\n";
  f << "  - id: 0\n";
  f << "    name: \"initialize\"\n";
  f << "    params:\n";
  f << "      duration: " << std::fixed << std::setprecision(4) << duration << "\n";
  f << "      wait_time: 0.0\n";
  f << "      stay_here: true\n";
  f << "      target_jpos: [";
  for (int i = 0; i < kNJoints; ++i) {
    if (i) f << ", ";
    f << std::fixed << std::setprecision(5) << target[i];
  }
  f << "]\n";
  f << "    task_hierarchy:\n";
  f << "      - name: \"jpos_task\"\n";

  f.close();
}

// Write joint_pid_gains.yaml (used only when PID is enabled).
void WritePidYaml(const std::filesystem::path& dir) {
  auto path = dir / "joint_pid_gains.yaml";
  std::ofstream f(path);
  f << "default:\n";
  f << "  kp_pos: 0.0\n";
  f << "  ki_pos: 0.0\n";
  f << "  kd_pos: 0.0\n";
  f.close();
}

struct SimConfig {
  double sim_duration_s{5.0};
  ControllerFlags flags{};
  bool mujoco_gravity{true};
  double traj_duration{0.5};
};

SimResult RunSim(const std::array<double, kNJoints>& kp,
                 const std::array<double, kNJoints>& kd,
                 const std::array<double, kNJoints>& target,
                 const SimConfig& cfg = {}) {
  // Create temp config directory.
  auto tmp_dir = std::filesystem::temp_directory_path() / "wbc_gain_tune";
  std::filesystem::create_directories(tmp_dir);

  WriteTaskYaml(tmp_dir, kp, kd);
  WriteWbcYaml(tmp_dir, cfg.flags);
  WriteStateMachineYaml(tmp_dir, target, cfg.traj_duration);
  WritePidYaml(tmp_dir);

  // Build WBC architecture.
  std::string yaml_path = (tmp_dir / "optimo_wbc.yaml").string();
  auto arch_config = wbc::ControlArchitectureConfig::FromYaml(yaml_path, kDt);
  arch_config.state_provider = std::make_unique<wbc::StateProvider>(kDt);
  auto arch = std::make_unique<wbc::ControlArchitecture>(std::move(arch_config));
  arch->Initialize();

  // Load MuJoCo model.
  std::string mjcf_path = ResolvePackagePath(
    "optimo_description", "mjcf/optimo.xml");

  char error[1000] = "";
  mjModel* m = mj_loadXML(mjcf_path.c_str(), nullptr, error, sizeof(error));
  if (!m) {
    std::cerr << "MuJoCo load error: " << error << std::endl;
    return {999.0, 999.0, {}, false};
  }
  // Optionally disable gravity in MuJoCo for diagnostic purposes.
  if (!cfg.mujoco_gravity) {
    m->opt.gravity[0] = 0.0;
    m->opt.gravity[1] = 0.0;
    m->opt.gravity[2] = 0.0;
  }

  mjData* d = mj_makeData(m);

  // Set initial pose to keyframe (spawn_home).
  if (m->nkey > 0) {
    mju_copy(d->qpos, m->key_qpos, m->nq);
  }
  mj_forward(m, d);

  SimResult result{};
  result.stable = true;

  wbc::RobotJointState joint_state;
  joint_state.Reset(kNJoints);

  const int n_steps = static_cast<int>(cfg.sim_duration_s / kDt);

  for (int step = 0; step < n_steps; ++step) {
    double t = step * kDt;

    // Read MuJoCo state into WBC joint state.
    for (int i = 0; i < kNJoints; ++i) {
      joint_state.q[i] = d->qpos[i];
      joint_state.qdot[i] = d->qvel[i];
      joint_state.tau[i] = d->qfrc_actuator[i];
    }

    // Check for NaN.
    for (int i = 0; i < kNJoints; ++i) {
      if (!std::isfinite(joint_state.q[i])) {
        result.stable = false;
        goto cleanup;
      }
    }

    // WBC update.
    arch->Update(joint_state, t, kDt);
    const auto& cmd = arch->GetCommand();

    // Apply torques to MuJoCo actuators.
    for (int i = 0; i < kNJoints; ++i) {
      d->ctrl[i] = cmd.tau[i];
    }

    // Step MuJoCo physics.
    mj_step(m, d);
  }

  // Compute steady-state error (use last state).
  result.max_abs_error = 0.0;
  result.sum_abs_error = 0.0;
  for (int i = 0; i < kNJoints; ++i) {
    double err = std::abs(d->qpos[i] - target[i]);
    result.per_joint_error[i] = err;
    result.max_abs_error = std::max(result.max_abs_error, err);
    result.sum_abs_error += err;
  }
  // Store final torque commands for diagnosis.
  {
    const auto& cmd = arch->GetCommand();
    for (int i = 0; i < kNJoints; ++i) {
      result.final_tau[i] = cmd.tau[i];
    }
  }

cleanup:
  mj_deleteData(d);
  mj_deleteModel(m);
  std::filesystem::remove_all(tmp_dir);
  return result;
}

void PrintResult(const std::string& label,
                 const std::array<double, kNJoints>& kp,
                 const std::array<double, kNJoints>& kd,
                 const SimResult& r) {
  std::cout << std::fixed << std::setprecision(5);
  std::cout << label
            << "  kp=" << kp[0] << "/" << kp[4]
            << "  kd=" << kd[0] << "/" << kd[4];
  if (!r.stable) {
    std::cout << "  UNSTABLE\n";
    return;
  }
  std::cout << "  max_err=" << r.max_abs_error
            << "  sum_err=" << r.sum_abs_error
            << "  per_joint=[";
  for (int i = 0; i < kNJoints; ++i) {
    if (i) std::cout << ", ";
    std::cout << r.per_joint_error[i];
  }
  std::cout << "]\n";
}

// Uniform gains (same for all joints).
std::array<double, kNJoints> Uniform(double v) {
  std::array<double, kNJoints> a;
  a.fill(v);
  return a;
}

// Per-joint gains: first 4 (large joints) get kp_big, last 3 (wrist) get kp_small.
std::array<double, kNJoints> Split(double big, double small) {
  return {big, big, big, big, small, small, small};
}

}  // namespace

// Step-by-step diagnostic: track how error develops over time.
TEST(GainTuning, StepByStepDiag) {
  std::cout << "\n===== Step-by-Step Diagnostic =====\n";

  auto tmp_dir = std::filesystem::temp_directory_path() / "wbc_step_diag";
  std::filesystem::create_directories(tmp_dir);

  auto kp = Uniform(400.0);
  auto kd = Uniform(40.0);
  WriteTaskYaml(tmp_dir, kp, kd);
  WriteWbcYaml(tmp_dir);
  WriteStateMachineYaml(tmp_dir, kHomeQpos, 0.5);
  WritePidYaml(tmp_dir);

  std::string yaml_path = (tmp_dir / "optimo_wbc.yaml").string();
  auto arch_config = wbc::ControlArchitectureConfig::FromYaml(yaml_path, kDt);
  arch_config.state_provider = std::make_unique<wbc::StateProvider>(kDt);
  auto arch = std::make_unique<wbc::ControlArchitecture>(std::move(arch_config));
  arch->Initialize();

  std::string mjcf_path = ResolvePackagePath("optimo_description", "mjcf/optimo.xml");
  char error[1000] = "";
  mjModel* m = mj_loadXML(mjcf_path.c_str(), nullptr, error, sizeof(error));
  ASSERT_NE(m, nullptr);
  mjData* d = mj_makeData(m);

  if (m->nkey > 0) {
    mju_copy(d->qpos, m->key_qpos, m->nq);
  }
  mj_forward(m, d);

  wbc::RobotJointState js;
  js.Reset(kNJoints);

  std::cout << std::fixed << std::setprecision(6);
  std::cout << "step | time    | qpos[0]     | qpos[1]     | qpos[2]     | tau[0]      | tau[1]      | tau[2]\n";
  std::cout << "-----+---------+-------------+-------------+-------------+-------------+-------------+-------\n";

  const int n_steps = 5000;  // 5 seconds
  for (int step = 0; step < n_steps; ++step) {
    double t = step * kDt;

    for (int i = 0; i < kNJoints; ++i) {
      js.q[i] = d->qpos[i];
      js.qdot[i] = d->qvel[i];
    }

    arch->Update(js, t, kDt);
    const auto& cmd = arch->GetCommand();

    for (int i = 0; i < kNJoints; ++i) {
      d->ctrl[i] = cmd.tau[i];
    }

    // Print at step 0, 1, 2, 5, 10, 50, 100, 500, 1000, 2000, 3000, 4000, 4999
    if (step == 0 || step == 1 || step == 2 || step == 5 || step == 10 ||
        step == 50 || step == 100 || step == 500 || step == 1000 ||
        step == 2000 || step == 3000 || step == 4000 || step == n_steps - 1) {
      std::cout << std::setw(4) << step << " | "
                << std::setw(7) << t << " | "
                << std::setw(11) << d->qpos[0] << " | "
                << std::setw(11) << d->qpos[1] << " | "
                << std::setw(11) << d->qpos[2] << " | "
                << std::setw(11) << cmd.tau[0] << " | "
                << std::setw(11) << cmd.tau[1] << " | "
                << std::setw(11) << cmd.tau[2] << "\n";
    }

    mj_step(m, d);
  }

  std::cout << "\nFinal qpos: [";
  for (int i = 0; i < kNJoints; ++i) {
    if (i) std::cout << ", ";
    std::cout << d->qpos[i];
  }
  std::cout << "]\n";
  std::cout << "Target:     [";
  for (int i = 0; i < kNJoints; ++i) {
    if (i) std::cout << ", ";
    std::cout << kHomeQpos[i];
  }
  std::cout << "]\n";

  mj_deleteData(d);
  mj_deleteModel(m);
  std::filesystem::remove_all(tmp_dir);
}

// Pure MuJoCo test: apply exact qfrc_bias as ctrl and check for drift.
TEST(GainTuning, PureMujocoGravComp) {
  std::cout << "\n===== Pure MuJoCo Gravity Compensation (no WBC) =====\n";

  std::string mjcf_path = ResolvePackagePath("optimo_description", "mjcf/optimo.xml");
  char error[1000] = "";
  mjModel* m = mj_loadXML(mjcf_path.c_str(), nullptr, error, sizeof(error));
  ASSERT_NE(m, nullptr);
  mjData* d = mj_makeData(m);

  if (m->nkey > 0) mju_copy(d->qpos, m->key_qpos, m->nq);
  mju_zero(d->qvel, m->nv);
  mju_zero(d->ctrl, m->nu);
  mj_forward(m, d);

  std::cout << std::fixed << std::setprecision(10);
  std::cout << "Initial qfrc_bias: [";
  for (int i = 0; i < kNJoints; ++i) {
    if (i) std::cout << ", ";
    std::cout << d->qfrc_bias[i];
  }
  std::cout << "]\n\n";

  // Set ctrl = qfrc_bias and recompute to get the true qacc.
  for (int i = 0; i < kNJoints; ++i) d->ctrl[i] = d->qfrc_bias[i];
  mj_forward(m, d);

  std::cout << "After ctrl=qfrc_bias, mj_forward:\n";
  auto print_vec = [&](const char* name, const mjtNum* v) {
    std::cout << "  " << name << ": [";
    for (int i = 0; i < kNJoints; ++i) {
      if (i) std::cout << ", ";
      std::cout << v[i];
    }
    std::cout << "]\n";
  };
  print_vec("qacc          ", d->qacc);
  print_vec("qfrc_actuator ", d->qfrc_actuator);
  print_vec("qfrc_bias     ", d->qfrc_bias);
  print_vec("qfrc_passive  ", d->qfrc_passive);
  print_vec("qfrc_constraint", d->qfrc_constraint);
  print_vec("qfrc_applied  ", d->qfrc_applied);
  print_vec("qfrc_inverse  ", d->qfrc_inverse);
  std::cout << "  ncon = " << d->ncon << "\n";
  for (int c = 0; c < d->ncon; ++c) {
    std::cout << "  contact[" << c << "]: geom1=" << d->contact[c].geom1
              << " geom2=" << d->contact[c].geom2
              << " dist=" << d->contact[c].dist
              << " pos=[" << d->contact[c].pos[0] << ", "
              << d->contact[c].pos[1] << ", "
              << d->contact[c].pos[2] << "]\n";
  }

  std::cout << "\nstep | qpos[1]           | qvel[1]          | max_err\n";

  for (int step = 0; step < 2000; ++step) {
    // Recompute bias at current state.
    mj_forward(m, d);
    for (int i = 0; i < kNJoints; ++i) {
      d->ctrl[i] = d->qfrc_bias[i];
    }

    if (step < 5 || step == 10 || step == 50 || step == 100 || step == 500 ||
        step == 1000 || step == 1999) {
      double max_err = 0;
      for (int i = 0; i < kNJoints; ++i) {
        max_err = std::max(max_err, std::abs(d->qpos[i] - kHomeQpos[i]));
      }
      std::cout << std::setw(4) << step
                << " | " << std::setw(17) << d->qpos[1]
                << " | " << std::setw(16) << d->qvel[1]
                << " | " << max_err << "\n";
    }

    mj_step(m, d);
  }

  std::cout << "\nFinal qpos: [";
  for (int i = 0; i < kNJoints; ++i) {
    if (i) std::cout << ", ";
    std::cout << d->qpos[i];
  }
  std::cout << "]\n";

  mj_deleteData(d);
  mj_deleteModel(m);
}

TEST(GainTuning, SweepUniformGains) {
  std::cout << "\n===== Uniform Gain Sweep (no joint PID) =====\n";

  struct Trial { double kp; double kd; };
  std::vector<Trial> trials = {
    {10.0,   1.0},    // baseline (original)
    {50.0,   10.0},
    {100.0,  20.0},
    {200.0,  28.0},   // ~critically damped: kd = 2*sqrt(kp)
    {400.0,  40.0},
    {600.0,  49.0},   // kd = 2*sqrt(kp)
    {800.0,  57.0},
    {1000.0, 63.0},
    {1500.0, 77.0},
    {2000.0, 89.0},
  };

  for (const auto& t : trials) {
    auto kp = Uniform(t.kp);
    auto kd = Uniform(t.kd);
    auto r = RunSim(kp, kd, kHomeQpos);
    PrintResult("uniform", kp, kd, r);
  }
}

TEST(GainTuning, SweepSplitGains) {
  std::cout << "\n===== Split Gain Sweep (big/wrist, no joint PID) =====\n";

  struct Trial { double kp_big; double kd_big; double kp_sm; double kd_sm; };
  std::vector<Trial> trials = {
    {400.0, 40.0, 100.0, 20.0},
    {600.0, 49.0, 150.0, 24.0},
    {800.0, 57.0, 200.0, 28.0},
    {1000.0, 63.0, 250.0, 31.0},
    {1500.0, 77.0, 400.0, 40.0},
    {2000.0, 89.0, 500.0, 45.0},
  };

  for (const auto& t : trials) {
    auto kp = Split(t.kp_big, t.kp_sm);
    auto kd = Split(t.kd_big, t.kd_sm);
    auto r = RunSim(kp, kd, kHomeQpos);
    PrintResult("split  ", kp, kd, r);
  }
}

TEST(GainTuning, DifferentTargetPoses) {
  std::cout << "\n===== Best gains across different poses =====\n";

  auto kp = Uniform(400.0);
  auto kd = Uniform(40.0);

  std::vector<std::pair<std::string, std::array<double, kNJoints>>> poses = {
    {"home       ", kHomeQpos},
    {"zeros      ", {0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0}},
    {"bent       ", {0.5, 2.5, 0.5, -1.0, 0.5, -1.0, 0.5}},
    {"stretched  ", {0.0, 2.0, 0.0, -0.5, 0.0, -0.5, 0.0}},
  };

  for (const auto& [name, target] : poses) {
    auto r = RunSim(kp, kd, target);
    PrintResult(name, kp, kd, r);
  }
}

TEST(GainTuning, TrajectoryDurationEffect) {
  std::cout << "\n===== Trajectory Duration Effect (kp=400, kd=40) =====\n";
  std::cout << "Testing if init trajectory causes the persistent error.\n\n";

  auto kp = Uniform(400.0);
  auto kd = Uniform(40.0);

  for (double dur : {0.001, 0.01, 0.1, 0.5, 1.0, 2.0}) {
    SimConfig cfg{.sim_duration_s = 5.0, .traj_duration = dur};
    auto r = RunSim(kp, kd, kHomeQpos, cfg);
    std::cout << "dur=" << std::fixed << std::setprecision(3) << dur << "  ";
    PrintResult("", kp, kd, r);
  }
}

TEST(GainTuning, DiagnoseFinal) {
  std::cout << "\n===== Final Diagnosis =====\n";
  std::cout << "MuJoCo actuator limits: [95, 95, 40, 40, 15, 15, 15] Nm\n\n";

  auto kp = Uniform(400.0);
  auto kd = Uniform(40.0);

  SimConfig cfg{.sim_duration_s = 10.0};
  auto r = RunSim(kp, kd, kHomeQpos, cfg);

  std::cout << std::fixed << std::setprecision(5);
  std::cout << "Joint |  target     | error [rad] | tau_cmd [Nm] | limit [Nm]\n";
  std::cout << "------+-------------+-------------+--------------+-----------\n";
  const double limits[] = {95.0, 95.0, 40.0, 40.0, 15.0, 15.0, 15.0};
  for (int i = 0; i < kNJoints; ++i) {
    std::cout << "  " << i
              << "   |  " << std::setw(9) << kHomeQpos[i]
              << "  |  " << r.per_joint_error[i]
              << "  |  " << std::setw(12) << r.final_tau[i]
              << "  |  " << std::setw(8) << limits[i] << "\n";
  }
}

// Compare Pinocchio gravity vs MuJoCo qfrc_bias at the SAME configuration.
TEST(GainTuning, GravityComparison) {
  std::cout << "\n===== Pinocchio vs MuJoCo Gravity Comparison =====\n";

  // --- MuJoCo side ---
  std::string mjcf_path = ResolvePackagePath(
    "optimo_description", "mjcf/optimo.xml");
  char error[1000] = "";
  mjModel* m = mj_loadXML(mjcf_path.c_str(), nullptr, error, sizeof(error));
  ASSERT_NE(m, nullptr) << "MuJoCo load error: " << error;
  mjData* d = mj_makeData(m);

  // Set to home keyframe.
  if (m->nkey > 0) {
    mju_copy(d->qpos, m->key_qpos, m->nq);
  }
  // Zero velocity.
  mju_zero(d->qvel, m->nv);
  mju_zero(d->ctrl, m->nu);
  mj_forward(m, d);

  std::cout << std::fixed << std::setprecision(6);
  std::cout << "\nMuJoCo qpos (home):  [";
  for (int i = 0; i < kNJoints; ++i) {
    if (i) std::cout << ", ";
    std::cout << d->qpos[i];
  }
  std::cout << "]\n";

  std::cout << "MuJoCo qfrc_bias:    [";
  for (int i = 0; i < kNJoints; ++i) {
    if (i) std::cout << ", ";
    std::cout << d->qfrc_bias[i];
  }
  std::cout << "]\n";

  // --- Pinocchio side (via WBC) ---
  auto tmp_dir = std::filesystem::temp_directory_path() / "wbc_grav_diag";
  std::filesystem::create_directories(tmp_dir);

  auto kp_arr = Uniform(100.0);
  auto kd_arr = Uniform(10.0);
  WriteTaskYaml(tmp_dir, kp_arr, kd_arr);
  WriteWbcYaml(tmp_dir);
  std::array<double, kNJoints> target = kHomeQpos;
  WriteStateMachineYaml(tmp_dir, target, 0.001);
  WritePidYaml(tmp_dir);

  std::string yaml_path = (tmp_dir / "optimo_wbc.yaml").string();
  auto arch_config = wbc::ControlArchitectureConfig::FromYaml(yaml_path, kDt);
  arch_config.state_provider = std::make_unique<wbc::StateProvider>(kDt);
  auto arch = std::make_unique<wbc::ControlArchitecture>(std::move(arch_config));
  arch->Initialize();

  // Feed the SAME joint angles as MuJoCo home config.
  wbc::RobotJointState js;
  js.Reset(kNJoints);
  for (int i = 0; i < kNJoints; ++i) {
    js.q[i] = d->qpos[i];
    js.qdot[i] = 0.0;
  }
  arch->Update(js, 0.0, kDt);
  // Do a second update so the trajectory finishes and we get steady-state.
  arch->Update(js, 1.0, kDt);

  // Get Pinocchio gravity from the robot system.
  auto* robot = arch->GetRobot();
  Eigen::VectorXd pin_grav = robot->GetGravity();
  Eigen::VectorXd pin_cori = robot->GetCoriolis();

  std::cout << "Pinocchio gravity:   [";
  for (int i = 0; i < kNJoints; ++i) {
    if (i) std::cout << ", ";
    std::cout << pin_grav[i];
  }
  std::cout << "]\n";

  std::cout << "Pinocchio coriolis:  [";
  for (int i = 0; i < kNJoints; ++i) {
    if (i) std::cout << ", ";
    std::cout << pin_cori[i];
  }
  std::cout << "]\n";

  std::cout << "Pinocchio q (internal): [";
  Eigen::VectorXd pin_q = robot->GetQ();
  for (int i = 0; i < pin_q.size(); ++i) {
    if (i) std::cout << ", ";
    std::cout << pin_q[i];
  }
  std::cout << "]\n";

  // Difference.
  std::cout << "\nGravity diff (Pinocchio - MuJoCo):\n";
  std::cout << "Joint |  Pinocchio  |   MuJoCo    |    diff\n";
  std::cout << "------+-------------+-------------+----------\n";
  for (int i = 0; i < kNJoints; ++i) {
    double diff = pin_grav[i] - d->qfrc_bias[i];
    std::cout << "  " << i
              << "   |  " << std::setw(10) << pin_grav[i]
              << "  |  " << std::setw(10) << d->qfrc_bias[i]
              << "  |  " << std::setw(10) << diff << "\n";
  }

  // WBC command.
  const auto& cmd = arch->GetCommand();
  std::cout << "\nWBC command at home config:\n";
  std::cout << "  q_cmd:  [";
  for (int i = 0; i < kNJoints; ++i) {
    if (i) std::cout << ", ";
    std::cout << cmd.q[i];
  }
  std::cout << "]\n";
  std::cout << "  tau:    [";
  for (int i = 0; i < kNJoints; ++i) {
    if (i) std::cout << ", ";
    std::cout << cmd.tau[i];
  }
  std::cout << "]\n";
  std::cout << "  tau_ff: [";
  for (int i = 0; i < kNJoints; ++i) {
    if (i) std::cout << ", ";
    std::cout << cmd.tau_ff[i];
  }
  std::cout << "]\n";

  mj_deleteData(d);
  mj_deleteModel(m);
  std::filesystem::remove_all(tmp_dir);
}

// =============================================================================
// Multi-state YAML writer for testing state transitions, teleop, cartesian etc.
// =============================================================================

// Write a state machine YAML with multiple states (init→home→joint_teleop→cartesian_teleop).
void WriteMultiStateYaml(const std::filesystem::path& dir,
                         const std::array<double, kNJoints>& home_target,
                         double init_duration = 0.5,
                         double home_duration = 2.0) {
  auto path = dir / "state_machine.yaml";
  std::ofstream f(path);

  auto arr = [](const std::array<double, kNJoints>& v) {
    std::ostringstream os;
    os << "[";
    for (int i = 0; i < kNJoints; ++i) {
      if (i) os << ", ";
      os << std::fixed << std::setprecision(5) << v[i];
    }
    os << "]";
    return os.str();
  };

  f << "state_machine:\n";
  f << "  - id: 0\n";
  f << "    name: \"initialize\"\n";
  f << "    params:\n";
  f << "      duration: " << std::fixed << std::setprecision(4) << init_duration << "\n";
  f << "      wait_time: 0.0\n";
  f << "      stay_here: true\n";
  f << "      target_jpos: " << arr(home_target) << "\n";
  f << "    task_hierarchy:\n";
  f << "      - name: \"jpos_task\"\n";

  f << "  - id: 1\n";
  f << "    name: \"home\"\n";
  f << "    type: \"initialize\"\n";
  f << "    params:\n";
  f << "      duration: " << std::fixed << std::setprecision(4) << home_duration << "\n";
  f << "      wait_time: 0.0\n";
  f << "      stay_here: true\n";
  f << "      target_jpos: " << arr(home_target) << "\n";
  f << "    task_hierarchy:\n";
  f << "      - name: \"jpos_task\"\n";

  f << "  - id: 2\n";
  f << "    name: \"joint_teleop\"\n";
  f << "    params:\n";
  f << "      stay_here: true\n";
  f << "      joint_vel_limit: [0.5, 0.5, 0.5, 0.5, 0.3, 0.3, 0.3]\n";
  f << "    task_hierarchy:\n";
  f << "      - name: \"jpos_task\"\n";

  f << "  - id: 3\n";
  f << "    name: \"cartesian_teleop\"\n";
  f << "    params:\n";
  f << "      stay_here: true\n";
  f << "      linear_vel_max: 0.1\n";
  f << "      angular_vel_max: 0.5\n";
  f << "      manipulability:\n";
  f << "        step_size: 0.5\n";
  f << "        w_threshold: 0.01\n";
  f << "    task_hierarchy:\n";
  f << "      - name: \"ee_pos_task\"\n";
  f << "        priority: 0\n";
  f << "      - name: \"ee_ori_task\"\n";
  f << "        priority: 0\n";
  f << "      - name: \"jpos_task\"\n";
  f << "        priority: 1\n";

  f.close();
}

// Extended task YAML with configurable EE gains.
std::string WriteTaskYamlFull(const std::filesystem::path& dir,
                              const std::array<double, kNJoints>& kp,
                              const std::array<double, kNJoints>& kd,
                              double ee_pos_kp, double ee_pos_kd,
                              double ee_ori_kp, double ee_ori_kd) {
  auto path = dir / "task_list.yaml";
  std::ofstream f(path);

  auto arr = [](const std::array<double, kNJoints>& v) {
    std::ostringstream os;
    os << "[";
    for (int i = 0; i < kNJoints; ++i) {
      if (i) os << ", ";
      os << std::fixed << std::setprecision(1) << v[i];
    }
    os << "]";
    return os.str();
  };

  f << "task_pool:\n";
  f << "  - name: \"jpos_task\"\n";
  f << "    type: \"JointTask\"\n";
  f << "    kp: " << arr(kp) << "\n";
  f << "    kd: " << arr(kd) << "\n";
  f << "    kp_ik: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]\n";
  f << "\n";
  f << "  - name: \"ee_pos_task\"\n";
  f << "    type: \"LinkPosTask\"\n";
  f << "    target_frame: \"end_effector\"\n";
  f << "    reference_frame: \"base_link\"\n";
  f << "    kp: [" << ee_pos_kp << ", " << ee_pos_kp << ", " << ee_pos_kp << "]\n";
  f << "    kd: [" << ee_pos_kd << ", " << ee_pos_kd << ", " << ee_pos_kd << "]\n";
  f << "    kp_ik: [1.0, 1.0, 1.0]\n";
  f << "\n";
  f << "  - name: \"ee_ori_task\"\n";
  f << "    type: \"LinkOriTask\"\n";
  f << "    target_frame: \"end_effector\"\n";
  f << "    reference_frame: \"base_link\"\n";
  f << "    kp: [" << ee_ori_kp << ", " << ee_ori_kp << ", " << ee_ori_kp << "]\n";
  f << "    kd: [" << ee_ori_kd << ", " << ee_ori_kd << ", " << ee_ori_kd << "]\n";
  f << "    kp_ik: [1.0, 1.0, 1.0]\n";

  f.close();
  return path.string();
}

// Helper: build full WBC stack with multi-state config.
struct MultiStateEnv {
  std::filesystem::path tmp_dir;
  std::unique_ptr<wbc::ControlArchitecture> arch;
  mjModel* m{nullptr};
  mjData* d{nullptr};
  wbc::RobotJointState js;

  ~MultiStateEnv() {
    if (d) mj_deleteData(d);
    if (m) mj_deleteModel(m);
    if (std::filesystem::exists(tmp_dir))
      std::filesystem::remove_all(tmp_dir);
  }
};

std::unique_ptr<MultiStateEnv> BuildMultiStateEnv(
    const std::array<double, kNJoints>& jpos_kp,
    const std::array<double, kNJoints>& jpos_kd,
    const std::array<double, kNJoints>& home_target,
    double ee_pos_kp = 200.0, double ee_pos_kd = 28.0,
    double ee_ori_kp = 200.0, double ee_ori_kd = 28.0,
    double init_dur = 0.5, double home_dur = 2.0) {
  auto env = std::make_unique<MultiStateEnv>();
  env->tmp_dir = std::filesystem::temp_directory_path() / "wbc_multistate";
  std::filesystem::create_directories(env->tmp_dir);

  WriteTaskYamlFull(env->tmp_dir, jpos_kp, jpos_kd,
                    ee_pos_kp, ee_pos_kd, ee_ori_kp, ee_ori_kd);
  WriteWbcYaml(env->tmp_dir);
  WriteMultiStateYaml(env->tmp_dir, home_target, init_dur, home_dur);
  WritePidYaml(env->tmp_dir);

  std::string yaml_path = (env->tmp_dir / "optimo_wbc.yaml").string();
  auto arch_config = wbc::ControlArchitectureConfig::FromYaml(yaml_path, kDt);
  arch_config.state_provider = std::make_unique<wbc::StateProvider>(kDt);
  env->arch = std::make_unique<wbc::ControlArchitecture>(std::move(arch_config));
  env->arch->Initialize();

  std::string mjcf_path = ResolvePackagePath("optimo_description", "mjcf/optimo.xml");
  char error[1000] = "";
  env->m = mj_loadXML(mjcf_path.c_str(), nullptr, error, sizeof(error));
  if (!env->m) {
    throw std::runtime_error(std::string("MuJoCo load error: ") + error);
  }
  env->d = mj_makeData(env->m);

  if (env->m->nkey > 0) {
    mju_copy(env->d->qpos, env->m->key_qpos, env->m->nq);
  }
  mj_forward(env->m, env->d);

  env->js.Reset(kNJoints);
  return env;
}

void ReadJointState(MultiStateEnv* env) {
  for (int i = 0; i < kNJoints; ++i) {
    env->js.q[i] = env->d->qpos[i];
    env->js.qdot[i] = env->d->qvel[i];
    env->js.tau[i] = env->d->qfrc_actuator[i];
  }
}

void ApplyCommand(MultiStateEnv* env) {
  const auto& cmd = env->arch->GetCommand();
  for (int i = 0; i < kNJoints; ++i) {
    env->d->ctrl[i] = cmd.tau[i];
  }
}

#include "optimo_controller/state_machines/joint_teleop.hpp"
#include "optimo_controller/state_machines/cartesian_teleop.hpp"
#include "wbc_handlers/manipulability_handler.hpp"

// =============================================================================
// Test: Initialize state tracking from non-home start position
// =============================================================================
TEST(StateMachine, InitializeStateTracking) {
  std::cout << "\n===== Initialize State Tracking =====\n";

  auto kp = Uniform(200.0);
  auto kd = Uniform(28.0);

  std::array<double, kNJoints> start_pos = {0.0, 2.0, 0.0, -0.5, 0.0, -0.5, 0.0};

  auto env = BuildMultiStateEnv(kp, kd, kHomeQpos, 200.0, 28.0, 200.0, 28.0, 2.0);

  // Set MuJoCo to non-home start position
  for (int i = 0; i < kNJoints; ++i) env->d->qpos[i] = start_pos[i];
  mju_zero(env->d->qvel, env->m->nv);
  mj_forward(env->m, env->d);

  std::cout << std::fixed << std::setprecision(5);
  std::cout << "Start:  [";
  for (int i = 0; i < kNJoints; ++i) { if (i) std::cout << ", "; std::cout << start_pos[i]; }
  std::cout << "]\nTarget: [";
  for (int i = 0; i < kNJoints; ++i) { if (i) std::cout << ", "; std::cout << kHomeQpos[i]; }
  std::cout << "]\n\ntime   | max_err   | per_joint_err\n-------+-----------+------------------------------------------\n";

  const int n_steps = 5000;
  for (int step = 0; step < n_steps; ++step) {
    double t = step * kDt;
    ReadJointState(env.get());
    env->arch->Update(env->js, t, kDt);
    ApplyCommand(env.get());
    mj_step(env->m, env->d);

    if (step == 0 || step == 100 || step == 500 || step == 1000 ||
        step == 2000 || step == 3000 || step == n_steps - 1) {
      double max_err = 0;
      for (int i = 0; i < kNJoints; ++i)
        max_err = std::max(max_err, std::abs(env->d->qpos[i] - kHomeQpos[i]));
      std::cout << std::setw(6) << t << " | " << std::setw(9) << max_err << " | [";
      for (int i = 0; i < kNJoints; ++i) {
        if (i) std::cout << ", ";
        std::cout << std::setw(8) << (env->d->qpos[i] - kHomeQpos[i]);
      }
      std::cout << "]\n";
    }
  }

  double max_err = 0;
  for (int i = 0; i < kNJoints; ++i)
    max_err = std::max(max_err, std::abs(env->d->qpos[i] - kHomeQpos[i]));
  std::cout << "\nFinal max error: " << max_err << " rad\n";
  EXPECT_LT(max_err, 0.01) << "Initialize state should track to target within 0.01 rad";
}

// =============================================================================
// Test: Dynamic tracking — verify trajectory duration matches actual behavior
// =============================================================================
TEST(StateMachine, DynamicTrackingAndDuration) {
  std::cout << "\n===== Dynamic Tracking & Duration Verification =====\n";

  auto kp = Uniform(200.0);
  auto kd = Uniform(28.0);
  std::array<double, kNJoints> start_pos = {0.3, 2.5, 0.3, -1.0, 0.3, -1.0, 0.3};

  for (double traj_dur : {0.5, 1.0, 2.0, 3.0}) {
    auto env = BuildMultiStateEnv(kp, kd, kHomeQpos, 200.0, 28.0, 200.0, 28.0, traj_dur);
    for (int i = 0; i < kNJoints; ++i) env->d->qpos[i] = start_pos[i];
    mju_zero(env->d->qvel, env->m->nv);
    mj_forward(env->m, env->d);

    std::cout << "\n--- traj_duration = " << std::fixed << std::setprecision(1)
              << traj_dur << "s ---\ntime   | max_err   | joint1_err\n";

    double err_at_traj_end = -1;
    double max_tracking_err = 0;
    const int n_steps = static_cast<int>((traj_dur + 3.0) / kDt);

    for (int step = 0; step < n_steps; ++step) {
      double t = step * kDt;
      ReadJointState(env.get());
      env->arch->Update(env->js, t, kDt);
      ApplyCommand(env.get());
      mj_step(env->m, env->d);

      double max_err = 0;
      for (int i = 0; i < kNJoints; ++i)
        max_err = std::max(max_err, std::abs(env->d->qpos[i] - kHomeQpos[i]));
      max_tracking_err = std::max(max_tracking_err, max_err);

      if (std::abs(t - traj_dur) < kDt * 0.5) err_at_traj_end = max_err;

      if (step == 0 || std::abs(t - traj_dur * 0.25) < kDt * 0.5 ||
          std::abs(t - traj_dur * 0.5) < kDt * 0.5 ||
          std::abs(t - traj_dur * 0.75) < kDt * 0.5 ||
          std::abs(t - traj_dur) < kDt * 0.5 ||
          std::abs(t - (traj_dur + 1.0)) < kDt * 0.5 ||
          step == n_steps - 1) {
        std::cout << std::setw(6) << t << " | " << std::setw(9) << max_err
                  << " | " << std::setw(9) << (env->d->qpos[1] - kHomeQpos[1]) << "\n";
      }
    }
    double final_max_err = 0;
    for (int i = 0; i < kNJoints; ++i)
      final_max_err = std::max(final_max_err, std::abs(env->d->qpos[i] - kHomeQpos[i]));
    std::cout << "  err_at_traj_end=" << std::setprecision(5) << err_at_traj_end
              << "  final_err=" << final_max_err
              << "  worst_transient=" << max_tracking_err << "\n";
  }
}

// =============================================================================
// Test: Home state with different target configurations
// =============================================================================
TEST(StateMachine, HomeStateDifferentConfigs) {
  std::cout << "\n===== Home State — Different Target Configs =====\n";

  auto kp = Uniform(200.0);
  auto kd = Uniform(28.0);

  struct Pose { std::string name; std::array<double, kNJoints> start, target; };
  std::vector<Pose> poses = {
    {"home→home (zero-motion)", kHomeQpos, kHomeQpos},
    {"zeros→home", {0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0}, kHomeQpos},
    {"bent→home", {0.5, 2.5, 0.5, -1.0, 0.5, -1.0, 0.5}, kHomeQpos},
    {"home→bent", kHomeQpos, {0.5, 2.5, 0.5, -1.0, 0.5, -1.0, 0.5}},
    {"home→stretched", kHomeQpos, {0.0, 2.0, 0.0, -0.5, 0.0, -0.5, 0.0}},
  };

  for (const auto& p : poses) {
    auto env = BuildMultiStateEnv(kp, kd, p.target, 200.0, 28.0, 200.0, 28.0, 2.0);
    for (int i = 0; i < kNJoints; ++i) env->d->qpos[i] = p.start[i];
    mju_zero(env->d->qvel, env->m->nv);
    mj_forward(env->m, env->d);

    for (int step = 0; step < 5000; ++step) {
      ReadJointState(env.get());
      env->arch->Update(env->js, step * kDt, kDt);
      ApplyCommand(env.get());
      mj_step(env->m, env->d);
    }
    double max_err = 0;
    for (int i = 0; i < kNJoints; ++i)
      max_err = std::max(max_err, std::abs(env->d->qpos[i] - p.target[i]));
    std::cout << std::fixed << std::setprecision(6)
              << p.name << "  final_max_err=" << max_err << "\n";
    EXPECT_LT(max_err, 0.01) << "Config: " << p.name;
  }
}

// =============================================================================
// Test: Joint Teleop State
// =============================================================================
TEST(StateMachine, JointTeleopState) {
  std::cout << "\n===== Joint Teleop State =====\n";

  auto kp = Uniform(200.0);
  auto kd = Uniform(28.0);
  auto env = BuildMultiStateEnv(kp, kd, kHomeQpos);

  // Run init for 1s
  for (int step = 0; step < 1000; ++step) {
    ReadJointState(env.get());
    env->arch->Update(env->js, step * kDt, kDt);
    ApplyCommand(env.get());
    mj_step(env->m, env->d);
  }

  std::cout << std::fixed << std::setprecision(4) << "After init, qpos: [";
  for (int i = 0; i < kNJoints; ++i) { if (i) std::cout << ", "; std::cout << env->d->qpos[i]; }
  std::cout << "]\n";

  // Transition to joint_teleop (state 2)
  env->arch->RequestState(2);

  std::array<double, kNJoints> pos_before;
  for (int i = 0; i < kNJoints; ++i) pos_before[i] = env->d->qpos[i];

  auto* fsm = env->arch->GetFsmHandler();
  auto* jt = dynamic_cast<wbc::JointTeleop*>(fsm->FindStateById(2));
  ASSERT_NE(jt, nullptr) << "JointTeleop state not found";

  // Phase 1: no commands → watchdog fires → hold position (0.5s)
  std::cout << "\n--- Phase 1: hold (no commands, 0.5s) ---\n";
  Eigen::VectorXd dummy_pos = Eigen::VectorXd::Zero(kNJoints);
  for (int step = 0; step < 500; ++step) {
    ReadJointState(env.get());
    env->arch->Update(env->js, 1.0 + step * kDt, kDt);
    ApplyCommand(env.get());
    mj_step(env->m, env->d);
  }
  double max_drift = 0;
  for (int i = 0; i < kNJoints; ++i)
    max_drift = std::max(max_drift, std::abs(env->d->qpos[i] - pos_before[i]));
  std::cout << "Drift: " << std::setprecision(6) << max_drift << " rad\n";
  EXPECT_LT(max_drift, 0.02);

  // Phase 2: velocity on joint 0 (+0.3 rad/s, 1s)
  std::cout << "\n--- Phase 2: joint0 vel=+0.3 rad/s (1s) ---\n";
  double ref_j0 = env->d->qpos[0];
  Eigen::VectorXd vel_cmd = Eigen::VectorXd::Zero(kNJoints);
  vel_cmd[0] = 0.3;
  int64_t ts = 1;

  for (int step = 0; step < 1000; ++step) {
    ts += 1000000;
    jt->UpdateCommand(vel_cmd, ts, dummy_pos, 0);
    ReadJointState(env.get());
    env->arch->Update(env->js, 1.5 + step * kDt, kDt);
    ApplyCommand(env.get());
    mj_step(env->m, env->d);
  }
  double actual = env->d->qpos[0] - ref_j0;
  std::cout << "Expected delta=0.3, actual=" << std::setprecision(4) << actual
            << "  err=" << std::abs(actual - 0.3) << "\n";

  // Phase 3: stop, hold (0.5s)
  std::cout << "\n--- Phase 3: stop, hold (0.5s) ---\n";
  std::array<double, kNJoints> pos_after;
  for (int i = 0; i < kNJoints; ++i) pos_after[i] = env->d->qpos[i];
  vel_cmd.setZero();
  for (int step = 0; step < 500; ++step) {
    ts += 1000000;
    jt->UpdateCommand(vel_cmd, ts, dummy_pos, 0);
    ReadJointState(env.get());
    env->arch->Update(env->js, 2.5 + step * kDt, kDt);
    ApplyCommand(env.get());
    mj_step(env->m, env->d);
  }
  double hold_err = 0;
  for (int i = 0; i < kNJoints; ++i)
    hold_err = std::max(hold_err, std::abs(env->d->qpos[i] - pos_after[i]));
  std::cout << "Hold error: " << std::setprecision(6) << hold_err << "\n";
  EXPECT_LT(hold_err, 0.01);
}

// =============================================================================
// Test: Cartesian Teleop State
// =============================================================================
TEST(StateMachine, CartesianTeleopState) {
  std::cout << "\n===== Cartesian Teleop State =====\n";

  auto kp = Uniform(200.0);
  auto kd = Uniform(28.0);
  auto env = BuildMultiStateEnv(kp, kd, kHomeQpos, 200.0, 28.0, 200.0, 28.0);

  // Run init for 1s
  for (int step = 0; step < 1000; ++step) {
    ReadJointState(env.get());
    env->arch->Update(env->js, step * kDt, kDt);
    ApplyCommand(env.get());
    mj_step(env->m, env->d);
  }

  auto* robot = env->arch->GetRobot();
  ReadJointState(env.get());
  env->arch->Update(env->js, 1.0, kDt);
  int ee_idx = robot->GetFrameIndex("end_effector");
  Eigen::Vector3d home_ee = robot->GetLinkIsometry(ee_idx).translation();
  Eigen::Quaterniond home_quat(robot->GetLinkIsometry(ee_idx).rotation());

  std::cout << std::fixed << std::setprecision(5);
  std::cout << "Home EE pos: [" << home_ee.transpose() << "]\n";
  std::cout << "Home EE quat(xyzw): [" << home_quat.coeffs().transpose() << "]\n";

  // Transition to cartesian_teleop
  env->arch->RequestState(3);
  auto* ct = dynamic_cast<wbc::CartesianTeleop*>(
      env->arch->GetFsmHandler()->FindStateById(3));
  ASSERT_NE(ct, nullptr);

  // Phase 1: hold (1s)
  std::cout << "\n--- Phase 1: hold EE (1s) ---\n";
  for (int step = 0; step < 1000; ++step) {
    ReadJointState(env.get());
    env->arch->Update(env->js, 1.001 + step * kDt, kDt);
    ApplyCommand(env.get());
    mj_step(env->m, env->d);
  }
  ReadJointState(env.get());
  env->arch->Update(env->js, 2.001, kDt);
  Eigen::Vector3d ee_after_hold = robot->GetLinkIsometry(ee_idx).translation();
  std::cout << "EE drift: " << (ee_after_hold - home_ee).norm() << " m\n";

  // Phase 2: xdot = [0.05, 0, 0] for 1s
  std::cout << "\n--- Phase 2: xdot=[0.05,0,0] for 1s ---\n";
  Eigen::Vector3d xdot(0.05, 0.0, 0.0);
  Eigen::Vector3d zero3 = Eigen::Vector3d::Zero();
  Eigen::Quaterniond ident = Eigen::Quaterniond::Identity();
  int64_t ts = 1;
  Eigen::Vector3d ee_before_vel = ee_after_hold;

  for (int step = 0; step < 1000; ++step) {
    double t = 2.002 + step * kDt;
    ts += 1000000;
    ct->UpdateCommand(xdot, zero3, ts, zero3, ident, 0);
    ReadJointState(env.get());
    env->arch->Update(env->js, t, kDt);
    ApplyCommand(env.get());
    mj_step(env->m, env->d);

    if (step % 250 == 0 || step == 999) {
      ReadJointState(env.get());
      env->arch->Update(env->js, t + kDt, kDt);
      Eigen::Vector3d ee = robot->GetLinkIsometry(ee_idx).translation();
      std::cout << "  t=" << std::setw(5) << (t - 2.002) << "  ee=[" << ee.transpose() << "]\n";
    }
  }
  ReadJointState(env.get());
  env->arch->Update(env->js, 3.002, kDt);
  Eigen::Vector3d ee_after_vel = robot->GetLinkIsometry(ee_idx).translation();
  Eigen::Vector3d delta = ee_after_vel - ee_before_vel;
  std::cout << "EE delta: [" << delta.transpose() << "]\n";
  std::cout << "Expected x~0.05, actual x=" << delta.x() << "\n";

  // Phase 3: stop, hold 1s
  std::cout << "\n--- Phase 3: stop, hold (1s) ---\n";
  for (int step = 0; step < 1000; ++step) {
    double t = 3.003 + step * kDt;
    ts += 1000000;
    ct->UpdateCommand(zero3, zero3, ts, zero3, ident, 0);
    ReadJointState(env.get());
    env->arch->Update(env->js, t, kDt);
    ApplyCommand(env.get());
    mj_step(env->m, env->d);
  }
  ReadJointState(env.get());
  env->arch->Update(env->js, 4.003, kDt);
  Eigen::Vector3d ee_final = robot->GetLinkIsometry(ee_idx).translation();
  std::cout << "EE drift after stop: " << (ee_final - ee_after_vel).norm() << " m\n";
}

// =============================================================================
// Test: Cartesian teleop step-by-step diagnostics
// =============================================================================
TEST(StateMachine, CartesianTeleopDiag) {
  std::cout << "\n===== Cartesian Teleop Step-by-Step Diagnostics =====\n";

  auto kp = Uniform(200.0);
  auto kd = Uniform(28.0);
  auto env = BuildMultiStateEnv(kp, kd, kHomeQpos, 200.0, 28.0, 200.0, 28.0);

  // Init for 1s
  for (int step = 0; step < 1000; ++step) {
    ReadJointState(env.get());
    env->arch->Update(env->js, step * kDt, kDt);
    ApplyCommand(env.get());
    mj_step(env->m, env->d);
  }

  auto* robot = env->arch->GetRobot();
  int ee_idx = robot->GetFrameIndex("end_effector");

  // Transition to cartesian_teleop
  env->arch->RequestState(3);
  auto* ct = dynamic_cast<wbc::CartesianTeleop*>(
      env->arch->GetFsmHandler()->FindStateById(3));
  ASSERT_NE(ct, nullptr);

  // Run 1 tick with no commands to see initial state
  ReadJointState(env.get());
  env->arch->Update(env->js, 1.001, kDt);
  ApplyCommand(env.get());

  auto* ee_pos_task = env->arch->GetRobot();  // Get task from formulation
  // Print command torques
  const auto& cmd0 = env->arch->GetCommand();
  std::cout << std::fixed << std::setprecision(6);
  std::cout << "Tick 0 (transition, no vel cmd):\n";
  std::cout << "  EE pos: [" << robot->GetLinkIsometry(ee_idx).translation().transpose() << "]\n";
  std::cout << "  tau: [";
  for (int i = 0; i < kNJoints; ++i) { if (i) std::cout << ", "; std::cout << cmd0.tau[i]; }
  std::cout << "]\n";

  mj_step(env->m, env->d);

  // Now send velocity command and track step-by-step
  Eigen::Vector3d xdot(0.05, 0.0, 0.0);
  Eigen::Vector3d zero3 = Eigen::Vector3d::Zero();
  Eigen::Quaterniond ident = Eigen::Quaterniond::Identity();
  int64_t ts = 1;

  std::cout << "\nstep | ee_x       | ee_y       | ee_z       | tau[0]     | tau[1]     | tau[3]     | max_tau\n";

  for (int step = 0; step < 50; ++step) {
    double t = 1.002 + step * kDt;
    ts += 1000000;
    ct->UpdateCommand(xdot, zero3, ts, zero3, ident, 0);
    ReadJointState(env.get());
    env->arch->Update(env->js, t, kDt);
    const auto& cmd = env->arch->GetCommand();
    ApplyCommand(env.get());

    double max_tau = 0;
    for (int i = 0; i < kNJoints; ++i) max_tau = std::max(max_tau, std::abs(cmd.tau[i]));

    Eigen::Vector3d ee = robot->GetLinkIsometry(ee_idx).translation();
    std::cout << std::setw(4) << step << " | "
              << std::setw(10) << ee.x() << " | "
              << std::setw(10) << ee.y() << " | "
              << std::setw(10) << ee.z() << " | "
              << std::setw(10) << cmd.tau[0] << " | "
              << std::setw(10) << cmd.tau[1] << " | "
              << std::setw(10) << cmd.tau[3] << " | "
              << std::setw(10) << max_tau << "\n";

    mj_step(env->m, env->d);
  }

  // Print qpos for diagnosis
  std::cout << "\nFinal qpos: [";
  for (int i = 0; i < kNJoints; ++i) {
    if (i) std::cout << ", ";
    std::cout << env->d->qpos[i];
  }
  std::cout << "]\nHome qpos:  [";
  for (int i = 0; i < kNJoints; ++i) {
    if (i) std::cout << ", ";
    std::cout << kHomeQpos[i];
  }
  std::cout << "]\n";
}

// =============================================================================
// Test: Cartesian gain sweep — find stable gains for LinkPosTask/LinkOriTask
// =============================================================================
TEST(StateMachine, CartesianGainSweep) {
  std::cout << "\n===== Cartesian Gain Sweep =====\n";
  std::cout << "Testing ee_pos/ori gains while keeping jpos gains at kp=200,kd=28\n";
  std::cout << "Sending xdot=[0.05,0,0] for 1s then holding 1s\n\n";

  auto jpos_kp = Uniform(200.0);
  auto jpos_kd = Uniform(28.0);

  struct Trial { double kp; double kd; };
  std::vector<Trial> trials = {
    {10.0,   6.0},
    {25.0,  10.0},
    {50.0,  14.0},
    {100.0, 20.0},
    {150.0, 24.0},
    {200.0, 28.0},
    {400.0, 40.0},
  };

  std::cout << "ee_kp  | ee_kd | stable | ee_x_delta | ee_hold_drift | max_tau_peak\n";
  std::cout << "-------+-------+--------+------------+---------------+-------------\n";

  for (const auto& t : trials) {
    auto env = BuildMultiStateEnv(jpos_kp, jpos_kd, kHomeQpos,
                                  t.kp, t.kd, t.kp, t.kd);
    // Init for 1s
    for (int step = 0; step < 1000; ++step) {
      ReadJointState(env.get());
      env->arch->Update(env->js, step * kDt, kDt);
      ApplyCommand(env.get());
      mj_step(env->m, env->d);
    }

    auto* robot = env->arch->GetRobot();
    int ee_idx = robot->GetFrameIndex("end_effector");

    // Transition to cartesian_teleop
    env->arch->RequestState(3);
    auto* ct = dynamic_cast<wbc::CartesianTeleop*>(
        env->arch->GetFsmHandler()->FindStateById(3));

    // 1 transition tick
    ReadJointState(env.get());
    env->arch->Update(env->js, 1.0, kDt);
    ApplyCommand(env.get());
    mj_step(env->m, env->d);

    Eigen::Vector3d ee_before = robot->GetLinkIsometry(ee_idx).translation();

    // Send xdot=[0.05,0,0] for 1s
    Eigen::Vector3d xdot(0.05, 0.0, 0.0);
    Eigen::Vector3d zero3 = Eigen::Vector3d::Zero();
    Eigen::Quaterniond ident = Eigen::Quaterniond::Identity();
    int64_t ts = 1;
    double max_tau_peak = 0;
    bool stable = true;

    for (int step = 0; step < 1000; ++step) {
      double time = 1.001 + step * kDt;
      ts += 1000000;
      ct->UpdateCommand(xdot, zero3, ts, zero3, ident, 0);
      ReadJointState(env.get());
      env->arch->Update(env->js, time, kDt);
      const auto& cmd = env->arch->GetCommand();
      ApplyCommand(env.get());

      for (int i = 0; i < kNJoints; ++i) {
        max_tau_peak = std::max(max_tau_peak, std::abs(cmd.tau[i]));
        if (!std::isfinite(cmd.tau[i])) { stable = false; break; }
      }
      mj_step(env->m, env->d);
    }

    ReadJointState(env.get());
    env->arch->Update(env->js, 2.001, kDt);
    Eigen::Vector3d ee_after_vel = robot->GetLinkIsometry(ee_idx).translation();
    double x_delta = ee_after_vel.x() - ee_before.x();

    // Hold for 1s
    for (int step = 0; step < 1000; ++step) {
      double time = 2.002 + step * kDt;
      ts += 1000000;
      ct->UpdateCommand(zero3, zero3, ts, zero3, ident, 0);
      ReadJointState(env.get());
      env->arch->Update(env->js, time, kDt);
      ApplyCommand(env.get());
      mj_step(env->m, env->d);
    }
    ReadJointState(env.get());
    env->arch->Update(env->js, 3.002, kDt);
    Eigen::Vector3d ee_final = robot->GetLinkIsometry(ee_idx).translation();
    double hold_drift = (ee_final - ee_after_vel).norm();

    // Check stability: if EE moved more than 1m total, consider unstable
    double total_ee_movement = (ee_final - ee_before).norm();
    if (total_ee_movement > 0.5) stable = false;

    std::cout << std::fixed << std::setprecision(1)
              << std::setw(6) << t.kp << " | "
              << std::setw(5) << t.kd << " | "
              << (stable ? "  yes  " : "  NO   ") << " | "
              << std::setprecision(4)
              << std::setw(10) << x_delta << " | "
              << std::setw(13) << hold_drift << " | "
              << std::setw(11) << max_tau_peak << "\n";
  }
}

// =============================================================================
// Test: Manipulability Handler diagnostics
// =============================================================================
TEST(StateMachine, ManipulabilityDiagnostics) {
  std::cout << "\n===== Manipulability Handler Diagnostics =====\n";

  auto kp = Uniform(200.0);
  auto kd = Uniform(28.0);
  auto env = BuildMultiStateEnv(kp, kd, kHomeQpos, 200.0, 28.0, 200.0, 28.0);

  for (int step = 0; step < 1000; ++step) {
    ReadJointState(env.get());
    env->arch->Update(env->js, step * kDt, kDt);
    ApplyCommand(env.get());
    mj_step(env->m, env->d);
  }

  auto* robot = env->arch->GetRobot();
  int ee_idx = robot->GetFrameIndex("end_effector");

  std::cout << std::fixed << std::setprecision(6);
  std::cout << "Config                              | w (manipulability)\n";
  std::cout << "------------------------------------+-------------------\n";

  struct Cfg { std::string name; std::array<double, kNJoints> q; };
  std::vector<Cfg> configs = {
    {"home",             kHomeQpos},
    {"zeros(min j2)",    {0.0, 1.047, 0.0, 0.0, 0.0, 0.0, 0.0}},
    {"stretched",        {0.0, 2.0, 0.0, -0.5, 0.0, -0.5, 0.0}},
    {"near_singular(j4=0)", {0.0, 3.14159, 0.0, 0.0, 0.0, 0.0, 0.0}},
    {"fully extended",   {0.0, 3.14159, 0.0, -3.14, 0.0, 0.0, 0.0}},
    {"bent",             {0.5, 2.5, 0.5, -1.0, 0.5, -1.0, 0.5}},
  };

  for (const auto& c : configs) {
    Eigen::VectorXd q(kNJoints);
    for (int i = 0; i < kNJoints; ++i) q[i] = c.q[i];
    double w = robot->ComputeManipulability(ee_idx, q);
    std::cout << std::left << std::setw(35) << c.name << " | " << w << "\n";
  }

  // Test handler at home
  std::cout << "\n--- Handler at home config ---\n";
  wbc::ManipulabilityHandler manip;
  wbc::ManipulabilityHandler::Config mcfg{0.5, 0.01};
  manip.Init(robot, ee_idx, mcfg);
  for (int i = 0; i < kNJoints; ++i) manip.Update(kDt);
  std::cout << "w=" << manip.manipulability() << "  active=" << manip.is_active()
            << "  avoid=[" << manip.avoidance_velocity().transpose() << "]\n";

  // Test at near-singular
  std::cout << "\n--- Handler at near-singular config ---\n";
  Eigen::Vector3d z3 = Eigen::Vector3d::Zero();
  Eigen::Quaterniond iq = Eigen::Quaterniond::Identity();
  Eigen::VectorXd sing_q(kNJoints), sing_v(kNJoints);
  sing_q << 0.0, 3.14159, 0.0, 0.0, 0.0, 0.0, 0.0;
  sing_v.setZero();
  robot->UpdateRobotModel(z3, iq, z3, z3, sing_q, sing_v, false);

  wbc::ManipulabilityHandler manip2;
  manip2.Init(robot, ee_idx, mcfg);
  for (int i = 0; i < kNJoints * 3; ++i) manip2.Update(kDt);
  std::cout << "w=" << manip2.manipulability() << "  active=" << manip2.is_active()
            << "  avoid=[" << manip2.avoidance_velocity().transpose() << "]\n";
}

// =============================================================================
// Test: Task introspection during Cartesian teleop
// =============================================================================
TEST(StateMachine, CartesianTaskIntrospection) {
  std::cout << "\n===== Cartesian Task Introspection =====\n";

  auto kp = Uniform(200.0);
  auto kd = Uniform(28.0);
  auto env = BuildMultiStateEnv(kp, kd, kHomeQpos, 50.0, 14.0, 50.0, 14.0);

  // Init for 1s
  for (int step = 0; step < 1000; ++step) {
    ReadJointState(env.get());
    env->arch->Update(env->js, step * kDt, kDt);
    ApplyCommand(env.get());
    mj_step(env->m, env->d);
  }

  auto* robot = env->arch->GetRobot();
  int ee_idx = robot->GetFrameIndex("end_effector");
  auto* registry = env->arch->GetConfig()->taskRegistry();
  auto* ee_pos_task = registry->GetMotionTask("ee_pos_task");
  auto* ee_ori_task = registry->GetMotionTask("ee_ori_task");
  auto* jpos_task   = registry->GetMotionTask("jpos_task");
  ASSERT_NE(ee_pos_task, nullptr);

  // Transition to cartesian_teleop
  env->arch->RequestState(3);
  auto* ct = dynamic_cast<wbc::CartesianTeleop*>(
      env->arch->GetFsmHandler()->FindStateById(3));
  ASSERT_NE(ct, nullptr);

  // Run 1 tick (FirstVisit)
  ReadJointState(env.get());
  env->arch->Update(env->js, 1.0, kDt);
  ApplyCommand(env.get());
  mj_step(env->m, env->d);

  Eigen::Vector3d ee_home = robot->GetLinkIsometry(ee_idx).translation();
  std::cout << std::fixed << std::setprecision(6);
  std::cout << "EE home: [" << ee_home.transpose() << "]\n";
  std::cout << "Task kp: [" << ee_pos_task->Kp().transpose() << "]\n";
  std::cout << "Task kd: [" << ee_pos_task->Kd().transpose() << "]\n";

  // Print Jacobian structure
  std::cout << "\nJacobian (3x7):\n";
  const auto& J = ee_pos_task->Jacobian();
  for (int r = 0; r < 3; ++r) {
    std::cout << "  [";
    for (int c = 0; c < kNJoints; ++c) {
      if (c) std::cout << ", ";
      std::cout << std::setw(10) << J(r, c);
    }
    std::cout << "]\n";
  }

  // Send velocity command
  Eigen::Vector3d xdot(0.05, 0.0, 0.0);
  Eigen::Vector3d zero3 = Eigen::Vector3d::Zero();
  Eigen::Quaterniond ident = Eigen::Quaterniond::Identity();
  int64_t ts = 1;

  std::cout << "\nstep | des_pos_x  | act_pos_x  | pos_err_x  | op_cmd_x   | op_cmd_y   | op_cmd_z   | tau[0]    | tau[1]    | tau[3]\n";

  for (int step = 0; step < 20; ++step) {
    double t = 1.001 + step * kDt;
    ts += 1000000;
    ct->UpdateCommand(xdot, zero3, ts, zero3, ident, 0);
    ReadJointState(env.get());
    env->arch->Update(env->js, t, kDt);
    const auto& cmd = env->arch->GetCommand();

    Eigen::Vector3d ee = robot->GetLinkIsometry(ee_idx).translation();

    std::cout << std::setw(4) << step << " | "
              << std::setw(10) << ee_pos_task->DesiredPos()[0] << " | "
              << std::setw(10) << ee_pos_task->CurrentPos()[0] << " | "
              << std::setw(10) << ee_pos_task->PosError()[0] << " | "
              << std::setw(10) << ee_pos_task->OpCommand()[0] << " | "
              << std::setw(10) << ee_pos_task->OpCommand()[1] << " | "
              << std::setw(10) << ee_pos_task->OpCommand()[2] << " | "
              << std::setw(9) << cmd.tau[0] << " | "
              << std::setw(9) << cmd.tau[1] << " | "
              << std::setw(9) << cmd.tau[3] << "\n";

    ApplyCommand(env.get());
    mj_step(env->m, env->d);
  }

  // Also print ori task state
  std::cout << "\n--- Orientation task at step 20 ---\n";
  std::cout << "ori des: [" << ee_ori_task->DesiredPos().transpose() << "]\n";
  std::cout << "ori act: [" << ee_ori_task->CurrentPos().transpose() << "]\n";
  std::cout << "ori err: [" << ee_ori_task->PosError().transpose() << "]\n";
  std::cout << "ori cmd: [" << ee_ori_task->OpCommand().transpose() << "]\n";

  std::cout << "\n--- Joint task at step 20 ---\n";
  std::cout << "jpos des: [" << jpos_task->DesiredPos().transpose() << "]\n";
  std::cout << "jpos act: [" << jpos_task->CurrentPos().transpose() << "]\n";
  std::cout << "jpos err: [" << jpos_task->PosError().head(3).transpose() << " ...]\n";
  std::cout << "jpos cmd: [" << jpos_task->OpCommand().head(3).transpose() << " ...]\n";
}

// =============================================================================
// Test: Forward dynamics verification — check if MuJoCo qacc matches WBIC intent
// =============================================================================
TEST(StateMachine, ForwardDynamicsCheck) {
  std::cout << "\n===== Forward Dynamics Check =====\n";

  auto kp = Uniform(200.0);
  auto kd = Uniform(28.0);
  auto env = BuildMultiStateEnv(kp, kd, kHomeQpos, 50.0, 14.0, 50.0, 14.0);

  // Init for 1s at home
  for (int step = 0; step < 1000; ++step) {
    ReadJointState(env.get());
    env->arch->Update(env->js, step * kDt, kDt);
    ApplyCommand(env.get());
    mj_step(env->m, env->d);
  }

  auto* robot = env->arch->GetRobot();
  int ee_idx = robot->GetFrameIndex("end_effector");
  auto* registry = env->arch->GetConfig()->taskRegistry();
  auto* ee_pos_task = registry->GetMotionTask("ee_pos_task");

  // Transition to cartesian_teleop
  env->arch->RequestState(3);
  auto* ct = dynamic_cast<wbc::CartesianTeleop*>(
      env->arch->GetFsmHandler()->FindStateById(3));

  // Run 1 tick (FirstVisit)
  ReadJointState(env.get());
  env->arch->Update(env->js, 1.0, kDt);
  ApplyCommand(env.get());
  mj_step(env->m, env->d);

  // Send +x velocity and check 3 ticks
  Eigen::Vector3d xdot(0.05, 0.0, 0.0);
  Eigen::Vector3d zero3 = Eigen::Vector3d::Zero();
  Eigen::Quaterniond ident = Eigen::Quaterniond::Identity();
  int64_t ts = 1;

  std::cout << std::fixed << std::setprecision(6);

  for (int step = 0; step < 3; ++step) {
    double t = 1.001 + step * kDt;
    ts += 1000000;
    ct->UpdateCommand(xdot, zero3, ts, zero3, ident, 0);
    ReadJointState(env.get());
    env->arch->Update(env->js, t, kDt);

    const auto& cmd = env->arch->GetCommand();
    const auto& J = ee_pos_task->Jacobian();
    const auto& Jdot_qdot = ee_pos_task->JacobianDotQdot();
    const auto& op_cmd = ee_pos_task->OpCommand();

    // Read qdot before step
    Eigen::VectorXd qdot_before(kNJoints);
    for (int i = 0; i < kNJoints; ++i) qdot_before[i] = env->d->qvel[i];

    // Apply torques
    ApplyCommand(env.get());
    // Call mj_forward to get qacc WITHOUT stepping
    mj_forward(env->m, env->d);

    Eigen::VectorXd qacc(kNJoints);
    for (int i = 0; i < kNJoints; ++i) qacc[i] = env->d->qacc[i];

    // Compute J*qacc + Jdot*qdot (this should equal op_cmd if dynamics match)
    Eigen::VectorXd xddot_actual = J * qacc + Jdot_qdot;
    Eigen::VectorXd xddot_expected = op_cmd;

    // Also compute inferred qddot from Pinocchio's model
    Eigen::VectorXd tau_cmd(kNJoints);
    for (int i = 0; i < kNJoints; ++i) tau_cmd[i] = cmd.tau[i];
    Eigen::VectorXd nle = robot->GetGravity() + robot->GetCoriolis();
    Eigen::MatrixXd Minv = robot->GetMassMatrixInverse();
    Eigen::VectorXd qddot_inferred = Minv * (tau_cmd - nle);
    Eigen::VectorXd xddot_inferred = J * qddot_inferred + Jdot_qdot;

    std::cout << "--- Step " << step << " ---\n";
    std::cout << "  tau_cmd:      [";
    for (int i = 0; i < kNJoints; ++i) { if (i) std::cout << ", "; std::cout << cmd.tau[i]; }
    std::cout << "]\n";
    std::cout << "  nle:          [" << nle.transpose() << "]\n";
    std::cout << "  mj_qacc:      [" << qacc.transpose() << "]\n";
    std::cout << "  pin_qddot:    [" << qddot_inferred.transpose() << "]\n";
    std::cout << "  qacc diff:    [" << (qacc - qddot_inferred).transpose() << "]\n";
    std::cout << "  J*mj_qacc:    [" << xddot_actual.transpose() << "]\n";
    std::cout << "  J*pin_qddot:  [" << xddot_inferred.transpose() << "]\n";
    std::cout << "  op_cmd(pos):  [" << xddot_expected.transpose() << "]\n";

    // Now actually step
    mj_step(env->m, env->d);
  }

  // Also: check what Pinocchio's forward dynamics would give
  std::cout << "\n--- Pinocchio FD check at home ---\n";
  Eigen::VectorXd q_home(kNJoints), qdot_zero(kNJoints);
  for (int i = 0; i < kNJoints; ++i) { q_home[i] = kHomeQpos[i]; qdot_zero[i] = 0.0; }
  robot->UpdateRobotModel(zero3, ident, zero3, zero3, q_home, qdot_zero, false);

  Eigen::VectorXd grav = robot->GetGravity();
  Eigen::MatrixXd M = robot->GetMassMatrix();
  std::cout << "  Pinocchio grav: [" << grav.transpose() << "]\n";
  std::cout << "  M diag: [";
  for (int i = 0; i < kNJoints; ++i) { if (i) std::cout << ", "; std::cout << M(i,i); }
  std::cout << "]\n";

  // Apply grav torques in MuJoCo: set ctrl = grav, forward, check qacc ≈ 0
  for (int i = 0; i < kNJoints; ++i) {
    env->d->qpos[i] = kHomeQpos[i];
    env->d->qvel[i] = 0.0;
    env->d->ctrl[i] = grav[i];
  }
  mj_forward(env->m, env->d);
  std::cout << "  MJ qacc with grav comp: [";
  for (int i = 0; i < kNJoints; ++i) { if (i) std::cout << ", "; std::cout << env->d->qacc[i]; }
  std::cout << "]\n";
  std::cout << "  (Should be ~0 if models match)\n";
}

// =============================================================================
// Test: Null-space method comparison — trajectory precision + Hz
// Tracks a sinusoidal x-direction EE trajectory and compares methods.
// =============================================================================
TEST(StateMachine, NullSpaceMethodComparison) {
  std::cout << "\n===== Null-Space Method Comparison =====\n";
  std::cout << "Trajectory: x += 0.03*sin(2π*t), hold y/z, 2s duration\n\n";

  struct MethodTrial {
    const char* name;
    wbc::NullSpaceMethod method;
  };
  std::vector<MethodTrial> methods = {
    {"SVD_EXACT",  wbc::NullSpaceMethod::SVD_EXACT},
    {"DLS(0.05)",  wbc::NullSpaceMethod::DLS},
    {"DLS_MICRO",  wbc::NullSpaceMethod::DLS_MICRO},
  };

  auto jpos_kp = Uniform(200.0);
  auto jpos_kd = Uniform(28.0);

  std::cout << std::fixed << std::setprecision(6);
  std::cout << "method     | rms_pos(m)  | max_pos(m)  | rms_ori(rad) | max_ori(rad) | avg_us  | Hz\n";
  std::cout << "-----------+-------------+-------------+--------------+--------------+---------+---------\n";

  for (const auto& trial : methods) {
    auto env = BuildMultiStateEnv(jpos_kp, jpos_kd, kHomeQpos, 200.0, 28.0, 200.0, 28.0);

    // Set the null-space method
    env->arch->GetSolver()->SetNullSpaceMethod(trial.method);

    // Init for 1s
    for (int step = 0; step < 1000; ++step) {
      ReadJointState(env.get());
      env->arch->Update(env->js, step * kDt, kDt);
      ApplyCommand(env.get());
      mj_step(env->m, env->d);
    }

    // Transition to cartesian_teleop
    env->arch->RequestState(3);
    auto* ct = dynamic_cast<wbc::CartesianTeleop*>(
        env->arch->GetFsmHandler()->FindStateById(3));
    ASSERT_NE(ct, nullptr);

    auto* robot = env->arch->GetRobot();
    int ee_idx = robot->GetFrameIndex("end_effector");

    // Hold 0.5s to settle
    for (int step = 0; step < 500; ++step) {
      ReadJointState(env.get());
      env->arch->Update(env->js, 1.0 + step * kDt, kDt);
      ApplyCommand(env.get());
      mj_step(env->m, env->d);
    }

    ReadJointState(env.get());
    env->arch->Update(env->js, 1.5, kDt);
    Eigen::Vector3d home_ee = robot->GetLinkIsometry(ee_idx).translation();
    Eigen::Quaterniond home_quat(robot->GetLinkIsometry(ee_idx).rotation());

    // Phase: sinusoidal trajectory for 2s
    // Desired: x(t) = home_x + A*sin(2π*f*t), y=home_y, z=home_z
    // Velocity: xdot = A*2π*f*cos(2π*f*t), 0, 0
    const double A = 0.03;  // amplitude [m]
    const double freq = 1.0;  // [Hz]
    const int traj_steps = 2000;  // 2s

    double sum_sq_pos = 0.0, max_pos_err = 0.0;
    double sum_sq_ori = 0.0, max_ori_err = 0.0;
    Eigen::Vector3d zero3 = Eigen::Vector3d::Zero();
    Eigen::Quaterniond ident = Eigen::Quaterniond::Identity();
    int64_t ts = 1;

    auto t_start = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < traj_steps; ++step) {
      double t = step * kDt;
      double t_base = 1.501 + t;

      // Desired velocity: xdot = A*2π*f*cos(2πft)
      double xdot = A * 2.0 * M_PI * freq * std::cos(2.0 * M_PI * freq * t);
      Eigen::Vector3d vel_cmd(xdot, 0.0, 0.0);

      ts += 1000000;
      ct->UpdateCommand(vel_cmd, zero3, ts, zero3, ident, 0);
      ReadJointState(env.get());
      env->arch->Update(env->js, t_base, kDt);
      ApplyCommand(env.get());
      mj_step(env->m, env->d);

      // Compute desired EE position
      Eigen::Vector3d des_ee = home_ee;
      des_ee.x() += A * std::sin(2.0 * M_PI * freq * t);

      // Measure actual
      ReadJointState(env.get());
      env->arch->Update(env->js, t_base + kDt * 0.5, kDt);
      Eigen::Vector3d act_ee = robot->GetLinkIsometry(ee_idx).translation();

      double pos_err = (act_ee - des_ee).norm();
      sum_sq_pos += pos_err * pos_err;
      max_pos_err = std::max(max_pos_err, pos_err);

      // Orientation error (should stay near home orientation)
      Eigen::Quaterniond act_quat(robot->GetLinkIsometry(ee_idx).rotation());
      double ori_err = act_quat.angularDistance(home_quat);
      sum_sq_ori += ori_err * ori_err;
      max_ori_err = std::max(max_ori_err, ori_err);
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed_us = std::chrono::duration<double, std::micro>(t_end - t_start).count();
    double avg_us = elapsed_us / traj_steps;

    double rms_pos = std::sqrt(sum_sq_pos / traj_steps);
    double rms_ori = std::sqrt(sum_sq_ori / traj_steps);

    std::cout << std::left << std::setw(10) << trial.name << " | "
              << std::setw(11) << rms_pos << " | "
              << std::setw(11) << max_pos_err << " | "
              << std::setw(12) << rms_ori << " | "
              << std::setw(12) << max_ori_err << " | "
              << std::setw(7) << std::setprecision(1) << avg_us << " | "
              << std::setw(7) << std::setprecision(0) << (1e6 / avg_us)
              << std::setprecision(6) << "\n";
  }
}

// =============================================================================
// Test: Jacobian Convention Verification
// Empirically determine which rows of FillLinkJacobian are linear vs angular
// by comparing J*qdot with actual link velocity from Pinocchio.
// =============================================================================
TEST(StateMachine, JacobianVerification) {
  std::cout << "\n===== Jacobian Convention Verification =====\n";

  auto kp = Uniform(200.0);
  auto kd = Uniform(28.0);
  auto env = BuildMultiStateEnv(kp, kd, kHomeQpos);

  // Init for 0.5s to let robot settle
  for (int step = 0; step < 500; ++step) {
    ReadJointState(env.get());
    env->arch->Update(env->js, step * kDt, kDt);
    ApplyCommand(env.get());
    mj_step(env->m, env->d);
  }

  auto* robot = env->arch->GetRobot();
  int ee_idx = robot->GetFrameIndex("end_effector");

  // Use a non-trivial velocity configuration
  Eigen::VectorXd q(kNJoints), qdot(kNJoints);
  for (int i = 0; i < kNJoints; ++i) {
    q[i] = env->d->qpos[i];
    qdot[i] = (i % 2 == 0) ? 0.5 : -0.3;  // non-zero pattern
  }

  Eigen::Vector3d z3 = Eigen::Vector3d::Zero();
  Eigen::Quaterniond iq = Eigen::Quaterniond::Identity();
  robot->UpdateRobotModel(z3, iq, z3, z3, q, qdot, false);

  // Get spatial velocity from Pinocchio: [angular(0-2); linear(3-5)]
  Eigen::Matrix<double, 6, 1> spatial_vel = robot->GetLinkSpatialVel(ee_idx);
  Eigen::Vector3d ang_vel = spatial_vel.head<3>();
  Eigen::Vector3d lin_vel = spatial_vel.tail<3>();

  // Get Jacobian from FillLinkJacobian
  Eigen::MatrixXd full_jac(6, kNJoints);
  robot->FillLinkJacobian(ee_idx, full_jac);

  // Compute J * qdot
  Eigen::Matrix<double, 6, 1> jac_times_qdot = full_jac * qdot;
  Eigen::Vector3d top_rows = jac_times_qdot.head<3>();
  Eigen::Vector3d bot_rows = jac_times_qdot.tail<3>();

  std::cout << std::fixed << std::setprecision(6);
  std::cout << "Pinocchio angular vel: [" << ang_vel.transpose() << "]\n";
  std::cout << "Pinocchio linear  vel: [" << lin_vel.transpose() << "]\n";
  std::cout << "\nFillLinkJacobian * qdot:\n";
  std::cout << "  top rows (0-2): [" << top_rows.transpose() << "]\n";
  std::cout << "  bot rows (3-5): [" << bot_rows.transpose() << "]\n";

  double top_matches_linear  = (top_rows - lin_vel).norm();
  double top_matches_angular = (top_rows - ang_vel).norm();
  double bot_matches_linear  = (bot_rows - lin_vel).norm();
  double bot_matches_angular = (bot_rows - ang_vel).norm();

  std::cout << "\nMatching:\n";
  std::cout << "  top rows vs linear:  " << top_matches_linear  << "\n";
  std::cout << "  top rows vs angular: " << top_matches_angular << "\n";
  std::cout << "  bot rows vs linear:  " << bot_matches_linear  << "\n";
  std::cout << "  bot rows vs angular: " << bot_matches_angular << "\n";

  if (top_matches_linear < top_matches_angular) {
    std::cout << "\n>>> FillLinkJacobian layout: [LINEAR(0-2); ANGULAR(3-5)]\n";
    std::cout << ">>> LinkPosTask takes block(3,0,3,n) = ANGULAR rows — WRONG!\n";
    std::cout << ">>> LinkOriTask takes block(0,0,3,n) = LINEAR rows  — WRONG!\n";
  } else {
    std::cout << "\n>>> FillLinkJacobian layout: [ANGULAR(0-2); LINEAR(3-5)]\n";
    std::cout << ">>> LinkPosTask takes block(3,0,3,n) = LINEAR rows  — CORRECT\n";
    std::cout << ">>> LinkOriTask takes block(0,0,3,n) = ANGULAR rows — CORRECT\n";
  }

  // Also verify JacobianDotQdot convention
  Eigen::Matrix<double, 6, 1> jdotqdot = robot->GetLinkJacobianDotQdot(ee_idx);
  std::cout << "\nGetLinkJacobianDotQdot: [" << jdotqdot.transpose() << "]\n";
  std::cout << "  Layout: [angular(0-2); linear(3-5)] (per code inspection)\n";

  // Assert that we have clarity
  EXPECT_LT(std::min(top_matches_linear, top_matches_angular), 1e-6)
      << "Top rows should clearly match either linear or angular";
  EXPECT_LT(std::min(bot_matches_linear, bot_matches_angular), 1e-6)
      << "Bot rows should clearly match either linear or angular";
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
