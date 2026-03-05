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
    auto r = RunSim(kp, kd, kHomeQpos, 5.0);
    std::ostringstream label;
    label << "uniform";
    PrintResult(label.str(), kp, kd, r);
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
    auto r = RunSim(kp, kd, kHomeQpos, 5.0);
    std::ostringstream label;
    label << "split  ";
    PrintResult(label.str(), kp, kd, r);
  }
}

TEST(GainTuning, DifferentTargetPoses) {
  std::cout << "\n===== Best gains across different poses =====\n";

  auto kp = Uniform(400.0);
  auto kd = Uniform(40.0);

  std::vector<std::pair<std::string, std::array<double, kNJoints>>> poses = {
    {"home       ", kHomeQpos},
    {"zeros      ", {0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0}},  // joint2 min ~1.047
    {"bent       ", {0.5, 2.5, 0.5, -1.0, 0.5, -1.0, 0.5}},
    {"stretched  ", {0.0, 2.0, 0.0, -0.5, 0.0, -0.5, 0.0}},
  };

  for (const auto& [name, target] : poses) {
    auto r = RunSim(kp, kd, target, 5.0);
    PrintResult(name, kp, kd, r);
  }
}

TEST(GainTuning, DynamicsCompensation) {
  std::cout << "\n===== Dynamics Compensation Flags (kp=400, kd=40) =====\n";

  auto kp = Uniform(400.0);
  auto kd = Uniform(40.0);

  struct FlagTrial {
    std::string name;
    ControllerFlags flags;
  };

  std::vector<FlagTrial> trials = {
    {"grav_only       ", {.gravity=true,  .coriolis=false, .inertia=false}},
    {"grav+coriolis   ", {.gravity=true,  .coriolis=true,  .inertia=false}},
    {"grav+inertia    ", {.gravity=true,  .coriolis=false, .inertia=true}},
    {"full_dynamics   ", {.gravity=true,  .coriolis=true,  .inertia=true}},
    {"no_compensation ", {.gravity=false, .coriolis=false, .inertia=false}},
  };

  for (const auto& t : trials) {
    auto r = RunSim(kp, kd, kHomeQpos, 5.0, t.flags);
    PrintResult(t.name, kp, kd, r);
  }
}

TEST(GainTuning, HighGainsWithDynamics) {
  std::cout << "\n===== High Gains + Full Dynamics =====\n";

  ControllerFlags full_dyn{.gravity=true, .coriolis=true, .inertia=true};

  struct Trial { double kp; double kd; };
  std::vector<Trial> trials = {
    {100.0,  20.0},
    {400.0,  40.0},
    {1000.0, 63.0},
    {2000.0, 89.0},
    {4000.0, 126.0},
  };

  for (const auto& t : trials) {
    auto kp = Uniform(t.kp);
    auto kd = Uniform(t.kd);
    auto r = RunSim(kp, kd, kHomeQpos, 5.0, full_dyn);
    PrintResult("full_dyn", kp, kd, r);
  }
}

TEST(GainTuning, DiagnoseTorqueClipping) {
  std::cout << "\n===== Torque Clipping Diagnosis =====\n";
  std::cout << "MuJoCo actuator limits: [95, 95, 40, 40, 15, 15, 15] Nm\n\n";

  auto kp = Uniform(400.0);
  auto kd = Uniform(40.0);

  // Use a longer sim to ensure full settling.
  auto r = RunSim(kp, kd, kHomeQpos, 10.0);

  std::cout << std::fixed << std::setprecision(5);
  std::cout << "Joint |  target     |  q_final    | error [rad] | tau_cmd [Nm] | limit [Nm]\n";
  std::cout << "------+-------------+-------------+-------------+--------------+-----------\n";
  const double limits[] = {95.0, 95.0, 40.0, 40.0, 15.0, 15.0, 15.0};
  for (int i = 0; i < kNJoints; ++i) {
    double q_final = kHomeQpos[i] + (r.per_joint_error[i]);  // approximate
    std::cout << "  " << i
              << "   |  " << std::setw(9) << kHomeQpos[i]
              << "  |  " << std::setw(9) << (kHomeQpos[i] - r.per_joint_error[i])
              << "  |  " << r.per_joint_error[i]
              << "  |  " << std::setw(12) << r.final_tau[i]
              << "  |  " << std::setw(8) << limits[i] << "\n";
  }

  // Also test with gravity disabled in MuJoCo to confirm gravity is the cause.
  // Test with MuJoCo gravity=0 to confirm gravity model mismatch.
  std::cout << "\nWith MuJoCo gravity=0:\n";
  auto r0 = RunSim(kp, kd, kHomeQpos, 5.0, {}, false);
  PrintResult("zero_grav", kp, kd, r0);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
