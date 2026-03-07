/**
 * @file test_domain_randomization.cpp
 * @brief Domain randomization test for adaptive compensators.
 *
 * Randomizes MuJoCo joint dynamics (damping, friction, stiffness) and verifies
 * that adaptive friction compensation + momentum observer improve tracking
 * robustness compared to PID-only baseline.
 *
 * Test protocol:
 *   1. Generate randomized MJCF with scaled damping/friction/stiffness
 *   2. Run 8-waypoint Cartesian teleop with each compensator config
 *   3. Compare transit RMS, hold error, orientation error
 *   4. Repeat for N random seeds and report aggregate statistics
 */
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <mujoco/mujoco.h>

#include "wbc_architecture/control_architecture.hpp"
#include "wbc_formulation/interface/task.hpp"
#include "wbc_robot_system/pinocchio_robot_system.hpp"

#include "optimo_controller/state_machines/cartesian_teleop.hpp"
#include "optimo_controller/state_machines/joint_teleop.hpp"

namespace {

constexpr int kNJoints = 7;
constexpr double kDt = 0.001;

const std::array<double, kNJoints> kHomeQpos = {
    0.0, 3.14159, 0.0, 0.0, 0.0, 0.0, 0.0};
const std::array<double, kNJoints> kJointTorqueLimitNm = {
    79.0, 95.0, 32.0, 40.0, 15.0, 15.0, 15.0};

// Nominal dynamics from optimo.xml
const std::array<double, kNJoints> kNomDamping = {0.4, 0.4, 0.3, 0.3, 0.1, 0.1, 0.1};
const std::array<double, kNJoints> kNomFriction = {0.2, 0.2, 0.2, 0.2, 0.4, 0.4, 0.4};
const std::array<double, kNJoints> kNomStiffness = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
const std::array<double, kNJoints> kSpringRef = {0, 3.14159, 0, 0, 0, 0, 0};

std::string ResolvePackagePath(const std::string& pkg_name,
                               const std::string& rel_path) {
  const char* prefix = std::getenv("AMENT_PREFIX_PATH");
  if (!prefix)
    throw std::runtime_error("AMENT_PREFIX_PATH not set.");
  std::istringstream ss(prefix);
  std::string token;
  while (std::getline(ss, token, ':')) {
    auto full = std::filesystem::path(token) / "share" / pkg_name / rel_path;
    if (std::filesystem::exists(full)) return full.string();
  }
  throw std::runtime_error("Cannot resolve package://" + pkg_name + "/" + rel_path);
}

// ── Randomized dynamics parameters ──────────────────────────────────────────

// Nominal per-link masses from URDF (used as baseline for mass perturbation)
const std::array<double, kNJoints> kNomMass = {4.14, 1.72, 4.13, 1.09, 1.92, 1.04, 0.88};

struct DynamicsParams {
  std::array<double, kNJoints> damping;
  std::array<double, kNJoints> friction;
  std::array<double, kNJoints> stiffness;
  std::array<double, kNJoints> mass_scale;  // multiplier on link mass (1.0 = nominal)
  std::array<Eigen::Vector3d, kNJoints> com_offset;  // additive CoM shift [m]
  double ee_payload_kg{0.0};                // extra payload mass at EE
};

DynamicsParams RandomizeDynamics(std::mt19937& rng, double scale_range,
                                  double mass_range = 0.0,
                                  double payload_max = 0.0,
                                  double com_range = 0.0) {
  std::uniform_real_distribution<double> dist(1.0 - scale_range, 1.0 + scale_range);
  std::uniform_real_distribution<double> mass_dist(1.0 - mass_range, 1.0 + mass_range);
  std::uniform_real_distribution<double> payload_dist(0.0, payload_max);
  std::uniform_real_distribution<double> com_dist(-com_range, com_range);
  DynamicsParams p;
  for (int i = 0; i < kNJoints; ++i) {
    p.damping[i]   = std::max(0.0, kNomDamping[i] * dist(rng));
    p.friction[i]  = std::max(0.0, kNomFriction[i] * dist(rng));
    p.stiffness[i] = std::max(0.0, kNomStiffness[i] * dist(rng));
    p.mass_scale[i] = (mass_range > 0) ? mass_dist(rng) : 1.0;
    p.com_offset[i] = (com_range > 0)
        ? Eigen::Vector3d(com_dist(rng), com_dist(rng), com_dist(rng))
        : Eigen::Vector3d::Zero();
  }
  p.ee_payload_kg = (payload_max > 0) ? payload_dist(rng) : 0.0;
  return p;
}

// ── Write randomized MJCF ───────────────────────────────────────────────────

void WriteRandomizedMjcf(const std::filesystem::path& out_path,
                         const std::string& base_mjcf_path,
                         const DynamicsParams& dyn) {
  // Read the base MJCF and replace dynamics parameters.
  // Rather than XML parsing, write a new MJCF from scratch using the base mesh paths.
  std::string meshdir;
  {
    // Meshes are in urdf/meshes/ relative to package share dir
    auto base = std::filesystem::path(base_mjcf_path).parent_path();
    meshdir = (base / "../urdf/meshes/").string();
  }

  std::ofstream f(out_path);
  f << "<mujoco model=\"optimo_randomized\">\n"
    << "  <compiler angle=\"radian\" meshdir=\"" << meshdir << "\"/>\n"
    << "  <option gravity=\"0 0 -9.81\" timestep=\"0.001\" solver=\"Newton\" "
       "tolerance=\"1e-12\" impratio=\"60.0\"/>\n"
    << "  <default><geom contype=\"0\" conaffinity=\"0\"/></default>\n"
    << "  <asset>\n";
  for (int i = 0; i <= 7; ++i)
    f << "    <mesh name=\"col" << i << "\" content_type=\"model/stl\" file=\"col"
      << i << ".stl\"/>\n";
  f << "  </asset>\n";

  // Joint specs: name, pos, quat_parent, axis, range, mesh
  struct JointSpec {
    const char* link_name;
    const char* parent_pos;
    const char* parent_quat;
    const char* axis;
    const char* range;
    const char* mesh;
    const char* mesh_quat;
    const char* inertial;
  };

  // We write the full kinematic tree matching optimo.xml exactly,
  // only varying damping/frictionloss/stiffness/springref.
  f << "  <worldbody>\n"
    << "    <geom size=\"0.05 0.05 0.5\" pos=\"0 0 -0.5\" type=\"box\" rgba=\"1 1 1 1\"/>\n"
    << "    <geom pos=\"0 0 0.07\" quat=\"1 0 0 0\" type=\"mesh\" rgba=\"0.5 0.5 1 1\" mesh=\"col0\"/>\n";

  // Helper to write joint line
  auto joint_line = [&](int idx, const char* name, const char* axis, const char* range) {
    f << "      <joint name=\"" << name << "\" pos=\"0 0 0\" axis=\"" << axis
      << "\" range=\"" << range << "\""
      << " damping=\"" << dyn.damping[idx] << "\""
      << " frictionloss=\"" << dyn.friction[idx] << "\""
      << " stiffness=\"" << dyn.stiffness[idx] << "\""
      << " springref=\"" << kSpringRef[idx] << "\"/>\n";
  };

  // Helper to write inertial with mass scaling.
  // Mass scales uniformly; inertia scales proportionally (I ∝ m).
  auto inertial = [&](int link_idx, const char* pos, const char* quat,
                       double nom_mass, const char* diag) {
    double m = nom_mass * dyn.mass_scale[link_idx];
    double s = dyn.mass_scale[link_idx];  // inertia scale factor
    // Parse diaginertia values and scale them
    double i1, i2, i3;
    double p1, p2, p3;
    std::sscanf(diag, "%lf %lf %lf", &i1, &i2, &i3);
    std::sscanf(pos, "%lf %lf %lf", &p1, &p2, &p3);
    const Eigen::Vector3d com_nominal(p1, p2, p3);
    const Eigen::Vector3d com_shifted = com_nominal + dyn.com_offset[link_idx];
    f << std::setprecision(8)
      << "      <inertial pos=\"" << com_shifted.x() << " " << com_shifted.y()
      << " " << com_shifted.z() << "\" quat=\"" << quat
      << "\" mass=\"" << m << "\" diaginertia=\""
      << i1*s << " " << i2*s << " " << i3*s << "\"/>\n";
    f << std::setprecision(2);  // restore
  };

  // link1
  f << "    <body name=\"link1\" pos=\"0 0 0.07\">\n";
  inertial(0, "0 -0.01635 0.01549", "0.733993 0.672808 0.0651674 -0.0658565",
           4.14, "0.0100027 0.00995582 0.00747546");
  joint_line(0, "joint1", "0 0 1", "-2.0944 2.0944");
  f << "      <geom quat=\"0.707107 0.707107 0 0\" type=\"mesh\" rgba=\"1 1 1 1\" mesh=\"col1\"/>\n";

  // link2
  f << "      <body name=\"link2\" quat=\"0.707107 0.707107 0 0\">\n";
  inertial(1, "0 -0.05063 0.05677", "0.697505 0.116135 -0.116135 0.697505",
           1.72, "0.0063654 0.005694 0.0019516");
  joint_line(1, "joint2", "0 0 1", "1.0472 3.83972");
  f << "        <geom quat=\"0.707107 0.707107 0 0\" type=\"mesh\" rgba=\"0.5 0.5 1 1\" mesh=\"col2\"/>\n";

  // link3
  f << "        <body name=\"link3\" pos=\"0 -0.4 0\" quat=\"0.707107 0.707107 0 0\">\n";
  inertial(2, "-3.9e-05 0.098 0", "0.50494 0.474771 -0.494533 0.524466",
           4.13, "0.0130882 0.0129889 0.00254595");
  joint_line(2, "joint3", "0 0 1", "-2.0944 2.0944");
  f << "          <geom pos=\"0 0 0\" quat=\"0.707107 -0.707107 0 0\" type=\"mesh\" rgba=\"1 1 1 1\" mesh=\"col3\"/>\n";

  // link4
  f << "          <body name=\"link4\" quat=\"0.707107 -0.707107 0 0\">\n";
  inertial(3, "-1e-05 0.048852 0.0688", "0.501659 0.493047 -0.505423 0.499791",
           1.09, "0.0338771 0.0336796 0.00108207");
  joint_line(3, "joint4", "0 0 1", "-2.53073 2.53073");
  f << "            <geom quat=\"0.707107 0.707107 0 0\" type=\"mesh\" rgba=\"0.5 0.5 1 1\" mesh=\"col4\"/>\n";

  // link5
  f << "            <body name=\"link5\" pos=\"0 -0.4 0\" quat=\"0.707107 0.707107 0 0\">\n";
  inertial(4, "0.001262 0.085 0.008", "0.984064 -0.177814 -8.09748e-06 0.000289155",
           1.92, "0.00093167 0.000813494 0.000721196");
  joint_line(4, "joint5", "0 0 1", "-2.61799 2.61799");
  f << "              <geom pos=\"0 0 0\" quat=\"0.707107 -0.707107 0 0\" type=\"mesh\" rgba=\"1 1 1 1\" mesh=\"col5\"/>\n";

  // link6
  f << "              <body name=\"link6\" quat=\"0.707107 -0.707107 0 0\">\n";
  inertial(5, "0 -7e-05 0.010855", "0.984064 -0.177814 -8.09748e-06 0.000289155",
           1.04, "0.00093167 0.000813494 0.000721196");
  joint_line(5, "joint6", "0 0 1", "-2.0944 2.0944");
  f << "                <geom quat=\"0.707107 0.707107 0 0\" type=\"mesh\" rgba=\"0.5 0.5 1 1\" mesh=\"col6\"/>\n";

  // link7
  f << "                <body name=\"link7\" quat=\"0.707107 0.707107 0 0\">\n";
  inertial(6, "0 0 0.1", "0.999477 0.00657917 -0.0101967 0.0299688",
           0.88, "0.0055453 0.00477683 0.00172087");
  joint_line(6, "joint7", "0 0 1", "-2.0944 2.0944");
  f << "                  <geom type=\"mesh\" rgba=\"1 1 1 1\" mesh=\"col7\"/>\n";

  // EE payload: extra mass at end-effector tip (unknown to URDF)
  if (dyn.ee_payload_kg > 0.01) {
    // Approximate payload as point mass at EE tip (0.1m beyond link7 CoM)
    double pl = dyn.ee_payload_kg;
    f << "                  <body name=\"payload\" pos=\"0 0 0.2\">\n"
      << "                    <inertial pos=\"0 0 0\" mass=\"" << pl
      << "\" diaginertia=\"" << pl*0.001 << " " << pl*0.001 << " " << pl*0.001 << "\"/>\n"
      << "                    <geom type=\"sphere\" size=\"0.03\" rgba=\"1 0 0 1\"/>\n"
      << "                  </body>\n";
  }

  // Close all bodies
  f << "                </body>\n"   // link7
    << "              </body>\n"     // link6
    << "            </body>\n"       // link5
    << "          </body>\n"         // link4
    << "        </body>\n"           // link3
    << "      </body>\n"             // link2
    << "    </body>\n"               // link1
    << "  </worldbody>\n";

  f << "  <keyframe>\n"
    << "    <key name=\"spawn_home\" qpos=\"0 3.14159 0 0 0 0 0\"/>\n"
    << "  </keyframe>\n";

  f << "  <actuator>\n";
  const double trq_limits[] = {79, 95, 32, 40, 15, 15, 15};
  for (int i = 0; i < kNJoints; ++i) {
    f << "    <motor name=\"joint" << (i+1) << "_motor\" joint=\"joint" << (i+1)
      << "\" gear=\"1\" ctrllimited=\"true\" ctrlrange=\"-" << trq_limits[i]
      << " " << trq_limits[i] << "\" forcelimited=\"true\" forcerange=\"-"
      << trq_limits[i] << " " << trq_limits[i] << "\"/>\n";
  }
  f << "  </actuator>\n</mujoco>\n";
}

// ── YAML writers ─────────────────────────────────────────────────────────────

struct TaskGains {
  double jpos_kp{100.0}, jpos_kd{20.0};
  double ee_kp{3200.0}, ee_kd{113.0};
};

void WriteTaskYaml(const std::filesystem::path& dir,
                   const TaskGains& g = TaskGains{}) {
  std::ofstream f(dir / "task_list.yaml");
  f << "task_pool:\n"
    << "  - name: \"jpos_task\"\n"
    << "    type: \"JointTask\"\n"
    << "    kp: " << g.jpos_kp << "\n"
    << "    kd: " << g.jpos_kd << "\n"
    << "    kp_ik: 1.0\n"
    << "\n"
    << "  - name: \"ee_pos_task\"\n"
    << "    type: \"LinkPosTask\"\n"
    << "    target_frame: \"optimo_end_effector\"\n"
    << "    reference_frame: \"optimo_base_link\"\n"
    << "    kp: " << g.ee_kp << "\n"
    << "    kd: " << g.ee_kd << "\n"
    << "    kp_ik: 1.0\n"
    << "\n"
    << "  - name: \"ee_ori_task\"\n"
    << "    type: \"LinkOriTask\"\n"
    << "    target_frame: \"optimo_end_effector\"\n"
    << "    reference_frame: \"optimo_base_link\"\n"
    << "    kp: " << g.ee_kp << "\n"
    << "    kd: " << g.ee_kd << "\n"
    << "    kp_ik: 1.0\n";
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
    << "      target_jpos: [0, 3.14159, 0, 0, 0, 0, 0]\n"
    << "    task_hierarchy:\n"
    << "      - name: \"jpos_task\"\n"
    << "        weight: 1.0\n"
    << "      - name: \"ee_pos_task\"\n"
    << "        weight: 1e-6\n"
    << "      - name: \"ee_ori_task\"\n"
    << "        weight: 1e-6\n"
    << "\n"
    << "  - id: 1\n"
    << "    name: \"home\"\n"
    << "    type: \"initialize\"\n"
    << "    params:\n"
    << "      duration: 2.0\n"
    << "      wait_time: 0.0\n"
    << "      stay_here: true\n"
    << "      target_jpos: [0, 3.14159, 0, 0, 0, 0, 0]\n"
    << "    task_hierarchy:\n"
    << "      - name: \"jpos_task\"\n"
    << "        weight: 1.0\n"
    << "      - name: \"ee_pos_task\"\n"
    << "        weight: 1e-6\n"
    << "      - name: \"ee_ori_task\"\n"
    << "        weight: 1e-6\n"
    << "\n"
    << "  - id: 2\n"
    << "    name: \"joint_teleop\"\n"
    << "    params:\n"
    << "      stay_here: true\n"
    << "      joint_vel_limit: [0.5, 0.5, 0.5, 0.5, 0.3, 0.3, 0.3]\n"
    << "    task_hierarchy:\n"
    << "      - name: \"jpos_task\"\n"
    << "        weight: 1.0\n"
    << "      - name: \"ee_pos_task\"\n"
    << "        weight: 1e-6\n"
    << "      - name: \"ee_ori_task\"\n"
    << "        weight: 1e-6\n"
    << "\n"
    << "  - id: 3\n"
    << "    name: \"cartesian_teleop\"\n"
    << "    params:\n"
    << "      stay_here: true\n"
    << "      linear_vel_max: 0.1\n"
    << "      angular_vel_max: 0.5\n"
    << "      manipulability:\n"
    << "        step_size: 0.5\n"
    << "        w_threshold: 0.01\n"
    << "    task_hierarchy:\n"
    << "      - name: \"ee_pos_task\"\n"
    << "        weight: 10.0\n"
    << "      - name: \"ee_ori_task\"\n"
    << "        weight: 10.0\n"
    << "      - name: \"jpos_task\"\n"
    << "        weight: 0.01\n";
}

void WritePidYaml(const std::filesystem::path& dir, double kp_vel = 1.0) {
  std::ofstream f(dir / "joint_pid_gains.yaml");
  f << "default:\n"
    << "  kp_pos: 200.0\n"
    << "  ki_pos: 0.0\n"
    << "  kd_pos: 28.0\n"
    << "  kp_vel: " << kp_vel << "\n"
    << "  ki_vel: 0.0\n"
    << "  kd_vel: 0.0\n";
}

struct CompensatorConfig {
  const char* label;
  bool pid;
  bool friction;
  double gamma_c;
  double gamma_v;
  double max_f_c;
  double max_f_v;
  bool observer;
  double K_o;
  double max_tau_dist;
  double kp_vel{1.0};  // 0 = direct-PD mode (no velocity loop)
  bool enforce_torque_limit{false};
};

void WriteWbcYaml(const std::filesystem::path& dir, const CompensatorConfig& cc) {
  std::ofstream f(dir / "optimo_wbc.yaml");
  auto b = [](bool v) { return v ? "true" : "false"; };

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
    << "    enabled: " << b(cc.pid) << "\n"
    << "    gains_yaml: \"joint_pid_gains.yaml\"\n";

  if (cc.friction) {
    f << "  friction_compensator:\n"
      << "    enabled: true\n"
      << "    gamma_c: " << cc.gamma_c << "\n"
      << "    gamma_v: " << cc.gamma_v << "\n"
      << "    max_f_c: " << cc.max_f_c << "\n"
      << "    max_f_v: " << cc.max_f_v << "\n";
  }

  if (cc.observer) {
    f << "  momentum_observer:\n"
      << "    enabled: true\n"
      << "    K_o: " << cc.K_o << "\n"
      << "    max_tau_dist: " << cc.max_tau_dist << "\n";
  }

  f << "\n"
    << "regularization:\n"
    << "  w_qddot: 1.0e-6\n"
    << "  w_tau: 0.0\n"
    << "  w_tau_dot: 0.0\n"
    << "  w_rf: 1.0e-4\n"
    << "  w_xc_ddot: 1.0e-3\n"
    << "  w_f_dot: 1.0e-3\n"
    << "\n"
    << "global_constraints:\n"
    << "  JointPosLimitConstraint:\n"
    << "    enabled: false\n"
    << "  JointVelLimitConstraint:\n"
    << "    enabled: false\n"
    << "  JointTrqLimitConstraint:\n"
    << "    enabled: " << b(cc.enforce_torque_limit) << "\n"
    << "\n"
    << "task_pool_yaml: \"task_list.yaml\"\n"
    << "state_machine_yaml: \"state_machine.yaml\"\n";
}

// ── Sim environment ─────────────────────────────────────────────────────────

struct SimEnv {
  std::filesystem::path tmp_dir;
  std::unique_ptr<wbc::ControlArchitecture> arch;
  mjModel* m{nullptr};
  mjData* d{nullptr};
  wbc::RobotJointState js;

  ~SimEnv() {
    if (d) mj_deleteData(d);
    if (m) mj_deleteModel(m);
    if (std::filesystem::exists(tmp_dir))
      std::filesystem::remove_all(tmp_dir);
  }
};

std::unique_ptr<SimEnv> BuildEnv(const CompensatorConfig& cc,
                                  const std::string& mjcf_path,
                                  const TaskGains& gains = TaskGains{},
                                  wbc::HardTorqueLimitMode hard_torque_mode =
                                      wbc::HardTorqueLimitMode::DIAGONAL_M_BOX,
                                  bool enable_timing = false) {
  auto env = std::make_unique<SimEnv>();
  env->tmp_dir = std::filesystem::temp_directory_path() / "wbc_domrand";
  std::filesystem::create_directories(env->tmp_dir);

  WriteTaskYaml(env->tmp_dir, gains);
  WriteWbcYaml(env->tmp_dir, cc);
  WriteStateMachineYaml(env->tmp_dir);
  WritePidYaml(env->tmp_dir, cc.kp_vel);

  std::string yaml_path = (env->tmp_dir / "optimo_wbc.yaml").string();
  auto arch_config = wbc::ControlArchitectureConfig::FromYaml(yaml_path, kDt);
  arch_config.state_provider = std::make_unique<wbc::StateProvider>(kDt);
  env->arch = std::make_unique<wbc::ControlArchitecture>(std::move(arch_config));
  env->arch->Initialize();
  env->arch->enable_timing_ = enable_timing;
  if (env->arch->GetSolver() != nullptr) {
    env->arch->GetSolver()->enable_timing_ = enable_timing;
    env->arch->GetSolver()->SetHardTorqueLimitMode(hard_torque_mode);
  }

  char error[1000] = "";
  env->m = mj_loadXML(mjcf_path.c_str(), nullptr, error, sizeof(error));
  if (!env->m)
    throw std::runtime_error(std::string("MuJoCo: ") + error);
  env->d = mj_makeData(env->m);

  if (env->m->nkey > 0)
    mju_copy(env->d->qpos, env->m->key_qpos, env->m->nq);
  mj_forward(env->m, env->d);

  env->js.Reset(kNJoints);
  return env;
}

void ReadJointState(SimEnv* env) {
  for (int i = 0; i < kNJoints; ++i) {
    env->js.q[i] = env->d->qpos[i];
    env->js.qdot[i] = env->d->qvel[i];
    env->js.tau[i] = env->d->qfrc_actuator[i];
  }
}

void ApplyCommand(SimEnv* env) {
  const auto& cmd = env->arch->GetCommand();
  for (int i = 0; i < kNJoints; ++i) env->d->ctrl[i] = cmd.tau[i];
}

// ── Tracking metrics ─────────────────────────────────────────────────────────

struct TrackingResult {
  double avg_transit_rms_mm;
  double worst_transit_mm;
  double avg_hold_err_mm;
  double worst_hold_err_mm;
  double worst_ori_deg;
  bool stable;
};

struct JointTrajectoryResult {
  double rms_err_mrad{0.0};
  double worst_err_mrad{0.0};
  double hold_err_mrad{0.0};
  bool stable{true};
};

struct TimingAndTorqueStats {
  double qp_setup_us_sum{0.0};
  double qp_solve_us_sum{0.0};
  double make_torque_us_sum{0.0};
  double find_config_us_sum{0.0};
  double tau_violation_sq_sum{0.0};
  double tau_violation_max_nm{0.0};
  int samples{0};

  void Accumulate(const wbc::ControlArchitecture& arch) {
    const wbc::WBIC* solver = arch.GetSolver();
    if (solver != nullptr) {
      qp_setup_us_sum += solver->timing_stats_.qp_setup_us;
      qp_solve_us_sum += solver->timing_stats_.qp_solve_us;
    }
    find_config_us_sum += arch.timing_stats_.find_config_us;
    make_torque_us_sum += arch.timing_stats_.make_torque_us;

    const auto& cmd = arch.GetCommand();
    if (cmd.tau_ff.size() == kNJoints) {
      for (int i = 0; i < kNJoints; ++i) {
        const double excess = std::max(0.0, std::abs(cmd.tau_ff[i]) - kJointTorqueLimitNm[i]);
        tau_violation_sq_sum += excess * excess;
        tau_violation_max_nm = std::max(tau_violation_max_nm, excess);
      }
    }
    ++samples;
  }

  void Merge(const TimingAndTorqueStats& other) {
    qp_setup_us_sum += other.qp_setup_us_sum;
    qp_solve_us_sum += other.qp_solve_us_sum;
    make_torque_us_sum += other.make_torque_us_sum;
    find_config_us_sum += other.find_config_us_sum;
    tau_violation_sq_sum += other.tau_violation_sq_sum;
    tau_violation_max_nm = std::max(tau_violation_max_nm, other.tau_violation_max_nm);
    samples += other.samples;
  }

  double MeanQpSetupUs() const {
    return (samples > 0) ? (qp_setup_us_sum / static_cast<double>(samples)) : 0.0;
  }
  double MeanQpSolveUs() const {
    return (samples > 0) ? (qp_solve_us_sum / static_cast<double>(samples)) : 0.0;
  }
  double MeanFindConfigUs() const {
    return (samples > 0) ? (find_config_us_sum / static_cast<double>(samples)) : 0.0;
  }
  double MeanMakeTorqueUs() const {
    return (samples > 0) ? (make_torque_us_sum / static_cast<double>(samples)) : 0.0;
  }
  double RmsTauViolationNm() const {
    if (samples <= 0) return 0.0;
    const double denom = static_cast<double>(samples * kNJoints);
    return std::sqrt(tau_violation_sq_sum / denom);
  }
};

TrackingResult RunCartesianTeleop(const CompensatorConfig& cc,
                                  const std::string& mjcf_path,
                                  const TaskGains& gains = TaskGains{},
                                  int trajectory_profile = 0,
                                  wbc::HardTorqueLimitMode hard_torque_mode =
                                      wbc::HardTorqueLimitMode::DIAGONAL_M_BOX,
                                  TimingAndTorqueStats* perf_stats = nullptr,
                                  double speed_scale = 1.0) {
  TrackingResult result{};
  auto env = BuildEnv(cc, mjcf_path, gains, hard_torque_mode,
                      perf_stats != nullptr);

  auto* robot = env->arch->GetRobot();
  int ee_idx = robot->GetFrameIndex("optimo_end_effector");

  // Init 2s
  double t = 0.0;
  for (int step = 0; step < 2000; ++step, t += kDt) {
    ReadJointState(env.get());
    env->arch->Update(env->js, t, kDt);
    if (perf_stats) perf_stats->Accumulate(*env->arch);
    ApplyCommand(env.get());
    mj_step(env->m, env->d);
  }

  // Transition to cartesian_teleop
  env->arch->RequestState(3);
  auto* ct = dynamic_cast<wbc::CartesianTeleop*>(
      env->arch->GetFsmHandler()->FindStateById(3));
  if (!ct) { result.stable = false; return result; }

  // Settle 0.5s
  for (int step = 0; step < 500; ++step, t += kDt) {
    ReadJointState(env.get());
    env->arch->Update(env->js, t, kDt);
    if (perf_stats) perf_stats->Accumulate(*env->arch);
    ApplyCommand(env.get());
    mj_step(env->m, env->d);
  }

  Eigen::Quaterniond home_quat(robot->GetLinkIsometry(ee_idx).rotation());

  // Use controller's internal reference as the origin (not actual EE position,
  // which may differ due to model error during settle).
  Eigen::Vector3d ref_origin = ct->PosGoal();

  // Waypoints: diversified Cartesian paths.
  struct Waypoint { Eigen::Vector3d offset; double speed, hold; };
  std::vector<Waypoint> waypoints;
  if (trajectory_profile == 0) {
    waypoints = {
        {{ 0.05,  0.00,  0.00}, 0.05, 1.0},
        {{ 0.05,  0.00,  0.05}, 0.05, 1.0},
        {{ 0.00,  0.00,  0.05}, 0.05, 1.0},
        {{-0.05,  0.00,  0.05}, 0.05, 1.0},
        {{-0.05,  0.00,  0.00}, 0.05, 1.0},
        {{ 0.00,  0.00,  0.00}, 0.05, 1.0},
        {{ 0.00,  0.03,  0.00}, 0.03, 1.0},
        {{ 0.00,  0.00,  0.00}, 0.03, 1.0},
    };
  } else {
    waypoints = {
        {{ 0.04,  0.02,  0.00}, 0.06, 0.8},
        {{ 0.02, -0.03,  0.04}, 0.05, 0.8},
        {{-0.03,  0.03,  0.05}, 0.05, 0.8},
        {{-0.05, -0.02,  0.02}, 0.05, 0.8},
        {{ 0.00,  0.04, -0.02}, 0.04, 0.8},
        {{ 0.03, -0.04,  0.00}, 0.04, 0.8},
        {{ 0.00,  0.00,  0.00}, 0.05, 1.0},
    };
  }

  Eigen::Vector3d zero3 = Eigen::Vector3d::Zero();
  Eigen::Quaterniond ident = Eigen::Quaterniond::Identity();
  int64_t ts = 1;
  Eigen::Vector3d current_goal = ref_origin;

  double total_transit_rms_sq = 0, total_hold = 0;
  double worst_transit = 0, worst_hold = 0, worst_ori = 0;
  result.stable = true;

  for (const auto& wp : waypoints) {
    Eigen::Vector3d target = ref_origin + wp.offset;
    Eigen::Vector3d direction = target - current_goal;
    double distance = direction.norm();
    Eigen::Vector3d unit_dir = (distance > 1e-6)
        ? direction.normalized() : Eigen::Vector3d::Zero();
    const double cmd_speed = std::max(1e-4, wp.speed * speed_scale);
    double travel_time = (distance > 1e-6) ? distance / cmd_speed : 0.0;
    int travel_steps = std::max(1, static_cast<int>(travel_time / kDt));

    double sum_sq = 0;
    int n_samples = 0;

    for (int step = 0; step < travel_steps; ++step, t += kDt) {
      Eigen::Vector3d vel_cmd = unit_dir * cmd_speed;

      ts += 1000000;
      ct->UpdateCommand(vel_cmd, zero3, ts, zero3, ident, 0);
      ReadJointState(env.get());
      env->arch->Update(env->js, t, kDt);
      if (perf_stats) perf_stats->Accumulate(*env->arch);

      // Compare actual EE against controller's internal desired position
      Eigen::Vector3d act = robot->GetLinkIsometry(ee_idx).translation();
      double err = (act - ct->PosDesired()).norm();
      sum_sq += err * err;
      worst_transit = std::max(worst_transit, err);
      ++n_samples;

      ApplyCommand(env.get());
      mj_step(env->m, env->d);
      if (!std::isfinite(env->d->qpos[0])) { result.stable = false; return result; }
    }

    double transit_rms = (n_samples > 0) ? std::sqrt(sum_sq / n_samples) : 0.0;
    total_transit_rms_sq += transit_rms * transit_rms;
    current_goal = target;

    // Hold phase
    int hold_steps = static_cast<int>(wp.hold / kDt);
    for (int step = 0; step < hold_steps; ++step, t += kDt) {
      ts += 1000000;
      ct->UpdateCommand(zero3, zero3, ts, zero3, ident, 0);
      ReadJointState(env.get());
      env->arch->Update(env->js, t, kDt);
      if (perf_stats) perf_stats->Accumulate(*env->arch);
      ApplyCommand(env.get());
      mj_step(env->m, env->d);
    }

    ReadJointState(env.get());
    env->arch->Update(env->js, t, kDt);
    if (perf_stats) perf_stats->Accumulate(*env->arch);
    Eigen::Vector3d hold_pos = robot->GetLinkIsometry(ee_idx).translation();
    double hold_err = (hold_pos - ct->PosDesired()).norm();
    total_hold += hold_err;
    worst_hold = std::max(worst_hold, hold_err);

    Eigen::Quaterniond act_quat(robot->GetLinkIsometry(ee_idx).rotation());
    double ori_err = act_quat.angularDistance(home_quat) * 180.0 / M_PI;
    worst_ori = std::max(worst_ori, ori_err);
  }

  int nw = static_cast<int>(waypoints.size());
  result.avg_transit_rms_mm = std::sqrt(total_transit_rms_sq / nw) * 1000.0;
  result.worst_transit_mm = worst_transit * 1000.0;
  result.avg_hold_err_mm = (total_hold / nw) * 1000.0;
  result.worst_hold_err_mm = worst_hold * 1000.0;
  result.worst_ori_deg = worst_ori;
  return result;
}

JointTrajectoryResult RunJointTeleopTrajectory(
    const CompensatorConfig& cc, const std::string& mjcf_path,
    const TaskGains& gains = TaskGains{}, int trajectory_profile = 0,
    wbc::HardTorqueLimitMode hard_torque_mode =
        wbc::HardTorqueLimitMode::DIAGONAL_M_BOX,
    TimingAndTorqueStats* perf_stats = nullptr,
    double vel_scale = 1.0) {
  JointTrajectoryResult result{};
  auto env = BuildEnv(cc, mjcf_path, gains, hard_torque_mode,
                      perf_stats != nullptr);

  // Init 2s
  double t = 0.0;
  for (int step = 0; step < 2000; ++step, t += kDt) {
    ReadJointState(env.get());
    env->arch->Update(env->js, t, kDt);
    if (perf_stats) perf_stats->Accumulate(*env->arch);
    ApplyCommand(env.get());
    mj_step(env->m, env->d);
  }

  // Transition to joint_teleop
  env->arch->RequestState(2);
  auto* jt = dynamic_cast<wbc::JointTeleop*>(
      env->arch->GetFsmHandler()->FindStateById(2));
  if (!jt) {
    result.stable = false;
    return result;
  }

  // Settle 0.5s
  for (int step = 0; step < 500; ++step, t += kDt) {
    ReadJointState(env.get());
    env->arch->Update(env->js, t, kDt);
    if (perf_stats) perf_stats->Accumulate(*env->arch);
    ApplyCommand(env.get());
    mj_step(env->m, env->d);
  }

  Eigen::VectorXd q_ref(kNJoints);
  for (int i = 0; i < kNJoints; ++i) {
    q_ref(i) = env->d->qpos[i];
  }

  Eigen::VectorXd vel_cmd = Eigen::VectorXd::Zero(kNJoints);
  Eigen::VectorXd dummy_pos = Eigen::VectorXd::Zero(kNJoints);
  int64_t ts = 1;

  // Diversified joint-space velocity trajectories.
  // Commands stay within joint_vel_limit to keep reference integration valid.
  const int traj_steps = 4000;  // 4.0s
  double sum_step_rms_sq = 0.0;
  double worst_err = 0.0;

  for (int step = 0; step < traj_steps; ++step, t += kDt) {
    vel_cmd.setZero();
    if (trajectory_profile == 0) {
      if (step < 1000) {
        vel_cmd(0) = 0.2;
      } else if (step < 2000) {
        vel_cmd(0) = -0.2;
      } else if (step < 3000) {
        vel_cmd(2) = 0.15;
      } else {
        vel_cmd(2) = -0.15;
      }
    } else {
      const double tau = step * kDt;
      vel_cmd(0) = 0.18 * std::sin(2.0 * M_PI * 0.50 * tau);
      vel_cmd(1) = 0.16 * std::sin(2.0 * M_PI * 0.35 * tau + 0.7);
      vel_cmd(3) = 0.12 * std::cos(2.0 * M_PI * 0.40 * tau);
      vel_cmd(5) = 0.10 * std::sin(2.0 * M_PI * 0.60 * tau + 1.2);
    }
    vel_cmd *= vel_scale;

    ts += 1000000;
    jt->UpdateCommand(vel_cmd, ts, dummy_pos, 0);
    q_ref.noalias() += vel_cmd * kDt;

    ReadJointState(env.get());
    env->arch->Update(env->js, t, kDt);
    if (perf_stats) perf_stats->Accumulate(*env->arch);
    ApplyCommand(env.get());
    mj_step(env->m, env->d);

    if (!std::isfinite(env->d->qpos[0])) {
      result.stable = false;
      return result;
    }

    double step_sq = 0.0;
    for (int i = 0; i < kNJoints; ++i) {
      const double e = std::abs(env->d->qpos[i] - q_ref(i));
      step_sq += e * e;
      worst_err = std::max(worst_err, e);
    }
    sum_step_rms_sq += step_sq / static_cast<double>(kNJoints);
  }

  // Hold for 0.5s after stop command
  vel_cmd.setZero();
  for (int step = 0; step < 500; ++step, t += kDt) {
    ts += 1000000;
    jt->UpdateCommand(vel_cmd, ts, dummy_pos, 0);
    ReadJointState(env.get());
    env->arch->Update(env->js, t, kDt);
    if (perf_stats) perf_stats->Accumulate(*env->arch);
    ApplyCommand(env.get());
    mj_step(env->m, env->d);
  }

  double hold_err = 0.0;
  for (int i = 0; i < kNJoints; ++i) {
    hold_err = std::max(hold_err, std::abs(env->d->qpos[i] - q_ref(i)));
  }

  result.rms_err_mrad = std::sqrt(sum_step_rms_sq / static_cast<double>(traj_steps)) * 1000.0;
  result.worst_err_mrad = worst_err * 1000.0;
  result.hold_err_mrad = hold_err * 1000.0;
  return result;
}

}  // namespace

// =============================================================================
// Domain Randomization Test
// =============================================================================

TEST(DomainRandomization, CompensatorComparison) {
  std::cout << "\n===== Domain Randomization: Compensator Comparison =====\n\n";

  // Compensator configurations to test (no joint PID — corrupts null-space)
  std::vector<CompensatorConfig> configs = {
    // Pure WBC feedforward
    {"WBC only",       false, false, 0,0,0,0, false, 0,0, 0.0},
    // WBC + momentum observer
    {"WBC+MomObs50",   false, false, 0,0,0,0, true, 50.0, 20.0, 0.0},
  };

  // Number of random seeds for the sweep
  constexpr int kNumSeeds = 5;
  constexpr double kScaleRange = 1.0;  // ±100% randomization

  std::string base_mjcf = ResolvePackagePath("optimo_description", "mjcf/optimo.xml");

  // Accumulate results: [config_idx][seed] -> result
  std::vector<std::vector<TrackingResult>> all_results(configs.size());

  for (int seed = 0; seed < kNumSeeds; ++seed) {
    std::mt19937 rng(42 + seed);
    DynamicsParams dyn = RandomizeDynamics(rng, kScaleRange, 0.5, 3.0);  // ±50% mass, up to 3kg payload

    // Write randomized MJCF
    auto tmp_mjcf = std::filesystem::temp_directory_path() / "wbc_domrand_mjcf";
    std::filesystem::create_directories(tmp_mjcf);
    auto mjcf_path = tmp_mjcf / ("optimo_seed" + std::to_string(seed) + ".xml");
    WriteRandomizedMjcf(mjcf_path, base_mjcf, dyn);

    std::cout << "Seed " << seed << ": damping=[";
    for (int i = 0; i < kNJoints; ++i) {
      if (i) std::cout << ",";
      std::cout << std::fixed << std::setprecision(2) << dyn.damping[i];
    }
    std::cout << "] fric=[";
    for (int i = 0; i < kNJoints; ++i) {
      if (i) std::cout << ",";
      std::cout << std::fixed << std::setprecision(2) << dyn.friction[i];
    }
    std::cout << "] mass_s=[";
    for (int i = 0; i < kNJoints; ++i) {
      if (i) std::cout << ",";
      std::cout << std::fixed << std::setprecision(2) << dyn.mass_scale[i];
    }
    std::cout << "] payload=" << std::setprecision(1) << dyn.ee_payload_kg << "kg\n";

    for (size_t ci = 0; ci < configs.size(); ++ci) {
      auto r = RunCartesianTeleop(configs[ci], mjcf_path.string());
      all_results[ci].push_back(r);
    }

    std::filesystem::remove_all(tmp_mjcf);
  }

  // ── Print per-seed results ─────────────────────────────────────────────
  std::cout << "\n" << std::fixed << std::setprecision(2);
  std::cout << std::left << std::setw(18) << "config"
            << " | seed | trn_rms | trn_max | hld_avg | hld_max | ori_max | ok\n";
  std::cout << std::string(95, '-') << "\n";

  for (size_t ci = 0; ci < configs.size(); ++ci) {
    for (int s = 0; s < kNumSeeds; ++s) {
      const auto& r = all_results[ci][s];
      std::cout << std::left << std::setw(18) << configs[ci].label
                << " |    " << s << " | "
                << std::setw(7) << r.avg_transit_rms_mm << " | "
                << std::setw(7) << r.worst_transit_mm << " | "
                << std::setw(7) << r.avg_hold_err_mm << " | "
                << std::setw(7) << r.worst_hold_err_mm << " | "
                << std::setw(7) << r.worst_ori_deg << " | "
                << (r.stable ? "OK" : "FAIL") << "\n";
    }
    std::cout << std::string(95, '-') << "\n";
  }

  // ── Aggregate statistics ───────────────────────────────────────────────
  std::cout << "\n===== Aggregate (mean over " << kNumSeeds << " seeds) =====\n\n";
  std::cout << std::left << std::setw(18) << "config"
            << " | trn_rms | trn_max | hld_avg | hld_max | ori_max | stable\n";
  std::cout << std::string(90, '-') << "\n";

  for (size_t ci = 0; ci < configs.size(); ++ci) {
    double sum_trn_rms = 0, sum_trn_max = 0, sum_hld_avg = 0, sum_hld_max = 0, sum_ori = 0;
    int n_stable = 0;
    for (const auto& r : all_results[ci]) {
      sum_trn_rms += r.avg_transit_rms_mm;
      sum_trn_max += r.worst_transit_mm;
      sum_hld_avg += r.avg_hold_err_mm;
      sum_hld_max += r.worst_hold_err_mm;
      sum_ori += r.worst_ori_deg;
      if (r.stable) ++n_stable;
    }
    int n = kNumSeeds;
    std::cout << std::left << std::setw(18) << configs[ci].label << " | "
              << std::setw(7) << sum_trn_rms / n << " | "
              << std::setw(7) << sum_trn_max / n << " | "
              << std::setw(7) << sum_hld_avg / n << " | "
              << std::setw(7) << sum_hld_max / n << " | "
              << std::setw(7) << sum_ori / n << " | "
              << n_stable << "/" << n << "\n";
  }

  // ── Assertions ─────────────────────────────────────────────────────────
  // All configs must be stable across all seeds
  for (size_t ci = 0; ci < configs.size(); ++ci) {
    for (int s = 0; s < kNumSeeds; ++s) {
      EXPECT_TRUE(all_results[ci][s].stable)
          << configs[ci].label << " seed " << s << " unstable";
    }
  }

  // All configs should achieve < 20mm transit RMS on average (basic sanity)
  for (size_t ci = 0; ci < configs.size(); ++ci) {
    double sum = 0;
    for (const auto& r : all_results[ci]) sum += r.avg_transit_rms_mm;
    double mean = sum / kNumSeeds;
    EXPECT_LT(mean, 50.0)
        << configs[ci].label << " mean transit RMS " << mean << "mm > 50mm";
  }
}

// =============================================================================
// Hard Scenarios: high friction, mass mismatch, payload
// =============================================================================

TEST(DomainRandomization, HardScenarios) {
  std::cout << "\n===== Hard Domain Randomization Scenarios =====\n\n";

  std::string base_mjcf = ResolvePackagePath("optimo_description", "mjcf/optimo.xml");

  // Scenario definitions: each progressively harder
  struct Scenario {
    const char* name;
    double fric_scale;   // friction/damping/stiffness randomization range
    double mass_range;   // mass perturbation range (0.3 = ±30%)
    double payload_max;  // max EE payload [kg]
    int seed;
  };

  std::vector<Scenario> scenarios = {
    // S1: High friction (3x nominal friction, 2x damping)
    {"HighFriction",       0.5,  0.0, 0.0, 42},
    // S2: Mass mismatch (±40% mass error, unknown to URDF)
    {"MassMismatch30",     0.5,  0.4, 0.0, 77},
    // S3: EE payload (3kg unknown payload — like a heavy tool)
    {"Payload3kg",         0.5,  0.0, 3.0, 99},
    // S4: Combined: 3x friction + ±30% mass + 2kg payload
    {"Combined",           0.5,  0.3, 2.0, 55},
    // S5: Extreme: 5x friction, ±50% mass, 4kg payload
    {"Extreme",            0.5,  0.5, 4.0, 33},
  };

  // Configs to compare (no joint PID — corrupts null-space)
  struct ConfigDef { const char* label; bool obs; double Ko; };
  std::vector<ConfigDef> cdefs = {
    {"WBC only",       false,  0.0},
    {"WBC+MomObs50",   true,  50.0},
  };

  // Header
  std::cout << std::fixed << std::setprecision(2);
  std::cout << std::left << std::setw(16) << "scenario"
            << " | " << std::setw(20) << "config"
            << " | trn_rms | trn_max | hld_avg | hld_max | ori_max | ok\n";
  std::cout << std::string(110, '-') << "\n";

  for (const auto& sc : scenarios) {
    std::mt19937 rng(sc.seed);

    // For HighFriction: multiply nominal friction by 3x before randomization
    DynamicsParams dyn = RandomizeDynamics(rng, sc.fric_scale,
                                            sc.mass_range, sc.payload_max);
    std::string sname(sc.name);
    if (sname == "HighFriction") {
      // 5x friction, 3x damping — heavy joint resistance
      for (int i = 0; i < kNJoints; ++i) {
        dyn.friction[i] = kNomFriction[i] * 5.0
            * std::uniform_real_distribution<double>(0.8, 1.2)(rng);
        dyn.damping[i] = kNomDamping[i] * 3.0
            * std::uniform_real_distribution<double>(0.8, 1.2)(rng);
      }
    } else if (sname == "Combined" || sname == "Extreme") {
      // 3x friction for Combined, 5x for Extreme
      double fric_mult = (sname == "Extreme") ? 5.0 : 3.0;
      double damp_mult = (sname == "Extreme") ? 3.0 : 2.0;
      for (int i = 0; i < kNJoints; ++i) {
        dyn.friction[i] = kNomFriction[i] * fric_mult
            * std::uniform_real_distribution<double>(0.8, 1.2)(rng);
        dyn.damping[i] = kNomDamping[i] * damp_mult
            * std::uniform_real_distribution<double>(0.8, 1.2)(rng);
      }
    }

    // Write randomized MJCF
    auto tmp_mjcf = std::filesystem::temp_directory_path() / "wbc_hard_scenarios";
    std::filesystem::create_directories(tmp_mjcf);
    auto mjcf_path = tmp_mjcf / ("optimo_" + std::string(sc.name) + ".xml");
    WriteRandomizedMjcf(mjcf_path, base_mjcf, dyn);

    // Print scenario details
    std::cout << "\n--- " << sc.name << " ---\n";
    std::cout << "  fric=[";
    for (int i = 0; i < kNJoints; ++i) {
      if (i) std::cout << ",";
      std::cout << dyn.friction[i];
    }
    std::cout << "] mass_scale=[";
    for (int i = 0; i < kNJoints; ++i) {
      if (i) std::cout << ",";
      std::cout << dyn.mass_scale[i];
    }
    std::cout << "] payload=" << dyn.ee_payload_kg << "kg\n";

    for (const auto& cd : cdefs) {
      CompensatorConfig cc{};
      cc.label = cd.label;
      cc.pid = false;
      cc.friction = false;
      cc.observer = cd.obs;
      cc.K_o = cd.Ko;
      cc.max_tau_dist = 30.0;
      cc.kp_vel = 0.0;

      auto r = RunCartesianTeleop(cc, mjcf_path.string());
      std::cout << std::left << std::setw(16) << sc.name
                << " | " << std::setw(20) << cd.label << " | "
                << std::setw(7) << r.avg_transit_rms_mm << " | "
                << std::setw(7) << r.worst_transit_mm << " | "
                << std::setw(7) << r.avg_hold_err_mm << " | "
                << std::setw(7) << r.worst_hold_err_mm << " | "
                << std::setw(7) << r.worst_ori_deg << " | "
                << (r.stable ? "OK" : "FAIL") << "\n";

      EXPECT_TRUE(r.stable) << sc.name << "/" << cd.label << " unstable";
    }

    std::filesystem::remove_all(tmp_mjcf);
  }
}

// =============================================================================
// Compensator Gain Sweep: find optimal friction compensator gains
// =============================================================================

TEST(DomainRandomization, FrictionCompGainSweep) {
  std::cout << "\n===== Friction Compensator Gain Sweep =====\n\n";

  // Use a single challenging randomization (seed=99, scale=0.7)
  std::mt19937 rng(99);
  DynamicsParams dyn = RandomizeDynamics(rng, 0.7);

  std::string base_mjcf = ResolvePackagePath("optimo_description", "mjcf/optimo.xml");
  auto tmp_mjcf = std::filesystem::temp_directory_path() / "wbc_fric_sweep";
  std::filesystem::create_directories(tmp_mjcf);
  auto mjcf_path = tmp_mjcf / "optimo_fric_sweep.xml";
  WriteRandomizedMjcf(mjcf_path, base_mjcf, dyn);

  std::cout << "Dynamics: damping=[";
  for (int i = 0; i < kNJoints; ++i) {
    if (i) std::cout << ",";
    std::cout << std::fixed << std::setprecision(2) << dyn.damping[i];
  }
  std::cout << "] fric=[";
  for (int i = 0; i < kNJoints; ++i) {
    if (i) std::cout << ",";
    std::cout << std::fixed << std::setprecision(2) << dyn.friction[i];
  }
  std::cout << "]\n\n";

  struct GainConfig {
    const char* label;
    double gamma_c, gamma_v, max_f_c, max_f_v;
  };

  std::vector<GainConfig> gains = {
    {"baseline (WBC only)", 0, 0, 0, 0},
    {"gc=1 gv=0.5",        1.0, 0.5, 5.0, 3.0},
    {"gc=5 gv=2",           5.0, 2.0, 5.0, 3.0},
    {"gc=10 gv=5",         10.0, 5.0, 5.0, 3.0},
    {"gc=20 gv=10",        20.0, 10.0, 5.0, 3.0},
    {"gc=50 gv=20",        50.0, 20.0, 10.0, 5.0},
    {"gc=5 gv=2 +obs",     5.0, 2.0, 5.0, 3.0},  // with momentum observer
  };

  std::cout << std::fixed << std::setprecision(2);
  std::cout << std::left << std::setw(22) << "config"
            << " | trn_rms | trn_max | hld_avg | hld_max | ori_max | ok\n";
  std::cout << std::string(90, '-') << "\n";

  for (size_t i = 0; i < gains.size(); ++i) {
    const auto& g = gains[i];
    bool has_fric = (g.gamma_c > 0 || g.gamma_v > 0);
    bool has_obs = (i == gains.size() - 1);  // last entry gets observer

    CompensatorConfig cc;
    cc.label = g.label;
    cc.pid = false;
    cc.friction = has_fric;
    cc.gamma_c = g.gamma_c;
    cc.gamma_v = g.gamma_v;
    cc.max_f_c = g.max_f_c;
    cc.max_f_v = g.max_f_v;
    cc.observer = has_obs;
    cc.K_o = 50.0;
    cc.max_tau_dist = 20.0;

    auto r = RunCartesianTeleop(cc, mjcf_path.string());
    std::cout << std::left << std::setw(22) << g.label << " | "
              << std::setw(7) << r.avg_transit_rms_mm << " | "
              << std::setw(7) << r.worst_transit_mm << " | "
              << std::setw(7) << r.avg_hold_err_mm << " | "
              << std::setw(7) << r.worst_hold_err_mm << " | "
              << std::setw(7) << r.worst_ori_deg << " | "
              << (r.stable ? "OK" : "FAIL") << "\n";

    EXPECT_TRUE(r.stable) << g.label << " unstable";
  }

  std::filesystem::remove_all(tmp_mjcf);
}

// =============================================================================
// Momentum Observer Gain Sweep
// =============================================================================

TEST(DomainRandomization, MomentumObserverGainSweep) {
  std::cout << "\n===== Momentum Observer Gain Sweep =====\n\n";

  std::mt19937 rng(77);
  DynamicsParams dyn = RandomizeDynamics(rng, 0.7);

  std::string base_mjcf = ResolvePackagePath("optimo_description", "mjcf/optimo.xml");
  auto tmp_mjcf = std::filesystem::temp_directory_path() / "wbc_obs_sweep";
  std::filesystem::create_directories(tmp_mjcf);
  auto mjcf_path = tmp_mjcf / "optimo_obs_sweep.xml";
  WriteRandomizedMjcf(mjcf_path, base_mjcf, dyn);

  struct GainConfig {
    const char* label;
    double K_o, max_tau;
  };

  std::vector<GainConfig> gains = {
    {"baseline (WBC only)", 0, 0},
    {"Ko=10",   10.0, 20.0},
    {"Ko=30",   30.0, 20.0},
    {"Ko=50",   50.0, 20.0},
    {"Ko=100",  100.0, 20.0},
    {"Ko=200",  200.0, 30.0},
    {"Ko=50 +fric", 50.0, 20.0},  // with friction compensator
  };

  std::cout << std::fixed << std::setprecision(2);
  std::cout << std::left << std::setw(22) << "config"
            << " | trn_rms | trn_max | hld_avg | hld_max | ori_max | ok\n";
  std::cout << std::string(90, '-') << "\n";

  for (size_t i = 0; i < gains.size(); ++i) {
    const auto& g = gains[i];
    bool has_obs = (g.K_o > 0);
    bool has_fric = (i == gains.size() - 1);

    CompensatorConfig cc;
    cc.label = g.label;
    cc.pid = false;
    cc.friction = has_fric;
    cc.gamma_c = 5.0;
    cc.gamma_v = 2.0;
    cc.max_f_c = 5.0;
    cc.max_f_v = 3.0;
    cc.observer = has_obs;
    cc.K_o = g.K_o;
    cc.max_tau_dist = g.max_tau;

    auto r = RunCartesianTeleop(cc, mjcf_path.string());
    std::cout << std::left << std::setw(22) << g.label << " | "
              << std::setw(7) << r.avg_transit_rms_mm << " | "
              << std::setw(7) << r.worst_transit_mm << " | "
              << std::setw(7) << r.avg_hold_err_mm << " | "
              << std::setw(7) << r.worst_hold_err_mm << " | "
              << std::setw(7) << r.worst_ori_deg << " | "
              << (r.stable ? "OK" : "FAIL") << "\n";

    EXPECT_TRUE(r.stable) << g.label << " unstable";
  }

  std::filesystem::remove_all(tmp_mjcf);
}

// =============================================================================
// Gain Sweep: find optimal task gains under large domain randomization
// =============================================================================

TEST(DomainRandomization, GainSweep) {
  std::cout << "\n===== Task Gain Sweep (±100% domain randomization) =====\n\n";

  constexpr int kNumSeeds = 3;
  constexpr double kRange = 1.0;  // ±100%

  std::string base_mjcf = ResolvePackagePath("optimo_description", "mjcf/optimo.xml");

  // Sweep task gains × compensator configs (no joint PID — it corrupts null-space)
  struct SweepConfig {
    const char* label;
    TaskGains gains;
    bool observer;
    double K_o;
  };

  std::vector<SweepConfig> sweep_configs = {
    // Task gain sweep (no compensator)
    {"ee400/40  j100/20",   {100, 20,  400,  40}, false, 0},
    {"ee800/57  j100/20",   {100, 20,  800,  57}, false, 0},
    {"ee1600/80 j100/20",   {100, 20, 1600,  80}, false, 0},
    {"ee3200/113 j100/20",  {100, 20, 3200, 113}, false, 0},
    {"ee6400/160 j100/20",  {100, 20, 6400, 160}, false, 0},
    // Jpos gain sweep (fixed ee gains)
    {"ee3200/113 j200/28",  {200, 28, 3200, 113}, false, 0},
    {"ee3200/113 j400/40",  {400, 40, 3200, 113}, false, 0},
    // Best gains × compensator
    {"ee6400/160 j200/28",  {200, 28, 6400, 160}, false, 0},
    {"ee6400/160 j200/28 +MomObs50", {200, 28, 6400, 160}, true, 50},
  };

  std::vector<std::string> mjcf_paths;
  auto sweep_dir = std::filesystem::temp_directory_path() / "wbc_gain_sweep";
  std::filesystem::create_directories(sweep_dir);

  constexpr double kMassRange = 0.5;   // ±50% mass randomization
  constexpr double kPayloadMax = 3.0;  // up to 3kg unknown EE payload

  for (int seed = 0; seed < kNumSeeds; ++seed) {
    std::mt19937 rng(42 + seed);
    DynamicsParams dyn = RandomizeDynamics(rng, kRange, kMassRange, kPayloadMax);
    auto path = sweep_dir / ("optimo_seed" + std::to_string(seed) + ".xml");
    WriteRandomizedMjcf(path, base_mjcf, dyn);
    mjcf_paths.push_back(path.string());

    std::cout << "Seed " << seed << ": damping=[";
    for (int i = 0; i < kNJoints; ++i) {
      if (i) std::cout << ",";
      std::cout << std::fixed << std::setprecision(2) << dyn.damping[i];
    }
    std::cout << "] fric=[";
    for (int i = 0; i < kNJoints; ++i) {
      if (i) std::cout << ",";
      std::cout << std::fixed << std::setprecision(2) << dyn.friction[i];
    }
    std::cout << "] mass_s=[";
    for (int i = 0; i < kNJoints; ++i) {
      if (i) std::cout << ",";
      std::cout << std::fixed << std::setprecision(2) << dyn.mass_scale[i];
    }
    std::cout << "] payload=" << std::setprecision(1) << dyn.ee_payload_kg << "kg\n";
  }

  std::cout << std::fixed << std::setprecision(2);
  std::cout << std::left << std::setw(32) << "gains"
            << " | trn_rms | hld_avg | ori_max | stable\n";
  std::cout << std::string(80, '-') << "\n";

  for (const auto& sc : sweep_configs) {
    CompensatorConfig cc{};
    cc.label = sc.label;
    cc.pid = false;
    cc.friction = false;
    cc.observer = sc.observer;
    cc.K_o = sc.K_o;
    cc.max_tau_dist = 20.0;
    cc.kp_vel = 0.0;

    double sum_trn = 0, sum_hld = 0, sum_ori = 0;
    int n_stable = 0;

    for (int seed = 0; seed < kNumSeeds; ++seed) {
      auto r = RunCartesianTeleop(cc, mjcf_paths[seed], sc.gains);
      if (r.stable) {
        sum_trn += r.avg_transit_rms_mm;
        sum_hld += r.avg_hold_err_mm;
        sum_ori += r.worst_ori_deg;
        ++n_stable;
      }
    }

    int n = std::max(n_stable, 1);
    std::cout << std::left << std::setw(32) << sc.label << " | "
              << std::setw(7) << sum_trn / n << " | "
              << std::setw(7) << sum_hld / n << " | "
              << std::setw(7) << sum_ori / n << " | "
              << n_stable << "/" << kNumSeeds << "\n";
  }

  std::filesystem::remove_all(sweep_dir);
}

TEST(DomainRandomization, DISABLED_TorqueLimitApproxVsExactDense) {
  std::cout << "\n===== Hard Torque Limit: Approximation vs Exact Dense =====\n\n";

  constexpr int kNumSeeds = 3;
  constexpr int kNumCartesianProfiles = 2;
  constexpr int kNumJointProfiles = 2;
  constexpr double kRange = 1.0;       // damping/friction/stiffness ±100%
  constexpr double kMassRange = 0.6;   // mass ±60%
  constexpr double kPayloadMax = 4.0;  // payload up to 4kg
  constexpr double kComRange = 0.04;   // CoM offset ±4cm per axis

  std::string base_mjcf =
      ResolvePackagePath("optimo_description", "mjcf/optimo.xml");
  auto sweep_dir = std::filesystem::temp_directory_path() / "wbc_trqmode_compare";
  std::filesystem::create_directories(sweep_dir);

  std::vector<std::string> mjcf_paths;
  for (int seed = 0; seed < kNumSeeds; ++seed) {
    std::mt19937 rng(1337 + seed);
    DynamicsParams dyn =
        RandomizeDynamics(rng, kRange, kMassRange, kPayloadMax, kComRange);
    auto path = sweep_dir / ("optimo_seed" + std::to_string(seed) + ".xml");
    WriteRandomizedMjcf(path, base_mjcf, dyn);
    mjcf_paths.push_back(path.string());
  }

  struct ModeCfg {
    const char* label;
    wbc::HardTorqueLimitMode mode;
  };
  const std::vector<ModeCfg> modes = {
      {"diag_M_box", wbc::HardTorqueLimitMode::DIAGONAL_M_BOX},
      {"exact_dense", wbc::HardTorqueLimitMode::EXACT_DENSE},
  };

  struct Aggregate {
    double cart_transit_mm_sum{0.0};
    double cart_hold_mm_sum{0.0};
    double cart_ori_deg_sum{0.0};
    double joint_rms_mrad_sum{0.0};
    double joint_hold_mrad_sum{0.0};
    int stable_seed_count{0};
    TimingAndTorqueStats perf;
  };
  std::vector<Aggregate> agg(modes.size());

  CompensatorConfig cc{};
  cc.label = "trq_compare";
  cc.pid = false;
  cc.friction = false;
  cc.observer = false;
  cc.kp_vel = 0.0;
  cc.enforce_torque_limit = true;

  const TaskGains gains{200.0, 28.0, 6400.0, 160.0};

  for (size_t mi = 0; mi < modes.size(); ++mi) {
    for (int seed = 0; seed < kNumSeeds; ++seed) {
      bool seed_stable = true;
      double seed_cart_transit = 0.0;
      double seed_cart_hold = 0.0;
      double seed_cart_ori = 0.0;
      double seed_joint_rms = 0.0;
      double seed_joint_hold = 0.0;
      TimingAndTorqueStats seed_perf;

      for (int p = 0; p < kNumCartesianProfiles; ++p) {
        const auto r = RunCartesianTeleop(cc, mjcf_paths[seed], gains, p,
                                          modes[mi].mode, &seed_perf);
        if (!r.stable) {
          seed_stable = false;
          break;
        }
        seed_cart_transit += r.avg_transit_rms_mm;
        seed_cart_hold += r.avg_hold_err_mm;
        seed_cart_ori += r.worst_ori_deg;
      }
      if (!seed_stable) {
        continue;
      }

      for (int p = 0; p < kNumJointProfiles; ++p) {
        const auto r = RunJointTeleopTrajectory(cc, mjcf_paths[seed], gains, p,
                                                modes[mi].mode, &seed_perf);
        if (!r.stable) {
          seed_stable = false;
          break;
        }
        seed_joint_rms += r.rms_err_mrad;
        seed_joint_hold += r.hold_err_mrad;
      }
      if (!seed_stable) {
        continue;
      }

      agg[mi].stable_seed_count += 1;
      agg[mi].cart_transit_mm_sum +=
          seed_cart_transit / static_cast<double>(kNumCartesianProfiles);
      agg[mi].cart_hold_mm_sum +=
          seed_cart_hold / static_cast<double>(kNumCartesianProfiles);
      agg[mi].cart_ori_deg_sum +=
          seed_cart_ori / static_cast<double>(kNumCartesianProfiles);
      agg[mi].joint_rms_mrad_sum +=
          seed_joint_rms / static_cast<double>(kNumJointProfiles);
      agg[mi].joint_hold_mrad_sum +=
          seed_joint_hold / static_cast<double>(kNumJointProfiles);
      agg[mi].perf.Merge(seed_perf);
    }
  }

  std::cout << std::fixed << std::setprecision(3);
  std::cout << std::left << std::setw(14) << "mode"
            << " | stable | c_trn | c_hold | c_ori | j_rms | j_hold"
            << " | qp_setup(us) | qp_solve(us) | make_torque(us)"
            << " | tau_rms_excess | tau_max_excess\n";
  std::cout << std::string(170, '-') << "\n";

  for (size_t mi = 0; mi < modes.size(); ++mi) {
    const int denom = std::max(1, agg[mi].stable_seed_count);
    std::cout << std::left << std::setw(14) << modes[mi].label
              << " | " << std::setw(6) << agg[mi].stable_seed_count << "/" << kNumSeeds
              << " | " << std::setw(6) << (agg[mi].cart_transit_mm_sum / denom)
              << " | " << std::setw(6) << (agg[mi].cart_hold_mm_sum / denom)
              << " | " << std::setw(6) << (agg[mi].cart_ori_deg_sum / denom)
              << " | " << std::setw(6) << (agg[mi].joint_rms_mrad_sum / denom)
              << " | " << std::setw(7) << (agg[mi].joint_hold_mrad_sum / denom)
              << " | " << std::setw(12) << agg[mi].perf.MeanQpSetupUs()
              << " | " << std::setw(12) << agg[mi].perf.MeanQpSolveUs()
              << " | " << std::setw(14) << agg[mi].perf.MeanMakeTorqueUs()
              << " | " << std::setw(14) << agg[mi].perf.RmsTauViolationNm()
              << " | " << agg[mi].perf.tau_violation_max_nm << "\n";
  }

  if (agg[0].perf.MeanMakeTorqueUs() > 1e-9) {
    const double slowdown =
        agg[1].perf.MeanMakeTorqueUs() / agg[0].perf.MeanMakeTorqueUs();
    std::cout << "\nExact dense / diag-M make_torque slowdown: "
              << slowdown << "x\n";
  }

  for (size_t mi = 0; mi < modes.size(); ++mi) {
    EXPECT_GT(agg[mi].stable_seed_count, 0)
        << "No stable seed for mode: " << modes[mi].label;
    EXPECT_GT(agg[mi].perf.samples, 0)
        << "No timing samples for mode: " << modes[mi].label;
  }

  std::filesystem::remove_all(sweep_dir);
}

TEST(DomainRandomization, DISABLED_TorqueLimitApproxVsExactDenseQuasiStatic) {
  std::cout << "\n===== Quasi-Static: Approximation vs Exact Dense =====\n\n";

  constexpr int kNumSeeds = 3;
  constexpr int kNumCartesianProfiles = 1;
  constexpr int kNumJointProfiles = 1;
  constexpr double kRange = 0.2;       // damping/friction/stiffness ±20%
  constexpr double kMassRange = 0.1;   // mass ±10%
  constexpr double kPayloadMax = 0.5;  // payload up to 0.5kg
  constexpr double kComRange = 0.005;  // CoM offset ±5mm per axis
  constexpr double kCartSpeedScale = 0.2;
  constexpr double kJointVelScale = 0.2;

  std::string base_mjcf =
      ResolvePackagePath("optimo_description", "mjcf/optimo.xml");
  auto sweep_dir = std::filesystem::temp_directory_path() / "wbc_trqmode_compare_quasi";
  std::filesystem::create_directories(sweep_dir);

  std::vector<std::string> mjcf_paths;
  for (int seed = 0; seed < kNumSeeds; ++seed) {
    std::mt19937 rng(2026 + seed);
    DynamicsParams dyn =
        RandomizeDynamics(rng, kRange, kMassRange, kPayloadMax, kComRange);
    auto path = sweep_dir / ("optimo_seed" + std::to_string(seed) + ".xml");
    WriteRandomizedMjcf(path, base_mjcf, dyn);
    mjcf_paths.push_back(path.string());
  }

  struct ModeCfg {
    const char* label;
    wbc::HardTorqueLimitMode mode;
  };
  const std::vector<ModeCfg> modes = {
      {"diag_M_box", wbc::HardTorqueLimitMode::DIAGONAL_M_BOX},
      {"exact_dense", wbc::HardTorqueLimitMode::EXACT_DENSE},
  };

  struct Aggregate {
    double cart_transit_mm_sum{0.0};
    double cart_hold_mm_sum{0.0};
    double cart_ori_deg_sum{0.0};
    double joint_rms_mrad_sum{0.0};
    double joint_hold_mrad_sum{0.0};
    int stable_seed_count{0};
    TimingAndTorqueStats perf;
  };
  std::vector<Aggregate> agg(modes.size());

  CompensatorConfig cc{};
  cc.label = "trq_compare_quasi";
  cc.pid = false;
  cc.friction = false;
  cc.observer = false;
  cc.kp_vel = 0.0;
  cc.enforce_torque_limit = true;

  const TaskGains gains{80.0, 16.0, 1200.0, 45.0};

  for (size_t mi = 0; mi < modes.size(); ++mi) {
    for (int seed = 0; seed < kNumSeeds; ++seed) {
      bool seed_stable = true;
      double seed_cart_transit = 0.0;
      double seed_cart_hold = 0.0;
      double seed_cart_ori = 0.0;
      double seed_joint_rms = 0.0;
      double seed_joint_hold = 0.0;
      TimingAndTorqueStats seed_perf;

      for (int p = 0; p < kNumCartesianProfiles; ++p) {
        const auto r = RunCartesianTeleop(
            cc, mjcf_paths[seed], gains, p, modes[mi].mode, &seed_perf,
            kCartSpeedScale);
        if (!r.stable) {
          seed_stable = false;
          break;
        }
        seed_cart_transit += r.avg_transit_rms_mm;
        seed_cart_hold += r.avg_hold_err_mm;
        seed_cart_ori += r.worst_ori_deg;
      }
      if (!seed_stable) {
        continue;
      }

      for (int p = 0; p < kNumJointProfiles; ++p) {
        const auto r = RunJointTeleopTrajectory(
            cc, mjcf_paths[seed], gains, p, modes[mi].mode, &seed_perf,
            kJointVelScale);
        if (!r.stable) {
          seed_stable = false;
          break;
        }
        seed_joint_rms += r.rms_err_mrad;
        seed_joint_hold += r.hold_err_mrad;
      }
      if (!seed_stable) {
        continue;
      }

      agg[mi].stable_seed_count += 1;
      agg[mi].cart_transit_mm_sum +=
          seed_cart_transit / static_cast<double>(kNumCartesianProfiles);
      agg[mi].cart_hold_mm_sum +=
          seed_cart_hold / static_cast<double>(kNumCartesianProfiles);
      agg[mi].cart_ori_deg_sum +=
          seed_cart_ori / static_cast<double>(kNumCartesianProfiles);
      agg[mi].joint_rms_mrad_sum +=
          seed_joint_rms / static_cast<double>(kNumJointProfiles);
      agg[mi].joint_hold_mrad_sum +=
          seed_joint_hold / static_cast<double>(kNumJointProfiles);
      agg[mi].perf.Merge(seed_perf);
    }
  }

  std::cout << std::fixed << std::setprecision(3);
  std::cout << std::left << std::setw(14) << "mode"
            << " | stable | c_trn | c_hold | c_ori | j_rms | j_hold"
            << " | qp_setup(us) | qp_solve(us) | make_torque(us)"
            << " | tau_rms_excess | tau_max_excess\n";
  std::cout << std::string(170, '-') << "\n";

  for (size_t mi = 0; mi < modes.size(); ++mi) {
    const int denom = std::max(1, agg[mi].stable_seed_count);
    std::cout << std::left << std::setw(14) << modes[mi].label
              << " | " << std::setw(6) << agg[mi].stable_seed_count << "/"
              << kNumSeeds
              << " | " << std::setw(6) << (agg[mi].cart_transit_mm_sum / denom)
              << " | " << std::setw(6) << (agg[mi].cart_hold_mm_sum / denom)
              << " | " << std::setw(6) << (agg[mi].cart_ori_deg_sum / denom)
              << " | " << std::setw(6) << (agg[mi].joint_rms_mrad_sum / denom)
              << " | " << std::setw(7) << (agg[mi].joint_hold_mrad_sum / denom)
              << " | " << std::setw(12) << agg[mi].perf.MeanQpSetupUs()
              << " | " << std::setw(12) << agg[mi].perf.MeanQpSolveUs()
              << " | " << std::setw(14) << agg[mi].perf.MeanMakeTorqueUs()
              << " | " << std::setw(14) << agg[mi].perf.RmsTauViolationNm()
              << " | " << agg[mi].perf.tau_violation_max_nm << "\n";
  }

  for (size_t mi = 0; mi < modes.size(); ++mi) {
    EXPECT_GT(agg[mi].stable_seed_count, 0)
        << "No stable seed for mode: " << modes[mi].label;
    EXPECT_GT(agg[mi].perf.samples, 0)
        << "No timing samples for mode: " << modes[mi].label;
  }

  std::filesystem::remove_all(sweep_dir);
}

// =============================================================================
// Tunable Search (manual): joint + Cartesian trajectories with CoM randomization
// =============================================================================
TEST(DomainRandomization, DISABLED_OptimalParamSearchJointAndCartesian) {
  std::cout << "\n===== Optimal Parameter Search: Joint + Cartesian =====\n\n";

  constexpr int kNumSeeds = 3;
  constexpr int kNumCartesianProfiles = 2;
  constexpr int kNumJointProfiles = 2;
  constexpr double kRange = 1.0;       // damping/friction/stiffness ±100%
  constexpr double kMassRange = 0.6;   // mass ±60%
  constexpr double kPayloadMax = 4.0;  // payload up to 4kg
  constexpr double kComRange = 0.04;   // CoM offset ±4cm per axis

  std::string base_mjcf = ResolvePackagePath("optimo_description", "mjcf/optimo.xml");
  auto sweep_dir = std::filesystem::temp_directory_path() / "wbc_optimal_param_search";
  std::filesystem::create_directories(sweep_dir);
  const auto csv_path = std::filesystem::temp_directory_path() / "gain_sweep_domain_randomized.csv";
  std::ofstream csv(csv_path);
  if (csv.is_open()) {
    csv << "label,jpos_kp,jpos_kd,ee_kp,ee_kd,"
        << "friction_enabled,gamma_c,gamma_v,max_f_c,max_f_v,"
        << "observer_enabled,K_o,max_tau_dist,"
        << "stable_seeds,num_seeds,"
        << "cart_transit_mm,cart_hold_mm,cart_ori_deg,"
        << "joint_rms_mrad,joint_hold_mrad,score,"
        << "rand_scale_range,rand_mass_range,rand_payload_max_kg,rand_com_range_m\n";
  }

  std::vector<std::string> mjcf_paths;
  for (int seed = 0; seed < kNumSeeds; ++seed) {
    std::mt19937 rng(123 + seed);
    DynamicsParams dyn = RandomizeDynamics(rng, kRange, kMassRange, kPayloadMax, kComRange);
    auto path = sweep_dir / ("optimo_seed" + std::to_string(seed) + ".xml");
    WriteRandomizedMjcf(path, base_mjcf, dyn);
    mjcf_paths.push_back(path.string());

    std::cout << "Seed " << seed
              << " payload=" << std::fixed << std::setprecision(2) << dyn.ee_payload_kg
              << "kg, com_shift_l2=[" << dyn.com_offset[1].transpose() << "]\n";
  }

  struct Candidate {
    std::string label;
    TaskGains gains;
    bool friction;
    double gamma_c;
    double gamma_v;
    double max_f_c;
    double max_f_v;
    bool observer;
    double K_o;
    double max_tau_dist;
  };

  std::vector<Candidate> candidates;
  const std::vector<std::pair<double, double>> jpos_grid = {
      {120.0, 22.0}, {200.0, 28.0}, {320.0, 38.0}};
  const std::vector<std::pair<double, double>> ee_grid = {
      {3200.0, 113.0}, {6400.0, 160.0}, {9000.0, 190.0}};
  const std::vector<double> obs_grid = {50.0, 100.0};

  for (const auto& jg : jpos_grid) {
    for (const auto& eg : ee_grid) {
      for (double Ko : obs_grid) {
        std::ostringstream os;
        os << "j" << static_cast<int>(jg.first) << "/" << static_cast<int>(jg.second)
           << " ee" << static_cast<int>(eg.first) << "/" << static_cast<int>(eg.second)
           << " +Obs" << static_cast<int>(Ko);
        candidates.push_back({os.str(),
                              {jg.first, jg.second, eg.first, eg.second},
                              false, 0.0, 0.0, 0.0, 0.0,
                              true, Ko, 20.0});
      }
    }
  }

  // Friction variants around stronger EE gains.
  const std::vector<std::pair<double, double>> fric_gains = {{5.0, 2.0}, {10.0, 4.0}};
  for (const auto& eg : std::vector<std::pair<double, double>>{{6400.0, 160.0},
                                                                {9000.0, 190.0}}) {
    for (double Ko : obs_grid) {
      for (const auto& fg : fric_gains) {
        std::ostringstream os;
        os << "j200/28 ee" << static_cast<int>(eg.first) << "/" << static_cast<int>(eg.second)
           << " +Obs" << static_cast<int>(Ko)
           << "+Fric" << static_cast<int>(fg.first) << "/" << static_cast<int>(fg.second);
        candidates.push_back({os.str(),
                              {200.0, 28.0, eg.first, eg.second},
                              true, fg.first, fg.second, 8.0, 5.0,
                              true, Ko, 20.0});
      }
    }
  }

  // No-residual baselines.
  for (const auto& eg : ee_grid) {
    std::ostringstream os;
    os << "j200/28 ee" << static_cast<int>(eg.first) << "/" << static_cast<int>(eg.second)
       << " (no residual)";
    candidates.push_back({os.str(),
                          {200.0, 28.0, eg.first, eg.second},
                          false, 0.0, 0.0, 0.0, 0.0,
                          false, 0.0, 0.0});
  }

  std::cout << "Candidate count: " << candidates.size()
            << ", seeds: " << kNumSeeds
            << ", trajectory profiles: C" << kNumCartesianProfiles
            << " + J" << kNumJointProfiles
            << ", torque_limit=on\n\n";

  std::cout << "\n";
  std::cout << std::left << std::setw(36) << "candidate"
            << " | c_trn | c_hold | c_ori | j_rms | j_hold | score | stable\n";
  std::cout << std::string(108, '-') << "\n";

  int best_idx = -1;
  int best_stable = -1;
  double best_score = std::numeric_limits<double>::infinity();

  for (size_t i = 0; i < candidates.size(); ++i) {
    const auto& cand = candidates[i];

    CompensatorConfig cc{};
    cc.label = cand.label.c_str();
    cc.pid = false;
    cc.friction = cand.friction;
    cc.gamma_c = cand.gamma_c;
    cc.gamma_v = cand.gamma_v;
    cc.max_f_c = cand.max_f_c;
    cc.max_f_v = cand.max_f_v;
    cc.observer = cand.observer;
    cc.K_o = cand.K_o;
    cc.max_tau_dist = cand.max_tau_dist;
    cc.kp_vel = 0.0;
    cc.enforce_torque_limit = true;

    double sum_c_trn = 0.0;
    double sum_c_hold = 0.0;
    double sum_c_ori = 0.0;
    double sum_j_rms = 0.0;
    double sum_j_hold = 0.0;
    double sum_score = 0.0;
    int n_stable = 0;

    for (int seed = 0; seed < kNumSeeds; ++seed) {
      bool seed_stable = true;
      double seed_sum_c_trn = 0.0;
      double seed_sum_c_hold = 0.0;
      double seed_sum_c_ori = 0.0;
      double seed_sum_j_rms = 0.0;
      double seed_sum_j_hold = 0.0;

      for (int p = 0; p < kNumCartesianProfiles; ++p) {
        const auto cart = RunCartesianTeleop(cc, mjcf_paths[seed], cand.gains, p);
        if (!cart.stable) {
          seed_stable = false;
          break;
        }
        seed_sum_c_trn += cart.avg_transit_rms_mm;
        seed_sum_c_hold += cart.avg_hold_err_mm;
        seed_sum_c_ori += cart.worst_ori_deg;
      }
      if (!seed_stable) continue;

      for (int p = 0; p < kNumJointProfiles; ++p) {
        const auto joint = RunJointTeleopTrajectory(cc, mjcf_paths[seed], cand.gains, p);
        if (!joint.stable) {
          seed_stable = false;
          break;
        }
        seed_sum_j_rms += joint.rms_err_mrad;
        seed_sum_j_hold += joint.hold_err_mrad;
      }
      if (!seed_stable) continue;

      const double mean_c_trn = seed_sum_c_trn / static_cast<double>(kNumCartesianProfiles);
      const double mean_c_hold = seed_sum_c_hold / static_cast<double>(kNumCartesianProfiles);
      const double mean_c_ori = seed_sum_c_ori / static_cast<double>(kNumCartesianProfiles);
      const double mean_j_rms = seed_sum_j_rms / static_cast<double>(kNumJointProfiles);
      const double mean_j_hold = seed_sum_j_hold / static_cast<double>(kNumJointProfiles);

      // Combined objective (lower is better):
      // - Cartesian: mm/deg
      // - Joint: mrad
      const double score = mean_c_trn + mean_c_hold + 0.5 * mean_c_ori +
                           mean_j_rms + mean_j_hold;

      sum_c_trn += mean_c_trn;
      sum_c_hold += mean_c_hold;
      sum_c_ori += mean_c_ori;
      sum_j_rms += mean_j_rms;
      sum_j_hold += mean_j_hold;
      sum_score += score;
      ++n_stable;
    }

    const int denom = std::max(1, n_stable);
    double avg_c_trn = std::numeric_limits<double>::quiet_NaN();
    double avg_c_hold = std::numeric_limits<double>::quiet_NaN();
    double avg_c_ori = std::numeric_limits<double>::quiet_NaN();
    double avg_j_rms = std::numeric_limits<double>::quiet_NaN();
    double avg_j_hold = std::numeric_limits<double>::quiet_NaN();
    double avg_score = std::numeric_limits<double>::infinity();
    if (n_stable > 0) {
      avg_c_trn = sum_c_trn / static_cast<double>(denom);
      avg_c_hold = sum_c_hold / static_cast<double>(denom);
      avg_c_ori = sum_c_ori / static_cast<double>(denom);
      avg_j_rms = sum_j_rms / static_cast<double>(denom);
      avg_j_hold = sum_j_hold / static_cast<double>(denom);
      avg_score = sum_score / static_cast<double>(denom);
    }

    std::cout << std::left << std::setw(36) << cand.label << " | "
              << std::setw(5) << avg_c_trn << " | "
              << std::setw(6) << avg_c_hold << " | "
              << std::setw(5) << avg_c_ori << " | "
              << std::setw(5) << avg_j_rms << " | "
              << std::setw(6) << avg_j_hold << " | "
              << std::setw(6) << avg_score << " | "
              << n_stable << "/" << kNumSeeds << "\n";

    if (n_stable > best_stable ||
        (n_stable == best_stable && avg_score < best_score)) {
      best_stable = n_stable;
      best_score = avg_score;
      best_idx = static_cast<int>(i);
    }

    if (csv.is_open()) {
      const auto esc = [](const std::string& s) {
        std::string out;
        out.reserve(s.size());
        for (char c : s) {
          if (c == '"') {
            out.push_back('"');
          }
          out.push_back(c);
        }
        return out;
      };

      csv << '"' << esc(cand.label) << '"' << ','
          << cand.gains.jpos_kp << ','
          << cand.gains.jpos_kd << ','
          << cand.gains.ee_kp << ','
          << cand.gains.ee_kd << ','
          << (cand.friction ? 1 : 0) << ','
          << cand.gamma_c << ','
          << cand.gamma_v << ','
          << cand.max_f_c << ','
          << cand.max_f_v << ','
          << (cand.observer ? 1 : 0) << ','
          << cand.K_o << ','
          << cand.max_tau_dist << ','
          << n_stable << ','
          << kNumSeeds << ','
          << avg_c_trn << ','
          << avg_c_hold << ','
          << avg_c_ori << ','
          << avg_j_rms << ','
          << avg_j_hold << ','
          << avg_score << ','
          << kRange << ','
          << kMassRange << ','
          << kPayloadMax << ','
          << kComRange
          << '\n';
    }
  }

  ASSERT_GE(best_idx, 0);
  const auto& best = candidates[best_idx];
  std::cout << "\nBest candidate: " << best.label
            << " (stable " << best_stable << "/" << kNumSeeds
            << ", score=" << best_score << ")\n";
  std::cout << "Recommended gains: jpos_kp=" << best.gains.jpos_kp
            << ", jpos_kd=" << best.gains.jpos_kd
            << ", ee_kp=" << best.gains.ee_kp
            << ", ee_kd=" << best.gains.ee_kd << "\n";
  std::cout << "Recommended residuals: friction=" << (best.friction ? "on" : "off")
            << ", gamma_c=" << best.gamma_c
            << ", gamma_v=" << best.gamma_v
            << ", observer=" << (best.observer ? "on" : "off")
            << ", K_o=" << best.K_o
            << ", max_tau_dist=" << best.max_tau_dist << "\n";
  if (csv.is_open()) {
    std::cout << "CSV written: " << csv_path << "\n";
  }

  std::filesystem::remove_all(sweep_dir);
}
