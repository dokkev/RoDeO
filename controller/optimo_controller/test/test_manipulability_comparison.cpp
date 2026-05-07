/**
 * @file test_manipulability_comparison.cpp
 * @brief Compare manipulability handler ON vs OFF via Pinocchio IK sim.
 *
 * Pure kinematic simulation — no MuJoCo, no WBC pipeline.
 * Supports both Optimo (7-DOF) and Plato (15-DOF: arm + hand).
 *
 * For each robot / trajectory combo, runs twice (handler ON vs OFF).
 * Logs CSV with: σ_min, cond#, qdot norm (raw & clipped), tracking error.
 */
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/SVD>

#include "wbc_robot_system/pinocchio_robot_system.hpp"
#include "wbc_handlers/manipulability_handler.hpp"

namespace {

constexpr double kDt = 0.002;  // 500 Hz
constexpr double kSigmaThreshold = 0.08;
constexpr double kDlsLambda = 0.002;
constexpr double kMaxJointVel = 0.5;

// ── Robot configuration ─────────────────────────────────────────────────────

struct RobotConfig {
  std::string pkg_name;        // ROS package name
  std::string urdf_rel_path;   // relative to share/pkg/
  std::string ee_frame;        // end-effector frame in URDF
  int num_joints;
  Eigen::VectorXd start_q;     // initial non-singular config
  std::string label;
};

RobotConfig MakeOptimoConfig() {
  RobotConfig c;
  c.pkg_name = "optimo_description";
  c.urdf_rel_path = "urdf/optimo.urdf";
  c.ee_frame = "optimo_end_effector";
  c.num_joints = 7;
  c.start_q = (Eigen::VectorXd(7) << 0.3, 2.0, 0.5, -1.5, 0.5, -0.8, 0.5).finished();
  c.label = "Optimo (7-DOF)";
  return c;
}

RobotConfig MakePlatoFingertipConfig() {
  RobotConfig c;
  c.pkg_name = "plato_description";
  c.urdf_rel_path = "urdf/plato.urdf";
  c.ee_frame = "aristo_index_fingertip";  // 9 DOF chain (7 arm + 2 finger)
  c.num_joints = 15;  // all active joints
  // Folded arm + moderately flexed fingers. σ_min should be well above 0.08.
  c.start_q = Eigen::VectorXd(15);
  c.start_q << 0.3, 2.0, 0.5, -1.5, 0.5, -0.8, 0.5,  // arm bent (same as optimo)
               0.5, 0.5, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0; // finger moderately flexed
  c.label = "Plato fingertip (15-DOF, 9 to tip)";
  return c;
}

// ── Utility ─────────────────────────────────────────────────────────────────

std::string ResolvePackagePath(const std::string& pkg_name,
                               const std::string& rel_path) {
  const char* prefix = std::getenv("AMENT_PREFIX_PATH");
  if (!prefix) throw std::runtime_error("AMENT_PREFIX_PATH not set.");
  std::istringstream ss(prefix);
  std::string token;
  while (std::getline(ss, token, ':')) {
    auto full = std::filesystem::path(token) / "share" / pkg_name / rel_path;
    if (std::filesystem::exists(full)) return full.string();
  }
  throw std::runtime_error("Cannot resolve package://" + pkg_name + "/" + rel_path);
}

// ── CSV ─────────────────────────────────────────────────────────────────────

void WriteCsvHeader(std::ofstream& f, int n_joints) {
  f << "time,ee_x,ee_y,ee_z,ee_x_des,ee_y_des,ee_z_des,"
    << "sigma_min,logw,is_active,bias_qdot_norm,"
    << "qdot_norm,qdot_raw_norm,cond_number,vel_clipped,";
  for (int i = 0; i < n_joints; ++i) f << "q" << i << ",";
  f << "phase\n";
}

struct CsvRow {
  double t;
  Eigen::Vector3d ee, ee_des;
  double sigma_min, logw;
  bool active;
  double bias_norm, qdot_norm, qdot_raw_norm, cond_number;
  bool vel_clipped;
  Eigen::VectorXd q;
  std::string phase;
};

void WriteCsvRow(std::ofstream& f, const CsvRow& r) {
  f << std::fixed << std::setprecision(6)
    << r.t << ","
    << r.ee.x() << "," << r.ee.y() << "," << r.ee.z() << ","
    << r.ee_des.x() << "," << r.ee_des.y() << "," << r.ee_des.z() << ","
    << r.sigma_min << "," << r.logw << ","
    << (r.active ? 1 : 0) << "," << r.bias_norm << ","
    << r.qdot_norm << "," << r.qdot_raw_norm << ","
    << r.cond_number << "," << (r.vel_clipped ? 1 : 0) << ",";
  for (int i = 0; i < r.q.size(); ++i) f << r.q[i] << ",";
  f << r.phase << "\n";
}

bool ClampJointVel(Eigen::VectorXd& v, double limit) {
  bool clipped = false;
  for (int i = 0; i < v.size(); ++i) {
    if (v[i] > limit)  { v[i] = limit;  clipped = true; }
    if (v[i] < -limit) { v[i] = -limit; clipped = true; }
  }
  return clipped;
}

Eigen::MatrixXd DlsPinv(const Eigen::MatrixXd& J, double lambda) {
  Eigen::MatrixXd JJt = J * J.transpose();
  JJt.diagonal().array() += lambda * lambda;
  return J.transpose() * JJt.inverse();
}

double ConditionNumber(const Eigen::MatrixXd& J_lin) {
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(J_lin);
  const auto& sv = svd.singularValues();
  if (sv(sv.size() - 1) < 1e-10) return 1e6;
  return sv(0) / sv(sv.size() - 1);
}

// ── Trajectory waypoint ─────────────────────────────────────────────────────

struct Waypoint {
  Eigen::Vector3d offset;
  double speed;
  double hold_time;
};

// ── Main sim loop ───────────────────────────────────────────────────────────

void RunKinematicSim(const RobotConfig& rcfg, double manip_gain,
                     double dls_lambda,
                     const std::string& csv_path,
                     const std::vector<Waypoint>& waypoints) {
  std::string urdf = ResolvePackagePath(rcfg.pkg_name, rcfg.urdf_rel_path);
  std::string pkg_dir = std::filesystem::path(urdf).parent_path().string();
  auto robot = std::make_unique<wbc::PinocchioRobotSystem>(
      urdf, pkg_dir, /*fixed_base=*/true);

  const int n = rcfg.num_joints;
  int ee_idx = robot->GetFrameIndex(rcfg.ee_frame);

  Eigen::VectorXd q = rcfg.start_q;
  Eigen::VectorXd qdot = Eigen::VectorXd::Zero(n);
  Eigen::Vector3d z3 = Eigen::Vector3d::Zero();
  Eigen::Quaterniond iq = Eigen::Quaterniond::Identity();
  robot->UpdateRobotModel(z3, iq, z3, z3, q, qdot, false);

  wbc::ManipulabilityHandler handler;
  wbc::ManipulabilityHandler::Config cfg;
  cfg.gain = manip_gain;
  cfg.sigma_threshold = kSigmaThreshold;
  handler.Init(robot.get(), ee_idx, cfg);
  handler.Update(kDt);

  Eigen::Vector3d start_ee = robot->GetLinkIsometry(ee_idx).translation();
  const Eigen::MatrixXd& jlim = robot->JointPosLimits();

  std::cout << "  Start EE: [" << std::fixed << std::setprecision(3)
            << start_ee.x() << ", " << start_ee.y() << ", " << start_ee.z()
            << "], σ_min=" << handler.sigma_min()
            << ", n_joints=" << n << "\n";

  std::ofstream csv(csv_path);
  WriteCsvHeader(csv, n);

  double t = 0.0;
  Eigen::MatrixXd J(6, n);
  int total_clipped = 0;

  auto ik_step = [&](const Eigen::Vector3d& xdot_des,
                     const Eigen::Vector3d& ee_des,
                     const std::string& phase) {
    robot->UpdateRobotModel(z3, iq, z3, z3, q, qdot, false);
    robot->FillLinkJacobian(ee_idx, J);
    handler.Update(kDt);

    Eigen::Vector3d ee = robot->GetLinkIsometry(ee_idx).translation();
    Eigen::MatrixXd J_lin = J.bottomRows(3);

    Eigen::MatrixXd J_pinv = DlsPinv(J_lin, dls_lambda);
    Eigen::VectorXd qdot_ik = J_pinv * xdot_des;

    if (manip_gain > 0.0 && handler.is_active()) {
      Eigen::MatrixXd N = Eigen::MatrixXd::Identity(n, n) - J_pinv * J_lin;
      qdot_ik += N * handler.bias_qdot();
    }

    double raw_norm = qdot_ik.norm();
    bool clipped = ClampJointVel(qdot_ik, kMaxJointVel);
    if (clipped) ++total_clipped;

    qdot = qdot_ik;
    q += qdot * kDt;
    for (int j = 0; j < n; ++j)
      q[j] = std::clamp(q[j], jlim(j, 0), jlim(j, 1));

    double cond = ConditionNumber(J_lin);
    CsvRow row{t, ee, ee_des, handler.sigma_min(), handler.logw(),
               handler.is_active(), handler.bias_qdot().norm(),
               qdot.norm(), raw_norm, cond, clipped, q, phase};
    WriteCsvRow(csv, row);
    t += kDt;
  };

  // Baseline hold
  for (int i = 0; i < static_cast<int>(1.0 / kDt); ++i) {
    Eigen::Vector3d ee = robot->GetLinkIsometry(ee_idx).translation();
    ik_step(5.0 * (start_ee - ee), start_ee, "baseline");
  }

  // Execute waypoints
  Eigen::Vector3d current_goal = start_ee;
  for (size_t wi = 0; wi < waypoints.size(); ++wi) {
    const auto& wp = waypoints[wi];
    Eigen::Vector3d target = start_ee + wp.offset;
    Eigen::Vector3d direction = target - current_goal;
    double distance = direction.norm();
    if (distance < 1e-6) { current_goal = target; continue; }
    Eigen::Vector3d unit_dir = direction.normalized();
    double travel_time = distance / wp.speed;
    int travel_steps = std::max(1, static_cast<int>(travel_time / kDt));

    std::string transit_ph = "wp" + std::to_string(wi) + "_transit";
    std::string hold_ph = "wp" + std::to_string(wi) + "_hold";

    for (int step = 0; step < travel_steps; ++step) {
      double frac = static_cast<double>(step + 1) / travel_steps;
      Eigen::Vector3d des = current_goal + direction * frac;
      Eigen::Vector3d ee = robot->GetLinkIsometry(ee_idx).translation();
      ik_step(unit_dir * wp.speed + 5.0 * (des - ee), des, transit_ph);
    }
    current_goal = target;

    int hold_steps = static_cast<int>(wp.hold_time / kDt);
    for (int step = 0; step < hold_steps; ++step) {
      Eigen::Vector3d ee = robot->GetLinkIsometry(ee_idx).translation();
      ik_step(5.0 * (current_goal - ee), current_goal, hold_ph);
    }
  }

  csv.close();
  std::cout << "  Written: " << csv_path
            << " (" << std::filesystem::file_size(csv_path) / 1024 << " KB)"
            << ", vel-clipped: " << total_clipped << "\n";
}

// ── Helper ──────────────────────────────────────────────────────────────────

// Run 4 variants: {DLS, no-DLS} × {handler ON, OFF}
void RunComparison(const RobotConfig& rcfg, const std::string& name,
                   const std::string& title,
                   const std::vector<Waypoint>& waypoints) {
  constexpr double kLambdaDls = 0.05;   // standard DLS
  constexpr double kLambdaNone = 0.0;   // pure pseudoinverse — no singularity protection

  struct Variant {
    std::string suffix;
    double lambda;
    double gain;
    std::string desc;
  };
  std::vector<Variant> variants = {
    {"dls_handler_on",   kLambdaDls,  0.15, "DLS + Handler ON"},
    {"dls_handler_off",  kLambdaDls,  0.0,  "DLS + Handler OFF"},
    {"nodls_handler_on", kLambdaNone, 0.15, "No DLS + Handler ON"},
    {"nodls_handler_off",kLambdaNone, 0.0,  "No DLS + Handler OFF"},
  };

  std::string dir = "/tmp/manip_comparison/" + name;
  std::filesystem::create_directories(dir);
  std::cout << "\n===== " << title << " (" << rcfg.label << ") =====\n";
  for (const auto& v : variants) {
    std::cout << v.desc << ":\n";
    RunKinematicSim(rcfg, v.gain, v.lambda,
                    dir + "/" + v.suffix + ".csv", waypoints);
  }
  EXPECT_TRUE(std::filesystem::exists(dir + "/dls_handler_on.csv"));
}

// ── Optimo tests (7-DOF) ───────────────────────────────────────────────────

TEST(OptimoComparison, DeepExtension) {
  auto cfg = MakeOptimoConfig();
  RunComparison(cfg, "optimo_deep", "Deep Extension", {
      {{0.10, 0.00, 0.00}, 0.05, 0.5},
      {{0.20, 0.00, 0.00}, 0.04, 0.5},
      {{0.30, 0.00, 0.00}, 0.03, 1.0},
      {{0.35, 0.00, 0.00}, 0.02, 2.0},
      {{0.30, 0.00, 0.05}, 0.04, 0.5},
      {{0.30, 0.00,-0.05}, 0.04, 0.5},
      {{0.20, 0.00, 0.00}, 0.05, 0.5},
      {{0.00, 0.00, 0.00}, 0.05, 1.0},
  });
}

TEST(OptimoComparison, ProgressiveApproach) {
  auto cfg = MakeOptimoConfig();
  std::vector<Waypoint> wps;
  for (int i = 1; i <= 9; ++i)
    wps.push_back({{0.05 * i, 0.0, 0.0}, 0.02, 1.0});
  for (int i = 8; i >= 0; --i)
    wps.push_back({{0.05 * i, 0.0, 0.0}, 0.03, 0.3});
  wps.push_back({{0.0, 0.0, 0.0}, 0.03, 1.0});
  RunComparison(cfg, "optimo_progressive", "Progressive Approach", wps);
}

// ── Plato fingertip tests (15-DOF) ─────────────────────────────────────────

// Push fingertip deep into workspace boundary — σ_min should drop dramatically.
TEST(PlatoComparison, FingertipDeepExtension) {
  auto cfg = MakePlatoFingertipConfig();
  RunComparison(cfg, "plato_deep", "Fingertip Deep Extension", {
      {{0.10, 0.00, 0.00}, 0.04, 0.5},
      {{0.20, 0.00, 0.00}, 0.03, 0.5},
      {{0.30, 0.00, 0.00}, 0.02, 1.0},
      {{0.40, 0.00, 0.00}, 0.01, 2.0},   // very deep
      {{0.35, 0.00, 0.08}, 0.04, 0.5},
      {{0.35, 0.00,-0.08}, 0.04, 0.5},
      {{0.20, 0.00, 0.00}, 0.05, 0.5},
      {{0.00, 0.00, 0.00}, 0.05, 1.0},
  });
}

// Fast lateral sweeps at deep extension
TEST(PlatoComparison, FingertipLateralSweep) {
  auto cfg = MakePlatoFingertipConfig();
  std::vector<Waypoint> wps;
  wps.push_back({{0.35, 0.00, 0.00}, 0.03, 0.5});
  for (int i = 0; i < 5; ++i) {
    wps.push_back({{0.35,  0.10, 0.00}, 0.06, 0.3});
    wps.push_back({{0.35, -0.10, 0.00}, 0.06, 0.3});
  }
  wps.push_back({{0.35, 0.00, 0.00}, 0.04, 0.5});
  wps.push_back({{0.00, 0.00, 0.00}, 0.05, 1.0});
  RunComparison(cfg, "plato_sweep", "Fingertip Fast Lateral Sweep", wps);
}

// Progressive march toward workspace limit
TEST(PlatoComparison, FingertipProgressive) {
  auto cfg = MakePlatoFingertipConfig();
  std::vector<Waypoint> wps;
  for (int i = 1; i <= 10; ++i)
    wps.push_back({{0.05 * i, 0.0, 0.0}, 0.02, 1.0});  // up to 0.50
  for (int i = 9; i >= 0; --i)
    wps.push_back({{0.05 * i, 0.0, 0.0}, 0.03, 0.3});
  wps.push_back({{0.0, 0.0, 0.0}, 0.03, 1.0});
  RunComparison(cfg, "plato_progressive", "Fingertip Progressive to Limit", wps);
}

// Large circle at deep extension
TEST(PlatoComparison, FingertipCircle) {
  auto cfg = MakePlatoFingertipConfig();
  const int N = 30;
  const double R = 0.12;
  std::vector<Waypoint> wps;
  wps.push_back({{0.30, 0.00, 0.00}, 0.03, 0.5});
  for (int i = 0; i <= N; ++i) {
    double theta = 2.0 * M_PI * i / N;
    double dx = 0.30 + R * std::cos(theta);
    double dy = R * std::sin(theta);
    wps.push_back({{dx, dy, 0.00}, 0.04, 0.1});
  }
  wps.push_back({{0.00, 0.00, 0.00}, 0.05, 1.0});
  RunComparison(cfg, "plato_circle", "Fingertip Circle Near Boundary", wps);
}

}  // namespace
