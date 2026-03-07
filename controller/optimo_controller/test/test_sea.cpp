/**
 * @file test_sea.cpp
 * @brief Verify spring actuator behavior (motor+gear → spring → link).
 *
 * Tests:
 * 1. Unit: spring produces tau = k*(q_des - q_link) + tau_ff
 * 2. Integration: spring actuator + MuJoCo — perturbed joint restores to q_des
 */
#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>

#include <Eigen/Dense>
#include <mujoco/mujoco.h>
#include <yaml-cpp/yaml.h>

#include "wbc_util/actuator_interface.hpp"
#include "wbc_util/ros_path_utils.hpp"

namespace {

constexpr int kNJoints = 7;
constexpr double kDt = 0.001;

std::string ResolvePackagePath(const std::string& pkg_name,
                               const std::string& rel_path) {
  const char* prefix = std::getenv("AMENT_PREFIX_PATH");
  if (!prefix) {
    throw std::runtime_error("AMENT_PREFIX_PATH not set.");
  }
  std::istringstream ss(prefix);
  std::string token;
  while (std::getline(ss, token, ':')) {
    auto full = std::filesystem::path(token) / "share" / pkg_name / rel_path;
    if (std::filesystem::exists(full)) return full.string();
  }
  throw std::runtime_error("Cannot resolve package://" + pkg_name + "/" + rel_path);
}

struct SpringParams {
  Eigen::VectorXd stiffness;
  Eigen::VectorXd damping;
};

SpringParams LoadSpringParams() {
  auto yaml_path = ResolvePackagePath("optimo_description", "config/joint_dynamics.yaml");
  auto yml = YAML::LoadFile(yaml_path);
  const auto hw = yml["robot_hardware"];

  auto load_vec = [&](const std::string& key) -> Eigen::VectorXd {
    auto node = hw[key];
    Eigen::VectorXd v(kNJoints);
    for (int i = 0; i < kNJoints; ++i)
      v[i] = node[i].as<double>();
    return v;
  };

  return {load_vec("stiffness"), load_vec("damping_cmp")};
}

wbc::ActuatorCommand MakeCmd(
    const Eigen::VectorXd& q_des, const Eigen::VectorXd& q_link,
    const Eigen::VectorXd& tau_ff = Eigen::VectorXd(),
    const Eigen::VectorXd& qdot_des = Eigen::VectorXd(),
    const Eigen::VectorXd& qdot_link = Eigen::VectorXd()) {
  wbc::ActuatorCommand cmd;
  cmd.q_des = q_des;
  cmd.q_link = q_link;
  cmd.tau_ff = tau_ff.size() > 0 ? tau_ff : Eigen::VectorXd::Zero(q_des.size());
  cmd.qdot_des = qdot_des.size() > 0 ? qdot_des : Eigen::VectorXd::Zero(q_des.size());
  cmd.qdot_link = qdot_link.size() > 0 ? qdot_link : Eigen::VectorXd::Zero(q_des.size());
  cmd.dt = kDt;
  return cmd;
}

}  // namespace

// ------------------------------------------------------------------
// Unit test: spring torque = k * (q_des - q_link) when tau_ff = 0
// ------------------------------------------------------------------
TEST(SpringActuator, SpringTorqueDirection) {
  auto params = LoadSpringParams();
  wbc::SpringActuator actuator(params.stiffness, params.damping);

  Eigen::VectorXd q_des = Eigen::VectorXd::Zero(kNJoints);
  Eigen::VectorXd q_link = Eigen::VectorXd::Constant(kNJoints, 0.1);

  auto cmd = MakeCmd(q_des, q_link);
  Eigen::VectorXd tau = actuator.ProcessTorque(cmd);

  std::cout << "\n===== Spring Torque Test =====\n";
  std::cout << "Stiffness:    " << params.stiffness.transpose() << "\n";
  std::cout << "q_des=0, q_link=+0.1 rad\n";
  std::cout << "tau_spring:   " << tau.transpose() << "\n";

  for (int i = 0; i < kNJoints; ++i) {
    double expected = -params.stiffness[i] * 0.1;
    EXPECT_NEAR(tau[i], expected, 1e-10)
      << "Joint " << i << ": tau should equal k * (q_des - q_link)";
  }
}

// ------------------------------------------------------------------
// Unit test: feedforward torque passes through
// ------------------------------------------------------------------
TEST(SpringActuator, FeedforwardPassthrough) {
  auto params = LoadSpringParams();
  wbc::SpringActuator actuator(params.stiffness, params.damping);

  // q_des == q_link → spring force = 0, only tau_ff remains.
  Eigen::VectorXd q = Eigen::VectorXd::Constant(kNJoints, 1.0);
  Eigen::VectorXd tau_ff = Eigen::VectorXd::Constant(kNJoints, 5.0);

  auto cmd = MakeCmd(q, q, tau_ff);
  Eigen::VectorXd tau = actuator.ProcessTorque(cmd);

  std::cout << "\n===== Feedforward Passthrough Test =====\n";
  std::cout << "tau_ff=5.0, q_des==q_link → tau should be 5.0\n";
  std::cout << "tau: " << tau.transpose() << "\n";

  for (int i = 0; i < kNJoints; ++i) {
    EXPECT_NEAR(tau[i], 5.0, 1e-10);
  }
}

// ------------------------------------------------------------------
// Unit test: DirectActuator ignores q_des, returns tau_ff
// ------------------------------------------------------------------
TEST(SpringActuator, DirectPassthrough) {
  wbc::DirectActuator actuator;

  Eigen::VectorXd q_des = Eigen::VectorXd::Zero(kNJoints);
  Eigen::VectorXd q_link = Eigen::VectorXd::Constant(kNJoints, 0.5);
  Eigen::VectorXd tau_ff = Eigen::VectorXd::Constant(kNJoints, 3.0);

  auto cmd = MakeCmd(q_des, q_link, tau_ff);
  Eigen::VectorXd tau = actuator.ProcessTorque(cmd);

  for (int i = 0; i < kNJoints; ++i) {
    EXPECT_NEAR(tau[i], 3.0, 1e-10);
  }
}

// ------------------------------------------------------------------
// Integration: spring actuator + MuJoCo — joint restores to q_des
// ------------------------------------------------------------------
TEST(SpringActuator, MujocoSpringRestore) {
  std::string mjcf_path = ResolvePackagePath("optimo_description", "mjcf/optimo.xml");
  char error[1000] = "";
  mjModel* m = mj_loadXML(mjcf_path.c_str(), nullptr, error, sizeof(error));
  ASSERT_NE(m, nullptr) << error;
  mjData* d = mj_makeData(m);

  // Disable gravity to isolate spring dynamics.
  m->opt.gravity[0] = m->opt.gravity[1] = m->opt.gravity[2] = 0.0;

  if (m->nkey > 0) mju_copy(d->qpos, m->key_qpos, m->nq);
  mju_zero(d->qvel, m->nv);
  mj_forward(m, d);

  Eigen::VectorXd q_home(kNJoints);
  for (int i = 0; i < kNJoints; ++i) q_home[i] = d->qpos[i];

  auto params = LoadSpringParams();
  wbc::SpringActuator actuator(params.stiffness, params.damping);

  // Perturb joint 0 by +0.2 rad.
  constexpr int kTestJoint = 0;
  constexpr double kPerturbation = 0.2;
  d->qpos[kTestJoint] += kPerturbation;
  mj_forward(m, d);

  constexpr double kSimTime = 5.0;
  const int n_steps = static_cast<int>(kSimTime / kDt);

  // q_des = q_home (spring pulls back to home).
  Eigen::VectorXd q_des = q_home;

  std::cout << "\n===== Spring + MuJoCo Restore Test =====\n";
  std::cout << "Joint " << kTestJoint << " perturbed by +" << kPerturbation << " rad\n";
  std::cout << "Stiffness[0] = " << params.stiffness[0] << " Nm/rad\n\n";
  std::cout << "  time  |  q[0]-q_home  |  tau[0]\n";
  std::cout << "--------+---------------+--------\n";

  for (int step = 0; step < n_steps; ++step) {
    Eigen::VectorXd q_link(kNJoints), qdot_link(kNJoints);
    for (int i = 0; i < kNJoints; ++i) {
      q_link[i] = d->qpos[i];
      qdot_link[i] = d->qvel[i];
    }

    auto cmd = MakeCmd(q_des, q_link, Eigen::VectorXd::Zero(kNJoints),
                        Eigen::VectorXd::Zero(kNJoints), qdot_link);
    Eigen::VectorXd tau = actuator.ProcessTorque(cmd);

    for (int i = 0; i < kNJoints; ++i) d->ctrl[i] = tau[i];
    mj_step(m, d);

    double t = step * kDt;
    if (step == 0 || step == 50 || step == 100 || step == 500 ||
        step == 1000 || step == 2000 || step == n_steps - 1) {
      double disp = d->qpos[kTestJoint] - q_home[kTestJoint];
      std::cout << std::fixed << std::setprecision(4)
                << std::setw(7) << t << " | "
                << std::setw(13) << disp << " | "
                << std::setw(13) << tau[kTestJoint] << "\n";
    }
  }

  double final_disp = std::abs(d->qpos[kTestJoint] - q_home[kTestJoint]);
  std::cout << "\nFinal displacement: " << final_disp << " rad\n";

  // Spring should restore joint to near q_des (home).
  EXPECT_LT(final_disp, 0.01)
    << "After 5s, spring should restore joint to within 0.01 rad of home";

  // System stable.
  for (int i = 0; i < kNJoints; ++i) {
    EXPECT_TRUE(std::isfinite(d->qpos[i])) << "Joint " << i << " is NaN";
  }

  mj_deleteData(d);
  mj_deleteModel(m);
}
