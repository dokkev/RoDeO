#include <algorithm>
#include <gtest/gtest.h>

#include <Eigen/Dense>

#include "residual_compensator/sysid.hpp"

namespace wbc {
namespace {

TEST(SysID, GravityGridProgressesAndFinishes) {
  SysID sys;
  sys.Setup(3);

  SysIDConfig cfg;
  cfg.enabled = true;
  cfg.mode = SysIDMode::GRAVITY_GRID;
  cfg.joint_idx = 1;
  cfg.start_delay = 0.0;
  cfg.ramp_time = 1.0;
  cfg.dwell_time = 0.5;
  cfg.offset = 0.0;
  cfg.gravity_offsets_rad = {-0.2, 0.2};
  sys.Configure(cfg);

  Eigen::VectorXd hold = Eigen::VectorXd::Zero(3);
  sys.Reset(hold);
  sys.Start(0.0);

  Eigen::VectorXd q = Eigen::VectorXd::Zero(3);
  Eigen::VectorXd qdot = Eigen::VectorXd::Zero(3);
  Eigen::VectorXd qddot = Eigen::VectorXd::Zero(3);

  sys.Update(0.0, 0.01, q, qdot, qddot);
  EXPECT_TRUE(sys.IsActive());
  EXPECT_EQ(sys.Phase(), SysIDPhase::RAMP);
  EXPECT_NEAR(q[1], 0.0, 1e-9);

  sys.Update(0.5, 0.01, q, qdot, qddot);
  EXPECT_LT(q[1], 0.0);
  EXPECT_GT(q[1], -0.2);

  sys.Update(1.2, 0.01, q, qdot, qddot);
  EXPECT_NEAR(q[1], -0.2, 1e-6);

  sys.Update(1.8, 0.01, q, qdot, qddot);
  EXPECT_GT(q[1], -0.2);

  sys.Update(3.2, 0.01, q, qdot, qddot);
  EXPECT_FALSE(sys.IsActive());
  EXPECT_TRUE(sys.IsFinished());
  EXPECT_EQ(sys.Phase(), SysIDPhase::DONE);
}

TEST(SysID, FrictionSweepStaysWithinBounds) {
  SysID sys;
  sys.Setup(2);

  SysIDConfig cfg;
  cfg.enabled = true;
  cfg.mode = SysIDMode::FRICTION_SWEEP;
  cfg.joint_idx = 0;
  cfg.start_delay = 0.0;
  cfg.duration = 2.0;
  cfg.amplitude = 0.1;
  cfg.offset = 0.05;
  cfg.cruise_vel = 0.2;
  sys.Configure(cfg);

  Eigen::VectorXd hold = Eigen::VectorXd::Zero(2);
  sys.Reset(hold);
  sys.Start(0.0);

  Eigen::VectorXd q = Eigen::VectorXd::Zero(2);
  Eigen::VectorXd qdot = Eigen::VectorXd::Zero(2);
  Eigen::VectorXd qddot = Eigen::VectorXd::Zero(2);

  double min_q = 1e9;
  double max_q = -1e9;
  for (int k = 0; k < 260; ++k) {
    const double t = 0.01 * static_cast<double>(k);
    sys.Update(t, 0.01, q, qdot, qddot);
    min_q = std::min(min_q, q[0]);
    max_q = std::max(max_q, q[0]);
    if (sys.IsActive()) {
      EXPECT_NEAR(std::abs(qdot[0]), 0.2, 1e-9);
    }
  }

  EXPECT_GE(min_q, -0.05 - 1e-9);  // offset - amplitude
  EXPECT_LE(max_q,  0.15 + 1e-9);  // offset + amplitude
  EXPECT_FALSE(sys.IsActive());
  EXPECT_TRUE(sys.IsFinished());
}

TEST(SysID, SafetyCheckCanAbort) {
  SysID sys;
  sys.Setup(1);

  SysIDConfig cfg;
  cfg.enabled = true;
  cfg.mode = SysIDMode::SINE;
  cfg.start_delay = 0.0;
  cfg.duration = 5.0;
  cfg.amplitude = 0.1;
  cfg.frequency_hz = 1.0;
  cfg.max_tracking_err = 0.01;
  cfg.max_meas_vel = 10.0;
  cfg.max_tau_ratio = 1.0;
  sys.Configure(cfg);

  Eigen::VectorXd hold = Eigen::VectorXd::Zero(1);
  sys.Reset(hold);
  sys.Start(0.0);

  Eigen::VectorXd q_ref = Eigen::VectorXd::Zero(1);
  Eigen::VectorXd qdot_ref = Eigen::VectorXd::Zero(1);
  Eigen::VectorXd qddot_ref = Eigen::VectorXd::Zero(1);
  sys.Update(0.1, 0.001, q_ref, qdot_ref, qddot_ref);

  Eigen::VectorXd q_meas = q_ref;
  Eigen::VectorXd qdot_meas = qdot_ref;
  Eigen::VectorXd tau_cmd = Eigen::VectorXd::Zero(1);
  Eigen::MatrixXd tau_limits(1, 2);
  tau_limits << -100.0, 100.0;

  q_meas[0] += 1.0;

  std::string reason;
  EXPECT_FALSE(sys.CheckSafety(q_meas, qdot_meas, tau_cmd, tau_limits, &reason));
  EXPECT_FALSE(reason.empty());

  reason.clear();
  EXPECT_FALSE(sys.CheckSafetyAndAbort(q_meas, qdot_meas, tau_cmd, tau_limits, &reason));
  EXPECT_TRUE(sys.IsAborted());
  EXPECT_EQ(sys.Phase(), SysIDPhase::ABORT);
  EXPECT_FALSE(reason.empty());
}

}  // namespace
}  // namespace wbc
