/**
 * @file wbc_core/wbc_architecture/test/test_cascade_pid.cpp
 * @brief Unit tests for the cascade PID controller (JointPID).
 *
 * Cascade structure:
 *
 *   Outer (position loop):
 *     pos_err  = q_des - q
 *     pos_int += pos_err * dt
 *     qdot_ref = Kp_pos*pos_err + Ki_pos*pos_int + Kd_pos*(qdot_des - qdot)
 *
 *   Inner (velocity loop):
 *     vel_err      = qdot_ref - qdot
 *     vel_int     += vel_err * dt
 *     vel_err_dot  = (vel_err - vel_err_prev) / dt
 *     tau          = Kp_vel*vel_err + Ki_vel*vel_int + Kd_vel*vel_err_dot
 */
#include <gtest/gtest.h>

#include "wbc_util/joint_pid.hpp"

namespace wbc {
namespace {

constexpr double kEps = 1.0e-12;

// Build a single-joint PID with given gains (all others zero).
JointPID MakePID(double kp_pos = 0, double ki_pos = 0, double kd_pos = 0,
                 double kp_vel = 0, double ki_vel = 0, double kd_vel = 0) {
  JointPID pid;
  pid.Setup(1);
  pid.SetPositionGains(Eigen::VectorXd::Constant(1, kp_pos),
                       Eigen::VectorXd::Constant(1, ki_pos),
                       Eigen::VectorXd::Constant(1, kd_pos));
  pid.SetVelocityGains(Eigen::VectorXd::Constant(1, kp_vel),
                       Eigen::VectorXd::Constant(1, ki_vel),
                       Eigen::VectorXd::Constant(1, kd_vel));
  return pid;
}

const Eigen::VectorXd k0 = Eigen::VectorXd::Zero(1);

// Helper: call Compute with scalar inputs.
double Compute(JointPID& pid,
               double q_des, double qdot_des, double q, double qdot,
               double dt = 0.001) {
  Eigen::VectorXd out(1);
  pid.Compute(Eigen::VectorXd::Constant(1, q_des),
              Eigen::VectorXd::Constant(1, qdot_des),
              Eigen::VectorXd::Constant(1, q),
              Eigen::VectorXd::Constant(1, qdot),
              dt, out);
  return out[0];
}

// ── Setup ──────────────────────────────────────────────────────────────────

TEST(CascadePID, SetupReportsReady) {
  JointPID pid;
  EXPECT_FALSE(pid.IsSetup());
  pid.Setup(3);
  EXPECT_TRUE(pid.IsSetup());
}

TEST(CascadePID, AllZeroGainsProduceZeroOutput) {
  auto pid = MakePID();  // all gains zero
  EXPECT_NEAR(Compute(pid, 99.0, 5.0, 0.0, 0.0), 0.0, kEps);
}

// ── Position loop: Kp_pos ──────────────────────────────────────────────────

TEST(CascadePID, PosLoop_Kp_DrivesVelocityReference) {
  // kp_pos=10, kp_vel=1, all others 0
  // pos_err=1 => qdot_ref=10 => vel_err=10-0=10 => tau=kp_vel*10=10
  auto pid = MakePID(/*kp_pos=*/10.0, 0, 0, /*kp_vel=*/1.0);
  EXPECT_NEAR(Compute(pid, 1.0, 0.0, 0.0, 0.0), 10.0, kEps);
}

TEST(CascadePID, PosLoop_Kp_MeasuredVelocityReducesVelErr) {
  // kp_pos=10, kp_vel=1; q_err=1 => qdot_ref=10; qdot=4 => vel_err=6 => tau=6
  auto pid = MakePID(10.0, 0, 0, 1.0);
  EXPECT_NEAR(Compute(pid, 1.0, 0.0, 0.0, 4.0), 6.0, kEps);
}

// ── Position loop: Ki_pos ──────────────────────────────────────────────────

TEST(CascadePID, PosLoop_Ki_AccumulatesPositionError) {
  // kp_pos=0, ki_pos=1, kp_vel=1, dt=0.1, pos_err=1
  // Tick 1: pos_int=0.1  => qdot_ref=0.1 => vel_err=0.1 => tau=0.1
  // Tick 2: pos_int=0.2  => qdot_ref=0.2 => vel_err=0.2 => tau=0.2
  auto pid = MakePID(0, /*ki_pos=*/1.0, 0, /*kp_vel=*/1.0);
  EXPECT_NEAR(Compute(pid, 1.0, 0.0, 0.0, 0.0, 0.1), 0.1, kEps);
  EXPECT_NEAR(Compute(pid, 1.0, 0.0, 0.0, 0.0, 0.1), 0.2, kEps);
}

TEST(CascadePID, PosLoop_Ki_IntegralIsClamped) {
  // integral_limit_pos=0.5, ki_pos=1, constant pos_err=1
  // After many ticks: pos_int clamped to 0.5 => qdot_ref=0.5 => tau=kp_vel*0.5
  JointPID pid;
  pid.Setup(1);
  pid.SetPositionGains(k0, Eigen::VectorXd::Constant(1, 1.0), k0);
  pid.SetVelocityGains(Eigen::VectorXd::Constant(1, 1.0), k0, k0);
  pid.SetPositionIntegralLimit(Eigen::VectorXd::Constant(1, 0.5));

  for (int i = 0; i < 100; ++i) Compute(pid, 1.0, 0.0, 0.0, 0.0, 0.01);
  EXPECT_NEAR(Compute(pid, 1.0, 0.0, 0.0, 0.0, 0.01), 0.5, kEps);
}

// ── Position loop: Kd_pos ──────────────────────────────────────────────────

TEST(CascadePID, PosLoop_Kd_UsesVelocityError) {
  // kd_pos=1 (D of position loop = qdot_des - qdot), kp_vel=1
  // q_des=q=0 (no pos error), qdot_des=5, qdot=0
  // qdot_ref = kd_pos*(qdot_des-qdot) = 1*(5-0)=5 => vel_err=5-0=5 => tau=5
  auto pid = MakePID(0, 0, /*kd_pos=*/1.0, /*kp_vel=*/1.0);
  EXPECT_NEAR(Compute(pid, 0.0, 5.0, 0.0, 0.0), 5.0, kEps);
}

// ── Velocity loop: Kp_vel ─────────────────────────────────────────────────

TEST(CascadePID, VelLoop_Kp_ScalesVelErr) {
  // kp_pos=1, kp_vel=3: pos_err=1 => qdot_ref=1 => vel_err=1-0=1 => tau=3
  auto pid = MakePID(/*kp_pos=*/1.0, 0, 0, /*kp_vel=*/3.0);
  EXPECT_NEAR(Compute(pid, 1.0, 0.0, 0.0, 0.0), 3.0, kEps);
}

// ── Velocity loop: Ki_vel ─────────────────────────────────────────────────

TEST(CascadePID, VelLoop_Ki_AccumulatesVelocityError) {
  // kp_pos=10, ki_vel=1, kp_vel=0, dt=0.1
  // Tick 1: qdot_ref=10, vel_err=10, vel_int=1.0, tau=ki_vel*1.0=1.0
  // Tick 2: same => vel_int=2.0, tau=2.0
  auto pid = MakePID(/*kp_pos=*/10.0, 0, 0, 0, /*ki_vel=*/1.0);
  EXPECT_NEAR(Compute(pid, 1.0, 0.0, 0.0, 0.0, 0.1), 1.0, kEps);
  EXPECT_NEAR(Compute(pid, 1.0, 0.0, 0.0, 0.0, 0.1), 2.0, kEps);
}

TEST(CascadePID, VelLoop_Ki_IntegralIsClamped) {
  JointPID pid;
  pid.Setup(1);
  pid.SetPositionGains(Eigen::VectorXd::Constant(1, 10.0), k0, k0);
  pid.SetVelocityGains(k0, Eigen::VectorXd::Constant(1, 1.0), k0);
  pid.SetVelocityIntegralLimit(Eigen::VectorXd::Constant(1, 0.5));

  for (int i = 0; i < 100; ++i) Compute(pid, 1.0, 0.0, 0.0, 0.0, 0.01);
  EXPECT_NEAR(Compute(pid, 1.0, 0.0, 0.0, 0.0, 0.01), 0.5, kEps);
}

// ── Velocity loop: Kd_vel ─────────────────────────────────────────────────

TEST(CascadePID, VelLoop_Kd_UsesVelErrDerivative) {
  // kp_pos=10, kd_vel=1, kp_vel=0, ki_vel=0
  // Tick 1: vel_err=10, vel_err_prev=0, vel_err_dot=(10-0)/dt => tau=kd_vel*vel_err_dot
  auto pid = MakePID(/*kp_pos=*/10.0, 0, 0, 0, 0, /*kd_vel=*/1.0);
  const double dt = 0.1;
  // vel_err_prev=0 initially; vel_err=10 => vel_err_dot=10/0.1=100 => tau=100
  EXPECT_NEAR(Compute(pid, 1.0, 0.0, 0.0, 0.0, dt), 100.0, kEps);
  // Tick 2: same steady state => vel_err_dot=(10-10)/dt=0 => tau=0
  EXPECT_NEAR(Compute(pid, 1.0, 0.0, 0.0, 0.0, dt), 0.0, kEps);
}

// ── Reset ─────────────────────────────────────────────────────────────────

TEST(CascadePID, ResetClearsBothIntegralsAndVelHistory) {
  auto pid = MakePID(/*kp_pos=*/10.0, /*ki_pos=*/1.0, 0, 0, /*ki_vel=*/1.0, /*kd_vel=*/1.0);
  for (int i = 0; i < 10; ++i) Compute(pid, 1.0, 0.0, 0.0, 0.0, 0.01);

  pid.Reset();

  // After reset: integrals = 0, vel_err_prev = 0
  // Compute one tick: pos_int grows from 0, vel_int grows from 0, vel_err_dot=(vel_err-0)/dt
  // kp_pos=10: qdot_ref=10, vel_err=10
  // ki_vel=1, dt=0.1 => vel_int=1.0; kd_vel=1, vel_err_dot=(10-0)/0.1=100
  // tau = 0*10 + 1*1.0 + 1*100 = 101
  // Also ki_pos=1, pos_int=0.1 => qdot_ref = 10+0.1=10.1, vel_err=10.1
  // But since kp_pos=10 and ki_pos=1: qdot_ref=10*1+1*0.1=10.1
  // vel_err=10.1, vel_int=10.1*0.1=1.01, vel_err_dot=(10.1-0)/0.1=101
  // tau = 1.01 + 101 = 102.01
  const double dt = 0.1;
  const double pos_err   = 1.0;
  const double pos_int   = pos_err * dt;             // ki_pos=1, dt=0.1 => 0.1
  const double qdot_ref  = 10.0 * pos_err + 1.0 * pos_int;  // 10.1
  const double vel_err   = qdot_ref;                 // qdot=0
  const double vel_int   = vel_err * dt;             // 1.01
  const double vel_err_dot = vel_err / dt;           // 101
  const double expected  = vel_int + vel_err_dot;    // 1.01 + 101 = 102.01
  EXPECT_NEAR(Compute(pid, 1.0, 0.0, 0.0, 0.0, dt), expected, kEps);
}

// ── Multi-joint independence ───────────────────────────────────────────────

TEST(CascadePID, MultiJointGainsAreIndependent) {
  // 2-joint: joint 0 kp_pos=5, joint 1 kp_pos=20; kp_vel=1 each
  JointPID pid;
  pid.Setup(2);
  pid.SetPositionGains(Eigen::Vector2d(5.0, 20.0), k0.replicate(2, 1), k0.replicate(2, 1));
  pid.SetVelocityGains(Eigen::Vector2d::Ones(), k0.replicate(2, 1), k0.replicate(2, 1));

  Eigen::VectorXd out(2);
  pid.Compute(Eigen::Vector2d::Ones(),  Eigen::Vector2d::Zero(),
              Eigen::Vector2d::Zero(),  Eigen::Vector2d::Zero(),
              0.001, out);
  // Joint 0: qdot_ref=5*1=5, vel_err=5, tau=1*5=5
  // Joint 1: qdot_ref=20*1=20, vel_err=20, tau=1*20=20
  EXPECT_NEAR(out[0], 5.0,  kEps);
  EXPECT_NEAR(out[1], 20.0, kEps);
}

}  // namespace
}  // namespace wbc
