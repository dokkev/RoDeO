/**
 * @file wbc_handlers/test/test_joint_teleop_handler.cpp
 * @brief Unit tests for JointTeleopHandler.
 *
 * Covers:
 *   - Init: Desired() and Goal() start at q_curr
 *   - SetVelocity: integration, velocity clamping, position clamping
 *   - SetPosition: updates Goal() only, Desired() unchanged until Update()
 *   - Update: rate-limited tracking of Goal(), task.UpdateDesired() called
 */
#include <gtest/gtest.h>

#include "wbc_handlers/joint_teleop_handler.hpp"

namespace wbc {
namespace {

constexpr double kEps = 1.0e-10;
constexpr int    kN   = 3;

// Minimal mock task — records the last UpdateDesired() call.
struct MockTask {
  Eigen::VectorXd pos{Eigen::VectorXd::Zero(kN)};
  Eigen::VectorXd vel{Eigen::VectorXd::Zero(kN)};
  Eigen::VectorXd acc{Eigen::VectorXd::Zero(kN)};

  void UpdateDesired(const Eigen::VectorXd& p,
                     const Eigen::VectorXd& v,
                     const Eigen::VectorXd& a) {
    pos = p;  vel = v;  acc = a;
  }
};

// Returns a handler initialized at q_curr, [-q_lim, +q_lim], qdot_max (uniform).
JointTeleopHandler MakeHandler(double q_curr  = 0.0,
                                double q_lim   = 1.0,
                                double qdot_max = 1.0) {
  JointTeleopHandler h;
  h.Init(Eigen::VectorXd::Constant(kN, q_curr),
         Eigen::VectorXd::Constant(kN, -q_lim),
         Eigen::VectorXd::Constant(kN, +q_lim),
         Eigen::VectorXd::Constant(kN, qdot_max));
  return h;
}

// ── Init ────────────────────────────────────────────────────────────────────

TEST(JointTeleopHandler, NotInitializedByDefault) {
  JointTeleopHandler h;
  EXPECT_FALSE(h.IsInitialized());
}

TEST(JointTeleopHandler, InitSetsDesiredAndGoalToCurrentPos) {
  auto h = MakeHandler(0.5);
  EXPECT_TRUE(h.IsInitialized());
  EXPECT_NEAR(h.Desired()[0], 0.5, kEps);
  EXPECT_NEAR(h.Goal()[0],    0.5, kEps);
}

// ── SetVelocity ─────────────────────────────────────────────────────────────

TEST(JointTeleopHandler, SetVelocityIntegratesIntoGoal) {
  // vel=0.5, dt=0.1 → step=0.05 (< qdot_max*dt=0.1) → goal=0.05
  auto h = MakeHandler(0.0);
  h.SetVelocity(Eigen::VectorXd::Constant(kN, 0.5), 0.1);
  EXPECT_NEAR(h.Goal()[0], 0.05, kEps);
}

TEST(JointTeleopHandler, SetVelocityKeepsDesiredInSyncWithGoal) {
  auto h = MakeHandler(0.0);
  h.SetVelocity(Eigen::VectorXd::Constant(kN, 0.5), 0.1);
  EXPECT_NEAR(h.Desired()[0], h.Goal()[0], kEps);
}

TEST(JointTeleopHandler, SetVelocityClampsPositiveExcess) {
  // vel=10 >> qdot_max=1, dt=0.1 → step clamped to 0.1 → goal=0.1
  auto h = MakeHandler(0.0);
  h.SetVelocity(Eigen::VectorXd::Constant(kN, 10.0), 0.1);
  EXPECT_NEAR(h.Goal()[0], 0.1, kEps);
}

TEST(JointTeleopHandler, SetVelocityClampsNegativeExcess) {
  auto h = MakeHandler(0.0);
  h.SetVelocity(Eigen::VectorXd::Constant(kN, -10.0), 0.1);
  EXPECT_NEAR(h.Goal()[0], -0.1, kEps);
}

TEST(JointTeleopHandler, SetVelocityClampsToPositionMax) {
  // start=0.95, vel=1, dt=0.1 → step=0.1 → would be 1.05, clamped to 1.0
  auto h = MakeHandler(0.95);
  h.SetVelocity(Eigen::VectorXd::Constant(kN, 1.0), 0.1);
  EXPECT_NEAR(h.Goal()[0], 1.0, kEps);
}

TEST(JointTeleopHandler, SetVelocityClampsToPositionMin) {
  auto h = MakeHandler(-0.95);
  h.SetVelocity(Eigen::VectorXd::Constant(kN, -1.0), 0.1);
  EXPECT_NEAR(h.Goal()[0], -1.0, kEps);
}

TEST(JointTeleopHandler, SetVelocityIgnoresNonPositiveDt) {
  auto h = MakeHandler(0.0);
  h.SetVelocity(Eigen::VectorXd::Constant(kN, 1.0), 0.0);
  EXPECT_NEAR(h.Goal()[0], 0.0, kEps);
  h.SetVelocity(Eigen::VectorXd::Constant(kN, 1.0), -0.05);
  EXPECT_NEAR(h.Goal()[0], 0.0, kEps);
}

// ── SetPosition ─────────────────────────────────────────────────────────────

TEST(JointTeleopHandler, SetPositionUpdatesGoalButNotDesired) {
  auto h = MakeHandler(0.0);
  h.SetPosition(Eigen::VectorXd::Constant(kN, 0.8));
  EXPECT_NEAR(h.Goal()[0],    0.8, kEps);
  EXPECT_NEAR(h.Desired()[0], 0.0, kEps);  // unchanged until Update()
}

TEST(JointTeleopHandler, SetPositionClampsToMax) {
  auto h = MakeHandler(0.0);
  h.SetPosition(Eigen::VectorXd::Constant(kN, 99.0));
  EXPECT_NEAR(h.Goal()[0], 1.0, kEps);
}

TEST(JointTeleopHandler, SetPositionClampsToMin) {
  auto h = MakeHandler(0.0);
  h.SetPosition(Eigen::VectorXd::Constant(kN, -99.0));
  EXPECT_NEAR(h.Goal()[0], -1.0, kEps);
}

// ── Update: rate-limited tracking ───────────────────────────────────────────

TEST(JointTeleopHandler, UpdateStepsDesiredTowardGoalAtRateLimit) {
  // q_des_smooth_=0, goal=1, qdot_max=1, dt=0.1 → one step = 0.1
  auto h = MakeHandler(0.0);
  h.SetPosition(Eigen::VectorXd::Constant(kN, 1.0));

  MockTask task;
  h.Update(&task, 0.1);

  EXPECT_NEAR(h.Desired()[0], 0.1, kEps);
  EXPECT_NEAR(task.pos[0],    0.1, kEps);
}

TEST(JointTeleopHandler, UpdateReachesGoalAfterSufficientTicks) {
  // goal=0.5, qdot_max=1, dt=0.1 → 5 ticks minimum; run 10 to be safe
  auto h = MakeHandler(0.0);
  h.SetPosition(Eigen::VectorXd::Constant(kN, 0.5));

  MockTask task;
  for (int i = 0; i < 10; ++i) h.Update(&task, 0.1);

  EXPECT_NEAR(h.Desired()[0], 0.5, kEps);
}

TEST(JointTeleopHandler, UpdatePassesZeroVelAndAccToTask) {
  auto h = MakeHandler(0.3);
  MockTask task;
  h.Update(&task, 0.01);

  EXPECT_NEAR(task.vel[0], 0.0, kEps);
  EXPECT_NEAR(task.acc[0], 0.0, kEps);
}

TEST(JointTeleopHandler, UpdateWithNullTaskDoesNotCrash) {
  auto h = MakeHandler(0.0);
  EXPECT_NO_FATAL_FAILURE(h.Update(static_cast<MockTask*>(nullptr), 0.01));
}

TEST(JointTeleopHandler, SetVelocityFollowedByUpdateHoldsDesired) {
  // SetVelocity keeps q_des_smooth_ in sync → Update with no goal change is a no-op
  auto h = MakeHandler(0.0);
  h.SetVelocity(Eigen::VectorXd::Constant(kN, 0.5), 0.1);  // desired=goal=0.05

  MockTask task;
  h.Update(&task, 0.1);  // goal==desired → no movement

  EXPECT_NEAR(h.Desired()[0], 0.05, kEps);
}

}  // namespace
}  // namespace wbc
