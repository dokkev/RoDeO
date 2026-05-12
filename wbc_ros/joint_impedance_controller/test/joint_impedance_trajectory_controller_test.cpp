#include <gtest/gtest.h>

#include <vector>

#include "joint_impedance_controller/impedance_trajectory_controller.hpp"

namespace {

std::vector<double> makeSeq(double start, double step, std::size_t n) {
  std::vector<double> out(n, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    out[i] = start + step * static_cast<double>(i);
  }
  return out;
}

}  // namespace

TEST(ImpedanceTrajectoryControllerTest, InitializesWithExpectedSizes) {
  ImpedanceTrajectoryController controller(8);
  controller.setGains({1.0, 2.0}, {0.1, 0.2});

  const auto cmd = controller.update(0.01);

  EXPECT_EQ(cmd.position.size(), 8u);
  EXPECT_EQ(cmd.velocity.size(), 8u);
  EXPECT_EQ(cmd.stiffness.size(), 8u);
  EXPECT_EQ(cmd.damping.size(), 8u);
  EXPECT_EQ(cmd.effort_ff.size(), 8u);
  EXPECT_DOUBLE_EQ(cmd.stiffness[0], 1.0);
  EXPECT_DOUBLE_EQ(cmd.stiffness[1], 2.0);
  EXPECT_DOUBLE_EQ(cmd.stiffness[2], 0.0);
}

TEST(ImpedanceTrajectoryControllerTest, IdleTracksMeasuredState) {
  ImpedanceTrajectoryController controller(8);

  const auto measured = makeSeq(0.0, 0.1, 8);
  controller.setMeasuredState(measured, std::vector<double>(8, 0.0));
  const auto cmd = controller.update(0.01);

  for (std::size_t i = 0; i < 8; ++i) {
    EXPECT_NEAR(cmd.position[i], measured[i], 1e-12);
    EXPECT_NEAR(cmd.velocity[i], 0.0, 1e-12);
  }
}

TEST(ImpedanceTrajectoryControllerTest, ExecutesTimeParameterizedGoal) {
  ImpedanceTrajectoryController controller(8);
  controller.setMeasuredState(std::vector<double>(8, 0.0), std::vector<double>(8, 0.0));

  const auto goal = std::vector<double>{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
  controller.setGoal(goal, 1.0);

  const auto halfway = controller.update(0.5);
  for (std::size_t i = 0; i < 8; ++i) {
    EXPECT_GT(halfway.position[i], 0.0);
    EXPECT_LT(halfway.position[i], 1.0);
  }

  const auto done = controller.update(0.5);
  for (std::size_t i = 0; i < 8; ++i) {
    EXPECT_NEAR(done.position[i], 1.0, 1e-9);
    EXPECT_NEAR(done.velocity[i], 0.0, 1e-9);
  }
}

TEST(ImpedanceTrajectoryControllerTest, PartialGoalDoesNotZeroUnspecifiedJoints) {
  ImpedanceTrajectoryController controller(8);
  const auto measured = makeSeq(10.0, 1.0, 8);
  controller.setMeasuredState(measured, std::vector<double>(8, 0.0));

  controller.setGoal({100.0, 200.0}, 0.1);
  const auto cmd = controller.update(0.1);

  EXPECT_NEAR(cmd.position[0], 100.0, 1e-9);
  EXPECT_NEAR(cmd.position[1], 200.0, 1e-9);
  for (std::size_t i = 2; i < 8; ++i) {
    EXPECT_NEAR(cmd.position[i], measured[i], 1e-9);
  }
}

TEST(ImpedanceTrajectoryControllerTest, HoldUsesLatestMeasuredState) {
  ImpedanceTrajectoryController controller(8);

  controller.setMeasuredState(std::vector<double>(8, 1.0), std::vector<double>(8, 0.0));
  (void)controller.update(0.01);

  controller.setMeasuredState(std::vector<double>(8, 2.0), std::vector<double>(8, 0.0));
  controller.holdPosition();
  const auto cmd = controller.update(0.01);

  for (double q : cmd.position) {
    EXPECT_NEAR(q, 2.0, 1e-9);
  }
}
