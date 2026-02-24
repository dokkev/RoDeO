#include <stdexcept>

#include <gtest/gtest.h>

#include "rpc_example/ExampleController.hpp"
#include "rpc_example/example_state_provider.hpp"

namespace wbc {
namespace {

TEST(ExampleStateProviderTest, DefaultInitialization) {
  ExampleStateProvider sp;

  EXPECT_DOUBLE_EQ(sp.servo_dt_, 0.001);
  EXPECT_DOUBLE_EQ(sp.current_time_, 0.0);
  EXPECT_EQ(sp.count_, 0);
  EXPECT_EQ(sp.state_, 0);
  EXPECT_EQ(sp.prev_state_, 0);

  EXPECT_TRUE(sp.des_ee_iso_.matrix().isApprox(Eigen::Matrix4d::Identity()));
  EXPECT_TRUE(
      sp.rot_world_local_.isApprox(Eigen::Matrix3d::Identity(), 1.0e-12));
  EXPECT_FALSE(sp.b_ee_contact_);

  EXPECT_EQ(sp.teleop_cmd_seq_, 0u);
  EXPECT_DOUBLE_EQ(sp.teleop_cmd_time_sec_, 0.0);
  EXPECT_TRUE(
      sp.teleop_raw_pose_.matrix().isApprox(Eigen::Matrix4d::Identity()));
}

TEST(ExampleStateProviderTest, CustomDtInitialization) {
  constexpr double kDt = 0.0025;
  ExampleStateProvider sp(kDt);

  EXPECT_DOUBLE_EQ(sp.servo_dt_, kDt);
  EXPECT_EQ(sp.data_save_freq_, 1);
}

TEST(ExampleStateProviderTest, DeterministicInitialization) {
  ExampleStateProvider a(0.001);
  ExampleStateProvider b(0.001);

  EXPECT_DOUBLE_EQ(a.servo_dt_, b.servo_dt_);
  EXPECT_DOUBLE_EQ(a.current_time_, b.current_time_);
  EXPECT_EQ(a.count_, b.count_);
  EXPECT_EQ(a.state_, b.state_);
  EXPECT_EQ(a.prev_state_, b.prev_state_);
  EXPECT_TRUE(a.des_ee_iso_.matrix().isApprox(b.des_ee_iso_.matrix()));
  EXPECT_TRUE(a.rot_world_local_.isApprox(b.rot_world_local_));
  EXPECT_EQ(a.b_ee_contact_, b.b_ee_contact_);
}

TEST(ExampleControllerTest, ThrowsWhenRobotIsNull) {
  EXPECT_THROW(
      {
        const ExampleController controller(nullptr, "dummy.yaml", 0.001);
        (void)controller;
      },
      std::runtime_error);
}

} // namespace
} // namespace wbc

