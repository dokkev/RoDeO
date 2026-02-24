#pragma once

#include <cstdint>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include "wbc_robot_system/state_provider.hpp"

namespace wbc {

// Robot-specific shared state container for rpc_example.
class ExampleStateProvider final : public StateProvider {
public:
  explicit ExampleStateProvider(double dt = 0.001);
  ~ExampleStateProvider() override = default;

  int data_save_freq_{1};

  Eigen::Isometry3d des_ee_iso_{Eigen::Isometry3d::Identity()};
  bool b_ee_contact_{false};

  std::uint64_t teleop_cmd_seq_{0};
  double teleop_cmd_time_sec_{0.0};
  Eigen::Isometry3d teleop_raw_pose_{Eigen::Isometry3d::Identity()};
};

} // namespace wbc

