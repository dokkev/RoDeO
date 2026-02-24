#pragma once

#include <cstdint>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include "wbc_robot_system/state_provider.hpp"

namespace wbc {

// Robot-specific shared state container for Optimo.
class OptimoStateProvider final : public StateProvider {
public:
  explicit OptimoStateProvider(double dt = 0.001);
  ~OptimoStateProvider() override = default;

  // ---------------------------------------------------------------------------
  // Optimo robot state shared across controller/state-machine layers.
  // Keep this container state-focused (no orchestration flags).
  // ---------------------------------------------------------------------------
  int data_save_freq_{1};

  Eigen::Isometry3d des_ee_iso_{Eigen::Isometry3d::Identity()};

  bool b_f1_contact_{false};
  bool b_f2_contact_{false};
  bool b_f3_contact_{false};

  int planning_id_{0};

  // Teleop raw target buffer (written by controller, consumed by state).
  std::uint64_t teleop_cmd_seq_{0};
  double teleop_cmd_time_sec_{0.0};
  Eigen::Isometry3d teleop_raw_pose_{Eigen::Isometry3d::Identity()};
};

} // namespace wbc
