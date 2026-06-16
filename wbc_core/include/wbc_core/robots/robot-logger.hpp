//
// Copyright (c) 2026
//
// Robot command/debug logging holder.
//

#ifndef WBC_CORE_ROBOTS_ROBOT_LOGGER_HPP_
#define WBC_CORE_ROBOTS_ROBOT_LOGGER_HPP_

#include <Eigen/Core>

#include "wbc_core/math/fwd.hpp"
#include "wbc_core/robots/robot-command.hpp"
#include "wbc_core/robots/fwd.hpp"

namespace wbc {
namespace robots {

/// Per-tick robot command trace.
///
/// This holder records the final command payload plus solver and command
/// components for debugging and telemetry. It is not the hardware command
/// writer.
class RobotLogger {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /// Initialize command payload and trace buffers for a robot model.
  void Initialize(const RobotSystem& robot);

  /// Update final model-side command payload.
  void UpdateCommand(const RobotCommand& command);

  RobotCommand cmd;         ///< Final model-side command payload.
  math::Vector qddot_sol;   ///< Solved generalized acceleration, size nv().
  math::Vector q_cmd;       ///< Joint position command, size nq_joints().
  math::Vector qdot_cmd;    ///< Joint velocity command, size nv_joints().
  math::Vector tau_ff_cmd;  ///< Feedforward actuator torque command, size na().
  math::Vector tau_fb_cmd;  ///< Feedback actuator torque command, size na().
  math::Vector tau_cmd;     ///< Final actuator torque command, size na().
};

}  // namespace robots
}  // namespace wbc

#endif  // WBC_CORE_ROBOTS_ROBOT_LOGGER_HPP_
