//
// Copyright (c) 2026
//
// Robot command/debug logging holder.
//

#ifndef WBC_CORE_ROBOTS_ROBOT_LOGGER_HPP_
#define WBC_CORE_ROBOTS_ROBOT_LOGGER_HPP_

#include <Eigen/Core>

#include "wbc_core/math/fwd.hpp"
#include "wbc_core/robots/robot-system.hpp"

namespace wbc {
namespace robots {

/// Per-tick robot command trace.
///
/// This holder records solver and command components separately for debugging
/// and telemetry. It is not the hardware command payload.
class RobotLogger {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /// Allocate zero log buffers for a robot model.
  void Initialize(const RobotSystem& robot) {
    qddot_sol = math::Vector::Zero(robot.nv());
    q_cmd = math::Vector::Zero(robot.nq_joints());
    qdot_cmd = math::Vector::Zero(robot.nv_joints());
    tau_ff_cmd = math::Vector::Zero(robot.na());
    tau_fb_cmd = math::Vector::Zero(robot.na());
    tau_cmd = math::Vector::Zero(robot.na());
  }

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
