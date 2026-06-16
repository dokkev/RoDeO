//
// Copyright (c) 2026
//
// Model-side robot command holder.
//

#ifndef WBC_CORE_ROBOTS_ROBOT_COMMAND_HPP_
#define WBC_CORE_ROBOTS_ROBOT_COMMAND_HPP_

#include <Eigen/Core>

#include "wbc_core/math/fwd.hpp"
#include "wbc_core/robots/fwd.hpp"

namespace wbc {
namespace robots {

/// Model-side command produced by the host controller.
///
/// The struct name carries the command-layer meaning, so fields intentionally
/// do not repeat the `_cmd` suffix.
struct RobotCommand {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  math::Vector q;     ///< Joint position command, size nq_joints().
  math::Vector qdot;  ///< Joint velocity command, size nv_joints().
  math::Vector tau;   ///< Actuator torque command, size na().

  /// Allocate zero command buffers for a robot model.
  void Initialize(const RobotSystem& robot);

  /// Allocate zero command buffers for fixed-size joint command users.
  void Initialize(Eigen::Index joint_dim);

  /// Allocate zero command buffers with explicit model dimensions.
  void Initialize(Eigen::Index nq_joints, Eigen::Index nv_joints,
                  Eigen::Index na);
};

}  // namespace robots
}  // namespace wbc

#endif  // WBC_CORE_ROBOTS_ROBOT_COMMAND_HPP_
