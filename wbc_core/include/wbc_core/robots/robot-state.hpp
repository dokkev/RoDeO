//
// Copyright (c) 2026
//
// Robot state holders.
//

#ifndef WBC_CORE_ROBOTS_ROBOT_STATE_HPP_
#define WBC_CORE_ROBOTS_ROBOT_STATE_HPP_

#include <optional>

#include <pinocchio/spatial/motion.hpp>
#include <pinocchio/spatial/se3.hpp>

#include "wbc_core/math/fwd.hpp"

namespace wbc {
namespace robots {

/// Joint state supplied by the caller.
struct JointState {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  math::Vector q;     ///< Joint position, size nq_joints().
  math::Vector qdot;  ///< Joint velocity, size nv_joints().
  math::Vector tau;   ///< Joint torque, size na().
};

/// Floating-base state supplied by the caller.
///
/// `pose_world_base` stores the base pose as SE(3). `twist_world_base` is the
/// 6D base twist used as Pinocchio generalized velocity for the free-flyer
/// root.
struct BaseState {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  pinocchio::SE3 pose_world_base{pinocchio::SE3::Identity()};
  pinocchio::Motion twist_world_base{pinocchio::Motion::Zero()};
};

/// User-facing semantic robot state.
struct RobotState {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  JointState joint;
  std::optional<BaseState> base;  ///< Only valid/used for floating-base robots.
};

/// Pinocchio-readable packed state cached by RobotSystem.
struct GeneralizedState {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  math::Vector q;  ///< Pinocchio generalized configuration, size nq().
  math::Vector v;  ///< Pinocchio generalized velocity, size nv().
};

}  // namespace robots
}  // namespace wbc

#endif  // WBC_CORE_ROBOTS_ROBOT_STATE_HPP_
