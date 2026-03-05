/**
 * @file wbc_handlers/src/manipulability_handler.cpp
 * @brief Manipulability-gradient singularity avoidance implementation.
 */
#include "wbc_handlers/manipulability_handler.hpp"

#include "wbc_robot_system/pinocchio_robot_system.hpp"

namespace wbc {

// --------------------------------------------------------------------------
void ManipulabilityHandler::Init(PinocchioRobotSystem* robot,
                                 int ee_frame_idx,
                                 const Config& config) {
  robot_ = robot;
  ee_frame_idx_ = ee_frame_idx;
  config_ = config;

  num_active_dof_ = robot_->NumActiveDof();
  num_float_dof_  = robot_->NumFloatDof();

  // Pre-allocate scratch buffers (zero heap allocation in RT loop).
  q_scratch_.setZero(robot_->GetQRef().size());
  gradient_.setZero(num_active_dof_);
  qdot_avoid_.setZero(num_active_dof_);

  w_ = 0.0;
  fd_current_dof_ = 0;
}

// --------------------------------------------------------------------------
void ManipulabilityHandler::Update(double dt) {
  if (robot_ == nullptr || dt <= 0.0) return;

  // 1. Amortized gradient computation: 1 DOF per tick (round-robin).
  const int qi = num_float_dof_ + fd_current_dof_;

  q_scratch_ = robot_->GetQRef();
  const double w_center = robot_->ComputeManipulability(ee_frame_idx_, q_scratch_);
  w_ = w_center;

  // Forward finite difference for current DOF only.
  q_scratch_[qi] += config_.fd_epsilon;
  const double w_plus = robot_->ComputeManipulability(ee_frame_idx_, q_scratch_);
  gradient_[fd_current_dof_] = (w_plus - w_center) / config_.fd_epsilon;

  // Advance round-robin index.
  fd_current_dof_ = (fd_current_dof_ + 1) % num_active_dof_;

  // 2. Avoidance velocity when w is below threshold.
  qdot_avoid_.setZero();

  if (w_ < config_.w_threshold) {
    const double gn = gradient_.norm();
    if (gn > 1e-10) {
      const double alpha =
          config_.step_size * (1.0 - w_ / config_.w_threshold);
      qdot_avoid_ = alpha * (gradient_ / gn);
    }
  }
}

}  // namespace wbc
