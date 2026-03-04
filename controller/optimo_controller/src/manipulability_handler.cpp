/**
 * @file optimo_controller/src/manipulability_handler.cpp
 * @brief Manipulability-gradient singularity avoidance implementation.
 */
#include "optimo_controller/manipulability_handler.hpp"

#include <algorithm>
#include <cmath>

#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/multibody/data.hpp>

#include "wbc_robot_system/pinocchio_robot_system.hpp"

namespace wbc {

// --------------------------------------------------------------------------
// Special members (must be in .cpp where pinocchio::Data is complete)
// --------------------------------------------------------------------------
ManipulabilityHandler::~ManipulabilityHandler() = default;
ManipulabilityHandler::ManipulabilityHandler(ManipulabilityHandler&&) noexcept = default;
ManipulabilityHandler& ManipulabilityHandler::operator=(ManipulabilityHandler&&) noexcept = default;

// --------------------------------------------------------------------------
void ManipulabilityHandler::Init(PinocchioRobotSystem* robot,
                                  int ee_frame_idx,
                                  const Config& config) {
  robot_ = robot;
  ee_frame_idx_ = ee_frame_idx;
  config_ = config;

  num_qdot_      = robot_->NumQdot();
  num_active_dof_ = robot_->NumActiveDof();
  num_float_dof_ = robot_->NumFloatDof();

  // Pre-allocate scratch buffers.
  data_scratch_ = std::make_unique<pinocchio::Data>(robot_->GetModel());
  J_.setZero(6, num_qdot_);
  q_scratch_.setZero(robot_->GetQRef().size());
  gradient_.setZero(num_active_dof_);
  qdot_avoid_.setZero(num_active_dof_);

  w_ = 0.0;
  sigma_min_ = 1.0;
  tick_count_ = 0;
}

// --------------------------------------------------------------------------
double ManipulabilityHandler::ComputeManipulability(
    const Eigen::VectorXd& q_full) {
  J_.setZero();
  pinocchio::computeFrameJacobian(
      robot_->GetModel(), *data_scratch_, q_full,
      static_cast<pinocchio::FrameIndex>(ee_frame_idx_),
      pinocchio::LOCAL_WORLD_ALIGNED,
      J_);
  // w = sqrt(det(J * J^T))   (6x6 for 6-row Jacobian)
  const Eigen::Matrix<double, 6, 6> JJt = J_ * J_.transpose();
  return std::sqrt(std::max(0.0, JJt.determinant()));
}

// --------------------------------------------------------------------------
void ManipulabilityHandler::ComputeGradient() {
  q_scratch_ = robot_->GetQRef();
  w_ = ComputeManipulability(q_scratch_);

  for (int i = 0; i < num_active_dof_; ++i) {
    const int qi = num_float_dof_ + i;
    q_scratch_[qi] += config_.fd_epsilon;
    const double w_plus = ComputeManipulability(q_scratch_);
    gradient_[i] = (w_plus - w_) / config_.fd_epsilon;
    q_scratch_[qi] -= config_.fd_epsilon;  // restore
  }
}

// --------------------------------------------------------------------------
void ManipulabilityHandler::Update(double dt) {
  if (robot_ == nullptr || dt <= 0.0) return;

  // Compute current manipulability + σ_min (every tick).
  w_ = ComputeManipulability(robot_->GetQRef());
  const Eigen::JacobiSVD<Eigen::MatrixXd> svd(J_, Eigen::ComputeThinV);
  const auto& sv = svd.singularValues();
  sigma_min_ = (sv.size() > 0) ? sv(sv.size() - 1) : 0.0;

  // Rate-limited gradient update.
  if (++tick_count_ >= config_.gradient_interval) {
    tick_count_ = 0;
    ComputeGradient();
  }

  // Compute avoidance velocity when near singularity.
  qdot_avoid_.setZero();
  if (sigma_min_ < config_.sigma_threshold) {
    const double gn = gradient_.norm();
    if (gn > 1e-10) {
      const double alpha =
          config_.step_size * (1.0 - sigma_min_ / config_.sigma_threshold);
      qdot_avoid_ = alpha * (gradient_ / gn);
    }
  }
}

}  // namespace wbc
