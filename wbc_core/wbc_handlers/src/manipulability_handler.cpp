/**
 * @file wbc_handlers/src/manipulability_handler.cpp
 * @brief SVD-based singularity avoidance implementation.
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

  qdot_avoid_.setZero(num_active_dof_);

  w_ = 0.0;
  sigma_min_ = 0.0;
}

// --------------------------------------------------------------------------
void ManipulabilityHandler::Update(double dt) {
  if (robot_ == nullptr || dt <= 0.0) return;

  // 1. Get the EE Jacobian and compute SVD.
  Eigen::MatrixXd jac = Eigen::MatrixXd::Zero(6, num_float_dof_ + num_active_dof_);
  robot_->FillLinkJacobian(ee_frame_idx_, jac);

  // Extract active-joint columns only (skip floating base).
  const Eigen::MatrixXd J_active = jac.rightCols(num_active_dof_);

  Eigen::JacobiSVD<Eigen::MatrixXd> svd(J_active, Eigen::ComputeThinV);
  const auto& S = svd.singularValues();
  sigma_min_ = S(S.size() - 1);

  // Yoshikawa manipulability = product of singular values.
  w_ = 1.0;
  for (int i = 0; i < S.size(); ++i) w_ *= S(i);

  // 2. Avoidance velocity when σ_min is below threshold.
  qdot_avoid_.setZero();

  if (sigma_min_ < config_.w_threshold) {
    // Right singular vector for smallest σ — the joint-space direction
    // that is "lost" at the singularity.
    const Eigen::VectorXd v_min = svd.matrixV().col(S.size() - 1);

    const double alpha =
        config_.step_size * (1.0 - sigma_min_ / config_.w_threshold);
    qdot_avoid_ = alpha * v_min;
  }
}

}  // namespace wbc
