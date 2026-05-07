#include "wbc_core/handlers/manipulability_handler.hpp"

#include <algorithm>
#include <iostream>
#include <cmath>
#include <limits>
#include <stdexcept>

#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/kinematics.hpp>

namespace wbc {

void ManipulabilityHandler::ClampEach(Eigen::VectorXd& x, double abs_limit) {
  for (int i = 0; i < x.size(); ++i) {
    x[i] = std::clamp(x[i], -abs_limit, abs_limit);
  }
}

void ManipulabilityHandler::Init(tsid::robots::RobotWrapper* robot,
                                 pinocchio::Data* data,
                                 int ee_frame_idx,
                                 const Config& config) {
  if (!robot) throw std::invalid_argument("ManipulabilityHandler::Init: robot is null");
  if (!data)  throw std::invalid_argument("ManipulabilityHandler::Init: data is null");
  if (config.sigma_threshold <= 0.0)
    throw std::invalid_argument("ManipulabilityHandler::Init: sigma_threshold must be > 0");
  if (config.fd_eps <= 0.0)
    throw std::invalid_argument("ManipulabilityHandler::Init: fd_eps must be > 0");

  // Validate ee_frame_idx against the model's frame count.
  const auto n_frames =
      static_cast<int>(robot->model().frames.size());
  if (ee_frame_idx < 0 || ee_frame_idx >= n_frames) {
    std::cerr << "ManipulabilityHandler::Init: ee_frame_idx=" << ee_frame_idx
              << " out of range [0, " << n_frames << ")\n";
    robot_ = nullptr;
    return;
  }

  robot_ = robot;
  data_ = data;
  ee_frame_idx_ = ee_frame_idx;
  config_ = config;

  na_ = robot_->na();
  nv_ = robot_->nv();
  nq_ = robot_->nq();
  is_fixed_base_ = robot_->is_fixed_base();

  // Create a separate pinocchio::Data for FD evaluations to avoid
  // corrupting the shared formulation data.
  fd_data_ = pinocchio::Data(robot_->model());

  grad_logw_.setZero(na_);
  bias_qdot_.setZero(na_);
  jac_buf_.setZero(6, nv_);
  v_zero_.setZero(nv_);
  q_scratch_.setZero(nq_);
  // J_active_ size depends on use_full_jacobian: 6 x na_ or 3 x na_
  const int jrows = config_.use_full_jacobian ? 6 : 3;
  J_active_.setZero(jrows, na_);
  sigma_min_ = 0.0;
  logw_ = 0.0;
  is_active_ = false;
}

std::pair<double, double> ManipulabilityHandler::ComputeMetrics() const {
  if (J_active_.rows() == 0 || J_active_.cols() == 0) {
    return {0.0, -std::numeric_limits<double>::infinity()};
  }
  svd_.compute(J_active_);
  const auto& S = svd_.singularValues();
  if (S.size() == 0) {
    return {0.0, -std::numeric_limits<double>::infinity()};
  }
  const double sigma_min = S.minCoeff();
  double logw = 0.0;
  for (int i = 0; i < S.size(); ++i) {
    logw += std::log(std::max(S[i], config_.sigma_eps));
  }
  return {sigma_min, logw};
}

void ManipulabilityHandler::ComputeTaskJacobian(
    const Eigen::VectorXd& q_full) {
  const auto& model = robot_->model();

  // Compute FK + Jacobians at q_full using the FD scratch data.
  pinocchio::forwardKinematics(model, fd_data_, q_full, v_zero_);
  pinocchio::updateFramePlacements(model, fd_data_);

  jac_buf_.setZero();
  pinocchio::computeFrameJacobian(
      model, fd_data_, q_full,
      static_cast<pinocchio::FrameIndex>(ee_frame_idx_),
      pinocchio::WORLD, jac_buf_);

  // Extract active-joint columns (skip floating base) into pre-allocated buffer.
  if (!config_.use_full_jacobian) {
    // Pinocchio WORLD Jacobian: [linear(0:3); angular(3:6)]
    J_active_.noalias() = jac_buf_.topRows(3).rightCols(na_);
  } else {
    J_active_ = jac_buf_.rightCols(na_);
    // Scale angular rows to reduce unit mismatch.
    J_active_.bottomRows(3) *= config_.characteristic_length;
  }
}

void ManipulabilityHandler::Update(const Eigen::VectorXd& q_current) {
  if (!robot_) return;

  grad_logw_.setZero();
  bias_qdot_.setZero();
  is_active_ = false;

  // Compute metrics at current configuration.
  ComputeTaskJacobian(q_current);
  auto [sigma_min, logw] = ComputeMetrics();
  sigma_min_ = sigma_min;
  logw_ = logw;

  // Smooth activation.
  const double activation =
      std::clamp((config_.sigma_threshold - sigma_min_) / config_.sigma_threshold,
                 0.0, 1.0);
  if (activation <= 0.0) return;

  // Central finite-difference gradient of log(manipulability).
  const double h = config_.fd_eps;
  const auto& model = robot_->model();

  // Index into q_full for active joints.
  // For fixed base: active joints start at index 0.
  // For floating base: active joints start at index 7 (nq - na for config).
  const int q_offset = is_fixed_base_ ? 0 : 7;

  // Use pre-allocated q_scratch_ instead of per-joint VectorXd copies.
  q_scratch_ = q_current;

  for (int i = 0; i < na_; ++i) {
    const int qi = q_offset + i;
    const double q_lo = model.lowerPositionLimit[qi];
    const double q_hi = model.upperPositionLimit[qi];
    const double q_i = q_current[qi];

    const bool can_plus  = (q_i + h <= q_hi);
    const bool can_minus = (q_i - h >= q_lo);

    if (can_plus && can_minus) {
      q_scratch_[qi] = q_i + h;
      ComputeTaskJacobian(q_scratch_);
      const double lw_plus = ComputeMetrics().second;
      q_scratch_[qi] = q_i - h;
      ComputeTaskJacobian(q_scratch_);
      const double lw_minus = ComputeMetrics().second;
      grad_logw_[i] = (lw_plus - lw_minus) / (2.0 * h);
    } else if (can_plus) {
      q_scratch_[qi] = q_i + h;
      ComputeTaskJacobian(q_scratch_);
      const double lw_plus = ComputeMetrics().second;
      grad_logw_[i] = (lw_plus - logw_) / h;
    } else if (can_minus) {
      q_scratch_[qi] = q_i - h;
      ComputeTaskJacobian(q_scratch_);
      const double lw_minus = ComputeMetrics().second;
      grad_logw_[i] = (logw_ - lw_minus) / h;
    }
    // Restore the original value for the next joint's perturbation.
    q_scratch_[qi] = q_i;
  }

  const double grad_norm = grad_logw_.norm();
  if (grad_norm < 1e-10) return;

  is_active_ = true;
  bias_qdot_ = config_.gain * activation * (grad_logw_ / grad_norm);
  ClampEach(bias_qdot_, config_.max_bias_qdot);
}

}  // namespace wbc
