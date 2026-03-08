/**
 * @file wbc_handlers/src/manipulability_handler.cpp
 * @brief Log-manipulability gradient posture bias implementation.
 */
#include "wbc_handlers/manipulability_handler.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

#include <Eigen/SVD>

#include "wbc_robot_system/pinocchio_robot_system.hpp"

namespace wbc {

// --------------------------------------------------------------------------
void ManipulabilityHandler::ClampEach(Eigen::VectorXd& x, double abs_limit) {
  for (int i = 0; i < x.size(); ++i) {
    x[i] = std::clamp(x[i], -abs_limit, abs_limit);
  }
}

// --------------------------------------------------------------------------
void ManipulabilityHandler::Init(PinocchioRobotSystem* robot,
                                 int ee_frame_idx,
                                 const Config& config) {
  if (robot == nullptr)
    throw std::invalid_argument("ManipulabilityHandler::Init: robot is null");
  if (config.sigma_threshold <= 0.0)
    throw std::invalid_argument("ManipulabilityHandler::Init: sigma_threshold must be > 0");
  if (config.fd_eps <= 0.0)
    throw std::invalid_argument("ManipulabilityHandler::Init: fd_eps must be > 0");
  if (config.sigma_eps <= 0.0)
    throw std::invalid_argument("ManipulabilityHandler::Init: sigma_eps must be > 0");
  if (config.gain < 0.0)
    throw std::invalid_argument("ManipulabilityHandler::Init: gain must be >= 0");
  if (config.max_bias_qdot < 0.0)
    throw std::invalid_argument("ManipulabilityHandler::Init: max_bias_qdot must be >= 0");
  if (config.use_full_jacobian && config.characteristic_length <= 0.0)
    throw std::invalid_argument("ManipulabilityHandler::Init: characteristic_length must be > 0 when use_full_jacobian is true");

  robot_ = robot;
  ee_frame_idx_ = ee_frame_idx;
  config_ = config;
  num_active_dof_ = robot_->NumActiveDof();
  num_float_dof_ = robot_->NumFloatDof();

  // Cache joint limits for FD boundary clamping. JointPosLimits() returns
  // an n_active × 2 matrix: col 0 = lower bound, col 1 = upper bound.
  joint_pos_limits_ = robot_->JointPosLimits();
  assert(joint_pos_limits_.rows() == num_active_dof_);
  assert(joint_pos_limits_.cols() == 2);

  grad_logw_.setZero(num_active_dof_);
  bias_qdot_.setZero(num_active_dof_);
  sigma_min_ = 0.0;
  logw_      = 0.0;
  is_active_ = false;
}

// --------------------------------------------------------------------------
// SVD note: singular values only — no U/V decomposition needed.
std::pair<double, double> ManipulabilityHandler::ComputeMetrics(
    const Eigen::MatrixXd& J) const {
  if (J.rows() == 0 || J.cols() == 0) {
    return {0.0, -std::numeric_limits<double>::infinity()};
  }
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(J);
  const Eigen::VectorXd& S = svd.singularValues();
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

// --------------------------------------------------------------------------
// Jacobian convention: FillLinkJacobian reorders Pinocchio's [linear;angular]
// output to [angular(0:3); linear(3:6)].
//   topRows(3)    = angular velocity Jacobian
//   bottomRows(3) = linear  velocity Jacobian
Eigen::MatrixXd ManipulabilityHandler::ComputeCurrentTaskJacobian() const {
  Eigen::MatrixXd jac =
      Eigen::MatrixXd::Zero(6, num_float_dof_ + num_active_dof_);
  robot_->FillLinkJacobian(ee_frame_idx_, jac);
  Eigen::MatrixXd J_active = jac.rightCols(num_active_dof_);

  if (!config_.use_full_jacobian) {
    // Return linear-velocity rows only (bottom 3 rows after reorder).
    return J_active.bottomRows(3);
  }
  // Scale angular rows (top 3) by characteristic_length to reduce unit
  // mismatch between angular [rad/s] and linear [m/s] contributions.
  J_active.topRows(3) *= config_.characteristic_length;
  return J_active;
}

// --------------------------------------------------------------------------
Eigen::MatrixXd ManipulabilityHandler::ComputeTaskJacobianAtQ(
    const Eigen::VectorXd& q_active) {
  // GetQRef()/GetQdotRef() return q_/qdot_ — the robot's current model state
  // set by UpdateRobotModel() on this tick. NOT a separate desired/reference.
  const Eigen::VectorXd q_full_orig = robot_->GetQRef();
  const Eigen::VectorXd qdot_full_orig = robot_->GetQdotRef();
  const Eigen::VectorXd qdot_active = qdot_full_orig.tail(num_active_dof_);

  // Reconstruct base state for floating-base robots.
  // For fixed-base (num_float_dof_==0) the base args are ignored by UpdateRobotModel.
  Eigen::Vector3d    base_pos     = Eigen::Vector3d::Zero();
  Eigen::Quaterniond base_quat    = Eigen::Quaterniond::Identity();
  Eigen::Vector3d    base_lin_vel = Eigen::Vector3d::Zero();
  Eigen::Vector3d    base_ang_vel = Eigen::Vector3d::Zero();

  if (num_float_dof_ > 0) {
    base_pos = q_full_orig.head<3>();
    // Pinocchio stores quaternion as [x, y, z, w] (coeffs() order).
    base_quat = Eigen::Quaterniond(
        q_full_orig[6], q_full_orig[3], q_full_orig[4], q_full_orig[5]);
    base_quat.normalize();
    base_lin_vel = qdot_full_orig.head<3>();
    base_ang_vel = qdot_full_orig.segment<3>(3);
  }

  // update_centroid=false: runs forwardKinematics + computeJointJacobians +
  // updateFramePlacements (everything FillLinkJacobian needs), skips only
  // centroidal quantities (CoM, momentum) — safe for Jacobian-only FD passes.
  robot_->UpdateRobotModel(base_pos, base_quat, base_lin_vel, base_ang_vel,
                           q_active, qdot_active, false);
  const Eigen::MatrixXd J = ComputeCurrentTaskJacobian();

  // Restore original state.
  robot_->UpdateRobotModel(base_pos, base_quat, base_lin_vel, base_ang_vel,
                           q_full_orig.tail(num_active_dof_), qdot_active, false);
  return J;
}

// --------------------------------------------------------------------------
void ManipulabilityHandler::Update(double /*dt*/) {
  if (robot_ == nullptr) return;

  grad_logw_.setZero();
  bias_qdot_.setZero();
  is_active_ = false;

  // GetQRef() returns the internal model state q_ — already synced to the
  // current tick by ControlArchitecture before any state machine step.
  const Eigen::VectorXd q_active = robot_->GetQRef().tail(num_active_dof_);

  const Eigen::MatrixXd J0      = ComputeCurrentTaskJacobian();
  auto [sigma_min, logw]         = ComputeMetrics(J0);
  sigma_min_ = sigma_min;
  logw_      = logw;

  // Smooth activation: ramps from 0 to 1 as σ_min falls from sigma_threshold to 0.
  const double activation =
      std::clamp((config_.sigma_threshold - sigma_min_) / config_.sigma_threshold,
                 0.0, 1.0);
  if (activation <= 0.0) return;

  // Central (or one-sided) finite-difference gradient of log(manipulability).
  // Near joint limits, use one-sided differences to stay within reachable set.
  const double h = config_.fd_eps;
  for (int i = 0; i < num_active_dof_; ++i) {
    const double q_i  = q_active[i];
    const double q_lo = joint_pos_limits_(i, 0);
    const double q_hi = joint_pos_limits_(i, 1);

    const bool can_plus  = (q_i + h <= q_hi);
    const bool can_minus = (q_i - h >= q_lo);

    if (can_plus && can_minus) {
      // Central difference — preferred for accuracy.
      Eigen::VectorXd q_plus  = q_active;
      Eigen::VectorXd q_minus = q_active;
      q_plus[i]  += h;
      q_minus[i] -= h;
      const double lw_plus  = ComputeMetrics(ComputeTaskJacobianAtQ(q_plus)).second;
      const double lw_minus = ComputeMetrics(ComputeTaskJacobianAtQ(q_minus)).second;
      grad_logw_[i] = (lw_plus - lw_minus) / (2.0 * h);
    } else if (can_plus) {
      // Forward difference (at lower limit).
      // Reuse current-state logw_ as the base term — saves one FD evaluation.
      Eigen::VectorXd q_plus = q_active;
      q_plus[i] += h;
      const double lw_plus = ComputeMetrics(ComputeTaskJacobianAtQ(q_plus)).second;
      grad_logw_[i] = (lw_plus - logw_) / h;
    } else if (can_minus) {
      // Backward difference (at upper limit).
      // Reuse current-state logw_ as the base term — saves one FD evaluation.
      Eigen::VectorXd q_minus = q_active;
      q_minus[i] -= h;
      const double lw_minus = ComputeMetrics(ComputeTaskJacobianAtQ(q_minus)).second;
      grad_logw_[i] = (logw_ - lw_minus) / h;
    } else {
      // Joint window smaller than 2h — skip (shouldn't happen with sane limits).
      grad_logw_[i] = 0.0;
    }
  }

  const double grad_norm = grad_logw_.norm();
  if (grad_norm < 1e-10) return;

  is_active_ = true;
  bias_qdot_ = config_.gain * activation * (grad_logw_ / grad_norm);
  ClampEach(bias_qdot_, config_.max_bias_qdot);
}

}  // namespace wbc
