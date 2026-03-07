/**
 * @file residual_compensator/src/adaptive_friction_compensator.cpp
 * @brief Adaptive friction compensator implementation.
 */
#include "residual_compensator/adaptive_friction_compensator.hpp"

#include <algorithm>

namespace wbc {

void AdaptiveFrictionCompensator::Setup(int n) {
  n_ = n;
  f_c_est_.setZero(n);
  f_v_est_.setZero(n);
  gamma_c_.setZero(n);
  gamma_v_.setZero(n);
  max_f_c_.setConstant(n, 10.0);
  max_f_v_.setConstant(n, 5.0);
}

void AdaptiveFrictionCompensator::SetGains(
    const Eigen::Ref<const Eigen::VectorXd>& gamma_c,
    const Eigen::Ref<const Eigen::VectorXd>& gamma_v) {
  gamma_c_ = gamma_c;
  gamma_v_ = gamma_v;
}

void AdaptiveFrictionCompensator::SetLimits(
    const Eigen::Ref<const Eigen::VectorXd>& max_f_c,
    const Eigen::Ref<const Eigen::VectorXd>& max_f_v) {
  max_f_c_ = max_f_c;
  max_f_v_ = max_f_v;
}

void AdaptiveFrictionCompensator::Reset() {
  f_c_est_.setZero();
  f_v_est_.setZero();
}

void AdaptiveFrictionCompensator::Compute(
    const Eigen::Ref<const Eigen::VectorXd>& qdot_des,
    const Eigen::Ref<const Eigen::VectorXd>& qdot,
    double dt,
    Eigen::Ref<Eigen::VectorXd> out) {
  for (int i = 0; i < n_; ++i) {
    const double e_v = qdot_des[i] - qdot[i];
    const double sgn = DeadbandSign(qdot[i]);

    // Gradient update.
    f_c_est_[i] += gamma_c_[i] * e_v * sgn * dt;
    f_v_est_[i] += gamma_v_[i] * e_v * qdot[i] * dt;

    // Projection to feasible set [0, max].
    f_c_est_[i] = std::clamp(f_c_est_[i], 0.0, max_f_c_[i]);
    f_v_est_[i] = std::clamp(f_v_est_[i], 0.0, max_f_v_[i]);

    // Compensation torque (cancel friction → add in direction of motion).
    out[i] = f_c_est_[i] * sgn + f_v_est_[i] * qdot[i];
  }
}

double AdaptiveFrictionCompensator::DeadbandSign(double x, double threshold) {
  if (x > threshold) return 1.0;
  if (x < -threshold) return -1.0;
  return x / threshold;
}

}  // namespace wbc
