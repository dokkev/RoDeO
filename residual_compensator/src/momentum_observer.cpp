/**
 * @file residual_compensator/src/momentum_observer.cpp
 * @brief Momentum observer implementation.
 */
#include "residual_compensator/momentum_observer.hpp"

namespace wbc {

void MomentumObserver::Setup(int n) {
  n_ = n;
  p_hat_.setZero(n);
  tau_dist_est_.setZero(n);
  K_o_.setConstant(n, 50.0);
  max_tau_dist_.setConstant(n, 50.0);
  is_initialized_ = false;
}

void MomentumObserver::SetGain(const Eigen::Ref<const Eigen::VectorXd>& K_o) {
  K_o_ = K_o;
}

void MomentumObserver::SetLimit(
    const Eigen::Ref<const Eigen::VectorXd>& max_tau_dist) {
  max_tau_dist_ = max_tau_dist;
}

void MomentumObserver::Reset() {
  is_initialized_ = false;
  p_hat_.setZero();
  tau_dist_est_.setZero();
}

void MomentumObserver::Compute(
    const Eigen::Ref<const Eigen::MatrixXd>& M,
    const Eigen::Ref<const Eigen::VectorXd>& coriolis,
    const Eigen::Ref<const Eigen::VectorXd>& gravity,
    const Eigen::Ref<const Eigen::VectorXd>& qdot,
    const Eigen::Ref<const Eigen::VectorXd>& tau_prev,
    double dt,
    Eigen::Ref<Eigen::VectorXd> out) {
  // Current generalized momentum.
  const Eigen::VectorXd p_curr = M * qdot;

  // First-tick initialization: synchronize integrator to current momentum.
  if (!is_initialized_) {
    p_hat_ = p_curr;
    is_initialized_ = true;
  }

  // Disturbance estimate from residual.
  tau_dist_est_ = K_o_.cwiseProduct(p_curr - p_hat_);

  // Clamp disturbance estimate for safety.
  tau_dist_est_ = tau_dist_est_.cwiseMax(-max_tau_dist_).cwiseMin(max_tau_dist_);

  // Momentum observer update (De Luca 2003 approximation).
  p_hat_.noalias() += (tau_prev + coriolis - gravity + tau_dist_est_) * dt;

  out = tau_dist_est_;
}

}  // namespace wbc
