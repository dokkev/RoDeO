/**
 * @file residual_compensator/src/momentum_observer.cpp
 * @brief Momentum observer implementation.
 */
#include "residual_compensator/momentum_observer.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>

namespace wbc {
namespace {

void ClampEach(Eigen::VectorXd& x, const Eigen::VectorXd& lim) {
  for (int i = 0; i < x.size(); ++i) {
    x[i] = std::max(-lim[i], std::min(x[i], lim[i]));
  }
}

// First-order IIR alpha from cutoff frequency.
double FirstOrderAlpha(double hz, double dt) {
  if (hz <= 0.0) return 1.0;
  return 1.0 - std::exp(-2.0 * M_PI * hz * dt);
}

}  // namespace

void MomentumObserver::Setup(int n) {
  assert(n > 0);
  n_ = n;

  z_.setZero(n);
  p_filt_.setZero(n);
  tau_uncertainty_est_.setZero(n);
  tau_bias_est_.setZero(n);
  tau_fast_residual_est_.setZero(n);

  K_o_.setConstant(n, 30.0);
  max_tau_uncertainty_.setConstant(n, 50.0);

  momentum_lpf_hz_ = 0.0;
  bias_lpf_hz_ = 0.0;
  is_initialized_ = false;
}

void MomentumObserver::SetGain(const Eigen::Ref<const Eigen::VectorXd>& K_o) {
  assert(K_o.size() == n_);
  K_o_ = K_o;
}

void MomentumObserver::SetLimit(
    const Eigen::Ref<const Eigen::VectorXd>& max_tau_uncertainty) {
  assert(max_tau_uncertainty.size() == n_);
  max_tau_uncertainty_ = max_tau_uncertainty;
}

void MomentumObserver::Reset() {
  is_initialized_ = false;
  z_.setZero();
  p_filt_.setZero();
  tau_uncertainty_est_.setZero();
  tau_bias_est_.setZero();
  tau_fast_residual_est_.setZero();
}

void MomentumObserver::Compute(
    const Eigen::Ref<const Eigen::VectorXd>& p_nom,
    const Eigen::Ref<const Eigen::VectorXd>& beta_nom,
    const Eigen::Ref<const Eigen::VectorXd>& tau_applied,
    double dt,
    Eigen::Ref<Eigen::VectorXd> out_uncertainty) {
  assert(n_ > 0);
  assert(p_nom.size() == n_);
  assert(beta_nom.size() == n_);
  assert(tau_applied.size() == n_);
  assert(out_uncertainty.size() == n_);
  assert(dt > 0.0);

  if (!is_initialized_) {
    z_      = p_nom;
    p_filt_ = p_nom;
    is_initialized_ = true;
  }

  // Optional low-pass filter on p_nom to suppress velocity noise.
  const double alpha_p = FirstOrderAlpha(momentum_lpf_hz_, dt);
  p_filt_ *= (1.0 - alpha_p);
  p_filt_ += alpha_p * p_nom;

  // Lumped uncertainty estimate from momentum residual.
  tau_uncertainty_est_ = K_o_.cwiseProduct(p_filt_ - z_);
  ClampEach(tau_uncertainty_est_, max_tau_uncertainty_);

  // Observer internal-state update — three separate in-place adds (RT-safe, no temporaries).
  z_ += tau_applied * dt;
  z_ += beta_nom * dt;
  z_ += tau_uncertainty_est_ * dt;

  // Optional slow bias / fast residual split for targeted compensation.
  if (bias_lpf_hz_ > 0.0) {
    const double alpha_b = FirstOrderAlpha(bias_lpf_hz_, dt);
    tau_bias_est_ *= (1.0 - alpha_b);
    tau_bias_est_ += alpha_b * tau_uncertainty_est_;
    tau_fast_residual_est_  = tau_uncertainty_est_;
    tau_fast_residual_est_ -= tau_bias_est_;
  } else {
    tau_bias_est_.setZero();
    tau_fast_residual_est_ = tau_uncertainty_est_;
  }

  out_uncertainty = tau_uncertainty_est_;
}

}  // namespace wbc
