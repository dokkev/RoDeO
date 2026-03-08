/**
 * @file residual_compensator/include/residual_compensator/momentum_observer.hpp
 * @brief Momentum-based lumped uncertainty observer for joint-space.
 *
 * Observer model (De Luca 2003):
 *   p     = M(q)*qdot                  (generalized momentum, computed by caller)
 *   p_dot = tau + beta(q,qdot) + w     (w = lumped uncertainty)
 *   ẑ    = tau + beta + ŵ              (observer state tracks predicted momentum)
 *   ŵ    = K_o * (p - z)              (residual drives uncertainty estimate)
 *
 * The caller is responsible for computing:
 *   p_nom    = M_active * qdot_active
 *   beta_nom = C_active*qdot_active - g_active
 *
 * This keeps the observer convention-independent — the robot model wrapper
 * owns the dynamics convention, the observer only integrates residuals.
 *
 * Optional features:
 *   momentum_lpf_hz  — low-pass on p_nom to suppress velocity noise
 *   bias_lpf_hz      — slow/fast splitter: tau_bias (slow model error) and
 *                      tau_fast_residual (contact-like transients)
 */
#pragma once

#include <Eigen/Dense>

namespace wbc {

struct MomentumObserverConfig {
  bool enabled{false};

  /// Diagonal observer gain (scalar broadcast or per-joint vector).
  Eigen::VectorXd K_o{Eigen::VectorXd::Constant(1, 30.0)};

  /// Safety clamp on lumped uncertainty estimate [Nm].
  Eigen::VectorXd max_tau_uncertainty{Eigen::VectorXd::Constant(1, 50.0)};

  /// Optional low-pass cutoff on nominal momentum p_nom [Hz] (0 = disabled).
  double momentum_lpf_hz{0.0};

  /// Optional slow bias splitter cutoff [Hz] (0 = disabled).
  double bias_lpf_hz{0.0};
};

class MomentumObserver {
 public:
  void Setup(int n);
  void SetGain(const Eigen::Ref<const Eigen::VectorXd>& K_o);
  void SetLimit(const Eigen::Ref<const Eigen::VectorXd>& max_tau_uncertainty);
  void SetMomentumLpfHz(double hz) { momentum_lpf_hz_ = hz; }
  void SetBiasLpfHz(double hz) { bias_lpf_hz_ = hz; }
  void Reset();

  bool IsSetup() const { return n_ > 0; }

  /**
   * @brief Compute lumped uncertainty estimate from generalized momentum.
   *
   * @param p_nom           Nominal generalized momentum M(q)*qdot  [n]
   * @param beta_nom        Nominal momentum rate bias C*qdot - g   [n]
   * @param tau_applied     Applied joint torque (previous tick)     [n]
   * @param dt              Control time step [s]
   * @param out_uncertainty Estimated lumped uncertainty torque      [n] (output)
   */
  void Compute(const Eigen::Ref<const Eigen::VectorXd>& p_nom,
               const Eigen::Ref<const Eigen::VectorXd>& beta_nom,
               const Eigen::Ref<const Eigen::VectorXd>& tau_applied,
               double dt,
               Eigen::Ref<Eigen::VectorXd> out_uncertainty);

  const Eigen::VectorXd& GetUncertaintyEstimate()  const { return tau_uncertainty_est_; }
  const Eigen::VectorXd& GetSlowBiasEstimate()     const { return tau_bias_est_; }
  const Eigen::VectorXd& GetFastResidualEstimate() const { return tau_fast_residual_est_; }

 private:
  int n_{0};
  bool is_initialized_{false};

  Eigen::VectorXd z_;                      ///< Observer internal momentum state
  Eigen::VectorXd p_filt_;                 ///< Filtered nominal momentum
  Eigen::VectorXd tau_uncertainty_est_;    ///< Total lumped uncertainty
  Eigen::VectorXd tau_bias_est_;           ///< Slow model-bias estimate
  Eigen::VectorXd tau_fast_residual_est_;  ///< Fast residual (contact-like)

  Eigen::VectorXd K_o_;
  Eigen::VectorXd max_tau_uncertainty_;

  double momentum_lpf_hz_{0.0};
  double bias_lpf_hz_{0.0};
};

}  // namespace wbc
