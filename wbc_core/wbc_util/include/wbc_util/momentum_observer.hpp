/**
 * @file wbc_util/include/wbc_util/momentum_observer.hpp
 * @brief Generalized momentum-based disturbance observer for joint-space.
 *
 * Estimates external/unmodeled disturbance torques from the residual between
 * the expected and actual generalized momentum evolution.
 *
 * Observer equation:
 *   p = M(q) * qdot                      (generalized momentum)
 *   r = K_o * (p - p_hat)                (residual = disturbance estimate)
 *   dp_hat = tau_cmd - g + C^T*qdot + r  (predicted momentum rate)
 *   p_hat += dp_hat * dt                  (integrated predicted momentum)
 *
 * K_o is a diagonal gain matrix controlling the observer bandwidth.
 * Higher K_o → faster tracking but more noise sensitivity.
 *
 * All buffers pre-allocated at Setup(); Compute() is RT-safe (no heap allocs).
 */
#pragma once

#include <Eigen/Dense>

namespace wbc {

struct MomentumObserverConfig {
  bool enabled{false};

  // Observer bandwidth gains (per-joint diagonal).
  Eigen::VectorXd K_o{Eigen::VectorXd::Constant(1, 50.0)};

  // Safety clamp on disturbance estimate [Nm].
  Eigen::VectorXd max_tau_dist{Eigen::VectorXd::Constant(1, 50.0)};
};

class MomentumObserver {
public:
  MomentumObserver() = default;

  void Setup(int n) {
    n_ = n;
    p_hat_.setZero(n);
    tau_dist_est_.setZero(n);
    K_o_.setConstant(n, 50.0);
    max_tau_dist_.setConstant(n, 50.0);
    is_initialized_ = false;
  }

  void SetGain(const Eigen::Ref<const Eigen::VectorXd>& K_o) {
    K_o_ = K_o;
  }

  void SetLimit(const Eigen::Ref<const Eigen::VectorXd>& max_tau_dist) {
    max_tau_dist_ = max_tau_dist;
  }

  void Reset() {
    is_initialized_ = false;
    p_hat_.setZero();
    tau_dist_est_.setZero();
  }

  bool IsSetup() const { return n_ > 0; }

  /**
   * @brief Compute disturbance torque estimate from momentum residual.
   *
   * @param M          Mass matrix — actuated block (n_act x n_act).
   * @param coriolis   Coriolis term C(q,qdot)*qdot — actuated part (n_act).
   * @param gravity    Gravity vector g(q) — actuated part (n_act).
   * @param qdot       Measured joint velocities (n_act).
   * @param tau_prev   Previous tick's commanded torque (n_act).
   * @param dt         Time step [s].
   * @param out        Pre-allocated output disturbance estimate (n_act).
   */
  void Compute(const Eigen::Ref<const Eigen::MatrixXd>& M,
               const Eigen::Ref<const Eigen::VectorXd>& coriolis,
               const Eigen::Ref<const Eigen::VectorXd>& gravity,
               const Eigen::Ref<const Eigen::VectorXd>& qdot,
               const Eigen::Ref<const Eigen::VectorXd>& tau_prev,
               double dt,
               Eigen::Ref<Eigen::VectorXd> out) {
    // Current generalized momentum
    Eigen::VectorXd p_curr = M * qdot;

    // First-tick initialization: synchronize integrator to current momentum
    if (!is_initialized_) {
      p_hat_ = p_curr;
      is_initialized_ = true;
    }

    // Disturbance estimate from residual
    tau_dist_est_ = K_o_.cwiseProduct(p_curr - p_hat_);

    // Clamp disturbance estimate for safety
    tau_dist_est_ = tau_dist_est_.cwiseMax(-max_tau_dist_)
                                  .cwiseMin( max_tau_dist_);

    // Momentum observer (De Luca 2003):
    //   dp/dt = tau + tau_ext + C^T*qdot - g
    //   dp_hat = tau_cmd + C*qdot - g + r    (using C*qdot ≈ C^T*qdot)
    // where coriolis = C(q,qdot)*qdot from Pinocchio (nle - gravity).
    // The C vs C^T approximation is absorbed by the observer gain K_o.
    p_hat_.noalias() += (tau_prev + coriolis - gravity + tau_dist_est_) * dt;

    out = tau_dist_est_;
  }

  const Eigen::VectorXd& GetDisturbanceEstimate() const { return tau_dist_est_; }

private:
  int n_{0};
  Eigen::VectorXd p_hat_;         // Integrated predicted momentum
  Eigen::VectorXd tau_dist_est_;  // Disturbance torque estimate
  Eigen::VectorXd K_o_;           // Observer gain (diagonal)
  Eigen::VectorXd max_tau_dist_;  // Safety clamp
  bool is_initialized_{false};
};

}  // namespace wbc
