/**
 * @file residual_compensator/include/residual_compensator/momentum_observer.hpp
 * @brief Generalized momentum-based disturbance observer for joint-space.
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

  void Setup(int n);
  void SetGain(const Eigen::Ref<const Eigen::VectorXd>& K_o);
  void SetLimit(const Eigen::Ref<const Eigen::VectorXd>& max_tau_dist);
  void Reset();

  bool IsSetup() const { return n_ > 0; }

  void Compute(const Eigen::Ref<const Eigen::MatrixXd>& M,
               const Eigen::Ref<const Eigen::VectorXd>& coriolis,
               const Eigen::Ref<const Eigen::VectorXd>& gravity,
               const Eigen::Ref<const Eigen::VectorXd>& qdot,
               const Eigen::Ref<const Eigen::VectorXd>& tau_prev,
               double dt,
               Eigen::Ref<Eigen::VectorXd> out);

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
