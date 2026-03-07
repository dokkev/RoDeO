/**
 * @file residual_compensator/include/residual_compensator/adaptive_friction_compensator.hpp
 * @brief Adaptive friction compensator: online estimation of Coulomb + viscous friction.
 */
#pragma once

#include <Eigen/Dense>

namespace wbc {

struct FrictionCompensatorConfig {
  bool enabled{false};

  // Adaptation rates (per-joint or scalar broadcast to all joints).
  Eigen::VectorXd gamma_c{Eigen::VectorXd::Zero(1)};  // Coulomb learning rate
  Eigen::VectorXd gamma_v{Eigen::VectorXd::Zero(1)};  // Viscous learning rate

  // Safety projection limits — estimates clamped to [0, max].
  Eigen::VectorXd max_f_c{Eigen::VectorXd::Constant(1, 10.0)};  // [Nm]
  Eigen::VectorXd max_f_v{Eigen::VectorXd::Constant(1, 5.0)};   // [Nm*s/rad]
};

class AdaptiveFrictionCompensator {
public:
  AdaptiveFrictionCompensator() = default;

  void Setup(int n);

  void SetGains(const Eigen::Ref<const Eigen::VectorXd>& gamma_c,
                const Eigen::Ref<const Eigen::VectorXd>& gamma_v);

  void SetLimits(const Eigen::Ref<const Eigen::VectorXd>& max_f_c,
                 const Eigen::Ref<const Eigen::VectorXd>& max_f_v);

  void Reset();

  bool IsSetup() const { return n_ > 0; }

  void Compute(const Eigen::Ref<const Eigen::VectorXd>& qdot_des,
               const Eigen::Ref<const Eigen::VectorXd>& qdot,
               double dt,
               Eigen::Ref<Eigen::VectorXd> out);

  const Eigen::VectorXd& GetCoulombEstimate() const { return f_c_est_; }
  const Eigen::VectorXd& GetViscousEstimate() const { return f_v_est_; }

private:
  static double DeadbandSign(double x, double threshold = 1e-3);

  int n_{0};
  Eigen::VectorXd f_c_est_;  // Coulomb friction estimate [Nm]
  Eigen::VectorXd f_v_est_;  // Viscous friction estimate [Nm*s/rad]
  Eigen::VectorXd gamma_c_;  // Coulomb learning rate
  Eigen::VectorXd gamma_v_;  // Viscous learning rate
  Eigen::VectorXd max_f_c_;  // Safety bound for Coulomb
  Eigen::VectorXd max_f_v_;  // Safety bound for viscous
};

}  // namespace wbc
