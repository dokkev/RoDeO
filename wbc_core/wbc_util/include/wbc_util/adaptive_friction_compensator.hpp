/**
 * @file wbc_util/include/wbc_util/adaptive_friction_compensator.hpp
 * @brief Adaptive friction compensator: online estimation of Coulomb + viscous friction.
 *
 * Runs alongside WBC feedforward to cancel unmodeled joint friction that varies
 * per robot unit. All buffers pre-allocated at Setup(); Compute() is RT-safe.
 *
 * Model:  tau_fric = f_c * sign(qdot) + f_v * qdot
 *
 * Adaptation law (gradient descent on velocity error):
 *   f_c += gamma_c * e_v * sign(qdot) * dt
 *   f_v += gamma_v * e_v * qdot * dt
 *
 * where e_v = qdot_des - qdot (velocity tracking error).
 */
#pragma once

#include <algorithm>
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

  void Setup(int n) {
    n_ = n;
    f_c_est_.setZero(n);
    f_v_est_.setZero(n);
    gamma_c_.setZero(n);
    gamma_v_.setZero(n);
    max_f_c_.setConstant(n, 10.0);
    max_f_v_.setConstant(n, 5.0);
    tau_comp_.setZero(n);
  }

  void SetGains(const Eigen::Ref<const Eigen::VectorXd>& gamma_c,
                const Eigen::Ref<const Eigen::VectorXd>& gamma_v) {
    gamma_c_ = gamma_c;
    gamma_v_ = gamma_v;
  }

  void SetLimits(const Eigen::Ref<const Eigen::VectorXd>& max_f_c,
                 const Eigen::Ref<const Eigen::VectorXd>& max_f_v) {
    max_f_c_ = max_f_c;
    max_f_v_ = max_f_v;
  }

  void Reset() {
    f_c_est_.setZero();
    f_v_est_.setZero();
  }

  bool IsSetup() const { return n_ > 0; }

  /**
   * @brief Update friction estimates and compute compensation torque.
   *
   * @param qdot_des Desired joint velocities (n).
   * @param qdot     Measured joint velocities (n).
   * @param dt       Time step [s].
   * @param out      Pre-allocated output torque (n) — written in-place.
   */
  void Compute(const Eigen::Ref<const Eigen::VectorXd>& qdot_des,
               const Eigen::Ref<const Eigen::VectorXd>& qdot,
               double dt,
               Eigen::Ref<Eigen::VectorXd> out) {
    for (int i = 0; i < n_; ++i) {
      const double e_v = qdot_des[i] - qdot[i];
      const double sgn = DeadbandSign(qdot[i]);

      // Gradient update
      f_c_est_[i] += gamma_c_[i] * e_v * sgn * dt;
      f_v_est_[i] += gamma_v_[i] * e_v * qdot[i] * dt;

      // Projection to feasible set [0, max]
      f_c_est_[i] = std::clamp(f_c_est_[i], 0.0, max_f_c_[i]);
      f_v_est_[i] = std::clamp(f_v_est_[i], 0.0, max_f_v_[i]);

      // Compensation torque (cancel friction → add in direction of motion)
      out[i] = f_c_est_[i] * sgn + f_v_est_[i] * qdot[i];
    }
  }

  const Eigen::VectorXd& GetCoulombEstimate() const { return f_c_est_; }
  const Eigen::VectorXd& GetViscousEstimate() const { return f_v_est_; }

private:
  static double DeadbandSign(double x, double threshold = 1e-3) {
    if (x > threshold) return 1.0;
    if (x < -threshold) return -1.0;
    return x / threshold;
  }

  int n_{0};
  Eigen::VectorXd f_c_est_;   // Coulomb friction estimate [Nm]
  Eigen::VectorXd f_v_est_;   // Viscous friction estimate [Nm*s/rad]
  Eigen::VectorXd gamma_c_;   // Coulomb learning rate
  Eigen::VectorXd gamma_v_;   // Viscous learning rate
  Eigen::VectorXd max_f_c_;   // Safety bound for Coulomb
  Eigen::VectorXd max_f_v_;   // Safety bound for viscous
  Eigen::VectorXd tau_comp_;  // Scratch (unused, kept for potential future use)
};

}  // namespace wbc
