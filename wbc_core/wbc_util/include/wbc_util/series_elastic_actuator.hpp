/**
 * @file wbc_util/include/wbc_util/series_elastic_actuator.hpp
 * @brief Controller-level SEA dynamics simulation.
 *
 * Simulates rotor inertia + torsional spring between motor and link.
 * MuJoCo models only the link side; this class sits between WBC output
 * and the effort command sent to MuJoCo.
 *
 * Flow:
 *   WBC → tau_cmd → SEA::Step(tau_cmd, q_link, dt) → tau_spring → MuJoCo
 *
 * Rotor EOM (per joint):
 *   J_eff * qddot_r = tau_cmd - k*(q_r - q_l) - d*qdot_r - f*sign(qdot_r)
 *
 * Spring torque applied to MuJoCo:
 *   tau_spring = k * (q_rotor - q_link)
 */
#pragma once

#include <cmath>

#include <Eigen/Dense>

namespace wbc {

struct SEAConfig {
  Eigen::VectorXd stiffness;      ///< k [Nm/rad] per joint
  Eigen::VectorXd rotor_inertia;  ///< N²·J_bare [kg·m²] per joint
  Eigen::VectorXd rotor_damping;  ///< Viscous friction [Nm·s/rad] per joint
  Eigen::VectorXd rotor_friction; ///< Coulomb friction [Nm] per joint
};

class SeriesElasticActuator {
public:
  /**
   * @brief Initialize SEA model. Must be called before Step().
   * All vectors must have the same size (number of actuated joints).
   */
  void Init(const SEAConfig& config) {
    n_ = static_cast<int>(config.stiffness.size());
    k_ = config.stiffness;
    J_ = config.rotor_inertia;
    d_ = config.rotor_damping;
    f_ = config.rotor_friction;
    q_rotor_.setZero(n_);
    qdot_rotor_.setZero(n_);
  }

  /**
   * @brief Set rotor positions to match link positions (zero deflection).
   * Call once after Init() with the initial link joint positions.
   */
  void SetInitialRotorPosition(const Eigen::VectorXd& q_link) {
    q_rotor_ = q_link;
    qdot_rotor_.setZero(n_);
  }

  /**
   * @brief Integrate rotor dynamics and return spring torque.
   *
   * @param tau_cmd  WBC torque command (what the motor wants to apply).
   * @param q_link   Measured link-side joint positions (from MuJoCo).
   * @param dt       Control timestep [s].
   * @return Spring torque to send to MuJoCo as effort command.
   */
  Eigen::VectorXd Step(const Eigen::VectorXd& tau_cmd,
                       const Eigen::VectorXd& q_link, double dt) {
    for (int i = 0; i < n_; ++i) {
      const double delta = q_rotor_[i] - q_link[i];
      const double tau_spring = k_[i] * delta;

      // Coulomb friction with zero-velocity deadband
      const double friction =
          (std::abs(qdot_rotor_[i]) > 1e-6)
              ? f_[i] * ((qdot_rotor_[i] > 0.0) ? 1.0 : -1.0)
              : 0.0;

      // Rotor acceleration
      const double qddot =
          (tau_cmd[i] - tau_spring - d_[i] * qdot_rotor_[i] - friction) /
          J_[i];

      // Semi-implicit Euler (velocity first, then position)
      qdot_rotor_[i] += qddot * dt;
      q_rotor_[i] += qdot_rotor_[i] * dt;
    }

    // Spring torque with updated rotor position
    return k_.cwiseProduct(q_rotor_ - q_link);
  }

  int NumJoints() const { return n_; }
  const Eigen::VectorXd& RotorPosition() const { return q_rotor_; }
  const Eigen::VectorXd& RotorVelocity() const { return qdot_rotor_; }

private:
  int n_{0};
  Eigen::VectorXd k_;           ///< Stiffness
  Eigen::VectorXd J_;           ///< Effective rotor inertia
  Eigen::VectorXd d_;           ///< Rotor damping
  Eigen::VectorXd f_;           ///< Rotor Coulomb friction
  Eigen::VectorXd q_rotor_;     ///< Rotor position state
  Eigen::VectorXd qdot_rotor_;  ///< Rotor velocity state
};

} // namespace wbc
