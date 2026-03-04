/**
 * @file wbc_core/wbc_formulation/include/wbc_formulation/kinematic_constraint.hpp
 * @brief Doxygen documentation for kinematic_constraint module.
 */
#pragma once

#include "wbc_formulation/constraint.hpp"
#include <string>

namespace wbc {

/**
 * @brief Joint Position Limit: q_min <= q + qdot*dt + 0.5*qddot*dt^2 <= q_max
 *
 * By default, limits come from the URDF (via PinocchioRobotSystem).
 * Call SetCustomLimits() to override with operational limits from YAML.
 */
class JointPosLimitConstraint : public Constraint {
public:
  explicit JointPosLimitConstraint(PinocchioRobotSystem* robot, double dt = 0.001);
  ~JointPosLimitConstraint() override = default;

  void UpdateJacobian() override;
  void UpdateJacobianDotQdot() override;
  void UpdateConstraint() override;

  /// Override URDF limits with custom operational limits (Nx2: [min, max]).
  void SetCustomLimits(const Eigen::MatrixXd& limits);

  /// Returns Nx2 matrix [min, max] — custom if set, otherwise URDF.
  const Eigen::MatrixXd& EffectiveLimits() const;

private:
  double dt_;
  Eigen::MatrixXd custom_limits_;
  bool use_custom_limits_{false};
};

/**
 * @brief Joint Velocity Limit: qdot_min <= qdot + qddot*dt <= qdot_max
 *
 * By default, limits come from the URDF (via PinocchioRobotSystem).
 * Call SetCustomLimits() to override with operational limits from YAML.
 */
class JointVelLimitConstraint : public Constraint {
public:
  explicit JointVelLimitConstraint(PinocchioRobotSystem* robot, double dt = 0.001);
  ~JointVelLimitConstraint() override = default;

  void UpdateJacobian() override;
  void UpdateJacobianDotQdot() override;
  void UpdateConstraint() override;

  /// Override URDF limits with custom operational limits (Nx2: [min, max]).
  void SetCustomLimits(const Eigen::MatrixXd& limits);

  /// Returns Nx2 matrix [min, max] — custom if set, otherwise URDF.
  const Eigen::MatrixXd& EffectiveLimits() const;

private:
  double dt_;
  Eigen::MatrixXd custom_limits_;
  bool use_custom_limits_{false};
};

/**
 * @brief Joint Torque Limit: tau_min <= tau <= tau_max
 * Note: Primarily used in the final QP stage of WBIC.
 *
 * By default, limits come from the URDF (via PinocchioRobotSystem).
 * Call SetCustomLimits() to override with operational limits from YAML.
 */
class JointTrqLimitConstraint : public Constraint {
public:
  explicit JointTrqLimitConstraint(PinocchioRobotSystem* robot);
  ~JointTrqLimitConstraint() override = default;

  void UpdateJacobian() override;
  void UpdateJacobianDotQdot() override;
  void UpdateConstraint() override;

  /// Override URDF limits with custom operational limits (Nx2: [min, max]).
  void SetCustomLimits(const Eigen::MatrixXd& limits);

  /// Returns Nx2 matrix [min, max] — custom if set, otherwise URDF.
  const Eigen::MatrixXd& EffectiveLimits() const;

private:
  Eigen::MatrixXd custom_limits_;
  bool use_custom_limits_{false};
};

} // namespace wbc
