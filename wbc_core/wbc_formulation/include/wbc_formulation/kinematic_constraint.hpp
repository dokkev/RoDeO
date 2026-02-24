#pragma once

#include "wbc_formulation/constraint.hpp"
#include <string>

namespace wbc {

/**
 * @brief Joint Position Limit: q_min <= q + qdot*dt + 0.5*qddot*dt^2 <= q_max
 */
class JointPosLimitConstraint : public Constraint {
public:
  explicit JointPosLimitConstraint(PinocchioRobotSystem* robot, double dt = 0.001);
  ~JointPosLimitConstraint() override = default;

  void UpdateJacobian() override;
  void UpdateJacobianDotQdot() override;
  void UpdateConstraint() override;

private:
  double dt_;
};

/**
 * @brief Joint Velocity Limit: qdot_min <= qdot + qddot*dt <= qdot_max
 */
class JointVelLimitConstraint : public Constraint {
public:
  explicit JointVelLimitConstraint(PinocchioRobotSystem* robot, double dt = 0.001);
  ~JointVelLimitConstraint() override = default;

  void UpdateJacobian() override;
  void UpdateJacobianDotQdot() override;
  void UpdateConstraint() override;

private:
  double dt_;
};

/**
 * @brief Joint Torque Limit: tau_min <= tau <= tau_max
 * Note: WBIC의 최종 QP 단계에서 주로 사용됩니다.
 */
class JointTrqLimitConstraint : public Constraint {
public:
  explicit JointTrqLimitConstraint(PinocchioRobotSystem* robot);
  ~JointTrqLimitConstraint() override = default;

  void UpdateJacobian() override;
  void UpdateJacobianDotQdot() override;
  void UpdateConstraint() override;
};

} // namespace wbc