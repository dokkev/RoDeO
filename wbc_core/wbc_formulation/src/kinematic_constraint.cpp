/**
 * @file wbc_core/wbc_formulation/src/kinematic_constraint.cpp
 * @brief Doxygen documentation for kinematic_constraint module.
 */
#include "wbc_formulation/kinematic_constraint.hpp"
#include "wbc_robot_system/pinocchio_robot_system.hpp"

namespace wbc {

// =============================================================================
// Joint Position Limit
// =============================================================================
JointPosLimitConstraint::JointPosLimitConstraint(PinocchioRobotSystem* robot, double dt)
    : Constraint(robot, robot->NumActiveDof(), -1), dt_(dt) {
  constraint_matrix_.resize(dim_ * 2, robot->NumQdot());
  constraint_vector_.resize(dim_ * 2);
}

void JointPosLimitConstraint::SetCustomLimits(const Eigen::MatrixXd& limits) {
  custom_limits_ = limits;
  use_custom_limits_ = true;
}

const Eigen::MatrixXd& JointPosLimitConstraint::EffectiveLimits() const {
  return use_custom_limits_ ? custom_limits_ : robot_->JointPosLimits();
}

void JointPosLimitConstraint::UpdateJacobian() { /* Not used for ineq */ }
void JointPosLimitConstraint::UpdateJacobianDotQdot() { /* Not used for ineq */ }

void JointPosLimitConstraint::UpdateConstraint() {
  const Eigen::VectorXd& q_ref    = robot_->GetQRef();
  const Eigen::VectorXd& qdot_ref = robot_->GetQdotRef();
  const Eigen::MatrixXd& limits =
      use_custom_limits_ ? custom_limits_ : robot_->JointPosLimits();
  const int nf = robot_->NumFloatDof();

  // Soft-bound via look-ahead time constants (replaces 2/dt² Taylor expansion).
  // Position margin: derive a "safe velocity" from distance to limit.
  // Velocity damper: derive a smooth acceleration to track the safe velocity.
  // This produces O(100) bounds instead of O(1e6) from 2/dt², preventing
  // crossed bounds with velocity constraints in the QP solver.
  constexpr double kTauPos = 0.05;  // position look-ahead [s]
  constexpr double kTauVel = 0.02;  // velocity damping [s]

  constraint_matrix_.setZero();
  for (int i = 0; i < dim_; ++i) {
    const int idx       = nf + i;
    const double q_i    = q_ref[robot_->GetQIdx(i)];
    const double qdot_i = qdot_ref[idx];

    // Step 1: safe velocity = (q_lim - q) / tau_p
    const double qdot_safe_min = (limits(i, 0) - q_i) / kTauPos;
    const double qdot_safe_max = (limits(i, 1) - q_i) / kTauPos;

    // Step 2: damped acceleration = (qdot_safe - qdot) / tau_v
    const double qddot_min = (qdot_safe_min - qdot_i) / kTauVel;
    const double qddot_max = (qdot_safe_max - qdot_i) / kTauVel;

    // Ax <= b form: -qddot <= -qddot_min  and  qddot <= qddot_max
    constraint_matrix_(i, idx) = -1.0;
    constraint_vector_(i) = -qddot_min;

    constraint_matrix_(i + dim_, idx) = 1.0;
    constraint_vector_(i + dim_) = qddot_max;
  }
}

// =============================================================================
// Joint Velocity Limit
// =============================================================================
JointVelLimitConstraint::JointVelLimitConstraint(PinocchioRobotSystem* robot, double dt)
    : Constraint(robot, robot->NumActiveDof(), -1), dt_(dt) {
  constraint_matrix_.resize(dim_ * 2, robot->NumQdot());
  constraint_vector_.resize(dim_ * 2);
}

void JointVelLimitConstraint::SetCustomLimits(const Eigen::MatrixXd& limits) {
  custom_limits_ = limits;
  use_custom_limits_ = true;
}

const Eigen::MatrixXd& JointVelLimitConstraint::EffectiveLimits() const {
  return use_custom_limits_ ? custom_limits_ : robot_->JointVelLimits();
}

void JointVelLimitConstraint::UpdateJacobian() { /* Not used */ }
void JointVelLimitConstraint::UpdateJacobianDotQdot() { /* Not used */ }

void JointVelLimitConstraint::UpdateConstraint() {
  const Eigen::VectorXd& qdot_ref = robot_->GetQdotRef();
  const Eigen::MatrixXd& limits =
      use_custom_limits_ ? custom_limits_ : robot_->JointVelLimits();
  const int nf = robot_->NumFloatDof();

  // Soft damping: approach the velocity limit over tau_v instead of
  // demanding instant compliance in one dt step.
  constexpr double kTauVel = 0.02;  // velocity damping [s]

  constraint_matrix_.setZero();
  for (int i = 0; i < dim_; ++i) {
    const int    idx    = nf + i;
    const double qdot_i = qdot_ref[idx];

    // qddot bounds: (qdot_lim - qdot) / tau_v
    const double qddot_min = (limits(i, 0) - qdot_i) / kTauVel;
    const double qddot_max = (limits(i, 1) - qdot_i) / kTauVel;

    constraint_matrix_(i, idx) = -1.0;     // -qddot <= -qddot_min
    constraint_vector_(i) = -qddot_min;

    constraint_matrix_(i + dim_, idx) = 1.0;  // qddot <= qddot_max
    constraint_vector_(i + dim_) = qddot_max;
  }
}

// =============================================================================
// Joint Torque Limit
// =============================================================================
JointTrqLimitConstraint::JointTrqLimitConstraint(PinocchioRobotSystem* robot)
    : Constraint(robot, robot->NumActiveDof(), -1) {
  // Torque is f(qddot, f_contact), so this enters as a constraint on QP decision variables.
  // constraint_matrix_ is intentionally not populated here; it is handled directly by the
  // solver which computes tau = M*qddot + cori + grav - Jc'*f and enforces tau limits.
  constraint_matrix_.resize(dim_ * 2, dim_);
  constraint_matrix_.setZero();
  constraint_vector_.resize(dim_ * 2);
}

void JointTrqLimitConstraint::SetCustomLimits(const Eigen::MatrixXd& limits) {
  custom_limits_ = limits;
  use_custom_limits_ = true;
}

const Eigen::MatrixXd& JointTrqLimitConstraint::EffectiveLimits() const {
  return use_custom_limits_ ? custom_limits_ : robot_->JointTrqLimits();
}

void JointTrqLimitConstraint::UpdateJacobian() { /* Not used */ }
void JointTrqLimitConstraint::UpdateJacobianDotQdot() { /* Not used */ }

void JointTrqLimitConstraint::UpdateConstraint() {
  const Eigen::MatrixXd& limits =
      use_custom_limits_ ? custom_limits_ : robot_->JointTrqLimits();
  for (int i = 0; i < dim_; ++i) {
    constraint_vector_(i) = -limits(i, 0); // -tau <= -tau_min
    constraint_vector_(i + dim_) = limits(i, 1); // tau <= tau_max
  }
}

} // namespace wbc
