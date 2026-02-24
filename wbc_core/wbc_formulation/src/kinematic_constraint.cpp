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

void JointPosLimitConstraint::UpdateJacobian() { /* Not used for ineq */ }
void JointPosLimitConstraint::UpdateJacobianDotQdot() { /* Not used for ineq */ }

void JointPosLimitConstraint::UpdateConstraint() {
  const Eigen::VectorXd q = robot_->GetJointPos();
  const Eigen::VectorXd qdot = robot_->GetJointVel();
  const Eigen::MatrixXd limits = robot_->JointPosLimits(); // [min, max]

  constraint_matrix_.setZero();
  for (int i = 0; i < dim_; ++i) {
    int idx = robot_->NumFloatDof() + i;
    
    // Taylor Expansion: q_next = q + qdot*dt + 0.5*qddot*dt^2
    // qddot_max = (2/dt^2) * (q_max - q - qdot*dt)
    double qddot_min = (2.0 / (dt_ * dt_)) * (limits(i, 0) - q[i] - qdot[i] * dt_);
    double qddot_max = (2.0 / (dt_ * dt_)) * (limits(i, 1) - q[i] - qdot[i] * dt_);

    // -I * qddot <= -qddot_min
    constraint_matrix_(i, idx) = -1.0;
    constraint_vector_(i) = -qddot_min;

    // I * qddot <= qddot_max
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

void JointVelLimitConstraint::UpdateJacobian() { /* Not used */ }
void JointVelLimitConstraint::UpdateJacobianDotQdot() { /* Not used */ }

void JointVelLimitConstraint::UpdateConstraint() {
  const Eigen::VectorXd qdot = robot_->GetJointVel();
  const Eigen::MatrixXd limits = robot_->JointVelLimits();

  constraint_matrix_.setZero();
  for (int i = 0; i < dim_; ++i) {
    int idx = robot_->NumFloatDof() + i;
    
    // qdot_next = qdot + qddot*dt
    // qddot_max = (qdot_max - qdot) / dt
    double qddot_min = (limits(i, 0) - qdot[i]) / dt_;
    double qddot_max = (limits(i, 1) - qdot[i]) / dt_;

    constraint_matrix_(i, idx) = -1.0;
    constraint_vector_(i) = -qddot_min;

    constraint_matrix_(i + dim_, idx) = 1.0;
    constraint_vector_(i + dim_) = qddot_max;
  }
}

// =============================================================================
// Joint Torque Limit
// =============================================================================
JointTrqLimitConstraint::JointTrqLimitConstraint(PinocchioRobotSystem* robot)
    : Constraint(robot, robot->NumActiveDof(), -1) {
  // 토크는 f(ddot{q}, f_contact) 이므로 QP의 결정 변수들에 대한 제약으로 들어감
  // 여기서는 단순하게 b 벡터(Limit 값)만 업데이트해둠
  constraint_matrix_.resize(dim_ * 2, dim_); 
  constraint_vector_.resize(dim_ * 2);
}

void JointTrqLimitConstraint::UpdateJacobian() { /* Not used */ }
void JointTrqLimitConstraint::UpdateJacobianDotQdot() { /* Not used */ }

void JointTrqLimitConstraint::UpdateConstraint() {
  const Eigen::MatrixXd limits = robot_->JointTrqLimits();
  for (int i = 0; i < dim_; ++i) {
    constraint_vector_(i) = -limits(i, 0); // -tau <= -tau_min
    constraint_vector_(i + dim_) = limits(i, 1); // tau <= tau_max
  }
}

} // namespace wbc
