#pragma once

#include <Eigen/Dense>

namespace wbc {
class PinocchioRobotSystem;

/**
 * @brief Abstract constraint interface for WBC solvers.
 *
 * A constraint provides:
 * - acceleration-level kinematics (`J`, `Jdot*qdot`)
 * - linear inequality/equality representation (`A`, `b`)
 */
class Constraint {
public:
  Constraint(PinocchioRobotSystem* robot, int dim, int target_link_idx);
  virtual ~Constraint() = default;

  virtual void UpdateJacobian() = 0;
  virtual void UpdateJacobianDotQdot() = 0;
  virtual void UpdateConstraint() = 0;

  int Dim() const { return dim_; }
  int TargetLinkIdx() const { return target_link_idx_; }

  Eigen::MatrixXd Jacobian() const { return jacobian_; }
  Eigen::VectorXd JacobianDotQdot() const { return jacobian_dot_q_dot_; }
  Eigen::MatrixXd ConstraintMatrix() const { return constraint_matrix_; }
  Eigen::VectorXd ConstraintVector() const { return constraint_vector_; }

protected:
  PinocchioRobotSystem* robot_;
  int dim_;
  int target_link_idx_;

  Eigen::MatrixXd jacobian_;
  Eigen::VectorXd jacobian_dot_q_dot_;
  Eigen::MatrixXd constraint_matrix_;
  Eigen::VectorXd constraint_vector_;
};

}  // namespace wbc
