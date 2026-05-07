/**
 * @file wbc_core/wbc_formulation/src/constraint.cpp
 * @brief Doxygen documentation for constraint module.
 */
#include "wbc_formulation/constraint.hpp"

#include "wbc_robot_system/pinocchio_robot_system.hpp"

namespace wbc {

Constraint::Constraint(PinocchioRobotSystem* robot, int dim, int target_link_idx)
    : robot_(robot),
      dim_(dim),
      target_link_idx_(target_link_idx),
      jacobian_(Eigen::MatrixXd::Zero(dim, robot ? robot->NumQdot() : 0)),
      jacobian_dot_q_dot_(Eigen::VectorXd::Zero(dim)),
      constraint_matrix_(Eigen::MatrixXd::Zero(0, dim)),
      constraint_vector_(Eigen::VectorXd::Zero(0)) {}

}  // namespace wbc
