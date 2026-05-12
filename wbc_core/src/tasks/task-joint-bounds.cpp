//
// Copyright (c) 2017 CNRS
//

#include "wbc_core/robots/robot-system.hpp"
#include <wbc_core/tasks/task-joint-bounds.hpp>

namespace wbc {
namespace tasks {
using namespace math;
using namespace trajectories;
using namespace pinocchio;

TaskJointBounds::TaskJointBounds(const std::string& name, RobotSystem& robot,
                                 double dt)
    : TaskMotion(name, robot),
      m_constraint(name, robot.nv()),
      m_dt(dt),
      m_nv(robot.nv()),
      m_na(robot.na()) {
  PINOCCHIO_CHECK_INPUT_ARGUMENT(dt > 0.0, "dt needs to be positive");
  m_v_lb = -1e10 * Vector::Ones(m_na);
  m_v_ub = +1e10 * Vector::Ones(m_na);
  m_a_lb = -1e10 * Vector::Ones(m_na);
  m_a_ub = +1e10 * Vector::Ones(m_na);
  m_ddq_max_due_to_vel.setZero(m_na);
  m_ddq_max_due_to_vel.setZero(m_na);

  int offset = m_nv - m_na;
  for (int i = 0; i < offset; i++) {
    m_constraint.upperBound()(i) = 1e10;
    m_constraint.lowerBound()(i) = -1e10;
  }
}

int TaskJointBounds::dim() const { return m_nv; }

const Vector& TaskJointBounds::getAccelerationLowerBounds() const {
  return m_a_lb;
}

const Vector& TaskJointBounds::getAccelerationUpperBounds() const {
  return m_a_ub;
}

const Vector& TaskJointBounds::getVelocityLowerBounds() const { return m_v_lb; }

const Vector& TaskJointBounds::getVelocityUpperBounds() const { return m_v_ub; }

void TaskJointBounds::setTimeStep(double dt) {
  PINOCCHIO_CHECK_INPUT_ARGUMENT(dt > 0.0, "dt needs to be positive");
  m_dt = dt;
}

void TaskJointBounds::setVelocityBounds(ConstRefVector lower,
                                        ConstRefVector upper) {
  PINOCCHIO_CHECK_INPUT_ARGUMENT(
      lower.size() == m_na,
      "The size of the lower velocity bounds vector needs to equal " +
          std::to_string(m_na));
  PINOCCHIO_CHECK_INPUT_ARGUMENT(
      upper.size() == m_na,
      "The size of the upper velocity bounds vector needs to equal " +
          std::to_string(m_na));
  m_v_lb = lower;
  m_v_ub = upper;
}

void TaskJointBounds::setAccelerationBounds(ConstRefVector lower,
                                            ConstRefVector upper) {
  PINOCCHIO_CHECK_INPUT_ARGUMENT(
      lower.size() == m_na,
      "The size of the lower acceleration bounds vector needs to equal " +
          std::to_string(m_na));
  PINOCCHIO_CHECK_INPUT_ARGUMENT(
      upper.size() == m_na,
      "The size of the upper acceleration bounds vector needs to equal " +
          std::to_string(m_na));
  m_a_lb = lower;
  m_a_ub = upper;
}

const ConstraintBase& TaskJointBounds::getConstraint() const {
  return m_constraint;
}

void TaskJointBounds::setMask(ConstRefVector mask) { m_mask = mask; }

const ConstraintBase& TaskJointBounds::compute(const double, ConstRefVector,
                                               ConstRefVector v, Data&) {
  // compute min/max joint acc imposed by velocity limits
  m_ddq_max_due_to_vel = (m_v_ub - v.tail(m_na)) / m_dt;
  m_ddq_min_due_to_vel = (m_v_lb - v.tail(m_na)) / m_dt;

  // take most conservative limit between vel and acc
  int offset = m_nv - m_na;
  const bool use_mask = (m_mask.size() == m_na);
  for (int i = 0; i < m_na; i++) {
    if (use_mask && m_mask(i) == 0.0) {
      // Masked-out joint: unconstrained
      m_constraint.upperBound()(offset + i) = 1e10;
      m_constraint.lowerBound()(offset + i) = -1e10;
    } else {
      m_constraint.upperBound()(offset + i) =
          std::min(m_ddq_max_due_to_vel(i), m_a_ub(i));
      m_constraint.lowerBound()(offset + i) =
          std::max(m_ddq_min_due_to_vel(i), m_a_lb(i));
    }
  }
  return m_constraint;
}

}  // namespace tasks
}  // namespace wbc
