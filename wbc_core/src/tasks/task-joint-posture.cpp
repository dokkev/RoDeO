//
// Copyright (c) 2017 CNRS
//

#include <wbc_core/tasks/task-joint-posture.hpp>
#include "wbc_core/math/linear_algebra/selection.hpp"
#include "wbc_core/robots/robot-system.hpp"
#include <pinocchio/algorithm/joint-configuration.hpp>

namespace wbc {
namespace tasks {
using namespace math;
using namespace trajectories;
using namespace pinocchio;

TaskJointPosture::TaskJointPosture(const std::string& name, RobotSystem& robot)
    : TaskMotion(name, robot),
      m_ref(robot.nq_joints(), robot.nv_joints()),
      m_constraint(name, robot.nv_joints(), robot.nv()) {
  m_ref_q_augmented = pinocchio::neutral(robot.model());
  m_Kp.setZero(robot.nv_joints());
  m_Kd.setZero(robot.nv_joints());
  Vector m = Vector::Ones(robot.nv_joints());
  setMask(m);
}

void TaskJointPosture::setMask(ConstRefVector m) {
  PINOCCHIO_CHECK_INPUT_ARGUMENT(
      m.size() == m_robot.nv_joints(),
      "The size of the mask needs to equal " +
          std::to_string(m_robot.nv_joints()));
  m_mask = m;
  Matrix S;
  buildSelectionMatrix(m, m_robot.nv(), m_robot.nv() - m_robot.nv_joints(),
                       m_activeAxes, S);
  const Vector::Index dim = S.rows();
  m_constraint.resize((unsigned int)dim, m_robot.nv());
  m_constraint.setMatrix(S);
}

int TaskJointPosture::dim() const { return (int)m_mask.sum(); }

const Vector& TaskJointPosture::Kp() { return m_Kp; }

const Vector& TaskJointPosture::Kd() { return m_Kd; }

void TaskJointPosture::Kp(ConstRefVector Kp) {
  PINOCCHIO_CHECK_INPUT_ARGUMENT(Kp.size() == m_robot.nv_joints(),
                                 "The size of the Kp vector needs to equal " +
                                     std::to_string(m_robot.nv_joints()));
  m_Kp = Kp;
}

void TaskJointPosture::Kd(ConstRefVector Kd) {
  PINOCCHIO_CHECK_INPUT_ARGUMENT(Kd.size() == m_robot.nv_joints(),
                                 "The size of the Kd vector needs to equal " +
                                     std::to_string(m_robot.nv_joints()));
  m_Kd = Kd;
}

void TaskJointPosture::setReference(const TrajectorySample& ref) {
  PINOCCHIO_CHECK_INPUT_ARGUMENT(
      ref.getValue().size() == m_robot.nq_joints(),
      "The size of the reference value vector needs to equal " +
          std::to_string(m_robot.nq_joints()));
  PINOCCHIO_CHECK_INPUT_ARGUMENT(
      ref.getDerivative().size() == m_robot.nv_joints(),
      "The size of the reference value derivative vector needs to equal " +
          std::to_string(m_robot.nv_joints()));
  PINOCCHIO_CHECK_INPUT_ARGUMENT(
      ref.getSecondDerivative().size() == m_robot.nv_joints(),
      "The size of the reference value second derivative vector needs to "
      "equal " +
          std::to_string(m_robot.nv_joints()));
  m_ref = ref;
}

const TrajectorySample& TaskJointPosture::getReference() const { return m_ref; }

const Vector& TaskJointPosture::getDesiredAcceleration() const {
  return m_a_des;
}

Vector TaskJointPosture::getAcceleration(ConstRefVector dv) const {
  return m_constraint.matrix() * dv;
}

const Vector& TaskJointPosture::position_error() const { return m_p_error; }

const Vector& TaskJointPosture::velocity_error() const { return m_v_error; }

const Vector& TaskJointPosture::position() const { return m_p; }

const Vector& TaskJointPosture::velocity() const { return m_v; }

const Vector& TaskJointPosture::position_ref() const {
  return m_ref.getValue();
}

const Vector& TaskJointPosture::velocity_ref() const {
  return m_ref.getDerivative();
}

const ConstraintBase& TaskJointPosture::getConstraint() const {
  return m_constraint;
}

const ConstraintBase& TaskJointPosture::compute(const double, ConstRefVector q,
                                                ConstRefVector v, Data&) {
  m_ref_q_augmented.tail(m_robot.nq_joints()) = m_ref.getValue();

  // Compute errors
  m_p_error = pinocchio::difference(m_robot.model(), m_ref_q_augmented, q)
                  .tail(m_robot.nv_joints());

  m_v = v.tail(m_robot.nv_joints());
  m_v_error = m_v - m_ref.getDerivative();
  m_a_des = -m_Kp.cwiseProduct(m_p_error) - m_Kd.cwiseProduct(m_v_error) +
            m_ref.getSecondDerivative();

  for (unsigned int i = 0; i < m_activeAxes.size(); i++)
    m_constraint.vector()(i) = m_a_des(m_activeAxes(i));
  return m_constraint;
}

}  // namespace tasks
}  // namespace wbc
