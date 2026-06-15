//
// Copyright (c) 2017 CNRS
//

#include "wbc_core/robots/robot-system.hpp"

#include <cmath>
#include <stdexcept>

#include <Eigen/Geometry>

#include <pinocchio/multibody/model.hpp>
#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/algorithm/center-of-mass.hpp>
#include <pinocchio/algorithm/compute-all-terms.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/centroidal.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>

namespace wbc {
namespace robots {

RobotSystem::RobotSystem(const std::string& filename,
                         const std::vector<std::string>&, bool verbose)
    : m_verbose(verbose) {
  pinocchio::urdf::buildModel(filename, m_model, m_verbose);
  m_model_filename = filename;
  m_nq_joints = m_model.nq;
  m_nv_joints = m_model.nv;
  m_na = m_nv_joints;
  m_is_fixed_base = true;
  init();
}

RobotSystem::RobotSystem(const std::string& filename,
                         const std::vector<std::string>&,
                         const pinocchio::JointModelVariant& rootJoint,
                         bool verbose)
    : m_verbose(verbose) {
  pinocchio::urdf::buildModel(filename, rootJoint, m_model, m_verbose);
  m_model_filename = filename;
  m_nq_joints = m_model.nq - 7;
  m_nv_joints = m_model.nv - 6;
  m_na = m_nv_joints;
  m_is_fixed_base = false;
  init();
}

RobotSystem::RobotSystem(const pinocchio::Model& m, RootJointType rootJoint,
                         bool verbose)
    : m_verbose(verbose) {
  m_model = m;
  m_model_filename = "";
  m_nq_joints = m_model.nq;
  m_nv_joints = m_model.nv;
  m_na = m_nv_joints;
  m_is_fixed_base = true;
  switch (rootJoint) {
    case FIXED_BASE_SYSTEM:
      break;
    case FLOATING_BASE_SYSTEM:
      m_nq_joints = m_model.nq - 7;
      m_nv_joints = m_model.nv - 6;
      m_na = m_nv_joints;
      m_is_fixed_base = false;
      break;
    default:
      break;
  }
  init();
}

void RobotSystem::init() {
  m_rotor_inertias.setZero(m_na);
  m_gear_ratios.setZero(m_na);
  m_state.joint.q = pinocchio::neutral(m_model).tail(m_nq_joints);
  m_state.joint.qdot.setZero(m_nv_joints);
  m_state.joint.tau.setZero(m_na);
  if (m_is_fixed_base) {
    m_state.base.reset();
  } else {
    m_state.base = BaseState{};
  }
  m_time = 0.0;
  m_generalized_state.q = pinocchio::neutral(m_model);
  m_generalized_state.v.setZero(m_model.nv);
  m_Md.setZero(m_na);
  m_M.setZero(m_model.nv, m_model.nv);
  m_zero_v.setZero(m_model.nv);
  m_has_state = false;
}

int RobotSystem::nq() const { return m_model.nq; }
int RobotSystem::nv() const { return m_model.nv; }
int RobotSystem::nq_joints() const { return m_nq_joints; }
int RobotSystem::nv_joints() const { return m_nv_joints; }
int RobotSystem::na() const { return m_na; }
bool RobotSystem::is_fixed_base() const { return m_is_fixed_base; }
bool RobotSystem::hasState() const { return m_has_state; }

const pinocchio::Model& RobotSystem::model() const { return m_model; }
pinocchio::Model& RobotSystem::model() { return m_model; }

void RobotSystem::updateState(const JointState& joint) {
  if (!m_is_fixed_base) {
    throw std::invalid_argument(
        "Fixed-base updateState called on a floating-base robot.");
  }
  validateJointState(joint);

  m_state.joint = joint;
  m_state.base.reset();

  m_generalized_state.q = m_state.joint.q;
  m_generalized_state.v = m_state.joint.qdot;
  m_has_state = true;
}

void RobotSystem::updateState(const JointState& joint, const BaseState& base) {
  if (m_is_fixed_base) {
    throw std::invalid_argument(
        "Floating-base updateState called on a fixed-base robot.");
  }
  validateJointState(joint);
  validateBaseState(base);

  m_state.joint = joint;
  m_state.base = base;

  m_generalized_state.q.setZero(m_model.nq);
  m_generalized_state.v.setZero(m_model.nv);

  Eigen::Quaterniond quat(base.pose_world_base.rotation());
  quat.normalize();

  m_generalized_state.q.head<3>() = base.pose_world_base.translation();
  m_generalized_state.q.segment<4>(3) << quat.x(), quat.y(), quat.z(),
      quat.w();
  m_generalized_state.q.tail(m_nq_joints) = m_state.joint.q;

  m_generalized_state.v.head<6>() = base.twist_world_base.toVector();
  m_generalized_state.v.tail(m_nv_joints) = m_state.joint.qdot;
  m_has_state = true;
}

void RobotSystem::updateState(const GeneralizedState& generalized) {
  updateState(generalized, m_state.joint.tau);
}

void RobotSystem::updateState(const GeneralizedState& generalized,
                              math::ConstRefVector tau_actuated) {
  validateGeneralizedState(generalized);
  if (tau_actuated.size() != m_na) {
    throw std::invalid_argument("The size of tau_actuated is incorrect!");
  }
  if (!tau_actuated.allFinite()) {
    throw std::invalid_argument("tau_actuated contains non-finite values!");
  }

  m_generalized_state = generalized;
  m_state.joint.tau = tau_actuated;

  if (m_is_fixed_base) {
    m_state.joint.q = generalized.q;
    m_state.joint.qdot = generalized.v;
    m_state.base.reset();
    m_has_state = true;
    return;
  }

  Eigen::Quaterniond quat(generalized.q(6), generalized.q(3),
                          generalized.q(4), generalized.q(5));
  quat.normalize();

  const math::Vector3 base_translation = generalized.q.head<3>();
  const math::Vector6 base_twist = generalized.v.head<6>();

  BaseState base;
  base.pose_world_base =
      pinocchio::SE3(quat.toRotationMatrix(), base_translation);
  base.twist_world_base = pinocchio::Motion(base_twist);

  m_state.joint.q = generalized.q.tail(m_nq_joints);
  m_state.joint.qdot = generalized.v.tail(m_nv_joints);
  m_state.base = base;
  m_has_state = true;
}

const RobotState& RobotSystem::state() const { return m_state; }
const JointState& RobotSystem::jointState() const { return m_state.joint; }

const BaseState& RobotSystem::baseState() const {
  PINOCCHIO_CHECK_INPUT_ARGUMENT(m_state.base.has_value(),
                                 "Base state is not available for this robot.");
  return *m_state.base;
}

const math::Vector& RobotSystem::generalized_q() const {
  return m_generalized_state.q;
}
const math::Vector& RobotSystem::generalized_v() const {
  return m_generalized_state.v;
}
const math::Vector& RobotSystem::tau_actuated() const {
  return m_state.joint.tau;
}

math::Vector RobotSystem::generalized_actuation_force() const {
  math::Vector force = math::Vector::Zero(m_model.nv);
  force.tail(m_na) = m_state.joint.tau;
  return force;
}

double RobotSystem::time() const { return m_time; }

void RobotSystem::setTime(double time) {
  if (!std::isfinite(time)) {
    throw std::invalid_argument("The time value is not finite!");
  }
  m_time = time;
}

bool RobotSystem::isValidJointState(const JointState& joint) const {
  return joint.q.size() == m_nq_joints && joint.qdot.size() == m_nv_joints &&
         joint.tau.size() == m_na && joint.q.allFinite() &&
         joint.qdot.allFinite() && joint.tau.allFinite();
}

bool RobotSystem::isValidBaseState(const BaseState& base) const {
  return base.pose_world_base.translation().allFinite() &&
         base.pose_world_base.rotation().allFinite() &&
         base.twist_world_base.toVector().allFinite();
}

bool RobotSystem::isValidGeneralizedState(
    const GeneralizedState& generalized) const {
  return generalized.q.size() == m_model.nq &&
         generalized.v.size() == m_model.nv && generalized.q.allFinite() &&
         generalized.v.allFinite();
}

void RobotSystem::validateJointState(const JointState& joint) const {
  if (joint.q.size() != m_nq_joints) {
    throw std::invalid_argument("The size of joint.q is incorrect!");
  }
  if (joint.qdot.size() != m_nv_joints) {
    throw std::invalid_argument("The size of joint.qdot is incorrect!");
  }
  if (joint.tau.size() != m_na) {
    throw std::invalid_argument("The size of joint.tau is incorrect!");
  }
  if (!joint.q.allFinite()) {
    throw std::invalid_argument("joint.q contains non-finite values!");
  }
  if (!joint.qdot.allFinite()) {
    throw std::invalid_argument("joint.qdot contains non-finite values!");
  }
  if (!joint.tau.allFinite()) {
    throw std::invalid_argument("joint.tau contains non-finite values!");
  }
}

void RobotSystem::validateBaseState(const BaseState& base) const {
  if (!isValidBaseState(base)) {
    throw std::invalid_argument("Base state contains non-finite values!");
  }
}

void RobotSystem::validateGeneralizedState(
    const GeneralizedState& generalized) const {
  if (generalized.q.size() != m_model.nq) {
    throw std::invalid_argument("The size of generalized.q is incorrect!");
  }
  if (generalized.v.size() != m_model.nv) {
    throw std::invalid_argument("The size of generalized.v is incorrect!");
  }
  if (!generalized.q.allFinite()) {
    throw std::invalid_argument("generalized.q contains non-finite values!");
  }
  if (!generalized.v.allFinite()) {
    throw std::invalid_argument("generalized.v contains non-finite values!");
  }
}

void RobotSystem::computeAllTerms(pinocchio::Data& data,
                                  const math::Vector& q,
                                  const math::Vector& v) const {
  pinocchio::computeAllTerms(m_model, data, q, v);
  data.M.triangularView<Eigen::StrictlyLower>() =
      data.M.transpose().triangularView<Eigen::StrictlyLower>();
  // computeAllTerms does not compute the com acceleration, so we need to call
  // centerOfMass Check this line, calling with zero acceleration at the last
  // phase compute the CoM acceleration.
  //      pinocchio::centerOfMass(m_model, data, q,v,false);
  pinocchio::updateFramePlacements(m_model, data);
  pinocchio::centerOfMass(m_model, data, q, v, m_zero_v);
  pinocchio::ccrba(m_model, data, q, v);
}

const math::Vector& RobotSystem::rotor_inertias() const {
  return m_rotor_inertias;
}
const math::Vector& RobotSystem::gear_ratios() const {
  return m_gear_ratios;
}

bool RobotSystem::rotor_inertias(math::ConstRefVector rotor_inertias) {
  PINOCCHIO_CHECK_INPUT_ARGUMENT(
      rotor_inertias.size() == m_rotor_inertias.size(),
      "The size of the rotor_inertias vector is incorrect!");
  m_rotor_inertias = rotor_inertias;
  updateMd();
  return true;
}

bool RobotSystem::gear_ratios(math::ConstRefVector gear_ratios) {
  PINOCCHIO_CHECK_INPUT_ARGUMENT(
      gear_ratios.size() == m_gear_ratios.size(),
      "The size of the gear_ratios vector is incorrect!");
  m_gear_ratios = gear_ratios;
  updateMd();
  return true;
}

void RobotSystem::updateMd() {
  m_Md =
      m_gear_ratios.cwiseProduct(m_gear_ratios.cwiseProduct(m_rotor_inertias));
}

void RobotSystem::com(const pinocchio::Data& data, math::RefVector com_pos,
                      math::RefVector com_vel,
                      math::RefVector com_acc) const {
  com_pos = data.com[0];
  com_vel = data.vcom[0];
  com_acc = data.acom[0];
}

const math::Vector3& RobotSystem::com(const pinocchio::Data& data) const {
  return data.com[0];
}

const math::Vector3& RobotSystem::com_vel(const pinocchio::Data& data) const {
  return data.vcom[0];
}

const math::Vector3& RobotSystem::com_acc(const pinocchio::Data& data) const {
  return data.acom[0];
}

const math::Matrix3x& RobotSystem::Jcom(
    const pinocchio::Data& data) const {
  return data.Jcom;
}

const math::Matrix& RobotSystem::mass(const pinocchio::Data& data) {
  m_M = data.M;
  m_M.diagonal().tail(m_na) += m_Md;
  return m_M;
}

const math::Vector& RobotSystem::nonLinearEffects(
    const pinocchio::Data& data) const {
  return data.nle;
}

const pinocchio::SE3& RobotSystem::position(
    const pinocchio::Data& data, pinocchio::Model::JointIndex index) const {
  PINOCCHIO_CHECK_INPUT_ARGUMENT(
      index < data.oMi.size(),
      "The index needs to be less than the size of the oMi vector");
  return data.oMi[index];
}

const pinocchio::Motion& RobotSystem::velocity(
    const pinocchio::Data& data, pinocchio::Model::JointIndex index) const {
  PINOCCHIO_CHECK_INPUT_ARGUMENT(
      index < data.v.size(),
      "The index needs to be less than the size of the v vector");
  return data.v[index];
}

const pinocchio::Motion& RobotSystem::acceleration(
    const pinocchio::Data& data, pinocchio::Model::JointIndex index) const {
  PINOCCHIO_CHECK_INPUT_ARGUMENT(
      index < data.a.size(),
      "The index needs to be less than the size of the a vector");
  return data.a[index];
}

void RobotSystem::jacobianWorld(const pinocchio::Data& data,
                                pinocchio::Model::JointIndex index,
                                pinocchio::Data::Matrix6x& J) const {
  PINOCCHIO_CHECK_INPUT_ARGUMENT(
      index < data.oMi.size(),
      "The index needs to be less than the size of the oMi vector");
  return pinocchio::getJointJacobian(m_model, data, index, pinocchio::WORLD, J);
}

void RobotSystem::jacobianLocal(const pinocchio::Data& data,
                                pinocchio::Model::JointIndex index,
                                pinocchio::Data::Matrix6x& J) const {
  PINOCCHIO_CHECK_INPUT_ARGUMENT(
      index < data.oMi.size(),
      "The index needs to be less than the size of the oMi vector");
  return pinocchio::getJointJacobian(m_model, data, index, pinocchio::LOCAL, J);
}

pinocchio::SE3 RobotSystem::framePosition(
    const pinocchio::Data& data, pinocchio::Model::FrameIndex index) const {
  PINOCCHIO_CHECK_INPUT_ARGUMENT(index < m_model.frames.size(),
                                 "Frame index greater than size of frame "
                                 "vector in model - frame may not exist");
  const pinocchio::Frame& f = m_model.frames[index];
  return data.oMi[f.parent].act(f.placement);
}

void RobotSystem::framePosition(const pinocchio::Data& data,
                                pinocchio::Model::FrameIndex index,
                                pinocchio::SE3& framePosition) const {
  PINOCCHIO_CHECK_INPUT_ARGUMENT(index < m_model.frames.size(),
                                 "Frame index greater than size of frame "
                                 "vector in model - frame may not exist");
  const pinocchio::Frame& f = m_model.frames[index];
  framePosition = data.oMi[f.parent].act(f.placement);
}

pinocchio::Motion RobotSystem::frameVelocity(
    const pinocchio::Data& data, pinocchio::Model::FrameIndex index) const {
  PINOCCHIO_CHECK_INPUT_ARGUMENT(index < m_model.frames.size(),
                                 "Frame index greater than size of frame "
                                 "vector in model - frame may not exist");
  const pinocchio::Frame& f = m_model.frames[index];
  return f.placement.actInv(data.v[f.parent]);
}

void RobotSystem::frameVelocity(const pinocchio::Data& data,
                                pinocchio::Model::FrameIndex index,
                                pinocchio::Motion& frameVelocity) const {
  PINOCCHIO_CHECK_INPUT_ARGUMENT(index < m_model.frames.size(),
                                 "Frame index greater than size of frame "
                                 "vector in model - frame may not exist");
  const pinocchio::Frame& f = m_model.frames[index];
  frameVelocity = f.placement.actInv(data.v[f.parent]);
}

pinocchio::Motion RobotSystem::frameVelocityWorldOriented(
    const pinocchio::Data& data,
    pinocchio::Model::FrameIndex index) const {
  pinocchio::Motion v_local, v_world;
  pinocchio::SE3 oMi;
  pinocchio::SE3 oMi_rotation_only = pinocchio::SE3::Identity();
  framePosition(data, index, oMi);
  frameVelocity(data, index, v_local);
  oMi_rotation_only.rotation(oMi.rotation());
  v_world = oMi_rotation_only.act(v_local);
  return v_world;
}

pinocchio::Motion RobotSystem::frameAcceleration(
    const pinocchio::Data& data, pinocchio::Model::FrameIndex index) const {
  PINOCCHIO_CHECK_INPUT_ARGUMENT(index < m_model.frames.size(),
                                 "Frame index greater than size of frame "
                                 "vector in model - frame may not exist");
  const pinocchio::Frame& f = m_model.frames[index];
  return f.placement.actInv(data.a[f.parent]);
}

void RobotSystem::frameAcceleration(
    const pinocchio::Data& data, pinocchio::Model::FrameIndex index,
    pinocchio::Motion& frameAcceleration) const {
  PINOCCHIO_CHECK_INPUT_ARGUMENT(index < m_model.frames.size(),
                                 "Frame index greater than size of frame "
                                 "vector in model - frame may not exist");
  const pinocchio::Frame& f = m_model.frames[index];
  frameAcceleration = f.placement.actInv(data.a[f.parent]);
}

pinocchio::Motion RobotSystem::frameAccelerationWorldOriented(
    const pinocchio::Data& data,
    pinocchio::Model::FrameIndex index) const {
  pinocchio::Motion a_local, a_world;
  pinocchio::SE3 oMi;
  pinocchio::SE3 oMi_rotation_only = pinocchio::SE3::Identity();
  framePosition(data, index, oMi);
  frameAcceleration(data, index, a_local);
  oMi_rotation_only.rotation(oMi.rotation());
  a_world = oMi_rotation_only.act(a_local);
  return a_world;
}

pinocchio::Motion RobotSystem::frameClassicAcceleration(
    const pinocchio::Data& data, pinocchio::Model::FrameIndex index) const {
  PINOCCHIO_CHECK_INPUT_ARGUMENT(index < m_model.frames.size(),
                                 "Frame index greater than size of frame "
                                 "vector in model - frame may not exist");
  const pinocchio::Frame& f = m_model.frames[index];
  pinocchio::Motion a = f.placement.actInv(data.a[f.parent]);
  pinocchio::Motion v = f.placement.actInv(data.v[f.parent]);
  a.linear() += v.angular().cross(v.linear());
  return a;
}

void RobotSystem::frameClassicAcceleration(
    const pinocchio::Data& data, pinocchio::Model::FrameIndex index,
    pinocchio::Motion& frameAcceleration) const {
  PINOCCHIO_CHECK_INPUT_ARGUMENT(index < m_model.frames.size(),
                                 "Frame index greater than size of frame "
                                 "vector in model - frame may not exist");
  const pinocchio::Frame& f = m_model.frames[index];
  frameAcceleration = f.placement.actInv(data.a[f.parent]);
  pinocchio::Motion v = f.placement.actInv(data.v[f.parent]);
  frameAcceleration.linear() += v.angular().cross(v.linear());
}

pinocchio::Motion RobotSystem::frameClassicAccelerationWorldOriented(
    const pinocchio::Data& data,
    pinocchio::Model::FrameIndex index) const {
  pinocchio::Motion a_local, a_world;
  pinocchio::SE3 oMi;
  pinocchio::SE3 oMi_rotation_only = pinocchio::SE3::Identity();
  framePosition(data, index, oMi);
  frameClassicAcceleration(data, index, a_local);
  oMi_rotation_only.rotation(oMi.rotation());
  a_world = oMi_rotation_only.act(a_local);
  return a_world;
}

void RobotSystem::frameJacobianWorld(pinocchio::Data& data,
                                     pinocchio::Model::FrameIndex index,
                                     pinocchio::Data::Matrix6x& J) const {
  PINOCCHIO_CHECK_INPUT_ARGUMENT(index < m_model.frames.size(),
                                 "Frame index greater than size of frame "
                                 "vector in model - frame may not exist");
  return pinocchio::getFrameJacobian(m_model, data, index, pinocchio::WORLD, J);
}

void RobotSystem::frameJacobianLocal(pinocchio::Data& data,
                                     pinocchio::Model::FrameIndex index,
                                     pinocchio::Data::Matrix6x& J) const {
  PINOCCHIO_CHECK_INPUT_ARGUMENT(index < m_model.frames.size(),
                                 "Frame index greater than size of frame "
                                 "vector in model - frame may not exist");
  return pinocchio::getFrameJacobian(m_model, data, index, pinocchio::LOCAL, J);
}

const pinocchio::Data::Matrix6x& RobotSystem::momentumJacobian(
    const pinocchio::Data& data) const {
  return data.Ag;
}

math::Vector3 RobotSystem::angularMomentumTimeVariation(
    const pinocchio::Data& data) const {
  return pinocchio::computeCentroidalMomentumTimeVariation(
             m_model, const_cast<pinocchio::Data&>(data))
      .angular();
}

void RobotSystem::setGravity(const pinocchio::Motion& gravity) {
  m_model.gravity = gravity;
}

}  // namespace robots
}  // namespace wbc
