//
// Copyright (c) 2017 CNRS
//

#include "wbc_core/robots/robot-system.hpp"

#include <pinocchio/multibody/model.hpp>
#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/algorithm/center-of-mass.hpp>
#include <pinocchio/algorithm/compute-all-terms.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/centroidal.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>

using namespace pinocchio;
using namespace wbc::math;

namespace wbc {
namespace robots {

RobotSystem::RobotSystem(const std::string& filename,
                         const std::vector<std::string>&, bool verbose)
    : m_verbose(verbose) {
  pinocchio::urdf::buildModel(filename, m_model, m_verbose);
  m_model_filename = filename;
  m_na = m_model.nv;
  m_nq_actuated = m_model.nq;
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
  m_na = m_model.nv - 6;
  m_nq_actuated = m_model.nq - 7;
  m_is_fixed_base = false;
  init();
}

RobotSystem::RobotSystem(const pinocchio::Model& m, bool verbose)
    : m_verbose(verbose) {
  m_model = m;
  m_model_filename = "";
  m_na = m_model.nv - 6;
  m_nq_actuated = m_model.nq - 7;
  m_is_fixed_base = false;
  init();
}

RobotSystem::RobotSystem(const pinocchio::Model& m, RootJointType rootJoint,
                         bool verbose)
    : m_verbose(verbose) {
  m_model = m;
  m_model_filename = "";
  m_na = m_model.nv;
  m_nq_actuated = m_model.nq;
  m_is_fixed_base = true;
  switch (rootJoint) {
    case FIXED_BASE_SYSTEM:
      break;
    case FLOATING_BASE_SYSTEM:
      m_na -= 6;
      m_nq_actuated = m_model.nq - 7;
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
  m_state.q = pinocchio::neutral(m_model);
  m_state.qdot.setZero(m_model.nv);
  m_state.time = 0.0;
  m_Md.setZero(m_na);
  m_M.setZero(m_model.nv, m_model.nv);
  m_zero_v.setZero(m_model.nv);
  m_has_state = false;
}

int RobotSystem::nq() const { return m_model.nq; }
int RobotSystem::nv() const { return m_model.nv; }
int RobotSystem::na() const { return m_na; }
int RobotSystem::nq_actuated() const { return m_nq_actuated; }
bool RobotSystem::is_fixed_base() const { return m_is_fixed_base; }
bool RobotSystem::hasState() const { return m_has_state; }

const Model& RobotSystem::model() const { return m_model; }
Model& RobotSystem::model() { return m_model; }

void RobotSystem::updateState(ConstRefVector q, ConstRefVector qdot) {
  updateState(q, qdot, m_state.time);
}

void RobotSystem::updateState(ConstRefVector q, ConstRefVector qdot,
                              double time) {
  PINOCCHIO_CHECK_INPUT_ARGUMENT(q.size() == m_model.nq,
                                 "The size of q is incorrect!");
  PINOCCHIO_CHECK_INPUT_ARGUMENT(qdot.size() == m_model.nv,
                                 "The size of qdot is incorrect!");
  m_state.q = q;
  m_state.qdot = qdot;
  m_state.time = time;
  m_has_state = true;
}

void RobotSystem::updateState(const RobotState& state) {
  updateState(state.q, state.qdot, state.time);
}

const RobotState& RobotSystem::state() const { return m_state; }
const Vector& RobotSystem::q() const { return m_state.q; }
const Vector& RobotSystem::qdot() const { return m_state.qdot; }
double RobotSystem::time() const { return m_state.time; }

void RobotSystem::computeAllTerms(Data& data, const Vector& q,
                                   const Vector& v) const {
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

const Vector& RobotSystem::rotor_inertias() const { return m_rotor_inertias; }
const Vector& RobotSystem::gear_ratios() const { return m_gear_ratios; }

bool RobotSystem::rotor_inertias(ConstRefVector rotor_inertias) {
  PINOCCHIO_CHECK_INPUT_ARGUMENT(
      rotor_inertias.size() == m_rotor_inertias.size(),
      "The size of the rotor_inertias vector is incorrect!");
  m_rotor_inertias = rotor_inertias;
  updateMd();
  return true;
}

bool RobotSystem::gear_ratios(ConstRefVector gear_ratios) {
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

void RobotSystem::com(const Data& data, RefVector com_pos, RefVector com_vel,
                       RefVector com_acc) const {
  com_pos = data.com[0];
  com_vel = data.vcom[0];
  com_acc = data.acom[0];
}

const Vector3& RobotSystem::com(const Data& data) const { return data.com[0]; }

const Vector3& RobotSystem::com_vel(const Data& data) const {
  return data.vcom[0];
}

const Vector3& RobotSystem::com_acc(const Data& data) const {
  return data.acom[0];
}

const Matrix3x& RobotSystem::Jcom(const Data& data) const { return data.Jcom; }

const Matrix& RobotSystem::mass(const Data& data) {
  m_M = data.M;
  m_M.diagonal().tail(m_na) += m_Md;
  return m_M;
}

const Vector& RobotSystem::nonLinearEffects(const Data& data) const {
  return data.nle;
}

const SE3& RobotSystem::position(const Data& data,
                                  const Model::JointIndex index) const {
  PINOCCHIO_CHECK_INPUT_ARGUMENT(
      index < data.oMi.size(),
      "The index needs to be less than the size of the oMi vector");
  return data.oMi[index];
}

const Motion& RobotSystem::velocity(const Data& data,
                                     const Model::JointIndex index) const {
  PINOCCHIO_CHECK_INPUT_ARGUMENT(
      index < data.v.size(),
      "The index needs to be less than the size of the v vector");
  return data.v[index];
}

const Motion& RobotSystem::acceleration(const Data& data,
                                         const Model::JointIndex index) const {
  PINOCCHIO_CHECK_INPUT_ARGUMENT(
      index < data.a.size(),
      "The index needs to be less than the size of the a vector");
  return data.a[index];
}

void RobotSystem::jacobianWorld(const Data& data,
                                 const Model::JointIndex index,
                                 Data::Matrix6x& J) const {
  PINOCCHIO_CHECK_INPUT_ARGUMENT(
      index < data.oMi.size(),
      "The index needs to be less than the size of the oMi vector");
  return pinocchio::getJointJacobian(m_model, data, index, pinocchio::WORLD, J);
}

void RobotSystem::jacobianLocal(const Data& data,
                                 const Model::JointIndex index,
                                 Data::Matrix6x& J) const {
  PINOCCHIO_CHECK_INPUT_ARGUMENT(
      index < data.oMi.size(),
      "The index needs to be less than the size of the oMi vector");
  return pinocchio::getJointJacobian(m_model, data, index, pinocchio::LOCAL, J);
}

SE3 RobotSystem::framePosition(const Data& data,
                                const Model::FrameIndex index) const {
  PINOCCHIO_CHECK_INPUT_ARGUMENT(index < m_model.frames.size(),
                                 "Frame index greater than size of frame "
                                 "vector in model - frame may not exist");
  const Frame& f = m_model.frames[index];
  return data.oMi[f.parent].act(f.placement);
}

void RobotSystem::framePosition(const Data& data,
                                 const Model::FrameIndex index,
                                 SE3& framePosition) const {
  PINOCCHIO_CHECK_INPUT_ARGUMENT(index < m_model.frames.size(),
                                 "Frame index greater than size of frame "
                                 "vector in model - frame may not exist");
  const Frame& f = m_model.frames[index];
  framePosition = data.oMi[f.parent].act(f.placement);
}

Motion RobotSystem::frameVelocity(const Data& data,
                                   const Model::FrameIndex index) const {
  PINOCCHIO_CHECK_INPUT_ARGUMENT(index < m_model.frames.size(),
                                 "Frame index greater than size of frame "
                                 "vector in model - frame may not exist");
  const Frame& f = m_model.frames[index];
  return f.placement.actInv(data.v[f.parent]);
}

void RobotSystem::frameVelocity(const Data& data,
                                 const Model::FrameIndex index,
                                 Motion& frameVelocity) const {
  PINOCCHIO_CHECK_INPUT_ARGUMENT(index < m_model.frames.size(),
                                 "Frame index greater than size of frame "
                                 "vector in model - frame may not exist");
  const Frame& f = m_model.frames[index];
  frameVelocity = f.placement.actInv(data.v[f.parent]);
}

Motion RobotSystem::frameVelocityWorldOriented(
    const Data& data, const Model::FrameIndex index) const {
  Motion v_local, v_world;
  SE3 oMi;
  SE3 oMi_rotation_only = SE3::Identity();
  framePosition(data, index, oMi);
  frameVelocity(data, index, v_local);
  oMi_rotation_only.rotation(oMi.rotation());
  v_world = oMi_rotation_only.act(v_local);
  return v_world;
}

Motion RobotSystem::frameAcceleration(const Data& data,
                                       const Model::FrameIndex index) const {
  PINOCCHIO_CHECK_INPUT_ARGUMENT(index < m_model.frames.size(),
                                 "Frame index greater than size of frame "
                                 "vector in model - frame may not exist");
  const Frame& f = m_model.frames[index];
  return f.placement.actInv(data.a[f.parent]);
}

void RobotSystem::frameAcceleration(const Data& data,
                                     const Model::FrameIndex index,
                                     Motion& frameAcceleration) const {
  PINOCCHIO_CHECK_INPUT_ARGUMENT(index < m_model.frames.size(),
                                 "Frame index greater than size of frame "
                                 "vector in model - frame may not exist");
  const Frame& f = m_model.frames[index];
  frameAcceleration = f.placement.actInv(data.a[f.parent]);
}

Motion RobotSystem::frameAccelerationWorldOriented(
    const Data& data, const Model::FrameIndex index) const {
  Motion a_local, a_world;
  SE3 oMi;
  SE3 oMi_rotation_only = SE3::Identity();
  framePosition(data, index, oMi);
  frameAcceleration(data, index, a_local);
  oMi_rotation_only.rotation(oMi.rotation());
  a_world = oMi_rotation_only.act(a_local);
  return a_world;
}

Motion RobotSystem::frameClassicAcceleration(
    const Data& data, const Model::FrameIndex index) const {
  PINOCCHIO_CHECK_INPUT_ARGUMENT(index < m_model.frames.size(),
                                 "Frame index greater than size of frame "
                                 "vector in model - frame may not exist");
  const Frame& f = m_model.frames[index];
  Motion a = f.placement.actInv(data.a[f.parent]);
  Motion v = f.placement.actInv(data.v[f.parent]);
  a.linear() += v.angular().cross(v.linear());
  return a;
}

void RobotSystem::frameClassicAcceleration(const Data& data,
                                            const Model::FrameIndex index,
                                            Motion& frameAcceleration) const {
  PINOCCHIO_CHECK_INPUT_ARGUMENT(index < m_model.frames.size(),
                                 "Frame index greater than size of frame "
                                 "vector in model - frame may not exist");
  const Frame& f = m_model.frames[index];
  frameAcceleration = f.placement.actInv(data.a[f.parent]);
  Motion v = f.placement.actInv(data.v[f.parent]);
  frameAcceleration.linear() += v.angular().cross(v.linear());
}

Motion RobotSystem::frameClassicAccelerationWorldOriented(
    const Data& data, const Model::FrameIndex index) const {
  Motion a_local, a_world;
  SE3 oMi;
  SE3 oMi_rotation_only = SE3::Identity();
  framePosition(data, index, oMi);
  frameClassicAcceleration(data, index, a_local);
  oMi_rotation_only.rotation(oMi.rotation());
  a_world = oMi_rotation_only.act(a_local);
  return a_world;
}

void RobotSystem::frameJacobianWorld(Data& data, const Model::FrameIndex index,
                                      Data::Matrix6x& J) const {
  PINOCCHIO_CHECK_INPUT_ARGUMENT(index < m_model.frames.size(),
                                 "Frame index greater than size of frame "
                                 "vector in model - frame may not exist");
  return pinocchio::getFrameJacobian(m_model, data, index, pinocchio::WORLD, J);
}

void RobotSystem::frameJacobianLocal(Data& data, const Model::FrameIndex index,
                                      Data::Matrix6x& J) const {
  PINOCCHIO_CHECK_INPUT_ARGUMENT(index < m_model.frames.size(),
                                 "Frame index greater than size of frame "
                                 "vector in model - frame may not exist");
  return pinocchio::getFrameJacobian(m_model, data, index, pinocchio::LOCAL, J);
}

const Data::Matrix6x& RobotSystem::momentumJacobian(const Data& data) const {
  return data.Ag;
}

Vector3 RobotSystem::angularMomentumTimeVariation(const Data& data) const {
  return pinocchio::computeCentroidalMomentumTimeVariation(
             m_model, const_cast<Data&>(data))
      .angular();
}

void RobotSystem::setGravity(const Motion& gravity) {
  m_model.gravity = gravity;
}

//    const Vector3 & com(Data & data,const Vector & q,
//                        const bool computeSubtreeComs = true,
//                        const bool updateKinematics = true)
//    {
//      return pinocchio::centerOfMass(m_model, data, q, computeSubtreeComs,
//      updateKinematics);
//    }
//    const Vector3 & com(Data & data, const Vector & q, const Vector & v,
//                 const bool computeSubtreeComs = true,
//                 const bool updateKinematics = true)
//    {
//      return pinocchio::centerOfMass(m_model, data, q, v, computeSubtreeComs,
//      updateKinematics);
//    }

}  // namespace robots
}  // namespace wbc
