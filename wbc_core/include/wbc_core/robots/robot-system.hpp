//
// Copyright (c) 2017 CNRS
//
// This file is part of tsid
// tsid is free software: you can redistribute it
// and/or modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation, either version
// 3 of the License, or (at your option) any later version.
// tsid is distributed in the hope that it will be
// useful, but WITHOUT ANY WARRANTY; without even the implied warranty
// of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
// General Lesser Public License for more details. You should have
// received a copy of the GNU Lesser General Public License along with
// tsid If not, see
// <http://www.gnu.org/licenses/>.
//

#ifndef WBC_CORE_ROBOTS_ROBOT_SYSTEM_HPP_
#define WBC_CORE_ROBOTS_ROBOT_SYSTEM_HPP_

#include "wbc_core/math/fwd.hpp"
#include "wbc_core/robots/fwd.hpp"
#include "wbc_core/robots/robot-state.hpp"

#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/spatial/motion.hpp>
#include <pinocchio/spatial/se3.hpp>
#include <pinocchio/spatial/fwd.hpp>

#include <string>
#include <vector>

namespace wbc {
namespace robots {

///
/// \brief Pinocchio-backed robot model helper.
///
class RobotSystem {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /* Possible root joints */
  enum RootJointType {
    FIXED_BASE_SYSTEM = 0,
    FLOATING_BASE_SYSTEM = 1,
  };

  RobotSystem(const std::string& filename,
              const std::vector<std::string>& package_dirs,
              bool verbose = false);

  RobotSystem(const std::string& filename,
              const std::vector<std::string>& package_dirs,
              const pinocchio::JointModelVariant& rootJoint,
              bool verbose = false);

  RobotSystem(const pinocchio::Model& m, RootJointType rootJoint,
              bool verbose = false);

  virtual ~RobotSystem() = default;

  virtual int nq() const;
  virtual int nq_joints() const;
  virtual int nv() const;
  virtual int nv_joints() const;
  virtual int na() const;
  virtual bool is_fixed_base() const;
  virtual bool hasState() const;

  ///
  /// \brief Accessor to model.
  ///
  /// \returns a const reference on the model.
  ///
  const pinocchio::Model& model() const;
  pinocchio::Model& model();

  void updateState(const JointState& joint);
  void updateState(const JointState& joint, const BaseState& base);
  void updateState(const GeneralizedState& generalized);
  void updateState(const GeneralizedState& generalized,
                   math::ConstRefVector tau_actuated);

  const RobotState& state() const;
  const JointState& jointState() const;
  const BaseState& baseState() const;

  const math::Vector& generalized_q() const;
  const math::Vector& generalized_v() const;
  const math::Vector& tau_actuated() const;
  math::Vector generalized_actuation_force() const;
  double time() const;
  void setTime(double time);

  bool isValidJointState(const JointState& joint) const;
  bool isValidBaseState(const BaseState& base) const;
  bool isValidGeneralizedState(const GeneralizedState& generalized) const;

  void computeAllTerms(pinocchio::Data& data, const math::Vector& q,
                       const math::Vector& v) const;

  const math::Vector& rotor_inertias() const;
  const math::Vector& gear_ratios() const;

  bool rotor_inertias(math::ConstRefVector rotor_inertias);
  bool gear_ratios(math::ConstRefVector gear_ratios);

  void com(const pinocchio::Data& data, math::RefVector com_pos,
           math::RefVector com_vel, math::RefVector com_acc) const;

  const math::Vector3& com(const pinocchio::Data& data) const;

  const math::Vector3& com_vel(const pinocchio::Data& data) const;

  const math::Vector3& com_acc(const pinocchio::Data& data) const;

  const math::Matrix3x& Jcom(const pinocchio::Data& data) const;

  const math::Matrix& mass(const pinocchio::Data& data);

  const math::Vector& nonLinearEffects(const pinocchio::Data& data) const;

  const pinocchio::SE3& position(const pinocchio::Data& data,
                                 pinocchio::Model::JointIndex index) const;

  const pinocchio::Motion& velocity(
      const pinocchio::Data& data, pinocchio::Model::JointIndex index) const;

  const pinocchio::Motion& acceleration(
      const pinocchio::Data& data, pinocchio::Model::JointIndex index) const;

  void jacobianWorld(const pinocchio::Data& data,
                     pinocchio::Model::JointIndex index,
                     pinocchio::Data::Matrix6x& J) const;

  void jacobianLocal(const pinocchio::Data& data,
                     pinocchio::Model::JointIndex index,
                     pinocchio::Data::Matrix6x& J) const;

  pinocchio::SE3 framePosition(const pinocchio::Data& data,
                               pinocchio::Model::FrameIndex index) const;

  void framePosition(const pinocchio::Data& data,
                     pinocchio::Model::FrameIndex index,
                     pinocchio::SE3& framePosition) const;

  pinocchio::Motion frameVelocity(
      const pinocchio::Data& data, pinocchio::Model::FrameIndex index) const;

  pinocchio::Motion frameVelocityWorldOriented(
      const pinocchio::Data& data, pinocchio::Model::FrameIndex index) const;

  void frameVelocity(const pinocchio::Data& data,
                     pinocchio::Model::FrameIndex index,
                     pinocchio::Motion& frameVelocity) const;

  pinocchio::Motion frameAcceleration(
      const pinocchio::Data& data, pinocchio::Model::FrameIndex index) const;

  pinocchio::Motion frameAccelerationWorldOriented(
      const pinocchio::Data& data, pinocchio::Model::FrameIndex index) const;

  void frameAcceleration(const pinocchio::Data& data,
                         pinocchio::Model::FrameIndex index,
                         pinocchio::Motion& frameAcceleration) const;

  pinocchio::Motion frameClassicAcceleration(
      const pinocchio::Data& data, pinocchio::Model::FrameIndex index) const;

  pinocchio::Motion frameClassicAccelerationWorldOriented(
      const pinocchio::Data& data, pinocchio::Model::FrameIndex index) const;

  void frameClassicAcceleration(const pinocchio::Data& data,
                                pinocchio::Model::FrameIndex index,
                                pinocchio::Motion& frameAcceleration) const;

  void frameJacobianWorld(pinocchio::Data& data,
                          pinocchio::Model::FrameIndex index,
                          pinocchio::Data::Matrix6x& J) const;

  void frameJacobianLocal(pinocchio::Data& data,
                          pinocchio::Model::FrameIndex index,
                          pinocchio::Data::Matrix6x& J) const;

  const pinocchio::Data::Matrix6x& momentumJacobian(
      const pinocchio::Data& data) const;

  math::Vector3 angularMomentumTimeVariation(
      const pinocchio::Data& data) const;

  void setGravity(const pinocchio::Motion& gravity);

 protected:
  void init();
  void validateJointState(const JointState& joint) const;
  void validateBaseState(const BaseState& base) const;
  void validateGeneralizedState(const GeneralizedState& generalized) const;
  void updateMd();

  /// \brief Robot model.
  pinocchio::Model m_model;
  std::string m_model_filename;
  bool m_verbose;

  int m_nq_joints;  ///< Joint configuration dimension excluding floating base.
  int m_nv_joints;  ///< Joint velocity dimension excluding floating base.
  int m_na;         ///< Actuator torque dimension.
  bool m_is_fixed_base;
  math::Vector m_rotor_inertias;
  math::Vector m_gear_ratios;
  RobotState m_state;
  GeneralizedState m_generalized_state;
  double m_time{0.0};
  math::Vector m_Md;  /// diagonal part of inertia matrix due to rotor inertias
  math::Matrix m_M;   /// inertia matrix including rotor inertias
  math::Vector m_zero_v;  /// pre-allocated zero velocity for computeAllTerms
  bool m_has_state{false};
};

}  // namespace robots

}  // namespace wbc

#endif  // WBC_CORE_ROBOTS_ROBOT_SYSTEM_HPP_
