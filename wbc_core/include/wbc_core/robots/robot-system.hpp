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

#include "wbc_core/deprecated.hh"
#include "wbc_core/math/fwd.hpp"
#include "wbc_core/robots/fwd.hpp"

#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/spatial/fwd.hpp>

#include <string>
#include <vector>

namespace wbc {
namespace robots {

/// Generalized robot state snapshot owned by RobotSystem.
struct RobotState {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  math::Vector q;     ///< Generalized configuration, size nq().
  math::Vector qdot;  ///< Generalized velocity, size nv().
  double time{0.0};   ///< State timestamp in seconds.
};

///
/// \brief Pinocchio-backed robot model helper.
///
class RobotSystem {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef math::Scalar Scalar;
  typedef pinocchio::Model Model;
  typedef pinocchio::Data Data;
  typedef pinocchio::Motion Motion;
  typedef pinocchio::Frame Frame;
  typedef pinocchio::SE3 SE3;
  typedef math::Vector Vector;
  typedef math::Vector3 Vector3;
  typedef math::Vector6 Vector6;
  typedef math::Matrix Matrix;
  typedef math::Matrix3x Matrix3x;
  typedef math::RefVector RefVector;
  typedef math::ConstRefVector ConstRefVector;

  /* Possible root joints */
  typedef enum e_RootJointType {
    FIXED_BASE_SYSTEM = 0,
    FLOATING_BASE_SYSTEM = 1,
  } RootJointType;

  RobotSystem(const std::string& filename,
              const std::vector<std::string>& package_dirs,
              bool verbose = false);

  RobotSystem(const std::string& filename,
              const std::vector<std::string>& package_dirs,
              const pinocchio::JointModelVariant& rootJoint,
              bool verbose = false);

  TSID_DEPRECATED RobotSystem(const Model& m, bool verbose = false);

  RobotSystem(const Model& m, RootJointType rootJoint, bool verbose = false);

  virtual ~RobotSystem() = default;

  virtual int nq() const;
  virtual int nq_actuated() const;
  virtual int nv() const;
  virtual int na() const;
  virtual bool is_fixed_base() const;
  virtual bool hasState() const;

  ///
  /// \brief Accessor to model.
  ///
  /// \returns a const reference on the model.
  ///
  const Model& model() const;
  Model& model();

  void updateState(ConstRefVector q, ConstRefVector qdot);
  void updateState(ConstRefVector q, ConstRefVector qdot, double time);
  void updateState(const RobotState& state);
  const RobotState& state() const;
  const Vector& q() const;
  const Vector& qdot() const;
  double time() const;

  void computeAllTerms(Data& data, const Vector& q, const Vector& v) const;

  const Vector& rotor_inertias() const;
  const Vector& gear_ratios() const;

  bool rotor_inertias(ConstRefVector rotor_inertias);
  bool gear_ratios(ConstRefVector gear_ratios);

  void com(const Data& data, RefVector com_pos, RefVector com_vel,
           RefVector com_acc) const;

  const Vector3& com(const Data& data) const;

  const Vector3& com_vel(const Data& data) const;

  const Vector3& com_acc(const Data& data) const;

  const Matrix3x& Jcom(const Data& data) const;

  const Matrix& mass(const Data& data);

  const Vector& nonLinearEffects(const Data& data) const;

  const SE3& position(const Data& data, const Model::JointIndex index) const;

  const Motion& velocity(const Data& data, const Model::JointIndex index) const;

  const Motion& acceleration(const Data& data,
                             const Model::JointIndex index) const;

  void jacobianWorld(const Data& data, const Model::JointIndex index,
                     Data::Matrix6x& J) const;

  void jacobianLocal(const Data& data, const Model::JointIndex index,
                     Data::Matrix6x& J) const;

  SE3 framePosition(const Data& data, const Model::FrameIndex index) const;

  void framePosition(const Data& data, const Model::FrameIndex index,
                     SE3& framePosition) const;

  Motion frameVelocity(const Data& data, const Model::FrameIndex index) const;

  Motion frameVelocityWorldOriented(const Data& data,
                                    const Model::FrameIndex index) const;

  void frameVelocity(const Data& data, const Model::FrameIndex index,
                     Motion& frameVelocity) const;

  Motion frameAcceleration(const Data& data,
                           const Model::FrameIndex index) const;

  Motion frameAccelerationWorldOriented(const Data& data,
                                        const Model::FrameIndex index) const;

  void frameAcceleration(const Data& data, const Model::FrameIndex index,
                         Motion& frameAcceleration) const;

  Motion frameClassicAcceleration(const Data& data,
                                  const Model::FrameIndex index) const;

  Motion frameClassicAccelerationWorldOriented(
      const Data& data, const Model::FrameIndex index) const;

  void frameClassicAcceleration(const Data& data, const Model::FrameIndex index,
                                Motion& frameAcceleration) const;

  void frameJacobianWorld(Data& data, const Model::FrameIndex index,
                          Data::Matrix6x& J) const;

  void frameJacobianLocal(Data& data, const Model::FrameIndex index,
                          Data::Matrix6x& J) const;

  const Data::Matrix6x& momentumJacobian(const Data& data) const;

  Vector3 angularMomentumTimeVariation(const Data& data) const;

  void setGravity(const Motion& gravity);

 protected:
  void init();
  void updateMd();

  /// \brief Robot model.
  Model m_model;
  std::string m_model_filename;
  bool m_verbose;

  int m_nq_actuated;  /// dimension of the configuration space of the actuated
                      /// DoF (nq for fixed-based, nq-7 for floating-base
                      /// robots)
  int m_na;  /// number of actuators (nv for fixed-based, nv-6 for floating-base
             /// robots)
  bool m_is_fixed_base;
  Vector m_rotor_inertias;
  Vector m_gear_ratios;
  RobotState m_state;
  Vector m_Md;      /// diagonal part of inertia matrix due to rotor inertias
  Matrix m_M;       /// inertia matrix including rotor inertias
  Vector m_zero_v;  /// pre-allocated zero velocity for computeAllTerms
  bool m_has_state{false};
};

}  // namespace robots

}  // namespace wbc

#endif  // WBC_CORE_ROBOTS_ROBOT_SYSTEM_HPP_
