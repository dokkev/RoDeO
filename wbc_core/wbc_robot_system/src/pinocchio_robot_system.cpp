/**
 * @file wbc_core/wbc_robot_system/src/pinocchio_robot_system.cpp
 * @brief Doxygen documentation for pinocchio_robot_system module.
 */
#include "wbc_robot_system/pinocchio_robot_system.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace wbc {
////////////////////////////////////////////////////////////////////////////////
PinocchioRobotSystem::PinocchioRobotSystem(
    const std::string& urdf_file, const std::string& package_dir,
    bool fixed_base, bool print_info,
    std::vector<std::string>* unactuated_joint_list)
    : RobotSystem(fixed_base, print_info),
      urdf_file_(urdf_file),
      package_dir_(package_dir) {
  Initialize();

  if (unactuated_joint_list != nullptr) {
    for (const std::string& joint_name : *unactuated_joint_list) {
      actuator_name_idx_map_.erase(joint_name);
      joint_name_to_actuated_idx_.erase(joint_name);
    }
  }

  if (print_info_) {
    PrintRobotInfo();
  }
}

////////////////////////////////////////////////////////////////////////////////
void PinocchioRobotSystem::Initialize() {
  if (fixed_base_) {
    pinocchio::urdf::buildModel(urdf_file_, model_);
    pinocchio::urdf::buildGeom(model_, urdf_file_, pinocchio::COLLISION,
                               collision_model_, package_dir_);
    pinocchio::urdf::buildGeom(model_, urdf_file_, pinocchio::VISUAL,
                               visual_model_, package_dir_);
    num_floating_dof_ = 0;
  } else {
    pinocchio::urdf::buildModel(urdf_file_, pinocchio::JointModelFreeFlyer(),
                                model_);
    pinocchio::urdf::buildGeom(model_, urdf_file_, pinocchio::COLLISION,
                               collision_model_, package_dir_);
    pinocchio::urdf::buildGeom(model_, urdf_file_, pinocchio::VISUAL,
                               visual_model_, package_dir_);
    num_floating_dof_ = 6;
  }

  data_ = pinocchio::Data(model_);
  visual_data_ = pinocchio::GeometryData(visual_model_);
  collision_data_ = pinocchio::GeometryData(collision_model_);

  num_q_ = model_.nq;
  num_qdot_ = model_.nv;
  num_actuated_ = num_qdot_ - num_floating_dof_;
  zero_qddot_ = Eigen::VectorXd::Zero(num_qdot_);

  InitializeRootFrame();

  total_mass_ = pinocchio::computeTotalMass(model_);
  total_mass_legacy_ = total_mass_;

  for (pinocchio::FrameIndex i(0);
       i < static_cast<pinocchio::FrameIndex>(model_.nframes); ++i) {
    const std::string frame_name = model_.frames[i].name;
    link_name_to_frame_idx_[frame_name] = static_cast<int>(i);

    if (frame_name != "universe" && frame_name != "root_joint") {
      const auto frame_type = model_.frames[i].type;
      if (frame_type == pinocchio::FrameType::BODY ||
          frame_type == pinocchio::FrameType::FIXED_JOINT) {
        link_idx_map_[static_cast<int>(model_.getFrameId(frame_name))] =
            frame_name;
      }
    }
  }

  actuated_joint_names_.clear();
  for (pinocchio::JointIndex i(0);
       i < static_cast<pinocchio::JointIndex>(model_.njoints); ++i) {
    const std::string joint_name = model_.names[i];
    if (joint_name == "universe" || joint_name == "root_joint") {
      continue;
    }

    const int q_idx = model_.joints[i].idx_q();
    const int qdot_idx = model_.joints[i].idx_v();
    joint_name_to_q_idx_[joint_name] = q_idx;
    joint_name_to_qdot_idx_[joint_name] = qdot_idx;

    const int idx_offset = fixed_base_ ? 1 : 2;
    const int legacy_joint_idx = static_cast<int>(i) - idx_offset;
    joint_idx_map_[legacy_joint_idx] = joint_name;
    joint_name_idx_map_[joint_name] = legacy_joint_idx;
    actuator_name_idx_map_[joint_name] = legacy_joint_idx;

    joint_name_to_actuated_idx_[joint_name] = legacy_joint_idx;
    actuated_joint_names_.push_back(joint_name);
  }

  assert(num_actuated_ == static_cast<int>(joint_idx_map_.size()));

  joint_position_limits_.resize(num_actuated_, 2);
  joint_velocity_limits_.resize(num_actuated_, 2);
  joint_torque_limits_.resize(num_actuated_, 2);

  if (fixed_base_) {
    joint_position_limits_.leftCols<1>() = model_.lowerPositionLimit;
    joint_position_limits_.rightCols<1>() = model_.upperPositionLimit;
    joint_velocity_limits_.leftCols<1>() = -model_.velocityLimit;
    joint_velocity_limits_.rightCols<1>() = model_.velocityLimit;
    joint_torque_limits_.leftCols<1>() = -model_.effortLimit;
    joint_torque_limits_.rightCols<1>() = model_.effortLimit;
  } else {
    // Position limits index into nq-space: floating base = 3 pos + 4 quat = 7
    // elements, so actuated joints start at num_floating_dof_ + 1 = 7.
    // Velocity/torque limits index into nv-space: floating base = 6 elements,
    // so actuated joints start at num_floating_dof_ = 6.
    joint_position_limits_.leftCols<1>() =
        model_.lowerPositionLimit.segment(num_floating_dof_ + 1, num_actuated_);
    joint_position_limits_.rightCols<1>() =
        model_.upperPositionLimit.segment(num_floating_dof_ + 1, num_actuated_);
    joint_velocity_limits_.leftCols<1>() =
        -model_.velocityLimit.segment(num_floating_dof_, num_actuated_);
    joint_velocity_limits_.rightCols<1>() =
        model_.velocityLimit.segment(num_floating_dof_, num_actuated_);
    joint_torque_limits_.leftCols<1>() =
        -model_.effortLimit.segment(num_floating_dof_, num_actuated_);
    joint_torque_limits_.rightCols<1>() =
        model_.effortLimit.segment(num_floating_dof_, num_actuated_);
  }

  // Soft limits default to hard (URDF) limits until overridden by YAML scale.
  soft_position_limits_ = joint_position_limits_;
  soft_velocity_limits_ = joint_velocity_limits_;
  soft_torque_limits_   = joint_torque_limits_;

  q_ = Eigen::VectorXd::Zero(num_q_);
  qdot_ = Eigen::VectorXd::Zero(num_qdot_);
  joint_positions_ = Eigen::VectorXd::Zero(num_actuated_);
  joint_velocities_ = Eigen::VectorXd::Zero(num_actuated_);
  centroidal_inertia_.setZero();
  centroidal_momentum_.setZero();
  centroidal_momentum_matrix_ = Eigen::MatrixXd::Zero(6, num_qdot_);
  nonlinear_effects_cache_ = Eigen::VectorXd::Zero(num_qdot_);
  gravity_cache_ = Eigen::VectorXd::Zero(num_qdot_);
  coriolis_cache_ = Eigen::VectorXd::Zero(num_qdot_);
  link_jac_scratch_ = Eigen::MatrixXd::Zero(6, num_qdot_);
  manip_data_ = pinocchio::Data(model_);
  manip_jac_ = Eigen::MatrixXd::Zero(6, num_qdot_);
  InvalidateDynamicsCache();
}

////////////////////////////////////////////////////////////////////////////////
void PinocchioRobotSystem::InitializeRootFrame() {
  const int joint_offset = 2;
  if (static_cast<int>(model_.frames.size()) > joint_offset) {
    root_frame_name_ = model_.frames[joint_offset].name;
  } else {
    root_frame_name_ = model_.frames[1].name;
  }
  base_local_com_pos_ = model_.inertias[1].lever();
}

////////////////////////////////////////////////////////////////////////////////
void PinocchioRobotSystem::InvalidateDynamicsCache() {
  acceleration_kinematics_valid_ = false;
  mass_matrix_valid_ = false;
  mass_matrix_inverse_valid_ = false;
  nonlinear_effects_valid_ = false;
  gravity_valid_ = false;
  coriolis_valid_ = false;
}

////////////////////////////////////////////////////////////////////////////////
void PinocchioRobotSystem::EnsureAccelerationKinematics() {
  if (acceleration_kinematics_valid_) {
    return;
  }
  pinocchio::forwardKinematics(model_, data_, q_, qdot_, zero_qddot_);
  acceleration_kinematics_valid_ = true;
}

////////////////////////////////////////////////////////////////////////////////
void PinocchioRobotSystem::EnsureMassMatrix() {
  if (mass_matrix_valid_) {
    return;
  }
  pinocchio::crba(model_, data_, q_);
  data_.M.triangularView<Eigen::StrictlyLower>() =
      data_.M.transpose().triangularView<Eigen::StrictlyLower>();
  mass_matrix_valid_ = true;
}

////////////////////////////////////////////////////////////////////////////////
void PinocchioRobotSystem::EnsureNonlinearEffects() {
  if (nonlinear_effects_valid_) {
    return;
  }
  nonlinear_effects_cache_ = pinocchio::nonLinearEffects(model_, data_, q_, qdot_);
  nonlinear_effects_valid_ = true;
  coriolis_valid_ = false;
}

////////////////////////////////////////////////////////////////////////////////
void PinocchioRobotSystem::EnsureGravity() {
  if (gravity_valid_) {
    return;
  }
  gravity_cache_ = pinocchio::computeGeneralizedGravity(model_, data_, q_);
  gravity_valid_ = true;
  coriolis_valid_ = false;
}

////////////////////////////////////////////////////////////////////////////////
void PinocchioRobotSystem::EnsureCoriolis() {
  if (coriolis_valid_) {
    return;
  }
  EnsureNonlinearEffects();
  EnsureGravity();
  coriolis_cache_ = nonlinear_effects_cache_ - gravity_cache_;
  coriolis_valid_ = true;
}

////////////////////////////////////////////////////////////////////////////////
void PinocchioRobotSystem::PrintRobotInfo() const {
  std::cout << "=======================" << std::endl;
  std::cout << "Pinocchio robot info" << std::endl;
  std::cout << "=======================" << std::endl;

  std::cout << "============ robot link ================" << std::endl;
  for (const auto& [link_idx, link_name] : link_idx_map_) {
    std::cout << "constexpr int " << link_name << " = " << link_idx << ";"
              << std::endl;
  }

  std::cout << "============ robot joint ================" << std::endl;
  for (const auto& [joint_idx, joint_name] : joint_idx_map_) {
    std::cout << "constexpr int " << joint_name << " = " << joint_idx << ";"
              << std::endl;
  }

  std::cout << "============ robot ================" << std::endl;
  std::cout << "constexpr int n_qdot = " << qdot_.size() << ";" << std::endl;
  std::cout << "constexpr int n_adof = " << qdot_.size() - num_floating_dof_
            << ";" << std::endl;
}

////////////////////////////////////////////////////////////////////////////////
int PinocchioRobotSystem::GetFrameIndex(const std::string& link_name) const {
  const auto it = link_name_to_frame_idx_.find(link_name);
  if (it == link_name_to_frame_idx_.end()) {
    throw std::runtime_error("Link/frame not found: " + link_name);
  }
  return it->second;
}

////////////////////////////////////////////////////////////////////////////////
int PinocchioRobotSystem::GetQIndex(const std::string& joint_name) const {
  const auto it = joint_name_to_q_idx_.find(joint_name);
  if (it == joint_name_to_q_idx_.end()) {
    throw std::runtime_error("Joint not found: " + joint_name);
  }
  return it->second;
}

////////////////////////////////////////////////////////////////////////////////
int PinocchioRobotSystem::GetQdotIndex(const std::string& joint_name) const {
  const auto it = joint_name_to_qdot_idx_.find(joint_name);
  if (it == joint_name_to_qdot_idx_.end()) {
    throw std::runtime_error("Joint not found: " + joint_name);
  }
  return it->second;
}

////////////////////////////////////////////////////////////////////////////////
int PinocchioRobotSystem::GetActuatedIndex(const std::string& joint_name) const {
  const auto it = joint_name_to_actuated_idx_.find(joint_name);
  if (it == joint_name_to_actuated_idx_.end()) {
    throw std::runtime_error("Joint not found: " + joint_name);
  }
  return it->second;
}

////////////////////////////////////////////////////////////////////////////////
std::string PinocchioRobotSystem::GetBaseLinkName() const {
  return root_frame_name_;
}

////////////////////////////////////////////////////////////////////////////////
void PinocchioRobotSystem::UpdateState(
    const Eigen::Vector3d& base_pos, const Eigen::Quaterniond& base_quat,
    const Eigen::Vector3d& base_lin_vel, const Eigen::Vector3d& base_ang_vel,
    const std::map<std::string, double>& joint_positions,
    const std::map<std::string, double>& joint_velocities,
    bool update_centroidal) {
  const std::size_t expected_joint_count = joint_name_to_q_idx_.size();
  if (joint_positions.size() != expected_joint_count ||
      joint_velocities.size() != expected_joint_count) {
    throw std::runtime_error(
        "UpdateState requires complete joint position/velocity maps");
  }

  for (const auto& [joint_name, q_idx] : joint_name_to_q_idx_) {
    const auto pos_it = joint_positions.find(joint_name);
    const auto vel_it = joint_velocities.find(joint_name);
    if (pos_it == joint_positions.end() || vel_it == joint_velocities.end()) {
      throw std::runtime_error("UpdateState missing joint entry: " + joint_name);
    }
    q_[q_idx] = pos_it->second;
    qdot_[GetQdotIndex(joint_name)] = vel_it->second;
  }

  if (!fixed_base_) {
    q_.segment<3>(0) = base_pos;
    q_.segment<4>(3) = base_quat.normalized().coeffs();

    const Eigen::Matrix3d rot_w_base = base_quat.normalized().toRotationMatrix();
    qdot_.segment<3>(0) = rot_w_base.transpose() * base_lin_vel;
    qdot_.segment<3>(3) = rot_w_base.transpose() * base_ang_vel;
  } else {
    const Eigen::Matrix3d rotation = base_quat.normalized().toRotationMatrix();
    pinocchio::SE3 base_transform(rotation, base_pos);
    model_.jointPlacements[1] = base_transform;
  }

  if (num_actuated_ > 0) {
    joint_positions_ = GetJointPos();
    joint_velocities_ = GetJointVel();
  }

  pinocchio::forwardKinematics(model_, data_, q_, qdot_);
  pinocchio::computeJointJacobians(model_, data_, q_);
  pinocchio::updateFramePlacements(model_, data_);
  InvalidateDynamicsCache();

  if (update_centroidal) {
    UpdateCentroidalQuantities();
  }
}

////////////////////////////////////////////////////////////////////////////////
void PinocchioRobotSystem::UpdateRobotModel(
    const Eigen::Vector3d& base_joint_pos,
    const Eigen::Quaterniond& base_joint_quat,
    const Eigen::Vector3d& base_joint_lin_vel,
    const Eigen::Vector3d& base_joint_ang_vel, const Eigen::VectorXd& joint_pos,
    const Eigen::VectorXd& joint_vel, bool update_centroid) {
  if (!fixed_base_) {
    q_.segment<3>(0) = base_joint_pos;
    q_.segment<4>(3) = base_joint_quat.normalized().coeffs();
    q_.tail(num_q_ - 7) = joint_pos;

    const Eigen::Matrix3d rot_w_base =
        base_joint_quat.normalized().toRotationMatrix();
    qdot_.segment<3>(0) = rot_w_base.transpose() * base_joint_lin_vel;
    qdot_.segment<3>(3) = rot_w_base.transpose() * base_joint_ang_vel;
    qdot_.tail(num_qdot_ - num_floating_dof_) = joint_vel;
  } else {
    const Eigen::Matrix3d rotation = base_joint_quat.normalized().toRotationMatrix();
    pinocchio::SE3 base_transform(rotation, base_joint_pos);
    model_.jointPlacements[1] = base_transform;

    q_ = joint_pos;
    qdot_ = joint_vel;
  }

  if (num_actuated_ > 0) {
    joint_positions_ = GetJointPos();
    joint_velocities_ = GetJointVel();
  }

  pinocchio::forwardKinematics(model_, data_, q_, qdot_);
  pinocchio::computeJointJacobians(model_, data_, q_);
  pinocchio::updateFramePlacements(model_, data_);
  InvalidateDynamicsCache();

  if (update_centroid) {
    UpdateCentroidalQuantities();
  }
}

////////////////////////////////////////////////////////////////////////////////
void PinocchioRobotSystem::UpdateRobotModel(
    const Eigen::Vector3d& /*base_com_pos*/,
    const Eigen::Quaternion<double>& /*base_com_quat*/,
    const Eigen::Vector3d& /*base_com_lin_vel*/,
    const Eigen::Vector3d& /*base_com_ang_vel*/,
    const Eigen::Vector3d& base_joint_pos,
    const Eigen::Quaternion<double>& base_joint_quat,
    const Eigen::Vector3d& base_joint_lin_vel,
    const Eigen::Vector3d& base_joint_ang_vel,
    const std::map<std::string, double>& joint_positions,
    const std::map<std::string, double>& joint_velocities,
    const bool update_centroid) {
  UpdateState(base_joint_pos, base_joint_quat, base_joint_lin_vel,
              base_joint_ang_vel, joint_positions, joint_velocities,
              update_centroid);
}

////////////////////////////////////////////////////////////////////////////////
Eigen::VectorXd PinocchioRobotSystem::GetQ() const { return q_; }

////////////////////////////////////////////////////////////////////////////////
Eigen::VectorXd PinocchioRobotSystem::GetQdot() const { return qdot_; }

////////////////////////////////////////////////////////////////////////////////
int PinocchioRobotSystem::GetQIdx(int joint_idx) const {
  // Range guaranteed by SelectedJointTask construction; assert fires in debug builds.
  assert(joint_idx >= 0 && joint_idx < num_actuated_);
  return (fixed_base_ ? 0 : 7) + joint_idx;
}

////////////////////////////////////////////////////////////////////////////////
int PinocchioRobotSystem::GetQdotIdx(int joint_idx) const {
  // Range guaranteed by SelectedJointTask construction; assert fires in debug builds.
  assert(joint_idx >= 0 && joint_idx < num_actuated_);
  return (fixed_base_ ? 0 : 6) + joint_idx;
}

////////////////////////////////////////////////////////////////////////////////
Eigen::Ref<const Eigen::VectorXd> PinocchioRobotSystem::GetJointPos() const {
  return q_.tail(num_actuated_);
}

////////////////////////////////////////////////////////////////////////////////
Eigen::Ref<const Eigen::VectorXd> PinocchioRobotSystem::GetJointVel() const {
  return qdot_.tail(num_actuated_);
}

////////////////////////////////////////////////////////////////////////////////
Eigen::Isometry3d PinocchioRobotSystem::GetLinkTransform(
    const std::string& link_name) {
  return GetLinkIsometry(link_name);
}

////////////////////////////////////////////////////////////////////////////////
Eigen::Isometry3d PinocchioRobotSystem::GetLinkIsometry(int link_idx) {
  Eigen::Isometry3d ret = Eigen::Isometry3d::Identity();
  const pinocchio::SE3 trans =
      pinocchio::updateFramePlacement(model_, data_, link_idx);
  ret.linear() = trans.rotation();
  ret.translation() = trans.translation();
  return ret;
}

////////////////////////////////////////////////////////////////////////////////
Eigen::Isometry3d PinocchioRobotSystem::GetLinkIsometry(
    const std::string& link_name) {
  return GetLinkIsometry(GetFrameIndex(link_name));
}

Eigen::Matrix<double, 6, 1>

////////////////////////////////////////////////////////////////////////////////
PinocchioRobotSystem::GetLinkVelocity(const std::string& link_name) {
  return GetLinkSpatialVel(link_name);
}

Eigen::Matrix<double, 6, 1>

////////////////////////////////////////////////////////////////////////////////
PinocchioRobotSystem::GetLinkSpatialVel(int link_idx) const {
  Eigen::Matrix<double, 6, 1> ret = Eigen::Matrix<double, 6, 1>::Zero();
  const pinocchio::Motion fv = pinocchio::getFrameVelocity(
      model_, data_, link_idx, pinocchio::LOCAL_WORLD_ALIGNED);
  ret.head<3>() = fv.angular();
  ret.tail<3>() = fv.linear();
  return ret;
}

Eigen::Matrix<double, 6, 1>

////////////////////////////////////////////////////////////////////////////////
PinocchioRobotSystem::GetLinkSpatialVel(const std::string& link_name) const {
  return GetLinkSpatialVel(GetFrameIndex(link_name));
}

Eigen::Matrix<double, 6, Eigen::Dynamic>

////////////////////////////////////////////////////////////////////////////////
PinocchioRobotSystem::GetLinkJacobian(const std::string& link_name) {
  return GetLinkJacobian(GetFrameIndex(link_name));
}

Eigen::Matrix<double, 6, Eigen::Dynamic>

////////////////////////////////////////////////////////////////////////////////
PinocchioRobotSystem::GetLinkJacobian(int link_idx) {
  Eigen::Matrix<double, 6, Eigen::Dynamic> jac(6, num_qdot_);
  jac.setZero();
  pinocchio::getFrameJacobian(model_, data_, link_idx,
                              pinocchio::LOCAL_WORLD_ALIGNED, jac);

  Eigen::Matrix<double, 6, Eigen::Dynamic> ret(6, num_qdot_);
  ret.setZero();
  ret.topRows<3>() = jac.bottomRows<3>();
  ret.bottomRows<3>() = jac.topRows<3>();
  return ret;
}

////////////////////////////////////////////////////////////////////////////////
Eigen::Matrix<double, 6, 1> PinocchioRobotSystem::GetLinkJacobianDotQdot(
    const std::string& link_name) {
  return GetLinkJacobianDotQdot(GetFrameIndex(link_name));
}

Eigen::Matrix<double, 6, 1>

////////////////////////////////////////////////////////////////////////////////
PinocchioRobotSystem::GetLinkJacobianDotQdot(int link_idx) {
  EnsureAccelerationKinematics();
  const pinocchio::Motion fa = pinocchio::getFrameClassicalAcceleration(
      model_, data_, link_idx, pinocchio::LOCAL_WORLD_ALIGNED);

  Eigen::Matrix<double, 6, 1> ret = Eigen::Matrix<double, 6, 1>::Zero();
  ret.segment<3>(0) = fa.angular();
  ret.segment<3>(3) = fa.linear();
  return ret;
}

////////////////////////////////////////////////////////////////////////////////
void PinocchioRobotSystem::FillLinkJacobian(int link_idx, Eigen::MatrixXd& out) {
  link_jac_scratch_.setZero();
  pinocchio::getFrameJacobian(model_, data_, link_idx,
                              pinocchio::LOCAL_WORLD_ALIGNED, link_jac_scratch_);
  out.topRows<3>()    = link_jac_scratch_.bottomRows<3>();
  out.bottomRows<3>() = link_jac_scratch_.topRows<3>();
}

////////////////////////////////////////////////////////////////////////////////
void PinocchioRobotSystem::FillLinkBodyJacobian(int link_idx, Eigen::MatrixXd& out) {
  link_jac_scratch_.setZero();
  pinocchio::getFrameJacobian(model_, data_, link_idx, pinocchio::LOCAL,
                              link_jac_scratch_);
  out.topRows<3>()    = link_jac_scratch_.bottomRows<3>();
  out.bottomRows<3>() = link_jac_scratch_.topRows<3>();
}

////////////////////////////////////////////////////////////////////////////////
void PinocchioRobotSystem::FillComJacobian(Eigen::MatrixXd& out) {
  pinocchio::jacobianCenterOfMass(model_, data_, q_);
  out = data_.Jcom;
}

////////////////////////////////////////////////////////////////////////////////
double PinocchioRobotSystem::ComputeManipulability(
    int frame_idx, const Eigen::VectorXd& q) {
  manip_jac_.setZero();
  pinocchio::computeFrameJacobian(
      model_, manip_data_, q,
      static_cast<pinocchio::FrameIndex>(frame_idx),
      pinocchio::LOCAL_WORLD_ALIGNED,
      manip_jac_);
  const Eigen::Matrix<double, 6, 6> JJt = manip_jac_ * manip_jac_.transpose();
  return std::sqrt(std::max(0.0, JJt.determinant()));
}

////////////////////////////////////////////////////////////////////////////////
Eigen::Matrix<double, 6, 1>
PinocchioRobotSystem::GetLinkLocalSpatialVelocity(int link_idx) const {
  Eigen::Matrix<double, 6, 1> ret = Eigen::Matrix<double, 6, 1>::Zero();
  const pinocchio::Motion fv =
      pinocchio::getFrameVelocity(model_, data_, link_idx, pinocchio::LOCAL);
  ret.segment<3>(0) = fv.angular();
  ret.segment<3>(3) = fv.linear();
  return ret;
}

////////////////////////////////////////////////////////////////////////////////
Eigen::Matrix<double, 6, Eigen::Dynamic>
PinocchioRobotSystem::GetLinkLocalJacobian(int link_idx) {
  Eigen::Matrix<double, 6, Eigen::Dynamic> jac(6, num_qdot_);
  jac.setZero();
  pinocchio::getFrameJacobian(model_, data_, link_idx, pinocchio::LOCAL, jac);
  Eigen::Matrix<double, 6, Eigen::Dynamic> ret(6, num_qdot_);
  ret.setZero();
  ret.topRows<3>() = jac.bottomRows<3>();
  ret.bottomRows<3>() = jac.topRows<3>();
  return ret;
}

////////////////////////////////////////////////////////////////////////////////
Eigen::Matrix<double, 6, 1>
PinocchioRobotSystem::GetLinkLocalJacobianDotQdot(int link_idx) {
  EnsureAccelerationKinematics();
  const pinocchio::Motion fa = pinocchio::getFrameClassicalAcceleration(
      model_, data_, link_idx, pinocchio::LOCAL);
  Eigen::Matrix<double, 6, 1> ret = Eigen::Matrix<double, 6, 1>::Zero();
  ret.segment<3>(0) = fa.angular();
  ret.segment<3>(3) = fa.linear();
  return ret;
}

////////////////////////////////////////////////////////////////////////////////
Eigen::Vector3d PinocchioRobotSystem::GetComPosition() {
  return pinocchio::centerOfMass(model_, data_, q_);
}

////////////////////////////////////////////////////////////////////////////////
Eigen::Vector3d PinocchioRobotSystem::GetComVelocity() {
  pinocchio::centerOfMass(model_, data_, q_, qdot_);
  return data_.vcom[0];
}

////////////////////////////////////////////////////////////////////////////////
Eigen::Matrix3Xd PinocchioRobotSystem::GetComJacobian() {
  return pinocchio::jacobianCenterOfMass(model_, data_, q_);
}

////////////////////////////////////////////////////////////////////////////////
Eigen::Matrix3Xd PinocchioRobotSystem::GetComJacobianDot() {
  return Eigen::Matrix<double, 3, Eigen::Dynamic>::Zero(3, num_qdot_);
}

////////////////////////////////////////////////////////////////////////////////
Eigen::Matrix3d PinocchioRobotSystem::GetBaseOrientation() {
  if (fixed_base_) {
    return Eigen::Matrix3d::Identity();
  }
  Eigen::Quaterniond world_q_body;
  world_q_body.coeffs() = q_.segment<4>(3);
  return world_q_body.normalized().toRotationMatrix();
}

////////////////////////////////////////////////////////////////////////////////
Eigen::Vector3d PinocchioRobotSystem::GetBaseOrientationYPR() {
  if (fixed_base_) {
    return Eigen::Vector3d::Zero();
  }
  Eigen::Quaterniond world_q_body;
  world_q_body.coeffs() = q_.segment<4>(3);
  return QuaternionToEulerZYX(world_q_body.normalized());
}

////////////////////////////////////////////////////////////////////////////////
Eigen::Vector3d PinocchioRobotSystem::GetBasePosition() {
  if (fixed_base_) {
    return Eigen::Vector3d::Zero();
  }
  Eigen::Quaterniond world_q_body;
  world_q_body.coeffs() = q_.segment<4>(3);
  const Eigen::Matrix3d world_r_body = world_q_body.normalized().toRotationMatrix();
  return q_.head<3>() + world_r_body * base_local_com_pos_;
}

////////////////////////////////////////////////////////////////////////////////
Eigen::Vector3d PinocchioRobotSystem::GetBaseLinearVelocity() {
  if (fixed_base_) {
    return Eigen::Vector3d::Zero();
  }
  const Eigen::Vector3d base_lin_vel_in_base = qdot_.head<3>();
  const Eigen::Vector3d base_ang_vel_in_base = qdot_.segment<3>(3);
  const Eigen::Vector3d base_com_lin_vel_in_base =
      base_lin_vel_in_base - base_local_com_pos_.cross(base_ang_vel_in_base);
  Eigen::Quaterniond world_q_body;
  world_q_body.coeffs() = q_.segment<4>(3);
  const Eigen::Matrix3d world_r_body = world_q_body.normalized().toRotationMatrix();
  return world_r_body * base_com_lin_vel_in_base;
}

////////////////////////////////////////////////////////////////////////////////
Eigen::Vector3d PinocchioRobotSystem::GetBaseAngularVelocity() {
  if (fixed_base_) {
    return Eigen::Vector3d::Zero();
  }
  return qdot_.segment<3>(3);
}

////////////////////////////////////////////////////////////////////////////////
Eigen::Matrix3d PinocchioRobotSystem::GetBaseYawRotationMatrix() {
  const Eigen::Vector3d ypr = GetBaseOrientationYPR();
  return SO3FromRPY(0.0, 0.0, ypr(0));
}

////////////////////////////////////////////////////////////////////////////////
Eigen::Isometry3d PinocchioRobotSystem::GetTransform(
    const std::string& ref_frame, const std::string& target_frame) {
  const Eigen::Isometry3d ref_iso = GetLinkIsometry(ref_frame);
  const Eigen::Isometry3d target_iso = GetLinkIsometry(target_frame);
  return ref_iso.inverse() * target_iso;
}

////////////////////////////////////////////////////////////////////////////////
Eigen::Vector3d PinocchioRobotSystem::GetLocomotionControlPointsInBody(
    int cp_idx) {
  const Eigen::Isometry3d world_iso_root_frame = GetLinkIsometry(root_frame_name_);
  Eigen::Isometry3d root_frame_iso_root_com_frame = Eigen::Isometry3d::Identity();
  root_frame_iso_root_com_frame.translation() = base_local_com_pos_;
  const Eigen::Isometry3d world_iso_root_com_frame =
      world_iso_root_frame * root_frame_iso_root_com_frame;

  const Eigen::Isometry3d base_com_iso_cp =
      world_iso_root_com_frame.inverse() *
      GetLinkIsometry(foot_cp_string_vec_.at(static_cast<size_t>(cp_idx)));

  return base_com_iso_cp.translation();
}

Eigen::Isometry3d

////////////////////////////////////////////////////////////////////////////////
PinocchioRobotSystem::GetLocomotionControlPointsIsometryInBody(int cp_idx) {
  const Eigen::Isometry3d world_iso_root_frame = GetLinkIsometry(root_frame_name_);
  Eigen::Isometry3d root_frame_iso_root_com_frame = Eigen::Isometry3d::Identity();
  root_frame_iso_root_com_frame.translation() = base_local_com_pos_;
  const Eigen::Isometry3d world_iso_root_com_frame =
      world_iso_root_frame * root_frame_iso_root_com_frame;

  return world_iso_root_com_frame.inverse() *
         GetLinkIsometry(foot_cp_string_vec_.at(static_cast<size_t>(cp_idx)));
}

std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>

////////////////////////////////////////////////////////////////////////////////
PinocchioRobotSystem::GetBaseToFootXYOffset() {
  const int kNumFeet = 2;
  std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> offset(
      kNumFeet, Eigen::Vector3d::Zero());

  for (int i = 0; i < kNumFeet; ++i) {
    offset[static_cast<size_t>(i)] = GetLocomotionControlPointsInBody(i);
    offset[static_cast<size_t>(i)][2] = 0.0;
  }

  return offset;
}

////////////////////////////////////////////////////////////////////////////////
Eigen::MatrixXd PinocchioRobotSystem::GetMassMatrix() {
  return GetMassMatrixRef();
}

////////////////////////////////////////////////////////////////////////////////
const Eigen::MatrixXd& PinocchioRobotSystem::GetMassMatrixRef() {
  EnsureMassMatrix();
  return data_.M;
}

////////////////////////////////////////////////////////////////////////////////
Eigen::MatrixXd PinocchioRobotSystem::GetMassMatrixInverse() {
  return GetMassMatrixInverseRef();
}

////////////////////////////////////////////////////////////////////////////////
const pinocchio::Data::RowMatrixXs&
PinocchioRobotSystem::GetMassMatrixInverseRef() {
  if (!mass_matrix_inverse_valid_) {
    pinocchio::computeMinverse(model_, data_, q_);
    data_.Minv.triangularView<Eigen::StrictlyLower>() =
        data_.Minv.transpose().triangularView<Eigen::StrictlyLower>();
    mass_matrix_inverse_valid_ = true;
  }
  return data_.Minv;
}

////////////////////////////////////////////////////////////////////////////////
Eigen::VectorXd PinocchioRobotSystem::GetGravity() {
  return GetGravityRef();
}

////////////////////////////////////////////////////////////////////////////////
const Eigen::VectorXd& PinocchioRobotSystem::GetGravityRef() {
  EnsureGravity();
  return gravity_cache_;
}

////////////////////////////////////////////////////////////////////////////////
Eigen::VectorXd PinocchioRobotSystem::GetCoriolis() {
  return GetCoriolisRef();
}

////////////////////////////////////////////////////////////////////////////////
const Eigen::VectorXd& PinocchioRobotSystem::GetCoriolisRef() {
  EnsureCoriolis();
  return coriolis_cache_;
}

////////////////////////////////////////////////////////////////////////////////
double PinocchioRobotSystem::GetTotalWeight() const {
  return -1.0 * pinocchio::computeTotalMass(model_) * model_.gravity981.coeff(2);
}

////////////////////////////////////////////////////////////////////////////////
void PinocchioRobotSystem::UpdateCentroidalQuantities() {
  pinocchio::ccrba(model_, data_, q_, qdot_);

  centroidal_inertia_.block<3, 3>(0, 0) = data_.Ig.matrix().block<3, 3>(3, 3);
  centroidal_inertia_.block<3, 3>(3, 3) = data_.Ig.matrix().block<3, 3>(0, 0);

  centroidal_momentum_.segment<3>(0) = data_.hg.angular();
  centroidal_momentum_.segment<3>(3) = data_.hg.linear();

  centroidal_momentum_matrix_.topRows<3>() = data_.Ag.bottomRows<3>();
  centroidal_momentum_matrix_.bottomRows<3>() = data_.Ag.topRows<3>();
}

////////////////////////////////////////////////////////////////////////////////
Eigen::Matrix<double, 6, 6> PinocchioRobotSystem::GetIg() const {
  return centroidal_inertia_;
}

////////////////////////////////////////////////////////////////////////////////
Eigen::Matrix<double, 6, 1> PinocchioRobotSystem::GetHg() const {
  return centroidal_momentum_;
}

////////////////////////////////////////////////////////////////////////////////
Eigen::Matrix<double, 6, Eigen::Dynamic> PinocchioRobotSystem::GetAg() const {
  return centroidal_momentum_matrix_;
}

////////////////////////////////////////////////////////////////////////////////
int PinocchioRobotSystem::NumQdot() const { return num_qdot_; }

////////////////////////////////////////////////////////////////////////////////
int PinocchioRobotSystem::NumActiveDof() const { return num_actuated_; }

////////////////////////////////////////////////////////////////////////////////
int PinocchioRobotSystem::NumFloatDof() const { return num_floating_dof_; }

////////////////////////////////////////////////////////////////////////////////
std::map<std::string, double> PinocchioRobotSystem::VectorToJointMap(
    const Eigen::VectorXd& joint_vector) const {
  if (joint_vector.size() != num_actuated_) {
    throw std::runtime_error("Joint vector size mismatch");
  }

  std::map<std::string, double> joint_map;
  for (int i = 0; i < num_actuated_; ++i) {
    joint_map[actuated_joint_names_[static_cast<size_t>(i)]] = joint_vector[i];
  }
  return joint_map;
}

////////////////////////////////////////////////////////////////////////////////
Eigen::VectorXd PinocchioRobotSystem::JointMapToVector(
    const std::map<std::string, double>& joint_map) const {
  Eigen::VectorXd joint_vector(num_actuated_);
  for (int i = 0; i < num_actuated_; ++i) {
    const std::string& joint_name = actuated_joint_names_[static_cast<size_t>(i)];
    const auto it = joint_map.find(joint_name);
    if (it == joint_map.end()) {
      throw std::runtime_error("Joint not found in map: " + joint_name);
    }
    joint_vector[i] = it->second;
  }
  return joint_vector;
}

////////////////////////////////////////////////////////////////////////////////
Eigen::Vector3d PinocchioRobotSystem::QuaternionToEulerZYX(
    const Eigen::Quaterniond& q) {
  const Eigen::Matrix3d r = q.toRotationMatrix();
  const double yaw = std::atan2(r(1, 0), r(0, 0));
  const double pitch = std::atan2(-r(2, 0),
                                  std::sqrt(r(2, 1) * r(2, 1) +
                                            r(2, 2) * r(2, 2)));
  const double roll = std::atan2(r(2, 1), r(2, 2));
  return Eigen::Vector3d(yaw, pitch, roll);
}

////////////////////////////////////////////////////////////////////////////////
Eigen::Matrix3d PinocchioRobotSystem::SO3FromRPY(double roll, double pitch,
                                                  double yaw) {
  const double cr = std::cos(roll);
  const double sr = std::sin(roll);
  const double cp = std::cos(pitch);
  const double sp = std::sin(pitch);
  const double cy = std::cos(yaw);
  const double sy = std::sin(yaw);

  Eigen::Matrix3d r;
  r << cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr,
      sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr,
      -sp, cp * sr, cp * cr;
  return r;
}

} // namespace wbc
