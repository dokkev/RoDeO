#pragma once

#include "wbc_robot_system/interface/robot_system.hpp"

#include <pinocchio/algorithm/aba.hpp>
#include <pinocchio/algorithm/center-of-mass.hpp>
#include <pinocchio/algorithm/centroidal.hpp>
#include <pinocchio/algorithm/crba.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/fwd.hpp>
#include <pinocchio/multibody/geometry.hpp>
#include <pinocchio/parsers/urdf.hpp>

#include <map>
#include <string>
#include <unordered_map>
#include <vector>

namespace wbc {
/**
 * @brief Pinocchio implementation of RobotSystem interface with RPC-compatible
 * API surface.
 *
 * This class intentionally exposes both:
 * - the newer WBC interfaces, and
 * - legacy RPC interfaces used in the original codebase.
 */
class PinocchioRobotSystem : public RobotSystem {
public:
  /**
   * @brief Construct a Pinocchio-backed robot system.
   * @param urdf_file Path to URDF file.
   * @param package_dir Package root for mesh/resource resolution.
   * @param fixed_base True for fixed-base model; false for floating-base.
   * @param print_info If true, prints model indexing and sizing info.
   * @param unactuated_joint_list Optional list of joints removed from actuator map.
   */
  PinocchioRobotSystem(const std::string& urdf_file,
                       const std::string& package_dir, bool fixed_base,
                       bool print_info = false,
                       std::vector<std::string>* unactuated_joint_list =
                           nullptr);

  ~PinocchioRobotSystem() override = default;

  // WBC interface
  /** @brief Get generalized-position index for a named joint. */
  int GetQIndex(const std::string& joint_name) const override;
  /** @brief Get generalized-velocity index for a named joint. */
  int GetQdotIndex(const std::string& joint_name) const override;
  /** @brief Get actuated-space index for a named joint. */
  int GetActuatedIndex(const std::string& joint_name) const override;
  /** @brief Get base/root link name used by the model. */
  std::string GetBaseLinkName() const override;

  // WBC interface
  /**
   * @brief Update robot state from base and joint maps.
   * @param base_pos Base position in world frame.
   * @param base_quat Base orientation quaternion.
   * @param base_lin_vel Base linear velocity in world frame.
   * @param base_ang_vel Base angular velocity in world frame.
   * @param joint_positions Joint position map.
   * @param joint_velocities Joint velocity map.
   * @param update_centroidal If true, refresh centroidal quantities.
   */
  void UpdateState(const Eigen::Vector3d& base_pos,
                   const Eigen::Quaterniond& base_quat,
                   const Eigen::Vector3d& base_lin_vel,
                   const Eigen::Vector3d& base_ang_vel,
                   const std::map<std::string, double>& joint_positions,
                   const std::map<std::string, double>& joint_velocities,
                   bool update_centroidal = false) override;

  // WBC interface
  /** @brief Get full generalized position vector. */
  Eigen::VectorXd GetQ() const override;
  /** @brief Get full generalized velocity vector. */
  Eigen::VectorXd GetQdot() const override;
  /** @brief Get mass matrix M(q). */
  Eigen::MatrixXd GetMassMatrix() override;
  /** @brief Get cached mass matrix by const reference. */
  const Eigen::MatrixXd& GetMassMatrixRef();
  /** @brief Get generalized gravity vector g(q). */
  Eigen::VectorXd GetGravity() override;
  /** @brief Get cached gravity vector by const reference. */
  const Eigen::VectorXd& GetGravityRef();
  /** @brief Get Coriolis/centrifugal vector C(q, qdot). */
  Eigen::VectorXd GetCoriolis() override;
  /** @brief Get cached Coriolis vector by const reference. */
  const Eigen::VectorXd& GetCoriolisRef();

  // WBC interface
  /** @brief Get center-of-mass position in world frame. */
  Eigen::Vector3d GetComPosition() override;
  /** @brief Get center-of-mass linear velocity in world frame. */
  Eigen::Vector3d GetComVelocity() override;
  /** @brief Get center-of-mass linear Jacobian. */
  Eigen::Matrix3Xd GetComJacobian() override;
  /** @brief Get time derivative of COM Jacobian (current implementation returns zeros). */
  Eigen::Matrix3Xd GetComJacobianDot() override;

  // WBC interface
  /** @brief Get link transform in world frame. */
  Eigen::Isometry3d GetLinkTransform(const std::string& link_name) override;
  /** @brief Get link spatial velocity (angular; linear). */
  Eigen::Matrix<double, 6, 1>
  GetLinkVelocity(const std::string& link_name) override;
  /** @brief Get link Jacobian with angular rows first. */
  Eigen::Matrix<double, 6, Eigen::Dynamic>
  GetLinkJacobian(const std::string& link_name) override;
  /** @brief Get Jdot*qdot for a link. */
  Eigen::Matrix<double, 6, 1>
  GetLinkJacobianDotQdot(const std::string& link_name) override;

  // WBC interface
  /** @brief Convert joint vector to ordered joint-name map. */
  std::map<std::string, double>
  VectorToJointMap(const Eigen::VectorXd& joint_vector) const override;
  /** @brief Convert joint-name map to ordered joint vector. */
  Eigen::VectorXd JointMapToVector(
      const std::map<std::string, double>& joint_map) const override;

  // WBC pinocchio helpers
  const pinocchio::Model& GetModel() const { return model_; }
  const pinocchio::Data& GetData() const { return data_; }

  /** @brief Get base orientation rotation matrix. */
  Eigen::Matrix3d GetBaseOrientation();
  /** @brief Get base orientation as yaw-pitch-roll (ZYX). */
  Eigen::Vector3d GetBaseOrientationYPR();
  /** @brief Get base position in world frame. */
  Eigen::Vector3d GetBasePosition();
  /** @brief Get base linear velocity in world frame. */
  Eigen::Vector3d GetBaseLinearVelocity();
  /** @brief Get base angular velocity in body/world convention used by Pinocchio state. */
  Eigen::Vector3d GetBaseAngularVelocity();
  /** @brief Get world yaw-only rotation matrix from base orientation. */
  Eigen::Matrix3d GetBaseYawRotationMatrix();
  /** @brief Get inverse mass matrix M(q)^-1. */
  Eigen::MatrixXd GetMassMatrixInverse();
  /** @brief Get inverse mass matrix by const reference. */
  const pinocchio::Data::RowMatrixXs& GetMassMatrixInverseRef();
  /** @brief Get total weight magnitude based on model gravity. */
  double GetTotalWeight() const;

  // Legacy RPC-compatible API
  /** @brief Get root frame name cached at initialization. */
  std::string GetRootFrameName() const { return root_frame_name_; }
  /** @brief Get q index from legacy joint index convention. */
  int GetQIdx(int joint_idx) const;
  /** @brief Get qdot index from legacy joint index convention. */
  int GetQdotIdx(int joint_idx) const;
  /** @brief Get actuated joint positions. */
  Eigen::VectorXd GetJointPos() const;
  /** @brief Get actuated joint velocities. */
  Eigen::VectorXd GetJointVel() const;

  /** @brief Get frame index by link name. */
  int GetFrameIndex(const std::string& link_name) const;
  /** @brief Get link isometry by frame index. */
  Eigen::Isometry3d GetLinkIsometry(int link_idx);
  /** @brief Get link isometry by frame name. */
  Eigen::Isometry3d GetLinkIsometry(const std::string& link_name);
  /** @brief Get LOCAL_WORLD_ALIGNED spatial velocity by index. */
  Eigen::Matrix<double, 6, 1> GetLinkSpatialVel(int link_idx) const;
  /** @brief Get LOCAL_WORLD_ALIGNED spatial velocity by name. */
  Eigen::Matrix<double, 6, 1> GetLinkSpatialVel(const std::string& link_name) const;
  /** @brief Get LOCAL_WORLD_ALIGNED Jacobian by frame index. */
  Eigen::Matrix<double, 6, Eigen::Dynamic> GetLinkJacobian(int link_idx);
  /** @brief Get LOCAL_WORLD_ALIGNED Jdot*qdot by frame index. */
  Eigen::Matrix<double, 6, 1> GetLinkJacobianDotQdot(int link_idx);
  /** @brief Get LOCAL spatial velocity by frame index. */
  Eigen::Matrix<double, 6, 1> GetLinkBodySpatialVel(int link_idx) const;
  /** @brief Get LOCAL Jacobian by frame index. */
  Eigen::Matrix<double, 6, Eigen::Dynamic> GetLinkBodyJacobian(int link_idx);
  /** @brief Get LOCAL Jdot*qdot by frame index. */
  Eigen::Matrix<double, 6, 1> GetLinkBodyJacobianDotQdot(int link_idx);

  /** @brief Get robot COM position in world frame. */
  Eigen::Vector3d GetRobotComPos();
  /** @brief Get robot COM linear velocity in world frame. */
  Eigen::Vector3d GetRobotComLinVel();
  /** @brief Get COM linear Jacobian. */
  Eigen::Matrix<double, 3, Eigen::Dynamic> GetComLinJacobian();
  /** @brief Get COM linear Jdot*qdot. */
  Eigen::Matrix<double, 3, 1> GetComLinJacobianDotQdot();

  /** @brief Get base orientation rotation matrix. */
  Eigen::Matrix3d GetBodyOriRot();
  /** @brief Get base orientation yaw-pitch-roll (ZYX). */
  Eigen::Vector3d GetBodyOriYPR();
  /** @brief Get base COM position in world frame. */
  Eigen::Vector3d GetBodyPos();
  /** @brief Get base COM linear velocity in world frame. */
  Eigen::Vector3d GetBodyVel();
  /** @brief Get world yaw-only rotation matrix from base orientation. */
  Eigen::Matrix3d GetBodyYawRotationMatrix();

  /** @brief Get transform from ref_frame to target_frame. */
  Eigen::Isometry3d GetTransform(const std::string& ref_frame,
                                 const std::string& target_frame);
  /** @brief Get locomotion control point position in base-COM frame. */
  Eigen::Vector3d GetLocomotionControlPointsInBody(int cp_idx);
  /** @brief Get locomotion control point isometry in base-COM frame. */
  Eigen::Isometry3d GetLocomotionControlPointsIsometryInBody(int cp_idx);
  /** @brief Get planar base-to-foot offsets (z forced to zero). */
  std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>
  GetBaseToFootXYOffset();

  /** @brief Get centroidal inertia. */
  Eigen::Matrix<double, 6, 6> GetIg() const;
  /** @brief Get centroidal momentum. */
  Eigen::Matrix<double, 6, 1> GetHg() const;
  /** @brief Get centroidal momentum matrix. */
  Eigen::Matrix<double, 6, Eigen::Dynamic> GetAg() const;

  /** @brief Get number of generalized velocities. */
  int NumQdot() const;
  /** @brief Get number of actuated DOFs. */
  int NumActiveDof() const;
  /** @brief Get number of floating DOFs. */
  int NumFloatDof() const;

  Eigen::Matrix<double, Eigen::Dynamic, 2> JointPosLimits() const {
    return joint_pos_limits_;
  }
  Eigen::Matrix<double, Eigen::Dynamic, 2> JointVelLimits() const {
    return joint_vel_limits_;
  }
  Eigen::Matrix<double, Eigen::Dynamic, 2> JointTrqLimits() const {
    return joint_trq_limits_;
  }

  std::unordered_map<std::string, int> GetJointNameAndIndexMap() const {
    return joint_name_idx_map_;
  }
  std::unordered_map<std::string, int> GetActuatorNameAndIndexMap() const {
    return actuator_name_idx_map_;
  }
  Eigen::Vector3d GetBaseLocalComPos() override { return base_local_com_pos_; }

  /**
   * @brief Set left/right locomotion control point frame names.
   * @param lfoot_cp_string Left foot control point frame name.
   * @param rfoot_cp_string Right foot control point frame name.
   */
  void SetFeetControlPoint(const std::string& lfoot_cp_string,
                           const std::string& rfoot_cp_string) {
    foot_cp_string_vec_.clear();
    lfoot_cp_string_ = lfoot_cp_string;
    rfoot_cp_string_ = rfoot_cp_string;
    foot_cp_string_vec_.push_back(lfoot_cp_string_);
    foot_cp_string_vec_.push_back(rfoot_cp_string_);
  }

  // RPC-style state update overloads
  /**
   * @brief Update model from base joint state and joint vectors.
   * @param base_joint_pos Base-joint position in world frame.
   * @param base_joint_quat Base-joint orientation.
   * @param base_joint_lin_vel Base-joint linear velocity in world frame.
   * @param base_joint_ang_vel Base-joint angular velocity in world frame.
   * @param joint_pos Actuated joint positions.
   * @param joint_vel Actuated joint velocities.
   * @param update_centroid If true, refresh centroidal quantities.
   */
  void UpdateRobotModel(const Eigen::Vector3d& base_joint_pos,
                        const Eigen::Quaterniond& base_joint_quat,
                        const Eigen::Vector3d& base_joint_lin_vel,
                        const Eigen::Vector3d& base_joint_ang_vel,
                        const Eigen::VectorXd& joint_pos,
                        const Eigen::VectorXd& joint_vel,
                        bool update_centroid = false);

  /**
   * @brief Legacy overload with both base COM and base joint arguments.
   *
   * The implementation uses base-joint quantities for model update.
   */
  void UpdateRobotModel(
      const Eigen::Vector3d& base_com_pos,
      const Eigen::Quaternion<double>& base_com_quat,
      const Eigen::Vector3d& base_com_lin_vel,
      const Eigen::Vector3d& base_com_ang_vel,
      const Eigen::Vector3d& base_joint_pos,
      const Eigen::Quaternion<double>& base_joint_quat,
      const Eigen::Vector3d& base_joint_lin_vel,
      const Eigen::Vector3d& base_joint_ang_vel,
      const std::map<std::string, double>& joint_positions,
      const std::map<std::string, double>& joint_velocities,
      const bool update_centroid = false) override;

protected:
  void Initialize() override;
  void UpdateCentroidalQuantities() override;

private:
  void _Initialize() { Initialize(); }
  void _InitializeRootFrame() { InitializeRootFrame(); }
  void _UpdateCentroidalQuantities() { UpdateCentroidalQuantities(); }
  void _PrintRobotInfo() const { PrintRobotInfo(); }

  void InitializeRootFrame();
  void PrintRobotInfo() const;
  static Eigen::Vector3d QuaternionToEulerZYX(const Eigen::Quaterniond& q);
  static Eigen::Matrix3d SO3FromRPY(double roll, double pitch, double yaw);
  void InvalidateDynamicsCache();
  void EnsureAccelerationKinematics();
  void EnsureMassMatrix();
  void EnsureNonlinearEffects();
  void EnsureGravity();
  void EnsureCoriolis();

  // Pinocchio model, geometry, and data
  pinocchio::Model model_;
  pinocchio::GeometryModel collision_model_;
  pinocchio::GeometryModel visual_model_;
  pinocchio::Data data_;
  pinocchio::GeometryData collision_data_;
  pinocchio::GeometryData visual_data_;

  // URDF info
  std::string urdf_file_;
  std::string package_dir_;

  // Configuration vectors
  Eigen::VectorXd q_;
  Eigen::VectorXd qdot_;
  Eigen::VectorXd zero_qddot_;

  // Index maps (new and legacy)
  std::unordered_map<std::string, int> joint_name_to_q_idx_;
  std::unordered_map<std::string, int> joint_name_to_qdot_idx_;
  std::unordered_map<std::string, int> joint_name_to_actuated_idx_;
  std::unordered_map<std::string, int> link_name_to_frame_idx_;
  std::unordered_map<std::string, int> joint_name_idx_map_;
  std::unordered_map<std::string, int> actuator_name_idx_map_;
  std::map<int, std::string> joint_idx_map_;
  std::map<int, std::string> link_idx_map_;

  // Joint info
  std::vector<std::string> actuated_joint_names_;
  double total_mass_legacy_{0.0};
  std::string root_frame_name_;
  Eigen::Vector3d base_local_com_pos_{Eigen::Vector3d::Zero()};
  std::string lfoot_cp_string_;
  std::string rfoot_cp_string_;
  std::vector<std::string> foot_cp_string_vec_;

  bool acceleration_kinematics_valid_{false};
  bool mass_matrix_valid_{false};
  bool nonlinear_effects_valid_{false};
  bool gravity_valid_{false};
  bool coriolis_valid_{false};
  Eigen::VectorXd nonlinear_effects_cache_;
  Eigen::VectorXd gravity_cache_;
  Eigen::VectorXd coriolis_cache_;
};

} // namespace wbc
