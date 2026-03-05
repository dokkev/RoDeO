/**
 * @file wbc_core/wbc_robot_system/include/wbc_robot_system/interface/robot_system.hpp
 * @brief Doxygen documentation for robot_system module.
 */
#pragma once

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <map>
#include <string>

namespace wbc {
/**
 * @brief Abstract interface for robot kinematic and dynamic models
 *
 * Provides access to:
 * - Robot configuration (DOF, mass, limits)
 * - Forward kinematics (link poses, velocities)
 * - Jacobians and derivatives
 * - Dynamics (mass matrix, gravity, coriolis)
 * - Centroidal momentum quantities
 */
class RobotSystem {
public:
  explicit RobotSystem(bool fixed_base, bool print_info = false)
      : fixed_base_(fixed_base),
        print_info_(print_info),
        num_floating_dof_(fixed_base ? 0 : 6),
        num_q_(0),
        num_qdot_(0),
        num_actuated_(0),
        total_mass_(0.0) {
    centroidal_inertia_.setZero();
    centroidal_momentum_matrix_.resize(6, 0);
    centroidal_momentum_.setZero();
  }

  virtual ~RobotSystem() = default;

  // ========================================================================
  // Configuration Queries
  // ========================================================================

  int GetNumFloatingDOF() const { return num_floating_dof_; }
  int GetNumQ() const { return num_q_; }
  int GetNumQdot() const { return num_qdot_; }
  int GetNumActuated() const { return num_actuated_; }
  double GetTotalMass() const { return total_mass_; }
  bool IsFixedBase() const { return fixed_base_; }
  bool ShouldPrintInfo() const { return print_info_; }

  const Eigen::MatrixXd& GetJointPositionLimits() const {
    return joint_position_limits_;
  }
  const Eigen::MatrixXd& GetJointVelocityLimits() const {
    return joint_velocity_limits_;
  }
  const Eigen::MatrixXd& GetJointTorqueLimits() const {
    return joint_torque_limits_;
  }

  // Soft limits (YAML-scaled operational limits; default = hard URDF limits).
  const Eigen::MatrixXd& SoftPositionLimits() const { return soft_position_limits_; }
  const Eigen::MatrixXd& SoftVelocityLimits() const { return soft_velocity_limits_; }
  const Eigen::MatrixXd& SoftTorqueLimits() const { return soft_torque_limits_; }
  void SetSoftPositionLimits(const Eigen::MatrixXd& limits) { soft_position_limits_ = limits; }
  void SetSoftVelocityLimits(const Eigen::MatrixXd& limits) { soft_velocity_limits_ = limits; }
  void SetSoftTorqueLimits(const Eigen::MatrixXd& limits) { soft_torque_limits_ = limits; }

  // ========================================================================
  // Index Queries (pure virtual - must be implemented)
  // ========================================================================

  virtual int GetQIndex(const std::string& joint_name) const = 0;
  virtual int GetQdotIndex(const std::string& joint_name) const = 0;
  virtual int GetActuatedIndex(const std::string& joint_name) const = 0;
  virtual std::string GetBaseLinkName() const = 0;

  // ========================================================================
  // State Update (pure virtual)
  // ========================================================================

  /**
   * @brief Update robot model with current state
   * @param base_pos Base position [x, y, z] (ignored if fixed base)
   * @param base_quat Base orientation [w, x, y, z]
   * @param base_lin_vel Base linear velocity
   * @param base_ang_vel Base angular velocity
   * @param joint_positions Map of joint_name -> position
   * @param joint_velocities Map of joint_name -> velocity
   * @param update_centroidal If true, compute centroidal quantities (expensive)
   */
  virtual void UpdateState(const Eigen::Vector3d& base_pos,
                           const Eigen::Quaterniond& base_quat,
                           const Eigen::Vector3d& base_lin_vel,
                           const Eigen::Vector3d& base_ang_vel,
                           const std::map<std::string, double>& joint_positions,
                           const std::map<std::string, double>& joint_velocities,
                           bool update_centroidal = false) = 0;

  // ========================================================================
  // State Access (pure virtual)
  // ========================================================================

  virtual Eigen::VectorXd GetQ() const = 0;
  virtual Eigen::VectorXd GetQdot() const = 0;

  // ========================================================================
  // Dynamics (pure virtual)
  // ========================================================================

  virtual Eigen::MatrixXd GetMassMatrix() = 0;
  virtual Eigen::VectorXd GetGravity() = 0;
  virtual Eigen::VectorXd GetCoriolis() = 0;

  // ========================================================================
  // Center of Mass Kinematics (pure virtual)
  // ========================================================================

  virtual Eigen::Vector3d GetComPosition() = 0;
  virtual Eigen::Vector3d GetComVelocity() = 0;
  virtual Eigen::Matrix3Xd GetComJacobian() = 0;
  virtual Eigen::Matrix3Xd GetComJacobianDot() = 0;

  // ========================================================================
  // Link Kinematics (pure virtual)
  // ========================================================================

  virtual Eigen::Isometry3d GetLinkTransform(const std::string& link_name) = 0;
  virtual Eigen::Matrix<double, 6, 1>
  GetLinkVelocity(const std::string& link_name) = 0;
  virtual Eigen::Matrix<double, 6, Eigen::Dynamic>
  GetLinkJacobian(const std::string& link_name) = 0;
  virtual Eigen::Matrix<double, 6, 1>
  GetLinkJacobianDotQdot(const std::string& link_name) = 0;

  // ========================================================================
  // Centroidal Momentum (computed when update_centroidal = true)
  // ========================================================================

  const Eigen::Matrix<double, 6, 6>& GetCentroidalInertia() const {
    return centroidal_inertia_;
  }
  const Eigen::Matrix<double, 6, Eigen::Dynamic>&
  GetCentroidalMomentumMatrix() const {
    return centroidal_momentum_matrix_;
  }
  const Eigen::Matrix<double, 6, 1>& GetCentroidalMomentum() const {
    return centroidal_momentum_;
  }

  // ========================================================================
  // Utility Functions (pure virtual)
  // ========================================================================

  virtual std::map<std::string, double>
  VectorToJointMap(const Eigen::VectorXd& joint_vector) const = 0;

  virtual Eigen::VectorXd
  JointMapToVector(const std::map<std::string, double>& joint_map) const = 0;

  // ========================================================================
  // Legacy RPC API compatibility
  // ========================================================================
  [[deprecated("Use GetQIndex().")]]
  int GetQIdx(const std::string& joint_name) { return GetQIndex(joint_name); }
  [[deprecated("Use GetQdotIndex().")]]
  int GetQdotIdx(const std::string& joint_name) {
    return GetQdotIndex(joint_name);
  }
  [[deprecated("Use GetActuatedIndex().")]]
  int GetJointIdx(const std::string& joint_name) {
    return GetActuatedIndex(joint_name);
  }

  [[deprecated("Use VectorToJointMap().")]]
  std::map<std::string, double>
  EigenVectorToMap(const Eigen::VectorXd& joint_cmd) {
    return VectorToJointMap(joint_cmd);
  }
  [[deprecated("Use JointMapToVector().")]]
  Eigen::VectorXd MapToEigenVector(std::map<std::string, double> joint_map) {
    return JointMapToVector(joint_map);
  }

  virtual Eigen::Vector3d GetBaseLocalComPos() { return Eigen::Vector3d::Zero(); }

  virtual void UpdateRobotModel(
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
      const bool update_centroid = false) {
    UpdateState(base_joint_pos, base_joint_quat, base_joint_lin_vel,
                base_joint_ang_vel, joint_positions, joint_velocities,
                update_centroid);
  }

  [[deprecated("Use GetComPosition().")]]
  Eigen::Vector3d GetRobotComPos() { return GetComPosition(); }
  [[deprecated("Use GetComVelocity().")]]
  Eigen::Vector3d GetRobotComLinVel() { return GetComVelocity(); }

  [[deprecated("Use GetLinkTransform().")]]
  Eigen::Isometry3d GetLinkIso(const std::string& link_name) {
    return GetLinkTransform(link_name);
  }
  [[deprecated("Use GetLinkVelocity().")]]
  Eigen::Matrix<double, 6, 1> GetLinkVel(const std::string& link_name) {
    return GetLinkVelocity(link_name);
  }

  [[deprecated("Use GetComJacobian().")]]
  Eigen::Matrix<double, 3, Eigen::Dynamic> GetComLinJacobian() {
    return GetComJacobian();
  }
  [[deprecated("Use GetComJacobianDot().")]]
  Eigen::Matrix<double, 3, Eigen::Dynamic> GetComLinJacobianDot() {
    return GetComJacobianDot();
  }

protected:
  bool fixed_base_;
  bool print_info_;

  // Dimensions
  int num_floating_dof_;
  int num_q_;
  int num_qdot_;
  int num_actuated_;

  // Robot properties
  double total_mass_;

  // Joint limits (num_actuated x 2: [min, max])
  Eigen::MatrixXd joint_position_limits_;   // hard (URDF)
  Eigen::MatrixXd joint_velocity_limits_;   // hard (URDF)
  Eigen::MatrixXd joint_torque_limits_;     // hard (URDF)
  Eigen::MatrixXd soft_position_limits_;    // operational (YAML-scaled; defaults to hard)
  Eigen::MatrixXd soft_velocity_limits_;
  Eigen::MatrixXd soft_torque_limits_;

  // Centroidal quantities
  Eigen::Matrix<double, 6, 6> centroidal_inertia_;
  Eigen::Matrix<double, 6, Eigen::Dynamic> centroidal_momentum_matrix_;
  Eigen::Matrix<double, 6, 1> centroidal_momentum_;

public:
  Eigen::VectorXd joint_positions_;
  Eigen::VectorXd joint_velocities_;

  // To be implemented by derived classes
  virtual void ConfigRobot() { Initialize(); }
  virtual void Initialize() = 0;
  virtual void UpdateCentroidalQuantities() = 0;
};

} // namespace wbc
