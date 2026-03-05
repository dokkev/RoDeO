/**
 * @file wbc_core/wbc_formulation/include/wbc_formulation/motion_task.hpp
 * @brief Doxygen documentation for motion_task module.
 */
#pragma once

#include "wbc_formulation/interface/task.hpp"
#include "wbc_robot_system/pinocchio_robot_system.hpp"

#include <vector>

namespace wbc {
/**
 * @brief Full-joint posture tracking task.
 */
class JointTask : public Task {
public:
  explicit JointTask(PinocchioRobotSystem* robot);
  ~JointTask() override = default;

  void UpdateOpCommand(
      const Eigen::Matrix3d& world_R_local = Eigen::Matrix3d::Identity()) override;
  void UpdateJacobian() override;
  void UpdateJacobianDotQdot() override;
};

/**
 * @brief Subset-joint posture tracking task.
 */
class SelectedJointTask : public Task {
public:
  SelectedJointTask(PinocchioRobotSystem* robot,
                    const std::vector<int>& joint_idx_container);
  ~SelectedJointTask() override = default;

  void UpdateOpCommand(
      const Eigen::Matrix3d& world_R_local = Eigen::Matrix3d::Identity()) override;
  void UpdateJacobian() override;
  void UpdateJacobianDotQdot() override;

  const std::vector<int>& JointIdxContainer() const { return joint_idx_container_; }

private:
  std::vector<int> joint_idx_container_;
};

/**
 * @brief Cartesian position task for a target link frame.
 */
class LinkPosTask : public Task {
public:
  LinkPosTask(PinocchioRobotSystem* robot, int target_idx);
  ~LinkPosTask() override = default;

  void UpdateOpCommand(
      const Eigen::Matrix3d& world_R_local = Eigen::Matrix3d::Identity()) override;
  void UpdateJacobian() override;
  void UpdateJacobianDotQdot() override;

private:
  Eigen::MatrixXd full_jac_scratch_;  ///< Pre-allocated 6 x num_qdot buffer
};

/**
 * @brief Cartesian orientation task for a target link frame.
 *
 * @note des_pos must be a 4-element quaternion [x, y, z, w]. Call the
 *       UpdateDesired override to enforce this contract at runtime.
 */
class LinkOriTask : public Task {
public:
  LinkOriTask(PinocchioRobotSystem* robot, int target_idx);
  ~LinkOriTask() override = default;

  /// Validates that des_pos has size 4 (quaternion [x,y,z,w]), then delegates to base.
  void UpdateDesired(const Eigen::VectorXd& des_pos, const Eigen::VectorXd& des_vel,
                     const Eigen::VectorXd& des_acc) override;

  void UpdateOpCommand(
      const Eigen::Matrix3d& world_R_local = Eigen::Matrix3d::Identity()) override;
  void UpdateJacobian() override;
  void UpdateJacobianDotQdot() override;

private:
  Eigen::MatrixXd full_jac_scratch_;  ///< Pre-allocated 6 x num_qdot buffer
};

/**
 * @brief Center-of-mass translation task.
 */
class ComTask : public Task {
public:
  explicit ComTask(PinocchioRobotSystem* robot);
  ~ComTask() override = default;

  void UpdateOpCommand(
      const Eigen::Matrix3d& world_R_local = Eigen::Matrix3d::Identity()) override;
  void UpdateJacobian() override;
  void UpdateJacobianDotQdot() override;
};

} // namespace wbc
