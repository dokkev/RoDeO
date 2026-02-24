#pragma once

#include "wbc_formulation/interface/task.hpp"
#include "wbc_robot_system/pinocchio_robot_system.hpp"

#include <vector>

namespace wbc {
class JointTask : public Task {
public:
  explicit JointTask(PinocchioRobotSystem* robot);
  ~JointTask() override = default;

  void UpdateOpCommand(
      const Eigen::Matrix3d& world_R_local = Eigen::Matrix3d::Identity()) override;
  void UpdateJacobian() override;
  void UpdateJacobianDotQdot() override;
};

class SelectedJointTask : public Task {
public:
  SelectedJointTask(PinocchioRobotSystem* robot,
                    const std::vector<int>& joint_idx_container);
  ~SelectedJointTask() override = default;

  void UpdateOpCommand(
      const Eigen::Matrix3d& world_R_local = Eigen::Matrix3d::Identity()) override;
  void UpdateJacobian() override;
  void UpdateJacobianDotQdot() override;

  std::vector<int> JointIdxContainer() const { return joint_idx_container_; }

private:
  std::vector<int> joint_idx_container_;
};

class LinkPosTask : public Task {
public:
  LinkPosTask(PinocchioRobotSystem* robot, int target_idx);
  ~LinkPosTask() override = default;

  void UpdateOpCommand(
      const Eigen::Matrix3d& world_R_local = Eigen::Matrix3d::Identity()) override;
  void UpdateJacobian() override;
  void UpdateJacobianDotQdot() override;

private:
  int target_link_idx_;
};

class LinkOriTask : public Task {
public:
  LinkOriTask(PinocchioRobotSystem* robot, int target_idx);
  ~LinkOriTask() override = default;

  void UpdateOpCommand(
      const Eigen::Matrix3d& world_R_local = Eigen::Matrix3d::Identity()) override;
  void UpdateJacobian() override;
  void UpdateJacobianDotQdot() override;

private:
  int target_link_idx_;
  Eigen::Quaterniond des_quat_prev_;
};

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

