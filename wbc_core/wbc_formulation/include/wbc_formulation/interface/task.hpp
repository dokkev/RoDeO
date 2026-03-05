/**
 * @file wbc_core/wbc_formulation/include/wbc_formulation/interface/task.hpp
 * @brief Doxygen documentation for task module.
 */
#pragma once

#include <Eigen/Dense>

#include <vector>

namespace wbc {
class PinocchioRobotSystem;
class Task;

/**
 * @brief WBC backend type for task-parameter interpretation.
 */
enum class WbcType { IHWBC, WBIC };

/**
 * @brief Typed task parameters (YAML-free).
 */
struct TaskConfig {
  Eigen::VectorXd kp;
  Eigen::VectorXd kd;
  Eigen::VectorXd ki;
  Eigen::VectorXd weight;
  Eigen::VectorXd kp_ik;

  /**
   * @brief Create zero-initialized task config for a given dimension.
   */
  static TaskConfig Defaults(int dim);

  /**
   * @brief Snapshot task's current gains/weights into typed config.
   */
  static TaskConfig FromTask(const Task& task);
};

/**
 * @brief Abstract operational-space task.
 */
class Task {
public:
  Task(PinocchioRobotSystem* robot, int dim);
  virtual ~Task() = default;

  /**
   * @brief Update desired task-space references.
   * @param des_pos Desired position/state. For LinkOriTask, must be size 4 (quaternion xyzw).
   * @param des_vel Desired velocity.
   * @param des_acc Desired acceleration.
   */
  virtual void UpdateDesired(const Eigen::VectorXd& des_pos,
                             const Eigen::VectorXd& des_vel,
                             const Eigen::VectorXd& des_acc);

  virtual void UpdateOpCommand(
      const Eigen::Matrix3d& world_R_local = Eigen::Matrix3d::Identity()) = 0;
  virtual void UpdateJacobian() = 0;
  virtual void UpdateJacobianDotQdot() = 0;

  virtual void SetParameters(const TaskConfig& config, WbcType wbc_type);

  void ModifyJacobian(const std::vector<int>& joint_idx, int num_float = 0);

  const Eigen::VectorXd& DesiredPos() const { return des_pos_; }
  const Eigen::VectorXd& DesiredVel() const { return des_vel_; }
  const Eigen::VectorXd& DesiredAcc() const { return des_acc_; }
  const Eigen::VectorXd& DesiredLocalPos() const { return local_des_pos_; }
  const Eigen::VectorXd& DesiredLocalVel() const { return local_des_vel_; }
  const Eigen::VectorXd& DesiredLocalAcc() const { return local_des_acc_; }
  const Eigen::VectorXd& CurrentPos() const { return pos_; }
  const Eigen::VectorXd& CurrentVel() const { return vel_; }
  const Eigen::VectorXd& CurrentLocalPos() const { return local_pos_; }
  const Eigen::VectorXd& CurrentLocalVel() const { return local_vel_; }
  const Eigen::MatrixXd& Jacobian() const { return jacobian_; }
  const Eigen::VectorXd& JacobianDotQdot() const { return jacobian_dot_q_dot_; }
  const Eigen::VectorXd& Weight() const { return weight_; }
  const Eigen::VectorXd& Kp() const { return kp_; }
  const Eigen::VectorXd& Kd() const { return kd_; }
  const Eigen::VectorXd& Ki() const { return ki_; }
  const Eigen::VectorXd& KpIK() const { return kp_ik_; }
  const Eigen::VectorXd& OpCommand() const { return op_cmd_; }
  int Dim() const { return dim_; }
  const Eigen::VectorXd& PosError() const { return pos_err_; }
  const Eigen::VectorXd& LocalPosError() const { return local_pos_err_; }
  int TargetIdx() const { return target_idx_; }
  int ReferenceFrameIdx() const { return ref_frame_idx_; }
  void SetReferenceFrameIdx(int idx) { ref_frame_idx_ = idx; }
  const Eigen::Matrix3d& Rot() const { return local_R_world_; }

  void SetWeight(const Eigen::VectorXd& weight) { weight_ = weight; }
  void SetKp(const Eigen::VectorXd& kp) { kp_ = kp; }
  void SetKd(const Eigen::VectorXd& kd) { kd_ = kd; }
  void SetKi(const Eigen::VectorXd& ki) { ki_ = ki; }
  void SetKpIK(const Eigen::VectorXd& kp_ik) { kp_ik_ = kp_ik; }

protected:
  PinocchioRobotSystem* robot_;
  int dim_;
  int target_idx_;
  int ref_frame_idx_{-1};

  Eigen::Matrix3d local_R_world_;

  Eigen::VectorXd pos_;
  Eigen::VectorXd vel_;
  Eigen::VectorXd local_pos_;
  Eigen::VectorXd local_vel_;
  Eigen::VectorXd pos_err_;
  Eigen::VectorXd vel_err_;
  Eigen::VectorXd local_pos_err_;
  Eigen::VectorXd local_vel_err_;

  Eigen::VectorXd kp_;
  Eigen::VectorXd kd_;
  Eigen::VectorXd ki_;  // Parsed from YAML but unused in any UpdateOpCommand (reserved for future integral action).
  Eigen::VectorXd kp_ik_;

  // Note: des_pos_ / pos_ are sized dim_ for all tasks except LinkOriTask,
  // where they are resized to 4 to store a quaternion [x,y,z,w].
  // des_pos_.size() != dim_ is intentional for orientation tracking.
  Eigen::VectorXd des_pos_;
  Eigen::VectorXd des_vel_;
  Eigen::VectorXd des_acc_;
  Eigen::VectorXd local_des_pos_;
  Eigen::VectorXd local_des_vel_;
  Eigen::VectorXd local_des_acc_;

  // Copy world-frame state into local-frame members (use in tasks where world_R_local == I).
  void SyncLocalToWorld() {
    local_des_pos_ = des_pos_;
    local_des_vel_ = des_vel_;
    local_des_acc_ = des_acc_;
    local_pos_     = pos_;
    local_vel_     = vel_;
    local_pos_err_ = pos_err_;
    local_vel_err_ = vel_err_;
  }

  Eigen::VectorXd op_cmd_;
  Eigen::MatrixXd jacobian_;
  Eigen::VectorXd jacobian_dot_q_dot_;
  Eigen::VectorXd weight_;
};

} // namespace wbc
