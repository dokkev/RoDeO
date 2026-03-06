/**
 * @file wbc_core/wbc_formulation/src/motion_task.cpp
 * @brief Doxygen documentation for motion_task module.
 */
#include "wbc_formulation/motion_task.hpp"

#include <pinocchio/spatial/explog.hpp>
#include <stdexcept>

namespace wbc {
////////////////////////////////////////////////////////////////////////////////
TaskConfig TaskConfig::Defaults(int dim) {
  TaskConfig config;
  config.kp = Eigen::VectorXd::Zero(dim);
  config.kd = Eigen::VectorXd::Zero(dim);
  config.ki = Eigen::VectorXd::Zero(dim);
  config.weight = Eigen::VectorXd::Zero(dim);
  config.kp_ik = Eigen::VectorXd::Zero(dim);
  return config;
}

////////////////////////////////////////////////////////////////////////////////
TaskConfig TaskConfig::FromTask(const Task& task) {
  TaskConfig config = TaskConfig::Defaults(task.Dim());
  config.kp = task.Kp();
  config.kd = task.Kd();
  config.ki = task.Ki();
  config.weight = task.Weight();
  config.kp_ik = task.KpIK();
  return config;
}

////////////////////////////////////////////////////////////////////////////////
Task::Task(PinocchioRobotSystem* robot, int dim)
    : robot_(robot),
      dim_(dim),
      target_idx_(0),
      local_R_world_(Eigen::Matrix3d::Identity()),
      pos_(Eigen::VectorXd::Zero(dim)),
      vel_(Eigen::VectorXd::Zero(dim)),
      local_pos_(Eigen::VectorXd::Zero(dim)),
      local_vel_(Eigen::VectorXd::Zero(dim)),
      pos_err_(Eigen::VectorXd::Zero(dim)),
      vel_err_(Eigen::VectorXd::Zero(dim)),
      local_pos_err_(Eigen::VectorXd::Zero(dim)),
      local_vel_err_(Eigen::VectorXd::Zero(dim)),
      kp_(Eigen::VectorXd::Zero(dim)),
      kd_(Eigen::VectorXd::Zero(dim)),
      ki_(Eigen::VectorXd::Zero(dim)),
      kp_ik_(Eigen::VectorXd::Zero(dim)),
      des_pos_(Eigen::VectorXd::Zero(dim)),
      des_vel_(Eigen::VectorXd::Zero(dim)),
      des_acc_(Eigen::VectorXd::Zero(dim)),
      local_des_pos_(Eigen::VectorXd::Zero(dim)),
      local_des_vel_(Eigen::VectorXd::Zero(dim)),
      local_des_acc_(Eigen::VectorXd::Zero(dim)),
      op_cmd_(Eigen::VectorXd::Zero(dim)),
      jacobian_(Eigen::MatrixXd::Zero(dim, robot->NumQdot())),
      jacobian_dot_q_dot_(Eigen::VectorXd::Zero(dim)),
      weight_(Eigen::VectorXd::Zero(dim)) {}

////////////////////////////////////////////////////////////////////////////////
void Task::UpdateDesired(const Eigen::VectorXd& des_pos,
                         const Eigen::VectorXd& des_vel,
                         const Eigen::VectorXd& des_acc) {
  des_pos_ = des_pos;
  des_vel_ = des_vel;
  des_acc_ = des_acc;
}

////////////////////////////////////////////////////////////////////////////////
void Task::SetParameters(const TaskConfig& config, WbcType wbc_type) {
  kp_ = config.kp;
  kd_ = config.kd;
  if (config.ki.size() == dim_) {
    ki_ = config.ki;
  }

  if (wbc_type == WbcType::IHWBC) {
    weight_ = config.weight;
  } else if (wbc_type == WbcType::WBIC) {
    kp_ik_ = config.kp_ik;
  }
}

////////////////////////////////////////////////////////////////////////////////
void Task::ModifyJacobian(const std::vector<int>& joint_idx, int num_float) {
  for (int idx : joint_idx) {
    jacobian_.col(num_float + idx).setZero();
  }
}

////////////////////////////////////////////////////////////////////////////////
JointTask::JointTask(PinocchioRobotSystem* robot)
    : Task(robot, robot->NumActiveDof()) {
  // Static Jacobian: [0 | I] — set once, never changes.
  jacobian_.setZero();
  jacobian_.block(0, robot_->NumFloatDof(), dim_, robot_->NumActiveDof()).setIdentity();
}

////////////////////////////////////////////////////////////////////////////////
void JointTask::UpdateOpCommand(const Eigen::Matrix3d& /*world_R_local*/) {
  pos_     = robot_->GetJointPos();
  vel_     = robot_->GetJointVel();
  pos_err_ = des_pos_ - pos_;
  vel_err_ = des_vel_ - vel_;
  SyncLocalToWorld();  // Joint-space: no local frame, keep local_* accessors consistent.
  op_cmd_  = des_acc_ + kp_.cwiseProduct(pos_err_) + kd_.cwiseProduct(vel_err_);
}

////////////////////////////////////////////////////////////////////////////////
void JointTask::UpdateJacobian() {
  // Jacobian is static [0|I], set once in constructor.
}

////////////////////////////////////////////////////////////////////////////////
void JointTask::UpdateJacobianDotQdot() {
  jacobian_dot_q_dot_.setZero();
}

////////////////////////////////////////////////////////////////////////////////
SelectedJointTask::SelectedJointTask(
    PinocchioRobotSystem* robot, const std::vector<int>& joint_idx_container)
    : Task(robot, static_cast<int>(joint_idx_container.size())),
      joint_idx_container_(joint_idx_container) {
  // Static sparse-identity Jacobian: set once, never changes.
  jacobian_.setZero();
  for (int i = 0; i < dim_; ++i) {
    const int idx = robot_->GetQdotIdx(joint_idx_container_[i]);
    jacobian_(i, idx) = 1.0;
  }
}

////////////////////////////////////////////////////////////////////////////////
void SelectedJointTask::UpdateOpCommand(const Eigen::Matrix3d& /*world_R_local*/) {
  const Eigen::VectorXd& q    = robot_->GetQRef();
  const Eigen::VectorXd& qdot = robot_->GetQdotRef();
  for (int i = 0; i < dim_; ++i) {
    pos_[i]     = q[robot_->GetQIdx(joint_idx_container_[i])];
    vel_[i]     = qdot[robot_->GetQdotIdx(joint_idx_container_[i])];
    pos_err_[i] = des_pos_[i] - pos_[i];
    vel_err_[i] = des_vel_[i] - vel_[i];
    op_cmd_[i]  = des_acc_[i] + kp_[i] * pos_err_[i] + kd_[i] * vel_err_[i];
  }
  SyncLocalToWorld();  // Joint-space: no local frame, keep local_* accessors consistent.
}

////////////////////////////////////////////////////////////////////////////////
void SelectedJointTask::UpdateJacobian() {
  // Jacobian is static sparse-identity, set once in constructor.
}

////////////////////////////////////////////////////////////////////////////////
void SelectedJointTask::UpdateJacobianDotQdot() {
  jacobian_dot_q_dot_.setZero();
}

////////////////////////////////////////////////////////////////////////////////
LinkPosTask::LinkPosTask(PinocchioRobotSystem* robot, int target_idx)
    : Task(robot, 3),
      full_jac_scratch_(Eigen::MatrixXd::Zero(6, robot->NumQdot())) {
  target_idx_ = target_idx;
}

////////////////////////////////////////////////////////////////////////////////
void LinkPosTask::UpdateOpCommand(const Eigen::Matrix3d& world_R_local) {
  pos_ = robot_->GetLinkIsometry(target_idx_).translation();
  vel_ = robot_->GetLinkSpatialVel(target_idx_).tail(dim_);

  // Determine local-frame rotation R and position offset.
  // With ref_frame: R = R_ref, des values are in reference frame.
  // Without: R = world_R_local, des values are in world frame (legacy).
  const bool has_ref = (ref_frame_idx_ >= 0);
  Eigen::Matrix3d R;
  Eigen::Vector3d pos_in_local;

  if (has_ref) {
    const Eigen::Isometry3d& T_ref = robot_->GetLinkIsometry(ref_frame_idx_);
    R = T_ref.linear();
    pos_in_local = R.transpose() * (pos_ - T_ref.translation());
  } else {
    R = world_R_local;
    pos_in_local = R.transpose() * pos_;
  }

  if (has_ref) {
    local_des_pos_ = des_pos_;
    local_des_vel_ = des_vel_;
    local_des_acc_ = des_acc_;
  } else {
    local_des_pos_.noalias() = R.transpose() * des_pos_;
    local_des_vel_.noalias() = R.transpose() * des_vel_;
    local_des_acc_.noalias() = R.transpose() * des_acc_;
  }

  local_pos_ = pos_in_local;
  local_vel_.noalias() = R.transpose() * vel_;
  local_pos_err_ = local_des_pos_ - local_pos_;
  local_vel_err_ = local_des_vel_ - local_vel_;

  pos_err_.noalias() = R * local_pos_err_;
  vel_err_.noalias() = R * local_vel_err_;

  op_cmd_.noalias() = R * (local_des_acc_ + kp_.cwiseProduct(local_pos_err_) +
                           kd_.cwiseProduct(local_vel_err_));
}

////////////////////////////////////////////////////////////////////////////////
void LinkPosTask::UpdateJacobian() {
  robot_->FillLinkJacobian(target_idx_, full_jac_scratch_);
  jacobian_.noalias() = full_jac_scratch_.block(3, 0, dim_, robot_->NumQdot());
}

////////////////////////////////////////////////////////////////////////////////
void LinkPosTask::UpdateJacobianDotQdot() {
  jacobian_dot_q_dot_ = robot_->GetLinkJacobianDotQdot(target_idx_).tail(dim_);
}

////////////////////////////////////////////////////////////////////////////////
LinkOriTask::LinkOriTask(PinocchioRobotSystem* robot, int target_idx)
    : Task(robot, 3),
      full_jac_scratch_(Eigen::MatrixXd::Zero(6, robot->NumQdot())) {
  target_idx_ = target_idx;
  des_pos_.resize(4);
  des_pos_.setZero();
  pos_.resize(4);
  pos_.setZero();
  local_des_pos_.resize(4);
  local_des_pos_.setZero();
  local_pos_.resize(4);
  local_pos_.setZero();
}

////////////////////////////////////////////////////////////////////////////////
void LinkOriTask::UpdateDesired(const Eigen::VectorXd& des_pos,
                                const Eigen::VectorXd& des_vel,
                                const Eigen::VectorXd& des_acc) {
  if (des_pos.size() != 4) {
    // Silently reject: throwing in the RT loop would crash the controller.
    return;
  }
  Task::UpdateDesired(des_pos, des_vel, des_acc);
}

////////////////////////////////////////////////////////////////////////////////
void LinkOriTask::UpdateOpCommand(const Eigen::Matrix3d& world_R_local) {
  const Eigen::Matrix3d R_cur_world =
      robot_->GetLinkIsometry(target_idx_).linear();
  local_R_world_ = R_cur_world.transpose();

  const bool has_ref = (ref_frame_idx_ >= 0);
  const Eigen::Matrix3d R =
      has_ref ? robot_->GetLinkIsometry(ref_frame_idx_).linear()
              : world_R_local;

  // des_pos_ stores quaternion [x,y,z,w].
  // With ref frame: quaternion is relative to reference frame.
  // Without: quaternion is in world frame.
  const Eigen::Quaterniond des_quat(des_pos_[3], des_pos_[0], des_pos_[1],
                                    des_pos_[2]);
  const Eigen::Matrix3d R_des_world =
      has_ref ? (R * des_quat.toRotationMatrix()) : des_quat.toRotationMatrix();

  // Store current quaternion for logging/compatibility.
  const Eigen::Quaterniond cur_quat_world(R_cur_world);
  pos_ << cur_quat_world.normalized().coeffs();
  const Eigen::Quaterniond cur_quat_local(R.transpose() * R_cur_world);
  local_pos_ << cur_quat_local.normalized().coeffs();
  const Eigen::Quaterniond des_quat_local(R.transpose() * R_des_world);
  local_des_pos_ << des_quat_local.normalized().coeffs();

  // Orientation error via Lie group log map: geometrically shortest rotation
  // vector on SO(3), free of gimbal lock and quaternion unwinding artifacts.
  const Eigen::Vector3d ori_err_local =
      pinocchio::log3(R_cur_world.transpose() * R_des_world);
  pos_err_.noalias() = R_cur_world * ori_err_local;
  local_pos_err_.noalias() = R.transpose() * pos_err_;

  vel_ = robot_->GetLinkSpatialVel(target_idx_).head(dim_);
  local_vel_.noalias() = R.transpose() * vel_;

  if (has_ref) {
    local_des_vel_ = des_vel_;
    local_des_acc_ = des_acc_;
  } else {
    local_des_vel_.noalias() = R.transpose() * des_vel_;
    local_des_acc_.noalias() = R.transpose() * des_acc_;
  }
  local_vel_err_ = local_des_vel_ - local_vel_;
  vel_err_.noalias() = R * local_vel_err_;

  op_cmd_.noalias() = R * (local_des_acc_ + kp_.cwiseProduct(local_pos_err_) +
                           kd_.cwiseProduct(local_vel_err_));
}

////////////////////////////////////////////////////////////////////////////////
void LinkOriTask::UpdateJacobian() {
  robot_->FillLinkJacobian(target_idx_, full_jac_scratch_);
  jacobian_.noalias() = full_jac_scratch_.block(0, 0, dim_, robot_->NumQdot());
}

////////////////////////////////////////////////////////////////////////////////
void LinkOriTask::UpdateJacobianDotQdot() {
  jacobian_dot_q_dot_ = robot_->GetLinkJacobianDotQdot(target_idx_).head(dim_);
}

////////////////////////////////////////////////////////////////////////////////
ComTask::ComTask(PinocchioRobotSystem* robot) : Task(robot, 3) {}

////////////////////////////////////////////////////////////////////////////////
void ComTask::UpdateOpCommand(const Eigen::Matrix3d& world_R_local) {
  pos_ = robot_->GetComPosition();
  vel_ = robot_->GetComVelocity();
  pos_err_ = des_pos_ - pos_;
  vel_err_ = des_vel_ - vel_;

  local_des_pos_.noalias() = world_R_local.transpose() * des_pos_;
  local_des_vel_.noalias() = world_R_local.transpose() * des_vel_;
  local_des_acc_.noalias() = world_R_local.transpose() * des_acc_;
  local_pos_.noalias()     = world_R_local.transpose() * pos_;
  local_pos_err_.noalias() = world_R_local.transpose() * pos_err_;
  local_vel_.noalias()     = world_R_local.transpose() * vel_;
  local_vel_err_.noalias() = world_R_local.transpose() * vel_err_;

  op_cmd_.noalias() = des_acc_ + world_R_local * (kp_.cwiseProduct(local_pos_err_) +
                                                   kd_.cwiseProduct(local_vel_err_));
}

////////////////////////////////////////////////////////////////////////////////
void ComTask::UpdateJacobian() {
  robot_->FillComJacobian(jacobian_);
}

////////////////////////////////////////////////////////////////////////////////
void ComTask::UpdateJacobianDotQdot() {
  // GetComJacobianDot() returns Matrix3Xd zeros — assigning that to a VectorXd
  // would be a shape mismatch.  The COM Jdot·qdot term is zero in practice
  // (Pinocchio does not expose an analytic dJcom/dt), so just zero the vector.
  jacobian_dot_q_dot_.setZero();
}

} // namespace wbc
