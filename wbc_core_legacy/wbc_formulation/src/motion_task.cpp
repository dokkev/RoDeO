/**
 * @file wbc_core/wbc_formulation/src/motion_task.cpp
 * @brief Doxygen documentation for motion_task module.
 */
#include "wbc_formulation/motion_task.hpp"
#include "wbc_formulation/se3_math.hpp"

#include <cassert>
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
void Task::SetParameters(const TaskConfig& config) {
  kp_ = config.kp;
  kd_ = config.kd;
  if (config.ki.size() == dim_) {
    ki_ = config.ki;
  }
  if (config.weight.size() == dim_) {
    weight_ = config.weight;
  }
  if (config.kp_ik.size() == dim_) {
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
  const Eigen::Isometry3d& T_target = robot_->GetLinkIsometry(target_idx_);
  pos_ = T_target.translation();
  vel_ = robot_->GetLinkSpatialVel(target_idx_).tail(dim_);

  const bool has_ref = (ref_frame_idx_ >= 0);
  Eigen::Matrix3d R;

  if (has_ref) {
    const Eigen::Isometry3d& T_ref = robot_->GetLinkIsometry(ref_frame_idx_);
    const Eigen::Matrix<double, 6, 1> V_ref =
        robot_->GetLinkSpatialVel(ref_frame_idx_);
    const Eigen::Vector3d w_ref_world = V_ref.head<3>();
    const Eigen::Vector3d v_ref_world = V_ref.tail<3>();

    R = T_ref.linear();

    local_pos_ = se3::RelativePositionInFrame(T_ref, pos_);
    local_vel_ = se3::RelativeLinearVelocityInFrame(
        T_ref, v_ref_world, w_ref_world, pos_, vel_);

    local_des_pos_ = des_pos_;
    local_des_vel_ = des_vel_;
    local_des_acc_ = des_acc_;
  } else {
    R = world_R_local;

    local_pos_.noalias()     = R.transpose() * pos_;
    local_vel_.noalias()     = R.transpose() * vel_;
    local_des_pos_.noalias() = R.transpose() * des_pos_;
    local_des_vel_.noalias() = R.transpose() * des_vel_;
    local_des_acc_.noalias() = R.transpose() * des_acc_;
  }

  local_pos_err_ = local_des_pos_ - local_pos_;
  local_vel_err_ = local_des_vel_ - local_vel_;

  pos_err_.noalias() = R * local_pos_err_;
  vel_err_.noalias() = R * local_vel_err_;

  op_cmd_.noalias() =
      R * (local_des_acc_ +
           kp_.cwiseProduct(local_pos_err_) +
           kd_.cwiseProduct(local_vel_err_));

  assert(pos_.allFinite()           && "LinkPosTask: pos_ is not finite");
  assert(vel_.allFinite()           && "LinkPosTask: vel_ is not finite");
  assert(local_pos_.allFinite()     && "LinkPosTask: local_pos_ is not finite");
  assert(local_vel_.allFinite()     && "LinkPosTask: local_vel_ is not finite");
  assert(pos_err_.allFinite()       && "LinkPosTask: pos_err_ is not finite");
  assert(vel_err_.allFinite()       && "LinkPosTask: vel_err_ is not finite");
  assert(local_pos_err_.allFinite() && "LinkPosTask: local_pos_err_ is not finite");
  assert(local_vel_err_.allFinite() && "LinkPosTask: local_vel_err_ is not finite");
  assert(op_cmd_.allFinite()        && "LinkPosTask: op_cmd_ is not finite");
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
  des_pos_.resize(4);       des_pos_       = se3::QuatIdentityXyzw();
  pos_.resize(4);           pos_           = se3::QuatIdentityXyzw();
  local_des_pos_.resize(4); local_des_pos_ = se3::QuatIdentityXyzw();
  local_pos_.resize(4);     local_pos_     = se3::QuatIdentityXyzw();
}

////////////////////////////////////////////////////////////////////////////////
void LinkOriTask::UpdateDesired(const Eigen::VectorXd& des_pos,
                                const Eigen::VectorXd& des_vel,
                                const Eigen::VectorXd& des_acc) {
  if (des_pos.size() != 4 || des_vel.size() != 3 || des_acc.size() != 3) {
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

  const Eigen::Vector3d w_target_world =
      robot_->GetLinkSpatialVel(target_idx_).head<3>();

  const bool has_ref = (ref_frame_idx_ >= 0);
  Eigen::Matrix3d R_ref_world;
  if (has_ref) {
    R_ref_world = robot_->GetLinkIsometry(ref_frame_idx_).linear();
  } else {
    R_ref_world = world_R_local;
  }

  // des_pos_ stores quaternion as [x, y, z, w].
  // With ref frame: quaternion is relative to reference frame.
  // Without: quaternion is in world frame.
  const Eigen::Quaterniond des_quat = se3::QuatFromXyzw(des_pos_.head<4>());
  const Eigen::Matrix3d R_des_world =
      has_ref ? (R_ref_world * des_quat.toRotationMatrix())
              : des_quat.toRotationMatrix();

  // Logging / compatibility: current and desired quaternions.
  pos_ = se3::QuatToXyzw(Eigen::Quaterniond(R_cur_world));
  local_pos_ = se3::QuatToXyzw(
      Eigen::Quaterniond(se3::RelativeOrientationInFrame(R_ref_world, R_cur_world)));
  local_des_pos_ = se3::QuatToXyzw(
      Eigen::Quaterniond(se3::RelativeOrientationInFrame(R_ref_world, R_des_world)));

  // Orientation error via SO(3) log map in world frame.
  pos_err_       = se3::RotationErrorWorld(R_cur_world, R_des_world);
  local_pos_err_.noalias() = R_ref_world.transpose() * pos_err_;

  vel_ = w_target_world;

  if (has_ref) {
    const Eigen::Vector3d w_ref_world =
        robot_->GetLinkSpatialVel(ref_frame_idx_).head<3>();
    local_vel_ = se3::RelativeAngularVelocityInFrame(
        R_ref_world, w_ref_world, w_target_world);
    local_des_vel_ = des_vel_;
    local_des_acc_ = des_acc_;
  } else {
    local_vel_.noalias()     = R_ref_world.transpose() * vel_;
    local_des_vel_.noalias() = R_ref_world.transpose() * des_vel_;
    local_des_acc_.noalias() = R_ref_world.transpose() * des_acc_;
  }

  local_vel_err_ = local_des_vel_ - local_vel_;
  vel_err_.noalias() = R_ref_world * local_vel_err_;

  op_cmd_.noalias() =
      R_ref_world * (local_des_acc_ +
                     kp_.cwiseProduct(local_pos_err_) +
                     kd_.cwiseProduct(local_vel_err_));

  assert(pos_.allFinite()           && "LinkOriTask: pos_ is not finite");
  assert(vel_.allFinite()           && "LinkOriTask: vel_ is not finite");
  assert(local_pos_.allFinite()     && "LinkOriTask: local_pos_ is not finite");
  assert(local_vel_.allFinite()     && "LinkOriTask: local_vel_ is not finite");
  assert(pos_err_.allFinite()       && "LinkOriTask: pos_err_ is not finite");
  assert(vel_err_.allFinite()       && "LinkOriTask: vel_err_ is not finite");
  assert(local_pos_err_.allFinite() && "LinkOriTask: local_pos_err_ is not finite");
  assert(local_vel_err_.allFinite() && "LinkOriTask: local_vel_err_ is not finite");
  assert(op_cmd_.allFinite()        && "LinkOriTask: op_cmd_ is not finite");
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
