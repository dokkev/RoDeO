//
// Copyright (c) 2026
//

#include "wbc_core/architecture/states/cartesian_teleop_state.hpp"

#include <pinocchio/algorithm/frames.hpp>

namespace wbc {

void CartesianTeleopState::SetParameters(const YAML::Node& node) {
  StateMachine::SetParameters(node);

  // Find SE3 tasks and joint posture task
  for (auto& [name, task] : assigned_tasks_) {
    auto se3 = std::dynamic_pointer_cast<tsid::tasks::TaskSE3Equality>(task);
    if (se3) {
      if (name.find("pos") != std::string::npos) {
        ee_pos_task_ = se3;
      } else if (name.find("ori") != std::string::npos) {
        ee_ori_task_ = se3;
      } else if (!ee_pos_task_) {
        ee_pos_task_ = se3;
      }
    }
    auto jt = std::dynamic_pointer_cast<tsid::tasks::TaskJointPosture>(task);
    if (jt) {
      jpos_task_ = jt;
    }
  }

  // Parse params
  if (node["preview_time"]) preview_time_ = node["preview_time"].as<double>();

  if (node["manipulability"]) {
    const auto& m = node["manipulability"];
    if (m["sigma_threshold"])       manip_config_.sigma_threshold       = m["sigma_threshold"].as<double>();
    if (m["gain"])                  manip_config_.gain                  = m["gain"].as<double>();
    if (m["max_bias_qdot"])         manip_config_.max_bias_qdot         = m["max_bias_qdot"].as<double>();
    if (m["fd_eps"])                manip_config_.fd_eps                = m["fd_eps"].as<double>();
    if (m["use_full_jacobian"])     manip_config_.use_full_jacobian     = m["use_full_jacobian"].as<bool>();
    if (m["characteristic_length"]) manip_config_.characteristic_length = m["characteristic_length"].as<double>();
  }
}

void CartesianTeleopState::FirstVisit() {
  ee_handler_.Init(preview_time_);
  ee_handler_.ResetCommand();

  // Initialize manipulability handler
  if (ee_pos_task_ && data_) {
    manip_handler_.Init(robot_, data_, ee_pos_task_->frame_id(), manip_config_);
  }

  watchdog_ = Watchdog{watchdog_.GetTimeout()};
  prev_vel_ts_ns_ = 0;
  prev_pose_ts_ns_ = 0;
  last_vel_ts_ns_ = 0;
  last_pose_ts_ns_ = 0;

  // Capture current EE pose
  if (!data_) {
    return;
  }
  pinocchio::Data& data = *data_;
  if (ee_pos_task_) {
    auto fid = ee_pos_task_->frame_id();
    pinocchio::SE3 H = data.oMf[fid];
    pos_des_ = H.translation();
    quat_des_ = Eigen::Quaterniond(H.rotation());
    pose_cmd_pos_ = pos_des_;
    pose_cmd_quat_ = quat_des_;
  }
}

void CartesianTeleopState::UpdateCommand(const Eigen::Vector3d& xdot,
                                         const Eigen::Vector3d& wdot,
                                         int64_t vel_ts_ns,
                                         const Eigen::Vector3d& x_des,
                                         const Eigen::Quaterniond& quat_des,
                                         int64_t pose_ts_ns) {
  if (vel_ts_ns > 0 && vel_ts_ns != prev_vel_ts_ns_) {
    prev_vel_ts_ns_ = vel_ts_ns;
    last_vel_ts_ns_ = vel_ts_ns;
    watchdog_.Reset();
    ee_handler_.SetLinearVelocity(xdot);
    ee_handler_.SetAngularVelocity(wdot);
  }
  if (pose_ts_ns > 0 && pose_ts_ns != prev_pose_ts_ns_) {
    prev_pose_ts_ns_ = pose_ts_ns;
    last_pose_ts_ns_ = pose_ts_ns;
    pose_cmd_pos_ = x_des;
    pose_cmd_quat_ = quat_des.normalized();
  }
}

void CartesianTeleopState::setSE3Ref(
    tsid::tasks::TaskSE3Equality& task,
    const Eigen::Vector3d& pos,
    const Eigen::Quaterniond& quat,
    const Eigen::Vector3d& linear_vel,
    const Eigen::Vector3d& angular_vel) {
  Eigen::Matrix3d R = quat.toRotationMatrix();
  // SE3 vector format: [translation(3); rotation(9)]
  se3_val_.head<3>() = pos;
  Eigen::Map<Eigen::Matrix<double, 9, 1>>(se3_val_.data() + 3) =
      Eigen::Map<const Eigen::Matrix<double, 9, 1>>(R.data());
  se3_sample_.setValue(se3_val_);
  // TSID twist convention: [angular; linear]
  se3_twist_.head<3>() = angular_vel;
  se3_twist_.tail<3>() = linear_vel;
  se3_sample_.setDerivative(se3_twist_);
  se3_sample_.setSecondDerivative(se3_zero_acc_);
  task.setReference(se3_sample_);
}

void CartesianTeleopState::OneStep() {
  const double dt = sp_->servo_dt;
  const int na = robot_->na();
  if (!data_) return;
  pinocchio::Data& data = *data_;

  // --- Task 1: Cartesian teleop (EE position + orientation) ---
  const bool use_pose_cmd =
      (last_pose_ts_ns_ > 0 && last_pose_ts_ns_ >= last_vel_ts_ns_);

  if (use_pose_cmd) {
    pos_des_ = pose_cmd_pos_;
    quat_des_ = pose_cmd_quat_;
    Eigen::Vector3d zeros3 = Eigen::Vector3d::Zero();
    if (ee_pos_task_) setSE3Ref(*ee_pos_task_, pos_des_, quat_des_, zeros3, zeros3);
    if (ee_ori_task_) setSE3Ref(*ee_ori_task_, pos_des_, quat_des_, zeros3, zeros3);
  } else {
    watchdog_.Update(dt);
    if (watchdog_.IsTimeout()) {
      ee_handler_.ResetCommand();
    }

    if (ee_pos_task_) {
      auto fid = ee_pos_task_->frame_id();
      pinocchio::SE3 H = data.oMf[fid];
      Eigen::Vector3d pos_curr = H.translation();
      Eigen::Quaterniond quat_curr(H.rotation());

      auto pos_result = ee_handler_.ComputePos(pos_curr);
      auto ori_result = ee_handler_.ComputeOri(quat_curr);

      pos_des_ = pos_result.pos_des;
      quat_des_ = ori_result.quat_des;

      setSE3Ref(*ee_pos_task_, pos_des_, quat_des_,
                      pos_result.vel_des, ori_result.omega_des);
      if (ee_ori_task_) {
        setSE3Ref(*ee_ori_task_, pos_des_, quat_des_,
                        pos_result.vel_des, ori_result.omega_des);
      }
    }
  }

  // --- Task 2: Soft posture bias (manipulability singularity avoidance) ---
  if (jpos_task_) {
    // Lazy-init posture buffers (na is only known at runtime)
    if (!jpos_buffers_initialized_) {
      q_active_.resize(na);
      q_des_jpos_.resize(na);
      zero_acc_jpos_ = Eigen::VectorXd::Zero(na);
      jpos_sample_ = tsid::trajectories::TrajectorySample(na);
      jpos_buffers_initialized_ = true;
    }

    manip_handler_.Update(sp_->nominal_jpos);
    const Eigen::VectorXd& qdot_avoid = manip_handler_.avoidance_velocity();

    q_active_ = sp_->nominal_jpos.tail(na);
    q_des_jpos_.noalias() = q_active_ + qdot_avoid * dt;

    jpos_sample_.setValue(q_des_jpos_);
    jpos_sample_.setDerivative(qdot_avoid);
    jpos_sample_.setSecondDerivative(zero_acc_jpos_);
    jpos_task_->setReference(jpos_sample_);
  }
}

void CartesianTeleopState::LastVisit() {}

void CartesianTeleopState::SetExternalInput(const TaskInput& input) {
  if (input.x_des) pos_des_ = *input.x_des;
  if (input.quat_des) quat_des_ = *input.quat_des;
}

}  // namespace wbc
