//
// Copyright (c) 2026
//

#include "wbc_core/architecture/states/joint_teleop_state.hpp"

namespace wbc {

void JointTeleopState::SetParameters(const YAML::Node& node) {
  StateMachine::SetParameters(node);

  for (auto& [name, task] : assigned_tasks_) {
    auto jt = std::dynamic_pointer_cast<tsid::tasks::TaskJointPosture>(task);
    if (jt) {
      jpos_task_ = jt;
      break;
    }
  }

  // Optional per-joint velocity limit override
  if (node["joint_vel_limit"]) {
    const auto& v = node["joint_vel_limit"];
    vel_limit_.resize(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) {
      vel_limit_[i] = v[i].as<double>();
    }
  }
}

void JointTeleopState::FirstVisit() {
  const int na = robot_->na();
  const auto& model = robot_->model();

  // Position limits from URDF (active joints only)
  const int q_offset = robot_->is_fixed_base() ? 0 : 7;
  Eigen::VectorXd q_min = model.lowerPositionLimit.segment(q_offset, na);
  Eigen::VectorXd q_max = model.upperPositionLimit.segment(q_offset, na);

  // Velocity limits from URDF or override
  const int v_offset = robot_->is_fixed_base() ? 0 : 6;
  Eigen::VectorXd qdot_max = (vel_limit_.size() == na)
      ? vel_limit_
      : model.velocityLimit.segment(v_offset, na);

  // Current active joint positions
  const auto& q = sp_->nominal_jpos;
  Eigen::VectorXd q_curr = (q.size() >= na) ? q.tail(na).eval()
                                              : Eigen::VectorXd::Zero(na);

  handler_.Init(q_curr, q_min, q_max, qdot_max);
  current_qdot_.setZero(na);
  watchdog_ = Watchdog{watchdog_.GetTimeout()};
  prev_vel_ts_ns_ = 0;
  prev_pos_ts_ns_ = 0;
}

void JointTeleopState::UpdateCommand(
    const Eigen::Ref<const Eigen::VectorXd>& qdot_cmd,
    int64_t vel_ts_ns,
    const Eigen::Ref<const Eigen::VectorXd>& q_des,
    int64_t pos_ts_ns) {
  if (vel_ts_ns > 0 && vel_ts_ns != prev_vel_ts_ns_) {
    prev_vel_ts_ns_ = vel_ts_ns;
    watchdog_.Reset();
    current_qdot_ = qdot_cmd;
  }
  if (pos_ts_ns > 0 && pos_ts_ns != prev_pos_ts_ns_) {
    prev_pos_ts_ns_ = pos_ts_ns;
    handler_.SetPosition(q_des);
  }
}

void JointTeleopState::OneStep() {
  if (!jpos_task_) return;

  const double dt = sp_->servo_dt;

  watchdog_.Update(dt);
  if (watchdog_.IsTimeout()) current_qdot_.setZero();

  handler_.SetVelocity(current_qdot_, dt);
  handler_.Update(dt);

  if (!jpos_buffers_initialized_) {
    const int na = robot_->na();
    zero_acc_ = Eigen::VectorXd::Zero(na);
    jpos_sample_ = tsid::trajectories::TrajectorySample(na);
    jpos_buffers_initialized_ = true;
  }

  jpos_sample_.setValue(handler_.Desired());
  jpos_sample_.setDerivative(handler_.Vel());
  jpos_sample_.setSecondDerivative(zero_acc_);
  jpos_task_->setReference(jpos_sample_);
}

void JointTeleopState::LastVisit() {}

void JointTeleopState::SetExternalInput(const TaskInput& input) {
  if (input.q_des) {
    handler_.SetPosition(*input.q_des);
  }
}

}  // namespace wbc
