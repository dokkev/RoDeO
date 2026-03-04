/**
 * @file controller/optimo_controller/src/state_machines/joint_teleop.cpp
 * @brief Joint-space teleoperation state for Optimo.
 */
#include "optimo_controller/state_machines/joint_teleop.hpp"

#include "wbc_fsm/state_factory.hpp"

namespace wbc {

JointTeleop::JointTeleop(StateId state_id, const std::string& state_name,
                         const StateMachineConfig& context)
    : StateMachine(state_id, state_name, context) {}

void JointTeleop::SetParameters(const YAML::Node& node) {
  SetCommonParameters(node);
  SetMotionTask("jpos_task", jpos_task_);
  if (node["joint_vel_limit"]) vel_limit_ = ParseVectorParam(node, "joint_vel_limit");
}

void JointTeleop::FirstVisit() {
  const Eigen::VectorXd qdot_max = (vel_limit_.size() > 0)
                                       ? vel_limit_
                                       : robot_->GetJointVelocityLimits().col(1);
  handler_.Init(
      robot_->GetJointPos(),
      robot_->GetJointPositionLimits().col(0),
      robot_->GetJointPositionLimits().col(1),
      qdot_max);
  current_qdot_.setZero(robot_->NumActiveDof());
  watchdog_ = Watchdog{watchdog_.GetTimeout()};  // re-initialize to expired state
  prev_vel_ts_ns_ = 0;
  prev_pos_ts_ns_ = 0;
}

void JointTeleop::UpdateCommand(const Eigen::Ref<const Eigen::VectorXd>& qdot_cmd,
                                int64_t vel_ts_ns,
                                const Eigen::Ref<const Eigen::VectorXd>& q_des,
                                int64_t pos_ts_ns) {
  if (vel_ts_ns > 0 && vel_ts_ns != prev_vel_ts_ns_) {
    prev_vel_ts_ns_ = vel_ts_ns;
    watchdog_.Reset();
    current_qdot_ = qdot_cmd;  // in-place copy, no alloc (pre-sized in FirstVisit)
  }
  if (pos_ts_ns > 0 && pos_ts_ns != prev_pos_ts_ns_) {
    prev_pos_ts_ns_ = pos_ts_ns;
    handler_.SetPosition(q_des);
  }
}

void JointTeleop::OneStep() {
  watchdog_.Update(sp_->servo_dt_);
  if (watchdog_.IsTimeout()) current_qdot_.setZero();  // watchdog: hold position
  handler_.SetVelocity(current_qdot_, sp_->servo_dt_);
  handler_.Update(jpos_task_, sp_->servo_dt_);
}

void JointTeleop::LastVisit() {}

bool JointTeleop::EndOfState() { return false; }

WBC_REGISTER_STATE(
    "joint_teleop",
    [](StateId id, const std::string& state_name,
       const StateMachineConfig& context) -> std::unique_ptr<StateMachine> {
      return std::make_unique<JointTeleop>(id, state_name, context);
    });

} // namespace wbc
