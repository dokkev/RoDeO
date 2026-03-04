/**
 * @file controller/optimo_controller/src/state_machines/ee_teleop.cpp
 * @brief Cartesian end-effector teleop state for Optimo.
 */
#include "optimo_controller/state_machines/ee_teleop.hpp"

#include "wbc_fsm/state_factory.hpp"
#include "wbc_util/yaml_parser.hpp"

namespace wbc {

EETeleop::EETeleop(StateId state_id, const std::string& state_name,
                   const StateMachineConfig& context)
    : StateMachine(state_id, state_name, context) {}

void EETeleop::SetParameters(const YAML::Node& node) {
  SetCommonParameters(node);
  SetMotionTask("ee_pos_task", ee_pos_task_);
  SetMotionTask("ee_ori_task", ee_ori_task_);
  SetMotionTask("jpos_task",   jpos_task_);

  const YAML::Node params = param::ResolveParamsNode(node);
  if (params["linear_vel_max"])  linear_vel_max_  = params["linear_vel_max"].as<double>();
  if (params["angular_vel_max"]) angular_vel_max_ = params["angular_vel_max"].as<double>();
  if (params["max_lookahead"])   max_lookahead_   = params["max_lookahead"].as<double>();

  if (params["manipulability"]) {
    const auto& m = params["manipulability"];
    if (m["step_size"])          manip_config_.step_size          = m["step_size"].as<double>();
    if (m["sigma_threshold"])    manip_config_.sigma_threshold    = m["sigma_threshold"].as<double>();
    if (m["gradient_interval"])  manip_config_.gradient_interval  = m["gradient_interval"].as<int>();
  }
}

void EETeleop::FirstVisit() {
  const Eigen::Isometry3d iso  = robot_->GetLinkIsometry(ee_pos_task_->TargetIdx());
  const Eigen::Quaterniond quat(iso.rotation());
  ee_handler_.Init(iso.translation(), quat, linear_vel_max_, angular_vel_max_);

  jpos_handler_.Init(
      robot_->GetJointPos(),
      robot_->GetJointPositionLimits().col(0),
      robot_->GetJointPositionLimits().col(1),
      robot_->GetJointVelocityLimits().col(1));

  manip_handler_.Init(robot_, ee_pos_task_->TargetIdx(), manip_config_);

  current_xdot_.setZero();
  current_wdot_.setZero();
  watchdog_ = Watchdog{watchdog_.GetTimeout()};  // re-initialize to expired state
  prev_vel_ts_ns_  = 0;
  prev_pose_ts_ns_ = 0;
}

void EETeleop::UpdateCommand(const Eigen::Vector3d& xdot,
                              const Eigen::Vector3d& wdot,
                              int64_t vel_ts_ns,
                              const Eigen::Vector3d& x_des,
                              const Eigen::Quaterniond& w_des,
                              int64_t pose_ts_ns) {
  if (vel_ts_ns > 0 && vel_ts_ns != prev_vel_ts_ns_) {
    prev_vel_ts_ns_ = vel_ts_ns;
    watchdog_.Reset();
    current_xdot_ = xdot;
    current_wdot_ = wdot;
  }
  if (pose_ts_ns > 0 && pose_ts_ns != prev_pose_ts_ns_) {
    prev_pose_ts_ns_ = pose_ts_ns;
    ee_handler_.SetPosition(x_des);
    ee_handler_.SetOrientation(w_des);
  }
}

void EETeleop::OneStep() {
  watchdog_.Update(sp_->servo_dt_);
  if (watchdog_.IsTimeout()) {
    current_xdot_.setZero();   // watchdog: hold position
    current_wdot_.setZero();
  }
  // Integrate stored velocity into the goal; no-op when zero (kEps guard in handler).
  ee_handler_.SetLinearVelocity(current_xdot_, sp_->servo_dt_);

  // Clamp pos_goal_ against workspace boundary and anti-windup radius.
  // actual_ee_pos is free (FK already updated by ctrl_arch_->Update before OneStep).
  const Eigen::Vector3d actual_ee_pos =
    robot_->GetLinkIsometry(ee_pos_task_->TargetIdx()).translation();
  ee_handler_.SetPosGoal(workspace_.Clamp(ee_handler_.PosGoal(), actual_ee_pos, max_lookahead_));

  ee_handler_.SetAngularVelocity(current_wdot_, sp_->servo_dt_);
  ee_handler_.UpdatePos(ee_pos_task_, sp_->servo_dt_);
  ee_handler_.UpdateOri(ee_ori_task_, sp_->servo_dt_);
  manip_handler_.Update(sp_->servo_dt_);
  if (manip_handler_.is_active()) {
    jpos_handler_.SetVelocity(manip_handler_.avoidance_velocity(), sp_->servo_dt_);
  }
  jpos_handler_.Update(jpos_task_, sp_->servo_dt_);
}

bool EETeleop::LoadWorkspace(const std::string& yaml_path) {
  return workspace_.Load(yaml_path);
}

void EETeleop::LastVisit() {}

bool EETeleop::EndOfState() { return false; }

WBC_REGISTER_STATE(
    "ee_teleop",
    [](StateId id, const std::string& state_name,
       const StateMachineConfig& context) -> std::unique_ptr<StateMachine> {
      return std::make_unique<EETeleop>(id, state_name, context);
    });

}  // namespace wbc
