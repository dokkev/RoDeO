/**
 * @file controller/optimo_controller/src/state_machines/sys_id.cpp
 * @brief Offline joint-space SysID excitation state.
 */
#include "optimo_controller/state_machines/sys_id.hpp"

#include <cstdio>

#include "wbc_fsm/state_factory.hpp"
#include "wbc_util/yaml_parser.hpp"

namespace wbc {

SysIdState::SysIdState(StateId state_id, const std::string& state_name,
                       const StateMachineConfig& context)
    : StateMachine(state_id, state_name, context) {}

void SysIdState::SetParameters(const YAML::Node& node) {
  SetCommonParameters(node);
  SetMotionTask("jpos_task", jpos_task_);

  const YAML::Node params = param::ResolveParamsNode(node);
  if (!params) {
    handler_.Configure(handler_config_);
    return;
  }

  if (params["mode"]) {
    handler_config_.sysid.mode = ParseMode(params["mode"].as<std::string>());
  }
  if (params["enabled"]) {
    handler_config_.sysid.enabled = params["enabled"].as<bool>();
  }
  if (params["joint_idx"]) {
    handler_config_.sysid.joint_idx = params["joint_idx"].as<int>();
  }

  if (params["start_delay"]) {
    handler_config_.sysid.start_delay = params["start_delay"].as<double>();
  }
  if (params["ramp_time"]) {
    handler_config_.sysid.ramp_time = params["ramp_time"].as<double>();
  }
  if (params["dwell_time"]) {
    handler_config_.sysid.dwell_time = params["dwell_time"].as<double>();
  }
  if (params["duration"]) {
    handler_config_.sysid.duration = params["duration"].as<double>();
  }

  if (params["amplitude"]) {
    handler_config_.sysid.amplitude = params["amplitude"].as<double>();
  }
  if (params["offset"]) {
    handler_config_.sysid.offset = params["offset"].as<double>();
  }
  if (params["cruise_vel"]) {
    handler_config_.sysid.cruise_vel = params["cruise_vel"].as<double>();
  }
  if (params["frequency_hz"]) {
    handler_config_.sysid.frequency_hz = params["frequency_hz"].as<double>();
  }
  if (params["chirp_f0_hz"]) {
    handler_config_.sysid.chirp_f0_hz = params["chirp_f0_hz"].as<double>();
  }
  if (params["chirp_f1_hz"]) {
    handler_config_.sysid.chirp_f1_hz = params["chirp_f1_hz"].as<double>();
  }
  if (params["chirp_duration"]) {
    handler_config_.sysid.chirp_duration = params["chirp_duration"].as<double>();
  }
  if (params["phase0"]) {
    handler_config_.sysid.phase0 = params["phase0"].as<double>();
  }

  if (params["gravity_offsets_rad"]) {
    handler_config_.sysid.gravity_offsets_rad =
        params["gravity_offsets_rad"].as<std::vector<double>>();
  }

  if (params["max_tracking_err"]) {
    handler_config_.sysid.max_tracking_err =
        params["max_tracking_err"].as<double>();
  }
  if (params["max_meas_vel"]) {
    handler_config_.sysid.max_meas_vel = params["max_meas_vel"].as<double>();
  }
  if (params["max_tau_ratio"]) {
    handler_config_.sysid.max_tau_ratio = params["max_tau_ratio"].as<double>();
  }

  if (params["abort_on_safety"]) {
    handler_config_.abort_on_safety = params["abort_on_safety"].as<bool>();
  }
  if (params["hold_on_abort"]) {
    handler_config_.hold_on_abort = params["hold_on_abort"].as<bool>();
  }
  if (params["end_on_finish"]) {
    end_on_finish_ = params["end_on_finish"].as<bool>();
  }

  handler_.Configure(handler_config_);
}

void SysIdState::FirstVisit() {
  const int n = robot_->NumActiveDof();
  q_ref_.setZero(n);
  qdot_ref_.setZero(n);
  qddot_ref_.setZero(n);
  tau_cmd_.setZero(n);
  tau_limits_ = robot_->SoftTorqueLimits();
  abort_reported_ = false;

  handler_.Initialize(n, robot_->GetJointPos(), current_time_);
}

void SysIdState::OneStep() {
  if (tau_limits_.rows() != robot_->SoftTorqueLimits().rows()) {
    tau_limits_ = robot_->SoftTorqueLimits();
  }

  handler_.Update(current_time_, sp_->servo_dt_,
                  robot_->GetJointPos(), robot_->GetJointVel(),
                  tau_cmd_, tau_limits_,
                  q_ref_, qdot_ref_, qddot_ref_);

  jpos_task_->UpdateDesired(q_ref_, qdot_ref_, qddot_ref_);

  if (handler_.IsAborted() && !abort_reported_) {
    std::fprintf(stderr, "[SysIdState] aborted: %s\n",
                 handler_.LastReason().c_str());
    abort_reported_ = true;
  }
}

void SysIdState::LastVisit() {
  handler_.Stop();
}

bool SysIdState::EndOfState() {
  if (stay_here_) {
    return false;
  }

  if (end_on_finish_ && (handler_.IsFinished() || handler_.IsAborted())) {
    return true;
  }

  return StateMachine::EndOfState();
}

SysIDMode SysIdState::ParseMode(const std::string& mode) {
  if (mode == "gravity_grid") {
    return SysIDMode::GRAVITY_GRID;
  }
  if (mode == "friction_sweep") {
    return SysIDMode::FRICTION_SWEEP;
  }
  if (mode == "sine") {
    return SysIDMode::SINE;
  }
  if (mode == "chirp") {
    return SysIDMode::CHIRP;
  }
  return SysIDMode::OFF;
}

WBC_REGISTER_STATE(
    "sys_id",
    [](StateId id, const std::string& state_name,
       const StateMachineConfig& context) -> std::unique_ptr<StateMachine> {
      return std::make_unique<SysIdState>(id, state_name, context);
    });

}  // namespace wbc
