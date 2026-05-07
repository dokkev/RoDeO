#include "plato_controller/state_machines/fingertip_teleop.hpp"

#include "wbc_fsm/state_factory.hpp"
#include "wbc_util/yaml_parser.hpp"

namespace wbc {

PlatoFingertipTeleop::PlatoFingertipTeleop(
    StateId state_id, const std::string& state_name,
    const StateMachineConfig& context)
    : StateMachine(state_id, state_name, context) {}

void PlatoFingertipTeleop::SetParameters(const YAML::Node& node) {
  SetCommonParameters(node);

  // Bind tasks
  SetMotionTask("ee_pos_task",     ee_pos_task_);
  SetMotionTask("ee_ori_task",     ee_ori_task_);
  SetMotionTask("finger_pos_task", finger_pos_task_);
  SetMotionTask("finger_ori_task", finger_ori_task_);
  SetMotionTask("jpos_task",       jpos_task_);

  const YAML::Node params = param::ResolveParamsNode(node);
  if (params["preview_time"]) preview_time_ = params["preview_time"].as<double>();

  // Manipulability config (for EE)
  if (params["manipulability"]) {
    const auto& m = params["manipulability"];
    if (m["sigma_threshold"])       manip_config_.sigma_threshold       = m["sigma_threshold"].as<double>();
    if (m["gain"])                  manip_config_.gain                  = m["gain"].as<double>();
    if (m["max_bias_qdot"])         manip_config_.max_bias_qdot         = m["max_bias_qdot"].as<double>();
    if (m["fd_eps"])                manip_config_.fd_eps                = m["fd_eps"].as<double>();
    if (m["use_full_jacobian"])     manip_config_.use_full_jacobian     = m["use_full_jacobian"].as<bool>();
    if (m["characteristic_length"]) manip_config_.characteristic_length = m["characteristic_length"].as<double>();
  }
}

void PlatoFingertipTeleop::FirstVisit() {
  // EE handler
  ee_handler_.Init(preview_time_);
  ee_handler_.ResetCommand();
  manip_handler_.Init(robot_, ee_pos_task_->TargetIdx(), manip_config_);
  ee_watchdog_ = Watchdog{ee_watchdog_.GetTimeout()};
  prev_ee_ts_ns_ = 0;

  // Finger handler
  finger_handler_.Init(preview_time_);
  finger_handler_.ResetCommand();
  finger_watchdog_ = Watchdog{finger_watchdog_.GetTimeout()};
  prev_finger_ts_ns_ = 0;
}

void PlatoFingertipTeleop::UpdateEECommand(
    const Eigen::Vector3d& xdot,
    const Eigen::Vector3d& wdot,
    int64_t vel_ts_ns) {
  if (vel_ts_ns > 0 && vel_ts_ns != prev_ee_ts_ns_) {
    prev_ee_ts_ns_ = vel_ts_ns;
    ee_watchdog_.Reset();
    ee_handler_.SetLinearVelocity(xdot);
    ee_handler_.SetAngularVelocity(wdot);
  }
}

void PlatoFingertipTeleop::UpdateFingerCommand(
    const Eigen::Vector3d& xdot,
    const Eigen::Vector3d& wdot,
    int64_t vel_ts_ns) {
  if (vel_ts_ns > 0 && vel_ts_ns != prev_finger_ts_ns_) {
    prev_finger_ts_ns_ = vel_ts_ns;
    finger_watchdog_.Reset();
    finger_handler_.SetLinearVelocity(xdot);
    finger_handler_.SetAngularVelocity(wdot);
  }
}

void PlatoFingertipTeleop::OneStep() {
  const double dt = sp_->servo_dt_;

  // --- EE Cartesian tracking ---
  ee_watchdog_.Update(dt);
  if (ee_watchdog_.IsTimeout()) {
    ee_handler_.ResetCommand();
  }

  const Eigen::Isometry3d ee_iso = robot_->GetLinkIsometry(ee_pos_task_->TargetIdx());
  ee_handler_.UpdatePos(ee_iso.translation(), ee_pos_task_);
  ee_handler_.UpdateOri(Eigen::Quaterniond(ee_iso.rotation()), ee_ori_task_);

  // --- Fingertip Cartesian tracking ---
  finger_watchdog_.Update(dt);
  if (finger_watchdog_.IsTimeout()) {
    finger_handler_.ResetCommand();
  }

  const Eigen::Isometry3d finger_iso = robot_->GetLinkIsometry(finger_pos_task_->TargetIdx());
  finger_handler_.UpdatePos(finger_iso.translation(), finger_pos_task_);
  finger_handler_.UpdateOri(Eigen::Quaterniond(finger_iso.rotation()), finger_ori_task_);

  // --- Soft posture bias (singularity avoidance for EE) ---
  manip_handler_.Update(dt);
  const Eigen::VectorXd& qdot_avoid = manip_handler_.avoidance_velocity();

  const Eigen::VectorXd q_des = robot_->GetJointPos() + qdot_avoid * dt;
  const Eigen::VectorXd zero_acc = Eigen::VectorXd::Zero(robot_->NumActiveDof());
  jpos_task_->UpdateDesired(q_des, qdot_avoid, zero_acc);
}

void PlatoFingertipTeleop::LastVisit() {}

bool PlatoFingertipTeleop::EndOfState() { return false; }

WBC_REGISTER_STATE(
    "plato_fingertip_teleop",
    [](StateId id, const std::string& state_name,
       const StateMachineConfig& context) -> std::unique_ptr<StateMachine> {
      return std::make_unique<PlatoFingertipTeleop>(id, state_name, context);
    });

}  // namespace wbc
