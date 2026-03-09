/**
 * @file controller/optimo_controller/src/state_machines/cartesian_teleop.cpp
 * @brief Cartesian end-effector teleop state for Optimo.
 */
#include "optimo_controller/state_machines/cartesian_teleop.hpp"

#include "wbc_fsm/state_factory.hpp"
#include "wbc_util/yaml_parser.hpp"

namespace wbc {

CartesianTeleop::CartesianTeleop(StateId state_id, const std::string& state_name,
                   const StateMachineConfig& context)
    : StateMachine(state_id, state_name, context) {}

void CartesianTeleop::SetParameters(const YAML::Node& node) {
  SetCommonParameters(node);
  SetMotionTask("ee_pos_task", ee_pos_task_);
  SetMotionTask("ee_ori_task", ee_ori_task_);
  SetMotionTask("jpos_task",   jpos_task_);

  const YAML::Node params = param::ResolveParamsNode(node);
  if (params["preview_time"]) preview_time_ = params["preview_time"].as<double>();

  if (params["manipulability"]) {
    const auto& m = params["manipulability"];
    if (m["sigma_threshold"])      manip_config_.sigma_threshold      = m["sigma_threshold"].as<double>();
    if (m["gain"])                 manip_config_.gain                 = m["gain"].as<double>();
    if (m["max_bias_qdot"])        manip_config_.max_bias_qdot        = m["max_bias_qdot"].as<double>();
    if (m["fd_eps"])               manip_config_.fd_eps               = m["fd_eps"].as<double>();
    if (m["use_full_jacobian"])    manip_config_.use_full_jacobian    = m["use_full_jacobian"].as<bool>();
    if (m["characteristic_length"]) manip_config_.characteristic_length = m["characteristic_length"].as<double>();
  }
}

void CartesianTeleop::FirstVisit() {
  ee_handler_.Init(preview_time_);
  ee_handler_.ResetCommand();

  manip_handler_.Init(robot_, ee_pos_task_->TargetIdx(), manip_config_);

  watchdog_ = Watchdog{watchdog_.GetTimeout()};  // re-initialize to expired state
  prev_vel_ts_ns_ = 0;
}

void CartesianTeleop::UpdateCommand(const Eigen::Vector3d& xdot,
                              const Eigen::Vector3d& wdot,
                              int64_t vel_ts_ns) {
  if (vel_ts_ns > 0 && vel_ts_ns != prev_vel_ts_ns_) {
    prev_vel_ts_ns_ = vel_ts_ns;
    watchdog_.Reset();
    ee_handler_.SetLinearVelocity(xdot);
    ee_handler_.SetAngularVelocity(wdot);
  }
}

void CartesianTeleop::OneStep() {
  const double dt = sp_->servo_dt_;

  // --- Task 1: Cartesian teleop (EE position + orientation) ---
  watchdog_.Update(dt);
  if (watchdog_.IsTimeout()) {
    ee_handler_.ResetCommand();
  }

  const Eigen::Isometry3d ee_iso = robot_->GetLinkIsometry(ee_pos_task_->TargetIdx());
  ee_handler_.UpdatePos(ee_iso.translation(), ee_pos_task_);
  ee_handler_.UpdateOri(Eigen::Quaterniond(ee_iso.rotation()), ee_ori_task_);

  // --- Task 2: Soft posture bias (manipulability singularity avoidance) ---
  manip_handler_.Update(dt);
  const Eigen::VectorXd& qdot_avoid = manip_handler_.avoidance_velocity();

  // 1-tick lookahead: target = current + avoidance_velocity * dt
  const Eigen::VectorXd q_des = robot_->GetJointPos() + qdot_avoid * dt;
  const Eigen::VectorXd zero_acc = Eigen::VectorXd::Zero(robot_->NumActiveDof());
  jpos_task_->UpdateDesired(q_des, qdot_avoid, zero_acc);
}

void CartesianTeleop::LastVisit() {}

bool CartesianTeleop::EndOfState() { return false; }

WBC_REGISTER_STATE(
    "cartesian_teleop",
    [](StateId id, const std::string& state_name,
       const StateMachineConfig& context) -> std::unique_ptr<StateMachine> {
      return std::make_unique<CartesianTeleop>(id, state_name, context);
    });

}  // namespace wbc
