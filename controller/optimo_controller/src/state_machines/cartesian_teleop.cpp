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
  if (params["linear_vel_max"])  linear_vel_max_  = params["linear_vel_max"].as<double>();
  if (params["angular_vel_max"]) angular_vel_max_ = params["angular_vel_max"].as<double>();
  if (params["max_lookahead"])   max_lookahead_   = params["max_lookahead"].as<double>();

  if (params["manipulability"]) {
    const auto& m = params["manipulability"];
    if (m["step_size"])     manip_config_.step_size     = m["step_size"].as<double>();
    if (m["w_threshold"])   manip_config_.w_threshold   = m["w_threshold"].as<double>();
  }
}

void CartesianTeleop::FirstVisit() {
  const Eigen::Isometry3d iso  = robot_->GetLinkIsometry(ee_pos_task_->TargetIdx());
  const Eigen::Quaterniond quat(iso.rotation());
  ee_handler_.Init(iso.translation(), quat, linear_vel_max_, angular_vel_max_);

  manip_handler_.Init(robot_, ee_pos_task_->TargetIdx(), manip_config_);

  current_xdot_.setZero();
  current_wdot_.setZero();
  watchdog_ = Watchdog{watchdog_.GetTimeout()};  // re-initialize to expired state
  prev_vel_ts_ns_  = 0;
  prev_pose_ts_ns_ = 0;
}

void CartesianTeleop::UpdateCommand(const Eigen::Vector3d& xdot,
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

void CartesianTeleop::OneStep() {
  const double dt = sp_->servo_dt_;

  // --- Task 1: Cartesian teleop (EE position + orientation) ---
  watchdog_.Update(dt);
  if (watchdog_.IsTimeout()) {
    current_xdot_.setZero();
    current_wdot_.setZero();
  }

  ee_handler_.SetLinearVelocity(current_xdot_, dt);
  ee_handler_.SetAngularVelocity(current_wdot_, dt);

  // Anti-windup: keep pos_goal within max_lookahead of actual EE position.
  const Eigen::Vector3d actual_ee_pos =
      robot_->GetLinkIsometry(ee_pos_task_->TargetIdx()).translation();
  {
    const Eigen::Vector3d diff = ee_handler_.PosGoal() - actual_ee_pos;
    const double dist = diff.norm();
    if (max_lookahead_ > 0.0 && dist > max_lookahead_) {
      ee_handler_.SetPosGoal(actual_ee_pos + diff * (max_lookahead_ / dist));
    }
  }

  ee_handler_.UpdatePos(ee_pos_task_, dt);
  ee_handler_.UpdateOri(ee_ori_task_, dt);

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
