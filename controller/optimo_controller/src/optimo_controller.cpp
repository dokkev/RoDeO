// Copyright 2024 Roboligent, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "optimo_controller/optimo_controller.hpp"

#include <algorithm>
#include <cmath>

#include <yaml-cpp/yaml.h>
#include <Eigen/Dense>
#include <hardware_interface/types/hardware_interface_type_values.hpp>
#include <pluginlib/class_list_macros.hpp>
#include <rclcpp/rclcpp.hpp>

#include "optimo_controller/state_machines/cartesian_teleop.hpp"
#include "optimo_controller/state_machines/joint_teleop.hpp"
#include "wbc_util/ros_path_utils.hpp"

namespace optimo_controller
{
////////////////////////////////////////////////////////////////////////

OptimoController::~OptimoController() = default;

////////////////////////////////////////////////////////////////////////

controller_interface::CallbackReturn OptimoController::on_init()
{
  auto_declare<std::vector<std::string>>("joints", {});
  auto_declare<std::string>(
    "wbc_yaml_path", "package://optimo_controller/config/optimo_wbc.yaml");
  auto_declare<double>("control_frequency", 1000.0);
  auto_declare<bool>("is_simulation", false);
  return controller_interface::CallbackReturn::SUCCESS;
}

////////////////////////////////////////////////////////////////////////

controller_interface::InterfaceConfiguration
OptimoController::command_interface_configuration() const
{
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;

  config.names.reserve(joint_count_ * kInterfacesPerJoint);
  for (const auto & joint : joints_)
    config.names.push_back(joint + "/" + hardware_interface::HW_IF_POSITION);
  for (const auto & joint : joints_)
    config.names.push_back(joint + "/" + hardware_interface::HW_IF_VELOCITY);
  for (const auto & joint : joints_)
    config.names.push_back(joint + "/" + hardware_interface::HW_IF_EFFORT);

  return config;
}

////////////////////////////////////////////////////////////////////////

controller_interface::InterfaceConfiguration
OptimoController::state_interface_configuration() const
{
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  config.names.reserve(joint_count_ * kInterfacesPerJoint);
  for (const auto & joint : joints_)
    config.names.push_back(joint + "/" + hardware_interface::HW_IF_POSITION);
  for (const auto & joint : joints_)
    config.names.push_back(joint + "/" + hardware_interface::HW_IF_VELOCITY);
  for (const auto & joint : joints_)
    config.names.push_back(joint + "/" + hardware_interface::HW_IF_EFFORT);
  return config;
}

////////////////////////////////////////////////////////////////////////

controller_interface::CallbackReturn
OptimoController::on_configure(const rclcpp_lifecycle::State & /*previous_state*/)
{
  joints_ = get_node()->get_parameter("joints").as_string_array();
  joint_count_ = joints_.size();
  wbc_yaml_path_ = get_node()->get_parameter("wbc_yaml_path").as_string();
  control_frequency_hz_ = get_node()->get_parameter("control_frequency").as_double();
  if (!std::isfinite(control_frequency_hz_) || control_frequency_hz_ <= 0.0)
  {
    RCLCPP_ERROR(
      get_node()->get_logger(),
      "[OptimoController] parameter 'control_frequency' must be finite and > 0. got=%.6f",
      control_frequency_hz_);
    return controller_interface::CallbackReturn::ERROR;
  }
  control_dt_ = 1.0 / control_frequency_hz_;
  if (joints_.empty())
  {
    RCLCPP_ERROR(get_node()->get_logger(), "[OptimoController] parameter 'joints' is empty.");
    return controller_interface::CallbackReturn::ERROR;
  }

  try
  {
    auto arch_config =
      wbc::ControlArchitectureConfig::FromYaml(wbc_yaml_path_, control_dt_);
    arch_config.state_provider = std::make_unique<wbc::StateProvider>(control_dt_);
    ctrl_arch_ = std::make_unique<wbc::ControlArchitecture>(std::move(arch_config));
    ctrl_arch_->Initialize();
    // Parse debug_mode from WBC YAML
    {
      const std::string resolved = wbc::path::ResolvePackageUri(wbc_yaml_path_);
      YAML::Node root = YAML::LoadFile(resolved);
      debug_mode_ = root["debug_mode"].as<bool>(false);
      debug_print_interval_s_ = root["debug_print_interval"].as<double>(5.0);
      if (debug_mode_) {
        RCLCPP_INFO(get_node()->get_logger(),
          "[OptimoController] debug_mode ON (print every %.1f s)", debug_print_interval_s_);
      }
    }
    ctrl_arch_->enable_timing_ = true;  // enable per-phase timing stats

    // Cache typed state pointers (non-RT, configure phase only).
    auto * fsm = ctrl_arch_->GetFsmHandler();
    if (const auto id = fsm->FindStateIdByName("joint_teleop")) {
      joint_teleop_state_ = dynamic_cast<wbc::JointTeleop *>(fsm->FindStateById(*id));
    }
    if (const auto id = fsm->FindStateIdByName("cartesian_teleop")) {
      cartesian_teleop_state_ = dynamic_cast<wbc::CartesianTeleop *>(fsm->FindStateById(*id));
    }
    if (const auto id = fsm->FindStateIdByName("safe_command")) {
      safe_command_state_id_ = *id;
    }
  }
  catch (const std::exception & e)
  {
    ctrl_arch_.reset();
    RCLCPP_ERROR(
      get_node()->get_logger(), "[OptimoController] failed to build control architecture: %s",
      e.what());
    return controller_interface::CallbackReturn::ERROR;
  }

  if (joint_teleop_state_ == nullptr) {
    RCLCPP_ERROR(get_node()->get_logger(),
      "[OptimoController] required state 'joint_teleop' not found in WBC config.");
    return controller_interface::CallbackReturn::ERROR;
  }
  if (cartesian_teleop_state_ == nullptr) {
    RCLCPP_ERROR(get_node()->get_logger(),
      "[OptimoController] required state 'cartesian_teleop' not found in WBC config.");
    return controller_interface::CallbackReturn::ERROR;
  }
  if (!safe_command_state_id_.has_value()) {
    RCLCPP_ERROR(get_node()->get_logger(),
      "[OptimoController] required state 'safe_command' not found in WBC config.");
    return controller_interface::CallbackReturn::ERROR;
  }

  // Build actuator interface (spring for sim, passthrough for real HW).
  {
    const bool is_sim = get_node()->get_parameter("is_simulation").as_bool();
    if (is_sim) {
      Eigen::VectorXd stiffness(7);
      stiffness << 966.4, 947.6, 509.3, 404.1, 484.3, 479.2, 455.6;
      Eigen::VectorXd damping(7);
      damping << 10.0, 10.0, 5.0, 5.0, 3.0, 3.0, 3.0;

      actuator_ = std::make_unique<wbc::SpringActuator>(stiffness, damping);
      RCLCPP_INFO(get_node()->get_logger(),
        "[OptimoController] Simulation mode: spring actuator enabled");
    } else {
      actuator_ = std::make_unique<wbc::DirectActuator>();
      RCLCPP_INFO(get_node()->get_logger(),
        "[OptimoController] Hardware mode: direct passthrough");
    }
  }

  // Pre-size / pre-initialize all command buffers.
  {
    const std::vector<double> zeros(joint_count_, 0.0);
    qdot_des_buf_.writeFromNonRT(JointVelRef{zeros, 0});
    q_des_buf_.writeFromNonRT(JointPosRef{zeros, 0});
  }
  xdot_des_buf_.writeFromNonRT(EEVelRef{{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, 0});

  // Joint velocity subscriber
  joint_vel_sub_ =
    get_node()->create_subscription<std_msgs::msg::Float64MultiArray>(
      "~/joint_vel_cmd",
      rclcpp::SensorDataQoS(),
      [this](std_msgs::msg::Float64MultiArray::ConstSharedPtr msg) {
        if (msg->data.size() != joint_count_) { return; }
        qdot_des_buf_.writeFromNonRT(JointVelRef{msg->data, get_node()->now().nanoseconds()});
      });

  // EE velocity subscriber
  ee_vel_sub_ =
    get_node()->create_subscription<geometry_msgs::msg::TwistStamped>(
      "~/ee_vel_cmd",
      rclcpp::SensorDataQoS(),
      [this](geometry_msgs::msg::TwistStamped::ConstSharedPtr msg) {
        xdot_des_buf_.writeFromNonRT(EEVelRef{
          {msg->twist.linear.x,  msg->twist.linear.y,  msg->twist.linear.z},
          {msg->twist.angular.x, msg->twist.angular.y, msg->twist.angular.z},
          rclcpp::Time(msg->header.stamp).nanoseconds()});
      });

  // Joint position subscriber
  joint_pos_sub_ =
    get_node()->create_subscription<std_msgs::msg::Float64MultiArray>(
      "~/joint_pos_cmd",
      rclcpp::SensorDataQoS(),
      [this](std_msgs::msg::Float64MultiArray::ConstSharedPtr msg) {
        if (msg->data.size() != joint_count_) { return; }
        q_des_buf_.writeFromNonRT(JointPosRef{msg->data, get_node()->now().nanoseconds()});
      });

  // State transition service
  set_state_srv_ = get_node()->create_service<wbc_msgs::srv::TransitionState>(
    "~/set_state",
    [this](const wbc_msgs::srv::TransitionState::Request::SharedPtr req,
           wbc_msgs::srv::TransitionState::Response::SharedPtr res) {
      if (!req->state_name.empty()) {
        if (ctrl_arch_->RequestState(req->state_name)) {
          res->success = true;
          res->message = "Transition requested: " + req->state_name;
        } else {
          res->success = false;
          res->message = "Unknown state name: " + req->state_name;
        }
      } else {
        ctrl_arch_->RequestState(req->state_id);
        res->success = true;
        res->message = "Transition requested: id=" + std::to_string(req->state_id);
      }
    });

  // Log available states
  {
    const auto& states = ctrl_arch_->GetFsmHandler()->GetStates();
    std::string state_list;
    for (const auto& [id, name] : states) {
      if (!state_list.empty()) state_list += ", ";
      state_list += std::to_string(id) + ":" + name;
    }
    RCLCPP_INFO(get_node()->get_logger(),
      "[OptimoController] Available states: [%s]", state_list.c_str());
  }

  return controller_interface::CallbackReturn::SUCCESS;
}

////////////////////////////////////////////////////////////////////////

controller_interface::CallbackReturn
OptimoController::on_activate(const rclcpp_lifecycle::State & /*previous_state*/)
{
  const std::size_t expected_state_interfaces  = joint_count_ * kInterfacesPerJoint;
  const std::size_t expected_command_interfaces = joint_count_ * kInterfacesPerJoint;
  if (state_interfaces_.size() < expected_state_interfaces ||
    command_interfaces_.size() < expected_command_interfaces)
  {
    RCLCPP_ERROR(
      get_node()->get_logger(),
      "[OptimoController] missing interfaces. expected state>=%zu command>=%zu (got state=%zu command=%zu)",
      expected_state_interfaces, expected_command_interfaces,
      state_interfaces_.size(), command_interfaces_.size());
    return controller_interface::CallbackReturn::ERROR;
  }

  robot_joint_state_.Reset(static_cast<Eigen::Index>(joint_count_));

  // Read initial joint positions and reset actuator state.
  {
    Eigen::VectorXd q0(joint_count_);
    for (std::size_t i = 0; i < joint_count_; ++i) {
      q0[static_cast<Eigen::Index>(i)] =
        state_interfaces_[InterfaceIndex(kPositionBlock, i, joint_count_)].get_value();
    }
    actuator_->Reset(q0);
  }

  // Position: hold current hardware positions. Velocity and effort: zero.
  for (std::size_t i = 0; i < joint_count_; ++i) {
    (void)command_interfaces_[InterfaceIndex(kPositionBlock, i, joint_count_)].set_value(
      state_interfaces_[InterfaceIndex(kPositionBlock, i, joint_count_)].get_value());
  }
  for (std::size_t i = 0; i < joint_count_; ++i) {
    (void)command_interfaces_[InterfaceIndex(kVelocityBlock, i, joint_count_)].set_value(0.0);
    (void)command_interfaces_[InterfaceIndex(kEffortBlock, i, joint_count_)].set_value(0.0);
  }

  // Sync active_state_id_ before the first update() tick.
  active_state_id_ = ctrl_arch_->GetCurrentStateId();

  return controller_interface::CallbackReturn::SUCCESS;
}

////////////////////////////////////////////////////////////////////////

controller_interface::CallbackReturn
OptimoController::on_deactivate(const rclcpp_lifecycle::State & /*previous_state*/)
{
  // Zero velocity and effort commands. Keep position to avoid swing.
  for (std::size_t i = 0; i < joint_count_; ++i) {
    (void)command_interfaces_[InterfaceIndex(kVelocityBlock, i, joint_count_)].set_value(0.0);
    (void)command_interfaces_[InterfaceIndex(kEffortBlock, i, joint_count_)].set_value(0.0);
  }
  return controller_interface::CallbackReturn::SUCCESS;
}

////////////////////////////////////////////////////////////////////////

controller_interface::return_type OptimoController::update(
  const rclcpp::Time & time, const rclcpp::Duration & /*period*/)
{
  if (!ctrl_arch_) return controller_interface::return_type::OK;

  // Route teleop commands to the active state.
  if (active_state_id_ == joint_teleop_state_->id()) {
    const auto* qdot_des = qdot_des_buf_.readFromRT();
    const auto* q_des = q_des_buf_.readFromRT();
    joint_teleop_state_->UpdateCommand(
      Eigen::Map<const Eigen::VectorXd>(qdot_des->qdot.data(), joint_count_),
      qdot_des->ts_ns,
      Eigen::Map<const Eigen::VectorXd>(q_des->q.data(), joint_count_),
      q_des->ts_ns);
  }

  if (active_state_id_ == cartesian_teleop_state_->id()) {
    const auto* xdot_des = xdot_des_buf_.readFromRT();
    cartesian_teleop_state_->UpdateCommand(
      xdot_des->xdot, xdot_des->wdot, xdot_des->ts_ns);
  }

  ctrl_arch_->Update(ReadJointState(), time.seconds(), control_dt_);
  // Refresh after FSM ran so auto-transitions are captured for the next tick.
  active_state_id_ = ctrl_arch_->GetCurrentStateId();

  // QP status monitoring
  {
    const auto& ts = ctrl_arch_->timing_stats_;
    const double total_us = ts.robot_model_us + ts.kinematics_us + ts.dynamics_us
                          + ts.find_config_us + ts.make_torque_us + ts.feedback_us;

    const auto* wbic_data = ctrl_arch_->GetSolver()->GetWbicData();

    // Track peak tick time
    if (total_us > max_tick_us_) max_tick_us_ = total_us;
    ++tick_count_;

    // Log warning if total WBC time exceeds 500 us (half of 1 kHz budget)
    if (total_us > 500.0) {
      RCLCPP_WARN_THROTTLE(get_node()->get_logger(), *get_node()->get_clock(), 1000,
        "[WBC] slow tick: %.0f us (model=%.0f kin=%.0f dyn=%.0f ik=%.0f id=%.0f fb=%.0f) "
        "qp_iter=%d qp_solve=%.0f us",
        total_us, ts.robot_model_us, ts.kinematics_us, ts.dynamics_us,
        ts.find_config_us, ts.make_torque_us, ts.feedback_us,
        wbic_data ? wbic_data->qp_iter_ : -1,
        wbic_data ? wbic_data->qp_solve_time_us_ : 0.0);
    }

    // Log error if QP did not solve
    if (wbic_data && !wbic_data->qp_solved_) {
      RCLCPP_ERROR_THROTTLE(get_node()->get_logger(), *get_node()->get_clock(), 1000,
        "[WBC] QP FAILED: status=%d iter=%d pri_res=%.2e dua_res=%.2e obj=%.4f",
        wbic_data->qp_status_, wbic_data->qp_iter_,
        wbic_data->qp_pri_res_, wbic_data->qp_dua_res_, wbic_data->qp_obj_);
    }

    // Periodic debug status print
    const double t_now = time.seconds();
    if (debug_mode_ && (t_now - last_debug_print_time_) >= debug_print_interval_s_) {
      const auto& cmd = ctrl_arch_->GetCommand();
      RCLCPP_INFO(get_node()->get_logger(),
        "[WBC] state=%d | tick=%.0f us (peak=%.0f us) | "
        "model=%.0f kin=%.0f dyn=%.0f ik=%.0f id=%.0f fb=%.0f | "
        "qp: solved=%d iter=%d solve=%.0f us | "
        "tau[0..2]=[%.2f, %.2f, %.2f]",
        active_state_id_,
        total_us, max_tick_us_,
        ts.robot_model_us, ts.kinematics_us, ts.dynamics_us,
        ts.find_config_us, ts.make_torque_us, ts.feedback_us,
        wbic_data ? static_cast<int>(wbic_data->qp_solved_) : -1,
        wbic_data ? wbic_data->qp_iter_ : -1,
        wbic_data ? wbic_data->qp_solve_time_us_ : 0.0,
        cmd.tau.size() > 0 ? cmd.tau[0] : 0.0,
        cmd.tau.size() > 1 ? cmd.tau[1] : 0.0,
        cmd.tau.size() > 2 ? cmd.tau[2] : 0.0);
      last_debug_print_time_ = t_now;
      max_tick_us_ = 0.0;  // reset peak for next interval
    }
  }

  // Get WBC command and apply actuator model.
  auto cmd = ctrl_arch_->GetCommand();
  {
    wbc::ActuatorCommand act_cmd;
    act_cmd.q_des = cmd.q;
    act_cmd.qdot_des = cmd.qdot;
    act_cmd.tau_ff = cmd.tau;
    act_cmd.q_link = robot_joint_state_.q;
    act_cmd.qdot_link = robot_joint_state_.qdot;
    act_cmd.dt = control_dt_;
    cmd.tau = actuator_->ProcessTorque(act_cmd);
  }

  WriteJointCommand(cmd);

  return controller_interface::return_type::OK;
}

////////////////////////////////////////////////////////////////////////

const wbc::RobotJointState & OptimoController::ReadJointState()
{
  for (std::size_t i = 0; i < joint_count_; ++i)
  {
    robot_joint_state_.q[i] =
      state_interfaces_[InterfaceIndex(kPositionBlock, i, joint_count_)].get_value();
    robot_joint_state_.qdot[i] =
      state_interfaces_[InterfaceIndex(kVelocityBlock, i, joint_count_)].get_value();
    robot_joint_state_.tau[i] =
      state_interfaces_[InterfaceIndex(kEffortBlock, i, joint_count_)].get_value();
  }
  return robot_joint_state_;
}

////////////////////////////////////////////////////////////////////////

void OptimoController::WriteJointCommand(const wbc::RobotCommand & cmd)
{
  for (std::size_t i = 0; i < joint_count_; ++i)
  {
    (void)command_interfaces_[InterfaceIndex(kPositionBlock, i, joint_count_)].set_value(cmd.q[i]);
    (void)command_interfaces_[InterfaceIndex(kVelocityBlock, i, joint_count_)].set_value(cmd.qdot[i]);
    (void)command_interfaces_[InterfaceIndex(kEffortBlock, i, joint_count_)].set_value(cmd.tau[i]);
  }
}

}  // namespace optimo_controller

PLUGINLIB_EXPORT_CLASS(
  optimo_controller::OptimoController, controller_interface::ControllerInterface)
