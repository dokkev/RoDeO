#include "plato_controller/plato_controller.hpp"

#include <algorithm>
#include <cmath>

#include <Eigen/Dense>
#include <hardware_interface/types/hardware_interface_type_values.hpp>
#include <pluginlib/class_list_macros.hpp>
#include <rclcpp/rclcpp.hpp>
#include <yaml-cpp/yaml.h>

#include "plato_controller/state_machines/fingertip_teleop.hpp"
#include "optimo_controller/state_machines/cartesian_teleop.hpp"
#include "optimo_controller/state_machines/joint_teleop.hpp"
#include "wbc_util/ros_path_utils.hpp"

namespace plato_controller {

PlatoController::~PlatoController() = default;

controller_interface::CallbackReturn PlatoController::on_init() {
  auto_declare<std::vector<std::string>>("joints", {});
  auto_declare<std::string>(
      "wbc_yaml_path", "package://plato_controller/config/plato_wbc.yaml");
  auto_declare<double>("control_frequency", 1000.0);
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::InterfaceConfiguration
PlatoController::command_interface_configuration() const {
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  config.names.reserve(joint_count_ * kInterfacesPerJoint);
  for (const auto& joint : joints_) {
    config.names.push_back(joint + "/" + hardware_interface::HW_IF_POSITION);
  }
  for (const auto& joint : joints_) {
    config.names.push_back(joint + "/" + hardware_interface::HW_IF_VELOCITY);
  }
  for (const auto& joint : joints_) {
    config.names.push_back(joint + "/" + hardware_interface::HW_IF_EFFORT);
  }
  return config;
}

controller_interface::InterfaceConfiguration
PlatoController::state_interface_configuration() const {
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  config.names.reserve(joint_count_ * kInterfacesPerJoint);
  for (const auto& joint : joints_) {
    config.names.push_back(joint + "/" + hardware_interface::HW_IF_POSITION);
  }
  for (const auto& joint : joints_) {
    config.names.push_back(joint + "/" + hardware_interface::HW_IF_VELOCITY);
  }
  for (const auto& joint : joints_) {
    config.names.push_back(joint + "/" + hardware_interface::HW_IF_EFFORT);
  }
  return config;
}

controller_interface::CallbackReturn PlatoController::on_configure(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  joints_ = get_node()->get_parameter("joints").as_string_array();
  joint_count_ = joints_.size();
  wbc_yaml_path_ = get_node()->get_parameter("wbc_yaml_path").as_string();
  control_frequency_hz_ = get_node()->get_parameter("control_frequency").as_double();
  if (!std::isfinite(control_frequency_hz_) || control_frequency_hz_ <= 0.0) {
    RCLCPP_ERROR(get_node()->get_logger(),
                 "[PlatoController] parameter 'control_frequency' must be finite and > 0.");
    return controller_interface::CallbackReturn::ERROR;
  }
  control_dt_ = 1.0 / control_frequency_hz_;
  if (joints_.empty()) {
    RCLCPP_ERROR(get_node()->get_logger(),
                 "[PlatoController] parameter 'joints' is empty.");
    return controller_interface::CallbackReturn::ERROR;
  }

  try {
    auto arch_config =
        wbc::ControlArchitectureConfig::FromYaml(wbc_yaml_path_, control_dt_);
    arch_config.state_provider = std::make_unique<wbc::StateProvider>(control_dt_);
    ctrl_arch_ = std::make_unique<wbc::ControlArchitecture>(std::move(arch_config));
    ctrl_arch_->Initialize();

    // Cache typed state pointers
    auto* fsm = ctrl_arch_->GetFsmHandler();
    if (const auto id = fsm->FindStateIdByName("joint_teleop")) {
      joint_teleop_state_ =
          dynamic_cast<wbc::JointTeleop*>(fsm->FindStateById(*id));
    }
    if (const auto id = fsm->FindStateIdByName("cartesian_teleop")) {
      cartesian_teleop_state_ =
          dynamic_cast<wbc::CartesianTeleop*>(fsm->FindStateById(*id));
    }
    if (const auto id = fsm->FindStateIdByName("fingertip_teleop")) {
      fingertip_teleop_state_ =
          dynamic_cast<wbc::PlatoFingertipTeleop*>(fsm->FindStateById(*id));
    }
    if (const auto id = fsm->FindStateIdByName("safe_command")) {
      safe_command_state_id_ = *id;
    }
  } catch (const std::exception& e) {
    ctrl_arch_.reset();
    RCLCPP_ERROR(get_node()->get_logger(),
                 "[PlatoController] failed to build control architecture: %s",
                 e.what());
    return controller_interface::CallbackReturn::ERROR;
  }

  // Pre-size RT buffers
  {
    const std::vector<double> zeros(joint_count_, 0.0);
    qdot_des_buf_.writeFromNonRT(JointVelRef{zeros, 0});
    q_des_buf_.writeFromNonRT(JointPosRef{zeros, 0});
  }
  ee_vel_buf_.writeFromNonRT(EEVelRef{});
  finger_vel_buf_.writeFromNonRT(EEVelRef{});

  // Joint velocity subscriber (all 15 joints)
  joint_vel_sub_ =
      get_node()->create_subscription<std_msgs::msg::Float64MultiArray>(
          "~/joint_vel_cmd", rclcpp::SensorDataQoS(),
          [this](std_msgs::msg::Float64MultiArray::ConstSharedPtr msg) {
            if (msg->data.size() != joint_count_) return;
            qdot_des_buf_.writeFromNonRT(
                JointVelRef{msg->data, get_node()->now().nanoseconds()});
          });

  // Joint position subscriber (all 15 joints)
  joint_pos_sub_ =
      get_node()->create_subscription<std_msgs::msg::Float64MultiArray>(
          "~/joint_pos_cmd", rclcpp::SensorDataQoS(),
          [this](std_msgs::msg::Float64MultiArray::ConstSharedPtr msg) {
            if (msg->data.size() != joint_count_) return;
            q_des_buf_.writeFromNonRT(
                JointPosRef{msg->data, get_node()->now().nanoseconds()});
          });

  // EE velocity subscriber (wrist Cartesian)
  ee_vel_sub_ =
      get_node()->create_subscription<geometry_msgs::msg::TwistStamped>(
          "~/ee_vel_cmd", rclcpp::SensorDataQoS(),
          [this](geometry_msgs::msg::TwistStamped::ConstSharedPtr msg) {
            ee_vel_buf_.writeFromNonRT(EEVelRef{
                {msg->twist.linear.x, msg->twist.linear.y, msg->twist.linear.z},
                {msg->twist.angular.x, msg->twist.angular.y, msg->twist.angular.z},
                rclcpp::Time(msg->header.stamp).nanoseconds()});
          });

  // Fingertip velocity subscriber (index fingertip Cartesian)
  finger_vel_sub_ =
      get_node()->create_subscription<geometry_msgs::msg::TwistStamped>(
          "~/finger_vel_cmd", rclcpp::SensorDataQoS(),
          [this](geometry_msgs::msg::TwistStamped::ConstSharedPtr msg) {
            finger_vel_buf_.writeFromNonRT(EEVelRef{
                {msg->twist.linear.x, msg->twist.linear.y, msg->twist.linear.z},
                {msg->twist.angular.x, msg->twist.angular.y, msg->twist.angular.z},
                rclcpp::Time(msg->header.stamp).nanoseconds()});
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
                "[PlatoController] Available states: [%s]", state_list.c_str());
  }

  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn PlatoController::on_activate(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  const std::size_t expected = joint_count_ * kInterfacesPerJoint;
  if (state_interfaces_.size() < expected || command_interfaces_.size() < expected) {
    RCLCPP_ERROR(get_node()->get_logger(),
                 "[PlatoController] missing interfaces. expected>=%zu (got state=%zu cmd=%zu)",
                 expected, state_interfaces_.size(), command_interfaces_.size());
    return controller_interface::CallbackReturn::ERROR;
  }

  robot_joint_state_.Reset(static_cast<Eigen::Index>(joint_count_));

  // Hold current hardware positions
  for (std::size_t i = 0; i < joint_count_; ++i) {
    (void)command_interfaces_[i].set_value(state_interfaces_[i].get_value());
  }
  for (std::size_t i = joint_count_; i < command_interfaces_.size(); ++i) {
    (void)command_interfaces_[i].set_value(0.0);
  }

  active_state_id_ = ctrl_arch_->GetCurrentStateId();

  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn PlatoController::on_deactivate(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  for (std::size_t i = joint_count_; i < command_interfaces_.size(); ++i) {
    (void)command_interfaces_[i].set_value(0.0);
  }
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::return_type PlatoController::update(
    const rclcpp::Time& time, const rclcpp::Duration& /*period*/) {
  if (!ctrl_arch_) return controller_interface::return_type::OK;

  // Dispatch teleop commands to active state
  if (joint_teleop_state_ && active_state_id_ == joint_teleop_state_->id()) {
    const auto* qdot_des = qdot_des_buf_.readFromRT();
    const auto* q_des = q_des_buf_.readFromRT();
    joint_teleop_state_->UpdateCommand(
        Eigen::Map<const Eigen::VectorXd>(qdot_des->qdot.data(), joint_count_),
        qdot_des->ts_ns,
        Eigen::Map<const Eigen::VectorXd>(q_des->q.data(), joint_count_),
        q_des->ts_ns);
  }

  if (cartesian_teleop_state_ && active_state_id_ == cartesian_teleop_state_->id()) {
    const auto* xdot_des = ee_vel_buf_.readFromRT();
    cartesian_teleop_state_->UpdateCommand(
        xdot_des->xdot, xdot_des->wdot, xdot_des->ts_ns);
  }

  if (fingertip_teleop_state_ && active_state_id_ == fingertip_teleop_state_->id()) {
    const auto* ee_vel = ee_vel_buf_.readFromRT();
    fingertip_teleop_state_->UpdateEECommand(
        ee_vel->xdot, ee_vel->wdot, ee_vel->ts_ns);
    const auto* finger_vel = finger_vel_buf_.readFromRT();
    fingertip_teleop_state_->UpdateFingerCommand(
        finger_vel->xdot, finger_vel->wdot, finger_vel->ts_ns);
  }

  ctrl_arch_->Update(ReadJointState(), time.seconds(), control_dt_);
  active_state_id_ = ctrl_arch_->GetCurrentStateId();

  WriteJointCommand(ctrl_arch_->GetCommand());

  return controller_interface::return_type::OK;
}

const wbc::RobotJointState& PlatoController::ReadJointState() {
  for (std::size_t i = 0; i < joint_count_; ++i) {
    robot_joint_state_.q[i] =
        state_interfaces_[InterfaceIndex(kPositionBlock, i, joint_count_)].get_value();
    robot_joint_state_.qdot[i] =
        state_interfaces_[InterfaceIndex(kVelocityBlock, i, joint_count_)].get_value();
    robot_joint_state_.tau[i] =
        state_interfaces_[InterfaceIndex(kEffortBlock, i, joint_count_)].get_value();
  }
  return robot_joint_state_;
}

void PlatoController::WriteJointCommand(const wbc::RobotCommand& cmd) {
  for (std::size_t i = 0; i < joint_count_; ++i) {
    (void)command_interfaces_[InterfaceIndex(kPositionBlock, i, joint_count_)].set_value(cmd.q[i]);
    (void)command_interfaces_[InterfaceIndex(kVelocityBlock, i, joint_count_)].set_value(cmd.qdot[i]);
    (void)command_interfaces_[InterfaceIndex(kEffortBlock, i, joint_count_)].set_value(cmd.tau[i]);
  }
}

}  // namespace plato_controller

PLUGINLIB_EXPORT_CLASS(
    plato_controller::PlatoController, controller_interface::ControllerInterface)
