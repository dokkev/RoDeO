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

#include <string>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <hardware_interface/types/hardware_interface_type_values.hpp>
#include <pluginlib/class_list_macros.hpp>
#include <rclcpp/rclcpp.hpp>

#include "wbc_util/ros_path_utils.hpp"

namespace optimo_controller
{

////////////////////////////////////////////////////////////////////////

CallbackReturn OptimoController::on_init()
{
  auto_declare<std::vector<std::string>>("joints", {});
  auto_declare<std::string>(
    "urdf_path", "package://optimo_description/urdf/optimo.urdf");
  return CallbackReturn::SUCCESS;
}

////////////////////////////////////////////////////////////////////////

CallbackReturn OptimoController::on_configure(const rclcpp_lifecycle::State &)
{
  joints_ = get_node()->get_parameter("joints").as_string_array();
  joint_count_ = joints_.size();
  if (joint_count_ == 0) {
    RCLCPP_ERROR(get_node()->get_logger(), "No joints configured.");
    return CallbackReturn::ERROR;
  }

  // Resolve URDF path (supports package:// syntax)
  const std::string urdf_param = get_node()->get_parameter("urdf_path").as_string();
  const std::string resolved_urdf = wbc::path::ResolvePackageUri(urdf_param);
  const std::string package_root = wbc::path::ResolveUrdfPackageRoot(
    urdf_param, resolved_urdf);

  RCLCPP_INFO(get_node()->get_logger(),
    "[OptimoController] Loading Pinocchio model from: %s", resolved_urdf.c_str());

  robot_ = std::make_unique<wbc::PinocchioRobotSystem>(
    resolved_urdf, package_root, /*fixed_base=*/true, /*print_info=*/true);

  // Pre-allocate state vectors
  q_.setZero(joint_count_);
  qdot_.setZero(joint_count_);

  RCLCPP_INFO(get_node()->get_logger(),
    "[OptimoController] Pinocchio gravity comp controller initialized (%zu joints).",
    joint_count_);

  return CallbackReturn::SUCCESS;
}

////////////////////////////////////////////////////////////////////////

controller_interface::InterfaceConfiguration
OptimoController::command_interface_configuration() const
{
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;

  for (const auto & joint : joints_) {
    config.names.push_back(joint + "/" + hardware_interface::HW_IF_EFFORT);
  }

  return config;
}

////////////////////////////////////////////////////////////////////////

controller_interface::InterfaceConfiguration
OptimoController::state_interface_configuration() const
{
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;

  for (const auto & joint : joints_) {
    config.names.push_back(joint + "/" + hardware_interface::HW_IF_POSITION);
    config.names.push_back(joint + "/" + hardware_interface::HW_IF_VELOCITY);
  }

  return config;
}

////////////////////////////////////////////////////////////////////////

controller_interface::return_type OptimoController::update(
  const rclcpp::Time &, const rclcpp::Duration &)
{
  // Read joint state from hardware interfaces.
  // State interface layout: [pos_0, vel_0, pos_1, vel_1, ...]
  for (std::size_t i = 0; i < joint_count_; ++i) {
    q_[i]    = state_interfaces_[2 * i].get_value();      // position [rad]
    qdot_[i] = state_interfaces_[2 * i + 1].get_value();  // velocity [rad/s]
  }

  // Update Pinocchio model with current joint state (all SI units).
  const Eigen::Vector3d zero3 = Eigen::Vector3d::Zero();
  const Eigen::Quaterniond identity = Eigen::Quaterniond::Identity();
  robot_->UpdateRobotModel(zero3, identity, zero3, zero3, q_, qdot_, false);

  // Compute gravity compensation torque [Nm].
  const Eigen::VectorXd& gravity = robot_->GetGravityRef();

  // Write gravity comp torque to effort command interfaces.
  // For fixed-base: gravity vector size = num_qdot = num_active_joints.
  const int n_active = robot_->NumActiveDof();
  for (std::size_t i = 0; i < joint_count_ && static_cast<int>(i) < n_active; ++i) {
    command_interfaces_[i].set_value(gravity[i]);
  }

  return controller_interface::return_type::OK;
}

}  // namespace optimo_controller

PLUGINLIB_EXPORT_CLASS(
  optimo_controller::OptimoController, controller_interface::ControllerInterface)
