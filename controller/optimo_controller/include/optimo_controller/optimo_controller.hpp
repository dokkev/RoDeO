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

#pragma once

#include <memory>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include <controller_interface/controller_interface.hpp>
#include <rclcpp/duration.hpp>
#include <rclcpp/time.hpp>

#include "wbc_robot_system/pinocchio_robot_system.hpp"

using CallbackReturn = rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;

namespace optimo_controller
{

class OptimoController : public controller_interface::ControllerInterface
{
public:
  CallbackReturn on_init() override;
  CallbackReturn on_configure(const rclcpp_lifecycle::State &) override;
  controller_interface::InterfaceConfiguration command_interface_configuration() const override;
  controller_interface::InterfaceConfiguration state_interface_configuration() const override;
  controller_interface::return_type update(const rclcpp::Time &, const rclcpp::Duration &) override;

private:
  // Robot model (Pinocchio) for gravity compensation
  std::unique_ptr<wbc::PinocchioRobotSystem> robot_;

  // Joint configuration
  std::vector<std::string> joints_;
  std::size_t joint_count_{0};

  // Pre-allocated state vectors (SI: rad, rad/s)
  Eigen::VectorXd q_;
  Eigen::VectorXd qdot_;
};

}  // namespace optimo_controller
