// Copyright 2024 Roboligent, Inc.
//
// Licensed under the Apache License, Version 2.0.

#include <memory>

#include <pluginlib/class_list_macros.hpp>

#include "control_architecture/state_machine/robot_control_profile.hpp"
#include "control_architecture/state_machine/state_factory.hpp"
#include "optimo_controller/state_machines/cartesian_reference.hpp"
#include "optimo_controller/state_machines/initialize.hpp"
#include "optimo_controller/state_machines/joint_reference.hpp"
#include "wbc_ros/whole_body_controller.hpp"

namespace optimo_controller {

void RegisterOptimoStates(wbc::StateFactory& factory) {
  factory.Register<state_machines::InitializeState>();
  factory.Register<state_machines::JointReferenceState>();
  factory.Register<state_machines::CartesianReferenceState>();
}

class OptimoController final : public wbc_ros::WholeBodyController {
 protected:
  std::unique_ptr<wbc::RobotControlProfile> CreateControlProfile() override {
    return std::make_unique<wbc::RobotControlProfile>(&RegisterOptimoStates);
  }
};

}  // namespace optimo_controller

PLUGINLIB_EXPORT_CLASS(optimo_controller::OptimoController,
                       controller_interface::ControllerInterface)
