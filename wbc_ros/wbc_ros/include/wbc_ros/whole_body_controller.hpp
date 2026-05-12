// Copyright 2026
//
// Licensed under the Apache License, Version 2.0.

#pragma once

#include <memory>
#include <string>
#include <vector>

#include <controller_interface/controller_interface.hpp>

#include "control_architecture/control_architecture.hpp"
#include "control_architecture/state_machine/robot_control_profile.hpp"
#include "wbc_core/adapters/command-adapter.hpp"
#include "wbc_core/robots/robot-system.hpp"
#include "wbc_core/utils/actuator_interface.hpp"

namespace wbc_ros {

class WholeBodyController : public controller_interface::ControllerInterface {
 public:
  ~WholeBodyController() override;

  controller_interface::CallbackReturn on_init() override;

  controller_interface::InterfaceConfiguration command_interface_configuration()
      const override;

  controller_interface::InterfaceConfiguration state_interface_configuration()
      const override;

  controller_interface::CallbackReturn on_configure(
      const rclcpp_lifecycle::State& previous_state) override;

  controller_interface::CallbackReturn on_activate(
      const rclcpp_lifecycle::State& previous_state) override;

  controller_interface::CallbackReturn on_deactivate(
      const rclcpp_lifecycle::State& previous_state) override;

  controller_interface::return_type update(
      const rclcpp::Time& time, const rclcpp::Duration& period) override;

 protected:
  virtual std::unique_ptr<wbc::RobotControlProfile> CreateControlProfile();

 private:
  static constexpr std::size_t kPositionBlock = 0U;
  static constexpr std::size_t kVelocityBlock = 1U;

  static constexpr std::size_t InterfaceIndex(
      std::size_t block, std::size_t joint_idx,
      std::size_t joint_count) noexcept {
    return block * joint_count + joint_idx;
  }

  controller_interface::InterfaceConfiguration JointInterfaceConfiguration(
      const std::vector<std::string>& interface_names) const;
  controller_interface::CallbackReturn ConfigureRuntime(
      const std::string& yaml_path);
  bool ConfigureCommandInterfaces();
  bool ConfigureCommandGains();
  bool ConfigureActuator();
  bool PrepareOutputCommand(const wbc::LowLevelCommand& cmd, double dt);
  void UpdateDebugStats(double time_sec);
  void LogAvailableStates() const;

  bool ReadRobotState(double time_sec);
  bool WriteJointCommand(const wbc::LowLevelCommand& cmd);
  bool WriteSafeCommand();
  controller_interface::return_type HandleRuntimeFault(const char* message);

  std::vector<std::string> joints_;
  std::vector<std::string> command_interface_names_;
  std::size_t joint_count_{0};
  double control_dt_{0.001};

  std::shared_ptr<wbc::robots::RobotSystem> robot_;
  std::unique_ptr<wbc::ControlArchitecture> ctrl_arch_;
  std::unique_ptr<wbc::RobotControlProfile> control_profile_;
  std::unique_ptr<wbc::ActuatorInterface> actuator_;
  wbc::robots::RobotState robot_state_;
  wbc::LowLevelCommand output_cmd_;
  wbc::LowLevelCommand safe_cmd_;
  Eigen::VectorXd command_kp_;
  Eigen::VectorXd command_kd_;

  bool debug_mode_{false};
  bool runtime_faulted_{false};
  double debug_print_interval_s_{5.0};
  double last_debug_print_time_{0.0};
  double max_tick_us_{0.0};
};

}  // namespace wbc_ros
