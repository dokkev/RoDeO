/**
 * @file hardware_interface/optimo_hardware_interface/include/optimo_hardware_interface/optimo_hardware_interface.hpp
 * @brief Doxygen documentation for optimo_hardware_interface module.
 */
#pragma once

#include <vector>

#include "hardware_interface/system_interface.hpp"

namespace optimo_hardware_interface
{

/**
 * @brief Minimal ros2_control system interface for Optimo hardware/simulation.
 */
class OptimoHardwareInterface : public hardware_interface::SystemInterface
{
public:
  hardware_interface::CallbackReturn on_init(
    const hardware_interface::HardwareComponentInterfaceParams & params) override;

  std::vector<hardware_interface::StateInterface> export_state_interfaces() override;
  std::vector<hardware_interface::CommandInterface> export_command_interfaces() override;

  hardware_interface::CallbackReturn on_configure(
    const rclcpp_lifecycle::State & previous_state) override;
  hardware_interface::CallbackReturn on_activate(
    const rclcpp_lifecycle::State & previous_state) override;
  hardware_interface::CallbackReturn on_deactivate(
    const rclcpp_lifecycle::State & previous_state) override;

  hardware_interface::return_type read(
    const rclcpp::Time & time, const rclcpp::Duration & period) override;
  hardware_interface::return_type write(
    const rclcpp::Time & time, const rclcpp::Duration & period) override;

private:
  std::vector<double> hw_positions_;
  std::vector<double> hw_velocities_;
  std::vector<double> hw_efforts_;
  std::vector<double> hw_position_commands_;
  std::vector<double> hw_velocity_commands_;
  std::vector<double> hw_effort_commands_;

  bool activated_{false};
};

}  // namespace optimo_hardware_interface
