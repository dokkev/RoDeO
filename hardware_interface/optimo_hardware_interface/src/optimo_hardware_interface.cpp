/**
 * @file hardware_interface/optimo_hardware_interface/src/optimo_hardware_interface.cpp
 * @brief Doxygen documentation for optimo_hardware_interface module.
 */
#include "optimo_hardware_interface/optimo_hardware_interface.hpp"

#include <algorithm>
#include <vector>

#include "hardware_interface/types/hardware_interface_type_values.hpp"
#include "pluginlib/class_list_macros.hpp"

namespace optimo_hardware_interface
{

hardware_interface::CallbackReturn OptimoHardwareInterface::on_init(
  const hardware_interface::HardwareComponentInterfaceParams & params)
{
  if (hardware_interface::SystemInterface::on_init(params) !=
    hardware_interface::CallbackReturn::SUCCESS)
  {
    return hardware_interface::CallbackReturn::ERROR;
  }

  if (info_.joints.empty()) {
    RCLCPP_ERROR(get_logger(), "No joints defined in hardware info.");
    return hardware_interface::CallbackReturn::ERROR;
  }

  const std::size_t n = info_.joints.size();
  hw_positions_.assign(n, 0.0);
  hw_velocities_.assign(n, 0.0);
  hw_efforts_.assign(n, 0.0);
  hw_position_commands_.assign(n, 0.0);
  hw_velocity_commands_.assign(n, 0.0);
  hw_effort_commands_.assign(n, 0.0);

  activated_ = false;

  RCLCPP_INFO(get_logger(), "Initialized OptimoHardwareInterface with %zu joints.", n);
  return hardware_interface::CallbackReturn::SUCCESS;
}

std::vector<hardware_interface::StateInterface>
OptimoHardwareInterface::export_state_interfaces()
{
  std::vector<hardware_interface::StateInterface> state_interfaces;
  state_interfaces.reserve(info_.joints.size() * 3U);

  for (std::size_t i = 0; i < info_.joints.size(); ++i) {
    const auto & name = info_.joints[i].name;
    state_interfaces.emplace_back(
      name, hardware_interface::HW_IF_POSITION, &hw_positions_[i]);
    state_interfaces.emplace_back(
      name, hardware_interface::HW_IF_VELOCITY, &hw_velocities_[i]);
    state_interfaces.emplace_back(
      name, hardware_interface::HW_IF_EFFORT, &hw_efforts_[i]);
  }
  return state_interfaces;
}

std::vector<hardware_interface::CommandInterface>
OptimoHardwareInterface::export_command_interfaces()
{
  std::vector<hardware_interface::CommandInterface> command_interfaces;
  command_interfaces.reserve(info_.joints.size() * 3U);

  for (std::size_t i = 0; i < info_.joints.size(); ++i) {
    command_interfaces.emplace_back(
      info_.joints[i].name, hardware_interface::HW_IF_POSITION, &hw_position_commands_[i]);
  }
  for (std::size_t i = 0; i < info_.joints.size(); ++i) {
    command_interfaces.emplace_back(
      info_.joints[i].name, hardware_interface::HW_IF_VELOCITY, &hw_velocity_commands_[i]);
  }
  for (std::size_t i = 0; i < info_.joints.size(); ++i) {
    command_interfaces.emplace_back(
      info_.joints[i].name, hardware_interface::HW_IF_EFFORT, &hw_effort_commands_[i]);
  }
  return command_interfaces;
}

hardware_interface::CallbackReturn OptimoHardwareInterface::on_configure(
  const rclcpp_lifecycle::State & /*previous_state*/)
{
  std::fill(hw_positions_.begin(), hw_positions_.end(), 0.0);
  std::fill(hw_velocities_.begin(), hw_velocities_.end(), 0.0);
  std::fill(hw_efforts_.begin(), hw_efforts_.end(), 0.0);
  std::fill(hw_position_commands_.begin(), hw_position_commands_.end(), 0.0);
  std::fill(hw_velocity_commands_.begin(), hw_velocity_commands_.end(), 0.0);
  std::fill(hw_effort_commands_.begin(), hw_effort_commands_.end(), 0.0);
  activated_ = false;
  return hardware_interface::CallbackReturn::SUCCESS;
}

hardware_interface::CallbackReturn OptimoHardwareInterface::on_activate(
  const rclcpp_lifecycle::State & /*previous_state*/)
{
  hw_position_commands_ = hw_positions_;
  hw_velocity_commands_ = hw_velocities_;
  hw_effort_commands_ = hw_efforts_;
  activated_ = true;
  return hardware_interface::CallbackReturn::SUCCESS;
}

hardware_interface::CallbackReturn OptimoHardwareInterface::on_deactivate(
  const rclcpp_lifecycle::State & /*previous_state*/)
{
  activated_ = false;
  std::fill(hw_position_commands_.begin(), hw_position_commands_.end(), 0.0);
  std::fill(hw_velocity_commands_.begin(), hw_velocity_commands_.end(), 0.0);
  std::fill(hw_effort_commands_.begin(), hw_effort_commands_.end(), 0.0);
  std::fill(hw_velocities_.begin(), hw_velocities_.end(), 0.0);
  std::fill(hw_efforts_.begin(), hw_efforts_.end(), 0.0);
  return hardware_interface::CallbackReturn::SUCCESS;
}

hardware_interface::return_type OptimoHardwareInterface::read(
  const rclcpp::Time & /*time*/, const rclcpp::Duration & /*period*/)
{
  for (std::size_t i = 0; i < hw_positions_.size(); ++i) {
    if (activated_) {
      hw_positions_[i] = hw_position_commands_[i];
      hw_velocities_[i] = hw_velocity_commands_[i];
      hw_efforts_[i] = hw_effort_commands_[i];
    } else {
      hw_velocities_[i] = 0.0;
      hw_efforts_[i] = 0.0;
    }
  }
  return hardware_interface::return_type::OK;
}

hardware_interface::return_type OptimoHardwareInterface::write(
  const rclcpp::Time & /*time*/, const rclcpp::Duration & /*period*/)
{
  return hardware_interface::return_type::OK;
}

}  // namespace optimo_hardware_interface

PLUGINLIB_EXPORT_CLASS(
  optimo_hardware_interface::OptimoHardwareInterface,
  hardware_interface::SystemInterface)
