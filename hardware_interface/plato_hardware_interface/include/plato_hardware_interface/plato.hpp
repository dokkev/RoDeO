#ifndef PLATO_HARDWARE_INTERFACE__PLATO_HPP_
#define PLATO_HARDWARE_INTERFACE__PLATO_HPP_

#include <memory>
#include <string>
#include <vector>

#include <hardware_interface/hardware_info.hpp>
#include <hardware_interface/system_interface.hpp>
#include <hardware_interface/types/hardware_interface_return_values.hpp>
#include <hardware_interface/types/hardware_interface_type_values.hpp>

#include "plato_hardware_interface/plato_hand.hpp"
#include "plato_hardware_interface/utils/visibility_control.h"

namespace plato_hardware_interface
{

class PlatoHardware : public hardware_interface::SystemInterface
{
public:
  PLATO_HARDWARE_INTERFACE_PUBLIC
  hardware_interface::CallbackReturn on_init(const hardware_interface::HardwareInfo & info) override;

  PLATO_HARDWARE_INTERFACE_PUBLIC
  hardware_interface::CallbackReturn on_configure(
    const rclcpp_lifecycle::State & previous_state) override;

  PLATO_HARDWARE_INTERFACE_PUBLIC
  std::vector<hardware_interface::StateInterface> export_state_interfaces() override;

  PLATO_HARDWARE_INTERFACE_PUBLIC
  std::vector<hardware_interface::CommandInterface> export_command_interfaces() override;

  PLATO_HARDWARE_INTERFACE_PUBLIC
  hardware_interface::CallbackReturn on_activate(
    const rclcpp_lifecycle::State & previous_state) override;

  PLATO_HARDWARE_INTERFACE_PUBLIC
  hardware_interface::CallbackReturn on_deactivate(
    const rclcpp_lifecycle::State & previous_state) override;

  PLATO_HARDWARE_INTERFACE_PUBLIC
  hardware_interface::return_type read(
    const rclcpp::Time & time, const rclcpp::Duration & period) override;

  PLATO_HARDWARE_INTERFACE_PUBLIC
  hardware_interface::return_type write(
    const rclcpp::Time & time, const rclcpp::Duration & period) override;

private:
  std::unique_ptr<plato_hand::Hand> hand_;
  bool zeroing_requested_ = false;
};

}  // namespace plato_hardware_interface

#endif  // PLATO_HARDWARE_INTERFACE__PLATO_HPP_
