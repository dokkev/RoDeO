#ifndef JOINT_IMPEDANCE_CONTROLLER__JOINT_IMPEDANCE_CONTROLLER_HPP_
#define JOINT_IMPEDANCE_CONTROLLER__JOINT_IMPEDANCE_CONTROLLER_HPP_

#include <memory>
#include <string>
#include <vector>

#include "controller_interface/controller_interface.hpp"
#include "rclcpp/subscription.hpp"
#include "rclcpp_lifecycle/state.hpp"
#include "realtime_tools/realtime_buffer.hpp"
#include "realtime_tools/realtime_publisher.hpp"
#include "wbc_msgs/msg/impedance_commands.hpp"
#include "wbc_msgs/msg/impedance_controller_state.hpp"

#include <joint_impedance_controller/joint_impedance_controller_parameters.hpp>

namespace joint_impedance_controller
{

using CmdType = wbc_msgs::msg::ImpedanceCommands;
using StateMsg = wbc_msgs::msg::ImpedanceControllerState;

class JointImpedanceController : public controller_interface::ControllerInterface
{
public:
  JointImpedanceController();

  controller_interface::CallbackReturn on_init() override;
  controller_interface::CallbackReturn on_configure(
    const rclcpp_lifecycle::State & previous_state) override;
  controller_interface::CallbackReturn on_activate(
    const rclcpp_lifecycle::State & previous_state) override;
  controller_interface::CallbackReturn on_deactivate(
    const rclcpp_lifecycle::State & previous_state) override;

  controller_interface::return_type update(
    const rclcpp::Time & time, const rclcpp::Duration & period) override;

  controller_interface::InterfaceConfiguration command_interface_configuration() const override;
  controller_interface::InterfaceConfiguration state_interface_configuration() const override;

private:
  std::shared_ptr<CmdType> make_hold_command() const;

  /// @brief Publish controller state (called from update loop)
  void publish_state(const rclcpp::Time & time, const CmdType& command);

  /// @brief Read state interfaces into member variables
  void read_state_interfaces();

  // Joint names
  std::vector<std::string> joint_names_;

  // State interface data (read from hardware)
  std::vector<double> positions_;
  std::vector<double> velocities_;
  std::vector<double> efforts_;
  std::vector<double> position_errors_;
  std::vector<double> velocity_errors_;
  std::vector<double> feedback_efforts_;
  std::vector<double> desired_efforts_;

  // Command interfaces (write to hardware)
  std::vector<std::reference_wrapper<hardware_interface::LoanedCommandInterface>>
    position_command_interfaces_;
  std::vector<std::reference_wrapper<hardware_interface::LoanedCommandInterface>>
    velocity_command_interfaces_;
  std::vector<std::reference_wrapper<hardware_interface::LoanedCommandInterface>>
    effort_command_interfaces_;
  std::vector<std::reference_wrapper<hardware_interface::LoanedCommandInterface>>
    stiffness_command_interfaces_;
  std::vector<std::reference_wrapper<hardware_interface::LoanedCommandInterface>>
    damping_command_interfaces_;

  // Real-time command buffer
  realtime_tools::RealtimeBuffer<std::shared_ptr<CmdType>> rt_command_ptr_;
  rclcpp::Subscription<CmdType>::SharedPtr joints_command_subscriber_;

  // State publisher
  using StatePublisher = realtime_tools::RealtimePublisher<StateMsg>;
  rclcpp::Publisher<StateMsg>::SharedPtr publisher_;
  std::unique_ptr<StatePublisher> state_publisher_;

  // Parameters
  std::shared_ptr<ParamListener> param_listener_;
  Params params_;
  size_t state_publish_divisor_ = 10;
  size_t state_publish_counter_ = 0;
};

}  // namespace joint_impedance_controller

#endif  // JOINT_IMPEDANCE_CONTROLLER__JOINT_IMPEDANCE_CONTROLLER_HPP_
