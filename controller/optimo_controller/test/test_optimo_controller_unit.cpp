/**
 * @file controller/optimo_controller/test/test_optimo_controller_unit.cpp
 * @brief Doxygen documentation for test_optimo_controller_unit module.
 */
#include <gtest/gtest.h>

#include <vector>

#include <controller_interface/controller_interface.hpp>
#include <hardware_interface/loaned_command_interface.hpp>
#include <hardware_interface/loaned_state_interface.hpp>
#include <hardware_interface/types/hardware_interface_type_values.hpp>
#include <rclcpp/duration.hpp>
#include <rclcpp/time.hpp>

#define private public
#include "optimo_controller/optimo_controller.hpp"
#undef private

namespace {

class TestableOptimoController : public optimo_controller::OptimoController {
public:
  void SetStateInterfaces(
      std::vector<hardware_interface::LoanedStateInterface>&& interfaces) {
    state_interfaces_ = std::move(interfaces);
  }

  void SetCommandInterfaces(
      std::vector<hardware_interface::LoanedCommandInterface>&& interfaces) {
    command_interfaces_ = std::move(interfaces);
  }
};

TEST(OptimoControllerTest, CommandInterfaceConfigurationUsesJointOrder) {
  TestableOptimoController controller;
  controller.joints_ = {"joint1", "joint2"};

  const auto config = controller.command_interface_configuration();
  EXPECT_EQ(config.type,
            controller_interface::interface_configuration_type::INDIVIDUAL);

  const std::vector<std::string> expected = {
      "joint1/position", "joint2/position", "joint1/velocity",
      "joint2/velocity", "joint1/effort",   "joint2/effort"};
  EXPECT_EQ(config.names, expected);
}

TEST(OptimoControllerTest, StateInterfaceConfigurationUsesJointOrder) {
  TestableOptimoController controller;
  controller.joints_ = {"joint1", "joint2"};

  const auto config = controller.state_interface_configuration();
  EXPECT_EQ(config.type,
            controller_interface::interface_configuration_type::INDIVIDUAL);

  const std::vector<std::string> expected = {
      "joint1/position", "joint2/position", "joint1/velocity",
      "joint2/velocity", "joint1/effort",   "joint2/effort"};
  EXPECT_EQ(config.names, expected);
}

TEST(OptimoControllerTest, UpdateReturnsOkWithoutModifyingCommands) {
  TestableOptimoController controller;
  controller.joints_ = {"joint1", "joint2"};

  double q1 = 0.1;
  double q2 = -0.2;
  double qdot1 = 1.2;
  double qdot2 = -1.1;
  double tau1 = 0.3;
  double tau2 = -0.4;

  std::vector<hardware_interface::StateInterface::ConstSharedPtr> owned_state_ifaces;
  std::vector<hardware_interface::LoanedStateInterface> loaned_state_ifaces;
  auto add_state_iface = [&](const std::string& joint,
                             const std::string& iface_name, double* value_ptr) {
    owned_state_ifaces.emplace_back(
        std::make_shared<hardware_interface::StateInterface>(
            joint, iface_name, value_ptr));
    loaned_state_ifaces.emplace_back(owned_state_ifaces.back());
  };
  add_state_iface("joint1", hardware_interface::HW_IF_POSITION, &q1);
  add_state_iface("joint2", hardware_interface::HW_IF_POSITION, &q2);
  add_state_iface("joint1", hardware_interface::HW_IF_VELOCITY, &qdot1);
  add_state_iface("joint2", hardware_interface::HW_IF_VELOCITY, &qdot2);
  add_state_iface("joint1", hardware_interface::HW_IF_EFFORT, &tau1);
  add_state_iface("joint2", hardware_interface::HW_IF_EFFORT, &tau2);
  controller.SetStateInterfaces(std::move(loaned_state_ifaces));

  double cmd_q1 = 0.0;
  double cmd_q2 = 0.0;
  double cmd_qdot1 = 5.0;
  double cmd_qdot2 = 5.0;
  double cmd_tau1 = 5.0;
  double cmd_tau2 = 5.0;

  std::vector<hardware_interface::CommandInterface::SharedPtr> owned_cmd_ifaces;
  std::vector<hardware_interface::LoanedCommandInterface> loaned_cmd_ifaces;
  auto add_cmd_iface = [&](const std::string& joint,
                           const std::string& iface_name, double* value_ptr) {
    owned_cmd_ifaces.emplace_back(
        std::make_shared<hardware_interface::CommandInterface>(
            joint, iface_name, value_ptr));
    loaned_cmd_ifaces.emplace_back(owned_cmd_ifaces.back(), nullptr);
  };
  add_cmd_iface("joint1", hardware_interface::HW_IF_POSITION, &cmd_q1);
  add_cmd_iface("joint2", hardware_interface::HW_IF_POSITION, &cmd_q2);
  add_cmd_iface("joint1", hardware_interface::HW_IF_VELOCITY, &cmd_qdot1);
  add_cmd_iface("joint2", hardware_interface::HW_IF_VELOCITY, &cmd_qdot2);
  add_cmd_iface("joint1", hardware_interface::HW_IF_EFFORT, &cmd_tau1);
  add_cmd_iface("joint2", hardware_interface::HW_IF_EFFORT, &cmd_tau2);
  controller.SetCommandInterfaces(std::move(loaned_cmd_ifaces));

  const auto ret =
      controller.update(rclcpp::Time(0, 0, RCL_ROS_TIME), rclcpp::Duration::from_seconds(0.001));
  EXPECT_EQ(ret, controller_interface::return_type::OK);
  EXPECT_DOUBLE_EQ(cmd_q1, 0.0);
  EXPECT_DOUBLE_EQ(cmd_q2, 0.0);
  EXPECT_DOUBLE_EQ(cmd_qdot1, 5.0);
  EXPECT_DOUBLE_EQ(cmd_qdot2, 5.0);
  EXPECT_DOUBLE_EQ(cmd_tau1, 5.0);
  EXPECT_DOUBLE_EQ(cmd_tau2, 5.0);
}

} // namespace
