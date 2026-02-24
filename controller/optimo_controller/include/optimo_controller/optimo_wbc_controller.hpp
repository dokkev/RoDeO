#pragma once

#include <memory>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <controller_interface/controller_interface.hpp>

#include "wbc_architecture/interface/control_architecture.hpp"
#include "wbc_robot_system/pinocchio_robot_system.hpp"

namespace optimo_controller {

class OptimoWbcController : public controller_interface::ControllerInterface {
public:
  controller_interface::CallbackReturn on_init() override;

  controller_interface::InterfaceConfiguration
  command_interface_configuration() const override;

  controller_interface::InterfaceConfiguration
  state_interface_configuration() const override;

  controller_interface::CallbackReturn
  on_configure(const rclcpp_lifecycle::State& previous_state) override;

  controller_interface::CallbackReturn
  on_activate(const rclcpp_lifecycle::State& previous_state) override;

  controller_interface::CallbackReturn
  on_deactivate(const rclcpp_lifecycle::State& previous_state) override;

  controller_interface::return_type update(const rclcpp::Time& time,
                                           const rclcpp::Duration& period) override;

private:
  static std::string ResolvePackageUri(const std::string& uri);
  static std::string ResolvePackageRootForUri(const std::string& uri);

  bool ReadJointState(Eigen::VectorXd& q, Eigen::VectorXd& qdot) const;
  bool WriteCommand(const wbc::RobotCommand& cmd);

  std::vector<std::string> joints_;

  bool fixed_base_{true};
  double fallback_dt_{0.001};
  std::string urdf_path_;
  std::string package_root_;
  std::string wbc_yaml_path_;

  std::unique_ptr<wbc::PinocchioRobotSystem> robot_;
  std::unique_ptr<wbc::ControlArchitecture> wbc_arch_;

  Eigen::VectorXd q_;
  Eigen::VectorXd qdot_;
};

} // namespace optimo_controller
