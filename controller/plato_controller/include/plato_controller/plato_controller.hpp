#pragma once

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <controller_interface/controller_interface.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>
#include <realtime_tools/realtime_buffer.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>

#include "wbc_architecture/control_architecture.hpp"
#include "wbc_msgs/srv/transition_state.hpp"
#include "wbc_util/actuator_interface.hpp"

namespace wbc {
class JointTeleop;
class CartesianTeleop;
class PlatoFingertipTeleop;
}  // namespace wbc

namespace plato_controller {

class PlatoController : public controller_interface::ControllerInterface {
public:
  ~PlatoController() override;

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

  controller_interface::return_type update(
      const rclcpp::Time& time, const rclcpp::Duration& period) override;

private:
  // RT input structs
  struct JointVelRef {
    std::vector<double> qdot;
    int64_t ts_ns{0};
  };
  struct JointPosRef {
    std::vector<double> q;
    int64_t ts_ns{0};
  };
  struct EEVelRef {
    Eigen::Vector3d xdot{Eigen::Vector3d::Zero()};
    Eigen::Vector3d wdot{Eigen::Vector3d::Zero()};
    int64_t ts_ns{0};
  };

  static constexpr std::size_t kInterfacesPerJoint = 3U;
  static constexpr std::size_t kPositionBlock = 0U;
  static constexpr std::size_t kVelocityBlock = 1U;
  static constexpr std::size_t kEffortBlock = 2U;

  static constexpr std::size_t InterfaceIndex(
      const std::size_t block, const std::size_t joint_idx,
      const std::size_t joint_count) noexcept {
    return block * joint_count + joint_idx;
  }

  const wbc::RobotJointState& ReadJointState();
  void WriteJointCommand(const wbc::RobotCommand& cmd);

  std::vector<std::string> joints_;
  std::size_t joint_count_{0};
  std::string wbc_yaml_path_;
  double control_frequency_hz_{1000.0};
  double control_dt_{0.001};
  std::unique_ptr<wbc::ControlArchitecture> ctrl_arch_;
  wbc::RobotJointState robot_joint_state_;

  // Per-topic RT buffers
  realtime_tools::RealtimeBuffer<JointVelRef> qdot_des_buf_;
  realtime_tools::RealtimeBuffer<JointPosRef> q_des_buf_;
  realtime_tools::RealtimeBuffer<EEVelRef> ee_vel_buf_;
  realtime_tools::RealtimeBuffer<EEVelRef> finger_vel_buf_;

  // ROS subscribers
  rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr joint_vel_sub_;
  rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr joint_pos_sub_;
  rclcpp::Subscription<geometry_msgs::msg::TwistStamped>::SharedPtr ee_vel_sub_;
  rclcpp::Subscription<geometry_msgs::msg::TwistStamped>::SharedPtr finger_vel_sub_;

  // State transition service
  rclcpp::Service<wbc_msgs::srv::TransitionState>::SharedPtr set_state_srv_;

  // Typed state pointers — cached at configure time
  wbc::JointTeleop* joint_teleop_state_{nullptr};
  wbc::CartesianTeleop* cartesian_teleop_state_{nullptr};
  wbc::PlatoFingertipTeleop* fingertip_teleop_state_{nullptr};
  std::optional<int> safe_command_state_id_;

  // Active FSM state id
  wbc::StateId active_state_id_{-1};
};

}  // namespace plato_controller
