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
#include <optional>
#include <string>
#include <vector>

#include <controller_interface/controller_interface.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>
#include <realtime_tools/realtime_buffer.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>

#include "wbc_architecture/control_architecture.hpp"
#include "wbc_msgs/srv/transition_state.hpp"
#include "wbc_util/actuator_interface.hpp"

namespace wbc {
class CartesianTeleop;
class JointTeleop;
}  // namespace wbc

namespace optimo_controller
{

class OptimoController : public controller_interface::ControllerInterface
{
public:
  ~OptimoController() override;

  controller_interface::CallbackReturn on_init() override;

  controller_interface::InterfaceConfiguration
  command_interface_configuration() const override;

  controller_interface::InterfaceConfiguration
  state_interface_configuration() const override;

  controller_interface::CallbackReturn
  on_configure(const rclcpp_lifecycle::State & previous_state) override;

  controller_interface::CallbackReturn
  on_activate(const rclcpp_lifecycle::State & previous_state) override;

  controller_interface::CallbackReturn
  on_deactivate(const rclcpp_lifecycle::State & previous_state) override;

  controller_interface::return_type update(
    const rclcpp::Time & time, const rclcpp::Duration & period) override;

private:
  // ---------------------------------------------------------------------------
  // Per-topic RT input structs.
  // ---------------------------------------------------------------------------
  struct JointVelRef {
    std::vector<double> qdot;
    int64_t             ts_ns{0};
  };
  struct JointPosRef {
    std::vector<double> q;
    int64_t             ts_ns{0};
  };
  struct EEVelRef {
    Eigen::Vector3d xdot{Eigen::Vector3d::Zero()};
    Eigen::Vector3d wdot{Eigen::Vector3d::Zero()};
    int64_t         ts_ns{0};
  };
  struct EEPoseRef {
    Eigen::Vector3d    x{Eigen::Vector3d::Zero()};
    Eigen::Quaterniond w{Eigen::Quaterniond::Identity()};
    int64_t            ts_ns{0};
  };

  static constexpr std::size_t kInterfacesPerJoint = 3U;
  static constexpr std::size_t kPositionBlock = 0U;
  static constexpr std::size_t kVelocityBlock = 1U;
  static constexpr std::size_t kEffortBlock = 2U;

  static constexpr std::size_t InterfaceIndex(
    const std::size_t block, const std::size_t joint_idx, const std::size_t joint_count) noexcept
  {
    return block * joint_count + joint_idx;
  }

  const wbc::RobotJointState & ReadJointState();
  void WriteJointCommand(const wbc::RobotCommand & cmd);

  std::vector<std::string> joints_;
  std::size_t joint_count_{0};
  std::string wbc_yaml_path_;
  double control_frequency_hz_{1000.0};
  double control_dt_{0.001};
  std::unique_ptr<wbc::ControlArchitecture> ctrl_arch_;
  std::unique_ptr<wbc::ActuatorInterface> actuator_;
  wbc::RobotJointState robot_joint_state_;

  // Per-topic RT buffers
  realtime_tools::RealtimeBuffer<JointVelRef> qdot_des_buf_;
  realtime_tools::RealtimeBuffer<JointPosRef> q_des_buf_;
  realtime_tools::RealtimeBuffer<EEVelRef>    xdot_des_buf_;
  realtime_tools::RealtimeBuffer<EEPoseRef>   x_des_buf_;

  // ROS subscribers
  rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr joint_vel_sub_;
  rclcpp::Subscription<geometry_msgs::msg::TwistStamped>::SharedPtr ee_vel_sub_;
  rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr joint_pos_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr  ee_pos_sub_;

  // State transition service
  rclcpp::Service<wbc_msgs::srv::TransitionState>::SharedPtr set_state_srv_;

  // Typed state pointers — cached at configure time
  wbc::JointTeleop* joint_teleop_state_{nullptr};
  wbc::CartesianTeleop* cartesian_teleop_state_{nullptr};
  std::optional<int> safe_command_state_id_;

  // Active FSM state id
  wbc::StateId active_state_id_{-1};

  // Debug mode: periodic QP/timing status print
  bool debug_mode_{false};
  double debug_print_interval_s_{5.0};
  double last_debug_print_time_{0.0};
  uint64_t tick_count_{0};
  double max_tick_us_{0.0};
};

}  // namespace optimo_controller
