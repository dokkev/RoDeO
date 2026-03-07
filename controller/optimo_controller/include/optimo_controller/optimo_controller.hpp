/**
 * @file controller/optimo_controller/include/optimo_controller/optimo_controller.hpp
 * @brief Doxygen documentation for optimo_controller module.
 */
#pragma once

#include <memory>
#include <string>
#include <vector>

#include <controller_interface/controller_interface.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>
#include <realtime_tools/realtime_buffer.h>
#include <realtime_tools/realtime_publisher.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include "wbc_architecture/control_architecture.hpp"
#include "wbc_logger/wbc_logger.hpp"
#include "wbc_msgs/msg/wbc_state.hpp"
#include "wbc_msgs/srv/transition_state.hpp"
#include "wbc_util/actuator_interface.hpp"

namespace wbc {
class CartesianTeleop;
class JointTeleop;
}  // namespace wbc

namespace optimo_controller
{

/**
 * @brief Fixed-base Optimo ROS2 controller.
 *
 * @details
 * This controller intentionally keeps a simple fixed-base data path:
 * `state_interfaces -> RobotJointState -> ControlArchitecture::Update(joint, t, dt)`.
 */
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
  // Per-topic RT input structs.  Each subscriber owns one RealtimeBuffer —
  // data and timestamp travel together so they are always consistent.
  // ---------------------------------------------------------------------------
  struct JointVelRef {
    std::vector<double> qdot;        // [rad/s], pre-sized in on_configure()
    int64_t             ts_ns{0};    // arrival time [ns], 0 = never received
  };
  struct JointPosRef {
    std::vector<double> q;           // [rad],   pre-sized in on_configure()
    int64_t             ts_ns{0};
  };
  struct EEVelRef {
    Eigen::Vector3d xdot{Eigen::Vector3d::Zero()};   // linear  vel [m/s],   world frame
    Eigen::Vector3d wdot{Eigen::Vector3d::Zero()};   // angular vel [rad/s], world frame
    int64_t         ts_ns{0};        // from msg->header.stamp [ns]
  };
  struct EEPoseRef {
    Eigen::Vector3d    x{Eigen::Vector3d::Zero()};         // position    [m],     world frame
    Eigen::Quaterniond w{Eigen::Quaterniond::Identity()};  // orientation,          world frame
    int64_t            ts_ns{0};     // from msg->header.stamp [ns]
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

  // Per-topic RT buffers — pre-sized to joint_count_ in on_configure().
  // Timestamp is embedded in each struct so data and ts are always consistent.
  realtime_tools::RealtimeBuffer<JointVelRef> qdot_des_buf_;
  realtime_tools::RealtimeBuffer<JointPosRef> q_des_buf_;
  realtime_tools::RealtimeBuffer<EEVelRef>    xdot_des_buf_;
  realtime_tools::RealtimeBuffer<EEPoseRef>   x_des_buf_;

  // ROS subscribers (non-RT)
  rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr joint_vel_sub_;
  rclcpp::Subscription<geometry_msgs::msg::TwistStamped>::SharedPtr ee_vel_sub_;
  rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr joint_pos_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr  ee_pos_sub_;
  rclcpp::Service<wbc_msgs::srv::TransitionState>::SharedPtr       set_state_srv_;

  // Typed state pointers — cached at configure time, valid for controller lifetime.
  // Allows RT-safe direct dispatch without dynamic_cast in the hot path.
  wbc::JointTeleop* joint_teleop_state_{nullptr};
  wbc::CartesianTeleop* cartesian_teleop_state_{nullptr};

  // Active state id — refreshed after each ctrl_arch_->Update() so the FSM's
  // latest state (including auto-transitions) is observed before the next tick's dispatch.
  wbc::StateId active_state_id_{-1};

  // RT-safe WBC state publisher — non-blocking trylock in the hot path.
  std::shared_ptr<realtime_tools::RealtimePublisher<wbc_msgs::msg::WbcState>> rt_wbc_pub_;

  /// Copy WbcStateData → WbcState msg and try-publish (non-blocking).
  void PublishWbcState(const rclcpp::Time& time);
};

}  // namespace optimo_controller
