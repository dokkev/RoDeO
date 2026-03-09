/**
 * @file controller/optimo_controller/include/optimo_controller/optimo_controller.hpp
 * @brief Doxygen documentation for optimo_controller module.
 */
#pragma once

#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <controller_interface/controller_interface.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>
#include <realtime_tools/realtime_buffer.hpp>
#include <realtime_tools/realtime_publisher.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include "wbc_architecture/control_architecture.hpp"
#include "wbc_logger/wbc_logger.hpp"
#include "wbc_msgs/msg/wbc_state.hpp"
#include "wbc_msgs/srv/residual_dynamics_service.hpp"
#include "wbc_msgs/srv/task_gain_service.hpp"
#include "wbc_msgs/srv/task_weight_service.hpp"
#include "wbc_msgs/srv/transition_state.hpp"
#include "wbc_util/actuator_interface.hpp"

// Roboligent SDK — gravity compensation bootstrap for hardware enabling.
namespace roboligent { class Model; class RobotConfiguration; }

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
  struct TaskGainUpdate {
    std::vector<std::string> task_names;
    std::vector<double> kp;
    std::vector<double> kd;
    int64_t ts_ns{0};
  };
  struct TaskWeightUpdate {
    std::vector<std::string> task_names;
    std::vector<double> weight;
    int64_t ts_ns{0};
  };
  struct ResidualDynamicsUpdate {
    bool friction_enabled{false};
    std::vector<double> gamma_c;
    std::vector<double> gamma_v;
    std::vector<double> max_f_c;
    std::vector<double> max_f_v;
    bool observer_enabled{false};
    std::vector<double> k_o;
    std::vector<double> max_tau_dist;
    int64_t ts_ns{0};
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

  // Index of the model_safety_error command interface.
  // Layout: [pos×n][vel×n][effort×n][model_safety_error]
  std::size_t ModelSafetyErrorCmdIndex() const noexcept
  {
    return joint_count_ * kInterfacesPerJoint;
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
  rclcpp::Service<wbc_msgs::srv::TransitionState>::SharedPtr         set_state_srv_;
  rclcpp::Service<wbc_msgs::srv::TaskGainService>::SharedPtr         task_gain_srv_;
  rclcpp::Service<wbc_msgs::srv::TaskWeightService>::SharedPtr       task_weight_srv_;
  rclcpp::Service<wbc_msgs::srv::ResidualDynamicsService>::SharedPtr residual_dyn_srv_;

  // Non-RT service requests are latched and consumed in update() (RT thread).
  realtime_tools::RealtimeBuffer<TaskGainUpdate>         task_gain_update_buf_;
  realtime_tools::RealtimeBuffer<TaskWeightUpdate>       task_weight_update_buf_;
  realtime_tools::RealtimeBuffer<ResidualDynamicsUpdate> residual_update_buf_;
  int64_t last_task_gain_update_ts_{0};
  int64_t last_task_weight_update_ts_{0};
  int64_t last_residual_update_ts_{0};

  // Tuned scalar maps keyed by task name. Reapplied on each state transition.
  // Pre-populated with NaN sentinel at configure time so runtime updates never
  // insert new keys (which would heap-allocate in the RT loop).
  std::unordered_map<std::string, double> tuned_task_kp_;
  std::unordered_map<std::string, double> tuned_task_kd_;
  std::unordered_map<std::string, double> tuned_task_weight_;
  wbc::StateId last_tuned_state_id_{-1};
  // Pre-allocated scratch for ReapplyTunedTaskParams (avoids VectorXd::Constant alloc).
  Eigen::VectorXd reapply_scratch_;

  // Typed state pointers — cached at configure time, valid for controller lifetime.
  // Allows RT-safe direct dispatch without dynamic_cast in the hot path.
  wbc::JointTeleop* joint_teleop_state_{nullptr};
  wbc::CartesianTeleop* cartesian_teleop_state_{nullptr};
  // State id for the safe_command FSM state. When the active state matches this,
  // model_safety_error command interface is set to 1.0 to trigger hardware disable.
  std::optional<int> safe_command_state_id_;

  // Active state id — refreshed after each ctrl_arch_->Update() so the FSM's
  // latest state (including auto-transitions) is observed before the next tick's dispatch.
  wbc::StateId active_state_id_{-1};

  // RT-safe WBC state publisher — non-blocking trylock in the hot path.
  std::shared_ptr<realtime_tools::RealtimePublisher<wbc_msgs::msg::WbcState>> rt_wbc_pub_;

  /// Copy WbcStateData → WbcState msg and try-publish (non-blocking).
  void PublishWbcState(const rclcpp::Time& time);
  void ApplyPendingRuntimeUpdates();
  void ReapplyTunedTaskParams();

  // Roboligent Model for gravity compensation bootstrap.
  // Provides non-zero torques on early ticks so hardware's check_for_controller() passes.
  std::shared_ptr<roboligent::Model> rl_model_;
  std::vector<double> rl_pos_deg_;    // model update scratch: joint pos [deg]
  std::vector<double> rl_vel_deg_;    // model update scratch: joint vel [deg/s]
  std::vector<int>    rl_trq_ref_;    // model update scratch: torque ref [mNm]
  bool rl_model_first_run_{true};     // first-run flag for model reset
  int  robot_index_{0};               // EtherCAT master index for config path
};

}  // namespace optimo_controller
