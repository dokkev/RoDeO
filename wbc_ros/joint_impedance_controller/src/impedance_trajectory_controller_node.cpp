#include <memory>
#include <vector>
#include <chrono>
#include <algorithm>
#include <stdexcept>
#include <string>

#include "ament_index_cpp/get_package_share_directory.hpp"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float64.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "wbc_msgs/msg/impedance_commands.hpp"
#include "joint_impedance_controller/impedance_trajectory_controller.hpp"
#include "joint_impedance_controller/impedance_handler.hpp"

class ImpedanceTrajectoryControllerNode : public rclcpp::Node {
public:
  ImpedanceTrajectoryControllerNode()
    : Node("impedance_trajectory_controller_node"),
      controller_(8),
      steady_clock_(RCL_STEADY_TIME) {
    const auto hand_namespace = this->declare_parameter<std::string>(
        "hand_namespace", "plato2");
    const auto impedance_preset_yaml_path = this->declare_parameter<std::string>(
        "impedance_preset_yaml_path",
        ament_index_cpp::get_package_share_directory("joint_impedance_controller") +
            "/config/impedance_preset.yaml");
    const auto default_impedance_level = this->declare_parameter<double>(
        "default_impedance_level", 6.0);
    const auto impedance_level_topic = this->declare_parameter<std::string>(
        "impedance_level_topic", "~/impedance_level");
    const auto position_topic = this->declare_parameter<std::string>(
        "position_command_topic",
        resolve_hand_topic_(hand_namespace, "/joint_impedance_trajectory_controller/commands"));
    const auto goal_command_topic = this->declare_parameter<std::string>(
        "goal_command_topic",
        resolve_hand_topic_(hand_namespace, "/joint_impedance_trajectory_controller/goal_command"));
    const auto joint_state_topic = this->declare_parameter<std::string>(
        "joint_state_topic", resolve_hand_topic_(hand_namespace, "/joint_states"));
    const auto impedance_topic = this->declare_parameter<std::string>(
        "impedance_command_topic",
        resolve_hand_topic_(hand_namespace, "/joint_impedance_controller/commands"));
    default_goal_duration_sec_ = this->declare_parameter<double>("default_goal_duration_sec", 0.25);
    const double control_rate_hz = this->declare_parameter<double>("control_rate_hz", 100.0);

    impedance_handler_ = std::make_unique<joint_impedance_controller::ImpedanceHandler>(
        static_cast<int>(controller_.dof()), impedance_preset_yaml_path);
    std::string error;
    if (!impedance_handler_->set_level(default_impedance_level, &error)) {
      throw std::runtime_error(
          "Failed to set default impedance level " + std::to_string(default_impedance_level) +
          "': " + error);
    }

    impedance_level_sub_ = this->create_subscription<std_msgs::msg::Float64>(
        impedance_level_topic, 10,
        std::bind(&ImpedanceTrajectoryControllerNode::impedanceLevelCallback, this, std::placeholders::_1));

    position_sub_ = this->create_subscription<std_msgs::msg::Float64MultiArray>(
        position_topic, 10,
        std::bind(&ImpedanceTrajectoryControllerNode::positionCallback, this, std::placeholders::_1));

    goal_command_sub_ = this->create_subscription<wbc_msgs::msg::ImpedanceCommands>(
        goal_command_topic, 10,
        std::bind(&ImpedanceTrajectoryControllerNode::goalCommandCallback, this, std::placeholders::_1));

    joint_state_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
        joint_state_topic, rclcpp::SensorDataQoS(),
        std::bind(&ImpedanceTrajectoryControllerNode::jointStateCallback, this, std::placeholders::_1));

    rclcpp::QoS qos_profile(rclcpp::KeepLast(10));
    qos_profile.durability(RMW_QOS_POLICY_DURABILITY_VOLATILE);
    impedance_pub_ = this->create_publisher<wbc_msgs::msg::ImpedanceCommands>(
        impedance_topic, qos_profile);

    const double safe_rate_hz = std::max(control_rate_hz, 1.0);
    const auto period = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::duration<double>(1.0 / safe_rate_hz));
    update_timer_ = this->create_wall_timer(
        period, std::bind(&ImpedanceTrajectoryControllerNode::updateLoop, this));

    RCLCPP_INFO(this->get_logger(),
                "impedance_trajectory_controller_node started (rate=%.1fHz, goal_duration=%.3fs, level=%.2f)",
                safe_rate_hz, default_goal_duration_sec_,
                impedance_handler_->active_level());
  }

private:
  void impedanceLevelCallback(const std_msgs::msg::Float64::SharedPtr msg) {
    if (!msg) {
      return;
    }

    std::string error;
    if (!impedance_handler_->set_level(msg->data, &error)) {
      RCLCPP_ERROR(this->get_logger(), "Failed to set impedance level %.3f: %s",
                   msg->data, error.c_str());
      return;
    }

    RCLCPP_INFO(this->get_logger(), "Impedance level set to %.3f",
                impedance_handler_->active_level());
  }

  void positionCallback(const std_msgs::msg::Float64MultiArray::SharedPtr msg) {
    if (!msg) {
      return;
    }
    if (msg->data.empty()) {
      controller_.holdPosition();
      return;
    }
    controller_.setGoal(msg->data, default_goal_duration_sec_);
  }

  void goalCommandCallback(const wbc_msgs::msg::ImpedanceCommands::SharedPtr msg) {
    if (!msg) {
      return;
    }
    if (msg->position.empty()) {
      controller_.holdPosition();
      return;
    }

    if (!msg->stiffness.empty() || !msg->damping.empty()) {
      impedance_handler_->set_custom_gains(msg->stiffness, msg->damping);
    }

    controller_.setGoal(msg->position, default_goal_duration_sec_, msg->effort_ff);
  }

  void jointStateCallback(const sensor_msgs::msg::JointState::SharedPtr msg) {
    if (!msg) {
      return;
    }
    controller_.setMeasuredState(msg->position, msg->velocity);
  }

  void updateLoop() {
    const auto now = steady_clock_.now();
    double dt_sec = 0.0;
    if (has_last_update_time_) {
      dt_sec = (now - last_update_time_).seconds();
    }
    last_update_time_ = now;
    has_last_update_time_ = true;

    auto gains = impedance_handler_->gains();
    controller_.setGains(gains.stiffness, gains.damping);
    const auto impedance_cmd = controller_.update(dt_sec);

    wbc_msgs::msg::ImpedanceCommands msg_out;
    msg_out.position = impedance_cmd.position;
    msg_out.velocity = impedance_cmd.velocity;
    msg_out.stiffness = impedance_cmd.stiffness;
    msg_out.damping = impedance_cmd.damping;
    msg_out.effort_ff = impedance_cmd.effort_ff;
    impedance_pub_->publish(msg_out);
  }

  static std::string resolve_hand_topic_(const std::string & hand_namespace, const char * suffix)
  {
    if (hand_namespace.empty()) {
      return std::string(suffix);
    }

    if (hand_namespace.front() == '/') {
      return hand_namespace + suffix;
    }

    return "/" + hand_namespace + suffix;
  }

  ImpedanceTrajectoryController controller_;
  std::unique_ptr<joint_impedance_controller::ImpedanceHandler> impedance_handler_;
  rclcpp::Subscription<std_msgs::msg::Float64>::SharedPtr impedance_level_sub_;
  rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr position_sub_;
  rclcpp::Subscription<wbc_msgs::msg::ImpedanceCommands>::SharedPtr goal_command_sub_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_sub_;
  rclcpp::Publisher<wbc_msgs::msg::ImpedanceCommands>::SharedPtr impedance_pub_;
  rclcpp::TimerBase::SharedPtr update_timer_;

  rclcpp::Clock steady_clock_;
  rclcpp::Time last_update_time_{0, 0, RCL_STEADY_TIME};
  bool has_last_update_time_{false};
  double default_goal_duration_sec_{0.25};
};

int main(int argc, char* argv[]) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ImpedanceTrajectoryControllerNode>());
  rclcpp::shutdown();
  return 0;
}
