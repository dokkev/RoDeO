#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>

#include "dynamixel_sdk/dynamixel_sdk.h"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float64.hpp"

namespace dynamixel_interface
{
namespace
{

constexpr int kDefaultBaudRate = 1000000;
constexpr int kDefaultDynamixelId = 101;
constexpr int kDefaultTorqueEnableAddress = 64;
constexpr int kDefaultGoalPositionAddress = 116;
constexpr int kDefaultPresentPositionAddress = 132;
constexpr int kDefaultPositionDataSize = 4;
constexpr double kPi = 3.14159265358979323846;
constexpr int kDefaultPositionMinTick = 0;
constexpr int kDefaultPositionMaxTick = 4095;

uint16_t checked_address(int address, const char * name)
{
  if (address < 0 || address > std::numeric_limits<uint16_t>::max()) {
    throw std::runtime_error(std::string(name) + " must fit in uint16");
  }
  return static_cast<uint16_t>(address);
}

uint8_t checked_dynamixel_id(int id)
{
  if (id < 0 || id > std::numeric_limits<uint8_t>::max()) {
    throw std::runtime_error("dxl_id must fit in uint8");
  }
  return static_cast<uint8_t>(id);
}

int checked_position_data_size(int data_size)
{
  if (data_size != 1 && data_size != 2 && data_size != 4) {
    throw std::runtime_error("position_data_size must be 1, 2, or 4");
  }
  return data_size;
}

}  // namespace

class DynamixelTopicController final : public rclcpp::Node
{
public:
  DynamixelTopicController()
  : Node("dynamixel_topic_controller"),
    device_name_(declare_parameter<std::string>("device_name", "/dev/ttyUSB0")),
    baud_rate_(declare_parameter<int>("baud_rate", kDefaultBaudRate)),
    protocol_version_(declare_parameter<double>("protocol_version", 2.0)),
    dxl_id_(checked_dynamixel_id(declare_parameter<int>("dxl_id", kDefaultDynamixelId))),
    torque_enable_address_(
      checked_address(
        declare_parameter<int>("torque_enable_address", kDefaultTorqueEnableAddress),
        "torque_enable_address")),
    operating_mode_address_(
      checked_address(declare_parameter<int>("operating_mode_address", 11), "operating_mode_address")),
    goal_position_address_(
      checked_address(
        declare_parameter<int>("goal_position_address", kDefaultGoalPositionAddress),
        "goal_position_address")),
    present_position_address_(
      checked_address(
        declare_parameter<int>("present_position_address", kDefaultPresentPositionAddress),
        "present_position_address")),
    position_data_size_(
      checked_position_data_size(
        declare_parameter<int>("position_data_size", kDefaultPositionDataSize))),
    enable_torque_on_start_(declare_parameter<bool>("enable_torque_on_start", true)),
    disable_torque_on_shutdown_(declare_parameter<bool>("disable_torque_on_shutdown", true)),
    set_operating_mode_on_start_(declare_parameter<bool>("set_operating_mode_on_start", false)),
    position_operating_mode_(declare_parameter<int>("position_operating_mode", 3)),
    read_period_ms_(declare_parameter<int>("read_period_ms", 50)),
    position_min_rad_(declare_parameter<double>("position_min_rad", -kPi)),
    position_max_rad_(declare_parameter<double>("position_max_rad", kPi)),
    position_min_tick_(declare_parameter<int>("position_min_tick", kDefaultPositionMinTick)),
    position_max_tick_(declare_parameter<int>("position_max_tick", kDefaultPositionMaxTick)),
    command_topic_(declare_parameter<std::string>("command_topic", "~/goal_position")),
    present_position_topic_(
      declare_parameter<std::string>("present_position_topic", "~/present_position"))
  {
    if (position_min_rad_ >= position_max_rad_) {
      throw std::runtime_error("position_min_rad must be smaller than position_max_rad");
    }
    if (position_min_tick_ == position_max_tick_) {
      throw std::runtime_error("position_min_tick and position_max_tick must differ");
    }

    packet_handler_ = dynamixel::PacketHandler::getPacketHandler(
      static_cast<float>(protocol_version_));
    port_handler_.reset(dynamixel::PortHandler::getPortHandler(device_name_.c_str()));

    if (packet_handler_ == nullptr || port_handler_ == nullptr) {
      throw std::runtime_error("failed to create Dynamixel SDK handlers");
    }
    if (!port_handler_->openPort()) {
      throw std::runtime_error("failed to open Dynamixel port: " + device_name_);
    }
    if (!port_handler_->setBaudRate(baud_rate_)) {
      throw std::runtime_error("failed to set Dynamixel baud rate: " + std::to_string(baud_rate_));
    }

    if (set_operating_mode_on_start_) {
      write_torque_enable(false);
      write_operating_mode(position_operating_mode_);
    }
    if (enable_torque_on_start_) {
      write_torque_enable(true);
    }

    command_sub_ = create_subscription<std_msgs::msg::Float64>(
      command_topic_,
      rclcpp::SystemDefaultsQoS(),
      [this](const std_msgs::msg::Float64::SharedPtr msg) {
        write_goal_position_rad(msg->data);
      });
    present_position_pub_ = create_publisher<std_msgs::msg::Float64>(
      present_position_topic_, rclcpp::SystemDefaultsQoS());

    if (read_period_ms_ > 0) {
      read_timer_ = create_wall_timer(
        std::chrono::milliseconds(read_period_ms_),
        [this]() {
          publish_present_position();
        });
    }

    RCLCPP_INFO(
      get_logger(),
      "Dynamixel topic controller ready: id=%u baud=%d command_topic=%s (Float64 radians)",
      dxl_id_,
      baud_rate_,
      command_topic_.c_str());
  }

  ~DynamixelTopicController() override
  {
    if (port_handler_ == nullptr) {
      return;
    }

    if (disable_torque_on_shutdown_) {
      write_torque_enable(false);
    }
    port_handler_->closePort();
  }

private:
  bool write_torque_enable(bool enabled)
  {
    return write_one_byte(
      torque_enable_address_, static_cast<uint8_t>(enabled ? 1 : 0), "torque enable");
  }

  bool write_operating_mode(int operating_mode)
  {
    if (operating_mode < 0 || operating_mode > std::numeric_limits<uint8_t>::max()) {
      RCLCPP_ERROR(get_logger(), "operating mode must fit in uint8: %d", operating_mode);
      return false;
    }
    return write_one_byte(
      operating_mode_address_, static_cast<uint8_t>(operating_mode), "operating mode");
  }

  bool write_goal_position_rad(double position_rad)
  {
    const auto raw_position = radians_to_raw_position(position_rad);
    if (!raw_position) {
      return false;
    }
    return write_goal_position_raw(*raw_position);
  }

  bool write_goal_position_raw(int32_t position)
  {
    uint8_t error = 0;
    int result = COMM_NOT_AVAILABLE;
    if (position_data_size_ == 1) {
      if (position < 0 || position > std::numeric_limits<uint8_t>::max()) {
        RCLCPP_ERROR(get_logger(), "1-byte goal position out of range: %d", position);
        return false;
      }
      result = packet_handler_->write1ByteTxRx(
        port_handler_.get(), dxl_id_, goal_position_address_, static_cast<uint8_t>(position), &error);
    } else if (position_data_size_ == 2) {
      if (position < 0 || position > std::numeric_limits<uint16_t>::max()) {
        RCLCPP_ERROR(get_logger(), "2-byte goal position out of range: %d", position);
        return false;
      }
      result = packet_handler_->write2ByteTxRx(
        port_handler_.get(), dxl_id_, goal_position_address_, static_cast<uint16_t>(position), &error);
    } else {
      result = packet_handler_->write4ByteTxRx(
        port_handler_.get(), dxl_id_, goal_position_address_, static_cast<uint32_t>(position), &error);
    }
    return check_sdk_result(result, error, "write goal position");
  }

  void publish_present_position()
  {
    uint8_t error = 0;
    int result = COMM_NOT_AVAILABLE;
    int32_t position = 0;

    if (position_data_size_ == 1) {
      uint8_t data = 0;
      result = packet_handler_->read1ByteTxRx(
        port_handler_.get(), dxl_id_, present_position_address_, &data, &error);
      position = data;
    } else if (position_data_size_ == 2) {
      uint16_t data = 0;
      result = packet_handler_->read2ByteTxRx(
        port_handler_.get(), dxl_id_, present_position_address_, &data, &error);
      position = data;
    } else {
      uint32_t data = 0;
      result = packet_handler_->read4ByteTxRx(
        port_handler_.get(), dxl_id_, present_position_address_, &data, &error);
      position = static_cast<int32_t>(data);
    }

    if (!check_sdk_result(result, error, "read present position")) {
      return;
    }

    std_msgs::msg::Float64 msg;
    msg.data = raw_position_to_radians(position);
    present_position_pub_->publish(msg);
  }

  std::optional<int32_t> radians_to_raw_position(double position_rad)
  {
    if (!std::isfinite(position_rad)) {
      RCLCPP_ERROR(get_logger(), "goal position must be finite: %f", position_rad);
      return std::nullopt;
    }

    const double clamped_position =
      std::clamp(position_rad, position_min_rad_, position_max_rad_);
    if (clamped_position != position_rad) {
      RCLCPP_WARN_THROTTLE(
        get_logger(),
        *get_clock(),
        1000,
        "goal position %.6f rad outside [%.6f, %.6f], clamping",
        position_rad,
        position_min_rad_,
        position_max_rad_);
    }

    const double normalized =
      (clamped_position - position_min_rad_) / (position_max_rad_ - position_min_rad_);
    const double raw_position =
      position_min_tick_ + normalized * (position_max_tick_ - position_min_tick_);
    return static_cast<int32_t>(std::llround(raw_position));
  }

  double raw_position_to_radians(int32_t raw_position) const
  {
    const double normalized =
      static_cast<double>(raw_position - position_min_tick_) /
      static_cast<double>(position_max_tick_ - position_min_tick_);
    return position_min_rad_ + normalized * (position_max_rad_ - position_min_rad_);
  }

  bool write_one_byte(uint16_t address, uint8_t value, const char * action)
  {
    uint8_t error = 0;
    const int result = packet_handler_->write1ByteTxRx(
      port_handler_.get(), dxl_id_, address, value, &error);
    return check_sdk_result(result, error, action);
  }

  bool check_sdk_result(int result, uint8_t error, const char * action)
  {
    if (result != COMM_SUCCESS) {
      RCLCPP_ERROR_THROTTLE(
        get_logger(),
        *get_clock(),
        1000,
        "%s failed: %s",
        action,
        packet_handler_->getTxRxResult(result));
      return false;
    }
    if (error != 0) {
      RCLCPP_ERROR_THROTTLE(
        get_logger(),
        *get_clock(),
        1000,
        "%s returned Dynamixel error: %s",
        action,
        packet_handler_->getRxPacketError(error));
      return false;
    }
    return true;
  }

  std::unique_ptr<dynamixel::PortHandler> port_handler_;
  dynamixel::PacketHandler * packet_handler_ = nullptr;

  std::string device_name_;
  int baud_rate_ = kDefaultBaudRate;
  double protocol_version_ = 2.0;
  uint8_t dxl_id_ = kDefaultDynamixelId;
  uint16_t torque_enable_address_ = kDefaultTorqueEnableAddress;
  uint16_t operating_mode_address_ = 11;
  uint16_t goal_position_address_ = kDefaultGoalPositionAddress;
  uint16_t present_position_address_ = kDefaultPresentPositionAddress;
  int position_data_size_ = kDefaultPositionDataSize;
  bool enable_torque_on_start_ = true;
  bool disable_torque_on_shutdown_ = true;
  bool set_operating_mode_on_start_ = false;
  int position_operating_mode_ = 3;
  int read_period_ms_ = 50;
  double position_min_rad_ = -kPi;
  double position_max_rad_ = kPi;
  int position_min_tick_ = kDefaultPositionMinTick;
  int position_max_tick_ = kDefaultPositionMaxTick;
  std::string command_topic_;
  std::string present_position_topic_;

  rclcpp::Subscription<std_msgs::msg::Float64>::SharedPtr command_sub_;
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr present_position_pub_;
  rclcpp::TimerBase::SharedPtr read_timer_;
};

}  // namespace dynamixel_interface

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  try {
    rclcpp::spin(std::make_shared<dynamixel_interface::DynamixelTopicController>());
  } catch (const std::exception & ex) {
    RCLCPP_FATAL(rclcpp::get_logger("dynamixel_topic_controller"), "%s", ex.what());
  }
  rclcpp::shutdown();
  return 0;
}
