#ifndef PLATO_HARDWARE_INTERFACE__GIM3505_PROTOCOL_HPP_
#define PLATO_HARDWARE_INTERFACE__GIM3505_PROTOCOL_HPP_

#include <cstdint>
#include <optional>

#include <PCANBasic.h>

#include "can_hardware_common/actuator_protocol.hpp"
#include "can_hardware_common/utils/can_helper.hpp"

namespace plato_hardware_interface::gim3505_protocol
{

constexpr uint32_t kDefaultQddPositionDurationMs = 10;

namespace CommandByte
{
constexpr uint8_t START_MOTOR = 0x91;
constexpr uint8_t STOP_MOTOR = 0x92;
constexpr uint8_t TORQUE_CONTROL = 0x93;
constexpr uint8_t SPEED_CONTROL = 0x94;
constexpr uint8_t POSITION_CONTROL = 0x95;
constexpr uint8_t STOP_CONTROL = 0x97;
}  // namespace CommandByte

namespace ResultByte
{
constexpr uint8_t SUCCESS = 0x00;
constexpr uint8_t FAILURE = 0x01;
constexpr uint8_t FAILURE_UNKNOWN_COMMAND = 0x02;
constexpr uint8_t FAILURE_UNKNOWN_ID = 0x03;
constexpr uint8_t FAILURE_READ_ONLY_REGISTER = 0x04;
constexpr uint8_t FAILURE_UNKNOWN_REGISTER = 0x05;
}  // namespace ResultByte

class MsgEncoder
{
public:
  static void start_motor(TPCANMsg & msg);
  static void stop_motor(TPCANMsg & msg);
  static void stop_control(TPCANMsg & msg);
  static void set_torque(TPCANMsg & msg, float torque, uint32_t duration);
  static void set_position(TPCANMsg & msg, float position, uint32_t duration);
};

class MsgDecoder
{
public:
  MsgDecoder(float gear_ratio, float torque_constant)
  : torque_scale_(450.0f * torque_constant * gear_ratio / 4095.0f),
    torque_offset_(225.0f * torque_constant * gear_ratio)
  {
  }

  static bool get_result(uint8_t error_byte);

  void get_states(
    const TPCANMsg & msg,
    uint8_t & temperature,
    float & position,
    float & velocity,
    float & torque) const;

private:
  float torque_scale_;
  float torque_offset_;
};

}  // namespace plato_hardware_interface::gim3505_protocol

namespace plato_actuator
{

class Gim3505Protocol : public can_hardware_common::ActuatorProtocol
{
public:
  explicit Gim3505Protocol(const can_hardware_common::ActuatorCoreConfig & config);

  std::optional<actuator::TxCommand> make_impedance_command(
    const can_hardware_common::ActuatorTarget & joint_target) override;
  actuator::TxCommand make_torque_command(float motor_torque) override;
  std::optional<can_hardware_common::DecodedFeedback> decode(const TPCANMsg & frame) override;

  actuator::TxCommand make_enable_motor_command();
  actuator::TxCommand make_disable_motor_command();
  actuator::TxCommand make_stop_control_command();
  actuator::TxCommand make_position_command(
    float joint_position,
    uint32_t duration = plato_hardware_interface::gim3505_protocol::kDefaultQddPositionDurationMs);
  actuator::TxCommand make_servo_position_command(
    float joint_position,
    uint32_t current_milliamps = 0);
  void set_position_offset(float position_offset) { config_.position_offset = position_offset; }

private:
  static TPCANMsg make_message_(uint32_t can_id, uint8_t len);
  static void validate_direction_(const can_hardware_common::ActuatorCoreConfig & config);
  float map_joint_to_motor_frame_(float joint_value, bool apply_offset = false) const;
  float map_motor_to_joint_frame_(float motor_value, bool apply_offset = false) const;

  can_hardware_common::ActuatorCoreConfig config_;
  uint8_t tx_id_;
  plato_hardware_interface::gim3505_protocol::MsgDecoder decoder_;
  TPCANMsg onoff_msg_;
  TPCANMsg cmd_msg_;
};

}  // namespace plato_actuator

#endif  // PLATO_HARDWARE_INTERFACE__GIM3505_PROTOCOL_HPP_
