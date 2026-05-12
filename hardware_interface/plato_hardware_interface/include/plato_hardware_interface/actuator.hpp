#ifndef PLATO_HARDWARE_INTERFACE__ACTUATOR_HPP_
#define PLATO_HARDWARE_INTERFACE__ACTUATOR_HPP_

#include <cstdint>
#include <limits>
#include <memory>

#include "can_hardware_common/actuator.hpp"
#include "plato_hardware_interface/gim3505_protocol.hpp"

namespace plato_actuator
{

enum class ProtocolKind
{
  kGim3505,
  kDynamixelBridge,
};

struct StaticConfig
{
  uint8_t can_tx_id = 0;
  uint8_t can_rx_id = 0;
  ProtocolKind protocol_kind = ProtocolKind::kGim3505;
  uint8_t dynamixel_servo_id = 0;
  int8_t direction = 1;
  float torque_constant = 0.0f;
  float gear_ratio = 0.0f;
  actuator::Limits limits;
  uint32_t servo_current_milliamps = 0;
  bool soft_stop_enabled = true;
  float effort_limit_nm = std::numeric_limits<float>::infinity();
};

struct Config
{
  StaticConfig static_config;
  float position_offset = 0.0f;
};

class Actuator
{
public:
  explicit Actuator(const Config & config);
  Actuator(const Actuator &) = delete;
  Actuator & operator=(const Actuator &) = delete;
  Actuator(Actuator &&) noexcept;
  Actuator & operator=(Actuator &&) = delete;
  ~Actuator();

  /// Low-level: returns TxCommand (used by protocol/zeroing internals).
  actuator::TxCommand enable_motor();
  actuator::TxCommand disable_motor();
  actuator::TxCommand stop_control();
  actuator::TxCommand set_joint_torque(float joint_torque);
  actuator::TxCommand set_joint_position(
    float joint_position,
    uint32_t duration_ms = plato_hardware_interface::gim3505_protocol::kDefaultQddPositionDurationMs);
  actuator::TxCommand set_servo_position(float joint_position, uint32_t current_milliamps = 0);
  actuator::TxCommand set_servo_hold(float joint_position);
  actuator::TxCommand set_servo_idle(float joint_position);
  actuator::TxCommand set_actuator_command(
    float position,
    float effort,
    float stiffness,
    double servo_stiffness_scale);
  bool set_current_position_as_zero();

  bool process_rx_frame(const TPCANMsg & msg);

  Config get_config() const { return {static_config_, position_offset_}; }
  const StaticConfig & get_static_config() const { return static_config_; }
  const can_hardware_common::ActuatorState & get_state() const { return state_; }
  const can_hardware_common::ActuatorStatus & get_status() const { return status_; }

  uint32_t get_tx_id() const { return static_config_.can_tx_id; }
  uint32_t get_rx_id() const { return static_config_.can_rx_id; }
  float get_motor_position() const { return motor_position_; }
  float get_position_offset() const { return position_offset_; }
  bool is_enabled() const { return motor_enabled_; }
  bool is_initialized() const { return is_initialized_; }
  bool uses_dynamixel_bridge() const
  {
    return static_config_.protocol_kind == ProtocolKind::kDynamixelBridge;
  }

private:
  actuator::TxCommand make_dynamixel_lifecycle_command_(bool enable) const;
  bool process_dynamixel_bridge_message_(const TPCANMsg & msg);
  uint32_t resolve_servo_current_command_(float stiffness, double servo_stiffness_scale) const;
  void apply_decoded_feedback_(const can_hardware_common::DecodedFeedback & decoded);
  float clamp_torque_near_bounds_(float joint_torque) const;
  float map_joint_to_motor_frame_(float joint_value, bool apply_offset = false) const;
  float map_motor_to_joint_frame_(float motor_value, bool apply_offset = false) const;

  const StaticConfig static_config_;
  float position_offset_ = 0.0f;
  std::unique_ptr<Gim3505Protocol> protocol_;
  can_hardware_common::ActuatorState state_;
  can_hardware_common::ActuatorStatus status_;
  float motor_position_ = 0.0f;
  bool motor_enabled_ = false;
  bool is_initialized_ = false;

  static constexpr float kMotorTorqueScale = 8.0f;
  static constexpr float kMotorTorqueLimit = 9.8f;
  static constexpr float kJointLimitSafetyMargin = 0.05f;
  static constexpr double kMaxDerivedServoCurrentMilliamps = 2000.0;
};

}  // namespace plato_actuator

#endif  // PLATO_HARDWARE_INTERFACE__ACTUATOR_HPP_
