#ifndef CAN_HARDWARE_COMMON__ACTUATOR_TYPES_HPP_
#define CAN_HARDWARE_COMMON__ACTUATOR_TYPES_HPP_

#include <cstdint>
#include <limits>

#include <PCANBasic.h>

namespace can_hardware_common
{

struct ActuatorTarget
{
  float position = 0.0f;
  float velocity = 0.0f;
  float stiffness = 0.0f;
  float damping = 0.0f;
  float torque = 0.0f;
};

struct ActuatorState
{
  float position = 0.0f;
  float velocity = 0.0f;
  float torque = 0.0f;
};

struct ActuatorStatus
{
  uint8_t temperature = 0;
  bool in_oc_mode = false;
  bool has_fault = false;
};

struct ActuatorCoreConfig
{
  uint8_t can_tx_id = 0;
  uint8_t can_rx_id = 0;
  float position_offset = 0.0f;
  int8_t direction = 1;
  float torque_constant = 0.0f;
  float gear_ratio = 0.0f;
};

}  // namespace can_hardware_common

namespace actuator
{

struct Limits
{
  float position_limit_max = std::numeric_limits<float>::quiet_NaN();
  float position_limit_min = std::numeric_limits<float>::quiet_NaN();
  float velocity_limit = std::numeric_limits<float>::quiet_NaN();
  float effort_limit = std::numeric_limits<float>::quiet_NaN();
  float stiffness_limit = std::numeric_limits<float>::quiet_NaN();
  float damping_limit = std::numeric_limits<float>::quiet_NaN();
};

struct Config
{
  can_hardware_common::ActuatorCoreConfig core;
  Limits limits;
};

struct TxCommand
{
  TPCANMsg frame{};
};

}  // namespace actuator

#endif  // CAN_HARDWARE_COMMON__ACTUATOR_TYPES_HPP_
