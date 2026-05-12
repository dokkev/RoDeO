#ifndef CAN_HARDWARE_COMMON__ACTUATOR_PROTOCOL_HPP_
#define CAN_HARDWARE_COMMON__ACTUATOR_PROTOCOL_HPP_

#include <optional>

#include <PCANBasic.h>

#include "can_hardware_common/actuator_types.hpp"

namespace can_hardware_common
{

struct DecodedFeedback
{
  bool has_state = false;
  ActuatorState state{};
  std::optional<float> motor_position;
  std::optional<uint8_t> temperature;
  std::optional<bool> in_oc_mode;
  std::optional<bool> has_fault;
  std::optional<bool> motor_enabled;
};

class ActuatorProtocol
{
public:
  virtual ~ActuatorProtocol() = default;

  virtual std::optional<actuator::TxCommand> make_impedance_command(
    const ActuatorTarget & joint_target) = 0;
  virtual actuator::TxCommand make_torque_command(float joint_torque) = 0;
  virtual std::optional<DecodedFeedback> decode(const TPCANMsg & frame) = 0;
};

}  // namespace can_hardware_common

#endif  // CAN_HARDWARE_COMMON__ACTUATOR_PROTOCOL_HPP_
