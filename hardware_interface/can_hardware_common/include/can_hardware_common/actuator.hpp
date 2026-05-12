#ifndef CAN_HARDWARE_COMMON__ACTUATOR_HPP_
#define CAN_HARDWARE_COMMON__ACTUATOR_HPP_

#include <memory>
#include <optional>

#include "can_hardware_common/actuator_protocol.hpp"
#include "can_hardware_common/actuator_types.hpp"

namespace can_hardware_common
{

class Actuator
{
public:
  Actuator(const actuator::Config & config, std::unique_ptr<ActuatorProtocol> protocol);
  ~Actuator();

  Actuator(const Actuator &) = delete;
  Actuator & operator=(const Actuator &) = delete;
  Actuator(Actuator &&) noexcept;
  Actuator & operator=(Actuator &&) noexcept;

  std::optional<actuator::TxCommand> set_impedance_command(
    const ActuatorTarget & joint_target);
  actuator::TxCommand set_torque_command(float joint_torque);
  bool process_rx_frame(const TPCANMsg & frame);
  const ActuatorState & get_states() const { return state_; }

private:
  void apply_decoded_feedback(const DecodedFeedback & decoded);
  float clamp_joint_torque(float joint_torque) const;
  ActuatorTarget clamp_impedance_target(const ActuatorTarget & joint_target) const;
  static void validate_direction(const ActuatorCoreConfig & core_config);
  static void validate_limits(const actuator::Limits & limits);

  actuator::Config config_;
  std::unique_ptr<ActuatorProtocol> protocol_;
  ActuatorState state_;
  ActuatorStatus status_;
  bool motor_enabled_ = false;
};

}  // namespace can_hardware_common

#endif  // CAN_HARDWARE_COMMON__ACTUATOR_HPP_
