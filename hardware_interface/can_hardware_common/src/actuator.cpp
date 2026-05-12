#include "can_hardware_common/actuator.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <utility>
#include <stdexcept>

namespace can_hardware_common
{

Actuator::Actuator(const actuator::Config & config, std::unique_ptr<ActuatorProtocol> protocol)
: config_(config),
  protocol_(std::move(protocol))
{
  validate_direction(config_.core);
  validate_limits(config_.limits);
  if (!protocol_) {
    throw std::invalid_argument("Actuator protocol must not be null");
  }
}

Actuator::~Actuator() = default;

Actuator::Actuator(Actuator &&) noexcept = default;

Actuator & Actuator::operator=(Actuator &&) noexcept = default;

std::optional<actuator::TxCommand> Actuator::set_impedance_command(
  const ActuatorTarget & joint_target)
{
  return protocol_->make_impedance_command(clamp_impedance_target(joint_target));
}

actuator::TxCommand Actuator::set_torque_command(float joint_torque)
{
  return protocol_->make_torque_command(clamp_joint_torque(joint_torque));
}

bool Actuator::process_rx_frame(const TPCANMsg & frame)
{
  if (frame.ID != config_.core.can_rx_id) {
    return false;
  }

  const auto decoded = protocol_->decode(frame);
  if (!decoded) {
    return false;
  }

  apply_decoded_feedback(*decoded);
  return true;
}

void Actuator::apply_decoded_feedback(const DecodedFeedback & decoded)
{
  if (decoded.has_state) {
    state_ = decoded.state;
  }

  if (decoded.temperature) {
    status_.temperature = *decoded.temperature;
  }

  if (decoded.in_oc_mode) {
    status_.in_oc_mode = *decoded.in_oc_mode;
  }

  if (decoded.has_fault) {
    status_.has_fault = *decoded.has_fault;
  }

  if (decoded.motor_enabled) {
    motor_enabled_ = *decoded.motor_enabled;
  }
}

float Actuator::clamp_joint_torque(float joint_torque) const
{
  return std::clamp(joint_torque, -config_.limits.effort_limit, config_.limits.effort_limit);
}

ActuatorTarget Actuator::clamp_impedance_target(const ActuatorTarget & joint_target) const
{
  ActuatorTarget clamped = joint_target;

  clamped.position = std::clamp(
    clamped.position,
    config_.limits.position_limit_min,
    config_.limits.position_limit_max);

  clamped.velocity = std::clamp(
    clamped.velocity,
    -config_.limits.velocity_limit,
    config_.limits.velocity_limit);

  clamped.torque = std::clamp(
    clamped.torque,
    -config_.limits.effort_limit,
    config_.limits.effort_limit);

  clamped.stiffness = std::clamp(
    clamped.stiffness,
    0.0f,
    config_.limits.stiffness_limit);

  clamped.damping = std::clamp(
    clamped.damping,
    0.0f,
    config_.limits.damping_limit);

  return clamped;
}

void Actuator::validate_direction(const ActuatorCoreConfig & core_config)
{
  if (core_config.direction != 1 && core_config.direction != -1) {
    throw std::invalid_argument("Actuator direction must be +1 or -1");
  }
}

void Actuator::validate_limits(const actuator::Limits & limits)
{
  if (!std::isfinite(limits.position_limit_min) ||
    !std::isfinite(limits.position_limit_max) ||
    !std::isfinite(limits.velocity_limit) ||
    !std::isfinite(limits.effort_limit) ||
    !std::isfinite(limits.stiffness_limit) ||
    !std::isfinite(limits.damping_limit))
  {
    throw std::invalid_argument("Actuator limits must be finite");
  }

  if (limits.position_limit_min > limits.position_limit_max) {
    throw std::invalid_argument("Actuator position limits are invalid");
  }

  if (limits.velocity_limit < 0.0f ||
    limits.effort_limit < 0.0f ||
    limits.stiffness_limit < 0.0f ||
    limits.damping_limit < 0.0f)
  {
    throw std::invalid_argument("Actuator magnitude limits must be non-negative");
  }
}

}  // namespace can_hardware_common
