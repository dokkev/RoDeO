#include "plato_hardware_interface/plato_state_helper.hpp"

#include <cmath>
#include <string>

#include <rclcpp/rclcpp.hpp>

#include "plato_hardware_interface/plato_layout.hpp"

namespace plato_hand
{
namespace
{
auto logger() { return rclcpp::get_logger("plato_hardware_interface"); }

rclcpp::Clock & throttle_clock()
{
  static rclcpp::Clock clock(RCL_STEADY_TIME);
  return clock;
}
}  // namespace

void PlatoStateHelper::update_joint_states(
  const std::vector<plato_actuator::Actuator> & actuators,
  const can_hardware_common::RobotIO::JointCommand & joint_commands,
  can_hardware_common::RobotIO::ActuatorState & actuator_states,
  can_hardware_common::RobotIO::JointState & joint_states) const
{
  bool used_fallback_feedback = false;
  std::string fallback_feedback_ids;
  for (std::size_t i = 0; i < actuators.size(); ++i) {
    if (!actuators[i].is_initialized()) {
      if (layout::is_thumb_servo(i)) {
        actuator_states.position_at(i) = static_cast<float>(joint_commands.position_at(i));
        actuator_states.velocity_at(i) = 0.0;
        actuator_states.effort_at(i) = 0.0;
        continue;
      }

      const bool have_cached_feedback =
        std::isfinite(actuator_states.position_at(i)) &&
        std::isfinite(actuator_states.velocity_at(i)) &&
        std::isfinite(actuator_states.effort_at(i));
      if (!have_cached_feedback) {
        actuator_states.position_at(i) = 0.0;
        actuator_states.velocity_at(i) = 0.0;
        actuator_states.effort_at(i) = 0.0;
      }

      used_fallback_feedback = true;
      if (!fallback_feedback_ids.empty()) {
        fallback_feedback_ids += ", ";
      }
      fallback_feedback_ids += std::to_string(i + 1);
      if (!have_cached_feedback) {
        fallback_feedback_ids += "*";
      }
      continue;
    }

    const auto & state = actuators[i].get_state();
    actuator_states.position_at(i) = state.position;
    actuator_states.velocity_at(i) = state.velocity;
    actuator_states.effort_at(i) = state.torque;
  }

  if (used_fallback_feedback) {
    RCLCPP_WARN_THROTTLE(
      logger(),
      throttle_clock(),
      1000,
      "Joint-state update using fallback feedback for actuator(s) [%s] ('*' means zero fallback; others use cached state)",
      fallback_feedback_ids.c_str());
  }

  can_hardware_common::RobotIO::JointState joint_state_candidate = joint_states;
  transmission_.actuator_to_joint(actuator_states, joint_state_candidate);
  if (!joint_state_candidate.all_finite()) {
    RCLCPP_WARN_THROTTLE(
      logger(),
      throttle_clock(),
      1000,
      "Skipping joint-state update: transmission output contains non-finite values.");
    return;
  }

  joint_states.copy_from(joint_state_candidate);
}

void PlatoStateHelper::joint_to_actuator_commands(
  const can_hardware_common::RobotIO::JointCommand & joint_commands,
  const can_hardware_common::RobotIO::ActuatorState & actuator_states,
  const can_hardware_common::RobotIO::JointState & joint_states,
  can_hardware_common::RobotIO::ActuatorCommand & actuator_commands) const
{
  transmission_.joint_to_actuator(
    joint_commands,
    actuator_states,
    joint_states,
    actuator_commands);
}

bool PlatoStateHelper::actuators_ready(const std::vector<plato_actuator::Actuator> & actuators) const
{
  return can_hardware_common::RobotIO::all_actuators_ready_from(
    actuators,
    layout::kFirstGimActuator,
    [](const auto & actuator) { return actuator.is_initialized(); });
}

void PlatoStateHelper::copy_feedback_snapshot(
  const std::vector<plato_actuator::Actuator> & actuators,
  can_hardware_common::core::StateSnapshot & snapshot) const
{
  can_hardware_common::RobotIO::copy_actuator_feedback_to_snapshot(
    actuators,
    snapshot,
    [](const auto & actuator) { return actuator.get_state(); });
}

}  // namespace plato_hand
