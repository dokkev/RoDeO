#ifndef CAN_HARDWARE_COMMON__CORE__ACTUATOR_FRAME_ROUTER_HPP_
#define CAN_HARDWARE_COMMON__CORE__ACTUATOR_FRAME_ROUTER_HPP_

#include <cstddef>
#include <vector>

#include <PCANBasic.h>

namespace can_hardware_common::core
{

template<typename ActuatorRange>
void append_enable_frames(ActuatorRange & actuators, std::vector<TPCANMsg> & frames)
{
  frames.reserve(frames.size() + actuators.size());
  for (auto & actuator : actuators) {
    frames.push_back(actuator.enable_motor().frame);
  }
}

template<typename ActuatorRange>
void append_disable_frames(ActuatorRange & actuators, std::vector<TPCANMsg> & frames)
{
  frames.reserve(frames.size() + actuators.size());
  for (auto & actuator : actuators) {
    frames.push_back(actuator.disable_motor().frame);
  }
}

template<typename ActuatorRange>
bool dispatch_rx_frame(const TPCANMsg & frame, ActuatorRange & actuators)
{
  if (frame.MSGTYPE != PCAN_MESSAGE_STANDARD) {
    return false;
  }

  for (auto & actuator : actuators) {
    if (actuator.process_rx_frame(frame)) {
      return true;
    }
  }

  return false;
}

template<typename ActuatorRange, typename ActuatorCommand>
void append_actuator_command_frames(
  ActuatorRange & actuators,
  const ActuatorCommand & actuator_command,
  double auxiliary_command_scale,
  std::vector<TPCANMsg> & frames)
{
  frames.reserve(frames.size() + actuators.size());
  for (std::size_t i = 0; i < actuators.size(); ++i) {
    frames.push_back(actuators[i].set_actuator_command(
      static_cast<float>(actuator_command.position_at(i)),
      static_cast<float>(actuator_command.effort_at(i)),
      static_cast<float>(actuator_command.stiffness_at(i)),
      auxiliary_command_scale).frame);
  }
}

}  // namespace can_hardware_common::core

#endif  // CAN_HARDWARE_COMMON__CORE__ACTUATOR_FRAME_ROUTER_HPP_
