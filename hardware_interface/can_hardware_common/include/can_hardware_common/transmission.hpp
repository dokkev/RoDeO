#ifndef CAN_HARDWARE_COMMON__TRANSMISSION_HPP_
#define CAN_HARDWARE_COMMON__TRANSMISSION_HPP_

#include "can_hardware_common/robot.hpp"

namespace can_hardware_common
{

class Transmission
{
public:
  virtual ~Transmission() = default;

  virtual void actuator_to_joint(
    const RobotIO::ActuatorState & actuator_state,
    RobotIO::JointState & joint_state) const = 0;

  virtual void joint_to_actuator(
    const RobotIO::JointCommand & joint_command,
    const RobotIO::ActuatorState & actuator_state,
    const RobotIO::JointState & joint_state,
    RobotIO::ActuatorCommand & actuator_command) const = 0;
};

}  // namespace can_hardware_common

#endif  // CAN_HARDWARE_COMMON__TRANSMISSION_HPP_
