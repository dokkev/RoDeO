#ifndef PLATO_HARDWARE_INTERFACE__PLATO_STATE_HELPER_HPP_
#define PLATO_HARDWARE_INTERFACE__PLATO_STATE_HELPER_HPP_

#include <vector>

#include "can_hardware_common/core/state_snapshot.hpp"
#include "can_hardware_common/robot.hpp"
#include "plato_hardware_interface/actuator.hpp"
#include "plato_hardware_interface/five_bar_linkage.hpp"

namespace plato_hand
{

class PlatoStateHelper
{
public:
  explicit PlatoStateHelper(FiveBarLinkage::Transmission & transmission)
  : transmission_(transmission)
  {
  }

  void update_joint_states(
    const std::vector<plato_actuator::Actuator> & actuators,
    const can_hardware_common::RobotIO::JointCommand & joint_commands,
    can_hardware_common::RobotIO::ActuatorState & actuator_states,
    can_hardware_common::RobotIO::JointState & joint_states) const;

  void joint_to_actuator_commands(
    const can_hardware_common::RobotIO::JointCommand & joint_commands,
    const can_hardware_common::RobotIO::ActuatorState & actuator_states,
    const can_hardware_common::RobotIO::JointState & joint_states,
    can_hardware_common::RobotIO::ActuatorCommand & actuator_commands) const;

  bool actuators_ready(const std::vector<plato_actuator::Actuator> & actuators) const;

  void copy_feedback_snapshot(
    const std::vector<plato_actuator::Actuator> & actuators,
    can_hardware_common::core::StateSnapshot & snapshot) const;

private:
  FiveBarLinkage::Transmission & transmission_;
};

}  // namespace plato_hand

#endif  // PLATO_HARDWARE_INTERFACE__PLATO_STATE_HELPER_HPP_
