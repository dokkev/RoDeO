#ifndef PLATO_HARDWARE_INTERFACE__FIVE_BAR_LINKAGE_HPP_
#define PLATO_HARDWARE_INTERFACE__FIVE_BAR_LINKAGE_HPP_

#include <array>
#include <cstddef>

#include "can_hardware_common/transmission.hpp"
#include "plato_hardware_interface/plato_layout.hpp"

namespace FiveBarLinkage
{

struct FiveBarLinkageConfig
{
  float L1;
  float L2;
  float L3;
  float L4;
  float L5;
  float eef_linkage_angle;
  float eef_length;
};

struct Kinematics
{
  float position_amplification = 1.0f;
  float torque_amplification = 1.0f;
};

class Transmission : public can_hardware_common::Transmission
{
public:
  static constexpr size_t kNumJoints = plato_hand::layout::kNumJoints;
  static constexpr size_t kNumActuators = plato_hand::layout::kNumActuators;
  using JointArray = std::array<float, kNumJoints>;

  explicit Transmission(FiveBarLinkageConfig config);
  Transmission(FiveBarLinkageConfig config, const JointArray & joint_effort_limits);

  void actuator_to_joint(
    const can_hardware_common::RobotIO::ActuatorState & actuator_state,
    can_hardware_common::RobotIO::JointState & joint_state) const override;

  void joint_to_actuator(
    const can_hardware_common::RobotIO::JointCommand & joint_command,
    const can_hardware_common::RobotIO::ActuatorState & actuator_state,
    const can_hardware_common::RobotIO::JointState & joint_state,
    can_hardware_common::RobotIO::ActuatorCommand & actuator_command) const override;

private:
  void compute_ratios_(
    const can_hardware_common::RobotIO::ActuatorState & actuator_state,
    JointArray & position_ratios,
    JointArray & velocity_ratios,
    JointArray & torque_ratios) const;

  FiveBarLinkageConfig config_;
  JointArray joint_effort_limits_;
};

Kinematics compute_kinematics(
  const FiveBarLinkageConfig & config,
  float mcp_motor_angle,
  float pip_motor_angle);

}  // namespace FiveBarLinkage

#endif  // PLATO_HARDWARE_INTERFACE__FIVE_BAR_LINKAGE_HPP_
