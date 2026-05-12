#ifndef CAN_HARDWARE_COMMON__CORE__STATE_SNAPSHOT_HPP_
#define CAN_HARDWARE_COMMON__CORE__STATE_SNAPSHOT_HPP_

#include <cstddef>
#include <vector>

#include <rclcpp/time.hpp>

#include "can_hardware_common/actuator_types.hpp"

namespace can_hardware_common::core
{

struct StateSnapshot
{
  struct SensorStatus
  {
    std::size_t expected_count = 0;
    std::size_t available_count = 0;
    std::size_t fresh_count = 0;

    bool ok() const
    {
      return expected_count == available_count && expected_count == fresh_count;
    }
  };

  rclcpp::Time stamp{};
  bool has_fresh_rx = false;
  bool calibrated = false;
  bool sensors_ok = false;
  bool actuators_ready = false;
  bool transport_healthy = false;
  bool lifecycle_busy = false;
  bool model_ready = false;
  SensorStatus sensor_status{};

  std::vector<double> joint_position;
  std::vector<double> joint_velocity;
  std::vector<double> joint_effort;
  std::vector<can_hardware_common::ActuatorState> actuator_states;

  void resize(std::size_t num_joints, std::size_t num_actuators)
  {
    joint_position.resize(num_joints);
    joint_velocity.resize(num_joints);
    joint_effort.resize(num_joints);
    actuator_states.assign(num_actuators, can_hardware_common::ActuatorState{});
  }
};

}  // namespace can_hardware_common::core

#endif  // CAN_HARDWARE_COMMON__CORE__STATE_SNAPSHOT_HPP_
