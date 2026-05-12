#ifndef CAN_HARDWARE_COMMON__CORE__CAN_HAND_BASE_HPP_
#define CAN_HARDWARE_COMMON__CORE__CAN_HAND_BASE_HPP_

#include <vector>

#include <PCANBasic.h>

#include "can_hardware_common/core/state_snapshot.hpp"
#include "can_hardware_common/robot.hpp"

namespace can_hardware_common::core
{

enum class DispatchPolicy
{
  kDirectFrames,
  kCustomExecution,
};

enum class LifecycleOperation
{
  kEnable,
  kDisable,
  kZero,
};

struct LifecyclePlan
{
  LifecycleOperation operation = LifecycleOperation::kEnable;
  DispatchPolicy dispatch_policy = DispatchPolicy::kDirectFrames;
  bool ready = false;
  std::vector<TPCANMsg> direct_frames;
};

struct WritePlan
{
  DispatchPolicy dispatch_policy = DispatchPolicy::kDirectFrames;
  bool ready = false;
  std::vector<TPCANMsg> direct_frames;
};

class CanHandBase
{
public:
  virtual ~CanHandBase() = default;

  bool read()
  {
    const bool ok = update_measurements_();
    refresh_state_snapshot_();
    return ok;
  }

  bool write()
  {
    WritePlan plan;
    plan.ready = joint_commands_.all_finite();
    if (!plan.ready) {
      previous_joint_command_valid_ = false;
      previous_actuator_command_valid_ = false;
      return true;
    }

    build_ready_write_plan_(plan);
    previous_joint_command_.copy_from(joint_commands_);
    previous_joint_command_valid_ = true;
    previous_actuator_command_.copy_from(actuator_commands_);
    previous_actuator_command_valid_ = actuator_commands_.size() > 0U;

    return execute_write_plan_(plan);
  }

  bool enable(bool zero_first = false)
  {
    if (zero_first) {
      const auto zero_plan = build_lifecycle_plan_(LifecycleOperation::kZero);
      if (!zero_plan.ready || !execute_lifecycle_plan_(zero_plan)) {
        return false;
      }
    }

    const auto plan = build_lifecycle_plan_(LifecycleOperation::kEnable);
    return plan.ready && execute_lifecycle_plan_(plan);
  }

  bool disable()
  {
    const auto plan = build_lifecycle_plan_(LifecycleOperation::kDisable);
    return plan.ready && execute_lifecycle_plan_(plan);
  }

  void initialize_joint_buffers(std::size_t num_joints, double value)
  {
    joint_commands_.resize(num_joints, value);
    joint_states_.resize(num_joints, value);
    previous_joint_command_.resize(num_joints, value);
  }

  void initialize_actuator_buffers(std::size_t num_actuators, float value)
  {
    actuator_commands_.resize(num_actuators, value);
    actuator_states_.resize(num_actuators, value);
    previous_actuator_command_.resize(num_actuators, value);
  }

  void reset_joint_commands(double value)
  {
    joint_commands_.fill(value);
    previous_joint_command_valid_ = false;
  }

  RobotIO::JointCommand & joint_commands() { return joint_commands_; }
  const RobotIO::JointCommand & joint_commands() const { return joint_commands_; }
  RobotIO::JointState & joint_states() { return joint_states_; }
  const RobotIO::JointState & joint_states() const { return joint_states_; }
  RobotIO::ActuatorCommand & actuator_commands() { return actuator_commands_; }
  const RobotIO::ActuatorCommand & actuator_commands() const { return actuator_commands_; }
  RobotIO::ActuatorState & actuator_states() { return actuator_states_; }
  const RobotIO::ActuatorState & actuator_states() const { return actuator_states_; }

  bool has_previous_joint_command() const { return previous_joint_command_valid_; }
  bool has_previous_actuator_command() const { return previous_actuator_command_valid_; }

  const RobotIO::JointCommand & previous_joint_command() const
  {
    return previous_joint_command_;
  }

  const RobotIO::ActuatorCommand & previous_actuator_command() const
  {
    return previous_actuator_command_;
  }

  const StateSnapshot & state_snapshot() const { return state_snapshot_; }

protected:
  virtual bool update_measurements_() = 0;
  virtual void refresh_state_snapshot_() = 0;
  virtual void build_ready_write_plan_(WritePlan & plan) = 0;
  virtual bool execute_write_plan_(const WritePlan & plan) = 0;
  virtual LifecyclePlan build_lifecycle_plan_(LifecycleOperation operation) = 0;
  virtual bool execute_lifecycle_plan_(const LifecyclePlan & plan) = 0;

  RobotIO::JointCommand joint_commands_;
  RobotIO::JointState joint_states_;
  RobotIO::ActuatorCommand actuator_commands_;
  RobotIO::ActuatorState actuator_states_;
  StateSnapshot state_snapshot_;

private:
  RobotIO::JointCommand previous_joint_command_;
  RobotIO::ActuatorCommand previous_actuator_command_;
  bool previous_joint_command_valid_ = false;
  bool previous_actuator_command_valid_ = false;
};

}  // namespace can_hardware_common::core

#endif  // CAN_HARDWARE_COMMON__CORE__CAN_HAND_BASE_HPP_
