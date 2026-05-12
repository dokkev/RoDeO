#ifndef CAN_HARDWARE_COMMON__ROBOT_HPP_
#define CAN_HARDWARE_COMMON__ROBOT_HPP_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace can_hardware_common
{

namespace RobotIO
{

namespace detail
{

template<typename Scalar>
inline bool all_finite(const std::vector<Scalar> & values)
{
  return std::all_of(
    values.begin(), values.end(), [](Scalar value) { return std::isfinite(value); });
}

template<typename Scalar>
inline void assign_all(std::vector<Scalar> & values, Scalar value)
{
  std::fill(values.begin(), values.end(), value);
}

template<typename Scalar>
inline void copy_values(std::vector<Scalar> & dst, const std::vector<Scalar> & src)
{
  if (dst.size() != src.size()) {
    dst.resize(src.size());
  }
  std::copy(src.begin(), src.end(), dst.begin());
}

template<typename Scalar>
class StateBuffer
{
public:
  void resize(std::size_t size, Scalar value = std::numeric_limits<Scalar>::quiet_NaN())
  {
    position_.assign(size, value);
    velocity_.assign(size, value);
    effort_.assign(size, value);
  }

  void fill(Scalar value)
  {
    assign_all(position_, value);
    assign_all(velocity_, value);
    assign_all(effort_, value);
  }

  void copy_from(const StateBuffer & other)
  {
    copy_values(position_, other.position_);
    copy_values(velocity_, other.velocity_);
    copy_values(effort_, other.effort_);
  }

  std::size_t size() const { return position_.size(); }
  bool all_finite() const
  {
    return detail::all_finite(position_) && detail::all_finite(velocity_) &&
           detail::all_finite(effort_);
  }

  Scalar & position_at(std::size_t index) { return position_.at(index); }
  Scalar & velocity_at(std::size_t index) { return velocity_.at(index); }
  Scalar & effort_at(std::size_t index) { return effort_.at(index); }
  const Scalar & position_at(std::size_t index) const { return position_.at(index); }
  const Scalar & velocity_at(std::size_t index) const { return velocity_.at(index); }
  const Scalar & effort_at(std::size_t index) const { return effort_.at(index); }

  Scalar position(std::size_t index) const { return position_.at(index); }
  Scalar velocity(std::size_t index) const { return velocity_.at(index); }
  Scalar effort(std::size_t index) const { return effort_.at(index); }

  Scalar * position_data() { return position_.data(); }
  Scalar * velocity_data() { return velocity_.data(); }
  Scalar * effort_data() { return effort_.data(); }
  const Scalar * position_data() const { return position_.data(); }
  const Scalar * velocity_data() const { return velocity_.data(); }
  const Scalar * effort_data() const { return effort_.data(); }

private:
  std::vector<Scalar> position_;
  std::vector<Scalar> velocity_;
  std::vector<Scalar> effort_;
};

template<typename Scalar>
class CommandBuffer
{
public:
  void resize(std::size_t size, Scalar value = std::numeric_limits<Scalar>::quiet_NaN())
  {
    position_.assign(size, value);
    velocity_.assign(size, value);
    effort_.assign(size, value);
    stiffness_.assign(size, value);
    damping_.assign(size, value);
  }

  void fill(Scalar value)
  {
    assign_all(position_, value);
    assign_all(velocity_, value);
    assign_all(effort_, value);
    assign_all(stiffness_, value);
    assign_all(damping_, value);
  }

  void copy_from(const CommandBuffer & other)
  {
    copy_values(position_, other.position_);
    copy_values(velocity_, other.velocity_);
    copy_values(effort_, other.effort_);
    copy_values(stiffness_, other.stiffness_);
    copy_values(damping_, other.damping_);
  }

  std::size_t size() const { return position_.size(); }
  bool all_finite() const
  {
    return detail::all_finite(position_) && detail::all_finite(velocity_) &&
           detail::all_finite(effort_) && detail::all_finite(stiffness_) &&
           detail::all_finite(damping_);
  }

  Scalar & position_at(std::size_t index) { return position_.at(index); }
  Scalar & velocity_at(std::size_t index) { return velocity_.at(index); }
  Scalar & effort_at(std::size_t index) { return effort_.at(index); }
  Scalar & stiffness_at(std::size_t index) { return stiffness_.at(index); }
  Scalar & damping_at(std::size_t index) { return damping_.at(index); }
  const Scalar & position_at(std::size_t index) const { return position_.at(index); }
  const Scalar & velocity_at(std::size_t index) const { return velocity_.at(index); }
  const Scalar & effort_at(std::size_t index) const { return effort_.at(index); }
  const Scalar & stiffness_at(std::size_t index) const { return stiffness_.at(index); }
  const Scalar & damping_at(std::size_t index) const { return damping_.at(index); }

  Scalar position(std::size_t index) const { return position_.at(index); }
  Scalar velocity(std::size_t index) const { return velocity_.at(index); }
  Scalar effort(std::size_t index) const { return effort_.at(index); }
  Scalar stiffness(std::size_t index) const { return stiffness_.at(index); }
  Scalar damping(std::size_t index) const { return damping_.at(index); }

  Scalar * position_data() { return position_.data(); }
  Scalar * velocity_data() { return velocity_.data(); }
  Scalar * effort_data() { return effort_.data(); }
  Scalar * stiffness_data() { return stiffness_.data(); }
  Scalar * damping_data() { return damping_.data(); }
  const Scalar * position_data() const { return position_.data(); }
  const Scalar * velocity_data() const { return velocity_.data(); }
  const Scalar * effort_data() const { return effort_.data(); }
  const Scalar * stiffness_data() const { return stiffness_.data(); }
  const Scalar * damping_data() const { return damping_.data(); }

private:
  std::vector<Scalar> position_;
  std::vector<Scalar> velocity_;
  std::vector<Scalar> effort_;
  std::vector<Scalar> stiffness_;
  std::vector<Scalar> damping_;
};

}  // namespace detail

using JointState = detail::StateBuffer<double>;
using JointCommand = detail::CommandBuffer<double>;
using ActuatorState = detail::StateBuffer<float>;
using ActuatorCommand = detail::CommandBuffer<float>;

template<typename Snapshot>
inline void copy_joint_state_to_snapshot(const JointState & joint_state, Snapshot & snapshot)
{
  snapshot.joint_position.resize(joint_state.size());
  snapshot.joint_velocity.resize(joint_state.size());
  snapshot.joint_effort.resize(joint_state.size());
  std::copy_n(joint_state.position_data(), joint_state.size(), snapshot.joint_position.begin());
  std::copy_n(joint_state.velocity_data(), joint_state.size(), snapshot.joint_velocity.begin());
  std::copy_n(joint_state.effort_data(), joint_state.size(), snapshot.joint_effort.begin());
}

template<typename ActuatorRange, typename StateGetter>
inline bool copy_actuator_feedback_to_buffer(
  const ActuatorRange & actuators,
  ActuatorState & actuator_state,
  StateGetter get_state)
{
  if (actuator_state.size() != actuators.size()) {
    return false;
  }

  for (std::size_t i = 0; i < actuators.size(); ++i) {
    const auto state = get_state(actuators[i]);
    actuator_state.position_at(i) = state.position;
    actuator_state.velocity_at(i) = state.velocity;
    actuator_state.effort_at(i) = state.torque;
  }

  return true;
}

template<typename ActuatorRange, typename Snapshot, typename StateGetter>
inline void copy_actuator_feedback_to_snapshot(
  const ActuatorRange & actuators,
  Snapshot & snapshot,
  StateGetter get_state)
{
  if (snapshot.actuator_states.size() != actuators.size()) {
    snapshot.actuator_states.assign(actuators.size(), {});
  }

  for (std::size_t i = 0; i < actuators.size(); ++i) {
    snapshot.actuator_states[i] = get_state(actuators[i]);
  }
}

template<typename ActuatorRange, typename ReadyPredicate>
inline bool any_actuator_ready(const ActuatorRange & actuators, ReadyPredicate is_ready)
{
  return std::any_of(actuators.begin(), actuators.end(), is_ready);
}

template<typename ActuatorRange, typename ReadyPredicate>
inline bool all_actuators_ready_from(
  const ActuatorRange & actuators,
  std::size_t first_index,
  ReadyPredicate is_ready)
{
  if (actuators.size() <= first_index) {
    return false;
  }

  return std::all_of(actuators.begin() + first_index, actuators.end(), is_ready);
}

}  // namespace RobotIO

}  // namespace can_hardware_common

#endif  // CAN_HARDWARE_COMMON__ROBOT_HPP_
