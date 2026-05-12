//
// Copyright (c) 2026
//
// RobotControlProfile: robot-specific control hooks without plugin plumbing.
//

#ifndef CONTROL_ARCHITECTURE_STATE_MACHINE_ROBOT_CONTROL_PROFILE_HPP_
#define CONTROL_ARCHITECTURE_STATE_MACHINE_ROBOT_CONTROL_PROFILE_HPP_

#include <functional>
#include <utility>

namespace wbc {

class ControlArchitecture;
class StateFactory;

class RobotControlProfile {
 public:
  using StateRegistrar = std::function<void(StateFactory&)>;

  RobotControlProfile() = default;
  explicit RobotControlProfile(StateRegistrar register_states)
      : register_states_(std::move(register_states)) {}

  virtual ~RobotControlProfile() = default;

  virtual void RegisterStates(StateFactory& factory) {
    if (register_states_) {
      register_states_(factory);
    }
  }

  virtual void Configure(ControlArchitecture&) {}

 private:
  StateRegistrar register_states_;
};

}  // namespace wbc

#endif  // CONTROL_ARCHITECTURE_STATE_MACHINE_ROBOT_CONTROL_PROFILE_HPP_
