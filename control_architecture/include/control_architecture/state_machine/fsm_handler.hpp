//
// Copyright (c) 2026
//
// FSM handler: manages state lifecycle and transitions.
//

#ifndef CONTROL_ARCHITECTURE_STATE_MACHINE_FSM_HANDLER_HPP_
#define CONTROL_ARCHITECTURE_STATE_MACHINE_FSM_HANDLER_HPP_

#include <atomic>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

#include "control_architecture/state_machine/state_machine.hpp"

namespace wbc {

class FSMHandler {
 public:
  FSMHandler() = default;

  using StateMap = std::unordered_map<StateId, std::unique_ptr<State>>;

  /// Register a state (takes ownership).
  void RegisterState(StateId id, std::unique_ptr<State> state) {
    if (!state) {
      throw std::invalid_argument("FSMHandler::RegisterState: state is null");
    }
    if (state->id() != id) {
      throw std::invalid_argument(
          "FSMHandler::RegisterState: id does not match state id");
    }
    if (!state_map_.emplace(id, std::move(state)).second) {
      throw std::invalid_argument(
          "FSMHandler::RegisterState: duplicate state id");
    }
  }

  /// Set the initial state.
  bool SetStartState(StateId id) {
    auto it = state_map_.find(id);
    if (it == state_map_.end()) return false;
    current_state_id_.store(id);
    current_state_ = it->second.get();
    is_first_visit_ = true;
    return true;
  }

  /// Thread-safe transition request.
  bool RequestState(StateId id) {
    if (state_map_.find(id) == state_map_.end()) return false;
    requested_state_.store(id);
    return true;
  }

  /// Look up state by name.
  std::optional<StateId> FindStateIdByName(const std::string& name) const {
    for (const auto& [id, state] : state_map_) {
      if (state->name() == name) return id;
    }
    return std::nullopt;
  }

  /// Current state ID (safe across threads).
  StateId GetCurrentStateId() const { return current_state_id_.load(); }

  /// Current state pointer (not thread-safe for non-owning access).
  State* GetCurrentState() const { return current_state_; }

  /// Update FSM: lifecycle + transitions.
  void Update(double global_time, double dt = 0.0) {
    if (!current_state_) return;

    StateId requested = requested_state_.exchange(-1);
    if (requested >= 0 && requested != current_state_id_.load()) {
      SwitchToState(requested, global_time);
    }

    EnterCurrentStateIfNeeded(global_time);

    current_state_->UpdateTime(global_time);

    if (current_state_->IsFinished()) {
      StateId next = current_state_->NextState();
      if (next >= 0 && SwitchToState(next, global_time)) {
        current_state_->UpdateTime(global_time);
      }
    }

    current_state_->Tick(dt);
  }

  /// Access all registered states.
  const StateMap& states() const {
    return state_map_;
  }

 private:
  void EnterCurrentStateIfNeeded(double global_time) {
    if (!is_first_visit_ || !current_state_) return;
    current_state_->Enter(global_time);
    is_first_visit_ = false;
  }

  bool SwitchToState(StateId id, double global_time) {
    auto it = state_map_.find(id);
    if (it == state_map_.end()) return false;
    if (current_state_) current_state_->Exit();
    current_state_ = it->second.get();
    current_state_id_.store(id);
    is_first_visit_ = true;
    EnterCurrentStateIfNeeded(global_time);
    return true;
  }

  std::atomic<StateId> current_state_id_{-1};
  State* current_state_{nullptr};
  bool is_first_visit_{false};
  std::atomic<StateId> requested_state_{-1};
  StateMap state_map_;
};

}  // namespace wbc

#endif  // CONTROL_ARCHITECTURE_STATE_MACHINE_FSM_HANDLER_HPP_
