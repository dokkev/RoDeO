//
// Copyright (c) 2026
//
// FSM handler: manages state lifecycle and transitions.
//

#ifndef WBC_CORE_ARCHITECTURE_FSM_HANDLER_HPP_
#define WBC_CORE_ARCHITECTURE_FSM_HANDLER_HPP_

#include <atomic>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "wbc_core/architecture/state_machine.hpp"

namespace wbc {

class FSMHandler {
 public:
  FSMHandler() = default;

  /// Register a state (takes ownership).
  void RegisterState(StateId id, std::unique_ptr<StateMachine> state) {
    state_map_[id] = std::move(state);
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
  StateMachine* GetCurrentState() const { return current_state_; }

  /// Update FSM: lifecycle + transitions.
  void Update(double global_time) {
    if (!current_state_) return;

    // Check for pending transition request
    StateId requested = requested_state_.exchange(-1);
    if (requested >= 0 && requested != current_state_id_.load()) {
      // Exit current state
      current_state_->LastVisit();
      // Switch
      current_state_ = state_map_[requested].get();
      current_state_id_.store(requested);
      is_first_visit_ = true;
    }

    // First visit
    if (is_first_visit_) {
      current_state_->EnterState(global_time);
      current_state_->FirstVisit();
      is_first_visit_ = false;
    }

    // Update time
    current_state_->UpdateStateTime(global_time);

    // Tick
    current_state_->OneStep();

    // Auto-transition
    if (current_state_->EndOfState()) {
      StateId next = current_state_->GetNextState();
      if (next >= 0 && state_map_.find(next) != state_map_.end()) {
        current_state_->LastVisit();
        current_state_ = state_map_[next].get();
        current_state_id_.store(next);
        is_first_visit_ = true;
      }
    }
  }

  /// Access all registered states.
  const std::unordered_map<StateId, std::unique_ptr<StateMachine>>&
  states() const {
    return state_map_;
  }

 private:
  std::atomic<StateId> current_state_id_{-1};
  StateMachine* current_state_{nullptr};
  bool is_first_visit_{false};
  std::atomic<StateId> requested_state_{-1};
  std::unordered_map<StateId, std::unique_ptr<StateMachine>> state_map_;
};

}  // namespace wbc

#endif  // WBC_CORE_ARCHITECTURE_FSM_HANDLER_HPP_
