/**
 * @file wbc_core/wbc_fsm/include/wbc_fsm/fsm_handler.hpp
 * @brief Doxygen documentation for fsm_handler module.
 */
#pragma once

#include <algorithm>
#include <atomic>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "wbc_fsm/interface/state_machine.hpp"

namespace wbc {

/**
 * @brief Runtime finite-state-machine executor.
 *
 * @details
 * `FSMHandler` owns state instances and drives lifecycle in update order:
 * `FirstVisit -> OneStep -> EndOfState/transition -> LastVisit`.
 *
 * State transition requests from non-RT sources can be latched via
 * `RequestState(...)` and consumed in the RT thread with `ConsumeRequestedState`.
 */
class FSMHandler {
public:
  FSMHandler()
      : current_state_id_(-1),
        current_state_(nullptr),
        is_first_visit_(true),
        transition_error_latched_(false),
        requested_state_(kNoRequest) {}
  ~FSMHandler() = default;

  /**
   * @brief Register (or replace) a state implementation.
   *
   * @note Does NOT rebuild the state catalog. Call `FinalizeStates()` once
   *       after all `RegisterState()` calls to make `GetStates()` consistent.
   */
  void RegisterState(StateId id, std::unique_ptr<StateMachine> state) {
    if (state_map_.find(id) != state_map_.end()) {
      std::cerr << "[FSMHandler] Warning: State " << id << " overwritten."
                << std::endl;
    }
    state_map_[id] = std::move(state);
  }

  /**
   * @brief Rebuild the sorted state catalog after all states are registered.
   *
   * @details Call once after the last `RegisterState()` to finalize the catalog
   *          returned by `GetStates()`. Typically invoked by
   *          `ConfigCompiler::InitializeFsm()`.
   */
  void FinalizeStates() { RebuildStateCatalog(); }

  /**
   * @brief Set initial active state id.
   * @return true if state exists and start state is set.
   */
  bool SetStartState(StateId id) {
    const auto it = state_map_.find(id);
    if (it == state_map_.end()) {
      std::cerr << "[FSMHandler] Error: Start State " << id << " not found!"
                << std::endl;
      current_state_ = nullptr;
      current_state_id_.store(-1, std::memory_order_release);
      is_first_visit_.store(true, std::memory_order_release);
      return false;
    }
    current_state_ = it->second.get();
    current_state_id_.store(id, std::memory_order_release);
    is_first_visit_.store(true, std::memory_order_release);
    transition_error_latched_ = false;
    return true;
  }

  /**
   * @brief Update FSM without externally provided global time.
   */
  void Update() {
    UpdateImpl(nullptr);
  }

  /**
   * @brief Update FSM with externally provided global time.
   */
  void Update(double current_global_time) {
    UpdateImpl(&current_global_time);
  }

  /**
   * @brief Current active state id, or -1 when inactive.
   */
  StateId GetCurrentStateId() const {
    return current_state_id_.load(std::memory_order_acquire);
  }

  /**
   * @brief Raw pointer to the active state, or nullptr when no state is active.
   *
   * @note Non-RT use only. Do not hold across ticks — FSM may transition.
   */
  StateMachine* GetCurrentState() const { return current_state_; }

  /**
   * @brief Returns true when current state is pending first-visit handling.
   */
  bool IsFirstVisit() const {
    return is_first_visit_.load(std::memory_order_acquire);
  }

  /**
   * @brief Find a registered state by id. Returns nullptr if not found.
   *
   * @note Non-RT use only (configure / activate phase).
   */
  StateMachine* FindStateById(StateId id) const {
    const auto it = state_map_.find(id);
    return (it != state_map_.end()) ? it->second.get() : nullptr;
  }

  /**
   * @brief Resolve state id by registered state name.
   */
  std::optional<StateId> FindStateIdByName(const std::string& name) const {
    if (name.empty()) {
      return std::nullopt;
    }
    for (const auto& [state_id, state] : state_map_) {
      if (state != nullptr && state->name() == name) {
        return state_id;
      }
    }
    return std::nullopt;
  }

  /**
   * @brief Request transition by registered state name.
   * @return true when state name exists and request was latched.
   */
  bool RequestStateByName(const std::string& name) {
    const std::optional<StateId> state_id = FindStateIdByName(name);
    if (!state_id.has_value()) {
      return false;
    }
    RequestState(*state_id);
    return true;
  }

  /**
   * @brief Sorted catalog of registered states as `(id, name)` pairs.
   *
   * @note Non-RT API. Access this after registration/configuration phase.
   * The returned reference is allocation-free and remains valid until the next
   * `RegisterState(...)` call.
   */
  const std::vector<std::pair<StateId, std::string>>& GetStates() const {
    return state_catalog_;
  }

  /**
   * @brief Thread-safe state transition request latch.
   */
  void RequestState(StateId id) { requested_state_.store(id); }

  /**
   * @brief Consume latest request once.
   * @return true when a request existed and was consumed.
   */
  bool ConsumeRequestedState(StateId& out_id) {
    const StateId requested = requested_state_.exchange(kNoRequest);
    if (requested == kNoRequest) {
      return false;
    }
    out_id = requested;
    return true;
  }

  /**
   * @brief Immediate transition helper intended for update thread usage.
   *
   * @warning Calls `LastVisit()` of the current state inline. Ensure the
   *          current state's `LastVisit()` is RT-safe (no heap allocation,
   *          no blocking I/O) before invoking from the RT thread.
   */
  bool ForceTransition(StateId id) {
    const auto it = state_map_.find(id);
    if (it == state_map_.end()) {
      std::cerr << "[FSMHandler] Error: Requested State " << id
                << " not found!" << std::endl;
      return false;
    }

    if (current_state_ != nullptr) {
      current_state_->LastVisit();
    }

    current_state_ = it->second.get();
    current_state_id_.store(id, std::memory_order_release);
    is_first_visit_.store(true, std::memory_order_release);
    transition_error_latched_ = false;
    return true;
  }

private:
  static constexpr StateId kNoRequest = -1;

  void RebuildStateCatalog() {
    state_catalog_.clear();
    state_catalog_.reserve(state_map_.size());
    for (const auto& [state_id, state] : state_map_) {
      state_catalog_.emplace_back(state_id,
                                  state != nullptr ? state->name() : "");
    }
    std::sort(state_catalog_.begin(), state_catalog_.end(),
              [](const auto& lhs, const auto& rhs) {
                return lhs.first < rhs.first;
              });
  }

  void UpdateImpl(const double* current_global_time) {
    if (!current_state_) {
      return;
    }
    const bool has_time = (current_global_time != nullptr);

    // (1) First Visit
    if (is_first_visit_.load(std::memory_order_acquire)) {
      if (has_time) {
        current_state_->EnterState(*current_global_time);
      }
      current_state_->FirstVisit();
      is_first_visit_.store(false, std::memory_order_release);
    }

    // (2) One Step (Nominal Behavior)
    if (has_time) {
      current_state_->UpdateStateTime(*current_global_time);
    }
    current_state_->OneStep();

    // (3) Transition Logic
    if (current_state_->EndOfState()) {
      current_state_->LastVisit(); // Clean up current state

      StateId next_id = current_state_->GetNextState();
      const auto next_it = state_map_.find(next_id);
      if (next_it == state_map_.end()) {
        if (!transition_error_latched_) {
          std::cerr << "[FSMHandler] Error: Next State " << next_id
                    << " not found. FSM is latched in safe-stop mode."
                    << std::endl;
          transition_error_latched_ = true;
        }
        current_state_ = nullptr;
        current_state_id_.store(-1, std::memory_order_release);
        is_first_visit_.store(true, std::memory_order_release);
        return;
      }

      // Switch and run FirstVisit immediately so the rest of this control tick
      // observes a state/configuration-consistent desired/task setup.
      current_state_ = next_it->second.get();
      current_state_id_.store(next_id, std::memory_order_release);
      if (has_time) {
        current_state_->EnterState(*current_global_time);
      }
      current_state_->FirstVisit();
      if (has_time) {
        current_state_->UpdateStateTime(*current_global_time);
      }
      is_first_visit_.store(false, std::memory_order_release);
      transition_error_latched_ = false;
      return;
    }
  }
  std::atomic<StateId> current_state_id_;
  StateMachine* current_state_;
  std::atomic<bool> is_first_visit_;
  bool transition_error_latched_;
  std::atomic<StateId> requested_state_;

  // Container that owns all registered states (managed here, not by ControlArchitecture).
  std::unordered_map<StateId, std::unique_ptr<StateMachine>> state_map_;
  std::vector<std::pair<StateId, std::string>> state_catalog_;
};

} // namespace wbc
