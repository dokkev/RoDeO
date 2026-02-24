#pragma once

#include <atomic>
#include <iostream>
#include <memory>
#include <unordered_map>

#include "wbc_fsm/interface/state_machine.hpp"

namespace wbc {

class FSMHandler {
public:
  FSMHandler()
      : current_state_id_(-1),
        current_state_(nullptr),
        b_first_visit_(true),
        transition_error_latched_(false),
        requested_state_(kNoRequest) {}
  ~FSMHandler() = default;

  // 1. 상태 등록 (소유권 이전)
  void RegisterState(StateId id, std::unique_ptr<StateMachine> state) {
    if (state_map_.find(id) != state_map_.end()) {
      std::cerr << "[FSMHandler] Warning: State " << id << " overwritten."
                << std::endl;
    }
    state_map_[id] = std::move(state);
  }

  // 2. 시작 상태 설정
  void SetStartState(StateId id) {
    if (state_map_.find(id) == state_map_.end()) {
      std::cerr << "[FSMHandler] Error: Start State " << id << " not found!"
                << std::endl;
      return;
    }
    current_state_ = state_map_[id].get();
    current_state_id_ = id;
    b_first_visit_ = true;
    transition_error_latched_ = false;
  }

  // 3. 메인 루프 (매 틱마다 호출)
  void Update() {
    UpdateImpl(nullptr);
  }

  // 3. 메인 루프 (전역 시간을 같이 전달하는 버전)
  void Update(double current_global_time) {
    UpdateImpl(&current_global_time);
  }

  // Getter
  StateId GetCurrentStateId() const { return current_state_id_; }
  StateId CurrentStateId() const { return current_state_id_; }

  // 외부(teleop/UI)에서 강제 상태 전환 요청 (thread-safe latch)
  void RequestState(StateId id) { requested_state_.store(id); }

  bool ConsumeRequestedState(StateId& out_id) {
    const StateId requested = requested_state_.exchange(kNoRequest);
    if (requested == kNoRequest) {
      return false;
    }
    out_id = requested;
    return true;
  }

  // update thread에서만 호출: 즉시 상태 전환
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
    current_state_id_ = id;
    b_first_visit_ = true;
    transition_error_latched_ = false;
    return true;
  }

private:
  static constexpr StateId kNoRequest = -1;

  void UpdateImpl(const double* current_global_time) {
    if (!current_state_) {
      return;
    }

    // (1) First Visit
    if (b_first_visit_) {
      if (current_global_time != nullptr) {
        current_state_->EnterState(*current_global_time);
      }
      current_state_->FirstVisit();
      b_first_visit_ = false;
    }

    // (2) One Step (Nominal Behavior)
    if (current_global_time != nullptr) {
      current_state_->UpdateStateTime(*current_global_time);
    }
    current_state_->OneStep();

    // (3) Transition Logic
    if (current_state_->EndOfState()) {
      current_state_->LastVisit(); // Clean up current state

      StateId next_id = current_state_->GetNextState();
      if (state_map_.find(next_id) == state_map_.end()) {
        if (!transition_error_latched_) {
          std::cerr << "[FSMHandler] Error: Next State " << next_id
                    << " not found. FSM is latched in safe-stop mode."
                    << std::endl;
          transition_error_latched_ = true;
        }
        current_state_ = nullptr;
        current_state_id_ = -1;
        b_first_visit_ = true;
        return;
      }

      // Switch
      current_state_ = state_map_[next_id].get();
      current_state_id_ = next_id;
      b_first_visit_ = true; // Trigger FirstVisit next cycle
    }
  }
  StateId current_state_id_;
  StateMachine* current_state_;
  bool b_first_visit_;
  bool transition_error_latched_;
  std::atomic<StateId> requested_state_;

  // 상태들을 보관하는 컨테이너 (Architecture 클래스 대신 얘가 관리)
  std::unordered_map<StateId, std::unique_ptr<StateMachine>> state_map_;
};

} // namespace wbc
