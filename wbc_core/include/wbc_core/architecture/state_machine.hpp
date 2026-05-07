//
// Copyright (c) 2026
//
// Base FSM state for WBMC architecture.
//

#ifndef WBC_CORE_ARCHITECTURE_STATE_MACHINE_HPP_
#define WBC_CORE_ARCHITECTURE_STATE_MACHINE_HPP_

#include <memory>
#include <string>
#include <unordered_map>

#include <pinocchio/multibody/data.hpp>
#include <yaml-cpp/yaml.h>

#include <wbc_core/robots/robot-wrapper.hpp>
#include <wbc_core/tasks/task-com-equality.hpp>
#include <wbc_core/tasks/task-joint-posture.hpp>
#include <wbc_core/tasks/task-se3-equality.hpp>

#include "wbc_core/architecture/state_provider.hpp"

namespace wbc {

using StateId = int;

/// Context passed to state constructors.
struct StateMachineContext {
  tsid::robots::RobotWrapper* robot{nullptr};
  pinocchio::Data* data{nullptr};
  StateProvider* state_provider{nullptr};
};

/// Base class for FSM states.
class StateMachine {
 public:
  StateMachine(StateId id, const std::string& name,
               const StateMachineContext& ctx)
      : state_id_(id),
        state_name_(name),
        robot_(ctx.robot),
        data_(ctx.data),
        sp_(ctx.state_provider) {}

  virtual ~StateMachine() = default;

  // ── Lifecycle (must override) ──────────────────────────────────────────
  virtual void FirstVisit() = 0;
  virtual void OneStep() = 0;
  virtual void LastVisit() = 0;

  // ── Optional overrides ─────────────────────────────────────────────────
  virtual void SetExternalInput(const TaskInput& /*input*/) {}

  virtual bool EndOfState() const {
    if (stay_here_) return false;
    return current_time_ >= (duration_ + wait_time_);
  }

  virtual StateId GetNextState() const { return next_state_id_; }

  virtual void SetParameters(const YAML::Node& node) {
    if (node["duration"]) duration_ = node["duration"].as<double>();
    if (node["wait_time"]) wait_time_ = node["wait_time"].as<double>();
    if (node["next_state_id"])
      next_state_id_ = node["next_state_id"].as<StateId>();
    if (node["stay_here"]) stay_here_ = node["stay_here"].as<bool>();
    if (node["b_stay_here"]) stay_here_ = node["b_stay_here"].as<bool>();
  }

  // ── State entry/exit (called by FSMHandler) ────────────────────────────
  void EnterState(double global_time) {
    start_time_ = global_time;
    current_time_ = 0.0;
  }

  void UpdateStateTime(double global_time) {
    current_time_ = global_time - start_time_;
  }

  // ── Accessors ──────────────────────────────────────────────────────────
  StateId id() const { return state_id_; }
  const std::string& name() const { return state_name_; }
  double elapsed_time() const { return current_time_; }

  // ── Task assignment (at config time) ───────────────────────────────────
  void assignTask(const std::string& name,
                  std::shared_ptr<tsid::tasks::TaskMotion> task) {
    assigned_tasks_[name] = task;
  }

  std::shared_ptr<tsid::tasks::TaskMotion> getTask(
      const std::string& name) const {
    auto it = assigned_tasks_.find(name);
    return (it != assigned_tasks_.end()) ? it->second : nullptr;
  }

 protected:
  StateId state_id_;
  std::string state_name_;
  tsid::robots::RobotWrapper* robot_;
  pinocchio::Data* data_;
  StateProvider* sp_;

  double duration_{3.0};
  double wait_time_{0.0};
  StateId next_state_id_{-1};
  bool stay_here_{false};

  double start_time_{-1.0};
  double current_time_{0.0};

  std::unordered_map<std::string, std::shared_ptr<tsid::tasks::TaskMotion>>
      assigned_tasks_;
};

}  // namespace wbc

#endif  // WBC_CORE_ARCHITECTURE_STATE_MACHINE_HPP_
