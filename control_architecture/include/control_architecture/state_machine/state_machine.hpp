//
// Copyright (c) 2026
//
// Base FSM state for IDHQP architecture.
//

#ifndef CONTROL_ARCHITECTURE_STATE_MACHINE_STATE_MACHINE_HPP_
#define CONTROL_ARCHITECTURE_STATE_MACHINE_STATE_MACHINE_HPP_

#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include <pinocchio/multibody/data.hpp>
#include <yaml-cpp/yaml.h>

#include <wbc_core/robots/robot-system.hpp>
#include <wbc_core/tasks/task-motion.hpp>

#ifndef STATE_NAME
#define STATE_NAME(name_literal) \
  static constexpr const char* kName = name_literal
#endif

namespace wbc {

using StateId = int;
using TaskHandle = std::shared_ptr<wbc::tasks::TaskMotion>;
using TaskMap = std::unordered_map<std::string, TaskHandle>;

struct StateLifecycle {
  double duration{3.0};
  double wait_time{0.0};
  StateId next_state_id{-1};
  bool stay_here{false};
};

/// Context passed to state constructors.
struct StateContext {
  wbc::robots::RobotSystem* robot{nullptr};
  pinocchio::Data* data{nullptr};
};

/// Base class for FSM states.
class State {
 public:
  State(StateId id, const std::string& name, const StateContext& ctx)
      : state_id_(id), state_name_(name), robot_(ctx.robot), data_(ctx.data) {}

  virtual ~State() = default;

  virtual void OnEnter() {}
  virtual void OnUpdate() {}
  virtual void OnExit() {}

  virtual void Configure(const YAML::Node& node) { (void)node; }

  void ConfigureLifecycle(const StateLifecycle& lifecycle) {
    lifecycle_ = lifecycle;
  }

  virtual bool IsFinished() const {
    if (lifecycle_.stay_here) return false;
    return current_time_ >= (lifecycle_.duration + lifecycle_.wait_time);
  }

  virtual StateId NextState() const { return lifecycle_.next_state_id; }

  void Enter(double global_time) {
    start_time_ = global_time;
    current_time_ = 0.0;
    OnEnter();
  }

  void UpdateTime(double global_time) {
    current_time_ = global_time - start_time_;
  }

  void Tick(double dt) {
    dt_ = dt;
    OnUpdate();
  }

  void Exit() { OnExit(); }

  // ── Accessors ──────────────────────────────────────────────────────────
  StateId id() const { return state_id_; }
  const std::string& name() const { return state_name_; }
  double elapsed_time() const { return current_time_; }
  double dt() const { return dt_; }

  // ── Task assignment (at config time) ───────────────────────────────────
  void AssignTask(const std::string& name, TaskHandle task) {
    task_map_[name] = task;
  }

  TaskHandle Task(const std::string& name) const {
    auto it = task_map_.find(name);
    return (it != task_map_.end()) ? it->second : nullptr;
  }

  template <typename TaskT>
  std::shared_ptr<TaskT> TaskAs(const std::string& name) const {
    return std::dynamic_pointer_cast<TaskT>(Task(name));
  }

  template <typename TaskT>
  std::shared_ptr<TaskT> RequireTask(const std::string& name,
                                     const std::string& owner) const {
    auto task = TaskAs<TaskT>(name);
    if (!task) {
      throw std::invalid_argument(owner + " requires assigned task '" + name +
                                  "'");
    }
    return task;
  }

 protected:
  StateId state_id_;
  std::string state_name_;
  wbc::robots::RobotSystem* robot_;
  pinocchio::Data* data_;

  StateLifecycle lifecycle_;

  double start_time_{-1.0};
  double current_time_{0.0};
  double dt_{0.0};

  TaskMap task_map_;
};

}  // namespace wbc

#endif  // CONTROL_ARCHITECTURE_STATE_MACHINE_STATE_MACHINE_HPP_
