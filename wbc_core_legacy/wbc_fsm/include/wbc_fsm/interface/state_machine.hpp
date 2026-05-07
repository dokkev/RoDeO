/**
 * @file wbc_core/wbc_fsm/include/wbc_fsm/interface/state_machine.hpp
 * @brief Doxygen documentation for state_machine module.
 */
#pragma once

#include <cmath>
#include <iostream>
#include <unordered_map>
#include <stdexcept>
#include <string>
#include <yaml-cpp/yaml.h> // added for YAML configuration

#include "wbc_robot_system/state_provider.hpp"
#include "wbc_formulation/interface/task.hpp"
#include "wbc_util/constraint_registry.hpp"
#include "wbc_util/task_registry.hpp"
#include "wbc_util/yaml_parser.hpp"

namespace wbc {

class PinocchioRobotSystem;
using StateId = int;

/**
 * @brief Construction context passed from runtime config to state instances.
 *
 * @details
 * This structure removes the need for global singletons during state creation.
 * `StateFactory` creators receive this context and forward it to concrete
 * state constructors.
 */
struct StateMachineConfig {
  PinocchioRobotSystem* robot{nullptr};
  TaskRegistry* task_registry{nullptr};
  ConstraintRegistry* constraint_registry{nullptr};
  StateProvider* state_provider{nullptr};

  /**
   * @brief Raw YAML node forwarded from the state's YAML entry.
   *
   * @note Factory lambdas passed to `WBC_REGISTER_STATE` may read `params`
   *       for constructor-time decisions, but MUST NOT call `SetParameters()`
   *       themselves. `ConfigCompiler::InitializeFsm` calls `SetParameters`
   *       once on every state after construction; a duplicate call from the
   *       factory would silently re-parse and overwrite the same fields.
   */
  YAML::Node params;
};

/**
 * @brief Base class for all runtime FSM states.
 *
 * @details
 * Responsibilities:
 * - define state lifecycle (`FirstVisit`, `OneStep`, `LastVisit`)
 * - parse state-local YAML params via `SetParameters`
 * - hold assigned task/contact/constraint handles resolved by name
 * - expose lookup accessors (`Get*` / `Require*`) for state logic
 */
class StateMachine {
public:
  // Constructor: directly injects registries to support a manager-less design.
  StateMachine(StateId state_id, const std::string& state_name,
               PinocchioRobotSystem* robot,
               TaskRegistry* /*task_reg*/,
               ConstraintRegistry* /*const_reg*/,
               StateProvider* state_provider = nullptr)
      : state_id_(state_id), state_name_(state_name), robot_(robot),
        sp_(state_provider),
        duration_(3.0), wait_time_(1.0), next_state_id_(state_id),
        stay_here_(false),
        start_time_(-1.0), current_time_(0.0) {} // start_time initializes to sentinel

  StateMachine(StateId state_id, const std::string& state_name,
               const StateMachineConfig& context)
      : StateMachine(state_id, state_name, context.robot, context.task_registry,
                     context.constraint_registry, context.state_provider) {}

  virtual ~StateMachine() = default;

  /**
   * @name Lifecycle
   * @brief Called by `FSMHandler` in deterministic order.
   */
  ///@{
  virtual void FirstVisit() = 0;
  virtual void OneStep() = 0;
  virtual void LastVisit() = 0;

  /**
   * @brief Accept external desired values from non-RT callers (e.g. ROS subscribers).
   *
   * @details Default implementation is a no-op. Teleop states override this to
   *          update their internal hold references. RT safety is the caller's
   *          responsibility (RealtimeBuffer to be added later).
   */
  virtual void SetExternalInput(const TaskInput& /*input*/) {}
  virtual bool EndOfState() {
    if (stay_here_) {
      return false;
    }
    return current_time_ >= (duration_ + wait_time_);
  }
  virtual StateId GetNextState() { return next_state_id_; }
  ///@}

  /**
   * @brief Parse common and state-local parameters.
   *
   * @details
   * Common keys:
   * - `duration`
   * - `wait_time`
   * - `next_state_id`
   * - `stay_here` (legacy alias `b_stay_here`)
   */
  virtual void SetParameters(const YAML::Node& node) {
    SetCommonParameters(node);
  }

  /**
   * @brief Mark state-entry timestamp and synchronize provider FSM metadata.
   */
  void EnterState(double current_global_time) {
    start_time_ = current_global_time;
    current_time_ = 0.0;
    if (sp_ != nullptr) {
      sp_->current_time_ = current_global_time;
      sp_->prev_state_ = sp_->state_;
      sp_->state_ = state_id_;
    }
  }

  /**
   * @brief Update elapsed time and provider flags for current tick.
   */
  void UpdateStateTime(double current_global_time) {
    if (start_time_ < 0.0) {
        // Guard: start_time not set (OneStep called before FirstVisit).
        start_time_ = current_global_time;
    }
    current_time_ = current_global_time - start_time_;
    if (sp_ != nullptr) {
      sp_->current_time_ = current_global_time;
      sp_->state_ = state_id_;
    }
  }

  // Getters
  StateId id() const { return state_id_; }
  const std::string& name() const { return state_name_; }
  double elapsed_time() const { return current_time_; }
  StateProvider* state_provider() const { return sp_; }

  /**
   * @name Assignment Interface
   * @brief Register handles resolved by runtime config.
   */
  ///@{
  void AssignTask(const std::string& task_name, Task* task) {
    if (task_name.empty() || task == nullptr) {
      return;
    }
    assigned_tasks_[task_name] = task;
  }

  void AssignForceTask(const std::string& task_name, ForceTask* task) {
    if (task_name.empty() || task == nullptr) {
      return;
    }
    assigned_force_tasks_[task_name] = task;
  }

  void AssignContact(const std::string& contact_name, Contact* contact) {
    if (contact_name.empty() || contact == nullptr) {
      return;
    }
    assigned_contacts_[contact_name] = contact;
  }

  void AssignConstraint(const std::string& constraint_name,
                        Constraint* constraint) {
    if (constraint_name.empty() || constraint == nullptr) {
      return;
    }
    assigned_constraints_[constraint_name] = constraint;
  }

  /**
   * @brief Assign all resolved entities from a runtime recipe-like object.
   *
   * @details
   * `recipe` is expected to expose:
   * - `motion_by_name`
   * - `force_by_name`
   * - `contact_by_name`
   * - `kin_by_name`
   *
   * This keeps FSM assembly code compact while avoiding direct dependency on
   * architecture-specific recipe types.
   */
  template <typename RecipeType>
  void AssignFromRecipe(const RecipeType& recipe) {
    for (const auto& [task_name, task] : recipe.motion_by_name) {
      AssignTask(task_name, task);
    }
    for (const auto& [task_name, force_task] : recipe.force_by_name) {
      AssignForceTask(task_name, force_task);
    }
    for (const auto& [contact_name, contact] : recipe.contact_by_name) {
      AssignContact(contact_name, contact);
    }
    for (const auto& [constraint_name, constraint] : recipe.kin_by_name) {
      AssignConstraint(constraint_name, constraint);
    }
  }
  ///@}

  /**
   * @name Access Interface
   * @brief Optional and no-throw task/contact lookup utilities.
   *
   * @details
   * `Require*` APIs intentionally do not throw to avoid exception paths in RT
   * call chains (`FSMHandler::UpdateImpl -> State::OneStep`).
   *
   * Recommended pattern:
   * 1) resolve and validate pointers in `FirstVisit()`
   * 2) cache pointers in state members
   * 3) in `OneStep()` use cached pointers only
   */
  ///@{
  ForceTask* GetForceTask(const std::string& task_name) const {
    auto it = assigned_force_tasks_.find(task_name);
    if (it == assigned_force_tasks_.end()) {
      return nullptr;
    }
    return it->second;
  }

  Contact* GetContact(const std::string& contact_name) const {
    auto it = assigned_contacts_.find(contact_name);
    if (it == assigned_contacts_.end()) {
      return nullptr;
    }
    return it->second;
  }

  Constraint* GetConstraint(const std::string& constraint_name) const {
    auto it = assigned_constraints_.find(constraint_name);
    if (it == assigned_constraints_.end()) {
      return nullptr;
    }
    return it->second;
  }

  // RT-safe lookup; returns nullptr if task is absent or wrong type.
  template <typename TaskType>
  [[nodiscard]]
  TaskType* GetMotionTask(const std::string& task_name) const {
    return dynamic_cast<TaskType*>(FindAssigned(assigned_tasks_, task_name));
  }

  // Config-time binding; throws if task is absent or wrong type.
  template <typename TaskType>
  void SetMotionTask(const std::string& task_name, TaskType*& out_task) const {
    out_task = GetMotionTask<TaskType>(task_name);
    if (out_task == nullptr) {
      throw std::runtime_error(
          "[" + state_name_ + "] Requires task '" + task_name + "'.");
    }
  }

  void SetForceTask(const std::string& task_name, ForceTask*& out_task) const {
    out_task = GetForceTask(task_name);
    if (out_task == nullptr) throw std::runtime_error(
        "[" + state_name_ + "] Requires force task '" + task_name + "'.");
  }

  void SetContact(const std::string& contact_name, Contact*& out_contact) const {
    out_contact = GetContact(contact_name);
    if (out_contact == nullptr) throw std::runtime_error(
        "[" + state_name_ + "] Requires contact '" + contact_name + "'.");
  }

  [[nodiscard]]
  Constraint* RequireConstraint(const std::string& constraint_name) const {
    auto* c = GetConstraint(constraint_name);
    if (c == nullptr) throw std::runtime_error(
        "[" + state_name_ + "] Requires constraint '" + constraint_name + "'.");
    return c;
  }
  ///@}

protected:
  // State identity
  StateId state_id_;
  std::string state_name_;

  // Robot and task handles (used directly by derived classes)
  PinocchioRobotSystem* robot_;
  StateProvider* sp_;
  std::unordered_map<std::string, Task*> assigned_tasks_;
  std::unordered_map<std::string, ForceTask*> assigned_force_tasks_;
  std::unordered_map<std::string, Contact*> assigned_contacts_;
  std::unordered_map<std::string, Constraint*> assigned_constraints_;

  // Common state parameters
  double duration_;
  double wait_time_;
  StateId next_state_id_;
  bool stay_here_;

  // Time tracking
  double start_time_;   // global time at state entry
  double current_time_; // elapsed time since state entry (used for trajectory interpolation)

  // Parses common state parameters: duration, wait_time, next_state_id, stay_here.
  void SetCommonParameters(const YAML::Node& node) {
    const YAML::Node params = param::ResolveParamsNode(node);
    if (!params) { return; }
    const bool has_next_state_id = static_cast<bool>(params["next_state_id"]);

    if (params["duration"])    { duration_  = params["duration"].as<double>(); }
    if (params["wait_time"])   { wait_time_ = params["wait_time"].as<double>(); }
    if (has_next_state_id)     { next_state_id_ = params["next_state_id"].as<StateId>(); }

    if (params["stay_here"])        { stay_here_ = params["stay_here"].as<bool>(); }
    else if (params["b_stay_here"]) { stay_here_ = params["b_stay_here"].as<bool>(); }

    if (stay_here_ && has_next_state_id) {
      std::cerr << "[StateMachine] State '" << state_name_
                << "' has stay_here=true and next_state_id set. "
                   "next_state_id will be ignored while stay_here is true.\n";
    }
    if (duration_ < 0.0 || wait_time_ < 0.0) {
      throw std::runtime_error(
          "[StateMachine] duration and wait_time must be non-negative for state '"
          + state_name_ + "'.");
    }
  }

  // Returns a double vector param from YAML; zero-size VectorXd if key absent.
  static Eigen::VectorXd ParseVectorParam(const YAML::Node& node,
                                          const std::string& key) {
    const YAML::Node params = param::ResolveParamsNode(node);
    if (!params || !params[key]) { return {}; }
    const auto vec = params[key].as<std::vector<double>>();
    return Eigen::Map<const Eigen::VectorXd>(vec.data(), vec.size());
  }

  template <typename MapT>
  static typename MapT::mapped_type FindAssigned(const MapT& assigned,
                                                 const std::string& key) {
    const auto it = assigned.find(key);
    if (it == assigned.end()) {
      return nullptr;
    }
    return it->second;
  }

};

} // namespace wbc
