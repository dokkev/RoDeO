/**
 * @file wbc_core/wbc_architecture/include/wbc_architecture/runtime_config.hpp
 * @brief Runtime configuration store for the WBC control loop.
 *
 * @details
 * RuntimeConfig holds the fully-constructed, runtime-ready objects produced by
 * ConfigCompiler at startup:
 * - task and constraint registries (ownership)
 * - per-state configs with pre-resolved task/contact pointers
 * - global solver settings and robot dimension hints
 *
 * RuntimeConfig is NOT responsible for YAML parsing or FSM construction —
 * that is ConfigCompiler's job. After startup, only RuntimeConfig is referenced
 * by the control loop (ControlArchitecture::Step).
 */
#pragma once

#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "wbc_formulation/wbc_formulation.hpp"
#include "wbc_fsm/interface/state_machine.hpp"
#include "wbc_util/constraint_registry.hpp"
#include "wbc_util/task_registry.hpp"
#include "wbc_util/yaml_parser.hpp"

namespace wbc {

class PinocchioRobotSystem;
class Constraint;
class ConfigCompiler;

/// Per-constraint-type soft/hard configuration parsed from YAML.
struct SoftConstraintConfig {
  bool is_soft{false};
  double weight{1e5};
};

// ---------------------------------------------------------------------------
// StateConfig — runtime-ready per-state descriptor.
//
// Contains non-owning pointers into the task/constraint registries, organized
// for direct use in the control loop (BuildFormulation, ApplyStateOverrides).
// ---------------------------------------------------------------------------
struct StateConfig {
  StateConfig() = default;
  StateConfig(StateConfig&&) = default;
  StateConfig& operator=(StateConfig&&) = default;
  StateConfig(const StateConfig&) = delete;
  StateConfig& operator=(const StateConfig&) = delete;

  StateId id{-1};
  // Factory registration key (WBC_REGISTER_STATE key).
  // If not specified in YAML, defaults to `name` for backward compatibility.
  std::string type;
  // Runtime instance name (state label exposed to FSM APIs/logging).
  std::string name;

  std::vector<Task*>              motion;
  std::vector<const TaskConfig*>  motion_cfg;

  std::vector<Contact*>           contacts;

  std::vector<ForceTask*>         forces;
  std::vector<const ForceTaskConfig*> force_cfg;

  std::vector<Constraint*>        kin;

  // Frequently used handles cached for O(1) runtime input mapping.
  Task*      ee_pos{nullptr};
  Task*      ee_ori{nullptr};
  Task*      com{nullptr};
  Task*      joint{nullptr};
  ForceTask* ee_force{nullptr};

  // Per-state weight ramp duration (seconds). Negative means use global default.
  double weight_ramp_duration{-1.0};

  // Override ownership (lifetime-safe pointers exposed via motion_cfg/force_cfg).
  std::vector<std::unique_ptr<TaskConfig>>      owned_task_cfg;
  std::vector<std::unique_ptr<ForceTaskConfig>> owned_force_cfg;
};

// ---------------------------------------------------------------------------
// RuntimeConfig — data store used by the real-time control loop.
//
// ConfigCompiler is declared as a friend so it can populate private members
// during startup without exposing a mutable public API to runtime callers.
// ---------------------------------------------------------------------------
class RuntimeConfig {
public:
  friend class ConfigCompiler;

  const param::RegularizationParams& Regularization() const { return regularization_; }
  /// @deprecated Use Regularization() instead.
  const param::RegularizationParams& SolverParams() const { return regularization_; }
  const std::unordered_map<StateId, StateConfig>& States() const { return states_; }

  const StateConfig& State(StateId state_id) const;
  const StateConfig* FindState(StateId state_id) const;

  /**
   * @brief Apply per-state task/force gain and weight overrides.
   *
   * @details
   * Called once on state entry (not every tick). Writes task gains using
   * pre-parsed config pointers, so no YAML parsing occurs at runtime.
   */
  void ApplyStateOverrides(const StateConfig& state, WbcType wbc_type) const;

  /**
   * @brief Build a WbcFormulation for the given state.
   *
   * @details
   * Shallow-copies non-owning pointers into `out`. Registry ownership remains
   * here. With Reserve() called at startup, no heap allocation occurs on
   * steady-state ticks.
   */
  void BuildFormulation(StateId state_id, WbcFormulation& out) const;
  void BuildFormulation(const StateConfig& state, WbcFormulation& out) const;

  TaskRegistry*       taskRegistry()       const { return task_registry_.get(); }
  ConstraintRegistry* constraintRegistry() const { return constraint_registry_.get(); }

  const std::vector<Constraint*>& GlobalConstraints() const {
    return global_constraints_;
  }

  const SoftConstraintConfig& SoftConfig(const std::string& type) const {
    static const SoftConstraintConfig kDefault{};
    auto it = soft_constraint_cfg_.find(type);
    return (it != soft_constraint_cfg_.end()) ? it->second : kDefault;
  }

  std::vector<bool> BuildActuationMask() const;
  int               MaxContactDim()      const;
  StateId           StartStateId()       const;
  const std::string& BaseFrameName()     const;
  const std::string& EndEffectorFrameName() const;
  void ValidateRobotDimensions()         const;

  const std::unordered_map<Task*, TaskConfig>& DefaultMotionTaskConfigs() const {
    return default_motion_task_cfg_;
  }

private:
  RuntimeConfig() = default;

  int num_qdot_{-1};
  int num_active_dof_{-1};
  int num_float_dof_{-1};

  std::unique_ptr<TaskRegistry>       task_registry_;
  std::unique_ptr<ConstraintRegistry> constraint_registry_;
  param::RobotModelParams             robot_model_hints_;
  param::RegularizationParams          regularization_;
  int                                  max_contact_dim_{-1};
  std::optional<StateId>              configured_start_state_id_;
  StateId                             first_state_id_{-1};
  std::unordered_map<StateId, StateConfig> states_;
  std::unordered_map<Task*,      TaskConfig>      default_motion_task_cfg_;
  std::unordered_map<ForceTask*, ForceTaskConfig> default_force_task_cfg_;

  std::vector<Constraint*> global_constraints_;
  std::unordered_map<std::string, SoftConstraintConfig> soft_constraint_cfg_;
};

} // namespace wbc
