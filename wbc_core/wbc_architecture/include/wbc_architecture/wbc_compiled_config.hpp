#pragma once

#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <yaml-cpp/yaml.h>

#include "wbc_formulation/wbc_formulation.hpp"
#include "wbc_util/constraint_registry.hpp"
#include "wbc_util/task_registry.hpp"

namespace wbc {

class PinocchioRobotSystem;
class Constraint;

struct GlobalParams {
  double w_qddot{1.0e-6};
  double w_rf{1.0e-4};
  double w_xc_ddot{1.0e-3};
  double w_force_rate_of_change{1.0e-3};
  // Legacy compatibility fallback from existing YAMLs.
  double w_tau{1.0e-3};
};

struct CompiledState {
  int id{-1};
  std::string name;

  std::vector<Task*> motion;
  std::vector<const TaskConfig*> motion_cfg;

  std::vector<Contact*> contacts;

  std::vector<ForceTask*> forces;
  std::vector<const ForceTaskConfig*> force_cfg;

  std::vector<Constraint*> kin;
  std::unordered_map<std::string, Task*> motion_by_name;
  std::unordered_map<std::string, ForceTask*> force_by_name;
  std::unordered_map<std::string, Contact*> contact_by_name;
  std::unordered_map<std::string, Constraint*> kin_by_name;

  // Frequently used handles cached for O(1) runtime input mapping.
  Task* ee_pos{nullptr};
  Task* ee_ori{nullptr};
  Task* joint{nullptr};
  ForceTask* ee_force{nullptr};

  // Override ownership (lifetime-safe pointers exposed via motion_cfg/force_cfg).
  std::vector<std::unique_ptr<TaskConfig>> owned_task_cfg;
  std::vector<std::unique_ptr<ForceTaskConfig>> owned_force_cfg;

  YAML::Node params;
};

class WbcConfigCompiler {
public:
  static std::unique_ptr<WbcConfigCompiler> Compile(PinocchioRobotSystem* robot,
                                                    const std::string& yaml_path);

  const GlobalParams& Globals() const { return globals_; }
  const std::unordered_map<int, CompiledState>& States() const { return states_; }
  const CompiledState& State(int state_id) const;
  const CompiledState* FindState(int state_id) const;

  // Build runtime formulation by shallow-copying non-owning pointers.
  // Lifetime contract: registries own all task/constraint objects for the
  // lifetime of this compiler instance and any runtime using these pointers.
  void BuildFormulation(int state_id, WbcFormulation& out) const;

  TaskRegistry* TaskRegistryPtr() const { return task_registry_.get(); }
  ConstraintRegistry* ConstraintRegistryPtr() const {
    return constraint_registry_.get();
  }

  const std::vector<Constraint*>& GlobalConstraints() const {
    return global_constraints_;
  }

  std::vector<bool> BuildActuationMask() const;
  int MaxContactDim() const;
  int StartStateId() const;
  std::string BaseFrameName() const;
  std::string EndEffectorFrameName() const;
  void ValidateRobotDimensions() const;

private:
  explicit WbcConfigCompiler(PinocchioRobotSystem* robot);

  void LoadYaml(const std::string& yaml_path);
  void ParseSolverParams(const YAML::Node& solver_params);
  int ResolveConfiguredStartStateId() const;
  void ParseConstraintPool(const YAML::Node& node);
  void ParseGlobalConstraints(const YAML::Node& node);
  void ParseTaskPool(const YAML::Node& node);
  void CompileStateMachine(const YAML::Node& node);
  void CompileState(const YAML::Node& state_node, CompiledState& out);

  static Eigen::VectorXd ParseVectorOrScalar(const YAML::Node& node, int dim,
                                             const std::string& object_name,
                                             const std::string& field_name);
  static YAML::Node ResolveParamsNode(const YAML::Node& node);

  PinocchioRobotSystem* robot_{nullptr};
  YAML::Node config_node_;

  std::unique_ptr<TaskRegistry> task_registry_;
  std::unique_ptr<ConstraintRegistry> constraint_registry_;
  GlobalParams globals_;
  int solver_start_state_id_{-1};
  int solver_max_contact_dim_{-1};
  std::optional<int> expected_num_qdot_;
  std::optional<int> expected_num_active_dof_;
  std::optional<int> expected_num_float_dof_;
  std::unordered_map<int, CompiledState> states_;

  std::vector<Constraint*> global_constraints_;
};

// Apply state-local task/force overrides once when a state becomes active.
inline void ApplyStateOverrides(const CompiledState& state, WbcType wbc_type) {
  for (std::size_t i = 0; i < state.motion.size(); ++i) {
    Task* task = state.motion[i];
    if (task == nullptr) {
      continue;
    }
    if (i >= state.motion_cfg.size()) {
      continue;
    }
    const TaskConfig* cfg = state.motion_cfg[i];
    if (cfg != nullptr) {
      task->SetParameters(*cfg, wbc_type);
    }
  }

  for (std::size_t i = 0; i < state.forces.size(); ++i) {
    ForceTask* force_task = state.forces[i];
    if (force_task == nullptr) {
      continue;
    }
    if (i >= state.force_cfg.size()) {
      continue;
    }
    const ForceTaskConfig* cfg = state.force_cfg[i];
    if (cfg != nullptr) {
      force_task->SetParameters(*cfg);
    }
  }
}

} // namespace wbc
