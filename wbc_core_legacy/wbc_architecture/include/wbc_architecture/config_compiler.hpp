/**
 * @file wbc_core/wbc_architecture/include/wbc_architecture/config_compiler.hpp
 * @brief One-time startup compiler: YAML → RuntimeConfig + FSM initialization.
 *
 * @details
 * ConfigCompiler exists only during system startup. It is never referenced by
 * the real-time control loop. Two-phase usage:
 *
 *   Phase 1 — parsing (ConfigCompiler::Compile):
 *     Read YAML, construct task/contact/constraint objects, build per-state
 *     configs. Returns a live RuntimeConfig and retains StateRecipes internally
 *     for Phase 2.
 *
 *   Phase 2 — FSM construction (ConfigCompiler::InitializeFsm):
 *     Instantiate concrete StateMachine subclasses from the StateFactory,
 *     populate task/contact handles, and register states with FSMHandler.
 *     Frees all recipe memory when done.
 *
 * After InitializeFsm() the ConfigCompiler can be destroyed.
 */
#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <Eigen/Dense>
#include <yaml-cpp/yaml.h>

#include "wbc_formulation/wbc_formulation.hpp"
#include "wbc_fsm/interface/state_machine.hpp"
#include "wbc_util/yaml_parser.hpp"

namespace wbc {

class PinocchioRobotSystem;
class RuntimeConfig;
class FSMHandler;
class StateProvider;
struct StateConfig;

// ---------------------------------------------------------------------------
// StateRecipe — startup-only construction blueprint for one FSM state.
//
// Produced by ConfigCompiler::ParseState, consumed by ConfigCompiler::InitializeFsm,
// then freed. Never used by the real-time control loop.
// ---------------------------------------------------------------------------
struct StateRecipe {
  StateId id{-1};
  std::string type;
  std::string name;

  // Name-keyed task/contact/constraint pointers (non-owning).
  // Used by StateMachine::AssignFromRecipe and SetParameters.
  std::unordered_map<std::string, Task*>       motion_by_name;
  std::unordered_map<std::string, ForceTask*>  force_by_name;
  std::unordered_map<std::string, Contact*>    contact_by_name;
  std::unordered_map<std::string, Constraint*> kin_by_name;

  // Raw YAML params blob forwarded to StateMachine::SetParameters.
  // Holding it here (not in RuntimeConfig) keeps YAML::Node out of the
  // runtime-only RuntimeConfig header.
  YAML::Node params;
};

// ---------------------------------------------------------------------------
// ConfigCompiler — stateful startup object; not used at runtime.
// ---------------------------------------------------------------------------
class ConfigCompiler {
public:
  // Phase 1: parse YAML file, build RuntimeConfig data structures.
  // Returns the compiler (holds recipes); call TakeConfig() to obtain the
  // RuntimeConfig, then call InitializeFsm().
  static std::unique_ptr<ConfigCompiler> Compile(PinocchioRobotSystem* robot,
                                                  const std::string& yaml_path);

  // Transfer RuntimeConfig ownership to the caller.
  // Must be called before InitializeFsm().
  std::unique_ptr<RuntimeConfig> TakeConfig();

  // Phase 2: instantiate FSM states from stored recipes and register them with
  // the given FSMHandler. Frees all recipe memory when done.
  // config must be the RuntimeConfig produced by Compile().
  void InitializeFsm(RuntimeConfig& config, FSMHandler& fsm_handler,
                     StateProvider& state_provider);

private:
  ConfigCompiler() = default;

  void LoadYaml(PinocchioRobotSystem* robot, const std::string& yaml_path);
  void LoadYaml(PinocchioRobotSystem* robot, const std::string& yaml_path,
                const YAML::Node& root);
  void ParseConstraintPool(const YAML::Node& node, PinocchioRobotSystem* robot);
  void ParseGlobalConstraints(const YAML::Node& node, PinocchioRobotSystem* robot);
  void ParseTaskPool(const YAML::Node& node, PinocchioRobotSystem* robot);
  void ParseStateMachine(const YAML::Node& node);
  void ParseState(const YAML::Node& state_node, StateConfig& out_config,
                  StateRecipe& out_recipe);

  static Eigen::VectorXd ParseVectorOrScalar(const YAML::Node& node, int dim,
                                              const std::string& object_name,
                                              const std::string& field_name);

  // Built during Compile(), transferred to caller via TakeConfig().
  std::unique_ptr<RuntimeConfig> runtime_config_;

  // Stored until InitializeFsm() consumes and frees them.
  std::unordered_map<StateId, StateRecipe> state_recipes_;

  // Non-owning robot pointer kept for InitializeFsm's StateMachineConfig.
  PinocchioRobotSystem* robot_{nullptr};
};

} // namespace wbc
