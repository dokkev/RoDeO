//
// Copyright (c) 2026
//
// ControlArchitecture: owns the WBMC runtime, FSM handler, and state provider.
//

#ifndef WBC_CORE_ARCHITECTURE_CONTROL_ARCHITECTURE_HPP_
#define WBC_CORE_ARCHITECTURE_CONTROL_ARCHITECTURE_HPP_

#include <chrono>
#include <memory>
#include <string>

#include <wbc_core/robots/robot-wrapper.hpp>

#include "wbc_core/adapters/command-adapter.hpp"
#include "wbc_core/architecture/fsm_handler.hpp"
#include "wbc_core/architecture/state_provider.hpp"
#include "wbc_core/controller/wbmc-registry.hpp"
#include "wbc_core/controller/wbmc.hpp"
#include "wbc_core/runtime/runtime_config.hpp"

namespace wbc {

using WBMC = tsid::WBMC;
using HardTorqueLimitMode = tsid::HardTorqueLimitMode;
using RobotCommand = LowLevelCommand;
using CommandMode = CommandOutputMode;

/// Timing statistics for each phase of the control loop.
struct ArchTimingStats {
  double robot_model_us{0};
  double kinematics_us{0};
  double dynamics_us{0};
  double find_config_us{0};
  double make_torque_us{0};
  double feedback_us{0};
};

class ControlArchitecture {
 public:
  /// Construct from YAML config and robot URDF.
  /// Reads robot_model.is_floating_base from the YAML (default: false).
  ControlArchitecture(const std::string& yaml_path,
                      const std::string& urdf_path,
                      const std::vector<std::string>& package_dirs);

  /// Construct from YAML node and existing RobotWrapper.
  ControlArchitecture(const YAML::Node& yaml_config,
                      std::shared_ptr<tsid::robots::RobotWrapper> robot);

  /// Initialize: parse config, create formulation, build FSM.
  void Initialize();

  /// Update robot state and run one control tick.
  void Update(const RobotJointState& state, double t, double dt);

  /// Get the last computed command.
  const RobotCommand& command() const { return cmd_; }

  /// Access runtime components.
  tsid::robots::RobotWrapper* robot() const { return robot_.get(); }
  tsid::WBMC* solver() const { return solver_.get(); }
  tsid::WBMCRegistry* registry() const { return registry_.get(); }
  FSMHandler* fsmHandler() { return &fsm_handler_; }
  const FSMHandler* fsmHandler() const { return &fsm_handler_; }
  StateProvider* stateProvider() { return &state_provider_; }
  const StateProvider* stateProvider() const { return &state_provider_; }
  RuntimeConfig* config() { return &config_; }
  const RuntimeConfig* config() const { return &config_; }

  /// Compatibility shims for legacy call sites.
  const RobotCommand& GetCommand() const { return command(); }
  tsid::robots::RobotWrapper* GetRobot() const { return robot(); }
  tsid::WBMC* GetFormulation() const { return solver(); }
  tsid::WBMC* GetSolver() const { return solver(); }
  tsid::WBMCRegistry* GetRegistry() const { return registry(); }
  FSMHandler* GetFsmHandler() { return fsmHandler(); }
  StateProvider* GetStateProvider() { return stateProvider(); }
  RuntimeConfig* GetConfig() { return config(); }
  void setTimingEnabled(bool enabled);
  bool timingEnabled() const { return timing_enabled_; }
  const ArchTimingStats& timingStats() const { return timing_stats_; }
  void setCommandOutputMode(CommandMode mode) {
    command_adapter_.setOutputMode(mode);
  }
  CommandMode commandOutputMode() const {
    return command_adapter_.outputMode();
  }

  /// State transition request by ID (thread-safe).
  bool RequestState(StateId id) { return fsm_handler_.RequestState(id); }

  /// State transition request by name (thread-safe).
  bool RequestState(const std::string& name) {
    auto id = fsm_handler_.FindStateIdByName(name);
    if (!id) return false;
    return fsm_handler_.RequestState(*id);
  }

  /// Current state ID.
  StateId GetCurrentStateId() const {
    return fsm_handler_.GetCurrentStateId();
  }

  /// External input to active state.
  void SetExternalInput(const TaskInput& input) {
    auto* state = fsm_handler_.GetCurrentState();
    if (state) state->SetExternalInput(input);
  }

 private:
  void Step();

  // Robot
  std::shared_ptr<tsid::robots::RobotWrapper> robot_;

  // Config
  std::string yaml_path_;  // non-empty when constructed from file
  YAML::Node yaml_config_;
  RuntimeConfig config_;
  bool initialized_{false};

  // Final-form WBMC runtime
  std::unique_ptr<tsid::WBMCRegistry> registry_;
  std::unique_ptr<tsid::WBMC> solver_;

  // FSM
  FSMHandler fsm_handler_;
  StateProvider state_provider_;

  // Output
  CommandAdapter command_adapter_;
  RobotCommand cmd_;

  // Time
  double current_time_{0.0};
  double dt_{0.001};
  bool timing_enabled_{false};
  ArchTimingStats timing_stats_;
};

}  // namespace wbc

#endif  // WBC_CORE_ARCHITECTURE_CONTROL_ARCHITECTURE_HPP_
