//
// Copyright (c) 2026
//
// ControlArchitecture: owns the IDHQP runtime and FSM handler.
//

#ifndef CONTROL_ARCHITECTURE_CONTROL_ARCHITECTURE_HPP_
#define CONTROL_ARCHITECTURE_CONTROL_ARCHITECTURE_HPP_

#include <chrono>
#include <memory>
#include <string>
#include <utility>

#include <wbc_core/robots/robot-system.hpp>

#include "control_architecture/state_machine/fsm_handler.hpp"
#include "control_architecture/state_machine/state_factory.hpp"
#include "wbc_core/controller/id-problem-registry.hpp"
#include "wbc_core/controller/id-hqp.hpp"
#include "wbc_core/robots/robot-command.hpp"
#include "wbc_core/robots/robot-logger.hpp"
#include "control_architecture/runtime/runtime_config.hpp"

namespace wbc {

/// Timing statistics for each phase of the control loop.
struct ArchTimingStats {
  double model_us{0};
  double fsm_us{0};
  double problem_us{0};
  double solve_us{0};
  double output_us{0};
};

class ControlArchitecture {
 public:
  /// Construct from assembled runtime config and an existing robot model.
  ControlArchitecture(RuntimeConfig config,
                      std::shared_ptr<wbc::robots::RobotSystem> robot);

  /// Initialize runtime objects and build the FSM.
  void Initialize();

  /// Update RobotSystem from estimator/controller state and run one tick.
  void Update(const robots::RobotState& state, double dt);

  /// Get the last computed command.
  const robots::RobotCommand& command() const { return cmd_; }

  /// Get the last solver-to-command trace.
  const robots::RobotLogger& logger() const { return logger_; }

  /// Access runtime components.
  wbc::robots::RobotSystem* robot() const { return robot_.get(); }
  wbc::IDHQP* solver() const { return solver_.get(); }
  wbc::IDProblemRegistry* registry() const { return registry_.get(); }
  FSMHandler* fsmHandler() { return &fsm_handler_; }
  const FSMHandler* fsmHandler() const { return &fsm_handler_; }
  RuntimeConfig* config() { return &config_; }
  const RuntimeConfig* config() const { return &config_; }
  StateFactory* stateFactory() { return &state_factory_; }
  const StateFactory* stateFactory() const { return &state_factory_; }
  void RegisterState(const std::string& key, StateFactory::Creator creator) {
    state_factory_.Register(key, std::move(creator));
  }

  void setTimingEnabled(bool enabled);
  bool timingEnabled() const { return timing_enabled_; }
  const ArchTimingStats& timingStats() const { return timing_stats_; }

  /// State transition request by ID (thread-safe).
  bool RequestState(StateId id) { return fsm_handler_.RequestState(id); }

  /// State transition request by name (thread-safe).
  bool RequestState(const std::string& name) {
    auto id = fsm_handler_.FindStateIdByName(name);
    if (!id) return false;
    return fsm_handler_.RequestState(*id);
  }

  /// Current state ID.
  StateId GetCurrentStateId() const { return fsm_handler_.GetCurrentStateId(); }

 private:
  void Step(double dt);
  void UpdateModelTerms();
  void UpdateStateMachine(double current_time, double dt);
  IDProblem BuildProblem(double current_time);
  const IDSolution& SolveProblem(const IDProblem& problem, double dt);
  void ApplySolution(const IDSolution& solution);
  void InitializeCommandFromRobotState();
  const StateConfig* ActiveStateConfig() const;

  // Robot
  std::shared_ptr<wbc::robots::RobotSystem> robot_;

  // Config
  RuntimeConfig config_;
  bool initialized_{false};

  // Final-form IDHQP runtime
  std::unique_ptr<wbc::IDProblemRegistry> registry_;
  std::unique_ptr<wbc::IDHQP> solver_;

  // FSM
  FSMHandler fsm_handler_;
  StateFactory state_factory_;

  // Output
  robots::RobotCommand cmd_;
  robots::RobotLogger logger_;
  bool command_initialized_{false};

  bool timing_enabled_{false};
  ArchTimingStats timing_stats_;
};

}  // namespace wbc

#endif  // CONTROL_ARCHITECTURE_CONTROL_ARCHITECTURE_HPP_
