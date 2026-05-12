//
// Copyright (c) 2026
//

#include "control_architecture/runtime/state_machine_assembler.hpp"

#include <stdexcept>
#include <string>
#include <utility>

namespace wbc {

void StateMachineAssembler::Assemble(RuntimeConfig& config,
                                     FSMHandler& fsm_handler,
                                     robots::RobotSystem& robot,
                                     pinocchio::Data& data,
                                     const StateFactory& state_factory) {
  StateContext context;
  context.robot = &robot;
  context.data = &data;

  for (auto& [state_id, state_cfg] : config.states) {
    (void)state_id;

    if (!state_factory.Contains(state_cfg.name)) {
      throw std::invalid_argument(
          "StateMachineAssembler::Assemble: state '" + state_cfg.name +
          "' uses unknown state factory key '" + state_cfg.name + "'");
    }

    auto state = state_factory.Create(state_cfg.name, state_cfg.id,
                                      state_cfg.name, context);

    for (const auto& task_name : state_cfg.task_names) {
      auto task_it = config.task_pool.find(task_name);
      if (task_it != config.task_pool.end()) {
        state->AssignTask(task_name, task_it->second.task);
      }
    }

    state->ConfigureLifecycle(state_cfg.lifecycle);
    state->Configure(state_cfg.params);
    fsm_handler.RegisterState(state_cfg.id, std::move(state));
  }

  if (!fsm_handler.SetStartState(config.start_state_id)) {
    throw std::invalid_argument(
        "StateMachineAssembler::Assemble: start_state_id " +
        std::to_string(config.start_state_id) + " is not registered");
  }
}

}  // namespace wbc
