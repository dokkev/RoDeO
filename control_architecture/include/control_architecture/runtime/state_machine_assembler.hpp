//
// Copyright (c) 2026
//
// StateMachineAssembler: RuntimeConfig -> FSMHandler wiring.
//

#ifndef CONTROL_ARCHITECTURE_RUNTIME_STATE_MACHINE_ASSEMBLER_HPP_
#define CONTROL_ARCHITECTURE_RUNTIME_STATE_MACHINE_ASSEMBLER_HPP_

#include <pinocchio/multibody/data.hpp>

#include "control_architecture/runtime/runtime_config.hpp"
#include "control_architecture/state_machine/fsm_handler.hpp"
#include "control_architecture/state_machine/state_factory.hpp"
#include "wbc_core/robots/robot-system.hpp"

namespace wbc {

class StateMachineAssembler {
 public:
  static void Assemble(RuntimeConfig& config,
                       FSMHandler& fsm_handler,
                       robots::RobotSystem& robot,
                       pinocchio::Data& data,
                       const StateFactory& state_factory);
};

}  // namespace wbc

#endif  // CONTROL_ARCHITECTURE_RUNTIME_STATE_MACHINE_ASSEMBLER_HPP_
