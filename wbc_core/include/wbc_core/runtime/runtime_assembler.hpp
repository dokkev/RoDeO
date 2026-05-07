//
// Copyright (c) 2026
//
// RuntimeAssembler: CompiledConfig -> RuntimeConfig + FSM/registry wiring.
//

#ifndef WBC_CORE_RUNTIME_RUNTIME_ASSEMBLER_HPP_
#define WBC_CORE_RUNTIME_RUNTIME_ASSEMBLER_HPP_

#include "wbc_core/architecture/fsm_handler.hpp"
#include "wbc_core/architecture/state_provider.hpp"
#include "wbc_core/controller/wbmc-registry.hpp"
#include "wbc_core/robots/robot-wrapper.hpp"
#include "wbc_core/runtime/runtime_config.hpp"
#include "wbc_core/runtime/compiled_config.hpp"

namespace wbc {

class RuntimeAssembler {
 public:
  /// Assemble runtime objects from validated CompiledConfig.
  static RuntimeConfig Assemble(const CompiledConfig& compiled_config,
                                tsid::robots::RobotWrapper& robot);

  /// Build FSM states from RuntimeConfig and register tasks/contacts in registry.
  static void InitializeFsm(
      RuntimeConfig& config,
      tsid::WBMCRegistry& registry,
      FSMHandler& fsm_handler,
      StateProvider& state_provider,
      tsid::robots::RobotWrapper& robot,
      pinocchio::Data& data);
};

}  // namespace wbc

#endif  // WBC_CORE_RUNTIME_RUNTIME_ASSEMBLER_HPP_
