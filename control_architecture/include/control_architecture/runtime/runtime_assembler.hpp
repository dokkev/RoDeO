//
// Copyright (c) 2026
//
// RuntimeAssembler: CompiledConfig -> RuntimeConfig.
//

#ifndef CONTROL_ARCHITECTURE_RUNTIME_RUNTIME_ASSEMBLER_HPP_
#define CONTROL_ARCHITECTURE_RUNTIME_RUNTIME_ASSEMBLER_HPP_

#include <pinocchio/multibody/data.hpp>

#include "control_architecture/runtime/compiled_config.hpp"
#include "control_architecture/runtime/runtime_config.hpp"
#include "wbc_core/controller/id-problem-registry.hpp"
#include "wbc_core/robots/robot-system.hpp"

namespace wbc {

class RuntimeAssembler {
 public:
  /// Assemble runtime objects from validated CompiledConfig.
  static RuntimeConfig Assemble(const CompiledConfig& compiled_config,
                                wbc::robots::RobotSystem& robot);
};

void BindRegistry(RuntimeConfig& config, IDProblemRegistry& registry,
                  robots::RobotSystem& robot, pinocchio::Data& data);

}  // namespace wbc

#endif  // CONTROL_ARCHITECTURE_RUNTIME_RUNTIME_ASSEMBLER_HPP_
