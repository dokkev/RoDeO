//
// Copyright (c) 2026
//
// ConfigValidator: CompiledConfig semantic validation.
//

#ifndef CONTROL_ARCHITECTURE_RUNTIME_CONFIG_VALIDATOR_HPP_
#define CONTROL_ARCHITECTURE_RUNTIME_CONFIG_VALIDATOR_HPP_

#include "control_architecture/runtime/compiled_config.hpp"

namespace wbc {

class ConfigValidator {
 public:
  /// Validate semantic contracts on a parsed CompiledConfig.
  /// Throws std::invalid_argument on contract violations.
  static void Validate(const CompiledConfig& compiled_config);
};

}  // namespace wbc

#endif  // CONTROL_ARCHITECTURE_RUNTIME_CONFIG_VALIDATOR_HPP_
