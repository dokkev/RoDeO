//
// Copyright (c) 2026
//
// ConfigValidator: CompiledConfig semantic validation.
//

#ifndef WBC_CORE_RUNTIME_CONFIG_VALIDATOR_HPP_
#define WBC_CORE_RUNTIME_CONFIG_VALIDATOR_HPP_

#include "wbc_core/runtime/compiled_config.hpp"

namespace wbc {

class ConfigValidator {
 public:
  /// Validate semantic contracts on a parsed CompiledConfig.
  /// Throws std::invalid_argument on contract violations.
  static void Validate(const CompiledConfig& compiled_config);
};

}  // namespace wbc

#endif  // WBC_CORE_RUNTIME_CONFIG_VALIDATOR_HPP_
