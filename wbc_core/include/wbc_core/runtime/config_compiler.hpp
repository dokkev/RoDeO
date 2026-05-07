//
// Copyright (c) 2026
//
// ConfigCompiler: YAML -> CompiledConfig (no TSID object creation).
//

#ifndef WBC_CORE_RUNTIME_CONFIG_COMPILER_HPP_
#define WBC_CORE_RUNTIME_CONFIG_COMPILER_HPP_

#include <yaml-cpp/yaml.h>

#include "wbc_core/runtime/compiled_config.hpp"

namespace wbc {

class ConfigCompiler {
 public:
  /// Compile a YAML root into a pure CompiledConfig.
  /// Throws std::invalid_argument if a state attempts to override
  /// WBMC solver hierarchy semantics.
  static CompiledConfig Compile(const YAML::Node& root);

 private:
  static void parseTaskPool(const YAML::Node& node,
                            CompiledConfig& compiled_config);
  static void parseContactPool(const YAML::Node& node,
                               CompiledConfig& compiled_config);
  static void parseStateMachine(const YAML::Node& node,
                                CompiledConfig& compiled_config);
  static void parseRegularization(const YAML::Node& node,
                                  CompiledConfig& compiled_config);
  static void parseController(const YAML::Node& node,
                              CompiledConfig& compiled_config);
  static void parseGlobalConstraints(const YAML::Node& node,
                                     CompiledConfig& compiled_config);
};

}  // namespace wbc

#endif  // WBC_CORE_RUNTIME_CONFIG_COMPILER_HPP_
