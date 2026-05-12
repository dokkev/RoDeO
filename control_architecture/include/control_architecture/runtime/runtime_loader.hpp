//
// Copyright (c) 2026
//
// RuntimeLoader: YAML file -> RobotSystem + RuntimeConfig.
//

#ifndef CONTROL_ARCHITECTURE_RUNTIME_RUNTIME_LOADER_HPP_
#define CONTROL_ARCHITECTURE_RUNTIME_RUNTIME_LOADER_HPP_

#include <memory>
#include <string>

#include <yaml-cpp/yaml.h>

#include "control_architecture/runtime/runtime_config.hpp"
#include "wbc_core/robots/robot-system.hpp"

namespace wbc {

struct LoadedRuntime {
  YAML::Node root;
  std::shared_ptr<robots::RobotSystem> robot;
  RuntimeConfig config;
};

class RuntimeLoader {
 public:
  static LoadedRuntime LoadFromYamlFile(const std::string& yaml_path);

  static std::shared_ptr<robots::RobotSystem> LoadRobotFromYaml(
      const YAML::Node& root);

  static RuntimeConfig MakeRuntimeConfig(const YAML::Node& root,
                                         robots::RobotSystem& robot);
};

}  // namespace wbc

#endif  // CONTROL_ARCHITECTURE_RUNTIME_RUNTIME_LOADER_HPP_
