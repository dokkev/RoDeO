//
// Copyright (c) 2026
//

#include "control_architecture/runtime/runtime_loader.hpp"

#include <stdexcept>
#include <vector>

#include <pinocchio/multibody/joint/joint-free-flyer.hpp>

#include "control_architecture/runtime/config_compiler.hpp"
#include "control_architecture/runtime/config_loader.hpp"
#include "control_architecture/runtime/config_validator.hpp"
#include "control_architecture/runtime/runtime_assembler.hpp"
#include "wbc_core/utils/ros_path_utils.hpp"

namespace wbc {

LoadedRuntime RuntimeLoader::LoadFromYamlFile(const std::string& yaml_path) {
  LoadedRuntime runtime;
  const std::string resolved_yaml = path::ResolvePackageUri(yaml_path);
  runtime.root = ConfigLoader::LoadFileAndResolve(resolved_yaml);
  runtime.robot = LoadRobotFromYaml(runtime.root);
  runtime.config = MakeRuntimeConfig(runtime.root, *runtime.robot);
  return runtime;
}

std::shared_ptr<robots::RobotSystem> RuntimeLoader::LoadRobotFromYaml(
    const YAML::Node& root) {
  if (!root["robot_model"] || !root["robot_model"]["urdf_path"]) {
    throw std::runtime_error("WBC YAML requires robot_model.urdf_path");
  }

  const std::string urdf_uri =
      root["robot_model"]["urdf_path"].as<std::string>();
  const std::string resolved_urdf = path::ResolvePackageUri(urdf_uri);
  const std::string package_root =
      path::ResolveUrdfPackageRoot(urdf_uri, resolved_urdf);
  const std::vector<std::string> package_dirs{package_root};

  const bool floating_base =
      root["robot_model"]["is_floating_base"].as<bool>(false);
  if (floating_base) {
    return std::make_shared<robots::RobotSystem>(
        resolved_urdf, package_dirs, pinocchio::JointModelFreeFlyer());
  }
  return std::make_shared<robots::RobotSystem>(resolved_urdf, package_dirs);
}

RuntimeConfig RuntimeLoader::MakeRuntimeConfig(const YAML::Node& root,
                                               robots::RobotSystem& robot) {
  auto compiled = ConfigCompiler::Compile(root);
  ConfigValidator::Validate(compiled);
  return RuntimeAssembler::Assemble(compiled, robot);
}

}  // namespace wbc
