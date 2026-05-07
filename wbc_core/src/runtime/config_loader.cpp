//
// Copyright (c) 2026
//

#include "wbc_core/runtime/config_loader.hpp"

#include <filesystem>

namespace wbc {

void ConfigLoader::ResolveExternalFiles(YAML::Node& root,
                                        const std::string& base_dir) {
  // task_pool_yaml: load external file and merge task_pool/contact_pool.
  if (root["task_pool_yaml"]) {
    const std::string rel = root["task_pool_yaml"].as<std::string>();
    const auto path =
        (std::filesystem::path(base_dir) / rel).lexically_normal().string();
    YAML::Node ext = YAML::LoadFile(path);

    if (ext["task_pool"] && !root["task_pool"]) {
      root["task_pool"] = ext["task_pool"];
    } else if (ext["task_pool"]) {
      for (const auto& item : ext["task_pool"]) {
        root["task_pool"].push_back(item);
      }
    }

    if (ext["contact_pool"] && !root["contact_pool"]) {
      root["contact_pool"] = ext["contact_pool"];
    } else if (ext["contact_pool"]) {
      for (const auto& item : ext["contact_pool"]) {
        root["contact_pool"].push_back(item);
      }
    }
  }

  // state_machine_yaml: load external file and merge state_machine.
  if (root["state_machine_yaml"]) {
    const std::string rel = root["state_machine_yaml"].as<std::string>();
    const auto path =
        (std::filesystem::path(base_dir) / rel).lexically_normal().string();
    YAML::Node ext = YAML::LoadFile(path);

    if (ext["state_machine"] && !root["state_machine"]) {
      root["state_machine"] = ext["state_machine"];
    }
  }
}

YAML::Node ConfigLoader::LoadFileAndResolve(const std::string& yaml_path) {
  YAML::Node root = YAML::LoadFile(yaml_path);
  const std::string base_dir =
      std::filesystem::path(yaml_path).parent_path().string();
  ResolveExternalFiles(root, base_dir);
  return root;
}

}  // namespace wbc
