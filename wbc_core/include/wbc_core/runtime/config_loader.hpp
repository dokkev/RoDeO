//
// Copyright (c) 2026
//
// ConfigLoader: file loading + external YAML resolution only.
//

#ifndef WBC_CORE_RUNTIME_CONFIG_LOADER_HPP_
#define WBC_CORE_RUNTIME_CONFIG_LOADER_HPP_

#include <string>

#include <yaml-cpp/yaml.h>

namespace wbc {

class ConfigLoader {
 public:
  /// Load a root YAML file, resolve external references, and return merged root.
  static YAML::Node LoadFileAndResolve(const std::string& yaml_path);

  /// Resolve external references in-place relative to base_dir.
  static void ResolveExternalFiles(YAML::Node& root,
                                   const std::string& base_dir);
};

}  // namespace wbc

#endif  // WBC_CORE_RUNTIME_CONFIG_LOADER_HPP_
