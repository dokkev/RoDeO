//
// Copyright (c) 2026
//
// Small YAML schema helpers for control architecture runtime config parsing.
//

#ifndef CONTROL_ARCHITECTURE_RUNTIME_YAML_PARSER_HPP_
#define CONTROL_ARCHITECTURE_RUNTIME_YAML_PARSER_HPP_

#include <initializer_list>
#include <string>

#include <yaml-cpp/yaml.h>

namespace wbc {

class YamlParser final {
 public:
  YamlParser() = delete;

  static void RequireField(const YAML::Node& node, const std::string& field,
                           const std::string& context);

  static void RejectField(const YAML::Node& node, const std::string& field,
                          const std::string& context,
                          const std::string& hint = "");

  static void RejectFields(const YAML::Node& node,
                           std::initializer_list<std::string> fields,
                           const std::string& context,
                           const std::string& hint = "");

  template <typename T>
  static T RequiredAs(const YAML::Node& node, const std::string& field,
                      const std::string& context) {
    RequireField(node, field, context);
    return node[field].as<T>();
  }
};

}  // namespace wbc

#endif  // CONTROL_ARCHITECTURE_RUNTIME_YAML_PARSER_HPP_
