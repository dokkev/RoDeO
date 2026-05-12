//
// Copyright (c) 2026
//

#include "control_architecture/runtime/yaml_parser.hpp"

#include <stdexcept>

namespace wbc {

void YamlParser::RequireField(const YAML::Node& node, const std::string& field,
                              const std::string& context) {
  if (!node[field]) {
    throw std::invalid_argument(context + ": requires field '" + field + "'");
  }
}

void YamlParser::RejectField(const YAML::Node& node, const std::string& field,
                             const std::string& context,
                             const std::string& hint) {
  if (!node[field]) {
    return;
  }

  std::string message = context + ": field '" + field + "' is not allowed";
  if (!hint.empty()) {
    message += ". " + hint;
  }
  throw std::invalid_argument(message);
}

void YamlParser::RejectFields(const YAML::Node& node,
                              std::initializer_list<std::string> fields,
                              const std::string& context,
                              const std::string& hint) {
  for (const auto& field : fields) {
    RejectField(node, field, context, hint);
  }
}

}  // namespace wbc
