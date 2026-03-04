/**
 * @file wbc_core/wbc_util/src/yaml_parser.cpp
 * @brief Doxygen documentation for yaml_parser module.
 */
#include "wbc_util/yaml_parser.hpp"

#include <stdexcept>

#include "wbc_util/ros_path_utils.hpp"

namespace wbc::param {
namespace {

YAML::Node ResolveRegularizationNode(const YAML::Node& root,
                                      const std::string& main_yaml_path) {
  // Primary: inline "regularization" section.
  if (root["regularization"]) {
    return root["regularization"];
  }
  // Legacy fallback: "solver_params" inline or external file.
  if (root["solver_params_yaml"]) {
    const std::string raw_path = root["solver_params_yaml"].as<std::string>();
    if (raw_path.empty()) {
      throw std::runtime_error("[YamlParser] solver_params_yaml is empty.");
    }
    const std::string resolved_path =
        path::ResolveRelativePath(raw_path, main_yaml_path);
    const YAML::Node external = LoadYamlFile(resolved_path);
    if (external["regularization"]) return external["regularization"];
    return external["solver_params"] ? external["solver_params"] : external;
  }
  return root["solver_params"];
}

std::optional<int> ParseOptionalNonNegativeInt(const YAML::Node& node,
                                               const char* key) {
  if (!node || !node[key]) {
    return std::nullopt;
  }
  const int value = node[key].as<int>();
  if (value < 0) {
    throw std::runtime_error(std::string("[YamlParser] '") + key +
                             "' must be non-negative.");
  }
  return value;
}

} // namespace

YAML::Node LoadYamlFile(const std::string& yaml_path) {
  if (yaml_path.empty()) {
    throw std::runtime_error("[YamlParser] YAML path is empty.");
  }

  try {
    return YAML::LoadFile(yaml_path);
  } catch (const std::exception& e) {
    throw std::runtime_error("[YamlParser] Failed to load YAML '" +
                             yaml_path + "': " + e.what());
  }
}

WbcDefinition ParseDefinition(const std::string& yaml_path) {
  const std::string resolved_yaml_path = path::ResolvePackageUri(yaml_path);
  const YAML::Node root = LoadYamlFile(resolved_yaml_path);

  WbcDefinition definition;
  definition.yaml_path = resolved_yaml_path;
  definition.root = root;
  definition.robot_model = ParseRequiredRobotModel(root);
  return definition;
}

RobotModelParams ParseRequiredRobotModel(const YAML::Node& root) {
  const YAML::Node robot_model = root["robot_model"];
  if (!robot_model) {
    throw std::runtime_error(
        "[YamlParser] Missing required 'robot_model' section.");
  }

  const YAML::Node urdf_path = robot_model["urdf_path"];
  if (!urdf_path || !urdf_path.IsScalar()) {
    throw std::runtime_error(
        "[YamlParser] 'robot_model.urdf_path' must be a scalar string.");
  }

  RobotModelParams params = ParseRobotModelHints(root);
  params.urdf_path = urdf_path.as<std::string>();
  if (params.urdf_path.empty()) {
    throw std::runtime_error(
        "[YamlParser] 'robot_model.urdf_path' is empty.");
  }
  return params;
}

RobotModelParams ParseRobotModelHints(const YAML::Node& root) {
  RobotModelParams params;
  const YAML::Node robot_model = root["robot_model"];
  if (!robot_model) {
    return params;
  }

  if (robot_model["urdf_path"]) {
    if (!robot_model["urdf_path"].IsScalar()) {
      throw std::runtime_error(
          "[YamlParser] 'robot_model.urdf_path' must be a scalar string.");
    }
    params.urdf_path = robot_model["urdf_path"].as<std::string>();
  }

  std::optional<bool> is_floating_base_param;
  if (robot_model["is_floating_base"]) {
    try {
      is_floating_base_param = robot_model["is_floating_base"].as<bool>();
    } catch (const std::exception& e) {
      throw std::runtime_error(
          "[YamlParser] 'robot_model.is_floating_base' must be bool: " +
          std::string(e.what()));
    }
  }

  std::optional<bool> fixed_base_alias_param;
  if (robot_model["fixed_base"]) {
    try {
      fixed_base_alias_param = robot_model["fixed_base"].as<bool>();
    } catch (const std::exception& e) {
      throw std::runtime_error(
          "[YamlParser] 'robot_model.fixed_base' must be bool: " +
          std::string(e.what()));
    }
  }

  if (is_floating_base_param.has_value() && fixed_base_alias_param.has_value()) {
    if (*is_floating_base_param == *fixed_base_alias_param) {
      throw std::runtime_error(
          "[YamlParser] robot_model.is_floating_base and "
          "robot_model.fixed_base are inconsistent.");
    }
    params.is_floating_base = *is_floating_base_param;
  } else if (is_floating_base_param.has_value()) {
    params.is_floating_base = *is_floating_base_param;
  } else if (fixed_base_alias_param.has_value()) {
    params.is_floating_base = !(*fixed_base_alias_param);
  }

  params.expected_num_qdot =
      ParseOptionalNonNegativeInt(robot_model, "expected_num_qdot");
  params.expected_num_active_dof =
      ParseOptionalNonNegativeInt(robot_model, "expected_num_active_dof");
  params.expected_num_float_dof =
      ParseOptionalNonNegativeInt(robot_model, "expected_num_float_dof");

  params.base_frame_name = robot_model["base_frame"].as<std::string>("");
  params.end_effector_frame_name =
      robot_model["end_effector_frame"].as<std::string>("");

  if (robot_model["floating_base"]) {
    params.floating_base_override = robot_model["floating_base"].as<bool>();
  }

  if (robot_model["unactuated_qdot_indices"]) {
    const YAML::Node indices = robot_model["unactuated_qdot_indices"];
    if (!indices.IsSequence()) {
      throw std::runtime_error(
          "[YamlParser] robot_model.unactuated_qdot_indices must be a YAML sequence.");
    }
    params.unactuated_qdot_indices.reserve(indices.size());
    for (const auto& idx_node : indices) {
      params.unactuated_qdot_indices.push_back(idx_node.as<int>());
    }
  }

  return params;
}

std::optional<int> ParseTopLevelStartStateId(const YAML::Node& root) {
  if (!root || !root["start_state_id"]) {
    return std::nullopt;
  }
  return root["start_state_id"].as<int>();
}

RegularizationParams ParseRegularization(const YAML::Node& root,
                                         const std::string& main_yaml_path) {
  RegularizationParams params;
  const YAML::Node node =
      ResolveRegularizationNode(root, main_yaml_path);
  if (!node) {
    return params;
  }

  params.w_qddot = node["w_qddot"].as<double>(params.w_qddot);
  params.w_rf = node["w_rf"].as<double>(params.w_rf);
  params.w_tau = node["w_tau"].as<double>(params.w_tau);
  params.w_tau_dot = node["w_tau_dot"].as<double>(params.w_tau_dot);
  params.w_xc_ddot = node["w_xc_ddot"].as<double>(params.w_xc_ddot);
  if (node["w_f_dot"]) {
    params.w_f_dot = node["w_f_dot"].as<double>();
  } else if (node["w_force_rate_of_change"]) {
    params.w_f_dot = node["w_force_rate_of_change"].as<double>();
  } else {
    params.w_f_dot = node["w_force_rate"].as<double>(params.w_f_dot);
  }

  return params;
}

int ParseMaxContactDim(const YAML::Node& root,
                       const std::string& main_yaml_path) {
  // Check inline first, then legacy solver_params locations.
  if (root["max_contact_dim"]) {
    return root["max_contact_dim"].as<int>();
  }
  const YAML::Node node =
      ResolveRegularizationNode(root, main_yaml_path);
  if (node && node["max_contact_dim"]) {
    return node["max_contact_dim"].as<int>();
  }
  return -1;
}

} // namespace wbc::param
