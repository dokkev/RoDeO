/**
 * @file wbc_core/wbc_util/include/wbc_util/yaml_parser.hpp
 * @brief Doxygen documentation for yaml_parser module.
 */
#pragma once

#include <optional>
#include <string>
#include <vector>

#include <yaml-cpp/yaml.h>

namespace wbc::param {

/**
 * @brief Resolve optional `params` sub-node with legacy fallback.
 *
 * @details
 * If `node["params"]` exists, return that sub-node; otherwise return `node`
 * itself. When `node` is null, an empty node is returned.
 */
inline YAML::Node ResolveParamsNode(const YAML::Node& node) {
  if (!node) {
    return YAML::Node();
  }
  return node["params"] ? node["params"] : node;
}

/**
 * @brief Parsed robot model section from YAML.
 */
struct RobotModelParams {
  std::string urdf_path;
  bool is_floating_base{false};

  std::optional<int> expected_num_qdot;
  std::optional<int> expected_num_active_dof;
  std::optional<int> expected_num_float_dof;

  std::string base_frame_name;
  std::string end_effector_frame_name;
  std::vector<int> unactuated_qdot_indices;
  std::optional<bool> floating_base_override;
};

/**
 * @brief QP regularization weights parsed from YAML.
 */
struct RegularizationParams {
  double w_qddot{1.0e-6};
  double w_rf{1.0e-4};
  double w_tau{0.0};
  double w_tau_dot{0.0};
  double w_xc_ddot{1.0e-3};
  double w_f_dot{1.0e-3};
};

/// @deprecated Use RegularizationParams directly.
using SolverParams = RegularizationParams;

/**
 * @brief Top-level parsed YAML definition bundle.
 */
struct WbcDefinition {
  // Resolved main YAML path (absolute or package://-resolved path).
  std::string yaml_path;
  YAML::Node root;
  RobotModelParams robot_model;
};

/**
 * @brief Load YAML file after URI/path resolution.
 */
YAML::Node LoadYamlFile(const std::string& yaml_path);
/**
 * @brief Parse complete definition for architecture assembly.
 */
WbcDefinition ParseDefinition(const std::string& yaml_path);

// Used by builders that must instantiate robot model from YAML.
RobotModelParams ParseRequiredRobotModel(const YAML::Node& root);

// Used by runtime config compiler for optional robot_model hints.
RobotModelParams ParseRobotModelHints(const YAML::Node& root);

std::optional<int> ParseTopLevelStartStateId(const YAML::Node& root);
RegularizationParams ParseRegularization(const YAML::Node& root,
                                         const std::string& main_yaml_path);
int ParseMaxContactDim(const YAML::Node& root,
                       const std::string& main_yaml_path);

} // namespace wbc::param
