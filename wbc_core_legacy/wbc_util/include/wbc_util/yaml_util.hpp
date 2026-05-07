/**
 * @file wbc_core/wbc_util/include/wbc_util/yaml_util.hpp
 * @brief Doxygen documentation for yaml_util module.
 */
#pragma once

#include <iostream>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <yaml-cpp/yaml.h>

namespace wbc_util {

/**
 * @brief Read required scalar parameter from YAML node.
 *
 * @warning Exits process on missing key (legacy behavior).
 */
template <typename T>
T ReadParameter(const YAML::Node& node, const std::string& name) {
  if (!node[name]) {
    std::cerr << "[YamlUtil] Error: Parameter '" << name
              << "' does not exist!" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  return node[name].as<T>();
}

/**
 * @brief Read optional scalar parameter from YAML node.
 */
template <typename T>
T ReadParameter(const YAML::Node& node, const std::string& name,
                const T& default_value) {
  if (!node[name]) {
    return default_value;
  }
  return node[name].as<T>();
}

/**
 * @brief Read required vector parameter as Eigen vector.
 *
 * @warning Exits process on missing key (legacy behavior).
 */
inline Eigen::VectorXd ReadVector(const YAML::Node& node,
                                  const std::string& name) {
  if (!node[name]) {
    std::cerr << "[YamlUtil] Error: Vector parameter '" << name
              << "' does not exist!" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  std::vector<double> vec_std = node[name].as<std::vector<double>>();
  return Eigen::Map<Eigen::VectorXd>(vec_std.data(), vec_std.size());
}

/**
 * @brief Read optional vector parameter as Eigen vector.
 */
inline Eigen::VectorXd ReadVector(const YAML::Node& node,
                                  const std::string& name,
                                  const Eigen::VectorXd& default_val) {
  if (!node[name]) {
    return default_val;
  }
  std::vector<double> vec_std = node[name].as<std::vector<double>>();
  return Eigen::Map<Eigen::VectorXd>(vec_std.data(), vec_std.size());
}

} // namespace wbc_util
