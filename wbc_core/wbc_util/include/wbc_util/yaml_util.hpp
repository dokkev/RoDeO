#pragma once

#include <iostream>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <yaml-cpp/yaml.h>

namespace wbc_util {

template <typename T>
T ReadParameter(const YAML::Node& node, const std::string& name) {
  if (!node[name]) {
    std::cerr << "[YamlUtil] Error: Parameter '" << name
              << "' does not exist!" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  return node[name].as<T>();
}

template <typename T>
T ReadParameter(const YAML::Node& node, const std::string& name,
                const T& default_value) {
  if (!node[name]) {
    return default_value;
  }
  return node[name].as<T>();
}

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
