// Copyright 2026
//
// Licensed under the Apache License, Version 2.0.

#include "mppi_core/config/mppi_config.hpp"

#include <stdexcept>
#include <string>
#include <utility>

namespace mppi_core {
namespace {

bool HasValue(const YAML::Node& node) {
  return node && node.Type() != YAML::NodeType::Undefined &&
         node.Type() != YAML::NodeType::Null;
}

int ReadInt(const YAML::Node& params, const char* key, int default_value) {
  if (!HasValue(params)) {
    return default_value;
  }
  const YAML::Node value = params[key];
  if (!HasValue(value)) {
    return default_value;
  }
  return value.as<int>();
}

double ReadDouble(const YAML::Node& params, const char* key,
                  double default_value) {
  if (!HasValue(params)) {
    return default_value;
  }
  const YAML::Node value = params[key];
  if (!HasValue(value)) {
    return default_value;
  }
  return value.as<double>();
}

YAML::Node ReadSection(const YAML::Node& node, const char* key) {
  if (!HasValue(node)) {
    return YAML::Node();
  }
  const YAML::Node value = node[key];
  if (!HasValue(value)) {
    return YAML::Node();
  }
  if (!value.IsMap()) {
    throw std::invalid_argument(std::string("Section '") + key +
                                "' must be a map");
  }
  return value;
}

Eigen::VectorXd ReadVectorXdOrScalar(const YAML::Node& params, const char* key,
                                     std::size_t expected_size,
                                     const Eigen::VectorXd& default_value) {
  if (!HasValue(params)) {
    return default_value;
  }
  const YAML::Node value = params[key];
  if (!HasValue(value)) {
    return default_value;
  }
  if (value.IsScalar()) {
    return Eigen::VectorXd::Constant(static_cast<Eigen::Index>(expected_size),
                                     value.as<double>());
  }
  if (!value.IsSequence()) {
    throw std::invalid_argument(std::string("Field '") + key +
                                "' must be a scalar or sequence");
  }
  if (value.size() != expected_size) {
    throw std::invalid_argument(std::string("Field '") + key +
                                "' dimension mismatch");
  }

  Eigen::VectorXd out(static_cast<Eigen::Index>(expected_size));
  for (std::size_t i = 0; i < expected_size; ++i) {
    out[static_cast<Eigen::Index>(i)] = value[i].as<double>();
  }
  return out;
}

YAML::Node MPPIConfigNode(const YAML::Node& root) {
  if (!HasValue(root)) {
    throw std::invalid_argument("MPPI YAML root is empty");
  }
  if (!root.IsMap()) {
    throw std::invalid_argument("MPPI YAML root must be a map");
  }

  const YAML::Node mppi = root["mppi"];
  if (HasValue(mppi)) {
    return mppi;
  }
  return root;
}

void ValidatePositiveInt(int value, const char* name) {
  if (value <= 0) {
    throw std::invalid_argument(std::string("MPPIConfig: ") + name +
                                " must be positive");
  }
}

}  // namespace

MPPIConfig ParseMPPIConfig(const YAML::Node& params, std::size_t action_dim,
                           MPPIConfig defaults) {
  if (action_dim == 0) {
    throw std::invalid_argument("ParseMPPIConfig: action_dim is zero");
  }

  const YAML::Node safe_params = HasValue(params) ? params : YAML::Node();
  if (HasValue(safe_params) && !safe_params.IsMap()) {
    throw std::invalid_argument("ParseMPPIConfig: params must be a map");
  }

  const int horizon_steps = ReadInt(safe_params, "horizon_steps",
                                    static_cast<int>(defaults.horizon_steps));
  const int num_rollouts = ReadInt(safe_params, "num_rollouts",
                                   static_cast<int>(defaults.num_rollouts));
  ValidatePositiveInt(horizon_steps, "horizon_steps");
  ValidatePositiveInt(num_rollouts, "num_rollouts");

  defaults.horizon_steps = static_cast<std::size_t>(horizon_steps);
  defaults.num_rollouts = static_cast<std::size_t>(num_rollouts);
  defaults.action_dim = action_dim;
  defaults.dt = ReadDouble(safe_params, "dt", defaults.dt);
  defaults.temperature =
      ReadDouble(safe_params, "temperature", defaults.temperature);

  const int random_seed = ReadInt(safe_params, "random_seed",
                                  static_cast<int>(defaults.random_seed));
  if (random_seed < 0) {
    throw std::invalid_argument("MPPIConfig: random_seed must be nonnegative");
  }
  defaults.random_seed = static_cast<std::uint32_t>(random_seed);

  const YAML::Node action = ReadSection(safe_params, "action");
  defaults.action_lower_bound = ReadVectorXdOrScalar(
      action, "lower_bound", action_dim, defaults.action_lower_bound);
  defaults.action_upper_bound = ReadVectorXdOrScalar(
      action, "upper_bound", action_dim, defaults.action_upper_bound);
  defaults.action_noise_std = ReadVectorXdOrScalar(
      action, "noise_std", action_dim, defaults.action_noise_std);

  // Backward-compatible flat fields for older experimental configs.
  defaults.action_lower_bound =
      ReadVectorXdOrScalar(safe_params, "action_lower_bound", action_dim,
                           defaults.action_lower_bound);
  defaults.action_upper_bound =
      ReadVectorXdOrScalar(safe_params, "action_upper_bound", action_dim,
                           defaults.action_upper_bound);
  defaults.action_noise_std = ReadVectorXdOrScalar(
      safe_params, "action_noise_std", action_dim, defaults.action_noise_std);

  return defaults;
}

MPPIConfig LoadMPPIConfigFromYamlFile(const std::string& yaml_path,
                                      std::size_t action_dim,
                                      MPPIConfig defaults) {
  try {
    const YAML::Node root = YAML::LoadFile(yaml_path);
    return ParseMPPIConfig(MPPIConfigNode(root), action_dim,
                           std::move(defaults));
  } catch (const YAML::Exception& ex) {
    throw std::runtime_error("LoadMPPIConfigFromYamlFile: failed to load '" +
                             yaml_path + "': " + ex.what());
  }
}

}  // namespace mppi_core
