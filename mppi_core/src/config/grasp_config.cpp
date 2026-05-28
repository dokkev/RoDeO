// Copyright 2026
//
// Licensed under the Apache License, Version 2.0.

#include "mppi_core/config/grasp_config.hpp"

#include <stdexcept>
#include <string>
#include <utility>

namespace mppi_core {
namespace {

bool HasValue(const YAML::Node& node) {
  return node && node.Type() != YAML::NodeType::Undefined &&
         node.Type() != YAML::NodeType::Null;
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

bool ReadBool(const YAML::Node& node, const char* key, bool default_value) {
  if (!HasValue(node)) {
    return default_value;
  }
  const YAML::Node value = node[key];
  if (!HasValue(value)) {
    return default_value;
  }
  return value.as<bool>();
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

Eigen::Vector2d ReadVector2(const YAML::Node& params, const char* key,
                            const Eigen::Vector2d& default_value) {
  if (!HasValue(params)) {
    return default_value;
  }
  const YAML::Node value = params[key];
  if (!HasValue(value)) {
    return default_value;
  }
  if (!value.IsSequence() || value.size() != 2) {
    throw std::invalid_argument(std::string("Field '") + key +
                                "' must be a length-2 sequence");
  }
  return Eigen::Vector2d{value[0].as<double>(), value[1].as<double>()};
}

Eigen::VectorXd ReadVectorXd(const YAML::Node& params, const char* key,
                             std::size_t expected_size,
                             const Eigen::VectorXd& default_value) {
  if (!HasValue(params)) {
    return default_value;
  }
  const YAML::Node value = params[key];
  if (!HasValue(value)) {
    return default_value;
  }
  if (!value.IsSequence()) {
    throw std::invalid_argument(std::string("Field '") + key +
                                "' must be a sequence");
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

YAML::Node GraspConfigNode(const YAML::Node& root) {
  if (!HasValue(root)) {
    throw std::invalid_argument("Grasp YAML root is empty");
  }
  if (!root.IsMap()) {
    throw std::invalid_argument("Grasp YAML root must be a map");
  }

  const YAML::Node grasp = root["grasp"];
  if (HasValue(grasp)) {
    return grasp;
  }

  return root;
}

}  // namespace

GraspStabilityCostConfig ParseGraspConfig(const YAML::Node& params,
                                          std::size_t action_dim,
                                          GraspStabilityCostConfig defaults) {
  if (action_dim == 0) {
    throw std::invalid_argument("ParseGraspConfig: action_dim is zero");
  }

  const YAML::Node safe_params = HasValue(params) ? params : YAML::Node();
  if (HasValue(safe_params) && !safe_params.IsMap()) {
    throw std::invalid_argument("ParseGraspConfig: params must be a map");
  }

  const YAML::Node force_window =
      ReadSection(safe_params, "normal_force_window");
  defaults.force_min_n =
      ReadDouble(force_window, "min_n", defaults.force_min_n);
  defaults.force_max_n =
      ReadDouble(force_window, "max_n", defaults.force_max_n);
  defaults.force_under_weight =
      ReadDouble(force_window, "under_weight", defaults.force_under_weight);
  defaults.force_over_weight =
      ReadDouble(force_window, "over_weight", defaults.force_over_weight);
  defaults.use_object_weight_lower_bound =
      ReadBool(force_window, "use_object_weight_lower_bound",
               defaults.use_object_weight_lower_bound);
  defaults.object_force_safety_factor =
      ReadDouble(force_window, "weight_safety_factor",
                 defaults.object_force_safety_factor);

  const YAML::Node friction_margin =
      ReadSection(safe_params, "friction_margin");
  defaults.friction_margin_enabled =
      ReadBool(friction_margin, "enabled", defaults.friction_margin_enabled);
  defaults.friction_margin_weight =
      ReadDouble(friction_margin, "weight", defaults.friction_margin_weight);
  defaults.friction_coefficient =
      ReadDouble(friction_margin, "mu_nominal", defaults.friction_coefficient);
  defaults.friction_mu_min =
      ReadDouble(friction_margin, "mu_min", defaults.friction_mu_min);
  defaults.required_force_weight = ReadDouble(
      friction_margin, "required_force_weight", defaults.required_force_weight);

  const YAML::Node tangential_load =
      ReadSection(safe_params, "tangential_load_proxy");
  defaults.gravity_tangential_load_weight =
      ReadDouble(tangential_load, "gravity_weight",
                 defaults.gravity_tangential_load_weight);
  defaults.motion_tangential_load_weight = ReadDouble(
      tangential_load, "motion_weight", defaults.motion_tangential_load_weight);
  defaults.slip_tangential_load_weight = ReadDouble(
      tangential_load, "slip_weight", defaults.slip_tangential_load_weight);
  defaults.force_spike_tangential_load_weight =
      ReadDouble(tangential_load, "force_spike_weight",
                 defaults.force_spike_tangential_load_weight);

  const YAML::Node normal_force_proxy =
      ReadSection(safe_params, "normal_force_proxy");
  defaults.closing_force_gain_n_per_rad =
      ReadDouble(normal_force_proxy, "closing_force_gain",
                 defaults.closing_force_gain_n_per_rad);
  defaults.opening_force_gain_n_per_rad =
      ReadDouble(normal_force_proxy, "opening_force_gain",
                 defaults.opening_force_gain_n_per_rad);
  defaults.force_proxy_max_n =
      ReadDouble(normal_force_proxy, "max_force_n", defaults.force_proxy_max_n);

  const YAML::Node tactile_prediction =
      ReadSection(safe_params, "tactile_prediction");
  defaults.contact_patch_force_per_node_n =
      ReadDouble(tactile_prediction, "force_per_node_n",
                 defaults.contact_patch_force_per_node_n);
  defaults.slip_prediction_decay = ReadDouble(tactile_prediction, "slip_decay",
                                              defaults.slip_prediction_decay);
  defaults.slip_prediction_margin_gain_per_n =
      ReadDouble(tactile_prediction, "slip_margin_gain_per_n",
                 defaults.slip_prediction_margin_gain_per_n);
  defaults.centroid_slip_drift_gain_m_per_n =
      ReadDouble(tactile_prediction, "centroid_drift_gain_m_per_n",
                 defaults.centroid_slip_drift_gain_m_per_n);
  defaults.slip_velocity_decay =
      ReadDouble(tactile_prediction, "slip_velocity_decay",
                 defaults.slip_velocity_decay);
  defaults.slip_velocity_margin_gain_per_nps =
      ReadDouble(tactile_prediction, "slip_velocity_margin_gain_per_nps",
                 defaults.slip_velocity_margin_gain_per_nps);
  defaults.action_slip_damping_gain_per_rad =
      ReadDouble(tactile_prediction, "action_slip_damping_gain_per_rad",
                 defaults.action_slip_damping_gain_per_rad);
  defaults.max_slip_velocity = ReadDouble(
      tactile_prediction, "max_slip_velocity", defaults.max_slip_velocity);
  defaults.centroid_velocity_decay =
      ReadDouble(tactile_prediction, "centroid_velocity_decay",
                 defaults.centroid_velocity_decay);
  defaults.centroid_velocity_slip_gain =
      ReadDouble(tactile_prediction, "centroid_velocity_slip_gain",
                 defaults.centroid_velocity_slip_gain);
  defaults.max_centroid_velocity_mps =
      ReadDouble(tactile_prediction, "max_centroid_velocity_mps",
                 defaults.max_centroid_velocity_mps);

  const YAML::Node slip_risk = ReadSection(safe_params, "slip_risk");
  defaults.slip_threshold =
      ReadDouble(slip_risk, "threshold", defaults.slip_threshold);
  defaults.slip_risk_weight =
      ReadDouble(slip_risk, "weight", defaults.slip_risk_weight);
  defaults.slip_velocity_weight =
      ReadDouble(slip_risk, "velocity_weight", defaults.slip_velocity_weight);

  const YAML::Node contact_centroid =
      ReadSection(safe_params, "contact_centroid");
  defaults.contact_centroid_enabled =
      ReadBool(contact_centroid, "enabled", defaults.contact_centroid_enabled);
  defaults.centroid_boundary_weight = ReadDouble(
      contact_centroid, "boundary_weight", defaults.centroid_boundary_weight);
  defaults.centroid_x_min =
      ReadDouble(contact_centroid, "x_min", defaults.centroid_x_min);
  defaults.centroid_x_max =
      ReadDouble(contact_centroid, "x_max", defaults.centroid_x_max);
  defaults.centroid_y_min =
      ReadDouble(contact_centroid, "y_min", defaults.centroid_y_min);
  defaults.centroid_y_max =
      ReadDouble(contact_centroid, "y_max", defaults.centroid_y_max);

  const YAML::Node contact_patch = ReadSection(safe_params, "contact_patch");
  defaults.contact_patch_enabled =
      ReadBool(contact_patch, "enabled", defaults.contact_patch_enabled);
  defaults.contact_patch_target_node_count =
      ReadDouble(contact_patch, "target_node_count",
                 defaults.contact_patch_target_node_count);
  defaults.contact_patch_weight =
      ReadDouble(contact_patch, "weight", defaults.contact_patch_weight);

  const YAML::Node tracking_guard = ReadSection(safe_params, "tracking_guard");
  defaults.tracking_weight =
      ReadDouble(tracking_guard, "weight", defaults.tracking_weight);
  defaults.tracking_action_scale_weight =
      ReadDouble(tracking_guard, "action_scale_weight",
                 defaults.tracking_action_scale_weight);

  const YAML::Node action_smoothness =
      ReadSection(safe_params, "action_smoothness");
  defaults.action_smoothness_weight = ReadDouble(
      action_smoothness, "weight", defaults.action_smoothness_weight);

  const YAML::Node joint_limit = ReadSection(safe_params, "joint_limit");
  defaults.joint_limit_weight =
      ReadDouble(joint_limit, "weight", defaults.joint_limit_weight);
  defaults.joint_lower_bound = ReadVectorXd(
      joint_limit, "lower_bound", action_dim, defaults.joint_lower_bound);
  defaults.joint_upper_bound = ReadVectorXd(
      joint_limit, "upper_bound", action_dim, defaults.joint_upper_bound);

  const YAML::Node action = ReadSection(safe_params, "action");

  // Backward-compatible flat fields for older experimental configs.
  defaults.force_min_n =
      ReadDouble(safe_params, "force_min_n", defaults.force_min_n);
  defaults.force_max_n =
      ReadDouble(safe_params, "force_max_n", defaults.force_max_n);
  const double force_window_weight =
      ReadDouble(safe_params, "force_window_weight", -1.0);
  if (force_window_weight >= 0.0) {
    defaults.force_under_weight = force_window_weight;
    defaults.force_over_weight = force_window_weight;
  }
  defaults.force_under_weight = ReadDouble(safe_params, "force_under_weight",
                                           defaults.force_under_weight);
  defaults.force_over_weight =
      ReadDouble(safe_params, "force_over_weight", defaults.force_over_weight);
  defaults.friction_coefficient = ReadDouble(
      safe_params, "friction_coefficient", defaults.friction_coefficient);
  defaults.friction_mu_min =
      ReadDouble(safe_params, "friction_mu_min", defaults.friction_mu_min);
  defaults.object_force_safety_factor =
      ReadDouble(safe_params, "object_force_safety_factor",
                 defaults.object_force_safety_factor);
  defaults.supporting_contact_count =
      ReadDouble(safe_params, "supporting_contact_count",
                 defaults.supporting_contact_count);
  defaults.slip_threshold =
      ReadDouble(safe_params, "slip_threshold", defaults.slip_threshold);
  defaults.slip_risk_weight =
      ReadDouble(safe_params, "slip_weight", defaults.slip_risk_weight);
  const Eigen::Vector2d centroid_abs_limit_m = ReadVector2(
      safe_params, "centroid_abs_limit_m", Eigen::Vector2d{-1.0, -1.0});
  if ((centroid_abs_limit_m.array() > 0.0).all()) {
    defaults.centroid_x_min = -centroid_abs_limit_m.x();
    defaults.centroid_x_max = centroid_abs_limit_m.x();
    defaults.centroid_y_min = -centroid_abs_limit_m.y();
    defaults.centroid_y_max = centroid_abs_limit_m.y();
  }
  defaults.centroid_boundary_weight = ReadDouble(
      safe_params, "centroid_weight", defaults.centroid_boundary_weight);
  defaults.contact_loss_weight = ReadDouble(safe_params, "contact_loss_weight",
                                            defaults.contact_loss_weight);
  defaults.action_smoothness_weight = ReadDouble(
      safe_params, "action_weight", defaults.action_smoothness_weight);

  defaults.closing_force_gain_n_per_rad =
      ReadDouble(safe_params, "closing_force_gain_n_per_rad",
                 defaults.closing_force_gain_n_per_rad);
  defaults.opening_force_gain_n_per_rad =
      ReadDouble(safe_params, "opening_force_gain_n_per_rad",
                 defaults.opening_force_gain_n_per_rad);

  const Eigen::VectorXd default_direction =
      defaults.closing_direction.size() == 0
          ? Eigen::VectorXd::Ones(static_cast<Eigen::Index>(action_dim))
          : defaults.closing_direction;
  defaults.closing_direction =
      ReadVectorXd(action, "closing_direction", action_dim, default_direction);
  defaults.closing_direction = ReadVectorXd(
      safe_params, "closing_direction", action_dim, defaults.closing_direction);
  return defaults;
}

GraspStabilityCostConfig LoadGraspConfigFromYamlFile(
    const std::string& yaml_path, std::size_t action_dim,
    GraspStabilityCostConfig defaults) {
  try {
    const YAML::Node root = YAML::LoadFile(yaml_path);
    return ParseGraspConfig(GraspConfigNode(root), action_dim,
                            std::move(defaults));
  } catch (const YAML::Exception& ex) {
    throw std::runtime_error("LoadGraspConfigFromYamlFile: failed to load '" +
                             yaml_path + "': " + ex.what());
  }
}

}  // namespace mppi_core
