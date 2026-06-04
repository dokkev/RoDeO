// Copyright 2026
//
// Licensed under the Apache License, Version 2.0.

#include "mppi_core/config/grasp_config.hpp"

#include <cstddef>
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

std::size_t ReadSize(const YAML::Node& node, const char* key,
                     std::size_t default_value) {
  if (!HasValue(node)) {
    return default_value;
  }
  const YAML::Node value = node[key];
  if (!HasValue(value)) {
    return default_value;
  }
  const int parsed = value.as<int>();
  if (parsed < 0) {
    throw std::invalid_argument(std::string("Field '") + key +
                                "' must be nonnegative");
  }
  return static_cast<std::size_t>(parsed);
}

TactileRolloutPolicy ReadTactileRolloutPolicy(
    const YAML::Node& node, const char* key,
    TactileRolloutPolicy default_value) {
  if (!HasValue(node)) {
    return default_value;
  }
  const YAML::Node value = node[key];
  if (!HasValue(value)) {
    return default_value;
  }
  const std::string policy = value.as<std::string>();
  if (policy == "force_aware_required") {
    return TactileRolloutPolicy::kForceAwareRequired;
  }
  if (policy == "force_then_kinematic_fallback") {
    return TactileRolloutPolicy::kForceThenKinematicFallback;
  }
  throw std::invalid_argument(
      "Field 'rollout_policy' must be one of: force_aware_required, "
      "force_then_kinematic_fallback");
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

DeltaQReferenceRolloutConfig ParseDeltaQReferenceRolloutConfig(
    const YAML::Node& params, DeltaQReferenceRolloutConfig defaults) {
  const YAML::Node safe_params = HasValue(params) ? params : YAML::Node();
  if (HasValue(safe_params) && !safe_params.IsMap()) {
    throw std::invalid_argument(
        "ParseDeltaQReferenceRolloutConfig: params must be a map");
  }

  const YAML::Node tactile_prediction =
      ReadSection(safe_params, "tactile_prediction");
  defaults.tactile_rollout_policy = ReadTactileRolloutPolicy(
      tactile_prediction, "rollout_policy", defaults.tactile_rollout_policy);
  return defaults;
}

DeltaQReferenceRolloutConfig LoadDeltaQReferenceRolloutConfigFromYamlFile(
    const std::string& yaml_path, DeltaQReferenceRolloutConfig defaults) {
  try {
    const YAML::Node root = YAML::LoadFile(yaml_path);
    return ParseDeltaQReferenceRolloutConfig(GraspConfigNode(root),
                                             std::move(defaults));
  } catch (const YAML::Exception& ex) {
    throw std::runtime_error(
        "LoadDeltaQReferenceRolloutConfigFromYamlFile: failed to load '" +
        yaml_path + "': " + ex.what());
  }
}

ContactForceRolloutConfig ParseContactForceRolloutConfig(
    const YAML::Node& params, ContactForceRolloutConfig defaults) {
  const YAML::Node safe_params = HasValue(params) ? params : YAML::Node();
  if (HasValue(safe_params) && !safe_params.IsMap()) {
    throw std::invalid_argument(
        "ParseContactForceRolloutConfig: params must be a map");
  }

  const YAML::Node rollout = ReadSection(safe_params, "contact_force_rollout");
  defaults.enable_force_projection_update = ReadBool(
      rollout, "enable_force_projection_update",
      defaults.enable_force_projection_update);
  defaults.force_lowpass_alpha =
      ReadDouble(rollout, "force_lowpass_alpha", defaults.force_lowpass_alpha);
  defaults.max_predicted_normal_force_n = ReadDouble(
      rollout, "max_predicted_normal_force_n",
      defaults.max_predicted_normal_force_n);
  defaults.shear_force_gain_m_per_n_s = ReadDouble(
      rollout, "shear_force_gain_m_per_n_s",
      defaults.shear_force_gain_m_per_n_s);
  defaults.rotational_shear_gain_rad_per_nm_s = ReadDouble(
      rollout, "rotational_shear_gain_rad_per_nm_s",
      defaults.rotational_shear_gain_rad_per_nm_s);
  defaults.friction_violation_confidence_decay = ReadDouble(
      rollout, "friction_violation_confidence_decay",
      defaults.friction_violation_confidence_decay);
  defaults.negative_normal_confidence_decay = ReadDouble(
      rollout, "negative_normal_confidence_decay",
      defaults.negative_normal_confidence_decay);
  defaults.min_stable_support_count = ReadSize(
      rollout, "min_stable_support_count", defaults.min_stable_support_count);
  defaults.min_contact_confidence = ReadDouble(
      rollout, "min_contact_confidence", defaults.min_contact_confidence);
  defaults.shear_ref_m =
      ReadDouble(rollout, "shear_ref_m", defaults.shear_ref_m);
  defaults.rotational_shear_ref_rad = ReadDouble(
      rollout, "rotational_shear_ref_rad",
      defaults.rotational_shear_ref_rad);
  defaults.rollout_torque_stiffness_nm_per_rad = ReadDouble(
      rollout, "rollout_torque_stiffness_nm_per_rad",
      defaults.rollout_torque_stiffness_nm_per_rad);
  defaults.rollout_torque_damping_nms_per_rad = ReadDouble(
      rollout, "rollout_torque_damping_nms_per_rad",
      defaults.rollout_torque_damping_nms_per_rad);
  return defaults;
}

ContactForceRolloutConfig LoadContactForceRolloutConfigFromYamlFile(
    const std::string& yaml_path, ContactForceRolloutConfig defaults) {
  try {
    const YAML::Node root = YAML::LoadFile(yaml_path);
    return ParseContactForceRolloutConfig(GraspConfigNode(root),
                                          std::move(defaults));
  } catch (const YAML::Exception& ex) {
    throw std::runtime_error(
        "LoadContactForceRolloutConfigFromYamlFile: failed to load '" +
        yaml_path + "': " + ex.what());
  }
}

}  // namespace mppi_core
