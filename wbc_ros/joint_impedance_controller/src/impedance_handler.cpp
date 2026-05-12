#include "joint_impedance_controller/impedance_handler.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <yaml-cpp/yaml.h>

namespace joint_impedance_controller
{

namespace
{

constexpr const char * kPresetRootKey = "impedance_preset";
constexpr const char * kMinPresetKey = "min";
constexpr const char * kMaxPresetKey = "max";
constexpr double kMinSupportedLevel = 0.0;
constexpr double kMaxSupportedLevel = 10.0;

double clamp_level(double level)
{
  return std::clamp(level, kMinSupportedLevel, kMaxSupportedLevel);
}

bool parse_required_vector(
  const YAML::Node & preset_node,
  const std::string & preset_label,
  const std::string & field_name,
  std::vector<double> * values_out,
  std::string * error_out)
{
  if (values_out == nullptr) {
    if (error_out != nullptr) {
      *error_out = "Impedance vector output pointer is null.";
    }
    return false;
  }

  const auto field_node = preset_node[field_name];
  if (!field_node || !field_node.IsSequence()) {
    if (error_out != nullptr) {
      *error_out =
        "Impedance preset '" + preset_label + "'" +
        " is missing sequence field '" + field_name + "'.";
    }
    return false;
  }

  try {
    *values_out = field_node.as<std::vector<double>>();
  } catch (const std::exception & ex) {
    if (error_out != nullptr) {
      *error_out =
        "Failed to parse '" + field_name + "' for impedance preset '" +
        preset_label + "'" +
        "': " + ex.what();
    }
    return false;
  }

  if (values_out->empty()) {
    if (error_out != nullptr) {
      *error_out =
        "Impedance preset '" + preset_label + "'" +
        " field '" + field_name + "' is empty.";
    }
    return false;
  }

  return true;
}

}  // namespace

ImpedanceHandler::ImpedanceHandler(
  int joint_count,
  std::string preset_yaml_path)
: joint_count_(std::clamp(joint_count, 0, 8)),
  preset_yaml_path_(std::move(preset_yaml_path)),
  gains_{
    std::vector<double>(static_cast<size_t>(joint_count_), 0.0),
    std::vector<double>(static_cast<size_t>(joint_count_), 0.0)}
{
  std::string error;
  if (!load_presets(&error)) {
    throw std::runtime_error(error);
  }
}

bool ImpedanceHandler::set_level(
  double level,
  std::string * error_out)
{
  std::lock_guard<std::mutex> lock(mutex_);
  if (anchors_.empty()) {
    if (error_out != nullptr) {
      *error_out = "No impedance level anchors are loaded.";
    }
    return false;
  }

  const auto clamped_level = clamp_level(level);
  gains_ = interpolate_gains(clamped_level);
  active_level_ = clamped_level;
  using_custom_gains_ = false;
  return true;
}

void ImpedanceHandler::set_custom_gains(
  const std::vector<double> & stiffness,
  const std::vector<double> & damping)
{
  std::lock_guard<std::mutex> lock(mutex_);

  if (!stiffness.empty()) {
    gains_.stiffness = sanitize_vector(stiffness, stiffness.front());
  }
  if (!damping.empty()) {
    gains_.damping = sanitize_vector(damping, damping.front());
  }

  active_level_.reset();
  using_custom_gains_ = true;
}

ImpedanceGains ImpedanceHandler::gains() const
{
  std::lock_guard<std::mutex> lock(mutex_);
  return gains_;
}

double ImpedanceHandler::active_level() const
{
  std::lock_guard<std::mutex> lock(mutex_);
  return active_level_.value_or(kMinSupportedLevel);
}

std::string ImpedanceHandler::active_label() const
{
  std::lock_guard<std::mutex> lock(mutex_);
  if (using_custom_gains_) {
    return "custom";
  }
  if (active_level_.has_value()) {
    return std::to_string(*active_level_);
  }
  return "unset";
}

std::vector<double> ImpedanceHandler::anchor_levels() const
{
  std::lock_guard<std::mutex> lock(mutex_);
  return available_anchor_levels_;
}

bool ImpedanceHandler::load_presets(std::string * error_out)
{
  YAML::Node root;
  try {
    root = YAML::LoadFile(preset_yaml_path_);
  } catch (const std::exception & ex) {
    if (error_out != nullptr) {
      *error_out = "Failed to load impedance preset YAML '" + preset_yaml_path_ + "': " +
        ex.what();
    }
    return false;
  }

  if (!root || !root.IsMap()) {
    if (error_out != nullptr) {
      *error_out =
        "Impedance preset YAML must contain a map at the document root: " +
        preset_yaml_path_;
    }
    return false;
  }

  const auto preset_root = root[kPresetRootKey];
  if (!preset_root || !preset_root.IsMap()) {
    if (error_out != nullptr) {
      *error_out =
        "Impedance preset YAML must contain a map at root key '" +
        std::string(kPresetRootKey) + "'.";
    }
    return false;
  }

  std::vector<LevelAnchor> loaded_anchors;
  std::vector<double> loaded_anchor_levels;
  const auto min_node = preset_root[kMinPresetKey];
  const auto max_node = preset_root[kMaxPresetKey];
  if (!min_node || !max_node) {
    if (error_out != nullptr) {
      *error_out =
        "Impedance preset YAML must define both '" + std::string(kMinPresetKey) +
        "' and '" + std::string(kMaxPresetKey) + "' under '" + std::string(kPresetRootKey) +
        "'.";
    }
    return false;
  }

  for (const auto & entry : preset_root) {
    std::string key;
    try {
      key = entry.first.as<std::string>();
    } catch (const std::exception & ex) {
      if (error_out != nullptr) {
        *error_out =
          std::string("Failed to parse impedance preset key: ") + ex.what();
      }
      return false;
    }

    if (key != kMinPresetKey && key != kMaxPresetKey) {
      if (error_out != nullptr) {
        *error_out =
          "Impedance preset YAML only supports '" + std::string(kMinPresetKey) +
          "' and '" + std::string(kMaxPresetKey) + "' keys. Unsupported key: '" + key + "'.";
      }
      return false;
    }
  }

  for (const auto & named_entry :
    {std::pair<const char *, double>{kMinPresetKey, kMinSupportedLevel},
      std::pair<const char *, double>{kMaxPresetKey, kMaxSupportedLevel}})
  {
    const auto preset_node = preset_root[named_entry.first];
    if (!preset_node || !preset_node.IsMap()) {
      if (error_out != nullptr) {
        *error_out =
          "Impedance preset must contain a map for key '" +
          std::string(named_entry.first) + "'.";
      }
      return false;
    }

    std::vector<double> stiffness;
    std::vector<double> damping;
    if (!parse_required_vector(
        preset_node, named_entry.first, "stiffness", &stiffness, error_out))
    {
      return false;
    }
    if (!parse_required_vector(
        preset_node, named_entry.first, "damping", &damping, error_out))
    {
      return false;
    }

    loaded_anchors.push_back(LevelAnchor{
      named_entry.second,
      ImpedanceGains{
        sanitize_vector(stiffness, stiffness.front()),
        sanitize_vector(damping, damping.front())}});
  }

  if (loaded_anchors.empty()) {
    if (error_out != nullptr) {
      *error_out = "Impedance preset YAML does not define any level anchors.";
    }
    return false;
  }

  std::sort(
    loaded_anchors.begin(), loaded_anchors.end(),
    [](const LevelAnchor & lhs, const LevelAnchor & rhs) {
      return lhs.level < rhs.level;
    });

  for (size_t i = 1; i < loaded_anchors.size(); ++i) {
    if (std::abs(loaded_anchors[i - 1].level - loaded_anchors[i].level) < 1e-9) {
      if (error_out != nullptr) {
        *error_out =
          "Duplicate impedance level anchor: " + std::to_string(loaded_anchors[i].level);
      }
      return false;
    }
  }

  loaded_anchor_levels.reserve(loaded_anchors.size());
  for (const auto & anchor : loaded_anchors) {
    loaded_anchor_levels.push_back(anchor.level);
  }

  std::lock_guard<std::mutex> lock(mutex_);
  anchors_ = std::move(loaded_anchors);
  available_anchor_levels_ = std::move(loaded_anchor_levels);
  return true;
}

ImpedanceGains ImpedanceHandler::interpolate_gains(double level) const
{
  if (anchors_.empty()) {
    return gains_;
  }

  if (anchors_.size() == 1 || level <= anchors_.front().level) {
    return anchors_.front().gains;
  }
  if (level >= anchors_.back().level) {
    return anchors_.back().gains;
  }

  for (size_t i = 1; i < anchors_.size(); ++i) {
    const auto & lower = anchors_[i - 1];
    const auto & upper = anchors_[i];
    if (level > upper.level) {
      continue;
    }

    const auto span = upper.level - lower.level;
    const auto alpha = span > 0.0 ? (level - lower.level) / span : 0.0;

    ImpedanceGains interpolated;
    interpolated.stiffness.resize(static_cast<size_t>(joint_count_), 0.0);
    interpolated.damping.resize(static_cast<size_t>(joint_count_), 0.0);
    for (size_t joint_index = 0; joint_index < static_cast<size_t>(joint_count_); ++joint_index) {
      interpolated.stiffness[joint_index] =
        lower.gains.stiffness[joint_index] +
        alpha * (upper.gains.stiffness[joint_index] - lower.gains.stiffness[joint_index]);
      interpolated.damping[joint_index] =
        lower.gains.damping[joint_index] +
        alpha * (upper.gains.damping[joint_index] - lower.gains.damping[joint_index]);
    }
    return interpolated;
  }

  return anchors_.back().gains;
}

std::vector<double> ImpedanceHandler::sanitize_vector(
  const std::vector<double> & in,
  double fallback) const
{
  std::vector<double> out = in;
  const auto target_size = static_cast<size_t>(joint_count_);
  if (out.size() < target_size) {
    out.resize(target_size, fallback);
  } else if (out.size() > target_size) {
    out.resize(target_size);
  }
  return out;
}

}  // namespace joint_impedance_controller
