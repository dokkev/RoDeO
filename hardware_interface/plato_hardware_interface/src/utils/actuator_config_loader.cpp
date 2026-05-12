#include "plato_hardware_interface/utils/actuator_config_loader.hpp"

#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <yaml-cpp/yaml.h>

#include "plato_hardware_interface/plato_layout.hpp"

namespace plato_actuator
{

namespace
{

std::string require_scalar(const YAML::Node & node, const char * key)
{
  const auto value = node[key];
  if (!value || !value.IsScalar()) {
    throw std::runtime_error(std::string("Missing scalar field: ") + key);
  }
  return value.as<std::string>();
}

uint8_t parse_u8(const YAML::Node & node, const char * key)
{
  const auto raw = require_scalar(node, key);
  const auto parsed = std::stoul(raw, nullptr, 0);
  if (parsed > UINT8_MAX) {
    throw std::runtime_error(std::string("Value out of range for ") + key + ": " + raw);
  }
  return static_cast<uint8_t>(parsed);
}

uint32_t parse_u32_or_default(const YAML::Node & node, const char * key, uint32_t default_value)
{
  const auto value = node[key];
  if (!value) {
    return default_value;
  }
  if (!value.IsScalar()) {
    throw std::runtime_error(std::string("Missing scalar field: ") + key);
  }
  return static_cast<uint32_t>(std::stoul(value.as<std::string>(), nullptr, 0));
}

int8_t parse_direction(const YAML::Node & node)
{
  const auto raw = require_scalar(node, "direction");
  const auto parsed = std::stoi(raw, nullptr, 0);
  if (parsed != -1 && parsed != 1) {
    throw std::runtime_error("direction must be either -1 or 1");
  }
  return static_cast<int8_t>(parsed);
}

float parse_float(const YAML::Node & node, const char * key)
{
  const auto value = node[key];
  if (!value || !value.IsScalar()) {
    throw std::runtime_error(std::string("Missing scalar field: ") + key);
  }
  return value.as<float>();
}

float parse_float_or_default(const YAML::Node & node, const char * key, float default_value)
{
  const auto value = node[key];
  if (!value) {
    return default_value;
  }
  if (!value.IsScalar()) {
    throw std::runtime_error(std::string("Missing scalar field: ") + key);
  }
  return value.as<float>();
}

bool parse_bool_or_default(const YAML::Node & node, const char * key, bool default_value)
{
  const auto value = node[key];
  if (!value) {
    return default_value;
  }
  if (!value.IsScalar()) {
    throw std::runtime_error(std::string("Missing scalar field: ") + key);
  }
  return value.as<bool>();
}

actuator::Limits parse_limits(const YAML::Node & node)
{
  const auto limits_node = node["joint_pos_limit_rad"];
  if (!limits_node || !limits_node.IsMap()) {
    throw std::runtime_error("Missing joint_pos_limit_rad map");
  }

  actuator::Limits limits;
  limits.position_limit_min = parse_float(limits_node, "min");
  limits.position_limit_max = parse_float(limits_node, "max");
  return limits;
}

Config parse_config(const YAML::Node & node)
{
  Config config;
  config.static_config.can_tx_id = parse_u8(node, "tx_id");
  config.static_config.can_rx_id = parse_u8(node, "rx_id");
  config.static_config.direction = parse_direction(node);
  config.static_config.torque_constant = parse_float(node, "torque_constant");
  config.static_config.gear_ratio = parse_float(node, "gear_ratio");
  config.static_config.limits = parse_limits(node);
  config.static_config.servo_current_milliamps =
    parse_u32_or_default(node, "servo_current_milliamps", 0);
  config.static_config.soft_stop_enabled = parse_bool_or_default(node, "soft_stop_enabled", true);
  config.static_config.effort_limit_nm = parse_float_or_default(
    node, "effort_limit_nm", std::numeric_limits<float>::infinity());
  return config;
}

void validate_can_ids(const std::vector<Config> & configs)
{
  for (std::size_t i = 0; i < configs.size(); ++i) {
    for (std::size_t j = i + 1; j < configs.size(); ++j) {
      const bool allowed_shared_dynamixel_mcu =
        plato_hand::layout::is_thumb_servo(i) && plato_hand::layout::is_thumb_servo(j);
      if (!allowed_shared_dynamixel_mcu &&
        configs[i].static_config.can_tx_id == configs[j].static_config.can_tx_id)
      {
        throw std::runtime_error("Duplicate tx_id found in actuator config");
      }
      if (!allowed_shared_dynamixel_mcu &&
        configs[i].static_config.can_rx_id == configs[j].static_config.can_rx_id)
      {
        throw std::runtime_error("Duplicate rx_id found in actuator config");
      }
    }
  }
}

}  // namespace

const std::array<const char *, plato_hand::layout::kNumActuators> &
expected_plato_actuator_names()
{
  return plato_hand::layout::kActuatorNames;
}

std::vector<Config> load_plato_actuator_configs()
{
  const auto yaml_path =
    ament_index_cpp::get_package_share_directory("plato_hardware_interface") +
    "/config/plato_actuators.yaml";
  return load_plato_actuator_configs(yaml_path);
}

std::vector<Config> load_plato_actuator_configs(const std::string & yaml_path)
{
  const auto root = YAML::LoadFile(yaml_path);
  const auto actuators_node = root["actuators"];
  if (!actuators_node || !actuators_node.IsSequence()) {
    throw std::runtime_error("Expected 'actuators' sequence in " + yaml_path);
  }

  std::unordered_map<std::string, Config> configs_by_name;
  configs_by_name.reserve(actuators_node.size());

  for (const auto & actuator_node : actuators_node) {
    const auto name = require_scalar(actuator_node, "name");
    const auto inserted = configs_by_name.emplace(name, parse_config(actuator_node));
    if (!inserted.second) {
      throw std::runtime_error("Duplicate actuator name in YAML: " + name);
    }
  }

  if (configs_by_name.size() != plato_hand::layout::kActuatorNames.size()) {
    throw std::runtime_error(
      "Expected exactly " + std::to_string(plato_hand::layout::kActuatorNames.size()) +
      " actuators in YAML");
  }

  std::vector<Config> configs;
  configs.reserve(plato_hand::layout::kActuatorNames.size());

  for (std::size_t i = 0; i < plato_hand::layout::kActuatorNames.size(); ++i) {
    const auto * expected_name = plato_hand::layout::kActuatorNames[i];
    const auto it = configs_by_name.find(expected_name);
    if (it == configs_by_name.end()) {
      throw std::runtime_error(std::string("Missing actuator entry in YAML: ") + expected_name);
    }
    auto config = it->second;
    if (plato_hand::layout::is_thumb_servo(i)) {
      config.static_config.protocol_kind = ProtocolKind::kDynamixelBridge;
      config.static_config.dynamixel_servo_id = plato_hand::layout::dynamixel_id_for_index(i);
    }
    configs.push_back(config);
  }

  validate_can_ids(configs);
  return configs;
}

}  // namespace plato_actuator
