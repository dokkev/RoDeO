#include "plato_hardware_interface/utils/actuator_offset_loader.hpp"

#include <fstream>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <yaml-cpp/yaml.h>

#include "plato_hardware_interface/utils/actuator_config_loader.hpp"

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

float require_float(const YAML::Node & node, const char * key)
{
  const auto value = node[key];
  if (!value || !value.IsScalar()) {
    throw std::runtime_error(std::string("Missing scalar field: ") + key);
  }
  return value.as<float>();
}

void validate_offsets_size(const PositionOffsets & offsets)
{
  if (offsets.size() != expected_plato_actuator_names().size()) {
    throw std::runtime_error(
      "Expected exactly " + std::to_string(expected_plato_actuator_names().size()) +
      " actuator offsets");
  }
}

}  // namespace

PositionOffsets load_plato_actuator_position_offsets()
{
  const auto yaml_path =
    ament_index_cpp::get_package_share_directory("plato_hardware_interface") +
    "/config/plato_actuator_offsets.yaml";
  return load_plato_actuator_position_offsets(yaml_path);
}

PositionOffsets load_plato_actuator_position_offsets(const std::string & yaml_path)
{
  const auto root = YAML::LoadFile(yaml_path);
  const auto offsets_node = root["offsets"];
  if (!offsets_node || !offsets_node.IsSequence()) {
    throw std::runtime_error("Expected 'offsets' sequence in " + yaml_path);
  }

  std::unordered_map<std::string, float> offsets_by_name;
  offsets_by_name.reserve(offsets_node.size());

  for (const auto & offset_node : offsets_node) {
    const auto name = require_scalar(offset_node, "name");
    const auto offset = require_float(offset_node, "offset");
    const auto inserted = offsets_by_name.emplace(name, offset);
    if (!inserted.second) {
      throw std::runtime_error("Duplicate actuator offset entry in YAML: " + name);
    }
  }

  PositionOffsets offsets;
  offsets.reserve(expected_plato_actuator_names().size());

  for (const auto * expected_name : expected_plato_actuator_names()) {
    const auto it = offsets_by_name.find(expected_name);
    if (it == offsets_by_name.end()) {
      throw std::runtime_error(
        std::string("Missing actuator offset entry in YAML: ") + expected_name);
    }
    offsets.push_back(it->second);
  }

  return offsets;
}

void save_plato_actuator_position_offsets(const PositionOffsets & offsets)
{
  const auto yaml_path =
    ament_index_cpp::get_package_share_directory("plato_hardware_interface") +
    "/config/plato_actuator_offsets.yaml";
  save_plato_actuator_position_offsets(offsets, yaml_path);
}

void save_plato_actuator_position_offsets(
  const PositionOffsets & offsets,
  const std::string & yaml_path)
{
  validate_offsets_size(offsets);

  YAML::Emitter out;
  out << YAML::BeginMap;
  out << YAML::Key << "offsets" << YAML::Value << YAML::BeginSeq;

  for (size_t i = 0; i < offsets.size(); ++i) {
    out << YAML::BeginMap;
    out << YAML::Key << "name" << YAML::Value << expected_plato_actuator_names()[i];
    out << YAML::Key << "offset" << YAML::Value << offsets[i];
    out << YAML::EndMap;
  }

  out << YAML::EndSeq;
  out << YAML::EndMap;

  std::ofstream file(yaml_path, std::ios::out | std::ios::trunc);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open actuator offset YAML for writing: " + yaml_path);
  }
  file << out.c_str() << '\n';
}

}  // namespace plato_actuator
