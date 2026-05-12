#include "plato_hardware_interface/utils/linkage_config_loader.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <yaml-cpp/yaml.h>

namespace FiveBarLinkage
{

namespace
{
constexpr float kPi = 3.14159265358979323846f;
constexpr float kDegreesToRadians = kPi / 180.0f;

float require_float(const YAML::Node & node, const char * key)
{
  const auto value = node[key];
  if (!value || !value.IsScalar()) {
    throw std::runtime_error(std::string("Missing scalar field in linkage config: ") + key);
  }
  return value.as<float>();
}

}  // namespace

FiveBarLinkageConfig load_plato_linkage_config()
{
  const auto yaml_path =
    ament_index_cpp::get_package_share_directory("plato_hardware_interface") +
    "/config/plato_linkage.yaml";
  return load_plato_linkage_config(yaml_path);
}

FiveBarLinkageConfig load_plato_linkage_config(const std::string & yaml_path)
{
  const auto root = YAML::LoadFile(yaml_path);
  const auto node = root["linkage"];
  if (!node || !node.IsMap()) {
    throw std::runtime_error("Expected 'linkage' map in " + yaml_path);
  }

  return FiveBarLinkageConfig{
    require_float(node, "L1"),
    require_float(node, "L2"),
    require_float(node, "L3"),
    require_float(node, "L4"),
    require_float(node, "L5"),
    require_float(node, "pip_motor_zero_angle_offset_deg") * kDegreesToRadians,
    require_float(node, "eef_length"),
  };
}

}  // namespace FiveBarLinkage
