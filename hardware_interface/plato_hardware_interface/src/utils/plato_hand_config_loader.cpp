#include "plato_hardware_interface/utils/plato_hand_config_loader.hpp"

#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>

#include <ament_index_cpp/get_package_share_directory.hpp>

#include "plato_hardware_interface/utils/actuator_config_loader.hpp"
#include "plato_hardware_interface/utils/actuator_offset_loader.hpp"
#include "plato_hardware_interface/utils/linkage_config_loader.hpp"

namespace plato_hand
{

namespace
{

std::string default_config_dir()
{
  return ament_index_cpp::get_package_share_directory("plato_hardware_interface") + "/config";
}

bool is_path_writable(const std::filesystem::path & path)
{
  std::error_code ec;
  const auto status = std::filesystem::status(path, ec);
  if (ec) {
    return false;
  }

  const auto perms = status.permissions();
  using perms_t = std::filesystem::perms;
  const auto write_mask = perms_t::owner_write | perms_t::group_write | perms_t::others_write;
  return (perms & write_mask) != perms_t::none;
}

std::string resolve_offset_save_path(const std::filesystem::path & default_offset_path)
{
  if (is_path_writable(default_offset_path)) {
    return default_offset_path.string();
  }

  const char * home = std::getenv("HOME");
  if (home == nullptr || std::string(home).empty()) {
    return default_offset_path.string();
  }

  std::error_code ec;
  const auto fallback_dir =
    std::filesystem::path(home) / ".config" / "plato_hardware_interface";
  std::filesystem::create_directories(fallback_dir, ec);
  if (ec) {
    return default_offset_path.string();
  }

  const auto fallback_path = fallback_dir / default_offset_path.filename();
  if (!std::filesystem::exists(fallback_path, ec)) {
    ec.clear();
    std::filesystem::copy_file(
      default_offset_path,
      fallback_path,
      std::filesystem::copy_options::overwrite_existing,
      ec);
  }

  return fallback_path.string();
}

}  // namespace

PlatoHandConfig load_default_plato_hand_config()
{
  return load_plato_hand_config(default_config_dir());
}

PlatoHandConfig load_plato_hand_config(const std::string & config_dir)
{
  const std::filesystem::path config_path(config_dir);
  const auto default_offset_path = config_path / "plato_actuator_offsets.yaml";
  auto actuator_configs =
    plato_actuator::load_plato_actuator_configs((config_path / "plato_actuators.yaml").string());
  const auto actuator_offsets =
    plato_actuator::load_plato_actuator_position_offsets(
    default_offset_path.string());

  if (actuator_configs.size() != actuator_offsets.size()) {
    throw std::runtime_error("Actuator config and actuator offset YAML size mismatch");
  }

  for (size_t i = 0; i < actuator_configs.size(); ++i) {
    actuator_configs[i].position_offset = actuator_offsets[i];
  }

  return PlatoHandConfig{
    std::move(actuator_configs),
    FiveBarLinkage::load_plato_linkage_config((config_path / "plato_linkage.yaml").string()),
    resolve_offset_save_path(default_offset_path),
  };
}

}  // namespace plato_hand
