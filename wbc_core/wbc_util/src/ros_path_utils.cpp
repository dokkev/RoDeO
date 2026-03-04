/**
 * @file wbc_core/wbc_util/src/ros_path_utils.cpp
 * @brief Doxygen documentation for ros_path_utils module.
 */
#include "wbc_util/ros_path_utils.hpp"

#include <ament_index_cpp/get_package_share_directory.hpp>

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <utility>

namespace wbc::path {
namespace {

constexpr const char kPackagePrefix[] = "package://";

bool IsPackageUri(const std::string& uri) {
  return uri.rfind(kPackagePrefix, 0) == 0U;
}

std::pair<std::string, std::string> SplitPackageUri(const std::string& uri) {
  const std::string suffix = uri.substr(sizeof(kPackagePrefix) - 1U);
  const std::size_t slash = suffix.find('/');
  const std::string package_name =
      (slash == std::string::npos) ? suffix : suffix.substr(0, slash);
  const std::string relative =
      (slash == std::string::npos) ? std::string() : suffix.substr(slash + 1U);

  if (package_name.empty()) {
    throw std::runtime_error("[ros_path_utils] Invalid package URI: " + uri);
  }
  return std::make_pair(package_name, relative);
}

bool UsesMeshesPackageAlias(const std::filesystem::path& urdf_path) {
  std::ifstream in(urdf_path);
  if (!in.is_open()) {
    return false;
  }

  std::string line;
  while (std::getline(in, line)) {
    if (line.find("package://meshes/") != std::string::npos) {
      return true;
    }
  }
  return false;
}

} // namespace

std::string ResolvePackageUri(const std::string& uri) {
  if (!IsPackageUri(uri)) {
    return uri;
  }

  const auto [package_name, relative] = SplitPackageUri(uri);
  const std::filesystem::path share_dir =
      ament_index_cpp::get_package_share_directory(package_name);
  if (relative.empty()) {
    return share_dir.string();
  }
  return (share_dir / relative).string();
}

std::string ResolveUrdfPackageRoot(const std::string& urdf_uri,
                                   const std::string& resolved_urdf_path) {
  const std::filesystem::path urdf_path(ResolvePackageUri(resolved_urdf_path));
  if (urdf_path.empty()) {
    throw std::runtime_error(
        "[ros_path_utils] URDF path is empty while resolving package root.");
  }

  const bool uses_mesh_alias = UsesMeshesPackageAlias(urdf_path);

  if (IsPackageUri(urdf_uri)) {
    const auto parsed = SplitPackageUri(urdf_uri);
    const std::string& package_name = parsed.first;
    const std::filesystem::path share_dir =
        ament_index_cpp::get_package_share_directory(package_name);
    if (uses_mesh_alias) {
      return share_dir.string();
    }
    return share_dir.parent_path().string();
  }

  if (uses_mesh_alias) {
    const std::filesystem::path urdf_dir = urdf_path.parent_path();
    if (!urdf_dir.empty()) {
      return urdf_dir.parent_path().string();
    }
  }

  std::filesystem::path prefix;
  for (const auto& part : urdf_path) {
    prefix /= part;
    if (part == "share") {
      return prefix.string();
    }
  }

  return urdf_path.parent_path().string();
}

std::string ResolveRelativePath(const std::string& path,
                                const std::string& base_yaml_path) {
  if (path.empty()) {
    throw std::runtime_error(
        "[ros_path_utils] Cannot resolve an empty path.");
  }

  const std::string resolved = ResolvePackageUri(path);
  const std::filesystem::path target_path(resolved);
  if (target_path.is_absolute()) {
    return target_path.lexically_normal().string();
  }

  if (base_yaml_path.empty()) {
    throw std::runtime_error(
        "[ros_path_utils] base_yaml_path is empty for relative path '" + path +
        "'.");
  }

  const std::filesystem::path base_path(ResolvePackageUri(base_yaml_path));
  return (base_path.parent_path() / target_path).lexically_normal().string();
}

} // namespace wbc::path
