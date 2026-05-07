/**
 * @file wbc_core/wbc_util/include/wbc_util/ros_path_utils.hpp
 * @brief Doxygen documentation for ros_path_utils module.
 */
#pragma once

#include <string>

namespace wbc::path {

/**
 * @brief Resolve `package://` URI to concrete filesystem path.
 */
std::string ResolvePackageUri(const std::string& uri);

/**
 * @brief Resolve ROS package root directory for a URDF path.
 */
std::string ResolveUrdfPackageRoot(const std::string& urdf_uri,
                                   const std::string& resolved_urdf_path);

/**
 * @brief Resolve relative path using the directory of `base_yaml_path`.
 */
std::string ResolveRelativePath(const std::string& path,
                                const std::string& base_yaml_path);

} // namespace wbc::path
