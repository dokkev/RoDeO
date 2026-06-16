#pragma once

#include <optional>
#include <stdexcept>
#include <string>

#include <Eigen/Geometry>

namespace wbc::math {

inline std::string referenceFrameName(
    const std::optional<std::string>& reference_frame,
    const std::string& default_frame,
    const std::string& fallback = "world") {
  if (reference_frame.has_value() && !reference_frame->empty()) {
    return *reference_frame;
  }
  if (!default_frame.empty()) {
    return default_frame;
  }
  return fallback;
}

template <typename FrameResolver>
Eigen::Isometry3d worldReferenceFrame(
    const std::string& frame_name, FrameResolver&& frame_resolver,
    const std::string& error_prefix = "[MathUtil] Invalid reference_frame") {
  if (frame_name.empty() || frame_name == "world") {
    return Eigen::Isometry3d::Identity();
  }
  try {
    return frame_resolver(frame_name);
  } catch (const std::exception& e) {
    throw std::runtime_error(error_prefix + " '" + frame_name +
                             "': " + e.what());
  }
}

}  // namespace wbc::math
