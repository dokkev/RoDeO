//
// Copyright (c) 2026
//
// Logging utilities.
//

#pragma once

#include <iostream>
#include <string>

namespace wbc::log {

/// Severity levels for log messages.
enum class Level { kDebug, kInfo, kWarn, kError };

/// Simple log macro (can be replaced with spdlog or ROS logging later).
inline void Log(Level level, const std::string& msg) {
  const char* prefix = "";
  switch (level) {
    case Level::kDebug: prefix = "[DEBUG] "; break;
    case Level::kInfo:  prefix = "[INFO]  "; break;
    case Level::kWarn:  prefix = "[WARN]  "; break;
    case Level::kError: prefix = "[ERROR] "; break;
  }
  std::cerr << prefix << msg << std::endl;
}

}  // namespace wbc::log
