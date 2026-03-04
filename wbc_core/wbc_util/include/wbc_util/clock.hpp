/**
 * @file wbc_core/wbc_util/include/wbc_util/clock.hpp
 * @brief Doxygen documentation for clock module.
 */
#pragma once

#include <chrono>

/**
 * @brief Lightweight wall-clock timer utility (milliseconds).
 */
class Clock {
public:
  Clock() {}
  ~Clock() = default;

  void Start() { start_time_ = std::chrono::high_resolution_clock::now(); }

  void Stop() {
    end_time_ = std::chrono::high_resolution_clock::now();
    duration_ = std::chrono::duration_cast<std::chrono::microseconds>(
        end_time_ - start_time_);
  }

  double duration() const { return double(duration_.count()) * 1e-3; }

private:
  std::chrono::microseconds duration_;
  std::chrono::high_resolution_clock::time_point start_time_, end_time_;
};
