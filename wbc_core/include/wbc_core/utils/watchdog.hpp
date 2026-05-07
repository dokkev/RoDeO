/**
 * @file wbc_core/include/wbc_core/utils/watchdog.hpp
 * @brief Generic signal-age watchdog for real-time control loops.
 */
#pragma once

#include <algorithm>

namespace wbc {

/**
 * @brief Tracks the age of a signal and detects stale-command timeouts.
 *
 * Typical RT-loop usage:
 * @code
 *   // Non-RT: on new message
 *   watchdog_.Reset();
 *   current_cmd_ = msg;
 *
 *   // RT OneStep():
 *   watchdog_.Update(sp_->servo_dt_);
 *   if (watchdog_.IsTimeout()) current_cmd_.setZero();
 * @endcode
 *
 * @note All methods are trivially RT-safe (no allocation, no locks).
 */
class Watchdog {
public:
  explicit Watchdog(double timeout = 0.2, double init_age = 999.0)
      : timeout_(std::max(0.0, timeout)), age_(init_age) {}

  void SetTimeout(double timeout) { timeout_ = std::max(0.0, timeout); }
  double GetTimeout() const { return timeout_; }

  void Reset() { age_ = 0.0; }
  void Update(double dt) { if (dt > 0.0) age_ += dt; }

  [[nodiscard]] bool IsTimeout() const { return age_ > timeout_; }
  [[nodiscard]] double GetAge() const { return age_; }

private:
  double timeout_;
  double age_;
};

}  // namespace wbc
