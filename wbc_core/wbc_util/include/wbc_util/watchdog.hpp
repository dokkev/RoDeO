/**
 * @file wbc_util/include/wbc_util/watchdog.hpp
 * @brief Generic signal-age watchdog for real-time control loops.
 */
#pragma once

#include <algorithm>

namespace wbc {

/**
 * @brief Tracks the age of a signal and detects stale-command timeouts.
 *
 * Separates two distinct responsibilities:
 *   - Age logic:      how old is the current command? (incremented by Update())
 *   - Watchdog logic: has it exceeded the safe limit? (checked by IsTimeout())
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
  /**
   * @param timeout   Timeout threshold [s].  Watchdog fires when age > timeout.
   * @param init_age  Initial age [s].  Default 999 s starts the watchdog in
   *                  the fired state — the caller must receive at least one
   *                  valid signal before commands are applied.
   */
  explicit Watchdog(double timeout = 0.2, double init_age = 999.0)
      : timeout_(std::max(0.0, timeout)), age_(init_age) {}

  /** @brief Override the timeout threshold. */
  void SetTimeout(double timeout) { timeout_ = std::max(0.0, timeout); }

  /** @brief Returns the current timeout threshold [s]. */
  double GetTimeout() const { return timeout_; }

  /**
   * @brief Signal that a fresh command was received — resets age to zero.
   *
   * Call this whenever UpdateCommand() sees a new message.
   */
  void Reset() { age_ = 0.0; }

  /**
   * @brief Advance the age by one control tick.
   *
   * Call this once per control tick (e.g. inside OneStep()).
   * @param dt Control step size [s], e.g. servo_dt_.
   */
  void Update(double dt) { if (dt > 0.0) age_ += dt; }

  /**
   * @brief Returns true when the signal is stale and commands should be zeroed.
   *
   * Fires when age > timeout_.
   */
  [[nodiscard]] bool IsTimeout() const { return age_ > timeout_; }

  /** @brief Current signal age [s]. */
  [[nodiscard]] double GetAge() const { return age_; }

private:
  double timeout_;
  double age_;
};

}  // namespace wbc
