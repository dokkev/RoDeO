#include "can_hardware_common/core/can_frame_executor.hpp"

#include <utility>

namespace can_hardware_common::core
{

CanFrameExecutor::CanFrameExecutor(CanBus & transport, rclcpp::Logger logger)
: transport_(transport),
  logger_(std::move(logger))
{
}

bool CanFrameExecutor::poll_rx()
{
  const auto rx = transport_.poll_rx();
  return update_read_status_(rx.status);
}

bool CanFrameExecutor::poll_rx_for(std::chrono::microseconds budget)
{
  const auto rx = transport_.poll_rx_for(budget);
  return update_read_status_(rx.status);
}

bool CanFrameExecutor::send_frame_blocking(
  const TPCANMsg & frame,
  std::chrono::microseconds timeout)
{
  const auto tx = transport_.send_tx_frame_blocking(frame, timeout);
  if (tx.ok()) {
    return true;
  }

  if (tx.timed_out) {
    RCLCPP_WARN_THROTTLE(
      logger_,
      throttle_clock_,
      1000,
      "CAN direct TX pacing timeout on ID 0x%X (wait=%ldus)",
      frame.ID,
      static_cast<long>(tx.required_wait.count()));
    return false;
  }

  if (tx.status == PCAN_ERROR_QXMTFULL) {
    RCLCPP_WARN_THROTTLE(
      logger_,
      throttle_clock_,
      1000,
      "CAN direct TX dropped by pacing on ID 0x%X",
      frame.ID);
    return false;
  }

  RCLCPP_WARN_THROTTLE(
    logger_,
    throttle_clock_,
    1000,
    "CAN direct TX error on ID 0x%X: status 0x%X",
    frame.ID,
    tx.status);
  return false;
}

bool CanFrameExecutor::execute_direct_frames(
  const std::vector<TPCANMsg> & frames,
  std::chrono::microseconds timeout)
{
  for (const auto & frame : frames) {
    if (!send_frame_blocking(frame, timeout)) {
      return false;
    }
  }

  return true;
}

void CanFrameExecutor::mark_rx_frame()
{
  std::lock_guard<std::mutex> lock(rx_mutex_);
  last_rx_time_ = SteadyClock::now();
  has_observed_rx_ = true;
  ++rx_frame_count_;
}

bool CanFrameExecutor::has_fresh_rx(std::chrono::microseconds stale_timeout) const
{
  std::lock_guard<std::mutex> lock(rx_mutex_);
  if (!has_observed_rx_) {
    return false;
  }
  return (SteadyClock::now() - last_rx_time_) <= stale_timeout;
}

std::size_t CanFrameExecutor::rx_frame_count() const
{
  std::lock_guard<std::mutex> lock(rx_mutex_);
  return rx_frame_count_;
}

bool CanFrameExecutor::update_read_status_(TPCANStatus status)
{
  const bool healthy = CanBus::is_nonfatal_read_status(status);
  last_rx_healthy_.store(healthy);
  if (!healthy) {
    RCLCPP_WARN_THROTTLE(
      logger_,
      throttle_clock_,
      1000,
      "CAN receive error: status 0x%X",
      status);
  }

  return healthy;
}

}  // namespace can_hardware_common::core
