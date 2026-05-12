#ifndef CAN_HARDWARE_COMMON__CORE__CAN_FRAME_EXECUTOR_HPP_
#define CAN_HARDWARE_COMMON__CORE__CAN_FRAME_EXECUTOR_HPP_

#include <PCANBasic.h>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <mutex>
#include <vector>

#include <rclcpp/rclcpp.hpp>

#include "can_hardware_common/can_bus.hpp"

namespace can_hardware_common::core
{

class CanFrameExecutor
{
public:
  CanFrameExecutor(CanBus & transport, rclcpp::Logger logger);

  bool poll_rx();
  bool poll_rx_for(std::chrono::microseconds budget);
  bool send_frame_blocking(const TPCANMsg & frame, std::chrono::microseconds timeout);
  bool execute_direct_frames(
    const std::vector<TPCANMsg> & frames,
    std::chrono::microseconds timeout);

  void mark_rx_frame();
  bool has_fresh_rx(std::chrono::microseconds stale_timeout) const;
  bool transport_healthy() const { return last_rx_healthy_.load(); }
  std::size_t rx_frame_count() const;

private:
  using SteadyClock = std::chrono::steady_clock;

  bool update_read_status_(TPCANStatus status);

  CanBus & transport_;
  rclcpp::Logger logger_;
  rclcpp::Clock throttle_clock_{RCL_STEADY_TIME};
  mutable std::mutex rx_mutex_;
  SteadyClock::time_point last_rx_time_{};
  std::size_t rx_frame_count_ = 0;
  bool has_observed_rx_ = false;
  std::atomic_bool last_rx_healthy_{true};
};

}  // namespace can_hardware_common::core

#endif  // CAN_HARDWARE_COMMON__CORE__CAN_FRAME_EXECUTOR_HPP_
