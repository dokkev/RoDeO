#ifndef CAN_HARDWARE_COMMON__CAN_BUS_HPP_
#define CAN_HARDWARE_COMMON__CAN_BUS_HPP_

#include <PCANBasic.h>

#include <chrono>
#include <cstddef>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

#include "can_hardware_common/pcan_interface.hpp"

namespace can_hardware_common
{

class CanBus
{
public:
  using RxObserver = std::function<void(const TPCANMsg & rx_frame)>;

  struct RxPollResult
  {
    std::size_t processed_frames = 0;
    TPCANStatus status = PCAN_ERROR_QRCVEMPTY;
  };

  struct BlockingTxResult
  {
    TPCANStatus status = PCAN_ERROR_OK;
    bool timed_out = false;
    std::chrono::microseconds required_wait{0};

    bool ok() const { return status == PCAN_ERROR_OK && !timed_out; }
  };

  struct BusDiagnostics
  {
    TPCANStatus bus_status = PCAN_ERROR_OK;
    uint32_t channel_condition = 0;
    uint32_t receive_status = 0;
  };

  CanBus() = default;
  ~CanBus();

  CanBus(const CanBus &) = delete;
  CanBus & operator=(const CanBus &) = delete;
  CanBus(CanBus &&) = delete;
  CanBus & operator=(CanBus &&) = delete;

  void set_tx_gap(std::chrono::microseconds tx_gap);
  RxPollResult poll_rx();
  RxPollResult poll_rx_for(
    std::chrono::microseconds budget,
    std::chrono::microseconds idle_sleep = std::chrono::microseconds(500));
  RxPollResult read_frames_for(
    std::vector<TPCANMsg> & rx_frames,
    std::size_t max_frames,
    std::chrono::microseconds budget);
  RxPollResult read_frames_until(
    std::vector<TPCANMsg> & rx_frames,
    std::size_t max_frames,
    std::chrono::steady_clock::time_point deadline);

  void add_rx_observer(RxObserver observer);
  void clear_rx_observers();

  TPCANStatus send_tx_frame(const TPCANMsg & tx_frame);
  BlockingTxResult send_tx_frame_blocking(
    const TPCANMsg & tx_frame,
    std::chrono::microseconds timeout);
  std::chrono::steady_clock::time_point next_tx_time() const;

  static bool is_nonfatal_read_status(TPCANStatus status);
  BusDiagnostics get_diagnostics();
  static std::string bus_status_string(TPCANStatus status);
  static std::string channel_condition_string(uint32_t channel_condition);
  static std::string receive_status_string(uint32_t receive_status);
  static std::string format_diagnostics(const BusDiagnostics & diagnostics);

private:
  static constexpr std::size_t kMaxRxPerPoll = 30U;

  pcan_interface::PCANInterface channel_;
  std::vector<RxObserver> rx_observers_;

  mutable std::mutex tx_mutex_;
  std::chrono::microseconds tx_gap_{0};
  std::chrono::steady_clock::time_point last_tx_time_{};
};

}  // namespace can_hardware_common

#endif  // CAN_HARDWARE_COMMON__CAN_BUS_HPP_
