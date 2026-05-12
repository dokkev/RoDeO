#include "can_hardware_common/can_bus.hpp"

#include <cstdio>
#include <exception>
#include <thread>

namespace can_hardware_common
{

namespace
{

std::string hex_u32(uint32_t value)
{
  char buffer[32];
  std::snprintf(buffer, sizeof(buffer), "0x%X", value);
  return buffer;
}

}  // namespace

CanBus::~CanBus() = default;

void CanBus::set_tx_gap(std::chrono::microseconds tx_gap)
{
  std::lock_guard<std::mutex> lock(tx_mutex_);
  tx_gap_ = tx_gap;
}

CanBus::RxPollResult CanBus::poll_rx()
{
  RxPollResult result;
  TPCANMsg rx_frame{};
  const auto dispatch_rx = [this](const TPCANMsg & frame) {
      for (auto & observer : rx_observers_) {
        try {
          observer(frame);
        } catch (const std::exception & error) {
          std::fprintf(stderr, "[CanBus] RX observer threw for CAN ID 0x%X: %s\n", frame.ID, error.what());
        } catch (...) {
          std::fprintf(stderr, "[CanBus] RX observer threw unknown for CAN ID 0x%X\n", frame.ID);
        }
      }
    };

  for (std::size_t index = 0; index < kMaxRxPerPoll; ++index) {
    const TPCANStatus status = channel_.read(rx_frame);
    if (status == PCAN_ERROR_OK) {
      dispatch_rx(rx_frame);
      ++result.processed_frames;
      continue;
    }

    result.status = status;
    return result;
  }

  result.status = PCAN_ERROR_OK;
  return result;
}

CanBus::RxPollResult CanBus::poll_rx_for(
  std::chrono::microseconds budget,
  std::chrono::microseconds idle_sleep)
{
  const auto deadline = std::chrono::steady_clock::now() + budget;

  RxPollResult result;
  while (std::chrono::steady_clock::now() < deadline) {
    const auto rx = poll_rx();
    result.processed_frames += rx.processed_frames;
    result.status = rx.status;

    if (!is_nonfatal_read_status(rx.status)) {
      return result;
    }

    if (rx.processed_frames == 0U && idle_sleep.count() > 0) {
      std::this_thread::sleep_for(idle_sleep);
    }
  }

  return result;
}

CanBus::RxPollResult CanBus::read_frames_for(
  std::vector<TPCANMsg> & rx_frames,
  std::size_t max_frames,
  std::chrono::microseconds budget)
{
  return read_frames_until(rx_frames, max_frames, std::chrono::steady_clock::now() + budget);
}

CanBus::RxPollResult CanBus::read_frames_until(
  std::vector<TPCANMsg> & rx_frames,
  std::size_t max_frames,
  std::chrono::steady_clock::time_point deadline)
{
  rx_frames.clear();

  RxPollResult result;
  if (max_frames == 0U) {
    result.status = PCAN_ERROR_OK;
    return result;
  }

  while (result.processed_frames < max_frames) {
    const auto now = std::chrono::steady_clock::now();
    if (now >= deadline) {
      result.status = PCAN_ERROR_QRCVEMPTY;
      return result;
    }

    TPCANMsg rx_frame{};
    const auto timeout = std::chrono::duration_cast<std::chrono::microseconds>(deadline - now);
    const TPCANStatus status = channel_.read_with_timeout(rx_frame, timeout);
    if (status != PCAN_ERROR_OK) {
      result.status = status;
      return result;
    }

    rx_frames.push_back(rx_frame);
    ++result.processed_frames;
  }

  result.status = PCAN_ERROR_OK;
  return result;
}

void CanBus::add_rx_observer(RxObserver observer)
{
  rx_observers_.push_back(std::move(observer));
}

void CanBus::clear_rx_observers()
{
  rx_observers_.clear();
}

TPCANStatus CanBus::send_tx_frame(const TPCANMsg & tx_frame)
{
  std::lock_guard<std::mutex> lock(tx_mutex_);

  const auto now = std::chrono::steady_clock::now();
  if (last_tx_time_ != std::chrono::steady_clock::time_point{} &&
      tx_gap_.count() > 0 &&
      now - last_tx_time_ < tx_gap_)
  {
    return PCAN_ERROR_QXMTFULL;
  }

  const TPCANStatus status = channel_.write(tx_frame);
  if (status == PCAN_ERROR_OK) {
    last_tx_time_ = now;
  }

  return status;
}

CanBus::BlockingTxResult CanBus::send_tx_frame_blocking(
  const TPCANMsg & tx_frame,
  std::chrono::microseconds timeout)
{
  BlockingTxResult result;

  const auto now = std::chrono::steady_clock::now();
  const auto next_send = next_tx_time();
  if (next_send > now) {
    result.required_wait =
      std::chrono::duration_cast<std::chrono::microseconds>(next_send - now);
    if (result.required_wait > timeout) {
      result.status = PCAN_ERROR_QXMTFULL;
      result.timed_out = true;
      return result;
    }
    std::this_thread::sleep_until(next_send);
  }

  result.status = send_tx_frame(tx_frame);
  return result;
}

std::chrono::steady_clock::time_point CanBus::next_tx_time() const
{
  std::lock_guard<std::mutex> lock(tx_mutex_);
  if (last_tx_time_ == std::chrono::steady_clock::time_point{}) {
    return {};
  }
  return last_tx_time_ + tx_gap_;
}

bool CanBus::is_nonfatal_read_status(TPCANStatus status)
{
  return status == PCAN_ERROR_OK || status == PCAN_ERROR_QRCVEMPTY;
}

CanBus::BusDiagnostics CanBus::get_diagnostics()
{
  BusDiagnostics diagnostics;
  diagnostics.bus_status = channel_.get_bus_status();
  (void)channel_.get_value(
    PCAN_CHANNEL_CONDITION,
    &diagnostics.channel_condition,
    sizeof(diagnostics.channel_condition));
  (void)channel_.get_value(
    PCAN_RECEIVE_STATUS,
    &diagnostics.receive_status,
    sizeof(diagnostics.receive_status));
  return diagnostics;
}

std::string CanBus::bus_status_string(TPCANStatus status)
{
  if (status == PCAN_ERROR_OK) {
    return "OK";
  }

  std::string result;
  auto append = [&](const char * label) {
      if (!result.empty()) {
        result += " | ";
      }
      result += label;
    };

  if (status & PCAN_ERROR_BUSLIGHT) { append("BUS_LIGHT"); }
  if (status & PCAN_ERROR_BUSHEAVY) { append("BUS_HEAVY"); }
  if (status & PCAN_ERROR_BUSPASSIVE) { append("BUS_PASSIVE"); }
  if (status & PCAN_ERROR_BUSOFF) { append("BUS_OFF"); }
  if (status & PCAN_ERROR_XMTFULL) { append("TX_BUFFER_FULL"); }
  if (status & PCAN_ERROR_OVERRUN) { append("RX_OVERRUN"); }
  if (status & PCAN_ERROR_QOVERRUN) { append("RX_QUEUE_OVERRUN"); }
  if (status & PCAN_ERROR_QXMTFULL) { append("TX_QUEUE_FULL"); }
  if (result.empty()) {
    char buffer[32];
    std::snprintf(buffer, sizeof(buffer), "0x%X", status);
    result = buffer;
  }
  return result;
}

std::string CanBus::channel_condition_string(uint32_t channel_condition)
{
  switch (channel_condition) {
    case PCAN_CHANNEL_UNAVAILABLE:
      return "UNAVAILABLE";
    case PCAN_CHANNEL_AVAILABLE:
      return "AVAILABLE";
    case PCAN_CHANNEL_OCCUPIED:
      return "OCCUPIED";
    case PCAN_CHANNEL_PCANVIEW:
      return "PCANVIEW";
    default:
      return hex_u32(channel_condition);
  }
}

std::string CanBus::receive_status_string(uint32_t receive_status)
{
  switch (receive_status) {
    case PCAN_PARAMETER_OFF:
      return "OFF";
    case PCAN_PARAMETER_ON:
      return "ON";
    default:
      return hex_u32(receive_status);
  }
}

std::string CanBus::format_diagnostics(const BusDiagnostics & diagnostics)
{
  return
    "bus=" + bus_status_string(diagnostics.bus_status) +
    ", channel=" + channel_condition_string(diagnostics.channel_condition) +
    ", receive=" + receive_status_string(diagnostics.receive_status);
}

}  // namespace can_hardware_common
