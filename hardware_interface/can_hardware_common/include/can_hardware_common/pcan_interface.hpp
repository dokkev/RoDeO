#ifndef CAN_HARDWARE_COMMON__PCAN_INTERFACE_HPP_
#define CAN_HARDWARE_COMMON__PCAN_INTERFACE_HPP_

#include <PCANBasic.h>

#include <chrono>
#include <cstdint>
#include <mutex>
#include <string>

namespace pcan_interface
{

class PCANInterface
{
public:
  PCANInterface();
  ~PCANInterface() noexcept;

  PCANInterface(const PCANInterface &) = delete;
  PCANInterface & operator=(const PCANInterface &) = delete;
  PCANInterface(PCANInterface &&) = delete;
  PCANInterface & operator=(PCANInterface &&) = delete;

  TPCANStatus write(const TPCANMsg & tx_frame);
  TPCANStatus read(TPCANMsg & rx_frame, TPCANTimestamp * timestamp = nullptr);
  TPCANStatus read_with_timeout(TPCANMsg & rx_frame, std::chrono::microseconds timeout);
  TPCANStatus get_bus_status();
  TPCANStatus get_value(TPCANParameter parameter, void * buffer, uint32_t buffer_length);

  static std::string format_error(TPCANStatus status);

private:
  static constexpr TPCANHandle kChannelHandle_ = PCAN_USBBUS1;
  static constexpr TPCANBaudrate kChannelBitrate_ = PCAN_BAUD_1M;
  mutable std::mutex io_mutex_;
  int receive_event_fd_ = -1;
};

}  // namespace pcan_interface

#endif  // CAN_HARDWARE_COMMON__PCAN_INTERFACE_HPP_
