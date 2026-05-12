#include "can_hardware_common/pcan_interface.hpp"

#include <poll.h>

#include <cstdio>
#include <stdexcept>
#include <string>
#include <thread>

namespace pcan_interface
{

namespace
{
constexpr WORD kPcanLangEnglish = 0x09;

std::string handle_name(TPCANHandle handle)
{
  switch (handle) {
    case PCAN_PCIBUS1:
    case PCAN_PCIBUS2:
    case PCAN_PCIBUS3:
    case PCAN_PCIBUS4:
    case PCAN_PCIBUS5:
    case PCAN_PCIBUS6:
    case PCAN_PCIBUS7:
    case PCAN_PCIBUS8:
    case PCAN_PCIBUS9:
    case PCAN_PCIBUS10:
    case PCAN_PCIBUS11:
    case PCAN_PCIBUS12:
    case PCAN_PCIBUS13:
    case PCAN_PCIBUS14:
    case PCAN_PCIBUS15:
    case PCAN_PCIBUS16:
      return "PCAN_PCI";
    case PCAN_USBBUS1:
    case PCAN_USBBUS2:
    case PCAN_USBBUS3:
    case PCAN_USBBUS4:
    case PCAN_USBBUS5:
    case PCAN_USBBUS6:
    case PCAN_USBBUS7:
    case PCAN_USBBUS8:
    case PCAN_USBBUS9:
    case PCAN_USBBUS10:
    case PCAN_USBBUS11:
    case PCAN_USBBUS12:
    case PCAN_USBBUS13:
    case PCAN_USBBUS14:
    case PCAN_USBBUS15:
    case PCAN_USBBUS16:
      return "PCAN_USB";
    case PCAN_LANBUS1:
    case PCAN_LANBUS2:
    case PCAN_LANBUS3:
    case PCAN_LANBUS4:
    case PCAN_LANBUS5:
    case PCAN_LANBUS6:
    case PCAN_LANBUS7:
    case PCAN_LANBUS8:
    case PCAN_LANBUS9:
    case PCAN_LANBUS10:
    case PCAN_LANBUS11:
    case PCAN_LANBUS12:
    case PCAN_LANBUS13:
    case PCAN_LANBUS14:
    case PCAN_LANBUS15:
    case PCAN_LANBUS16:
      return "PCAN_LAN";
    default:
      return "UNKNOWN";
  }
}

std::string format_channel_name(TPCANHandle handle)
{
  const BYTE channel_number =
    (handle < 0x100) ? static_cast<BYTE>(handle & 0x0F) : static_cast<BYTE>(handle & 0xFF);
  char buffer[64] = {};
  std::snprintf(
    buffer, sizeof(buffer), "%s %u (%Xh)", handle_name(handle).c_str(), channel_number, handle);
  return buffer;
}

std::string bitrate_to_string(TPCANBaudrate bitrate)
{
  switch (bitrate) {
    case PCAN_BAUD_1M: return "1 MBit/sec";
    case PCAN_BAUD_800K: return "800 kBit/sec";
    case PCAN_BAUD_500K: return "500 kBit/sec";
    case PCAN_BAUD_250K: return "250 kBit/sec";
    case PCAN_BAUD_125K: return "125 kBit/sec";
    default: return "Unknown Bitrate";
  }
}
}  // namespace

PCANInterface::PCANInterface()
{
  const TPCANStatus init_status = CAN_Initialize(kChannelHandle_, kChannelBitrate_);
  if (init_status != PCAN_ERROR_OK) {
    throw std::runtime_error(
            "Failed to initialize PCAN channel " + format_channel_name(kChannelHandle_) +
            " at " + bitrate_to_string(kChannelBitrate_) + ": " + format_error(init_status));
  }

  int fd = -1;
  if (CAN_GetValue(kChannelHandle_, PCAN_RECEIVE_EVENT, &fd, sizeof(fd)) == PCAN_ERROR_OK) {
    receive_event_fd_ = fd;
  }
}

PCANInterface::~PCANInterface() noexcept
{
  std::lock_guard<std::mutex> lock(io_mutex_);
  (void)CAN_Uninitialize(kChannelHandle_);
}

TPCANStatus PCANInterface::write(const TPCANMsg & tx_frame)
{
  TPCANMsg writable_msg = tx_frame;
  std::lock_guard<std::mutex> lock(io_mutex_);
  return CAN_Write(kChannelHandle_, &writable_msg);
}

TPCANStatus PCANInterface::read(TPCANMsg & rx_frame, TPCANTimestamp * timestamp)
{
  TPCANTimestamp local_timestamp{};
  TPCANTimestamp * timestamp_ptr = timestamp != nullptr ? timestamp : &local_timestamp;
  std::lock_guard<std::mutex> lock(io_mutex_);
  return CAN_Read(kChannelHandle_, &rx_frame, timestamp_ptr);
}

TPCANStatus PCANInterface::read_with_timeout(
  TPCANMsg & rx_frame,
  std::chrono::microseconds timeout)
{
  {
    std::lock_guard<std::mutex> lock(io_mutex_);
    TPCANTimestamp timestamp{};
    const TPCANStatus status = CAN_Read(kChannelHandle_, &rx_frame, &timestamp);
    if (status != PCAN_ERROR_QRCVEMPTY) {
      return status;
    }
  }

  if (receive_event_fd_ >= 0) {
    struct pollfd poll_fd{};
    poll_fd.fd = receive_event_fd_;
    poll_fd.events = POLLIN;
    const long timeout_us = timeout.count();
    struct timespec timeout_spec{};
    timeout_spec.tv_sec = timeout_us / 1000000;
    timeout_spec.tv_nsec = (timeout_us % 1000000) * 1000;
    if (::ppoll(&poll_fd, 1, &timeout_spec, nullptr) <= 0) {
      return PCAN_ERROR_QRCVEMPTY;
    }
  } else {
    std::this_thread::sleep_for(timeout);
  }

  std::lock_guard<std::mutex> lock(io_mutex_);
  TPCANTimestamp timestamp{};
  return CAN_Read(kChannelHandle_, &rx_frame, &timestamp);
}

TPCANStatus PCANInterface::get_bus_status()
{
  std::lock_guard<std::mutex> lock(io_mutex_);
  return CAN_GetStatus(kChannelHandle_);
}

TPCANStatus PCANInterface::get_value(
  TPCANParameter parameter,
  void * buffer,
  uint32_t buffer_length)
{
  std::lock_guard<std::mutex> lock(io_mutex_);
  return CAN_GetValue(kChannelHandle_, parameter, buffer, buffer_length);
}

std::string PCANInterface::format_error(TPCANStatus status)
{
  char buffer[256] = {};
  if (CAN_GetErrorText(status, kPcanLangEnglish, buffer) != PCAN_ERROR_OK) {
    std::snprintf(
      buffer,
      sizeof(buffer),
      "An error occurred. Error-code text (%Xh) could not be retrieved",
      status);
  }
  return buffer;
}

}  // namespace pcan_interface
