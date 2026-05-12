#ifndef CAN_HARDWARE_COMMON__UTILS__CAN_HELPER_HPP_
#define CAN_HARDWARE_COMMON__UTILS__CAN_HELPER_HPP_

#include <cstddef>
#include <cstdint>
#include <cstring>

#include "PCANBasic.h"

namespace can_hardware_common
{

namespace can_protocol_helpers
{

inline void encode_float_le(TPCANMsg & msg, float value, std::size_t offset)
{
  std::memcpy(&msg.DATA[offset], &value, sizeof(float));
}

inline void encode_u24_le(TPCANMsg & msg, uint32_t value, std::size_t offset)
{
  msg.DATA[offset] = value & 0xFF;
  msg.DATA[offset + 1] = (value >> 8) & 0xFF;
  msg.DATA[offset + 2] = (value >> 16) & 0xFF;
}

inline void encode_u32_le(TPCANMsg & msg, uint32_t value, std::size_t offset)
{
  msg.DATA[offset] = value & 0xFF;
  msg.DATA[offset + 1] = (value >> 8) & 0xFF;
  msg.DATA[offset + 2] = (value >> 16) & 0xFF;
  msg.DATA[offset + 3] = (value >> 24) & 0xFF;
}

inline uint32_t decode_u32_le(const TPCANMsg & msg, std::size_t offset)
{
  return msg.DATA[offset] |
         (msg.DATA[offset + 1] << 8) |
         (msg.DATA[offset + 2] << 16) |
         (msg.DATA[offset + 3] << 24);
}

inline float decode_float_le(const TPCANMsg & msg, std::size_t offset)
{
  float value = 0.0f;
  std::memcpy(&value, &msg.DATA[offset], sizeof(float));
  return value;
}

}  // namespace can_protocol_helpers

}  // namespace can_hardware_common

#endif  // CAN_HARDWARE_COMMON__UTILS__CAN_HELPER_HPP_
