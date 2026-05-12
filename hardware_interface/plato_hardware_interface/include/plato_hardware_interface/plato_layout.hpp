#ifndef PLATO_HARDWARE_INTERFACE__PLATO_LAYOUT_HPP_
#define PLATO_HARDWARE_INTERFACE__PLATO_LAYOUT_HPP_

#include <array>
#include <cstddef>
#include <cstdint>

namespace plato_hand::layout
{

constexpr std::size_t kNumJoints = 8;
constexpr std::size_t kNumActuators = 8;

constexpr std::size_t kThumbRoll = 0;
constexpr std::size_t kThumbYaw = 1;
constexpr std::size_t kThumbMcp = 2;
constexpr std::size_t kThumbPip = 3;
constexpr std::size_t kIndexMcp = 4;
constexpr std::size_t kIndexPip = 5;
constexpr std::size_t kMiddleMcp = 6;
constexpr std::size_t kMiddlePip = 7;

constexpr std::size_t kFirstGimActuator = kThumbMcp;

constexpr std::array<std::size_t, 2> kThumbServoActuators = {
  kThumbRoll,
  kThumbYaw,
};

constexpr std::array<const char *, kNumActuators> kActuatorNames = {
  "thumb_roll",
  "thumb_yaw",
  "thumb_mcp",
  "thumb_pip",
  "index_mcp",
  "index_pip",
  "middle_mcp",
  "middle_pip",
};

constexpr bool is_thumb_servo(std::size_t actuator_index)
{
  return actuator_index == kThumbRoll || actuator_index == kThumbYaw;
}

constexpr bool is_gim_actuator(std::size_t actuator_index)
{
  return actuator_index >= kFirstGimActuator && actuator_index < kNumActuators;
}

constexpr uint8_t dynamixel_id_for_index(std::size_t actuator_index)
{
  return static_cast<uint8_t>(actuator_index + 1U);
}

}  // namespace plato_hand::layout

#endif  // PLATO_HARDWARE_INTERFACE__PLATO_LAYOUT_HPP_
