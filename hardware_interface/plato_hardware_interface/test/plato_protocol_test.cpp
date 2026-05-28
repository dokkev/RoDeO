#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include "can_hardware_common/core/actuator_frame_router.hpp"
#include "plato_hardware_interface/actuator.hpp"
#include "plato_hardware_interface/dynamixel_can_protocol.hpp"
#include "plato_hardware_interface/gim3505_protocol.hpp"
#include "plato_hardware_interface/plato_layout.hpp"

namespace
{

plato_actuator::Config make_plato_config(std::uint8_t tx_id, std::uint8_t rx_id)
{
  plato_actuator::Config config;
  config.static_config.can_tx_id = tx_id;
  config.static_config.can_rx_id = rx_id;
  config.static_config.direction = 1;
  config.static_config.torque_constant = 0.1f;
  config.static_config.gear_ratio = 1.0f;
  config.static_config.servo_current_milliamps = 100;
  return config;
}

std::vector<plato_actuator::Actuator> make_plato_actuators(std::size_t count)
{
  std::vector<plato_actuator::Actuator> actuators;
  actuators.reserve(count);
  for (std::size_t i = 0; i < count; ++i) {
    auto config = make_plato_config(
      static_cast<std::uint8_t>(0x10 + i + 1),
      static_cast<std::uint8_t>(0x20 + i + 1));
    if (plato_hand::layout::is_thumb_servo(i)) {
      config.static_config.protocol_kind = plato_actuator::ProtocolKind::kDynamixelBridge;
      config.static_config.dynamixel_servo_id = plato_hand::layout::dynamixel_id_for_index(i);
    }
    actuators.emplace_back(config);
  }
  return actuators;
}

uint16_t pack_position(float position_rad)
{
  const float packed = (position_rad + 12.5f) * 65535.0f / 25.0f;
  return static_cast<uint16_t>(std::clamp(packed, 0.0f, 65535.0f));
}

uint16_t pack_velocity(float velocity_rpm)
{
  const float packed = (velocity_rpm + 65.0f) * 4095.0f / 130.0f;
  return static_cast<uint16_t>(std::clamp(packed, 0.0f, 4095.0f));
}

uint16_t pack_torque(float torque_nm, float torque_constant, float gear_ratio)
{
  const float scale = 450.0f * torque_constant * gear_ratio;
  const float offset = 225.0f * torque_constant * gear_ratio;
  const float packed = (torque_nm + offset) * 4095.0f / scale;
  return static_cast<uint16_t>(std::clamp(packed, 0.0f, 4095.0f));
}

float rpm_to_rad_s(float rpm)
{
  constexpr float kTwoPi = 6.28318530717958647692f;
  return rpm * kTwoPi / 60.0f;
}

TEST(ActuatorFrameRouterTest, AppendEnableFramesUsesActuatorEnableCommands)
{
  auto actuators = make_plato_actuators(4);
  std::vector<TPCANMsg> frames;

  can_hardware_common::core::append_enable_frames(actuators, frames);

  ASSERT_EQ(frames.size(), actuators.size());
  EXPECT_EQ(frames[0].ID, actuators[0].get_tx_id());
  EXPECT_EQ(
    frames.front().DATA[0],
    plato_hardware_interface::gim3505_protocol::CommandByte::START_MOTOR);
  EXPECT_EQ(frames.front().DATA[1], 1);
  EXPECT_EQ(frames[2].ID, actuators[2].enable_motor().frame.ID);
  EXPECT_EQ(
    frames[2].DATA[0],
    plato_hardware_interface::gim3505_protocol::CommandByte::START_MOTOR);
}

TEST(ActuatorFrameRouterTest, AppendDisableFramesUsesActuatorDisableCommands)
{
  auto actuators = make_plato_actuators(3);
  std::vector<TPCANMsg> frames;

  can_hardware_common::core::append_disable_frames(actuators, frames);

  ASSERT_EQ(frames.size(), actuators.size());
  EXPECT_EQ(frames[0].ID, actuators[0].get_tx_id());
  EXPECT_EQ(
    frames.front().DATA[0],
    plato_hardware_interface::gim3505_protocol::CommandByte::STOP_MOTOR);
  EXPECT_EQ(frames.front().DATA[1], 1);
  EXPECT_EQ(frames[2].ID, actuators[2].disable_motor().frame.ID);
  EXPECT_EQ(
    frames[2].DATA[0],
    plato_hardware_interface::gim3505_protocol::CommandByte::STOP_MOTOR);
}

TEST(ActuatorFrameRouterTest, DynamixelPackedFeedbackPayloadUpdatesMatchingServoState)
{
  auto actuators = make_plato_actuators(2);
  constexpr float kPosition = -0.42f;
  constexpr float kVelocityRpm = 12.0f;
  constexpr float kTorqueNm = 0.2f;
  constexpr float kTorqueConstant = 0.1f;
  constexpr float kGearRatio = 1.0f;

  const uint16_t packed_position = pack_position(kPosition);
  const uint16_t packed_velocity = pack_velocity(kVelocityRpm);
  const uint16_t packed_torque = pack_torque(kTorqueNm, kTorqueConstant, kGearRatio);

  TPCANMsg frame;
  std::memset(&frame, 0, sizeof(frame));
  frame.ID = actuators[1].get_rx_id();
  frame.MSGTYPE = PCAN_MESSAGE_STANDARD;
  frame.LEN = plato_hardware_interface::dynamixel_can_protocol::kFeedbackResponseLength;
  frame.DATA[0] =
    static_cast<std::uint8_t>(plato_hardware_interface::dynamixel_can_protocol::Command::kSetPosition);
  frame.DATA[1] = 2;
  frame.DATA[2] =
    static_cast<std::uint8_t>(plato_hardware_interface::dynamixel_can_protocol::Result::kSuccess);
  frame.DATA[3] = packed_position & 0xFF;
  frame.DATA[4] = (packed_position >> 8) & 0xFF;
  frame.DATA[5] = (packed_velocity >> 4) & 0xFF;
  frame.DATA[6] = static_cast<std::uint8_t>(
    ((packed_velocity & 0x0F) << 4) | ((packed_torque >> 8) & 0x0F));
  frame.DATA[7] = packed_torque & 0xFF;

  EXPECT_TRUE(can_hardware_common::core::dispatch_rx_frame(frame, actuators));
  EXPECT_FALSE(actuators[0].is_initialized());
  ASSERT_TRUE(actuators[1].is_initialized());
  EXPECT_NEAR(actuators[1].get_state().position, kPosition, 1.0e-3f);
  EXPECT_NEAR(actuators[1].get_state().velocity, rpm_to_rad_s(kVelocityRpm), 5.0e-3f);
  EXPECT_NEAR(actuators[1].get_state().torque, kTorqueNm, 2.0e-2f);
}

}  // namespace
