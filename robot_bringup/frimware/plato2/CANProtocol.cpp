#include "CANProtocol.hpp"

namespace can_protocol {

MsgDecoder::MsgDecoder(const float &gear_ratio, const float &torque_constant)
    : gear_ratio_(gear_ratio), torque_constant_(torque_constant) {}

/////////////////////////////////////////////////////////////////////////////////

void MsgDecoder::get_position_command(const CANMsg &msg, float &value, uint32_t &current) {
    // DATA[0] command, DATA[1] servo_id, DATA[2..5] float32 position rad.
    memcpy(&value, &msg.DATA[2], sizeof(float));

    // DATA[6..7] uint16 current limit in mA.
    current = msg.DATA[6] | (msg.DATA[7] << 8);
}

/////////////////////////////////////////////////////////////////////////////////

MsgEncoder::MsgEncoder(const float &gear_ratio, const float &torque_constant)
    : gear_ratio_(gear_ratio), torque_constant_(torque_constant) {}

/////////////////////////////////////////////////////////////////////////////////

void MsgEncoder::set_states(CANMsg &msg, const uint8_t servo_id, const uint8_t result, const float &position, const float &velocity, const float &torque) {
    // Set the message ID
    msg.DATA[0] = CommandByte::POSITION_CONTROL;
    msg.LEN = 8;

    // Dynamixel servo ID and result of the motor control.
    msg.DATA[1] = servo_id;
    msg.DATA[2] = result;

    // Encode position into bytes 3 and 4.
    uint16_t pos_int = static_cast<uint16_t>((position + 12.5f) * 65535.0f / 25.0f);
    msg.DATA[3] = pos_int & 0xFF;         // LSB of position
    msg.DATA[4] = (pos_int >> 8) & 0xFF;  // MSB of position

    // Encode velocity into bytes 5 and p6
    uint16_t velocity_int = static_cast<uint16_t>((velocity + 65.0f) * 4095.0f / 130.0f);
    msg.DATA[5] = (velocity_int >> 4) & 0xFF;        // Most significant 8 bits of velocity
    msg.DATA[6] = (velocity_int & 0x0F) << 4;        // Least significant 4 bits of velocity

    // Encode torque into the remaining part of byte 6 and byte 7
    uint16_t torque_int = static_cast<uint16_t>((torque + 225.0f * torque_constant_ * gear_ratio_) * 4095.0f / (450.0f * torque_constant_ * gear_ratio_));
    msg.DATA[6] |= (torque_int >> 8) & 0x0F;         // Most significant 4 bits of torque
    msg.DATA[7] = torque_int & 0xFF;                 // Least significant 8 bits of torque

}

void MsgEncoder::start_motor_response(CANMsg &msg, const uint8_t servo_id, const uint8_t result) {
    msg.DATA[0] = CommandByte::START_MOTOR;
    msg.DATA[1] = servo_id;
    msg.DATA[2] = result;
    msg.LEN = 3;
}

void MsgEncoder::stop_motor_response(CANMsg &msg, const uint8_t servo_id, const uint8_t result) {
    msg.DATA[0] = CommandByte::STOP_MOTOR;
    msg.DATA[1] = servo_id;
    msg.DATA[2] = result;
    msg.LEN = 3;
}

}  // namespace can_protocol
