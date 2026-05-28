#ifndef MAIN__CANPROTOCOL_HPP_
#define MAIN__CANPROTOCOL_HPP_

#include <cstdint>
#include <cstring>
#include <cmath> 

#include "Configs.h"

namespace  can_protocol{

/// @brief Decode host commands for the Dynamixel bridge protocol.
class MsgDecoder{

public:
    /// @brief Default constructor
    MsgDecoder(const float &gear_ratio, const float &torque_constant);

    /// @brief Get the desired position command and current limit from the received message.
    /// @param value desired position in radian
    /// @param current desired current limit in mA
    /// @param msg CANMsg reference to store the command message
    void get_position_command(const CANMsg &msg, float &value, uint32_t &current);

private:
    /// @brief gear ratio of the motor initialized in the actuator constructor
    const float &gear_ratio_;

    /// @brief torque constant of the motor initialized in the actuator constructor
    const float &torque_constant_;

};

class MsgEncoder{
public:
    /// @brief constructor
    MsgEncoder(const float &gear_ratio, const float &torque_constant);


    /// @brief set the state response for a Dynamixel position command.
    /// @param msg message to store the response
    /// @param servo_id Dynamixel servo ID
    /// @param result result of the command
    /// @param position reference to store the decoded position value
    /// @param velocity reference to store the decoded velocity value
    /// @param torque reference to store the decoded torque value
    void set_states(CANMsg &msg, const uint8_t servo_id, const uint8_t result, const float &position, const float &velocity, const float &torque);

    /// @brief set the motor response message for the start motor command
    /// @param msg message to store the response
    /// @param servo_id Dynamixel servo ID
    /// @param result result of the command
    void start_motor_response(CANMsg &msg, const uint8_t servo_id, const uint8_t result);

    /// @brief set the motor response message for the stop motor command
    /// @param msg message to store the response
    /// @param servo_id Dynamixel servo ID
    /// @param result result of the command
    void stop_motor_response(CANMsg &msg, const uint8_t servo_id, const uint8_t result);

private:
    /// @brief gear ratio of the motor
    const float &gear_ratio_ ;

    /// @brief torque constant of the motor
    const float &torque_constant_; 

};

} // namespace can_protocol
#endif  // MAIN__CANPROTOCOL_HPP_
