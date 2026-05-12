#ifndef PLATO_HARDWARE_INTERFACE__UTILS__CAN_IDS_HPP_
#define PLATO_HARDWARE_INTERFACE__UTILS__CAN_IDS_HPP_

#include <cstdint>
#include <cstring>
#include <cmath> 


//// These Values need to match with the values in the motor driver. Don't Modify them
namespace CommandByte{

    // Configuration
    constexpr uint8_t RESET_CONFIGURATION = 0x81;
    constexpr uint8_t REFRESH_CONFIGURATION = 0x82;
    constexpr uint8_t MODIFY_CONFIGURATION = 0x83;
    constexpr uint8_t RETRIVE_CONFIGURATION = 0x84;
    
    // Control
    constexpr uint8_t START_MOTOR = 0x91;
    constexpr uint8_t STOP_MOTOR = 0x92;
    constexpr uint8_t TORQUE_CONTROL = 0x93;
    constexpr uint8_t SPEED_CONTROL = 0x94;
    constexpr uint8_t POSITION_CONTROL = 0x95;
    // constexpr uint8_t PTS= 0x96; // Unsupported
    constexpr uint8_t STOP_CONTROL = 0x97;

    // Parameter
    constexpr uint8_t MODIFY_PARAMETER = 0xA1;
    constexpr uint8_t RETRIVE_PARAMETER = 0xA2;

    // Status
    constexpr uint8_t GET_VERSION = 0xB1;
    constexpr uint8_t GET_FAULT = 0xB2;
    constexpr uint8_t ACKNOWLEDGE_FAULT = 0xB3;
    constexpr uint8_t RETRIVE_INDICATOR = 0xB4;
    constexpr uint8_t CALIBRATE = 0xB5;

    // Update
    // constexpr uint8_t UPDATE_FIRMWARE = 0xC1; // Don't use it
} // namespace CommandByte

namespace ConfigByte{
    constexpr uint8_t ZERO_POSITION = 0x14;
}

namespace ResultByte{
    constexpr uint8_t SUCCESS = 0x00;
    constexpr uint8_t FAILURE = 0x01;
    constexpr uint8_t FAILURE_UNKNOWN_COMMAND = 0x02;
    constexpr uint8_t FAILURE_UNKNOWN_ID = 0x03;
    constexpr uint8_t FAILURE_READ_ONLY_REGISTER = 0x04;
    constexpr uint8_t FAILURE_UNKNOWN_REGISTER = 0x05;
    constexpr uint8_t FAILURE_STRING_FORMAT = 0x06;
    constexpr uint8_t FAILURE_DATA_FORMAT_ERROR = 0x07;
    constexpr uint8_t FAILURE_WRITE_ONLY_REGISTER = 0x08;
} // namespace ResultByte


namespace ConfigType{
    constexpr uint8_t INT32 = 0x00;
    constexpr uint8_t FLOAT32 = 0x01;
} // namespace ConfigType

namespace IntConfigID{
    constexpr uint8_t POLE_PAIRS = 0x00;
    constexpr uint8_t RATED_CURRENT = 0x01; // Ampere
    constexpr uint8_t MAX_SPEED = 0x02; // RPM
    constexpr uint8_t RATED_VOLTAGE = 0x06; // V
    // constexpr uint8_t PWM_FREQUENCY = 0x07; // Hz  // don't use it
    constexpr uint8_t KP_CURRENT = 0x08;
    constexpr uint8_t KI_CURRENT = 0x09;
    constexpr uint8_t KP_SPEED = 0x0C;
    constexpr uint8_t KI_SPEED = 0x0D;
    constexpr uint8_t KP_POSITION = 0x0E;
    constexpr uint8_t KI_POSITION = 0x0F;
    constexpr uint8_t KD_POSITION = 0x10;
    constexpr uint8_t GEAR_RATIO = 0x11;
    constexpr uint8_t CAN_ID = 0x12;
    constexpr uint8_t HOST_CAN_ID = 0x13;
    constexpr uint8_t ZERO_POSITION = 0x14;
    constexpr uint8_t POWER_OFF_POSITION = 0x15; // read-only
    constexpr uint8_t OVER_VOLTAGE_THRESHOLD = 0x16; // V
    constexpr uint8_t UNDER_VOLTAGE_THRESHOLD = 0x17; // V
    constexpr uint8_t CAN_BAUDRATE = 0x18;
    // constexpr uint8_t KP_FLUX_WEAKENING = 0x19; // don't use it
    // constexpr uint8_t KI_FLUX_WEAKENING = 0x1A; // don't use it
    constexpr uint8_t OVER_TEMPERATURE_THRESHOLD = 0x20;
    // constexpr uint8_t PROTOCOL_OVER_CAN = 0x1C;  // don't use it
} // namespace IntConfigID

namespace FloatConfigID{
    constexpr uint8_t Rs = 0x00; // Ohm
    constexpr uint8_t Ls = 0x01; // Henry
    constexpr uint8_t BACK_EMF_CONSTANT = 0x02; // Vrms/krpm
    constexpr uint8_t TORQUE_CONSTANT = 0x03; // Nm/A
    constexpr uint8_t SAMPLING_RESISTOR = 0x04; // Ohm
    constexpr uint8_t AMPLIFICATION_GAIN = 0x05;
} // namespace FloatConfigID

namespace ParamID{
    constexpr uint8_t KP_CURRENT = 0x00;
    constexpr uint8_t KI_CURRENT = 0x01;
    constexpr uint8_t KP_SPEED = 0x02;
    constexpr uint8_t KI_SPEED = 0x03;
    constexpr uint8_t KP_POSITION = 0x04;
    constexpr uint8_t KI_POSITION = 0x05;
    constexpr uint8_t KD_POSITION = 0x06;
    // constexpr uint8_t KP_FLUX_WEAKENING = 0x07; // don't use it
    // constexpr uint8_t KI_FLUX_WEAKENING = 0x08; // don't use it

}

namespace IndicatorID{

    // TODO: Add more indicators
    constexpr uint8_t BUS_VOLTAGE = 0x00;
    constexpr uint8_t IQ = 0x09; // Ampere
    constexpr uint8_t IQ_TARGET = 0x0B; // Ampere
    constexpr uint8_t ROTOR_ANGLE = 0x12; // rad
    constexpr uint8_t SHAFT_ANGLE = 0x13; // rad
    constexpr uint8_t SHAFT_SPEED = 0x14; // rpm

} // namespace IndicatorID


namespace FaultID{
    constexpr uint8_t NO_FAULT = 0x00;
    constexpr uint8_t FOC_FREQ_TOO_HIGH = 0x01;
    constexpr uint8_t OVER_VOLTAGE = 0x02;
    constexpr uint8_t UNDER_VOLTAGE = 0x03;
    constexpr uint8_t OVER_TEMPARATURE = 0x08;
    constexpr uint8_t OVER_CURRENT = 0x10;
} // namespace FaultID


namespace CalibrationID{
    constexpr uint8_t PHASE_ORDER = 0x00;
    constexpr uint8_t ENCODER = 0x01;
} // namespace CalibrationID

namespace FTSensorID{
    constexpr uint8_t THUMB_FORCE = 0x2A;
    constexpr uint8_t THUMB_TORQUE = 0x2B;
    constexpr uint8_t INDEX_FORCE = 0x3A;
    constexpr uint8_t INDEX_TORQUE = 0x3B;
    constexpr uint8_t MIDDLE_FORCE = 0x1A;
    constexpr uint8_t MIDDLE_TORQUE = 0x1B;
}

#endif // PLATO_HARDWARE_INTERFACE__UTILS__CAN_IDS_HPP_
