#ifndef MAIN__CONFIGS_H_
#define MAIN__CONFIGS_H_

#include <cstdint>
#include <cstring>
#include <cmath> 

constexpr uint32_t CAN_BAUDRATE_1M = 1000E3;

constexpr uint32_t SERVO_MCU_TX_ID = 0x21;
constexpr uint32_t SERVO_MCU_RX_ID = 0x11;

constexpr uint32_t SERVO1_TX_ID = SERVO_MCU_TX_ID;
constexpr uint32_t SERVO1_RX_ID = SERVO_MCU_RX_ID;
constexpr uint8_t  SERVO1_DXL_ID = 1;

constexpr uint32_t SERVO2_TX_ID = SERVO_MCU_TX_ID;
constexpr uint32_t SERVO2_RX_ID = SERVO_MCU_RX_ID;
constexpr uint8_t  SERVO2_DXL_ID = 2;

constexpr float GEAR_RATIO = 350.0f;
constexpr float TORQUE_CONSTANT = 1.17f;

// Uncomment this line to enable Arduino IDE Serial Monitor debug prints.
// #define PLATO_ENABLE_SERIAL_DEBUG

#ifdef PLATO_ENABLE_SERIAL_DEBUG
#define PLATO_DEBUG_BEGIN(baudrate) Serial.begin(baudrate)
#define PLATO_DEBUG_PRINT(value) Serial.print(value)
#define PLATO_DEBUG_PRINT_HEX(value) Serial.print(value, HEX)
#define PLATO_DEBUG_PRINTLN(value) Serial.println(value)
#define PLATO_DEBUG_NEWLINE() Serial.println()
#else
#define PLATO_DEBUG_BEGIN(baudrate) do {} while (0)
#define PLATO_DEBUG_PRINT(value) do {} while (0)
#define PLATO_DEBUG_PRINT_HEX(value) do {} while (0)
#define PLATO_DEBUG_PRINTLN(value) do {} while (0)
#define PLATO_DEBUG_NEWLINE() do {} while (0)
#endif


struct CANMsg{
    uint32_t ID;
    uint8_t DATA[8];
    uint8_t LEN;
};


//// These Values need to match with the values in the motor driver. Don't Modify them
namespace CommandByte{
    // Control
    constexpr uint8_t START_MOTOR = 0x91;
    constexpr uint8_t STOP_MOTOR = 0x92;
    constexpr uint8_t POSITION_CONTROL = 0x95;
} // namespace CommandByte

namespace ResultByte{
    constexpr uint8_t SUCCESS = 0x00;
    constexpr uint8_t FAILURE = 0x01;
    constexpr uint8_t FAILURE_MOTOR_DISABLED = 0x02;
} // namespace ResultByte


#endif  // MAIN_CONFIGS_H_
