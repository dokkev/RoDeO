#ifndef MAIN_CANINTERFACE_H_
#define MAIN_CANINTERFACE_H_

#include <CAN.h>
#include "Configs.h"


namespace can_interface
{


/// @brief CANInterface class
class CANInterface{
   public:
 
    /// @brief Default constructor for CANInterface
    explicit CANInterface(uint32_t baudrate);

    /// @brief Default destructor for CANInterface
    ~CANInterface();

    /// @brief Initialize the CAN bus.
    /// @return true on success
    bool init();

    bool set_can_filter(const uint16_t filter_id);

    bool send_cb(const CANMsg &msg);

    bool receive_cb(CANMsg &msg);

    void print_message(const CANMsg &msg);

    private:
        uint32_t baudrate_;

        


};

}  // namespace can_interface

#endif  // MAIN_CANINTERFACE_H_
