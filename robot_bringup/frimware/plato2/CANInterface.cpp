#include "CANInterface.hpp"

namespace can_interface {

CANInterface::CANInterface(uint32_t baudrate) : baudrate_(baudrate) {}

CANInterface::~CANInterface() {
    // End CAN on object destruction
    CAN.end();
}

bool CANInterface::init() {
    // Start CAN bus with specified baudrate
    if (!CAN.begin(baudrate_)) {
        PLATO_DEBUG_PRINTLN("CANInterface::init: Starting CAN Failed");
        return false;
    }
    PLATO_DEBUG_PRINTLN("CANInterface::init: CAN started with baudrate: " + String(baudrate_));

    return true;
}

bool CANInterface::set_can_filter(const uint16_t filter_id) {
    const uint16_t standard_id_mask = 0x7FF;
    const uint16_t filter = filter_id & standard_id_mask;
    const uint16_t mask = standard_id_mask;

    if (CAN.filter(filter, mask)) {
        PLATO_DEBUG_PRINTLN("CANInterface::set_can_filter: Filter applied for ID: 0x" + String(filter_id, HEX));
    }
    else {
        PLATO_DEBUG_PRINTLN("CANInterface::set_can_filter: Failed to apply filter!");
        return false;
    }
    return true;
}

bool CANInterface::send_cb(const CANMsg &msg) {
    // Start packet transmission on the specified ID
    if (CAN.beginPacket(msg.ID)) {
        // Send each byte in DATA array, up to LEN
        for (int i = 0; i < msg.LEN; i++) {
            CAN.write(msg.DATA[i]);
        }
        // End packet transmission
        CAN.endPacket();
        // delay(1); 
        // PLATO_DEBUG_PRINTLN("CANInterface::send_cb: Sent message with ID: 0x" + String(msg.ID, HEX));
        return true;
    }
    return false;
}

bool CANInterface::receive_cb(CANMsg &msg) {
    // Check if a CAN packet is available
    int packet_size = CAN.parsePacket();
    if (packet_size > 0) {
        // Store packet ID and length
        msg.ID = CAN.packetId();
        msg.LEN = packet_size;

        // Read packet data into msg.DATA
        for (int i = 0; i < packet_size; i++) {
            int r = CAN.read();
            if (r == -1) {
                PLATO_DEBUG_PRINTLN("CANInterface::receive_cb: Error reading CAN data");
                return false; // Return false if there's an error reading
            }
            msg.DATA[i] = static_cast<uint8_t>(r);
        }
        return true; // Return true if packet was successfully read
    }
    return false; // No packet available
}

void CANInterface::print_message(const CANMsg &msg) {
    // Print the message ID and data
    PLATO_DEBUG_PRINT("ID: 0x");
    PLATO_DEBUG_PRINT_HEX(msg.ID);
    PLATO_DEBUG_PRINT(" Data: ");
    for (int i = 0; i < msg.LEN; i++) {
        PLATO_DEBUG_PRINT_HEX(msg.DATA[i]);
        PLATO_DEBUG_PRINT(" ");
    }
    PLATO_DEBUG_NEWLINE();
}

}  // namespace can_interface
