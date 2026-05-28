#include "Actuator.hpp"
using namespace ControlTableItem;

namespace actuator{

Actuator::Actuator(Dynamixel2Arduino &dxl, can_interface::CANInterface &can_interface, Config &config, Gains &gains)
    :   dxl_(dxl),
        can_interface_(can_interface), 
        config_(config),
        encoder_(config.gear_ratio, config.torque_constant),
        decoder_(config.gear_ratio, config.torque_constant),
        gains_(gains){

    // initilze messages
    state_msg_ = init_message_();
    onoff_msg_ = init_message_();
    
    PLATO_DEBUG_PRINTLN("Actuator Constructor");
    // Get DYNAMIXEL information
    dxl_.ping(config.dxl_id);

    // Always boot disabled. Host START_MOTOR commands are the only path to torque on.
    motor_enabled_ = false;
    dxl_.torqueOff(config.dxl_id);
  
    // Apply the offset
    apply_offset(config.encoder_tick, config.direction);

    // Re-assert torque off after configuration writes.
    dxl_.torqueOff(config.dxl_id);
}

Actuator::~Actuator(){
    // Turn off the torque
    

    if (dxl_.torqueOff(config_.dxl_id)){
        encoder_.stop_motor_response(state_msg_, config_.dxl_id, ResultByte::SUCCESS);
        can_interface_.send_cb(state_msg_);
        PLATO_DEBUG_PRINTLN("Motor Disabled");
    }
    else{
        encoder_.stop_motor_response(state_msg_, config_.dxl_id, ResultByte::FAILURE);
        can_interface_.send_cb(state_msg_);
        PLATO_DEBUG_PRINTLN("Motor Disable Failed");
    }
}

void Actuator::enable_motor(){
    // Turn on the torque
    motor_enabled_ = false;
    dxl_.torqueOff(config_.dxl_id);
    dxl_.setOperatingMode(config_.dxl_id, OP_CURRENT_BASED_POSITION);

    // set gains after enabling the motor because setting operation mode resets the gains to default
    set_gains(gains_);


    if (dxl_.torqueOn(config_.dxl_id)){
        motor_enabled_ = true;
        encoder_.start_motor_response(onoff_msg_, config_.dxl_id, ResultByte::SUCCESS);
        can_interface_.send_cb(onoff_msg_);
        PLATO_DEBUG_PRINTLN("Motor Enabled");
      
    }
    else{
        encoder_.start_motor_response(onoff_msg_, config_.dxl_id, ResultByte::FAILURE);
        can_interface_.send_cb(onoff_msg_);
        PLATO_DEBUG_PRINTLN("Motor Enable Failed");
    }
}

void Actuator::disable_motor(){
    // Turn off the torque

    if (dxl_.torqueOff(config_.dxl_id)){
        motor_enabled_ = false;
        encoder_.stop_motor_response(onoff_msg_, config_.dxl_id, ResultByte::SUCCESS);
        can_interface_.send_cb(onoff_msg_);
        PLATO_DEBUG_PRINTLN("Motor Disabled");
    }
    else{
        encoder_.stop_motor_response(onoff_msg_, config_.dxl_id, ResultByte::FAILURE);
        can_interface_.send_cb(onoff_msg_);
        PLATO_DEBUG_PRINTLN("Motor Disable Failed");
    }
}


bool Actuator::set_position(const float &position, const uint32_t &current){
    if (!motor_enabled_) {
        encoder_.set_states(
            state_msg_,
            config_.dxl_id,
            ResultByte::FAILURE_MOTOR_DISABLED,
            states_.position,
            states_.velocity,
            states_.torque);
        can_interface_.send_cb(state_msg_);
        PLATO_DEBUG_PRINTLN("Position command rejected: motor disabled");
        return false;
    }

    // set the current
    const bool current_ok = dxl_.setGoalCurrent(config_.dxl_id, current, UNIT_MILLI_AMPERE);
    float bounded_position = position;
    if (position > config_.joint_limit_max){
        bounded_position = config_.joint_limit_max;
    }
    else if (position < config_.joint_limit_min){
        bounded_position = config_.joint_limit_min;
    }

    // set the position
    const float cmd = rad2deg(bounded_position);
    const bool position_ok = dxl_.setGoalPosition(config_.dxl_id, cmd, UNIT_DEGREE);
    if (!current_ok || !position_ok) {
        encoder_.set_states(
            state_msg_,
            config_.dxl_id,
            ResultByte::FAILURE,
            states_.position,
            states_.velocity,
            states_.torque);
        can_interface_.send_cb(state_msg_);
        PLATO_DEBUG_PRINTLN("Position command failed");
        return false;
    }

    return true;

}

void Actuator::apply_offset(const uint16_t &encoder_tick, const char &direction){
    // set the zero position
    dxl_.writeControlTableItem(HOMING_OFFSET, config_.dxl_id, encoder_tick);
    dxl_.writeControlTableItem(DRIVE_MODE, config_.dxl_id, direction);

}

void Actuator::set_gains(const Gains &gains){
    // set the gains
    dxl_.writeControlTableItem(POSITION_P_GAIN, config_.dxl_id, gains.kp_position);
    dxl_.writeControlTableItem(POSITION_I_GAIN, config_.dxl_id, gains.ki_position);
    dxl_.writeControlTableItem(POSITION_D_GAIN, config_.dxl_id, gains.kd_position);
    dxl_.writeControlTableItem(VELOCITY_P_GAIN, config_.dxl_id, gains.kp_velocity);
    dxl_.writeControlTableItem(VELOCITY_I_GAIN, config_.dxl_id, gains.ki_velocity);
}

void Actuator::update_states(){
    // get the states
    states_.position = deg2rad(dxl_.getPresentPosition(config_.dxl_id, UNIT_DEGREE));
    states_.velocity = dxl_.getPresentVelocity(config_.dxl_id, UNIT_RPM);
    states_.torque = dxl_.getPresentCurrent(config_.dxl_id, UNIT_MILLI_AMPERE) / 1000 * config_.torque_constant;
    // states_.temperature = dxl_.readControlTableItem(PRESENT_TEMPERATURE, config_.dxl_id);

    // make a state message and send it
    encoder_.set_states(
        state_msg_,
        config_.dxl_id,
        ResultByte::SUCCESS,
        states_.position,
        states_.velocity,
        states_.torque);
    can_interface_.send_cb(state_msg_);

}

void Actuator::process_message(const CANMsg &msg){
    if (!accepts_message(msg)) {
        return;
    }

    switch (msg.DATA[0]){

        case CommandByte::START_MOTOR:
            enable_motor();
            return;

        case CommandByte::STOP_MOTOR:
            disable_motor();
            return;

        case CommandByte::POSITION_CONTROL:
            if (msg.LEN < 8) {
                return;
            }
            decoder_.get_position_command(msg, commands_.position, commands_.current);
            if (set_position(commands_.position, commands_.current)) {
                update_states();
            }
     
            break;
    }


	    
}

bool Actuator::accepts_message(const CANMsg &msg) const{
    return msg.ID == config_.can_rx_id && msg.LEN >= 2 && msg.DATA[1] == config_.dxl_id;
}




} // namespace actuator
