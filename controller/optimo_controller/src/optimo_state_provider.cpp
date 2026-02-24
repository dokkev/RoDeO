#include "optimo_controller/optimo_state_provider.hpp"

namespace wbc {

OptimoStateProvider::OptimoStateProvider(double dt)
    : StateProvider(dt) {
  servo_dt_ = dt;
  data_save_freq_ = 1;

  count_ = 0;
  current_time_ = 0.0;

  b_f1_contact_ = false;
  b_f2_contact_ = false;
  b_f3_contact_ = false;

  state_ = 0;
  prev_state_ = 0;

  des_ee_iso_.setIdentity();
  rot_world_local_.setIdentity();

  planning_id_ = 0;
  teleop_cmd_seq_ = 0;
  teleop_cmd_time_sec_ = 0.0;
  teleop_raw_pose_.setIdentity();
}

} // namespace wbc
