/**
 * @file controller/optimo_controller/include/optimo_controller/state_machines/joint_teleop.hpp
 * @brief Joint-space teleoperation state for Optimo.
 */
#pragma once

#include <string>

#include "wbc_formulation/motion_task.hpp"
#include "wbc_fsm/interface/state_machine.hpp"
#include "wbc_trajectory/joint_teleop_handler.hpp"
#include "wbc_util/watchdog.hpp"

namespace wbc {

/**
 * @brief Holds the current joint position and tracks external joint commands.
 *
 * YAML params (under `params:`):
 *   - `joint_vel_limit`: per-joint velocity limits [rad/s] (optional — falls back to URDF values)
 *
 * External input (typed, called via dynamic_cast from controller):
 *   - UpdateCommand(): batched update — velocity watchdog + optional position target.
 *     Called once per control tick before ctrl_arch_->Update() invokes OneStep().
 *
 * Registration key: "joint_teleop"
 */
class JointTeleop : public StateMachine {
public:
  JointTeleop(StateId state_id, const std::string& state_name,
              const StateMachineConfig& context);
  ~JointTeleop() override = default;

  void SetParameters(const YAML::Node& node) override;
  void FirstVisit() override;
  void OneStep() override;
  void LastVisit() override;
  bool EndOfState() override;

  /**
   * @brief Push one tick's worth of input to this state (RT-safe, no alloc).
   *
   * The state compares incoming timestamps against its own prev_*_ts_ns_ to
   * detect new arrivals — no boolean pre-processing in the controller needed.
   *
   * @param qdot_cmd    Latest velocity command from the RT buffer [rad/s].
   * @param vel_ts_ns   Timestamp of the velocity message [ns], 0 = never received.
   * @param q_des       Absolute position target [rad].
   * @param pos_ts_ns   Timestamp of the position message [ns], 0 = never received.
   */
  void UpdateCommand(const Eigen::Ref<const Eigen::VectorXd>& qdot_cmd,
                     int64_t vel_ts_ns,
                     const Eigen::Ref<const Eigen::VectorXd>& q_des,
                     int64_t pos_ts_ns);

private:
  JointTask*         jpos_task_{nullptr};
  JointTeleopHandler handler_;
  Eigen::VectorXd    vel_limit_;        // empty = use URDF limits
  Eigen::VectorXd    current_qdot_;     // RT command buffer, pre-sized in FirstVisit()
  Watchdog           watchdog_{0.2};    // starts expired; Reset() on new message, Update() in OneStep()
  int64_t            prev_vel_ts_ns_{0};
  int64_t            prev_pos_ts_ns_{0};
};

} // namespace wbc
