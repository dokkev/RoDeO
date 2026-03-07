/**
 * @file controller/optimo_controller/include/optimo_controller/state_machines/sys_id.hpp
 * @brief Offline joint-space SysID excitation state.
 */
#pragma once

#include <string>

#include "wbc_handlers/sys_id_handler.hpp"
#include "wbc_formulation/motion_task.hpp"
#include "wbc_fsm/interface/state_machine.hpp"

namespace wbc {

/**
 * @brief Runs residual-compensator SysID trajectories on the joint task.
 *
 * Required task binding:
 * - `jpos_task`
 *
 * YAML params (under `params:`):
 * - mode: off|gravity_grid|friction_sweep|sine|chirp
 * - joint_idx, start_delay, ramp_time, dwell_time, duration
 * - amplitude, offset, cruise_vel, frequency_hz
 * - chirp_f0_hz, chirp_f1_hz, chirp_duration, phase0
 * - gravity_offsets_rad: [..]
 * - max_tracking_err, max_meas_vel, max_tau_ratio
 * - abort_on_safety, hold_on_abort, end_on_finish
 */
class SysIdState : public StateMachine {
public:
  SysIdState(StateId state_id, const std::string& state_name,
             const StateMachineConfig& context);
  ~SysIdState() override = default;

  void SetParameters(const YAML::Node& node) override;
  void FirstVisit() override;
  void OneStep() override;
  void LastVisit() override;
  bool EndOfState() override;

private:
  static SysIDMode ParseMode(const std::string& mode);

  JointTask* jpos_task_{nullptr};
  SysIdHandler handler_;
  SysIdHandlerConfig handler_config_;

  bool end_on_finish_{true};

  // Pre-allocated scratch (RT-safe in OneStep).
  Eigen::VectorXd q_ref_;
  Eigen::VectorXd qdot_ref_;
  Eigen::VectorXd qddot_ref_;
  Eigen::VectorXd tau_cmd_;
  Eigen::MatrixXd tau_limits_;
  bool abort_reported_{false};
};

}  // namespace wbc
