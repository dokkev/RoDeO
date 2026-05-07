#pragma once

#include <Eigen/Geometry>

#include "wbc_formulation/motion_task.hpp"
#include "wbc_fsm/interface/state_machine.hpp"
#include "wbc_handlers/cartesian_velocity_teleop_handler.hpp"
#include "wbc_handlers/manipulability_handler.hpp"
#include "wbc_util/watchdog.hpp"

namespace wbc {

/**
 * @brief Dual Cartesian teleop: EE (wrist) + index fingertip.
 *
 * Tracks both the Optimo end_effector and the aristo_index_fingertip
 * simultaneously via two independent velocity command channels.
 *
 * YAML params (under `params:`):
 *   - `linear_vel_max`:        EE linear velocity limit [m/s]
 *   - `angular_vel_max`:       EE angular velocity limit [rad/s]
 *   - `finger_linear_vel_max`: Fingertip linear velocity limit [m/s]
 *   - `finger_angular_vel_max`: Fingertip angular velocity limit [rad/s]
 *   - `manipulability`:        Singularity avoidance config (applied to EE)
 *
 * External input:
 *   - UpdateEECommand():     EE velocity command + timestamp
 *   - UpdateFingerCommand(): Fingertip velocity command + timestamp
 *
 * Registration key: "plato_fingertip_teleop"
 */
class PlatoFingertipTeleop : public StateMachine {
public:
  PlatoFingertipTeleop(StateId state_id, const std::string& state_name,
                       const StateMachineConfig& context);
  ~PlatoFingertipTeleop() override = default;

  void SetParameters(const YAML::Node& node) override;
  void FirstVisit() override;
  void OneStep() override;
  void LastVisit() override;
  bool EndOfState() override;

  /// Push EE velocity command (RT-safe).
  void UpdateEECommand(const Eigen::Vector3d& xdot,
                       const Eigen::Vector3d& wdot,
                       int64_t vel_ts_ns);

  /// Push fingertip velocity command (RT-safe).
  void UpdateFingerCommand(const Eigen::Vector3d& xdot,
                           const Eigen::Vector3d& wdot,
                           int64_t vel_ts_ns);

private:
  // EE tasks (wrist)
  LinkPosTask*  ee_pos_task_{nullptr};
  LinkOriTask*  ee_ori_task_{nullptr};

  // Fingertip tasks
  LinkPosTask*  finger_pos_task_{nullptr};
  LinkOriTask*  finger_ori_task_{nullptr};

  // Posture
  JointTask*    jpos_task_{nullptr};

  // EE handler
  CartesianVelocityTeleopHandler ee_handler_;
  ManipulabilityHandler          manip_handler_;
  ManipulabilityHandler::Config  manip_config_;
  double preview_time_{0.02};
  Watchdog ee_watchdog_{0.2};
  int64_t  prev_ee_ts_ns_{0};

  // Finger handler
  CartesianVelocityTeleopHandler finger_handler_;
  Watchdog finger_watchdog_{0.2};
  int64_t  prev_finger_ts_ns_{0};
};

}  // namespace wbc
