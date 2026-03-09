/**
 * @file controller/optimo_controller/include/optimo_controller/state_machines/cartesian_teleop.hpp
 * @brief Cartesian end-effector teleop state for Optimo.
 */
#pragma once

#include <Eigen/Geometry>

#include "wbc_formulation/motion_task.hpp"
#include "wbc_fsm/interface/state_machine.hpp"
#include "wbc_handlers/cartesian_velocity_teleop_handler.hpp"
#include "wbc_util/watchdog.hpp"
#include "wbc_handlers/manipulability_handler.hpp"

namespace wbc {

/**
 * @brief Cartesian velocity teleop with bounded reference integrator.
 *
 * Integrates velocity into a goal pose, bounded by anti-windup:
 * - Conditional integration slows when tracking error grows.
 * - Hard clamp prevents goal from running ahead of robot.
 * - HOLD mode: fixed goal with kp stiffness when command is zero.
 *
 * Registration key: "cartesian_teleop"
 */
class CartesianTeleop : public StateMachine {
public:
  CartesianTeleop(StateId state_id, const std::string& state_name,
           const StateMachineConfig& context);
  ~CartesianTeleop() override = default;

  void SetParameters(const YAML::Node& node) override;
  void FirstVisit() override;
  void OneStep() override;
  void LastVisit() override;
  bool EndOfState() override;

  /**
   * @brief Push one tick's velocity command (RT-safe, no alloc).
   *
   * Timestamps are compared against prev_vel_ts_ns_ to detect new arrivals.
   *
   * @param xdot       Linear velocity [m/s], world frame.
   * @param wdot       Angular velocity [rad/s], world frame.
   * @param vel_ts_ns  Message timestamp [ns]; 0 = never received.
   */
  void UpdateCommand(const Eigen::Vector3d& xdot,
                     const Eigen::Vector3d& wdot,
                     int64_t vel_ts_ns);

private:
  LinkPosTask*              ee_pos_task_{nullptr};
  LinkOriTask*              ee_ori_task_{nullptr};
  JointTask*                jpos_task_{nullptr};
  CartesianVelocityTeleopHandler ee_handler_;
  ManipulabilityHandler     manip_handler_;
  ManipulabilityHandler::Config manip_config_;
  CartesianVelocityTeleopHandler::Config handler_cfg_;
  Watchdog watchdog_{0.2};  // starts expired; Reset() on new message, Update() in OneStep()
  int64_t  prev_vel_ts_ns_{0};
};

}  // namespace wbc
