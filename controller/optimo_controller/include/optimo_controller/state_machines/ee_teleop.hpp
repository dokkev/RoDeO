/**
 * @file controller/optimo_controller/include/optimo_controller/state_machines/ee_teleop.hpp
 * @brief Cartesian end-effector teleop state for Optimo.
 */
#pragma once

#include <Eigen/Geometry>

#include "wbc_formulation/motion_task.hpp"
#include "wbc_fsm/interface/state_machine.hpp"
#include "wbc_trajectory/ee_teleop_handler.hpp"
#include "wbc_trajectory/joint_teleop_handler.hpp"
#include "wbc_util/watchdog.hpp"
#include "optimo_controller/manipulability_handler.hpp"
#include "optimo_controller/workspace_hull.hpp"

namespace wbc {

/**
 * @brief Holds the current EE pose and tracks external Cartesian commands,
 *        with a joint posture task in the null-space.
 *
 * YAML params (under `params:`):
 *   - `linear_vel_max`:  translational speed limit [m/s]   (default: 0.1)
 *   - `angular_vel_max`: rotational rate limit    [rad/s]  (default: 0.5)
 *   - `max_lookahead`:   anti-windup radius [m] — goal stays within this
 *                        distance of the actual EE position  (default: 0.1)
 *
 * External input (typed, called via dynamic_cast from controller):
 *   - UpdateCommand(): batched update — EE velocity watchdog + optional pose target.
 *     Called once per control tick before ctrl_arch_->Update() invokes OneStep().
 *
 * Registration key: "ee_teleop"
 */
class EETeleop : public StateMachine {
public:
  EETeleop(StateId state_id, const std::string& state_name,
           const StateMachineConfig& context);
  ~EETeleop() override = default;

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
   * @param xdot         Linear velocity command [m/s], world frame.
   * @param wdot         Angular velocity command [rad/s], world frame.
   * @param vel_ts_ns    Timestamp of the velocity message [ns], 0 = never received.
   * @param x_des        Desired EE position [m], world frame.
   * @param w_des        Desired EE orientation, world frame.
   * @param pose_ts_ns   Timestamp of the pose message [ns], 0 = never received.
   */
  void UpdateCommand(const Eigen::Vector3d& xdot,
                     const Eigen::Vector3d& wdot,
                     int64_t vel_ts_ns,
                     const Eigen::Vector3d& x_des,
                     const Eigen::Quaterniond& w_des,
                     int64_t pose_ts_ns);

  /**
   * @brief Load workspace hull from a YAML file (configure phase, non-RT).
   * @return true on success; false means workspace clamping is disabled.
   */
  bool LoadWorkspace(const std::string& yaml_path);

private:
  LinkPosTask*       ee_pos_task_{nullptr};
  LinkOriTask*       ee_ori_task_{nullptr};
  JointTask*         jpos_task_{nullptr};
  EETeleopHandler    ee_handler_;
  JointTeleopHandler jpos_handler_;     // posture null-space
  ManipulabilityHandler manip_handler_; // singularity avoidance
  ManipulabilityHandler::Config manip_config_;
  double linear_vel_max_{0.1};
  double angular_vel_max_{0.5};
  double max_lookahead_{0.1};           // anti-windup radius [m]
  Eigen::Vector3d    current_xdot_{Eigen::Vector3d::Zero()};  // RT command buffer
  Eigen::Vector3d    current_wdot_{Eigen::Vector3d::Zero()};  // RT command buffer
  Watchdog           watchdog_{0.2};    // starts expired; Reset() on new message, Update() in OneStep()
  int64_t            prev_vel_ts_ns_{0};
  int64_t            prev_pose_ts_ns_{0};
  optimo_controller::WorkspaceHull workspace_;
};

}  // namespace wbc
