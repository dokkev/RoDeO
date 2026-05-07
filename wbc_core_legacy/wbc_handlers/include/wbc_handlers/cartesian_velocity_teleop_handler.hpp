/**
 * @file wbc_handlers/include/wbc_handlers/cartesian_velocity_teleop_handler.hpp
 * @brief Instantaneous velocity-servo Cartesian teleop handler.
 *
 * Unlike CartesianTeleopHandler (which integrates velocity into a goal and
 * smooths toward it), this handler keeps no pose backlog.  Each control tick
 * it projects the latest velocity command forward by preview_time_ from the
 * *current measured* pose, so tracking debt never accumulates.
 *
 * Usage (per input tick):
 *   handler.SetLinearVelocity(xdot);
 *   handler.SetAngularVelocity(omega);
 *   // call ResetCommand() when input stream is silent / timed-out
 *
 * Usage (per control tick):
 *   handler.UpdatePos(ee_pos_curr, ee_pos_task_);
 *   handler.UpdateOri(ee_quat_curr, ee_ori_task_);
 */
#pragma once

#include <cmath>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include "wbc_formulation/se3_math.hpp"

namespace wbc {

class CartesianVelocityTeleopHandler {
public:
  CartesianVelocityTeleopHandler() = default;

  /**
   * @brief Initialize handler.
   * @param preview_time  Look-ahead horizon [s].  Recommended starting value: 0.02.
   *                      Smaller → softer feel, larger → more aggressive.
   */
  void Init(double preview_time) {
    preview_time_ = std::max(0.0, preview_time);
    xdot_cmd_.setZero();
    omega_cmd_.setZero();
    zeros3_.setZero();
    initialized_ = true;
  }

  /** @brief Overwrite the latest linear velocity command [m/s], world frame. */
  void SetLinearVelocity(const Eigen::Vector3d& xdot) {
    if (!initialized_) return;
    xdot_cmd_ = xdot;
  }

  /** @brief Overwrite the latest angular velocity command [rad/s], world frame. */
  void SetAngularVelocity(const Eigen::Vector3d& omega) {
    if (!initialized_) return;
    omega_cmd_ = omega;
  }

  /** @brief Zero both velocity commands (call when input stream times out). */
  void ResetCommand() {
    xdot_cmd_.setZero();
    omega_cmd_.setZero();
  }

  /**
   * @brief Compute desired position from current measured pose and push to task.
   *
   * des_pos = pos_curr + xdot_cmd * preview_time
   * des_vel = xdot_cmd  (feedforward, no scaling)
   * des_acc = 0
   *
   * @param pos_curr  Current measured EE position, world frame.
   * @param task      Cartesian position WBC task. Non-null.
   */
  template <typename PosTaskLike>
  void UpdatePos(const Eigen::Vector3d& pos_curr, PosTaskLike* task) {
    if (!initialized_ || task == nullptr) return;
    const Eigen::Vector3d pos_des = pos_curr + xdot_cmd_ * preview_time_;
    task->UpdateDesired(pos_des, xdot_cmd_, zeros3_);
  }

  /**
   * @brief Compute desired orientation from current measured pose and push to task.
   *
   * des_ori = Exp(omega_cmd * preview_time) * quat_curr
   * des_vel = omega_cmd  (feedforward, no scaling)
   * des_acc = 0
   *
   * @param quat_curr  Current measured EE orientation, world frame.
   * @param task       Orientation WBC task. Non-null.
   */
  template <typename OriTaskLike>
  void UpdateOri(const Eigen::Quaterniond& quat_curr, OriTaskLike* task) {
    if (!initialized_ || task == nullptr) return;
    const Eigen::Quaterniond quat_des =
        se3::IntegrateAngularVelocityWorld(quat_curr, omega_cmd_, preview_time_);
    task->UpdateDesired(se3::QuatToXyzw(quat_des), omega_cmd_, zeros3_);
  }

  bool IsInitialized() const { return initialized_; }

  const Eigen::Vector3d& LinearVelocityCmd()  const { return xdot_cmd_; }
  const Eigen::Vector3d& AngularVelocityCmd() const { return omega_cmd_; }

private:
  double preview_time_{0.02};
  Eigen::Vector3d xdot_cmd_{Eigen::Vector3d::Zero()};
  Eigen::Vector3d omega_cmd_{Eigen::Vector3d::Zero()};
  Eigen::Vector3d zeros3_{Eigen::Vector3d::Zero()};
  bool initialized_{false};
};

}  // namespace wbc
