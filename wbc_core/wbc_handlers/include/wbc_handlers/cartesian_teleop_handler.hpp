/**
 * @file wbc_handlers/include/wbc_handlers/cartesian_teleop_handler.hpp
 * @brief Velocity-command-based Cartesian teleop with isotropic speed clamping.
 */
#pragma once

#include <cmath>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include "wbc_formulation/se3_math.hpp"

namespace wbc {

/**
 * @brief Integrates 6-DOF velocity commands for Cartesian teleop with isotropic
 *        translational-speed and angular-rate clamping.
 *
 * Two internal buffer pairs:
 *   pos_goal_ / pos_des_smooth_    — 3-D position (world frame)
 *   quat_goal_ / quat_des_smooth_  — orientation (world frame, Eigen::Quaterniond)
 *
 * SetLinearVelocity() / SetAngularVelocity(): integrate into goal only.
 * SetPosition() / SetOrientation(): update goal only.
 * In both cases UpdatePos() / UpdateOri() rate-limit pos_des_smooth_ /
 * quat_des_smooth_ toward the goal at control frequency and feed forward the
 * resulting velocity, preventing sudden jumps and tracking lag.
 *
 * Usage (per input tick, e.g. 100 Hz):
 *   handler.SetLinearVelocity(xdot, dt_input);
 *   handler.SetAngularVelocity(omega, dt_input);
 *
 * Usage (per control tick, e.g. 1 kHz):
 *   handler.UpdatePos(ee_pos_task_, dt_ctrl);
 *   handler.UpdateOri(ee_ori_task_, dt_ctrl);
 *
 * Must call Init() before any other method.
 */
class CartesianTeleopHandler {
public:
  CartesianTeleopHandler() = default;

  /**
   * @brief Initialize from current EE transform and speed limits.
   *
   * @param pos_curr        Current EE position in world frame.
   * @param quat_curr       Current EE orientation in world frame.
   * @param linear_vel_max  Translational speed limit [m/s].
   * @param angular_vel_max Rotational rate limit [rad/s].
   */
  void Init(const Eigen::Vector3d& pos_curr,
            const Eigen::Quaterniond& quat_curr,
            double linear_vel_max,
            double angular_vel_max) {
    pos_goal_        = pos_curr;
    pos_des_smooth_  = pos_curr;
    quat_goal_       = quat_curr.normalized();
    quat_des_smooth_ = quat_goal_;
    linear_vel_max_  = std::max(0.0, linear_vel_max);
    angular_vel_max_ = std::max(0.0, angular_vel_max);
    zeros3_.setZero();
    vel3_.setZero();
    initialized_ = true;
  }

  /**
   * @brief Integrate a Cartesian velocity command into pos_goal_.
   *        UpdatePos() smooths pos_des_smooth_ toward pos_goal_ at control frequency.
   *
   * @param xdot  Linear velocity in world frame [m/s].
   * @param dt    Input period [s].
   */
  void SetLinearVelocity(const Eigen::Vector3d& xdot, double dt) {
    if (!initialized_ || dt <= 0.0) return;
    const double speed = xdot.norm();
    if (speed < kEps) return;
    const double scale = (speed > linear_vel_max_) ? (linear_vel_max_ / speed) : 1.0;
    pos_goal_ += xdot * (scale * dt);
  }

  /**
   * @brief Integrate an angular velocity command into quat_goal_.
   *        UpdateOri() smooths quat_des_smooth_ toward quat_goal_ at control frequency.
   *
   * Uses exact exponential map (se3::IntegrateAngularVelocityWorld) — no axis
   * normalization artifacts for large or small rates.
   *
   * @param omega  Angular velocity in world frame [rad/s].
   * @param dt     Input period [s].
   */
  void SetAngularVelocity(const Eigen::Vector3d& omega, double dt) {
    if (!initialized_ || dt <= 0.0) return;
    const double rate = omega.norm();
    if (rate < kEps) return;
    const double scale = (rate > angular_vel_max_) ? (angular_vel_max_ / rate) : 1.0;
    quat_goal_ = se3::IntegrateAngularVelocityWorld(quat_goal_, omega * scale, dt);
  }

  /**
   * @brief Set an absolute position target (jumps goal, smoothed by UpdatePos).
   */
  void SetPosition(const Eigen::Vector3d& x_des) {
    if (!initialized_) return;
    pos_goal_ = x_des;
  }

  /**
   * @brief Overwrite pos_goal_ directly (for post-integration clamping).
   */
  void SetPosGoal(const Eigen::Vector3d& goal) {
    if (!initialized_) return;
    pos_goal_ = goal;
  }

  /**
   * @brief Set an absolute orientation target (jumps goal, smoothed by UpdateOri).
   */
  void SetOrientation(const Eigen::Quaterniond& quat_des) {
    if (!initialized_) return;
    quat_goal_ = quat_des.normalized();
  }

  /**
   * @brief Step pos_des_smooth_ toward pos_goal_ and push to the WBC task.
   *
   * @param task  Cartesian position WBC task (e.g. LinkPosTask). Non-null.
   * @param dt    Control time step [s].
   */
  template <typename PosTaskLike>
  void UpdatePos(PosTaskLike* task, double dt) {
    if (!initialized_ || task == nullptr || dt < kEps) return;
    const Eigen::Vector3d old_pos = pos_des_smooth_;
    const Eigen::Vector3d delta   = pos_goal_ - pos_des_smooth_;
    const double dist     = delta.norm();
    const double max_step = linear_vel_max_ * dt;
    if (dist > max_step && dist > kEps) {
      pos_des_smooth_ += delta * (max_step / dist);
    } else {
      pos_des_smooth_ = pos_goal_;
    }
    vel3_ = (pos_des_smooth_ - old_pos) / dt;
    task->UpdateDesired(pos_des_smooth_, vel3_, zeros3_);
  }

  /**
   * @brief SLERP quat_des_smooth_ toward quat_goal_ and push to the WBC task.
   *
   * Angular feedforward velocity is computed via se3::AngularVelocityFromQuatDeltaWorld
   * (quaternion log map), which is exact and avoids the small-angle approximation.
   *
   * Orientation is packed as [x, y, z, w] to match LinkOriTask::UpdateDesired().
   *
   * @param task  Orientation WBC task (e.g. LinkOriTask). Non-null.
   * @param dt    Control time step [s].
   */
  template <typename OriTaskLike>
  void UpdateOri(OriTaskLike* task, double dt) {
    if (!initialized_ || task == nullptr || dt < kEps) return;
    const Eigen::Quaterniond old_quat = quat_des_smooth_;
    const double angle = old_quat.angularDistance(quat_goal_);
    if (angle < kEps) {
      quat_des_smooth_ = quat_goal_;
    } else {
      const double max_angle = angular_vel_max_ * dt;
      const double t = (max_angle >= angle) ? 1.0 : (max_angle / angle);
      quat_des_smooth_ = old_quat.slerp(t, quat_goal_).normalized();
    }
    vel3_ = se3::AngularVelocityFromQuatDeltaWorld(old_quat, quat_des_smooth_, dt);
    const Eigen::Vector4d ori_des = se3::QuatToXyzw(quat_des_smooth_);
    task->UpdateDesired(ori_des, vel3_, zeros3_);
  }

  bool IsInitialized() const { return initialized_; }

  const Eigen::Vector3d&    PosDesired()  const { return pos_des_smooth_; }
  const Eigen::Vector3d&    PosGoal()     const { return pos_goal_; }
  const Eigen::Quaterniond& QuatDesired() const { return quat_des_smooth_; }
  const Eigen::Quaterniond& QuatGoal()    const { return quat_goal_; }

private:
  static constexpr double kEps = 1.0e-10;

  Eigen::Vector3d    pos_goal_{Eigen::Vector3d::Zero()};
  Eigen::Vector3d    pos_des_smooth_{Eigen::Vector3d::Zero()};
  Eigen::Quaterniond quat_goal_{Eigen::Quaterniond::Identity()};
  Eigen::Quaterniond quat_des_smooth_{Eigen::Quaterniond::Identity()};
  double linear_vel_max_{1.0};
  double angular_vel_max_{1.0};
  Eigen::Vector3d zeros3_{Eigen::Vector3d::Zero()};
  Eigen::Vector3d vel3_{Eigen::Vector3d::Zero()};
  bool initialized_{false};
};

}  // namespace wbc
