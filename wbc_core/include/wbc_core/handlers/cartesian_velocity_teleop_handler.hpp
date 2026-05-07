/**
 * @file wbc_core/include/wbc_core/handlers/cartesian_velocity_teleop_handler.hpp
 * @brief Instantaneous velocity-servo Cartesian teleop handler for TSID tasks.
 *
 * Each control tick projects the latest velocity command forward by preview_time_
 * from the *current measured* pose, so tracking debt never accumulates.
 */
#pragma once

#include <cmath>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include "wbc_core/utils/se3_math.hpp"

namespace wbc {

class CartesianVelocityTeleopHandler {
public:
  CartesianVelocityTeleopHandler() = default;

  void Init(double preview_time) {
    preview_time_ = std::max(0.0, preview_time);
    xdot_cmd_.setZero();
    omega_cmd_.setZero();
    initialized_ = true;
  }

  void SetLinearVelocity(const Eigen::Vector3d& xdot) {
    if (!initialized_) return;
    xdot_cmd_ = xdot;
  }

  void SetAngularVelocity(const Eigen::Vector3d& omega) {
    if (!initialized_) return;
    omega_cmd_ = omega;
  }

  void ResetCommand() {
    xdot_cmd_.setZero();
    omega_cmd_.setZero();
  }

  /// Compute desired position from current measured pose.
  /// Returns (pos_des, vel_des) for the caller to set the TSID reference.
  struct PosResult {
    Eigen::Vector3d pos_des;
    Eigen::Vector3d vel_des;
  };

  PosResult ComputePos(const Eigen::Vector3d& pos_curr) const {
    PosResult r;
    r.pos_des = pos_curr + xdot_cmd_ * preview_time_;
    r.vel_des = xdot_cmd_;
    return r;
  }

  /// Compute desired orientation from current measured pose.
  /// Returns (quat_des, omega_des) for the caller to set the TSID reference.
  struct OriResult {
    Eigen::Quaterniond quat_des;
    Eigen::Vector3d omega_des;
  };

  OriResult ComputeOri(const Eigen::Quaterniond& quat_curr) const {
    OriResult r;
    r.quat_des = se3::IntegrateAngularVelocityWorld(quat_curr, omega_cmd_, preview_time_);
    r.omega_des = omega_cmd_;
    return r;
  }

  bool IsInitialized() const { return initialized_; }
  const Eigen::Vector3d& LinearVelocityCmd()  const { return xdot_cmd_; }
  const Eigen::Vector3d& AngularVelocityCmd() const { return omega_cmd_; }

private:
  double preview_time_{0.02};
  Eigen::Vector3d xdot_cmd_{Eigen::Vector3d::Zero()};
  Eigen::Vector3d omega_cmd_{Eigen::Vector3d::Zero()};
  bool initialized_{false};
};

}  // namespace wbc
