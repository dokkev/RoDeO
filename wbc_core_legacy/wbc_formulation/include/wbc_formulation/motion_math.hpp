/**
 * @file wbc_core/wbc_formulation/include/wbc_formulation/motion_math.hpp
 * @brief Lightweight rotation and relative-motion helpers for motion task formulation.
 *
 * All functions are header-only (inline). No external dependencies beyond Eigen.
 *
 * Conventions:
 *   - Quaternion storage: Eigen native (w, x, y, z), but conversion helpers use xyzw order.
 *   - Angular velocity: always in world frame unless otherwise noted.
 *   - "InFrame" suffix: result expressed in the reference body frame.
 */
#pragma once

#include <cmath>

#include <Eigen/Dense>
#include <Eigen/Geometry>

namespace wbc {

// ---------------------------------------------------------------------------
// Quaternion conversions
// ---------------------------------------------------------------------------

/// Build a quaternion from a [x, y, z, w] vector (ROS/sensor convention).
inline Eigen::Quaterniond QuatFromXyzw(const Eigen::Vector4d& q_xyzw) {
  // Eigen::Quaterniond ctor order is (w, x, y, z).
  return Eigen::Quaterniond(q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]).normalized();
}

/// Convert a quaternion to a [x, y, z, w] vector (ROS/sensor convention).
inline Eigen::Vector4d QuatToXyzw(const Eigen::Quaterniond& q) {
  return Eigen::Vector4d(q.x(), q.y(), q.z(), q.w());
}

// ---------------------------------------------------------------------------
// Angular velocity integration
// ---------------------------------------------------------------------------

/// First-order exponential-map integration of orientation.
///
/// Integrates the ODE q_dot = 0.5 * [omega_world; 0] ⊗ q over dt using the
/// exact exponential map:
///   q_new = exp([omega * dt/2]) ⊗ q   (world-frame left multiplication)
///
/// @param q          Current orientation (need not be normalized on input).
/// @param omega_world Angular velocity in world frame [rad/s].
/// @param dt          Time step [s].
/// @return Normalized updated quaternion.
inline Eigen::Quaterniond IntegrateAngularVelocityWorld(
    const Eigen::Quaterniond& q,
    const Eigen::Vector3d& omega_world,
    double dt) {
  const double omega_norm = omega_world.norm();
  const double theta = omega_norm * dt;
  if (theta < 1e-10) return q;
  const Eigen::Quaterniond dq(Eigen::AngleAxisd(theta, omega_world / omega_norm));
  return (dq * q).normalized();
}

// ---------------------------------------------------------------------------
// Rotation error
// ---------------------------------------------------------------------------

/// Rotation error vector in world frame via matrix logarithm.
///
/// Computes log(R_des * R_cur^T), which gives the axis-angle vector e such that
/// exp([e]_x) * R_cur ≈ R_des.  The result is expressed in the world frame.
///
/// Numerically stable for |theta| in [0, pi).  Behaviour is undefined near pi
/// (R_err trace close to -1); clamp or singularity handling is left to the caller.
///
/// @return 3-vector [rad], world frame.  Zero when R_cur == R_des.
inline Eigen::Vector3d RotationLogErrorWorld(
    const Eigen::Matrix3d& R_cur,
    const Eigen::Matrix3d& R_des) {
  const Eigen::Matrix3d R_err = R_des * R_cur.transpose();
  const double cos_theta = std::clamp(0.5 * (R_err.trace() - 1.0), -1.0, 1.0);
  const double theta = std::acos(cos_theta);
  if (theta < 1e-7) return Eigen::Vector3d::Zero();
  const double coeff = theta / (2.0 * std::sin(theta));
  return coeff * Eigen::Vector3d(
      R_err(2, 1) - R_err(1, 2),
      R_err(0, 2) - R_err(2, 0),
      R_err(1, 0) - R_err(0, 1));
}

// ---------------------------------------------------------------------------
// Relative kinematics (position / velocity / orientation expressed in a
// moving reference body frame)
// ---------------------------------------------------------------------------

/// Position of target point relative to reference frame origin, in reference frame.
///
/// p_rel = R_ref^T * (p_target_world - p_ref_world)
inline Eigen::Vector3d RelativePositionInFrame(
    const Eigen::Isometry3d& T_ref,
    const Eigen::Vector3d& p_target_world) {
  return T_ref.rotation().transpose() * (p_target_world - T_ref.translation());
}

/// Linear velocity of target relative to the moving reference frame, in reference frame.
///
/// Accounts for the frame's translational and rotational motion:
///   v_rel_world = v_target - v_ref - omega_ref × (p_target - p_ref)
///   v_rel_in_ref = R_ref^T * v_rel_world
///
/// @param T_ref         Pose of reference frame (world ← frame).
/// @param v_ref_world   Linear velocity of reference frame origin, world frame [m/s].
/// @param w_ref_world   Angular velocity of reference frame, world frame [rad/s].
/// @param p_target_world Position of target point, world frame [m].
/// @param v_target_world Linear velocity of target point, world frame [m/s].
inline Eigen::Vector3d RelativeLinearVelocityInFrame(
    const Eigen::Isometry3d& T_ref,
    const Eigen::Vector3d& v_ref_world,
    const Eigen::Vector3d& w_ref_world,
    const Eigen::Vector3d& p_target_world,
    const Eigen::Vector3d& v_target_world) {
  const Eigen::Vector3d p_rel = p_target_world - T_ref.translation();
  const Eigen::Vector3d v_rel_world =
      v_target_world - v_ref_world - w_ref_world.cross(p_rel);
  return T_ref.rotation().transpose() * v_rel_world;
}

/// Orientation of target relative to reference, in reference frame.
///
/// R_rel = R_ref^T * R_target
inline Eigen::Matrix3d RelativeOrientationInFrame(
    const Eigen::Matrix3d& R_ref_world,
    const Eigen::Matrix3d& R_target_world) {
  return R_ref_world.transpose() * R_target_world;
}

/// Angular velocity of target relative to reference frame, in reference frame.
///
/// w_rel = R_ref^T * (w_target - w_ref)
inline Eigen::Vector3d RelativeAngularVelocityInFrame(
    const Eigen::Matrix3d& R_ref_world,
    const Eigen::Vector3d& w_ref_world,
    const Eigen::Vector3d& w_target_world) {
  return R_ref_world.transpose() * (w_target_world - w_ref_world);
}

}  // namespace wbc
