/**
 * @file wbc_core/wbc_formulation/include/wbc_formulation/se3_math.hpp
 * @brief Lightweight SE(3) / SO(3) math utilities for motion task formulation.
 *
 * Header-only, Eigen-only. Placed in wbc_formulation (not wbc_util) because
 * wbc_util already depends on wbc_formulation — adding the reverse link would
 * create a circular CMake package dependency.
 *
 * All functions are in namespace wbc::se3.
 */
#pragma once

#include <cmath>

#include <Eigen/Dense>
#include <Eigen/Geometry>

namespace wbc::se3 {

inline constexpr double kEps = 1.0e-10;

// ---------------------------------------------------------------------------
// Quaternion conversions
// ---------------------------------------------------------------------------

/// Build quaternion from [x, y, z, w] vector (ROS/sensor convention).
/// Returns identity if the input has near-zero norm (guards against zero-initialized storage).
inline Eigen::Quaterniond QuatFromXyzw(const Eigen::Vector4d& q_xyzw) {
  if (q_xyzw.norm() < kEps) return Eigen::Quaterniond::Identity();
  Eigen::Quaterniond q(q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]);
  return q.normalized();
}

/// Convert quaternion to [x, y, z, w] vector (ROS/sensor convention).
/// Result is coeffs() of the normalized quaternion: [x, y, z, w].
inline Eigen::Vector4d QuatToXyzw(const Eigen::Quaterniond& q_in) {
  return q_in.normalized().coeffs();  // [x, y, z, w]
}

/// Identity quaternion as [x, y, z, w]. Useful for initializing VectorXd quaternion buffers.
inline Eigen::Vector4d QuatIdentityXyzw() {
  return Eigen::Vector4d(0.0, 0.0, 0.0, 1.0);
}

// ---------------------------------------------------------------------------
// Rotation vector / exponential map
// ---------------------------------------------------------------------------

/// Exact exponential map: rotation vector → unit quaternion.
/// For small ||rot_vec|| uses a first-order approximation to avoid division by zero.
inline Eigen::Quaterniond DeltaQuatFromRotationVector(
    const Eigen::Vector3d& rot_vec) {
  const double theta = rot_vec.norm();
  if (theta < kEps) {
    Eigen::Quaterniond q;
    q.w()   = 1.0;
    q.vec() = 0.5 * rot_vec;
    return q.normalized();
  }
  return Eigen::Quaterniond(Eigen::AngleAxisd(theta, rot_vec / theta));
}

/// Logarithmic map: unit quaternion → rotation vector.
/// Chooses shortest path (flips sign if w < 0).
inline Eigen::Vector3d RotationVectorFromQuaternion(
    const Eigen::Quaterniond& q_in) {
  Eigen::Quaterniond q = q_in.normalized();
  if (q.w() < 0.0) q.coeffs() *= -1.0;  // shortest path

  const double vec_norm = q.vec().norm();
  if (vec_norm < kEps) return Eigen::Vector3d::Zero();

  const double angle = 2.0 * std::atan2(vec_norm, q.w());
  return (angle / vec_norm) * q.vec();
}

/// Logarithmic map on SO(3): rotation matrix → rotation vector.
inline Eigen::Vector3d RotationLog(const Eigen::Matrix3d& R) {
  return RotationVectorFromQuaternion(Eigen::Quaterniond(R));
}

// ---------------------------------------------------------------------------
// Angular velocity integration and differentiation
// ---------------------------------------------------------------------------

/// First-order angular velocity integration in world frame.
///
/// Assumes q maps body→world; omega_world is in world frame.
/// R_new = Exp(omega_world * dt) * R_old  →  q_new = delta_q * q_old.
///
/// @param q            Current orientation (need not be pre-normalized).
/// @param omega_world  Angular velocity in world frame [rad/s].
/// @param dt           Time step [s]. Returns q unchanged if dt <= 0.
inline Eigen::Quaterniond IntegrateAngularVelocityWorld(
    const Eigen::Quaterniond& q,
    const Eigen::Vector3d& omega_world,
    double dt) {
  if (dt <= 0.0) return q.normalized();
  return (DeltaQuatFromRotationVector(omega_world * dt) * q).normalized();
}

/// Recover world-frame angular velocity from two successive quaternions.
///
/// Assumes q_new ≈ Exp(omega_world * dt) * q_old.
///
/// @param dt  Time step [s]. Returns zero if dt <= 0.
inline Eigen::Vector3d AngularVelocityFromQuatDeltaWorld(
    const Eigen::Quaterniond& q_old,
    const Eigen::Quaterniond& q_new,
    double dt) {
  if (dt <= 0.0) return Eigen::Vector3d::Zero();
  const Eigen::Quaterniond q_err = q_new * q_old.inverse();
  return RotationVectorFromQuaternion(q_err) / dt;
}

// ---------------------------------------------------------------------------
// Rotation error
// ---------------------------------------------------------------------------

/// Orientation error expressed in world frame via SO(3) log map.
///
/// Returns the world-frame rotation vector e such that:
///   Exp([e]_x) * R_cur ≈ R_des
///
/// Derivation: e = R_cur * log(R_cur^T * R_des)
///
/// @return 3-vector [rad], world frame. Zero when R_cur == R_des.
inline Eigen::Vector3d RotationErrorWorld(
    const Eigen::Matrix3d& R_cur_world,
    const Eigen::Matrix3d& R_des_world) {
  return R_cur_world * RotationLog(R_cur_world.transpose() * R_des_world);
}

// ---------------------------------------------------------------------------
// Relative kinematics in a moving reference frame
// ---------------------------------------------------------------------------

/// Position of target relative to reference frame origin, in reference frame.
///
/// p_rel = R_ref^T * (p_target_world - p_ref_world)
inline Eigen::Vector3d RelativePositionInFrame(
    const Eigen::Isometry3d& T_ref_world,
    const Eigen::Vector3d& p_target_world) {
  return T_ref_world.linear().transpose() *
         (p_target_world - T_ref_world.translation());
}

/// Linear velocity of target relative to the moving reference frame, in reference frame.
///
/// v_rel = R_ref^T * (v_target - v_ref - omega_ref × (p_target - p_ref))
///
/// The cross-product term removes the velocity induced at p_target by the
/// frame's rotation about its own origin.
inline Eigen::Vector3d RelativeLinearVelocityInFrame(
    const Eigen::Isometry3d& T_ref_world,
    const Eigen::Vector3d& v_ref_world,
    const Eigen::Vector3d& w_ref_world,
    const Eigen::Vector3d& p_target_world,
    const Eigen::Vector3d& v_target_world) {
  const Eigen::Vector3d p_rel = p_target_world - T_ref_world.translation();
  return T_ref_world.linear().transpose() *
         (v_target_world - v_ref_world - w_ref_world.cross(p_rel));
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

}  // namespace wbc::se3
