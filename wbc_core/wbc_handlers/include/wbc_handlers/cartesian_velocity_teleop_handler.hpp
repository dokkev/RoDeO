/**
 * @file wbc_handlers/include/wbc_handlers/cartesian_velocity_teleop_handler.hpp
 * @brief Cartesian velocity teleop with bounded reference integrator.
 *
 * Integrates commanded velocity into a goal pose, but bounds how far
 * the goal can get ahead of the measured pose (anti-windup).
 *
 * Two layers of protection:
 * 1. Conditional integration: slows/freezes integration when tracking error
 *    exceeds soft/hard thresholds (prevents backlog from building).
 * 2. Bounded reference: clamps the stored goal within e_max of measured
 *    (prevents large catch-up jumps).
 *
 * HOLD mode: when command is zero, goal stays fixed -> kp provides
 * real holding stiffness.
 *
 * Result: kp means stiffness, kd means damping, no preview-time hacks,
 * no unbounded debt accumulation.
 */
#pragma once

#include <algorithm>
#include <cmath>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include "wbc_formulation/se3_math.hpp"

namespace wbc {

class CartesianVelocityTeleopHandler {
public:
  enum class Mode { MOVE, HOLD };

  struct Config {
    double lin_enter_hold_thresh = 0.005;
    double lin_exit_hold_thresh  = 0.01;
    double ang_enter_hold_thresh = 0.02;
    double ang_exit_hold_thresh  = 0.05;

    double pos_e_soft = 0.02;
    double pos_e_hard = 0.06;
    double pos_e_max  = 0.08;

    double ori_e_soft = 0.08;
    double ori_e_hard = 0.20;
    double ori_e_max  = 0.25;
  };

  CartesianVelocityTeleopHandler() = default;

  void Init() {
    cfg_ = Config{};
    InitCommon();
  }

  void Init(const Config& cfg) {
    cfg_ = cfg;
    InitCommon();
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

  /// Freeze goal to current measured pose (use on watchdog timeout).
  void FreezeToMeasured(const Eigen::Vector3d& pos_curr,
                        const Eigen::Quaterniond& quat_curr) {
    goal_pos_ = pos_curr;
    goal_quat_ = quat_curr.normalized();
    xdot_cmd_.setZero();
    omega_cmd_.setZero();
    mode_ = Mode::HOLD;
    goal_initialized_ = true;
  }

  /**
   * @brief Main update: mode switch + bounded reference integration.
   * @param pos_curr   Measured EE position.
   * @param quat_curr  Measured EE orientation.
   * @param dt         Control timestep [s].
   * @param pos_task   Position WBC task (non-null).
   * @param ori_task   Orientation WBC task (non-null).
   */
  template <typename PosTaskLike, typename OriTaskLike>
  void Update(const Eigen::Vector3d& pos_curr,
              const Eigen::Quaterniond& quat_curr,
              double dt,
              PosTaskLike* pos_task,
              OriTaskLike* ori_task) {
    if (!initialized_ || dt <= 0.0) return;

    // Initialize goal to measured pose on first tick
    if (!goal_initialized_) {
      goal_pos_ = pos_curr;
      goal_quat_ = quat_curr.normalized();
      goal_initialized_ = true;
    }

    // --- Mode switching with separate linear/angular thresholds ---
    const bool cmd_active_exit =
        (xdot_cmd_.norm() > cfg_.lin_exit_hold_thresh) ||
        (omega_cmd_.norm() > cfg_.ang_exit_hold_thresh);
    const bool cmd_active_enter =
        (xdot_cmd_.norm() > cfg_.lin_enter_hold_thresh) ||
        (omega_cmd_.norm() > cfg_.ang_enter_hold_thresh);

    if (mode_ == Mode::HOLD && cmd_active_exit) {
      // HOLD -> MOVE: snap goal to measured to avoid stale error
      goal_pos_ = pos_curr;
      goal_quat_ = quat_curr.normalized();
      mode_ = Mode::MOVE;
    } else if (mode_ == Mode::MOVE && !cmd_active_enter) {
      // MOVE -> HOLD: goal stays where it is (provides hold stiffness)
      mode_ = Mode::HOLD;
    }

    // --- Position: bounded reference integration ---
    if (pos_task != nullptr) {
      if (mode_ == Mode::MOVE) {
        // 1. Conditional integration: slow/freeze when tracking is poor
        Eigen::Vector3d e_track = goal_pos_ - pos_curr;
        double e_norm = e_track.norm();
        double alpha = IntegrationGain(e_norm, cfg_.pos_e_soft, cfg_.pos_e_hard);

        goal_pos_ += alpha * xdot_cmd_ * dt;

        // 2. Bounded reference: clamp goal within e_max of measured
        Eigen::Vector3d e_new = goal_pos_ - pos_curr;
        double n = e_new.norm();
        if (n > cfg_.pos_e_max && n > 1e-10) {
          goal_pos_ = pos_curr + e_new * (cfg_.pos_e_max / n);
        }

        if (!CheckFinite(pos_curr, quat_curr)) return;

        // Scale velocity FF consistently with integration gain
        const Eigen::Vector3d xdot_ff = alpha * xdot_cmd_;
        pos_task->UpdateDesired(goal_pos_, xdot_ff, zeros3_);
      } else {
        // HOLD: fixed goal, zero velocity
        pos_task->UpdateDesired(goal_pos_, zeros3_, zeros3_);
      }
    }

    // --- Orientation: bounded reference integration ---
    if (ori_task != nullptr) {
      if (mode_ == Mode::MOVE) {
        // 1. Conditional integration
        Eigen::Quaterniond q_err_track = goal_quat_ * quat_curr.inverse();
        Eigen::Vector3d r_track = se3::RotationVectorFromQuaternion(q_err_track);
        double theta = r_track.norm();
        double alpha = IntegrationGain(theta, cfg_.ori_e_soft, cfg_.ori_e_hard);

        goal_quat_ = se3::IntegrateAngularVelocityWorld(
            goal_quat_, alpha * omega_cmd_, dt);

        // 2. Bounded reference: clamp orientation error
        Eigen::Quaterniond q_err_new = goal_quat_ * quat_curr.inverse();
        Eigen::Vector3d r_new = se3::RotationVectorFromQuaternion(q_err_new);
        double theta_new = r_new.norm();
        if (theta_new > cfg_.ori_e_max && theta_new > 1e-10) {
          r_new *= cfg_.ori_e_max / theta_new;
          goal_quat_ = (se3::DeltaQuatFromRotationVector(r_new) * quat_curr).normalized();
        }

        if (!CheckFinite(pos_curr, quat_curr)) return;

        // Scale velocity FF consistently with integration gain
        const Eigen::Vector3d omega_ff = alpha * omega_cmd_;
        ori_task->UpdateDesired(se3::QuatToXyzw(goal_quat_), omega_ff, zeros3_);
      } else {
        // HOLD: fixed goal, zero velocity
        ori_task->UpdateDesired(se3::QuatToXyzw(goal_quat_), zeros3_, zeros3_);
      }
    }
  }

  bool IsInitialized() const { return initialized_; }
  Mode GetMode() const { return mode_; }

  const Eigen::Vector3d& GoalPos() const { return goal_pos_; }
  const Eigen::Quaterniond& GoalQuat() const { return goal_quat_; }
  const Eigen::Vector3d& LinearVelocityCmd()  const { return xdot_cmd_; }
  const Eigen::Vector3d& AngularVelocityCmd() const { return omega_cmd_; }

private:
  void InitCommon() {
    // Sanitize hysteresis: exit >= enter
    cfg_.lin_exit_hold_thresh = std::max(cfg_.lin_exit_hold_thresh, cfg_.lin_enter_hold_thresh);
    cfg_.ang_exit_hold_thresh = std::max(cfg_.ang_exit_hold_thresh, cfg_.ang_enter_hold_thresh);

    // Sanitize anti-windup: 0 < e_soft < e_hard <= e_max
    cfg_.pos_e_soft = std::max(0.0, cfg_.pos_e_soft);
    cfg_.pos_e_hard = std::max(cfg_.pos_e_hard, cfg_.pos_e_soft + 1e-6);
    cfg_.pos_e_max  = std::max(cfg_.pos_e_max,  cfg_.pos_e_hard);

    cfg_.ori_e_soft = std::max(0.0, cfg_.ori_e_soft);
    cfg_.ori_e_hard = std::max(cfg_.ori_e_hard, cfg_.ori_e_soft + 1e-6);
    cfg_.ori_e_max  = std::max(cfg_.ori_e_max,  cfg_.ori_e_hard);

    xdot_cmd_.setZero();
    omega_cmd_.setZero();
    zeros3_.setZero();
    goal_pos_.setZero();
    goal_quat_ = Eigen::Quaterniond::Identity();
    mode_ = Mode::HOLD;
    goal_initialized_ = false;
    initialized_ = true;
  }

  /// Debug check: freeze to safe state on NaN/Inf (should never happen).
  bool CheckFinite(const Eigen::Vector3d& pos_curr,
                   const Eigen::Quaterniond& quat_curr) {
    if (goal_pos_.allFinite() && std::isfinite(goal_quat_.norm())) return true;
    // NaN detected: freeze to measured and log
    goal_pos_ = pos_curr;
    goal_quat_ = quat_curr.normalized();
    xdot_cmd_.setZero();
    omega_cmd_.setZero();
    mode_ = Mode::HOLD;
    return false;
  }

  /// Smooth integration gain: 1 when e < e_soft, 0 when e >= e_hard, linear between.
  static double IntegrationGain(double e, double e_soft, double e_hard) {
    if (e <= e_soft) return 1.0;
    if (e >= e_hard) return 0.0;
    return (e_hard - e) / (e_hard - e_soft);
  }

  Config cfg_;
  Mode mode_{Mode::HOLD};

  Eigen::Vector3d xdot_cmd_{Eigen::Vector3d::Zero()};
  Eigen::Vector3d omega_cmd_{Eigen::Vector3d::Zero()};
  Eigen::Vector3d zeros3_{Eigen::Vector3d::Zero()};

  // Integrated goal pose (bounded)
  Eigen::Vector3d goal_pos_{Eigen::Vector3d::Zero()};
  Eigen::Quaterniond goal_quat_{Eigen::Quaterniond::Identity()};

  bool initialized_{false};
  bool goal_initialized_{false};
};

}  // namespace wbc
