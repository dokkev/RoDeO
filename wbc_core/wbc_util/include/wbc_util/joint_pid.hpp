/**
 * @file wbc_core/wbc_util/include/wbc_util/joint_pid.hpp
 * @brief Per-joint cascade PID controller with pre-allocated RT-safe buffers.
 */
#pragma once

#include <Eigen/Dense>

namespace wbc {

/**
 * @brief Configuration gains for the cascade PID controller.
 *
 * Each loop has its own full PID gains and integral clamp.
 * Any gain not set in YAML defaults to zero (loop disabled for that term).
 *
 *   Outer (position loop) — outputs velocity reference:
 *     qdot_ref = kp_pos*(q_des-q) + ki_pos*∫(q_des-q)dt + kd_pos*(qdot_des-qdot)
 *
 *   Inner (velocity loop) — outputs torque:
 *     tau = kp_vel*(qdot_ref-qdot) + ki_vel*∫(qdot_ref-qdot)dt + kd_vel*d(vel_err)/dt
 */
struct JointPIDConfig {
  bool enabled{false};

  // Outer (position) loop
  Eigen::VectorXd kp_pos{Eigen::VectorXd::Zero(1)};
  Eigen::VectorXd ki_pos{Eigen::VectorXd::Zero(1)};
  Eigen::VectorXd kd_pos{Eigen::VectorXd::Zero(1)};
  Eigen::VectorXd pos_integral_limit{Eigen::VectorXd::Constant(1, 1.0e6)};

  // Inner (velocity) loop
  Eigen::VectorXd kp_vel{Eigen::VectorXd::Zero(1)};
  Eigen::VectorXd ki_vel{Eigen::VectorXd::Zero(1)};
  Eigen::VectorXd kd_vel{Eigen::VectorXd::Zero(1)};
  Eigen::VectorXd vel_integral_limit{Eigen::VectorXd::Constant(1, 1.0e6)};
};

/**
 * @brief Per-joint cascade PID feedback controller with optional direct-PD mode.
 *
 * @details
 * All internal buffers are allocated once in Setup(). Compute() is allocation-free.
 *
 * **Cascade mode** (default when any velocity gain is non-zero):
 *
 *   Outer (position loop):
 *     pos_err     = q_des - q
 *     pos_int    += pos_err * dt                     [clamped]
 *     qdot_ref    = Kp_pos*pos_err + Ki_pos*pos_int + Kd_pos*(qdot_des - qdot)
 *
 *   Inner (velocity loop):
 *     vel_err     = qdot_ref - qdot
 *     vel_int    += vel_err * dt                     [clamped]
 *     vel_err_dot = (vel_err - vel_err_prev) / dt    [backward difference]
 *     tau         = Kp_vel*vel_err + Ki_vel*vel_int + Kd_vel*vel_err_dot
 *
 * **Direct-PD mode** (when all velocity gains are zero, i.e. SetVelocityGains is not
 * called or all components are 0): the velocity loop is skipped and the position
 * loop output is used directly as torque:
 *
 *     tau = Kp_pos*(q_des-q) + Kd_pos*(qdot_des-qdot)  [+ Ki_pos integral term]
 *
 * To select direct-PD mode, simply omit kp_vel/ki_vel/kd_vel from the gains YAML
 * (they default to zero). To select cascade mode, set at least kp_vel.
 */
class JointPID {
public:
  JointPID() = default;

  /**
   * @brief Allocate internal buffers for @p n joints. Must be called once
   *        before Set*Gains / Compute. Safe to call multiple times.
   */
  void Setup(int n) {
    n_ = n;
    kp_pos_.setZero(n);  ki_pos_.setZero(n);  kd_pos_.setZero(n);
    kp_vel_.setZero(n);  ki_vel_.setZero(n);  kd_vel_.setZero(n);
    pos_integral_.setZero(n);
    pos_integral_limit_.setConstant(n, 1.0e6);
    vel_integral_.setZero(n);
    vel_integral_limit_.setConstant(n, 1.0e6);
    qdot_ref_.setZero(n);
    pos_err_.setZero(n);
    vel_err_.setZero(n);
    vel_err_prev_.setZero(n);
    vel_err_dot_.setZero(n);
    vel_loop_disabled_ = true;  // all vel gains are zero until SetVelocityGains() is called
  }

  /** @brief Set outer position-loop PID gains. */
  void SetPositionGains(const Eigen::Ref<const Eigen::VectorXd>& kp,
                        const Eigen::Ref<const Eigen::VectorXd>& ki,
                        const Eigen::Ref<const Eigen::VectorXd>& kd) {
    kp_pos_ = kp;  ki_pos_ = ki;  kd_pos_ = kd;
  }

  /** @brief Set inner velocity-loop PID gains.
   *  Setting all three to zero (or never calling this) enables direct-PD mode. */
  void SetVelocityGains(const Eigen::Ref<const Eigen::VectorXd>& kp,
                        const Eigen::Ref<const Eigen::VectorXd>& ki,
                        const Eigen::Ref<const Eigen::VectorXd>& kd) {
    kp_vel_ = kp;  ki_vel_ = ki;  kd_vel_ = kd;
    vel_loop_disabled_ = kp_vel_.isZero() && ki_vel_.isZero() && kd_vel_.isZero();
  }

  /** @brief Per-joint position-integral clamping bound (symmetric). */
  void SetPositionIntegralLimit(const Eigen::Ref<const Eigen::VectorXd>& limit) {
    pos_integral_limit_ = limit;
  }

  /** @brief Per-joint velocity-integral clamping bound (symmetric). */
  void SetVelocityIntegralLimit(const Eigen::Ref<const Eigen::VectorXd>& limit) {
    vel_integral_limit_ = limit;
  }

  /** @brief Reset both integrators and velocity-error history to zero. */
  void Reset() {
    pos_integral_.setZero();
    vel_integral_.setZero();
    vel_err_prev_.setZero();
  }

  bool IsSetup() const { return n_ > 0; }

  /**
   * @brief Compute feedback torque and write into @p out.
   *
   * @param q_des    Desired joint positions (n).
   * @param qdot_des Desired joint velocities (n) — feedforward / D term for position loop.
   * @param q        Measured joint positions (n).
   * @param qdot     Measured joint velocities (n).
   * @param dt       Time step in seconds.
   * @param out      Pre-allocated output vector (n) — written in-place.
   */
  void Compute(const Eigen::Ref<const Eigen::VectorXd>& q_des,
               const Eigen::Ref<const Eigen::VectorXd>& qdot_des,
               const Eigen::Ref<const Eigen::VectorXd>& q,
               const Eigen::Ref<const Eigen::VectorXd>& qdot,
               double dt,
               Eigen::Ref<Eigen::VectorXd> out) {
    // --- Outer (position) loop ---
    pos_err_      = q_des - q;
    pos_integral_ += pos_err_ * dt;
    pos_integral_  = pos_integral_.cwiseMax(-pos_integral_limit_)
                                  .cwiseMin( pos_integral_limit_);
    // D term: rate of change of position error = qdot_des - qdot
    qdot_ref_ = kp_pos_.cwiseProduct(pos_err_)
              + ki_pos_.cwiseProduct(pos_integral_)
              + kd_pos_.cwiseProduct(qdot_des - qdot);

    // --- Inner (velocity) loop — skipped in direct-PD mode ---
    if (vel_loop_disabled_) {
      out = qdot_ref_;  // position loop output IS the torque
      return;
    }
    vel_err_      = qdot_ref_ - qdot;
    vel_integral_ += vel_err_ * dt;
    vel_integral_  = vel_integral_.cwiseMax(-vel_integral_limit_)
                                  .cwiseMin( vel_integral_limit_);
    if (dt > 0.0) { vel_err_dot_ = (vel_err_ - vel_err_prev_) / dt; }
    else          { vel_err_dot_.setZero(); }
    out = kp_vel_.cwiseProduct(vel_err_)
        + ki_vel_.cwiseProduct(vel_integral_)
        + kd_vel_.cwiseProduct(vel_err_dot_);

    vel_err_prev_ = vel_err_;
  }

private:
  int n_{0};
  bool vel_loop_disabled_{true};  // true when all vel gains are zero (direct-PD mode)
  // Gains
  Eigen::VectorXd kp_pos_, ki_pos_, kd_pos_;
  Eigen::VectorXd kp_vel_, ki_vel_, kd_vel_;
  // State
  Eigen::VectorXd pos_integral_, pos_integral_limit_;
  Eigen::VectorXd vel_integral_, vel_integral_limit_;
  // Scratch (pre-allocated)
  Eigen::VectorXd qdot_ref_;
  Eigen::VectorXd pos_err_;
  Eigen::VectorXd vel_err_, vel_err_prev_, vel_err_dot_;
};

}  // namespace wbc
