/**
 * @file wbc_handlers/include/wbc_handlers/joint_teleop_handler.hpp
 * @brief Velocity-command-based joint teleop with velocity and position clamping.
 */
#pragma once

#include <Eigen/Dense>

namespace wbc {

/**
 * @brief Integrates velocity commands for teleop with per-joint velocity
 *        and position-limit clamping.
 *
 * Two internal buffers:
 *   q_goal_       — integrated position goal (set by SetVelocity / SetPosition)
 *   q_des_smooth_ — rate-limited tracking reference sent to the WBC task
 *
 * SetVelocity() / SetPosition(): update q_goal_ only.
 * Update() rate-limits q_des_smooth_ toward q_goal_ at control frequency and
 * feeds forward the resulting velocity, preventing sudden jumps and tracking lag.
 *
 * Usage (per input tick, e.g. 100 Hz):
 *   handler.SetVelocity(qdot_cmd, dt_input);
 *
 * Usage (per control tick, e.g. 1 kHz):
 *   handler.Update(jpos_task_, dt_ctrl);
 */
class JointTeleopHandler {
public:
  JointTeleopHandler() = default;

  /**
   * @brief Initialize from current joint state and model limits.
   *
   * @param q_curr    Current joint positions (n).
   * @param q_min     Lower position limits (n). Use GetJointPositionLimits().col(0).
   * @param q_max     Upper position limits (n). Use GetJointPositionLimits().col(1).
   * @param qdot_max  Symmetric velocity limits (n). Use GetJointVelocityLimits().col(1).
   */
  void Init(const Eigen::VectorXd& q_curr,
            const Eigen::VectorXd& q_min,
            const Eigen::VectorXd& q_max,
            const Eigen::VectorXd& qdot_max) {
    q_goal_       = q_curr;
    q_des_smooth_ = q_curr;
    q_min_        = q_min;
    q_max_        = q_max;
    qdot_max_     = qdot_max;
    zeros_.setZero(q_curr.size());
    vel_.setZero(q_curr.size());
  }

  /**
   * @brief Integrate a velocity command into q_goal_, clamped by velocity and
   *        position limits. Update() smooths q_des_smooth_ at control frequency.
   *
   * @param qdot_cmd  Joint velocity command (n) from joystick / teleop input.
   * @param dt        Input period (seconds), e.g. 1/100 for 100 Hz input.
   */
  void SetVelocity(const Eigen::Ref<const Eigen::VectorXd>& qdot_cmd, double dt) {
    if (dt <= 0.0) return;
    const Eigen::VectorXd clamped = qdot_cmd.cwiseMax(-qdot_max_).cwiseMin(qdot_max_);
    q_goal_ = (q_goal_ + clamped * dt).cwiseMax(q_min_).cwiseMin(q_max_);
    q_des_smooth_ = q_goal_;  // velocity commands are already rate-limited
  }

  /**
   * @brief Set an absolute joint target, clamped to position limits.
   *
   * Only q_goal_ is updated. Update() will drive q_des_smooth_ toward
   * q_goal_ at qdot_max speed, preventing sudden jumps.
   */
  void SetPosition(const Eigen::Ref<const Eigen::VectorXd>& q_des) {
    q_goal_ = q_des.cwiseMax(q_min_).cwiseMin(q_max_);
  }

  /**
   * @brief Step q_des_smooth_ toward q_goal_ and push to the WBC task.
   *
   * Must be called once per control tick.
   *
   * @param task    WBC task to update (non-null).
   * @param dt      Control time step (seconds), e.g. 1/1000 for 1 kHz.
   */
  template <typename TaskLike>
  void Update(TaskLike* task, double dt) {
    if (task == nullptr || dt <= 0.0) return;
    const Eigen::VectorXd old_q    = q_des_smooth_;
    const Eigen::VectorXd delta    = q_goal_ - q_des_smooth_;
    const Eigen::VectorXd max_step = qdot_max_ * dt;
    q_des_smooth_ += delta.cwiseMax(-max_step).cwiseMin(max_step);
    vel_ = (q_des_smooth_ - old_q) / dt;
    task->UpdateDesired(q_des_smooth_, vel_, zeros_);
  }

  bool IsInitialized() const { return zeros_.size() > 0; }

  const Eigen::VectorXd& Desired() const { return q_des_smooth_; }
  const Eigen::VectorXd& Goal()    const { return q_goal_; }

private:
  Eigen::VectorXd q_goal_;
  Eigen::VectorXd q_des_smooth_;
  Eigen::VectorXd q_min_;
  Eigen::VectorXd q_max_;
  Eigen::VectorXd qdot_max_;
  Eigen::VectorXd zeros_;
  Eigen::VectorXd vel_;   // scratch buffer for feedforward velocity (pre-allocated)
};

}  // namespace wbc
