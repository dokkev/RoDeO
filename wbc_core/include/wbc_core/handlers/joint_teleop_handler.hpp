/**
 * @file wbc_core/include/wbc_core/handlers/joint_teleop_handler.hpp
 * @brief Velocity-command-based joint teleop with velocity and position clamping.
 */
#pragma once

#include <Eigen/Dense>

namespace wbc {

class JointTeleopHandler {
public:
  JointTeleopHandler() = default;

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
    old_q_.setZero(q_curr.size());
    scratch_.setZero(q_curr.size());
  }

  void SetVelocity(const Eigen::Ref<const Eigen::VectorXd>& qdot_cmd, double dt) {
    if (dt <= 0.0) return;
    // Clamp velocity, integrate, clamp position — no temporaries.
    vel_.noalias() = qdot_cmd.cwiseMax(-qdot_max_).cwiseMin(qdot_max_);
    q_goal_ = (q_goal_ + vel_ * dt).cwiseMax(q_min_).cwiseMin(q_max_);
    q_des_smooth_ = q_goal_;
  }

  void SetPosition(const Eigen::Ref<const Eigen::VectorXd>& q_des) {
    q_goal_ = q_des.cwiseMax(q_min_).cwiseMin(q_max_);
  }

  /// Step q_des_smooth_ toward q_goal_. Access results via Desired() / Vel().
  void Update(double dt) {
    if (dt <= 0.0) {
      vel_.setZero();
      return;
    }
    // old_q_ = q_des_smooth_ before step (reuse scratch)
    old_q_ = q_des_smooth_;
    // delta = q_goal_ - q_des_smooth_, max_step = qdot_max_ * dt
    // Step: q_des_smooth_ += clamp(delta, -max_step, max_step)
    scratch_ = qdot_max_ * dt;  // max_step
    vel_ = q_goal_ - q_des_smooth_;  // delta (borrow vel_ as scratch)
    q_des_smooth_ += vel_.cwiseMax(-scratch_).cwiseMin(scratch_);
    vel_ = (q_des_smooth_ - old_q_) / dt;
  }

  bool IsInitialized() const { return zeros_.size() > 0; }
  const Eigen::VectorXd& Desired() const { return q_des_smooth_; }
  const Eigen::VectorXd& Goal()    const { return q_goal_; }
  const Eigen::VectorXd& Vel()     const { return vel_; }
  const Eigen::VectorXd& ZeroAcc() const { return zeros_; }

private:
  Eigen::VectorXd q_goal_;
  Eigen::VectorXd q_des_smooth_;
  Eigen::VectorXd q_min_;
  Eigen::VectorXd q_max_;
  Eigen::VectorXd qdot_max_;
  Eigen::VectorXd zeros_;
  Eigen::VectorXd vel_;
  Eigen::VectorXd old_q_;     ///< scratch for Update()
  Eigen::VectorXd scratch_;   ///< scratch for Update()
};

}  // namespace wbc
