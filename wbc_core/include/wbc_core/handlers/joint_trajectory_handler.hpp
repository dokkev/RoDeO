/**
 * @file wbc_core/include/wbc_core/handlers/joint_trajectory_handler.hpp
 * @brief Time-parameterized min-jerk joint trajectory handler for TSID tasks.
 */
#pragma once

#include <algorithm>

#include <Eigen/Dense>

#include <wbc_core/tasks/task-joint-posture.hpp>
#include <wbc_core/trajectories/trajectory-base.hpp>

#include "wbc_core/trajectories/interpolation.hpp"

namespace wbc {

class JointTrajectoryHandler {
public:
  JointTrajectoryHandler() = default;

  bool SetTrajectory(const Eigen::VectorXd& start_pos,
                     const Eigen::VectorXd& target_pos, double duration) {
    if (start_pos.size() == 0 || start_pos.size() != target_pos.size() ||
        duration <= 0.0) {
      Reset();
      return false;
    }
    // Reuse scratch zero vector (allocated once, resized only if dim changes).
    if (zero_scratch_.size() != start_pos.size()) {
      zero_scratch_.setZero(start_pos.size());
    }
    return SetTrajectory(start_pos, zero_scratch_, zero_scratch_, target_pos,
                         zero_scratch_, zero_scratch_, duration);
  }

  bool SetTrajectory(const Eigen::VectorXd& start_pos,
                     const Eigen::VectorXd& start_vel,
                     const Eigen::VectorXd& start_acc,
                     const Eigen::VectorXd& target_pos,
                     const Eigen::VectorXd& target_vel,
                     const Eigen::VectorXd& target_acc, double duration) {
    if (duration <= 0.0 || start_pos.size() == 0 ||
        start_pos.size() != start_vel.size() ||
        start_pos.size() != start_acc.size() ||
        start_pos.size() != target_pos.size() ||
        start_pos.size() != target_vel.size() ||
        start_pos.size() != target_acc.size()) {
      Reset();
      return false;
    }
    curve_.Initialize(start_pos, start_vel, start_acc, target_pos, target_vel,
                      target_acc, duration);
    duration_ = duration;
    current_time_ = 0.0;
    is_running_ = true;
    return true;
  }

  /// Update with TSID TaskJointPosture.
  void Update(double elapsed_time,
              tsid::tasks::TaskJointPosture& task) {
    if (!is_running_) return;
    current_time_ = std::max(0.0, elapsed_time);
    UpdateAtCurrentTime(task);
  }

  void UpdateDelta(double dt,
                   tsid::tasks::TaskJointPosture& task) {
    if (!is_running_) return;
    current_time_ += std::max(0.0, dt);
    UpdateAtCurrentTime(task);
  }

  bool IsRunning() const { return is_running_; }
  bool IsFinished() const { return !is_running_; }

  void Reset() {
    duration_ = 0.0;
    current_time_ = 0.0;
    is_running_ = false;
  }

  double Duration() const { return duration_; }
  double CurrentTime() const { return current_time_; }

private:
  void UpdateAtCurrentTime(tsid::tasks::TaskJointPosture& task) {
    const double eval_time = std::min(current_time_, duration_);
    const auto& pos = curve_.Evaluate(eval_time);
    const auto& vel = curve_.EvaluateFirstDerivative(eval_time);
    const auto& acc = curve_.EvaluateSecondDerivative(eval_time);

    // Lazy-init sample buffer to match dimension.
    if (sample_.getValue().size() != pos.size()) {
      sample_ = tsid::trajectories::TrajectorySample(pos.size());
    }
    sample_.setValue(pos);
    sample_.setDerivative(vel);
    sample_.setSecondDerivative(acc);
    task.setReference(sample_);

    if (eval_time >= duration_) {
      is_running_ = false;
    }
  }

  util::MinJerkCurveVec curve_;
  double duration_{0.0};
  double current_time_{0.0};
  bool is_running_{false};
  tsid::trajectories::TrajectorySample sample_{0};
  Eigen::VectorXd zero_scratch_;  ///< scratch for SetTrajectory(2-arg)
};

} // namespace wbc
