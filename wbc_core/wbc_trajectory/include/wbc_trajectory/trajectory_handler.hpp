#pragma once

#include <algorithm>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include "wbc_trajectory/interpolation.hpp"

namespace wbc {

class VectorTrajectoryHandler {
public:
  VectorTrajectoryHandler() = default;

  bool SetTrajectory(const Eigen::VectorXd& start_pos,
                     const Eigen::VectorXd& target_pos, double duration) {
    if (start_pos.size() == 0 || start_pos.size() != target_pos.size() ||
        duration <= 0.0) {
      Reset();
      return false;
    }

    const Eigen::VectorXd zero = Eigen::VectorXd::Zero(start_pos.size());
    return SetTrajectory(start_pos, zero, zero, target_pos, zero, zero,
                         duration);
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

  template <typename TaskLike>
  void Update(double elapsed_time, TaskLike* task) {
    if (!is_running_ || task == nullptr) {
      return;
    }
    current_time_ = std::max(0.0, elapsed_time);
    UpdateAtCurrentTime(task);
  }

  template <typename TaskLike>
  void UpdateDelta(double dt, TaskLike* task) {
    if (!is_running_ || task == nullptr) {
      return;
    }
    current_time_ += std::max(0.0, dt);
    UpdateAtCurrentTime(task);
  }

  template <typename TaskLike>
  void UpdateWithTime(double elapsed_time, TaskLike* task) {
    Update(elapsed_time, task);
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
  template <typename TaskLike>
  void UpdateAtCurrentTime(TaskLike* task) {
    const double eval_time = std::min(current_time_, duration_);
    task->UpdateDesired(curve_.Evaluate(eval_time),
                        curve_.EvaluateFirstDerivative(eval_time),
                        curve_.EvaluateSecondDerivative(eval_time));
    if (eval_time >= duration_) {
      is_running_ = false;
    }
  }

  MinJerkCurveVec curve_;
  double duration_{0.0};
  double current_time_{0.0};
  bool is_running_{false};
};

class OrientationTrajectoryHandler {
public:
  OrientationTrajectoryHandler() = default;

  bool SetTrajectory(const Eigen::Quaterniond& start_quat,
                     const Eigen::Quaterniond& target_quat, double duration) {
    return SetTrajectory(start_quat, Eigen::Vector3d::Zero(), target_quat,
                         Eigen::Vector3d::Zero(), duration);
  }

  bool SetTrajectory(const Eigen::Quaterniond& start_quat,
                     const Eigen::Vector3d& start_ang_vel,
                     const Eigen::Quaterniond& target_quat,
                     const Eigen::Vector3d& target_ang_vel, double duration) {
    if (duration <= 0.0) {
      Reset();
      return false;
    }

    curve_.Initialize(start_quat.normalized(), start_ang_vel,
                      target_quat.normalized(), target_ang_vel, duration);
    duration_ = duration;
    current_time_ = 0.0;
    is_running_ = true;
    return true;
  }

  template <typename TaskLike>
  void Update(double elapsed_time, TaskLike* task) {
    if (!is_running_ || task == nullptr) {
      return;
    }
    current_time_ = std::max(0.0, elapsed_time);
    UpdateAtCurrentTime(task);
  }

  template <typename TaskLike>
  void UpdateDelta(double dt, TaskLike* task) {
    if (!is_running_ || task == nullptr) {
      return;
    }
    current_time_ += std::max(0.0, dt);
    UpdateAtCurrentTime(task);
  }

  template <typename TaskLike>
  void UpdateWithTime(double elapsed_time, TaskLike* task) {
    Update(elapsed_time, task);
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
  template <typename TaskLike>
  void UpdateAtCurrentTime(TaskLike* task) {
    const double eval_time = std::min(current_time_, duration_);

    Eigen::Quaterniond quat_des = Eigen::Quaterniond::Identity();
    Eigen::Vector3d ang_vel_des = Eigen::Vector3d::Zero();
    Eigen::Vector3d ang_acc_des = Eigen::Vector3d::Zero();
    curve_.Evaluate(eval_time, quat_des);
    curve_.GetAngularVelocity(eval_time, ang_vel_des);
    curve_.GetAngularAcceleration(eval_time, ang_acc_des);

    Eigen::VectorXd ori_des(4);
    ori_des << quat_des.normalized().coeffs();
    task->UpdateDesired(ori_des, ang_vel_des, ang_acc_des);

    if (eval_time >= duration_) {
      is_running_ = false;
    }
  }

  HermiteQuaternionCurve curve_;
  double duration_{0.0};
  double current_time_{0.0};
  bool is_running_{false};
};

} // namespace wbc
