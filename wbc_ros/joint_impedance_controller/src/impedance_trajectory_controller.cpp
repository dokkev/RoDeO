#include "joint_impedance_controller/impedance_trajectory_controller.hpp"

#include <algorithm>
#include <cmath>

namespace {
constexpr double kMinDurationSec = 1e-3;
}

ImpedanceTrajectoryController::ImpedanceTrajectoryController(std::size_t dof)
    : dof_(dof == 0 ? 8 : dof) {
  stiffness_.assign(dof_, 0.0);
  damping_.assign(dof_, 0.0);
  measured_position_.assign(dof_, 0.0);
  measured_velocity_.assign(dof_, 0.0);
  desired_position_.assign(dof_, 0.0);
  desired_velocity_.assign(dof_, 0.0);
  desired_acceleration_.assign(dof_, 0.0);
  effort_ff_.assign(dof_, 0.0);
  start_position_.assign(dof_, 0.0);
  start_velocity_.assign(dof_, 0.0);
  goal_position_.assign(dof_, 0.0);
  goal_velocity_.assign(dof_, 0.0);
  goal_effort_ff_.assign(dof_, 0.0);
}

void ImpedanceTrajectoryController::setGains(const std::vector<double>& stiffness,
                                       const std::vector<double>& damping) {
  stiffness_ = sanitizeToDof(stiffness);
  damping_ = sanitizeToDof(damping);
}

void ImpedanceTrajectoryController::setMeasuredState(const std::vector<double>& position,
                                               const std::vector<double>& velocity) {
  measured_position_ = sanitizeToDof(position);
  measured_velocity_ = sanitizeToDof(velocity);
  has_measured_state_ = true;

  if (!has_desired_state_) {
    desired_position_ = measured_position_;
    desired_velocity_.assign(dof_, 0.0);
    desired_acceleration_.assign(dof_, 0.0);
    effort_ff_.assign(dof_, 0.0);
    has_desired_state_ = true;
  }
}

void ImpedanceTrajectoryController::setGoal(const std::vector<double>& target_position,
                                      double duration_sec,
                                      const std::vector<double>& effort_ff) {
  initializeDesiredFromMeasurementIfNeeded();

  start_position_ = has_measured_state_ ? measured_position_ : desired_position_;
  start_velocity_ = has_measured_state_ ? measured_velocity_ : desired_velocity_;

  goal_position_ = start_position_;
  const std::size_t position_len = std::min(dof_, target_position.size());
  std::copy_n(target_position.begin(), position_len, goal_position_.begin());
  goal_velocity_.assign(dof_, 0.0);
  goal_effort_ff_.assign(dof_, 0.0);
  const std::size_t effort_len = std::min(dof_, effort_ff.size());
  std::copy_n(effort_ff.begin(), effort_len, goal_effort_ff_.begin());

  duration_sec_ = std::max(duration_sec, kMinDurationSec);
  elapsed_sec_ = 0.0;
  mode_ = ExecutionMode::Executing;
}

void ImpedanceTrajectoryController::holdPosition() {
  initializeDesiredFromMeasurementIfNeeded();

  const auto& hold_source = has_measured_state_ ? measured_position_ : desired_position_;
  desired_position_ = hold_source;
  desired_velocity_.assign(dof_, 0.0);
  desired_acceleration_.assign(dof_, 0.0);
  effort_ff_.assign(dof_, 0.0);

  mode_ = ExecutionMode::Holding;
  elapsed_sec_ = 0.0;
  duration_sec_ = kMinDurationSec;
}

ImpedanceCommand ImpedanceTrajectoryController::update(double dt_sec) {
  initializeDesiredFromMeasurementIfNeeded();

  if (dt_sec < 0.0) {
    dt_sec = 0.0;
  }

  if (mode_ == ExecutionMode::Executing) {
    elapsed_sec_ += dt_sec;
    const double clamped_t = std::clamp(elapsed_sec_, 0.0, duration_sec_);
    sampleActiveTrajectory(clamped_t);
    effort_ff_ = goal_effort_ff_;

    if (elapsed_sec_ >= duration_sec_) {
      mode_ = ExecutionMode::Holding;
      desired_position_ = goal_position_;
      desired_velocity_.assign(dof_, 0.0);
      desired_acceleration_.assign(dof_, 0.0);
    }
  } else if (mode_ == ExecutionMode::Idle && has_measured_state_) {
    desired_position_ = measured_position_;
    desired_velocity_.assign(dof_, 0.0);
    desired_acceleration_.assign(dof_, 0.0);
    effort_ff_.assign(dof_, 0.0);
  }

  ImpedanceCommand cmd;
  cmd.position = desired_position_;
  cmd.velocity = desired_velocity_;
  cmd.stiffness = stiffness_;
  cmd.damping = damping_;
  cmd.effort_ff = effort_ff_;
  return cmd;
}

std::vector<double> ImpedanceTrajectoryController::sanitizeToDof(const std::vector<double>& in,
                                                           double fallback) const {
  std::vector<double> out(dof_, fallback);
  const std::size_t copy_len = std::min(dof_, in.size());
  std::copy_n(in.begin(), copy_len, out.begin());
  return out;
}

void ImpedanceTrajectoryController::initializeDesiredFromMeasurementIfNeeded() {
  if (has_desired_state_) {
    return;
  }

  if (has_measured_state_) {
    desired_position_ = measured_position_;
  } else {
    desired_position_.assign(dof_, 0.0);
  }

  desired_velocity_.assign(dof_, 0.0);
  desired_acceleration_.assign(dof_, 0.0);
  effort_ff_.assign(dof_, 0.0);
  has_desired_state_ = true;
}

void ImpedanceTrajectoryController::sampleActiveTrajectory(double t_sec) {
  const double T = std::max(duration_sec_, kMinDurationSec);
  const double s = std::clamp(t_sec / T, 0.0, 1.0);
  const double s2 = s * s;
  const double s3 = s2 * s;

  const double h00 = 2.0 * s3 - 3.0 * s2 + 1.0;
  const double h10 = s3 - 2.0 * s2 + s;
  const double h01 = -2.0 * s3 + 3.0 * s2;
  const double h11 = s3 - s2;

  const double dh00_dt = (6.0 * s2 - 6.0 * s) / T;
  const double dh10_dt = 3.0 * s2 - 4.0 * s + 1.0;
  const double dh01_dt = (-6.0 * s2 + 6.0 * s) / T;
  const double dh11_dt = 3.0 * s2 - 2.0 * s;

  const double d2h00_dt2 = (12.0 * s - 6.0) / (T * T);
  const double d2h10_dt2 = (6.0 * s - 4.0) / T;
  const double d2h01_dt2 = (-12.0 * s + 6.0) / (T * T);
  const double d2h11_dt2 = (6.0 * s - 2.0) / T;

  for (std::size_t i = 0; i < dof_; ++i) {
    const double p0 = start_position_[i];
    const double p1 = goal_position_[i];
    const double v0 = start_velocity_[i];
    const double v1 = goal_velocity_[i];

    desired_position_[i] = h00 * p0 + h10 * T * v0 + h01 * p1 + h11 * T * v1;
    desired_velocity_[i] = dh00_dt * p0 + dh10_dt * v0 + dh01_dt * p1 + dh11_dt * v1;
    desired_acceleration_[i] = d2h00_dt2 * p0 + d2h10_dt2 * v0 +
                               d2h01_dt2 * p1 + d2h11_dt2 * v1;
  }
}
