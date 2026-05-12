#pragma once

#include <cstddef>
#include <vector>

struct ImpedanceCommand {
  std::vector<double> position;
  std::vector<double> velocity;
  std::vector<double> stiffness;
  std::vector<double> damping;
  std::vector<double> effort_ff;
};

class ImpedanceTrajectoryController {
public:
  enum class ExecutionMode { Idle, Executing, Holding };

  explicit ImpedanceTrajectoryController(std::size_t dof = 8);

  void setGains(const std::vector<double>& stiffness, const std::vector<double>& damping);
  void setMeasuredState(const std::vector<double>& position, const std::vector<double>& velocity);
  void setGoal(const std::vector<double>& target_position,
               double duration_sec,
               const std::vector<double>& effort_ff = {});
  void holdPosition();

  ImpedanceCommand update(double dt_sec);

  std::size_t dof() const { return dof_; }
  ExecutionMode mode() const { return mode_; }

private:
  std::vector<double> sanitizeToDof(const std::vector<double>& in, double fallback = 0.0) const;
  void initializeDesiredFromMeasurementIfNeeded();
  void sampleActiveTrajectory(double t_sec);

  std::size_t dof_{8};
  ExecutionMode mode_{ExecutionMode::Idle};

  std::vector<double> stiffness_;
  std::vector<double> damping_;

  std::vector<double> measured_position_;
  std::vector<double> measured_velocity_;
  std::vector<double> desired_position_;
  std::vector<double> desired_velocity_;
  std::vector<double> desired_acceleration_;
  std::vector<double> effort_ff_;

  std::vector<double> start_position_;
  std::vector<double> start_velocity_;
  std::vector<double> goal_position_;
  std::vector<double> goal_velocity_;
  std::vector<double> goal_effort_ff_;

  bool has_measured_state_{false};
  bool has_desired_state_{false};

  double elapsed_sec_{0.0};
  double duration_sec_{0.25};
};
