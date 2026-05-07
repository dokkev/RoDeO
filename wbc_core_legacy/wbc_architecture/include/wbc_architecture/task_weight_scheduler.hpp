/**
 * @file wbc_core/wbc_architecture/include/wbc_architecture/task_weight_scheduler.hpp
 * @brief Smooth cosine interpolation of task weights during state transitions.
 *
 * All tasks remain registered in the formulation at all times. State changes
 * trigger weight ramps: previous-state tasks ramp down while new-state tasks
 * ramp up, producing a smooth cross-fade with no discontinuity.
 */
#pragma once

#include <vector>

#include <Eigen/Dense>

#include "wbc_formulation/interface/task.hpp"

namespace wbc {

/**
 * @brief Per-task ramp entry managed by the scheduler.
 *
 * Stores start/target weights and timing for cosine interpolation.
 * All members are pre-allocated — no heap allocation during scheduling.
 */
struct WeightRamp {
  Task* task{nullptr};
  Eigen::VectorXd w_start;    ///< weight at ramp begin
  Eigen::VectorXd w_target;   ///< weight at ramp end
  double t_start{0.0};        ///< wall time when ramp begins
  double duration{0.0};        ///< ramp duration in seconds
  bool active{false};          ///< true while interpolating
};

/**
 * @brief Central scheduler for task weight transitions.
 *
 * Usage:
 *   1. At init: call RegisterTask() for every task in the pool.
 *   2. On state change: call ScheduleTransition() with the new state's task/weight arrays.
 *   3. Every tick: call Tick(current_time) to advance all active ramps.
 *
 * The scheduler writes directly to Task::SetWeight() — no intermediate buffer.
 * Zero-allocation on the hot path: all ramp entries are pre-allocated at RegisterTask() time.
 */
class TaskWeightScheduler {
public:
  static constexpr double kMinWeight = 1e-6;
  static constexpr double kDefaultRampDuration = 0.3;  // seconds

  /**
   * @brief Register a task for weight scheduling.
   * Must be called once per task at initialization (before any Tick).
   */
  void RegisterTask(Task* task);

  /**
   * @brief Schedule weight transitions for a state change (zero-allocation).
   *
   * All registered tasks are ramped: tasks in the state's motion list get their
   * configured target weight; tasks not listed fall back to their pool default.
   *
   * @param state_motion  Array of motion task pointers from StateConfig::motion.
   * @param state_cfg     Parallel array of per-state override configs (nullable entries).
   * @param default_cfgs  Fallback default task configs from the pool.
   * @param current_time  Current wall time in seconds.
   * @param ramp_duration Duration of the cosine ramp in seconds.
   */
  void ScheduleTransition(
      const std::vector<Task*>& state_motion,
      const std::vector<const TaskConfig*>& state_cfg,
      const std::unordered_map<Task*, TaskConfig>& default_cfgs,
      double current_time,
      double ramp_duration = kDefaultRampDuration);

  /**
   * @brief Advance all active ramps and write weights to tasks.
   *
   * @param current_time Current wall time in seconds.
   * @return true if any ramp is still active (transitioning).
   */
  bool Tick(double current_time);

  /**
   * @brief Check if any ramp is still transitioning.
   */
  bool IsTransitioning() const { return active_count_ > 0; }

  void SetRampDuration(double d) { default_ramp_duration_ = d; }
  double GetRampDuration() const { return default_ramp_duration_; }

  /// Weight clamping bounds (Weight Ratio Guard).
  /// All weights are clamped to [weight_min, weight_max] after interpolation.
  void SetWeightBounds(double w_min, double w_max) {
    weight_min_ = w_min;
    weight_max_ = w_max;
  }
  double GetWeightMin() const { return weight_min_; }
  double GetWeightMax() const { return weight_max_; }

private:
  /**
   * @brief Cosine interpolation: smooth ramp from 0 to 1 over [0, 1].
   * Returns 0.5 * (1 - cos(pi * alpha)).
   */
  static double CosineRamp(double alpha) {
    if (alpha <= 0.0) return 0.0;
    if (alpha >= 1.0) return 1.0;
    return 0.5 * (1.0 - std::cos(M_PI * alpha));
  }

  std::vector<WeightRamp> ramps_;
  int active_count_{0};
  double default_ramp_duration_{kDefaultRampDuration};
  double weight_min_{kMinWeight};   ///< Lower clamp for all weights
  double weight_max_{1e4};          ///< Upper clamp for all weights

  // Pre-allocated scratch: marks which ramps got an explicit target this transition.
  std::vector<bool> has_target_;
};

}  // namespace wbc
