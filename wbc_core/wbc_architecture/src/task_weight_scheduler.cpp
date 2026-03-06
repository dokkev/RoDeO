/**
 * @file wbc_core/wbc_architecture/src/task_weight_scheduler.cpp
 * @brief TaskWeightScheduler implementation — cosine-ramp weight transitions.
 */
#include "wbc_architecture/task_weight_scheduler.hpp"

#include <cassert>
#include <cmath>

namespace wbc {

void TaskWeightScheduler::RegisterTask(Task* task) {
  assert(task != nullptr);

  // Check for duplicates via linear scan (init-only, small N).
  for (const auto& ramp : ramps_) {
    if (ramp.task == task) return;
  }

  WeightRamp ramp;
  ramp.task = task;
  ramp.w_start = task->Weight();
  ramp.w_target = task->Weight();
  ramp.active = false;
  ramps_.push_back(std::move(ramp));
  has_target_.push_back(false);
}

void TaskWeightScheduler::ScheduleTransition(
    const std::vector<Task*>& state_motion,
    const std::vector<const TaskConfig*>& state_cfg,
    const std::unordered_map<Task*, TaskConfig>& default_cfgs,
    double current_time, double ramp_duration) {

  // Phase 1: reset all marks to false (no allocation — vector already sized).
  for (std::size_t i = 0; i < has_target_.size(); ++i) {
    has_target_[i] = false;
  }

  // Phase 2: mark tasks present in the new state and set their target weight.
  for (std::size_t mi = 0; mi < state_motion.size(); ++mi) {
    Task* task = state_motion[mi];
    if (task == nullptr) continue;

    // Find this task's ramp index by linear scan (small N, typically <10).
    for (std::size_t ri = 0; ri < ramps_.size(); ++ri) {
      if (ramps_[ri].task != task) continue;

      has_target_[ri] = true;

      // Determine target weight: per-state override > pool default > current.
      const Eigen::VectorXd* target = nullptr;
      if (mi < state_cfg.size() && state_cfg[mi] != nullptr) {
        target = &state_cfg[mi]->weight;
      } else {
        auto it = default_cfgs.find(task);
        if (it != default_cfgs.end()) {
          target = &it->second.weight;
        }
      }

      ramps_[ri].w_start = task->Weight();
      if (target != nullptr) {
        ramps_[ri].w_target = *target;
      } else {
        ramps_[ri].w_target = task->Weight();  // no change
      }
      ramps_[ri].t_start = current_time;
      ramps_[ri].duration = ramp_duration;
      ramps_[ri].active = true;
      break;
    }
  }

  // Phase 3: tasks NOT in the new state ramp to their pool default weight.
  active_count_ = 0;
  for (std::size_t i = 0; i < ramps_.size(); ++i) {
    if (!has_target_[i]) {
      ramps_[i].w_start = ramps_[i].task->Weight();
      auto it = default_cfgs.find(ramps_[i].task);
      if (it != default_cfgs.end()) {
        ramps_[i].w_target = it->second.weight;
      } else {
        ramps_[i].w_target = ramps_[i].task->Weight();  // no change
      }
      ramps_[i].t_start = current_time;
      ramps_[i].duration = ramp_duration;
      ramps_[i].active = true;
    }
    if (ramps_[i].active) ++active_count_;
  }
}

bool TaskWeightScheduler::Tick(double current_time) {
  active_count_ = 0;

  for (auto& ramp : ramps_) {
    if (!ramp.active) continue;

    const double elapsed = current_time - ramp.t_start;
    const double alpha = (ramp.duration > 0.0)
                             ? elapsed / ramp.duration
                             : 1.0;

    if (alpha >= 1.0) {
      // Ramp complete — snap to target (clamped).
      ramp.task->SetWeight(ramp.w_target.cwiseMax(weight_min_).cwiseMin(weight_max_));
      ramp.active = false;
      continue;
    }

    // Cosine interpolation: w = w_start + (w_target - w_start) * ramp(alpha)
    const double s = CosineRamp(alpha);
    const Eigen::VectorXd w =
        ramp.w_start + s * (ramp.w_target - ramp.w_start);

    // Clamp to [weight_min, weight_max] for numerical stability and ratio guard.
    ramp.task->SetWeight(w.cwiseMax(weight_min_).cwiseMin(weight_max_));
    ++active_count_;
  }

  return active_count_ > 0;
}

}  // namespace wbc
