// Copyright 2026
//
// Licensed under the Apache License, Version 2.0.

#pragma once

#include <algorithm>
#include <cmath>

#include "mppi_core/tactile/tactile_state.hpp"

namespace mppi_core {

struct ContactForceCorrectionState {
  double normal_force_bias_n{0.0};
};

struct ContactForceCorrectionConfig {
  bool enabled{true};

  double bias_update_rate{0.05};
  double max_abs_bias_n{5.0};

  // Only update if measurement is valid and contact exists.
  bool update_only_in_contact{true};
};

inline double ClampContactForceBias(double bias_n, double max_abs_bias_n) {
  if (!std::isfinite(bias_n)) {
    return 0.0;
  }
  const double limit =
      std::max(0.0, std::isfinite(max_abs_bias_n) ? max_abs_bias_n : 0.0);
  return std::clamp(bias_n, -limit, limit);
}

inline void UpdateContactForceCorrection(
    double predicted_normal_force_n, double measured_normal_force_n,
    const TactileState& measured_tactile,
    const ContactForceCorrectionConfig& config,
    ContactForceCorrectionState* state) {
  if (state == nullptr || !config.enabled ||
      !std::isfinite(predicted_normal_force_n) ||
      !std::isfinite(measured_normal_force_n) || !measured_tactile.valid) {
    return;
  }

  if (config.update_only_in_contact && !measured_tactile.hasContact()) {
    return;
  }

  const double alpha = std::clamp(
      std::isfinite(config.bias_update_rate) ? config.bias_update_rate : 0.0,
      0.0, 1.0);
  const double error_n = measured_normal_force_n - predicted_normal_force_n;
  const double next_bias_n =
      (1.0 - alpha) * state->normal_force_bias_n + alpha * error_n;

  state->normal_force_bias_n =
      ClampContactForceBias(next_bias_n, config.max_abs_bias_n);
}

inline double ApplyContactForceCorrection(
    double predicted_normal_force_n, const ContactForceCorrectionState& state) {
  if (!std::isfinite(predicted_normal_force_n) ||
      !std::isfinite(state.normal_force_bias_n)) {
    return 0.0;
  }
  return std::max(0.0, predicted_normal_force_n + state.normal_force_bias_n);
}

}  // namespace mppi_core
