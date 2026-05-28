// Copyright 2026
//
// Licensed under the Apache License, Version 2.0.

#pragma once

#include <algorithm>
#include <cmath>

#include "mppi_core/tactile/tactile_state.hpp"

namespace mppi_core {

inline double Square(double x) { return x * x; }

inline double SquarePositive(double x) {
  return x > 0.0 ? x * x : 0.0;
}

struct ContactStabilityCostConfig {
  // Keep the first version intentionally small.
  // Do not add more weights unless there is experimental evidence.
  double w_contact_loss{100.0};
  double w_preload{1.0};
  double w_slip{10.0};

  double min_normal_force_n{0.2};
  double max_normal_force_n{5.0};
};

struct ContactStabilityCostDebug {
  // Optional diagnostics only. These are not separate cost weights.
  bool has_contact{false};
  double normal_force_n{0.0};
  double incipient_slip_score{0.0};
  double contact_area_proxy{0.0};
  double edge_risk{0.0};
  double confidence{0.0};

  double contact_loss_cost{0.0};
  double preload_cost{0.0};
  double slip_cost{0.0};
};

inline double ComputeContactStabilityCost(
    const TactileState& tactile,
    const ContactStabilityCostConfig& config = {},
    ContactStabilityCostDebug* debug = nullptr) {
  const bool has_contact = tactile.hasContact();

  const double normal_force_n =
      tactile.has_normal_force && std::isfinite(tactile.normal_force_n)
          ? std::max(0.0, tactile.normal_force_n)
          : 0.0;

  const double incipient_slip_score =
      std::isfinite(tactile.incipient_slip_score)
          ? std::max(0.0, tactile.incipient_slip_score)
          : 0.0;

  const double contact_area_proxy =
      std::isfinite(tactile.contact_area_proxy)
          ? std::clamp(tactile.contact_area_proxy, 0.0, 1.0)
          : 0.0;

  const double edge_risk =
      std::isfinite(tactile.edge_risk) ? std::max(0.0, tactile.edge_risk)
                                       : 0.0;

  const double confidence =
      std::isfinite(tactile.confidence)
          ? std::clamp(tactile.confidence, 0.0, 1.0)
          : 0.0;

  const double contact_loss_cost = has_contact ? 0.0 : 1.0;

  const double low_force_cost =
      SquarePositive(config.min_normal_force_n - normal_force_n);
  const double overload_cost =
      SquarePositive(normal_force_n - config.max_normal_force_n);

  // Safe band cost:
  //   zero inside [min_normal_force_n, max_normal_force_n]
  //   penalize low force strongly
  //   penalize overload mildly
  const double preload_cost = low_force_cost + 0.25 * overload_cost;

  // Slip is meaningful only in contact.
  const double slip_cost =
      has_contact ? Square(incipient_slip_score) : 0.0;

  if (debug != nullptr) {
    debug->has_contact = has_contact;
    debug->normal_force_n = normal_force_n;
    debug->incipient_slip_score = incipient_slip_score;
    debug->contact_area_proxy = contact_area_proxy;
    debug->edge_risk = edge_risk;
    debug->confidence = confidence;
    debug->contact_loss_cost = contact_loss_cost;
    debug->preload_cost = preload_cost;
    debug->slip_cost = slip_cost;
  }

  return config.w_contact_loss * contact_loss_cost +
         config.w_preload * preload_cost +
         config.w_slip * slip_cost;
}

}  // namespace mppi_core
