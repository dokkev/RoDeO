// Copyright 2026
//
// Licensed under the Apache License, Version 2.0.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>

#include "mppi_core/tactile/nari_touch_state.hpp"
#include "mppi_core/tactile/tactile_state.hpp"

namespace mppi_core {

struct NariTouchAdapterConfig {
  double slip_velocity_weight{1.0};

  // Reduced unit layout extents. These are not the raw 12 x 24 mm sensing area.
  double reduced_half_width_x_m{0.003};
  double reduced_half_width_y_m{0.009};
};

inline ContactPresence ToContactPresence(NariTouchContactState state) {
  switch (state) {
    case NariTouchContactState::kNoContact:
      return ContactPresence::kNoContact;
    case NariTouchContactState::kFewContacts:
      return ContactPresence::kLightContact;
    case NariTouchContactState::kEnoughContacts:
      return ContactPresence::kStableContact;
  }
  return ContactPresence::kUnknown;
}

inline double ComputeNariTouchConfidence(
    const NariTouchState& nari, const TactileState& tactile) {
  double confidence = 0.0;

  if (nari.hasContact()) {
    confidence += 0.25;
  }

  if (nari.contact_state == NariTouchContactState::kEnoughContacts) {
    confidence += 0.40;
  } else if (nari.contact_state == NariTouchContactState::kFewContacts) {
    confidence += 0.20;
  }

  if (tactile.has_centroid) {
    confidence += 0.20;
  }

  if (tactile.has_normal_force && tactile.normal_force_n > 0.0) {
    confidence += 0.15;
  }

  return std::clamp(confidence, 0.0, 1.0);
}

inline double ComputeNariTouchEdgeRisk(
    const TactileState& tactile, const NariTouchAdapterConfig& config) {
  if (!tactile.has_centroid) {
    return 0.0;
  }

  const double x_norm = std::abs(tactile.centroid_m.x()) /
                        std::max(1.0e-9, config.reduced_half_width_x_m);
  const double y_norm = std::abs(tactile.centroid_m.y()) /
                        std::max(1.0e-9, config.reduced_half_width_y_m);

  return std::clamp(std::max(x_norm, y_norm), 0.0, 2.0);
}

inline TactileState ConvertNariTouchToTactileState(
    const NariTouchState& nari, const std::string& frame_name,
    double stamp_sec, const NariTouchAdapterConfig& config = {}) {
  TactileState out;

  out.valid = true;
  out.stamp_sec = stamp_sec;
  out.frame_name = frame_name;

  out.contact_presence = ToContactPresence(nari.contact_state);

  out.has_normal_force = true;
  out.normal_force_n = ComputeNariTouchTotalNormalForceN(nari);

  Eigen::Vector2d centroid_m;
  out.has_centroid = ComputeNariTouchContactCentroidM(nari, &centroid_m);
  if (out.has_centroid) {
    out.centroid_m = centroid_m;
    out.centroid_velocity_mps = nari.centroid_velocity_mps;
  }

  out.contact_points.clear();
  for (std::size_t i = 0; i < nari.units.size(); ++i) {
    const auto& unit = nari.units[i];
    if (!unit.contact) {
      continue;
    }

    TactileContactPoint point;
    point.active = true;
    point.support_index = i;

    Eigen::Vector2d xy = unit.position_m;
    if (unit.cop.allFinite()) {
      xy += unit.cop;
    }
    point.position_sensor_m = Eigen::Vector3d{xy.x(), xy.y(), 0.0};

    if (std::isfinite(unit.normal_force) && unit.normal_force > 0.0) {
      point.has_normal_force = true;
      point.normal_force_n = unit.normal_force;
    }

    point.confidence = 1.0;
    out.contact_points.push_back(point);
  }

  const auto shear = nari.slip_state.head<2>();
  if (shear.allFinite()) {
    out.has_shear = true;
    out.shear_displacement_m = shear;
  }

  if (std::isfinite(nari.slip_state.z())) {
    out.has_rotational_shear = true;
    out.rotational_shear_rad = nari.slip_state.z();
  }

  const auto shear_velocity = nari.slip_velocity_state.head<2>();
  if (shear_velocity.allFinite()) {
    out.has_shear_velocity = true;
    out.shear_velocity_mps = shear_velocity;
  }

  if (std::isfinite(nari.slip_velocity_state.z())) {
    out.has_rotational_shear_velocity = true;
    out.rotational_shear_velocity_radps = nari.slip_velocity_state.z();
  }

  double slip_score_squared = 0.0;
  if (out.has_shear) {
    slip_score_squared += out.shear_displacement_m.squaredNorm();
  }
  if (out.has_rotational_shear) {
    slip_score_squared += out.rotational_shear_rad * out.rotational_shear_rad;
  }
  out.slip_score = std::sqrt(slip_score_squared);

  double slip_velocity_score_squared = 0.0;
  if (out.has_shear_velocity) {
    slip_velocity_score_squared += out.shear_velocity_mps.squaredNorm();
  }
  if (out.has_rotational_shear_velocity) {
    slip_velocity_score_squared += out.rotational_shear_velocity_radps *
                                   out.rotational_shear_velocity_radps;
  }
  out.slip_velocity_score = std::sqrt(slip_velocity_score_squared);
  out.incipient_slip_score =
      out.slip_score + std::max(0.0, config.slip_velocity_weight) *
                           out.slip_velocity_score;

  out.contact_support_count = out.activeContactPointCount();
  out.support_count = kNariTouchUnitCount;

  if (out.support_count > 0) {
    out.contact_area_proxy =
        static_cast<double>(out.contact_support_count) /
        static_cast<double>(out.support_count);
  }

  out.edge_risk = ComputeNariTouchEdgeRisk(out, config);
  out.confidence = ComputeNariTouchConfidence(nari, out);

  return out;
}

}  // namespace mppi_core
