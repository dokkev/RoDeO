// Copyright 2026
//
// Licensed under the Apache License, Version 2.0.

#pragma once

#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include "mppi_core/grasp/contact_stability_cost.hpp"
#include "mppi_core/grasp/grasp_contact_kinematics.hpp"

namespace mppi_core {

inline const char* ContactPresenceDebugName(ContactPresence presence) {
  switch (presence) {
    case ContactPresence::kUnknown:
      return "unknown";
    case ContactPresence::kNoContact:
      return "none";
    case ContactPresence::kLightContact:
      return "light";
    case ContactPresence::kStableContact:
      return "stable";
  }
  return "unknown";
}

inline std::string FormatDebugVector2(const Eigen::Vector2d& value) {
  std::ostringstream out;
  out << std::fixed << std::setprecision(6);
  if (!value.allFinite()) {
    out << "nan,nan";
    return out.str();
  }
  out << value.x() << "," << value.y();
  return out.str();
}

inline std::string MakeGraspTactileRolloutDebugTable(
    const GraspState& state,
    const PinocchioContactKinematicsContext& contact_kinematics,
    const std::vector<std::pair<std::string, Eigen::VectorXd>>&
        named_delta_q_tangent_actions,
    double dt,
    const GraspRolloutConfig& rollout_config = {},
    const ContactStabilityCostConfig& cost_config = {}) {
  std::ostringstream out;
  out << std::fixed << std::setprecision(6);
  out << "action contact normal_n confidence support area centroid edge_risk "
         "shear rot_shear incipient_slip contact_loss preload slip_cost total\n";

  for (const auto& named_action : named_delta_q_tangent_actions) {
    const auto& action_name = named_action.first;
    const auto& delta_q_tangent = named_action.second;

    const auto motions =
        ComputeContactPointMotions(state, delta_q_tangent, contact_kinematics);
    const TactileState predicted_tactile =
        StepTactileContactPatch(state.tactile, motions, dt, rollout_config);

    ContactStabilityCostDebug debug;
    const double total_cost =
        ComputeContactStabilityCost(predicted_tactile, cost_config, &debug);

    const Eigen::Vector2d centroid =
        predicted_tactile.has_centroid
            ? predicted_tactile.centroid_m
            : Eigen::Vector2d::Constant(
                  std::numeric_limits<double>::quiet_NaN());
    const Eigen::Vector2d shear =
        predicted_tactile.has_shear
            ? predicted_tactile.shear_displacement_m
            : Eigen::Vector2d::Zero();
    const double rotational_shear =
        predicted_tactile.has_rotational_shear
            ? predicted_tactile.rotational_shear_rad
            : 0.0;

    out << action_name << ' '
        << ContactPresenceDebugName(predicted_tactile.contact_presence) << ' '
        << predicted_tactile.normal_force_n << ' '
        << predicted_tactile.confidence << ' '
        << predicted_tactile.contact_support_count << ' '
        << predicted_tactile.contact_area_proxy << ' '
        << FormatDebugVector2(centroid) << ' '
        << predicted_tactile.edge_risk << ' '
        << FormatDebugVector2(shear) << ' '
        << rotational_shear << ' '
        << predicted_tactile.incipient_slip_score << ' '
        << debug.contact_loss_cost << ' '
        << debug.preload_cost << ' '
        << debug.slip_cost << ' '
        << total_cost << '\n';
  }

  return out.str();
}

}  // namespace mppi_core
