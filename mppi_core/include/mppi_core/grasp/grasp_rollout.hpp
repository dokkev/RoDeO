// Copyright 2026
//
// Licensed under the Apache License, Version 2.0.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include <Eigen/Core>

#include "mppi_core/grasp/grasp_state.hpp"

namespace mppi_core {

struct ContactPointMotion {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  std::size_t support_index{0};
  Eigen::Vector3d position_sensor_m{Eigen::Vector3d::Zero()};

  // Displacement of this contact/support point over the current rollout step,
  // expressed in the tactile sensor frame.
  // Convention: positive z means closing/compression along the tactile normal.
  Eigen::Vector3d delta_position_sensor_m{Eigen::Vector3d::Zero()};
};

struct GraspRolloutConfig {
  // Reduced tactile support geometry limits.
  double sensor_half_width_x_m{0.003};
  double sensor_half_width_y_m{0.009};
  double edge_risk_limit{1.0};

  // Contact classification. These are rollout model parameters, not costs.
  std::size_t min_stable_support_count{4};
  double min_contact_confidence{1.0e-3};

  // Geometry update gains.
  double centroid_motion_gain{1.0};
  double tangential_shear_gain{1.0};
  double rotational_shear_gain{1.0};

  // Normal preload proxy. This is not exact force estimation.
  double normal_force_gain_n_per_m{100.0};
  double max_normal_force_n{10.0};

  // Confidence update from motion tendency.
  double closing_confidence_gain_per_m{20.0};
  double separating_confidence_loss_per_m{200.0};
  double tangential_confidence_loss_per_m{5.0};
  double edge_confidence_loss_gain{0.25};

  // State bounds for one-step rollout robustness.
  double max_centroid_step_m{1.0e-2};
  double max_shear_m{2.0e-2};
  double max_rotational_shear_rad{1.0};

  // Normalization references for predicted tactile slip scores.
  double shear_ref_m{1.0e-3};
  double rotational_shear_ref_rad{2.0e-2};
};

inline double Clamp01(double x) {
  if (!std::isfinite(x)) {
    return 0.0;
  }
  return std::clamp(x, 0.0, 1.0);
}

inline bool IsFiniteContactPointMotion(const ContactPointMotion& motion) {
  return motion.position_sensor_m.allFinite() &&
         motion.delta_position_sensor_m.allFinite();
}

inline Eigen::Vector2d ClampPatchVectorNorm(const Eigen::Vector2d& value,
                                            double max_norm) {
  if (!value.allFinite()) {
    return Eigen::Vector2d::Zero();
  }

  const double safe_max_norm =
      std::max(0.0, std::isfinite(max_norm) ? max_norm : 0.0);
  const double norm = value.norm();
  if (norm <= 1.0e-12) {
    return value;
  }
  if (safe_max_norm <= 0.0) {
    return Eigen::Vector2d::Zero();
  }
  if (norm <= safe_max_norm) {
    return value;
  }
  return value * (safe_max_norm / norm);
}

inline double ComputeTactilePatchEdgeRisk(
    const Eigen::Vector2d& centroid_m,
    const GraspRolloutConfig& config) {
  if (!centroid_m.allFinite()) {
    return 0.0;
  }

  const double x_norm =
      std::abs(centroid_m.x()) /
      std::max(1.0e-9, config.sensor_half_width_x_m);
  const double y_norm =
      std::abs(centroid_m.y()) /
      std::max(1.0e-9, config.sensor_half_width_y_m);

  return std::clamp(std::max(x_norm, y_norm), 0.0, 2.0);
}

inline double ComputeTactileSupportScore(
    std::size_t contact_support_count,
    const GraspRolloutConfig& config) {
  if (config.min_stable_support_count == 0) {
    return 0.0;
  }
  return Clamp01(static_cast<double>(contact_support_count) /
                 static_cast<double>(config.min_stable_support_count));
}

inline ContactPresence PredictContactPresence(
    const TactileState& tactile,
    const GraspRolloutConfig& config) {
  if (!tactile.valid ||
      !std::isfinite(tactile.normal_force_n) ||
      tactile.normal_force_n <= 0.0 ||
      Clamp01(tactile.confidence) <= config.min_contact_confidence ||
      tactile.contact_support_count == 0) {
    return ContactPresence::kNoContact;
  }

  if (ComputeTactileSupportScore(tactile.contact_support_count, config) >=
      1.0) {
    return ContactPresence::kStableContact;
  }
  return ContactPresence::kLightContact;
}

inline double ComputePatchRolloutSlipScore(
    const TactileState& tactile,
    const GraspRolloutConfig& config = {}) {
  const double shear_ref_m =
      std::max(1.0e-9, std::isfinite(config.shear_ref_m)
                            ? config.shear_ref_m
                            : 1.0e-3);
  const double rotational_ref_rad =
      std::max(1.0e-9, std::isfinite(config.rotational_shear_ref_rad)
                            ? config.rotational_shear_ref_rad
                            : 2.0e-2);

  const double shear_score =
      tactile.has_shear && tactile.shear_displacement_m.allFinite()
          ? tactile.shear_displacement_m.norm() / shear_ref_m
          : 0.0;
  const double rotational_score =
      tactile.has_rotational_shear &&
              std::isfinite(tactile.rotational_shear_rad)
          ? std::abs(tactile.rotational_shear_rad) / rotational_ref_rad
          : 0.0;

  return shear_score + rotational_score;
}

inline void RefreshPredictedTactileFields(
    TactileState* tactile, const GraspRolloutConfig& config) {
  if (tactile == nullptr) {
    return;
  }

  tactile->valid = true;

  if (!std::isfinite(tactile->normal_force_n)) {
    tactile->normal_force_n = 0.0;
  }
  tactile->has_normal_force = true;
  tactile->normal_force_n =
      std::clamp(tactile->normal_force_n, 0.0,
                 std::max(0.0, config.max_normal_force_n));

  if (!tactile->centroid_m.allFinite()) {
    tactile->has_centroid = false;
    tactile->centroid_m = Eigen::Vector2d::Zero();
  }

  if (!tactile->shear_displacement_m.allFinite()) {
    tactile->has_shear = false;
    tactile->shear_displacement_m = Eigen::Vector2d::Zero();
  } else {
    tactile->shear_displacement_m =
        ClampPatchVectorNorm(tactile->shear_displacement_m,
                             config.max_shear_m);
  }

  if (!std::isfinite(tactile->rotational_shear_rad)) {
    tactile->has_rotational_shear = false;
    tactile->rotational_shear_rad = 0.0;
  } else {
    const double max_rot =
        std::max(0.0, std::isfinite(config.max_rotational_shear_rad)
                          ? config.max_rotational_shear_rad
                          : 0.0);
    tactile->rotational_shear_rad =
        std::clamp(tactile->rotational_shear_rad, -max_rot, max_rot);
  }

  tactile->confidence = Clamp01(tactile->confidence);
  if (tactile->has_centroid) {
    tactile->edge_risk =
        ComputeTactilePatchEdgeRisk(tactile->centroid_m, config);
  } else if (!std::isfinite(tactile->edge_risk)) {
    tactile->edge_risk = 0.0;
  } else {
    tactile->edge_risk = std::max(0.0, tactile->edge_risk);
  }

  if (tactile->normal_force_n <= 0.0 ||
      tactile->confidence <= config.min_contact_confidence) {
    tactile->contact_support_count = 0;
    for (auto& point : tactile->contact_points) {
      point.active = false;
    }
  }

  if (tactile->support_count > 0) {
    tactile->contact_area_proxy =
        Clamp01(static_cast<double>(tactile->contact_support_count) /
                static_cast<double>(tactile->support_count));
  } else {
    tactile->contact_area_proxy = 0.0;
  }

  tactile->contact_presence = PredictContactPresence(*tactile, config);
  tactile->slip_score = ComputePatchRolloutSlipScore(*tactile, config);
  tactile->slip_velocity_score =
      tactile->has_shear_velocity && tactile->shear_velocity_mps.allFinite()
          ? tactile->shear_velocity_mps.norm()
          : 0.0;
  tactile->incipient_slip_score = tactile->slip_score;
}

inline double EstimateRotationalPatchMotionRad(
    const std::vector<ContactPointMotion>& motions,
    const Eigen::Vector2d& center_m,
    const Eigen::Vector2d& mean_tangent_delta_m) {
  double numerator = 0.0;
  double denominator = 0.0;

  for (const auto& motion : motions) {
    if (!IsFiniteContactPointMotion(motion)) {
      continue;
    }

    const Eigen::Vector2d radius_m =
        motion.position_sensor_m.head<2>() - center_m;
    const Eigen::Vector2d tangent_delta_m =
        motion.delta_position_sensor_m.head<2>() - mean_tangent_delta_m;

    numerator += radius_m.x() * tangent_delta_m.y() -
                 radius_m.y() * tangent_delta_m.x();
    denominator += radius_m.squaredNorm();
  }

  if (denominator <= 1.0e-12 || !std::isfinite(numerator)) {
    return 0.0;
  }
  return numerator / denominator;
}

inline TactileState StepTactileContactPatch(
    const TactileState& tactile,
    const std::vector<ContactPointMotion>& motions,
    double dt,
    const GraspRolloutConfig& config = {}) {
  TactileState out = tactile;

  if (!std::isfinite(dt) || dt <= 0.0 || !tactile.valid ||
      !tactile.hasContact()) {
    RefreshPredictedTactileFields(&out, config);
    return out;
  }

  Eigen::Vector2d position_sum_m = Eigen::Vector2d::Zero();
  Eigen::Vector2d tangent_delta_sum_m = Eigen::Vector2d::Zero();
  double normal_delta_sum_m = 0.0;
  std::size_t valid_motion_count = 0;

  for (const auto& motion : motions) {
    if (!IsFiniteContactPointMotion(motion)) {
      continue;
    }

    position_sum_m += motion.position_sensor_m.head<2>();
    tangent_delta_sum_m += motion.delta_position_sensor_m.head<2>();
    normal_delta_sum_m += motion.delta_position_sensor_m.z();
    ++valid_motion_count;
  }

  if (valid_motion_count == 0) {
    RefreshPredictedTactileFields(&out, config);
    return out;
  }

  const double count = static_cast<double>(valid_motion_count);
  const Eigen::Vector2d mean_position_m = position_sum_m / count;
  const Eigen::Vector2d mean_tangent_delta_m = tangent_delta_sum_m / count;
  const double mean_normal_delta_m = normal_delta_sum_m / count;
  const Eigen::Vector2d base_centroid_m =
      tactile.has_centroid && tactile.centroid_m.allFinite()
          ? tactile.centroid_m
          : mean_position_m;

  const Eigen::Vector2d centroid_delta_m = ClampPatchVectorNorm(
      config.centroid_motion_gain * mean_tangent_delta_m,
      config.max_centroid_step_m);
  out.has_centroid = true;
  out.centroid_m = base_centroid_m + centroid_delta_m;
  out.centroid_velocity_mps = Eigen::Vector2d::Zero();
  if (dt > 0.0) {
    out.centroid_velocity_mps = centroid_delta_m / dt;
  }

  const Eigen::Vector2d current_shear_m =
      tactile.has_shear && tactile.shear_displacement_m.allFinite()
          ? tactile.shear_displacement_m
          : Eigen::Vector2d::Zero();
  const Eigen::Vector2d shear_delta_m =
      config.tangential_shear_gain * mean_tangent_delta_m;
  out.has_shear = true;
  out.shear_displacement_m =
      ClampPatchVectorNorm(current_shear_m + shear_delta_m,
                           config.max_shear_m);
  out.has_shear_velocity = true;
  out.shear_velocity_mps = Eigen::Vector2d::Zero();
  if (dt > 0.0) {
    out.shear_velocity_mps = shear_delta_m / dt;
  }

  const double rotation_delta_rad =
      config.rotational_shear_gain *
      EstimateRotationalPatchMotionRad(motions, base_centroid_m,
                                       mean_tangent_delta_m);
  const double current_rotation_rad =
      tactile.has_rotational_shear &&
              std::isfinite(tactile.rotational_shear_rad)
          ? tactile.rotational_shear_rad
          : 0.0;
  out.has_rotational_shear =
      tactile.has_rotational_shear || std::abs(rotation_delta_rad) > 0.0;
  out.rotational_shear_rad = current_rotation_rad + rotation_delta_rad;
  out.has_rotational_shear_velocity = true;
  out.rotational_shear_velocity_radps =
      dt > 0.0 ? rotation_delta_rad / dt : 0.0;

  const double current_normal_force_n =
      std::isfinite(tactile.normal_force_n)
          ? std::max(0.0, tactile.normal_force_n)
          : 0.0;
  out.normal_force_n =
      current_normal_force_n +
      config.normal_force_gain_n_per_m * mean_normal_delta_m;

  double confidence = Clamp01(tactile.confidence);
  if (mean_normal_delta_m >= 0.0) {
    confidence += config.closing_confidence_gain_per_m * mean_normal_delta_m;
  } else {
    confidence -= config.separating_confidence_loss_per_m *
                  std::abs(mean_normal_delta_m);
  }
  confidence -=
      config.tangential_confidence_loss_per_m * mean_tangent_delta_m.norm();

  if (out.has_centroid) {
    out.edge_risk = ComputeTactilePatchEdgeRisk(out.centroid_m, config);
    confidence -= config.edge_confidence_loss_gain *
                  std::max(0.0,
                           out.edge_risk - config.edge_risk_limit);
  }
  out.confidence = Clamp01(confidence);

  RefreshPredictedTactileFields(&out, config);
  return out;
}

inline GraspState StepGraspTactilePatch(
    const GraspState& state,
    const Eigen::VectorXd& q,
    const Eigen::VectorXd& dq,
    const Eigen::VectorXd& tau,
    const std::vector<ContactPointMotion>& motions,
    double dt,
    const GraspRolloutConfig& config = {}) {
  return MakeGraspState(
      q, dq, tau, StepTactileContactPatch(state.tactile, motions, dt, config));
}

}  // namespace mppi_core
