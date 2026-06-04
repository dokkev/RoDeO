// Copyright 2026
//
// Licensed under the Apache License, Version 2.0.

#include "mppi_core/costs/grasp_stability_cost.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace mppi_core {
namespace {

double Relu(double value) {
  return std::max(0.0, value);
}

double Square(double value) {
  return value * value;
}

bool IsFiniteAndNonnegative(double value) {
  return std::isfinite(value) && value >= 0.0;
}

double FiniteNonnegativeOrZero(double value) {
  if (!std::isfinite(value)) {
    return 0.0;
  }
  return std::max(0.0, value);
}

double TactileNormalForceN(const TactileState& tactile) {
  if (!tactile.has_normal_force || !std::isfinite(tactile.normal_force_n)) {
    return 0.0;
  }
  return std::max(0.0, tactile.normal_force_n);
}

double TactileSlipScore(const TactileState& tactile) {
  const double provided_score = FiniteNonnegativeOrZero(tactile.slip_score);
  if (provided_score > 0.0) {
    return provided_score;
  }

  double squared_score = 0.0;
  if (tactile.has_shear && tactile.shear_displacement_m.allFinite()) {
    squared_score += tactile.shear_displacement_m.squaredNorm();
  }
  if (tactile.has_rotational_shear &&
      std::isfinite(tactile.rotational_shear_rad)) {
    squared_score += tactile.rotational_shear_rad *
                     tactile.rotational_shear_rad;
  }
  return std::sqrt(squared_score);
}

double TactileSlipVelocityScore(const TactileState& tactile) {
  const double provided_score =
      FiniteNonnegativeOrZero(tactile.slip_velocity_score);
  if (provided_score > 0.0) {
    return provided_score;
  }

  double squared_score = 0.0;
  if (tactile.has_shear_velocity &&
      tactile.shear_velocity_mps.allFinite()) {
    squared_score += tactile.shear_velocity_mps.squaredNorm();
  }
  if (tactile.has_rotational_shear_velocity &&
      std::isfinite(tactile.rotational_shear_velocity_radps)) {
    squared_score += tactile.rotational_shear_velocity_radps *
                     tactile.rotational_shear_velocity_radps;
  }
  return std::sqrt(squared_score);
}

double TactileSlipRisk(const TactileState& tactile, double velocity_weight) {
  const double computed_risk =
      TactileSlipScore(tactile) +
      std::max(0.0, velocity_weight) * TactileSlipVelocityScore(tactile);
  return std::max(computed_risk,
                  FiniteNonnegativeOrZero(tactile.incipient_slip_score));
}

bool TactileContactCentroidM(const TactileState& tactile,
                             Eigen::Vector2d* centroid_m) {
  if (centroid_m == nullptr || !tactile.has_centroid ||
      !tactile.centroid_m.allFinite()) {
    return false;
  }
  *centroid_m = tactile.centroid_m;
  return true;
}

}  // namespace

GraspStabilityCost::GraspStabilityCost(GraspStabilityCostConfig config)
    : config_(std::move(config)) {
  if (!IsFiniteAndNonnegative(config_.force_min_n) ||
      !std::isfinite(config_.force_max_n) ||
      config_.force_max_n <= config_.force_min_n) {
    throw std::invalid_argument(
        "GraspStabilityCost: force bounds must be finite and ordered");
  }
  if (!std::isfinite(config_.centroid_x_min) ||
      !std::isfinite(config_.centroid_x_max) ||
      !std::isfinite(config_.centroid_y_min) ||
      !std::isfinite(config_.centroid_y_max) ||
      config_.centroid_x_min >= config_.centroid_x_max ||
      config_.centroid_y_min >= config_.centroid_y_max) {
    throw std::invalid_argument(
        "GraspStabilityCost: centroid bounds must be finite and ordered");
  }
  if (!IsFiniteAndNonnegative(config_.force_under_weight) ||
      !IsFiniteAndNonnegative(config_.force_over_weight) ||
      !IsFiniteAndNonnegative(config_.slip_threshold) ||
      !IsFiniteAndNonnegative(config_.slip_risk_weight) ||
      !IsFiniteAndNonnegative(config_.slip_velocity_weight) ||
      !IsFiniteAndNonnegative(config_.centroid_boundary_weight) ||
      !IsFiniteAndNonnegative(config_.contact_loss_weight) ||
      !IsFiniteAndNonnegative(config_.contact_patch_target_node_count) ||
      !IsFiniteAndNonnegative(config_.contact_patch_weight) ||
      !IsFiniteAndNonnegative(config_.tracking_weight) ||
      !IsFiniteAndNonnegative(config_.tracking_action_scale_weight) ||
      !IsFiniteAndNonnegative(config_.action_smoothness_weight) ||
      !IsFiniteAndNonnegative(config_.joint_limit_weight)) {
    throw std::invalid_argument(
        "GraspStabilityCost: weights, thresholds, and gains must be finite "
        "and nonnegative");
  }
  if (config_.joint_lower_bound.size() != 0 ||
      config_.joint_upper_bound.size() != 0) {
    if (config_.joint_lower_bound.size() != config_.joint_upper_bound.size()) {
      throw std::invalid_argument(
          "GraspStabilityCost: joint limit dimensions must match");
    }
    if (!config_.joint_lower_bound.allFinite() ||
        !config_.joint_upper_bound.allFinite() ||
        (config_.joint_lower_bound.array() > config_.joint_upper_bound.array())
            .any()) {
      throw std::invalid_argument(
          "GraspStabilityCost: joint limits must be finite and ordered");
    }
  }
}

double GraspStabilityCost::Evaluate(
    const RobotRolloutState& state,
    const Eigen::Ref<const Eigen::VectorXd>& action,
    const CostContext& context) const {
  if (context.rollout == nullptr || context.rollout->tactile == nullptr) {
    return 0.0;
  }

  const auto& tactile =
      state.tactile.valid ? state.tactile : *context.rollout->tactile;
  double slip_risk = TactileSlipRisk(tactile, config_.slip_velocity_weight);

  Eigen::Vector2d centroid_m = Eigen::Vector2d::Zero();
  bool centroid_valid = TactileContactCentroidM(tactile, &centroid_m);
  std::size_t contact_support_count = tactile.contact_support_count;

  const double contact_cost =
      ContactLocalCost(TactileNormalForceN(tactile), slip_risk, centroid_m,
                       centroid_valid, contact_support_count);

  return contact_cost + TrackingGuardCost(action, *context.rollout) +
         JointLimitCost(state, action) +
         config_.action_smoothness_weight * action.squaredNorm();
}

double GraspStabilityCost::TrackingGuardCost(
    const Eigen::Ref<const Eigen::VectorXd>& action,
    const RolloutContext& rollout) const {
  if (rollout.measured_state == nullptr ||
      rollout.initial_reference_state == nullptr) {
    return 0.0;
  }

  const auto& measured_q = rollout.measured_state->q;
  const auto& reference_q = rollout.initial_reference_state->q;
  if (measured_q.size() != reference_q.size() ||
      measured_q.size() != action.size()) {
    return 0.0;
  }

  const double tracking_error_norm = (reference_q - measured_q).norm();
  return config_.tracking_weight * Square(tracking_error_norm) +
         config_.tracking_action_scale_weight * tracking_error_norm *
             action.squaredNorm();
}

double GraspStabilityCost::JointLimitCost(
    const RobotRolloutState& state,
    const Eigen::Ref<const Eigen::VectorXd>& action) const {
  if (config_.joint_lower_bound.size() == 0 ||
      config_.joint_upper_bound.size() == 0 ||
      state.q.size() != action.size() ||
      config_.joint_lower_bound.size() != state.q.size()) {
    return 0.0;
  }

  const Eigen::VectorXd q_next = state.q;
  const Eigen::VectorXd lower_violation =
      (config_.joint_lower_bound - q_next)
          .cwiseMax(Eigen::VectorXd::Zero(q_next.size()));
  const Eigen::VectorXd upper_violation =
      (q_next - config_.joint_upper_bound)
          .cwiseMax(Eigen::VectorXd::Zero(q_next.size()));
  return config_.joint_limit_weight *
         (lower_violation.squaredNorm() + upper_violation.squaredNorm());
}

double GraspStabilityCost::ContactLocalCost(
    double normal_force_n, double slip_risk,
    const Eigen::Vector2d& predicted_centroid_m, bool centroid_valid,
    std::size_t contact_support_count) const {
  double cost = 0.0;
  cost += config_.force_under_weight *
          Square(Relu(config_.force_min_n - normal_force_n));
  cost += config_.force_over_weight *
          Square(Relu(normal_force_n - config_.force_max_n));

  cost += config_.slip_risk_weight *
          Square(Relu(slip_risk - config_.slip_threshold));

  if (config_.contact_centroid_enabled) {
    if (centroid_valid) {
      const double x = predicted_centroid_m.x();
      const double y = predicted_centroid_m.y();
      const double boundary_violation =
          Square(Relu(config_.centroid_x_min - x)) +
          Square(Relu(x - config_.centroid_x_max)) +
          Square(Relu(config_.centroid_y_min - y)) +
          Square(Relu(y - config_.centroid_y_max));
      cost += config_.centroid_boundary_weight * boundary_violation;
    } else {
      cost += config_.contact_loss_weight;
    }
  }

  cost += config_.contact_loss_weight *
          Square(Relu(config_.force_min_n - normal_force_n));

  if (config_.contact_patch_enabled) {
    const double contact_patch_deficit =
        Relu(config_.contact_patch_target_node_count -
             static_cast<double>(contact_support_count));
    cost += config_.contact_patch_weight * Square(contact_patch_deficit);
  }
  return cost;
}

}  // namespace mppi_core
