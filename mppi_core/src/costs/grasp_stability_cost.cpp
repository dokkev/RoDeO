// Copyright 2026
//
// Licensed under the Apache License, Version 2.0.

#include "mppi_core/costs/grasp_stability_cost.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

#include "mppi_core/contact/naritouch.hpp"

namespace mppi_core {
namespace {

constexpr double kGravityMps2 = 9.80665;

double Relu(double value) {
  return std::max(0.0, value);
}

double Square(double value) {
  return value * value;
}

bool IsFiniteAndNonnegative(double value) {
  return std::isfinite(value) && value >= 0.0;
}

double Clamp(double value, double lower, double upper) {
  return std::max(lower, std::min(value, upper));
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
  if (!std::isfinite(config_.friction_coefficient) ||
      config_.friction_coefficient <= 0.0) {
    throw std::invalid_argument(
        "GraspStabilityCost: friction_coefficient must be positive");
  }
  if (!std::isfinite(config_.friction_mu_min) ||
      config_.friction_mu_min <= 0.0) {
    throw std::invalid_argument(
        "GraspStabilityCost: friction_mu_min must be positive");
  }
  if (!std::isfinite(config_.supporting_contact_count) ||
      config_.supporting_contact_count <= 0.0) {
    throw std::invalid_argument(
        "GraspStabilityCost: supporting_contact_count must be positive");
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
  if (!IsFiniteAndNonnegative(config_.object_force_safety_factor) ||
      !IsFiniteAndNonnegative(config_.force_under_weight) ||
      !IsFiniteAndNonnegative(config_.force_over_weight) ||
      !IsFiniteAndNonnegative(config_.slip_threshold) ||
      !IsFiniteAndNonnegative(config_.slip_risk_weight) ||
      !IsFiniteAndNonnegative(config_.slip_velocity_weight) ||
      !IsFiniteAndNonnegative(config_.friction_margin_weight) ||
      !IsFiniteAndNonnegative(config_.required_force_weight) ||
      !IsFiniteAndNonnegative(config_.gravity_tangential_load_weight) ||
      !IsFiniteAndNonnegative(config_.motion_tangential_load_weight) ||
      !IsFiniteAndNonnegative(config_.slip_tangential_load_weight) ||
      !IsFiniteAndNonnegative(config_.force_spike_tangential_load_weight) ||
      !IsFiniteAndNonnegative(config_.closing_force_gain_n_per_rad) ||
      !IsFiniteAndNonnegative(config_.opening_force_gain_n_per_rad) ||
      !IsFiniteAndNonnegative(config_.force_proxy_max_n) ||
      !IsFiniteAndNonnegative(config_.contact_patch_force_per_node_n) ||
      !IsFiniteAndNonnegative(config_.slip_prediction_decay) ||
      !IsFiniteAndNonnegative(config_.slip_prediction_margin_gain_per_n) ||
      !IsFiniteAndNonnegative(config_.centroid_slip_drift_gain_m_per_n) ||
      !IsFiniteAndNonnegative(config_.centroid_boundary_weight) ||
      !IsFiniteAndNonnegative(config_.contact_loss_weight) ||
      !IsFiniteAndNonnegative(config_.tracking_weight) ||
      !IsFiniteAndNonnegative(config_.tracking_action_scale_weight) ||
      !IsFiniteAndNonnegative(config_.action_smoothness_weight) ||
      !IsFiniteAndNonnegative(config_.joint_limit_weight)) {
    throw std::invalid_argument(
        "GraspStabilityCost: weights, thresholds, and gains must be finite "
        "and nonnegative");
  }
  if (config_.force_proxy_max_n <= 0.0) {
    throw std::invalid_argument(
        "GraspStabilityCost: force_proxy_max_n must be positive");
  }
  if (config_.contact_patch_force_per_node_n <= 0.0) {
    throw std::invalid_argument(
        "GraspStabilityCost: contact_patch_force_per_node_n must be positive");
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
      state.tactile_initialized ? state.tactile : *context.rollout->tactile;
  double slip_risk =
      ComputeNariTouchSlipRisk(tactile, config_.slip_velocity_weight);

  Eigen::Vector2d centroid_m = Eigen::Vector2d::Zero();
  bool centroid_valid = ComputeNariTouchContactCentroidM(tactile, &centroid_m);

  double normal_force_proxy_n =
      NormalForceProxyN(tactile, state, action, *context.rollout);
  double tangential_load_proxy_n =
      TangentialLoadProxyN(slip_risk, *context.rollout);
  double friction_margin_n =
      config_.friction_coefficient * normal_force_proxy_n -
      tangential_load_proxy_n;

  if (state.contact.initialized) {
    slip_risk = state.contact.slip_risk;
    centroid_m = state.contact.centroid_m;
    centroid_valid =
        state.contact.centroid_valid && state.contact.contact_valid;
    normal_force_proxy_n = state.contact.normal_force_n;
    tangential_load_proxy_n = state.contact.tangential_load_n;
    friction_margin_n = std::min(state.contact.friction_margin_n,
                                 state.contact.friction_pyramid_margin_n);
  }
  const double force_min_n = MinimumForceN(*context.rollout);

  double robust_cost = ScenarioCost(
      normal_force_proxy_n, tangential_load_proxy_n, friction_margin_n,
      slip_risk, centroid_m, centroid_valid, force_min_n);

  const auto* disturbances = context.rollout->tactile_disturbances;
  if (disturbances != nullptr) {
    for (std::size_t i = 0; i < disturbances->count; ++i) {
      const auto& disturbance = disturbances->scenarios[i];
      const double disturbed_force_n =
          Clamp(normal_force_proxy_n + disturbance.normal_force_delta_n, 0.0,
                config_.force_proxy_max_n);
      const double disturbed_slip =
          std::max(0.0, slip_risk + disturbance.slip_delta);
      const double disturbed_tangential_load_n =
          tangential_load_proxy_n +
          config_.slip_tangential_load_weight * disturbance.slip_delta;
      const double safe_disturbed_tangential_load_n =
          std::max(0.0, disturbed_tangential_load_n);
      const double disturbed_friction_margin_n =
          config_.friction_coefficient * disturbed_force_n -
          safe_disturbed_tangential_load_n;
      const Eigen::Vector2d disturbed_centroid_m =
          centroid_m + disturbance.centroid_delta_m;
      robust_cost = std::max(
          robust_cost,
          ScenarioCost(disturbed_force_n, safe_disturbed_tangential_load_n,
                       disturbed_friction_margin_n, disturbed_slip,
                       disturbed_centroid_m, centroid_valid, force_min_n));
    }
  }

  return robust_cost + TrackingGuardCost(action, *context.rollout) +
         JointLimitCost(state, action) +
         config_.action_smoothness_weight * action.squaredNorm();
}

double GraspStabilityCost::ClosingDelta(
    const Eigen::Ref<const Eigen::VectorXd>& action) const {
  if (action.size() == 0) {
    return 0.0;
  }

  if (config_.closing_direction.size() == 0) {
    return action.mean();
  }
  if (config_.closing_direction.size() != action.size()) {
    throw std::invalid_argument(
        "GraspStabilityCost: closing_direction dimension mismatch");
  }

  const double normalizer =
      std::max(1.0, config_.closing_direction.cwiseAbs().sum());
  return config_.closing_direction.dot(action) / normalizer;
}

double GraspStabilityCost::CumulativeClosingDelta(
    const RobotRolloutState& state,
    const Eigen::Ref<const Eigen::VectorXd>& action,
    const RolloutContext& rollout) const {
  if (rollout.initial_reference_state == nullptr) {
    return ClosingDelta(action);
  }
  const auto& initial_q = rollout.initial_reference_state->q;
  if (state.q.size() != action.size() || initial_q.size() != action.size()) {
    return ClosingDelta(action);
  }

  if (config_.closing_direction.size() == 0) {
    return (state.q - initial_q).mean();
  }
  if (config_.closing_direction.size() != action.size()) {
    throw std::invalid_argument(
        "GraspStabilityCost: closing_direction dimension mismatch");
  }

  const double normalizer =
      std::max(1.0, config_.closing_direction.cwiseAbs().sum());
  return config_.closing_direction.dot(state.q - initial_q) / normalizer;
}

double GraspStabilityCost::NormalForceProxyN(
    const NariTouchState& tactile, const RobotRolloutState& state,
    const Eigen::Ref<const Eigen::VectorXd>& action,
    const RolloutContext& rollout) const {
  const double measured_force_n = ComputeNariTouchTotalNormalForceN(tactile);
  const double cumulative_closing_delta =
      CumulativeClosingDelta(state, action, rollout);
  const double predicted_force_n =
      measured_force_n +
      config_.closing_force_gain_n_per_rad * Relu(cumulative_closing_delta) -
      config_.opening_force_gain_n_per_rad * Relu(-cumulative_closing_delta);

  return Clamp(predicted_force_n, 0.0, config_.force_proxy_max_n);
}

double GraspStabilityCost::TangentialLoadProxyN(
    double slip_risk, const RolloutContext& rollout) const {
  double gravity_load_n = 0.0;
  if (rollout.object != nullptr && rollout.object->mass_kg > 0.0) {
    double gravity_magnitude = kGravityMps2;
    if (rollout.has_gravity_context &&
        rollout.gravity_in_sensor_frame.allFinite()) {
      const Eigen::Vector2d tangent_gravity =
          rollout.gravity_in_sensor_frame.head<2>();
      gravity_magnitude = tangent_gravity.norm();
      if (gravity_magnitude <= 0.0) {
        gravity_magnitude = kGravityMps2;
      }
    }
    gravity_load_n = rollout.object->mass_kg * gravity_magnitude /
                     config_.supporting_contact_count;
  }

  return config_.gravity_tangential_load_weight * gravity_load_n +
         config_.slip_tangential_load_weight * std::max(0.0, slip_risk);
}

double GraspStabilityCost::MinimumForceN(const RolloutContext& rollout) const {
  double force_min_n = config_.force_min_n;
  if (config_.use_object_weight_lower_bound && rollout.object != nullptr &&
      rollout.object->mass_kg > 0.0) {
    const double object_force_n =
        config_.object_force_safety_factor * rollout.object->mass_kg *
        kGravityMps2 /
        (config_.friction_coefficient * config_.supporting_contact_count);
    force_min_n = std::max(force_min_n, object_force_n);
  }
  return force_min_n;
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

double GraspStabilityCost::ScenarioCost(
    double normal_force_proxy_n, double tangential_load_proxy_n,
    double friction_margin_n, double slip_risk,
    const Eigen::Vector2d& predicted_centroid_m, bool centroid_valid,
    double force_min_n) const {
  double cost = 0.0;
  cost += config_.force_under_weight *
          Square(Relu(force_min_n - normal_force_proxy_n));
  cost += config_.force_over_weight *
          Square(Relu(normal_force_proxy_n - config_.force_max_n));

  if (config_.friction_margin_enabled) {
    cost += config_.friction_margin_weight * Square(Relu(-friction_margin_n));
  }

  const double required_normal_force_n =
      tangential_load_proxy_n /
      std::max(config_.friction_mu_min, config_.friction_coefficient);
  cost += config_.required_force_weight *
          Square(Relu(required_normal_force_n - config_.force_max_n));

  cost += config_.slip_risk_weight *
          Square(Relu(slip_risk - config_.slip_threshold));

  if (config_.contact_centroid_enabled && centroid_valid) {
    const double x = predicted_centroid_m.x();
    const double y = predicted_centroid_m.y();
    const double boundary_violation = Square(Relu(config_.centroid_x_min - x)) +
                                      Square(Relu(x - config_.centroid_x_max)) +
                                      Square(Relu(config_.centroid_y_min - y)) +
                                      Square(Relu(y - config_.centroid_y_max));
    cost += config_.centroid_boundary_weight * boundary_violation;
  } else {
    cost += config_.contact_loss_weight;
  }

  cost += config_.contact_loss_weight *
          Square(Relu(force_min_n - normal_force_proxy_n));
  return cost;
}

}  // namespace mppi_core
