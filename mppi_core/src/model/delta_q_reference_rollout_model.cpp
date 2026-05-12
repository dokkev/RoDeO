// Copyright 2026
//
// Licensed under the Apache License, Version 2.0.

#include "mppi_core/model/delta_q_reference_rollout_model.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <utility>

#include <Eigen/Geometry>

#include "mppi_core/contact/naritouch.hpp"

namespace mppi_core {
namespace {

constexpr double kGravityMps2 = 9.80665;
constexpr double kContactForceEpsN = 1.0e-6;

double Relu(double value) {
  return std::max(0.0, value);
}

double Clamp(double value, double lower, double upper) {
  return std::max(lower, std::min(value, upper));
}

Eigen::Vector3d BuildTangent(const Eigen::Vector3d& normal) {
  Eigen::Vector3d tangent = normal.cross(Eigen::Vector3d::UnitX());
  if (tangent.norm() < 1.0e-5) {
    tangent = normal.cross(Eigen::Vector3d::UnitY());
  }
  return tangent.normalized();
}

double FrictionPyramidMarginN(const Eigen::Vector3d& force,
                              const Eigen::Vector3d& contact_normal,
                              double friction_coefficient) {
  const Eigen::Vector3d normal = contact_normal.normalized();
  const Eigen::Vector3d t1 = BuildTangent(normal);
  const Eigen::Vector3d t2 = normal.cross(t1).normalized();

  // Same linearized pyramid as wbc_core::ContactPoint:
  // (-t - mu*n)^T f <= 0 and (t - mu*n)^T f <= 0.
  const double normal_force = normal.dot(force);
  const double margin_t1 =
      friction_coefficient * normal_force - std::abs(t1.dot(force));
  const double margin_t2 =
      friction_coefficient * normal_force - std::abs(t2.dot(force));
  return std::min(margin_t1, margin_t2);
}

bool ContactIsValid(const NariTouchState& tactile) {
  return tactile.hasContact() ||
         ComputeNariTouchTotalNormalForceN(tactile) > 0.0;
}

std::size_t ContactNodeCountFromForce(double normal_force_n,
                                      double force_per_node_n) {
  if (normal_force_n <= kContactForceEpsN) {
    return 0;
  }

  const double safe_force_per_node =
      std::max(kContactForceEpsN, force_per_node_n);
  const auto count =
      static_cast<std::size_t>(std::ceil(normal_force_n / safe_force_per_node));
  return std::max<std::size_t>(
      1, std::min<std::size_t>(count, kNariTouchNodeCount));
}

NariTouchContactState ContactStateFromNodeCount(std::size_t node_count) {
  if (node_count == 0) {
    return NariTouchContactState::kNoContact;
  }
  if (node_count < 3) {
    return NariTouchContactState::kFewContacts;
  }
  return NariTouchContactState::kEnoughContacts;
}

Eigen::Vector2d SlipDirection(const NariTouchState& tactile,
                              const ContactPredictionState& contact) {
  if (tactile.slip_state.allFinite()) {
    const Eigen::Vector2d slip_xy = tactile.slip_state.head<2>();
    const double slip_xy_norm = slip_xy.norm();
    if (slip_xy_norm > 1.0e-8) {
      return slip_xy / slip_xy_norm;
    }
  }

  const double centroid_norm = contact.centroid_m.norm();
  if (contact.centroid_valid && centroid_norm > 1.0e-8) {
    return contact.centroid_m / centroid_norm;
  }
  return Eigen::Vector2d::Zero();
}

void UpdatePredictedSlipState(const ContactPredictionState& contact,
                              const Eigen::Vector2d& slip_direction, double dt,
                              NariTouchState* tactile) {
  if (tactile == nullptr) {
    return;
  }

  Eigen::Vector3d old_slip = tactile->slip_state;
  if (!old_slip.allFinite()) {
    old_slip.setZero();
  }
  if (old_slip.allFinite() && old_slip.norm() > 1.0e-8) {
    tactile->slip_state = old_slip.normalized() * contact.slip_risk;
    if (dt > 0.0) {
      tactile->slip_velocity_state = (tactile->slip_state - old_slip) / dt;
    }
    return;
  }

  tactile->slip_state.setZero();
  if (slip_direction.norm() > 1.0e-8) {
    tactile->slip_state.x() = slip_direction.x() * contact.slip_risk;
    tactile->slip_state.y() = slip_direction.y() * contact.slip_risk;
  }
  if (dt > 0.0) {
    tactile->slip_velocity_state = (tactile->slip_state - old_slip) / dt;
  }
}

NariTouchState InitialTactilePrediction(const RolloutContext& context) {
  if (context.tactile == nullptr) {
    return NariTouchState{};
  }
  return *context.tactile;
}

double ClosingDelta(const Eigen::Ref<const Eigen::VectorXd>& action,
                    const Eigen::VectorXd& closing_direction) {
  if (action.size() == 0) {
    return 0.0;
  }
  if (closing_direction.size() == 0) {
    return action.mean();
  }
  if (closing_direction.size() != action.size()) {
    throw std::invalid_argument(
        "DeltaQReferenceRolloutModel: closing_direction dimension mismatch");
  }

  const double normalizer = std::max(1.0, closing_direction.cwiseAbs().sum());
  return closing_direction.dot(action) / normalizer;
}

double EstimateTangentialLoadN(const ContactPredictionState& contact,
                               const Eigen::Ref<const Eigen::VectorXd>& action,
                               double dt, const RolloutContext& rollout,
                               const ContactPredictionConfig& config) {
  double gravity_load_n = 0.0;
  if (rollout.object != nullptr && rollout.object->mass_kg > 0.0) {
    double gravity_magnitude = kGravityMps2;
    if (rollout.has_gravity_context &&
        rollout.gravity_in_sensor_frame.allFinite()) {
      gravity_magnitude = rollout.gravity_in_sensor_frame.head<2>().norm();
      if (gravity_magnitude <= 0.0) {
        gravity_magnitude = kGravityMps2;
      }
    }
    gravity_load_n = rollout.object->mass_kg * gravity_magnitude /
                     config.supporting_contact_count;
  }

  double motion_load_n = 0.0;
  if (dt > 0.0 && action.size() > 0) {
    motion_load_n = action.norm() / dt;
  }

  return config.gravity_tangential_load_weight * gravity_load_n +
         config.motion_tangential_load_weight * motion_load_n +
         config.slip_tangential_load_weight * std::max(0.0, contact.slip_risk);
}

ContactPredictionState InitialContactPrediction(const NariTouchState& tactile,
                                                double force_proxy_max_n,
                                                double force_per_node_n,
                                                double slip_velocity_weight,
                                                double friction_coefficient) {
  ContactPredictionState contact;
  contact.initialized = true;
  contact.contact_valid = ContactIsValid(tactile);
  contact.normal_force_n =
      Clamp(ComputeNariTouchTotalNormalForceN(tactile), 0.0, force_proxy_max_n);
  contact.slip_risk = ComputeNariTouchSlipRisk(tactile, slip_velocity_weight);
  contact.centroid_valid =
      ComputeNariTouchContactCentroidM(tactile, &contact.centroid_m);
  if (contact.contact_valid && !contact.centroid_valid) {
    contact.centroid_m.setZero();
    contact.centroid_valid = true;
  }
  contact.contact_node_count = tactile.contactNodeCount();
  if (contact.contact_valid && contact.contact_node_count == 0) {
    contact.contact_node_count =
        ContactNodeCountFromForce(contact.normal_force_n, force_per_node_n);
  }

  const Eigen::Vector3d force =
      Eigen::Vector3d{0.0, 0.0, contact.normal_force_n};
  contact.friction_pyramid_margin_n = FrictionPyramidMarginN(
      force, Eigen::Vector3d::UnitZ(), friction_coefficient);
  contact.friction_margin_n = contact.friction_pyramid_margin_n;
  return contact;
}

void UpdatePredictedTactile(const ContactPredictionState& contact,
                            NariTouchState* tactile) {
  if (tactile == nullptr) {
    return;
  }

  tactile->force_z = contact.normal_force_n;
  if (contact.contact_valid) {
    tactile->contact_state =
        ContactStateFromNodeCount(contact.contact_node_count);
  } else {
    tactile->contact_state = NariTouchContactState::kNoContact;
    tactile->slip_state.setZero();
    tactile->slip_velocity_state.setZero();
    for (auto& node : tactile->nodes) {
      node.contact = false;
      node.normal_force = 0.0;
    }
    return;
  }

  const std::size_t target_count =
      std::min<std::size_t>(contact.contact_node_count, kNariTouchNodeCount);
  if (target_count == 0) {
    tactile->contact_state = NariTouchContactState::kNoContact;
    tactile->slip_state.setZero();
    tactile->slip_velocity_state.setZero();
    for (auto& node : tactile->nodes) {
      node.contact = false;
      node.normal_force = 0.0;
    }
    return;
  }

  std::array<std::pair<double, std::size_t>, kNariTouchNodeCount> distances{};
  for (std::size_t i = 0; i < tactile->nodes.size(); ++i) {
    distances[i] = {
        (tactile->nodes[i].position_m - contact.centroid_m).squaredNorm(), i};
  }
  std::sort(distances.begin(), distances.end());

  for (auto& node : tactile->nodes) {
    node.contact = false;
    node.normal_force = 0.0;
  }

  const double node_force =
      contact.normal_force_n /
      static_cast<double>(std::max<std::size_t>(std::size_t{1}, target_count));
  for (std::size_t i = 0; i < target_count; ++i) {
    auto& node = tactile->nodes[distances[i].second];
    node.contact = true;
    node.normal_force = node_force;
  }
}

void ValidatePredictionConfig(const ContactPredictionConfig& config,
                              std::size_t joint_dim) {
  if (!std::isfinite(config.friction_coefficient) ||
      config.friction_coefficient <= 0.0) {
    throw std::invalid_argument(
        "DeltaQReferenceRolloutModel: friction_coefficient must be positive");
  }
  if (!std::isfinite(config.supporting_contact_count) ||
      config.supporting_contact_count <= 0.0) {
    throw std::invalid_argument(
        "DeltaQReferenceRolloutModel: supporting_contact_count must be "
        "positive");
  }
  if (!std::isfinite(config.force_proxy_max_n) ||
      config.force_proxy_max_n <= 0.0) {
    throw std::invalid_argument(
        "DeltaQReferenceRolloutModel: force_proxy_max_n must be positive");
  }
  if (!std::isfinite(config.contact_patch_force_per_node_n) ||
      config.contact_patch_force_per_node_n <= 0.0) {
    throw std::invalid_argument(
        "DeltaQReferenceRolloutModel: contact_patch_force_per_node_n must be "
        "positive");
  }
  if (!std::isfinite(config.gravity_tangential_load_weight) ||
      !std::isfinite(config.motion_tangential_load_weight) ||
      !std::isfinite(config.slip_tangential_load_weight) ||
      !std::isfinite(config.slip_velocity_weight) ||
      !std::isfinite(config.closing_force_gain_n_per_rad) ||
      !std::isfinite(config.opening_force_gain_n_per_rad) ||
      !std::isfinite(config.slip_prediction_decay) ||
      !std::isfinite(config.slip_prediction_margin_gain_per_n) ||
      !std::isfinite(config.centroid_slip_drift_gain_m_per_n)) {
    throw std::invalid_argument(
        "DeltaQReferenceRolloutModel: prediction weights and gains must be "
        "finite");
  }
  if (config.slip_velocity_weight < 0.0 || config.slip_prediction_decay < 0.0 ||
      config.slip_prediction_margin_gain_per_n < 0.0 ||
      config.centroid_slip_drift_gain_m_per_n < 0.0) {
    throw std::invalid_argument(
        "DeltaQReferenceRolloutModel: slip prediction gains must be "
        "nonnegative");
  }
  if (config.closing_direction.size() != 0 &&
      config.closing_direction.size() != static_cast<Eigen::Index>(joint_dim)) {
    throw std::invalid_argument(
        "DeltaQReferenceRolloutModel: closing_direction dimension mismatch");
  }
  if (config.closing_direction.size() != 0 &&
      !config.closing_direction.allFinite()) {
    throw std::invalid_argument(
        "DeltaQReferenceRolloutModel: closing_direction must be finite");
  }
}

}  // namespace

DeltaQReferenceRolloutModel::DeltaQReferenceRolloutModel(std::size_t joint_dim)
    : DeltaQReferenceRolloutModel(joint_dim, ContactPredictionConfig{}) {}

DeltaQReferenceRolloutModel::DeltaQReferenceRolloutModel(
    std::size_t joint_dim, ContactPredictionConfig prediction_config)
    : joint_dim_(joint_dim), prediction_config_(std::move(prediction_config)) {
  if (joint_dim_ == 0) {
    throw std::invalid_argument(
        "DeltaQReferenceRolloutModel: joint_dim must be nonzero");
  }
  ValidatePredictionConfig(prediction_config_, joint_dim_);
}

void DeltaQReferenceRolloutModel::Step(
    const RobotRolloutState& state,
    const Eigen::Ref<const Eigen::VectorXd>& action,
    const RolloutContext& context, double dt,
    RobotRolloutState* next_state) const {
  if (next_state == nullptr) {
    throw std::invalid_argument(
        "DeltaQReferenceRolloutModel::Step: next_state is null");
  }
  if (dt <= 0.0) {
    throw std::invalid_argument(
        "DeltaQReferenceRolloutModel::Step: dt must be positive");
  }
  if (state.q.size() != static_cast<Eigen::Index>(joint_dim_) ||
      action.size() != static_cast<Eigen::Index>(joint_dim_)) {
    throw std::invalid_argument(
        "DeltaQReferenceRolloutModel::Step: dimension mismatch");
  }

  next_state->q = state.q + action;
  next_state->v = action / dt;
  next_state->tactile = state.tactile_initialized
                            ? state.tactile
                            : InitialTactilePrediction(context);
  next_state->tactile_initialized = true;

  ContactPredictionState contact =
      state.contact.initialized
          ? state.contact
          : InitialContactPrediction(
                next_state->tactile, prediction_config_.force_proxy_max_n,
                prediction_config_.contact_patch_force_per_node_n,
                prediction_config_.slip_velocity_weight,
                prediction_config_.friction_coefficient);
  contact.initialized = true;

  const double closing_delta =
      ClosingDelta(action, prediction_config_.closing_direction);
  contact.normal_force_n =
      Clamp(contact.normal_force_n +
                prediction_config_.closing_force_gain_n_per_rad *
                    Relu(closing_delta) -
                prediction_config_.opening_force_gain_n_per_rad *
                    Relu(-closing_delta),
            0.0, prediction_config_.force_proxy_max_n);
  contact.tangential_load_n =
      EstimateTangentialLoadN(contact, action, dt, context, prediction_config_);
  contact.friction_margin_n =
      prediction_config_.friction_coefficient * contact.normal_force_n -
      contact.tangential_load_n;

  const Eigen::Vector3d predicted_force =
      Eigen::Vector3d{contact.tangential_load_n, 0.0, contact.normal_force_n};
  contact.friction_pyramid_margin_n =
      FrictionPyramidMarginN(predicted_force, Eigen::Vector3d::UnitZ(),
                             prediction_config_.friction_coefficient);
  contact.contact_valid = contact.normal_force_n > kContactForceEpsN;
  contact.contact_node_count = ContactNodeCountFromForce(
      contact.normal_force_n,
      prediction_config_.contact_patch_force_per_node_n);
  if (!contact.contact_valid) {
    contact.contact_node_count = 0;
    contact.centroid_valid = false;
    contact.centroid_m.setZero();
    contact.slip_risk = 0.0;
  } else {
    if (!contact.centroid_valid) {
      contact.centroid_m.setZero();
      contact.centroid_valid = true;
    }

    const double margin_deficit_n = Relu(-std::min(
        contact.friction_margin_n, contact.friction_pyramid_margin_n));
    const Eigen::Vector2d slip_direction =
        SlipDirection(next_state->tactile, contact);
    contact.slip_risk =
        prediction_config_.slip_prediction_decay * contact.slip_risk +
        prediction_config_.slip_prediction_margin_gain_per_n * margin_deficit_n;
    if (slip_direction.norm() > 1.0e-8) {
      contact.centroid_m +=
          prediction_config_.centroid_slip_drift_gain_m_per_n *
          margin_deficit_n * slip_direction;
    }
    UpdatePredictedSlipState(contact, slip_direction, dt, &next_state->tactile);
  }
  UpdatePredictedTactile(contact, &next_state->tactile);
  next_state->contact = contact;
}

}  // namespace mppi_core
