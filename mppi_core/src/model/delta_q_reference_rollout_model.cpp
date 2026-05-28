// Copyright 2026
//
// Licensed under the Apache License, Version 2.0.

#include "mppi_core/model/delta_q_reference_rollout_model.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <utility>

#include <Eigen/Geometry>
#include <pinocchio/algorithm/rnea.hpp>

#include "mppi_core/grasp/contact_force_correction.hpp"
#include "mppi_core/grasp/contact_force_projection.hpp"
#include "mppi_core/grasp/contact_force_rollout.hpp"
#include "mppi_core/grasp/grasp_contact_kinematics.hpp"

namespace mppi_core {
namespace {

constexpr double kGravityMps2 = 9.80665;
constexpr double kContactForceEpsN = 1.0e-6;

struct TactileOscillationState {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Eigen::Vector2d centroid_velocity_mps = Eigen::Vector2d::Zero();
  Eigen::Vector3d slip_velocity = Eigen::Vector3d::Zero();
  Eigen::Vector2d tangential_disturbance_direction = Eigen::Vector2d::Zero();
  double tangential_disturbance_n{0.0};
};

struct ContactPredictionState {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  bool initialized{false};
  bool contact_valid{false};
  bool centroid_valid{false};
  std::size_t contact_support_count{0};
  double normal_force_n{0.0};
  double tangential_load_n{0.0};
  double friction_margin_n{0.0};
  double friction_pyramid_margin_n{0.0};
  double slip_risk{0.0};
  Eigen::Vector2d centroid_m = Eigen::Vector2d::Zero();
  TactileOscillationState oscillation;
};

double Relu(double value) {
  return std::max(0.0, value);
}

double Clamp(double value, double lower, double upper) {
  return std::max(lower, std::min(value, upper));
}

Eigen::Vector2d ClampVectorNorm(const Eigen::Vector2d& value, double max_norm) {
  if (!value.allFinite()) {
    return Eigen::Vector2d::Zero();
  }
  if (!std::isfinite(max_norm) || max_norm <= 0.0) {
    return value;
  }

  const double norm = value.norm();
  if (norm <= max_norm || norm <= 0.0) {
    return value;
  }
  return value * (max_norm / norm);
}

Eigen::Vector3d ClampVectorNorm(const Eigen::Vector3d& value, double max_norm) {
  if (!value.allFinite()) {
    return Eigen::Vector3d::Zero();
  }
  if (!std::isfinite(max_norm) || max_norm <= 0.0) {
    return value;
  }

  const double norm = value.norm();
  if (norm <= max_norm || norm <= 0.0) {
    return value;
  }
  return value * (max_norm / norm);
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

Eigen::Vector3d TactileShearStateVector(const TactileState& tactile) {
  Eigen::Vector3d shear = Eigen::Vector3d::Zero();
  if (tactile.has_shear && tactile.shear_displacement_m.allFinite()) {
    shear.head<2>() = tactile.shear_displacement_m;
  }
  if (tactile.has_rotational_shear &&
      std::isfinite(tactile.rotational_shear_rad)) {
    shear.z() = tactile.rotational_shear_rad;
  }
  return shear;
}

Eigen::Vector3d TactileShearVelocityVector(const TactileState& tactile) {
  Eigen::Vector3d shear_velocity = Eigen::Vector3d::Zero();
  if (tactile.has_shear_velocity && tactile.shear_velocity_mps.allFinite()) {
    shear_velocity.head<2>() = tactile.shear_velocity_mps;
  }
  if (tactile.has_rotational_shear_velocity &&
      std::isfinite(tactile.rotational_shear_velocity_radps)) {
    shear_velocity.z() = tactile.rotational_shear_velocity_radps;
  }
  return shear_velocity;
}

double TactileSlipScore(const TactileState& tactile) {
  const double provided_score = FiniteNonnegativeOrZero(tactile.slip_score);
  if (provided_score > 0.0) {
    return provided_score;
  }
  return TactileShearStateVector(tactile).norm();
}

double TactileSlipVelocityScore(const TactileState& tactile) {
  const double provided_score =
      FiniteNonnegativeOrZero(tactile.slip_velocity_score);
  if (provided_score > 0.0) {
    return provided_score;
  }
  return TactileShearVelocityVector(tactile).norm();
}

double TactileSlipRisk(const TactileState& tactile, double velocity_weight) {
  const double computed_risk =
      TactileSlipScore(tactile) +
      std::max(0.0, velocity_weight) * TactileSlipVelocityScore(tactile);
  return std::max(computed_risk,
                  FiniteNonnegativeOrZero(tactile.incipient_slip_score));
}

void RefreshTactileSlipScores(TactileState* tactile, double velocity_weight) {
  if (tactile == nullptr) {
    return;
  }
  tactile->slip_score = TactileShearStateVector(*tactile).norm();
  tactile->slip_velocity_score = TactileShearVelocityVector(*tactile).norm();
  tactile->incipient_slip_score =
      tactile->slip_score +
      std::max(0.0, velocity_weight) * tactile->slip_velocity_score;
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

bool ContactIsValid(const TactileState& tactile) {
  return tactile.hasContact() || TactileNormalForceN(tactile) > 0.0;
}

std::size_t ContactSupportCountFromForce(double normal_force_n,
                                         double force_per_support_n,
                                         std::size_t max_support_count) {
  if (normal_force_n <= kContactForceEpsN) {
    return 0;
  }

  const double safe_force_per_support =
      std::max(kContactForceEpsN, force_per_support_n);
  const auto count = static_cast<std::size_t>(
      std::ceil(normal_force_n / safe_force_per_support));
  if (max_support_count > 0) {
    return std::max<std::size_t>(
        1, std::min<std::size_t>(count, max_support_count));
  }
  return std::max<std::size_t>(1, count);
}

ContactPresence ContactPresenceFromSupportCount(std::size_t support_count) {
  if (support_count == 0) {
    return ContactPresence::kNoContact;
  }
  if (support_count < 3) {
    return ContactPresence::kLightContact;
  }
  return ContactPresence::kStableContact;
}

Eigen::Vector2d SlipDirection(const TactileState& tactile,
                              const ContactPredictionState& contact) {
  const Eigen::Vector2d shear_xy = TactileShearStateVector(tactile).head<2>();
  const double shear_xy_norm = shear_xy.norm();
  if (shear_xy_norm > 1.0e-8) {
    return shear_xy / shear_xy_norm;
  }

  const double centroid_norm = contact.centroid_m.norm();
  if (contact.centroid_valid && centroid_norm > 1.0e-8) {
    return contact.centroid_m / centroid_norm;
  }
  return Eigen::Vector2d::Zero();
}

void UpdatePredictedShearState(const ContactPredictionState& contact, double dt,
                               double velocity_weight, TactileState* tactile) {
  if (tactile == nullptr) {
    return;
  }

  const Eigen::Vector3d old_shear = TactileShearStateVector(*tactile);

  Eigen::Vector3d shear_velocity = contact.oscillation.slip_velocity;
  if (!shear_velocity.allFinite()) {
    shear_velocity.setZero();
  }
  const Eigen::Vector3d shear = old_shear + shear_velocity * std::max(0.0, dt);

  tactile->has_shear = true;
  tactile->shear_displacement_m = shear.head<2>();
  tactile->has_shear_velocity = true;
  tactile->shear_velocity_mps = shear_velocity.head<2>();
  tactile->has_rotational_shear = true;
  tactile->rotational_shear_rad = shear.z();
  tactile->has_rotational_shear_velocity = true;
  tactile->rotational_shear_velocity_radps = shear_velocity.z();
  tactile->centroid_velocity_mps = contact.oscillation.centroid_velocity_mps;
  RefreshTactileSlipScores(tactile, velocity_weight);
}

TactileState InitialTactilePrediction(const RolloutContext& context) {
  if (context.tactile == nullptr) {
    return TactileState{};
  }
  return *context.tactile;
}

bool HasActiveTactileContactPoint(const TactileState& tactile) {
  for (const auto& point : tactile.contact_points) {
    if (point.active && point.position_sensor_m.allFinite()) {
      return true;
    }
  }
  return false;
}

bool HasValidMeasuredTau(const RobotRolloutState& state,
                         Eigen::Index tangent_dim) {
  return state.has_measured_tau && state.measured_tau.size() == tangent_dim &&
         state.measured_tau.allFinite();
}

bool BuildForceAwareTorqueSource(
    const RobotRolloutState& state,
    const Eigen::Ref<const Eigen::VectorXd>& action,
    const ContactForceRolloutConfig& config, bool allow_impedance_proxy,
    Eigen::Index tangent_dim, Eigen::VectorXd* tau_source) {
  if (tau_source == nullptr) {
    return false;
  }

  if (HasValidMeasuredTau(state, tangent_dim)) {
    *tau_source = state.measured_tau;
    return true;
  }

  if (!allow_impedance_proxy || !config.enable_impedance_torque_proxy ||
      action.size() != tangent_dim || state.dq.size() != tangent_dim ||
      !action.allFinite() || !state.dq.allFinite()) {
    return false;
  }

  const double stiffness = std::isfinite(config.impedance_stiffness_nm_per_rad)
                               ? config.impedance_stiffness_nm_per_rad
                               : 0.0;
  const double damping = std::isfinite(config.impedance_damping_nms_per_rad)
                             ? config.impedance_damping_nms_per_rad
                             : 0.0;
  *tau_source = stiffness * action - damping * state.dq;
  return tau_source->allFinite();
}

bool TryForceAwareTactileRollout(
    const RobotRolloutState& state,
    const Eigen::Ref<const Eigen::VectorXd>& action,
    const RolloutContext& context, double dt, TactileState* tactile_out) {
  if (tactile_out == nullptr || context.contact_kinematics == nullptr ||
      !state.valid || !state.tactile.valid ||
      !HasActiveTactileContactPoint(state.tactile) ||
      !IsValidContactKinematicsContext(*context.contact_kinematics)) {
    return false;
  }

  const auto& kinematics = *context.contact_kinematics;
  const auto& model = *kinematics.model;
  auto& data = *kinematics.data;
  if (state.q.size() != static_cast<Eigen::Index>(model.nq) ||
      state.dq.size() != static_cast<Eigen::Index>(model.nv) ||
      !state.q.allFinite() || !state.dq.allFinite()) {
    return false;
  }

  const ContactForceProjectionConfig projection_config =
      context.contact_force_projection_config != nullptr
          ? *context.contact_force_projection_config
          : ContactForceProjectionConfig{};
  ContactForceRolloutConfig force_rollout_config =
      context.contact_force_rollout_config != nullptr
          ? *context.contact_force_rollout_config
          : ContactForceRolloutConfig{};
  if (!projection_config.enabled ||
      !force_rollout_config.enable_force_projection_update) {
    return false;
  }

  Eigen::VectorXd tau_source;
  const bool allow_impedance_proxy =
      context.contact_force_rollout_config != nullptr;
  if (!BuildForceAwareTorqueSource(state, action, force_rollout_config,
                                   allow_impedance_proxy, model.nv,
                                   &tau_source)) {
    return false;
  }

  const Eigen::VectorXd zero_acceleration =
      Eigen::VectorXd::Zero(static_cast<Eigen::Index>(model.nv));
  const Eigen::VectorXd tau_model =
      pinocchio::rnea(model, data, state.q, state.dq, zero_acceleration);
  if (tau_model.size() != tau_source.size() || !tau_model.allFinite()) {
    return false;
  }

  ContactForceProjectionResult projection =
      ProjectContactForcesFromTorqueResidual(state, tau_source - tau_model,
                                             kinematics, projection_config);
  if (!projection.valid) {
    return false;
  }

  if (context.contact_force_correction_state != nullptr) {
    projection.total_normal_force_n =
        ApplyContactForceCorrection(projection.total_normal_force_n,
                                    *context.contact_force_correction_state);
  }

  *tactile_out = state.tactile;
  StepTactileStateFromProjectedForce(projection, dt, force_rollout_config,
                                     tactile_out);
  return tactile_out->valid;
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
         config.slip_tangential_load_weight * std::max(0.0, contact.slip_risk) +
         std::max(0.0, contact.oscillation.tangential_disturbance_n) +
         config.slip_velocity_weight *
             (contact.oscillation.slip_velocity.allFinite()
                  ? contact.oscillation.slip_velocity.norm()
                  : 0.0);
}

ContactPredictionState InitialContactPrediction(const TactileState& tactile,
                                                double force_proxy_max_n,
                                                double force_per_support_n,
                                                double slip_velocity_weight,
                                                double friction_coefficient) {
  ContactPredictionState contact;
  contact.initialized = true;
  contact.contact_valid = ContactIsValid(tactile);
  contact.normal_force_n =
      Clamp(TactileNormalForceN(tactile), 0.0, force_proxy_max_n);
  contact.slip_risk = TactileSlipRisk(tactile, slip_velocity_weight);
  contact.oscillation.slip_velocity = TactileShearVelocityVector(tactile);
  contact.oscillation.centroid_velocity_mps =
      tactile.centroid_velocity_mps.allFinite() ? tactile.centroid_velocity_mps
                                                : Eigen::Vector2d::Zero();
  contact.centroid_valid =
      TactileContactCentroidM(tactile, &contact.centroid_m);
  if (contact.contact_valid && !contact.centroid_valid) {
    contact.centroid_m.setZero();
    contact.centroid_valid = true;
  }
  contact.contact_support_count = tactile.contact_support_count;
  if (contact.contact_valid && contact.contact_support_count == 0) {
    contact.contact_support_count = ContactSupportCountFromForce(
        contact.normal_force_n, force_per_support_n, tactile.support_count);
  }

  const Eigen::Vector3d force =
      Eigen::Vector3d{0.0, 0.0, contact.normal_force_n};
  contact.friction_pyramid_margin_n = FrictionPyramidMarginN(
      force, Eigen::Vector3d::UnitZ(), friction_coefficient);
  contact.friction_margin_n = contact.friction_pyramid_margin_n;
  return contact;
}

void UpdatePredictedTactile(const ContactPredictionState& contact,
                            double velocity_weight, TactileState* tactile) {
  if (tactile == nullptr) {
    return;
  }

  tactile->valid = true;
  tactile->has_normal_force = true;
  tactile->normal_force_n = contact.normal_force_n;

  if (!contact.contact_valid || contact.contact_support_count == 0) {
    tactile->contact_presence = ContactPresence::kNoContact;
    tactile->has_centroid = false;
    tactile->centroid_m.setZero();
    tactile->centroid_velocity_mps.setZero();
    tactile->has_shear = false;
    tactile->shear_displacement_m.setZero();
    tactile->has_shear_velocity = false;
    tactile->shear_velocity_mps.setZero();
    tactile->has_rotational_shear = false;
    tactile->rotational_shear_rad = 0.0;
    tactile->has_rotational_shear_velocity = false;
    tactile->rotational_shear_velocity_radps = 0.0;
    tactile->slip_score = 0.0;
    tactile->slip_velocity_score = 0.0;
    tactile->incipient_slip_score = 0.0;
    tactile->contact_support_count = 0;
    tactile->contact_area_proxy = 0.0;
    tactile->edge_risk = 0.0;
    return;
  }

  tactile->contact_presence =
      ContactPresenceFromSupportCount(contact.contact_support_count);
  tactile->has_centroid =
      contact.centroid_valid && contact.centroid_m.allFinite();
  if (tactile->has_centroid) {
    tactile->centroid_m = contact.centroid_m;
  }
  tactile->centroid_velocity_mps = contact.oscillation.centroid_velocity_mps;
  tactile->contact_support_count = contact.contact_support_count;
  if (tactile->support_count == 0) {
    tactile->support_count = contact.contact_support_count;
  }
  if (tactile->support_count > 0) {
    tactile->contact_area_proxy =
        Clamp(static_cast<double>(tactile->contact_support_count) /
                  static_cast<double>(tactile->support_count),
              0.0, 1.0);
  } else {
    tactile->contact_area_proxy = 0.0;
  }
  RefreshTactileSlipScores(tactile, velocity_weight);
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
      !std::isfinite(config.centroid_slip_drift_gain_m_per_n) ||
      !std::isfinite(config.slip_velocity_decay) ||
      !std::isfinite(config.slip_velocity_margin_gain_per_nps) ||
      !std::isfinite(config.action_slip_damping_gain_per_rad) ||
      !std::isfinite(config.max_slip_velocity) ||
      !std::isfinite(config.centroid_velocity_decay) ||
      !std::isfinite(config.centroid_velocity_slip_gain) ||
      !std::isfinite(config.max_centroid_velocity_mps)) {
    throw std::invalid_argument(
        "DeltaQReferenceRolloutModel: prediction weights and gains must be "
        "finite");
  }
  if (config.slip_velocity_weight < 0.0 || config.slip_prediction_decay < 0.0 ||
      config.slip_prediction_margin_gain_per_n < 0.0 ||
      config.centroid_slip_drift_gain_m_per_n < 0.0 ||
      config.slip_velocity_decay < 0.0 ||
      config.slip_velocity_margin_gain_per_nps < 0.0 ||
      config.action_slip_damping_gain_per_rad < 0.0 ||
      config.max_slip_velocity <= 0.0 || config.centroid_velocity_decay < 0.0 ||
      config.centroid_velocity_slip_gain < 0.0 ||
      config.max_centroid_velocity_mps <= 0.0) {
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
  next_state->dq = action / dt;
  next_state->has_measured_tau = false;
  next_state->measured_tau.resize(0);
  next_state->tactile =
      state.tactile.valid ? state.tactile : InitialTactilePrediction(context);
  next_state->valid = next_state->tactile.valid &&
                      next_state->q.size() == next_state->dq.size();

  RobotRolloutState tactile_rollout_state = state;
  tactile_rollout_state.tactile = next_state->tactile;
  tactile_rollout_state.valid =
      tactile_rollout_state.tactile.valid &&
      tactile_rollout_state.q.size() == tactile_rollout_state.dq.size();

  TactileState force_predicted_tactile;
  if (TryForceAwareTactileRollout(tactile_rollout_state, action, context, dt,
                                  &force_predicted_tactile)) {
    next_state->tactile = force_predicted_tactile;
    next_state->valid = next_state->tactile.valid &&
                        next_state->q.size() == next_state->dq.size();
    return;
  }

  if (context.contact_kinematics != nullptr &&
      IsValidContactKinematicsInput(tactile_rollout_state, action,
                                    *context.contact_kinematics)) {
    const GraspRolloutConfig rollout_config =
        context.grasp_rollout_config != nullptr ? *context.grasp_rollout_config
                                                : GraspRolloutConfig{};
    const auto motions = ComputeContactPointMotions(
        tactile_rollout_state, action, *context.contact_kinematics);
    *next_state =
        StepGraspTactilePatch(tactile_rollout_state, next_state->q,
                              next_state->dq, motions, dt, rollout_config);
    return;
  }

  ContactPredictionState contact = InitialContactPrediction(
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
  contact.contact_support_count = ContactSupportCountFromForce(
      contact.normal_force_n, prediction_config_.contact_patch_force_per_node_n,
      next_state->tactile.support_count);
  if (!contact.contact_valid) {
    contact.contact_support_count = 0;
    contact.centroid_valid = false;
    contact.centroid_m.setZero();
    contact.slip_risk = 0.0;
    contact.oscillation = TactileOscillationState{};
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

    Eigen::Vector3d slip_velocity = contact.oscillation.slip_velocity;
    if (!slip_velocity.allFinite()) {
      slip_velocity.setZero();
    }
    slip_velocity *= prediction_config_.slip_velocity_decay;
    if (slip_direction.norm() > 1.0e-8) {
      slip_velocity.x() +=
          prediction_config_.slip_velocity_margin_gain_per_nps *
          margin_deficit_n * slip_direction.x();
      slip_velocity.y() +=
          prediction_config_.slip_velocity_margin_gain_per_nps *
          margin_deficit_n * slip_direction.y();
    }
    const double action_damping =
        Clamp(1.0 - prediction_config_.action_slip_damping_gain_per_rad *
                        Relu(closing_delta),
              0.0, 1.0);
    slip_velocity *= action_damping;
    slip_velocity =
        ClampVectorNorm(slip_velocity, prediction_config_.max_slip_velocity);
    contact.oscillation.slip_velocity = slip_velocity;

    Eigen::Vector2d centroid_velocity =
        contact.oscillation.centroid_velocity_mps;
    if (!centroid_velocity.allFinite()) {
      centroid_velocity.setZero();
    }
    centroid_velocity *= prediction_config_.centroid_velocity_decay;
    centroid_velocity += prediction_config_.centroid_velocity_slip_gain *
                         slip_velocity.head<2>();
    centroid_velocity = ClampVectorNorm(
        centroid_velocity, prediction_config_.max_centroid_velocity_mps);
    contact.oscillation.centroid_velocity_mps = centroid_velocity;

    Eigen::Vector2d disturbance_direction = slip_direction;
    const double velocity_xy_norm = slip_velocity.head<2>().norm();
    if (velocity_xy_norm > 1.0e-8) {
      disturbance_direction = slip_velocity.head<2>() / velocity_xy_norm;
    }
    if (disturbance_direction.norm() > 1.0e-8) {
      contact.oscillation.tangential_disturbance_direction =
          disturbance_direction;
      contact.centroid_m +=
          prediction_config_.centroid_slip_drift_gain_m_per_n *
              margin_deficit_n * disturbance_direction +
          centroid_velocity * dt;
    } else {
      contact.oscillation.tangential_disturbance_direction.setZero();
    }
    contact.oscillation.tangential_disturbance_n =
        prediction_config_.slip_tangential_load_weight *
        contact.oscillation.slip_velocity.norm();
    UpdatePredictedShearState(contact, dt,
                              prediction_config_.slip_velocity_weight,
                              &next_state->tactile);
    contact.slip_risk =
        std::max(contact.slip_risk,
                 TactileSlipRisk(next_state->tactile,
                                 prediction_config_.slip_velocity_weight));
  }
  UpdatePredictedTactile(contact, prediction_config_.slip_velocity_weight,
                         &next_state->tactile);
  next_state->valid = next_state->tactile.valid &&
                      next_state->q.size() == next_state->dq.size();
}

}  // namespace mppi_core
