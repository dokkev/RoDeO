// Copyright 2026
//
// Licensed under the Apache License, Version 2.0.

#include "mppi_core/model/delta_q_reference_rollout_model.hpp"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <utility>

#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/rnea.hpp>

#include "mppi_core/grasp/contact_force_correction.hpp"
#include "mppi_core/grasp/contact_force_projection.hpp"
#include "mppi_core/grasp/contact_force_rollout.hpp"
#include "mppi_core/grasp/grasp_contact_kinematics.hpp"

namespace mppi_core {
namespace {

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

bool ShouldTryKinematicPatchFallback(TactileRolloutPolicy policy) {
  return policy == TactileRolloutPolicy::kForceThenKinematicFallback;
}

Eigen::VectorXd ComputeRolloutPredictedTorque(
    const Eigen::Ref<const Eigen::VectorXd>& action,
    const Eigen::Ref<const Eigen::VectorXd>& dq,
    const ContactForceRolloutConfig& config) {
  const double stiffness =
      std::isfinite(config.rollout_torque_stiffness_nm_per_rad)
          ? config.rollout_torque_stiffness_nm_per_rad
          : 0.0;
  const double damping =
      std::isfinite(config.rollout_torque_damping_nms_per_rad)
          ? config.rollout_torque_damping_nms_per_rad
          : 0.0;
  return stiffness * action - damping * dq;
}

bool CanIntegrateWithPinocchio(
    const RobotRolloutState& state,
    const Eigen::Ref<const Eigen::VectorXd>& action,
    const PinocchioContactKinematicsContext* context) {
  return context != nullptr && IsValidContactKinematicsContext(*context) &&
         state.q.size() == static_cast<Eigen::Index>(context->model->nq) &&
         action.size() == static_cast<Eigen::Index>(context->model->nv);
}

Eigen::VectorXd IntegrateReference(
    const RobotRolloutState& state,
    const Eigen::Ref<const Eigen::VectorXd>& action,
    const PinocchioContactKinematicsContext* context) {
  if (CanIntegrateWithPinocchio(state, action, context)) {
    return pinocchio::integrate(*context->model, state.q, action);
  }
  if (state.q.size() != action.size()) {
    throw std::invalid_argument(
        "DeltaQReferenceRolloutModel::Step: q and action dimension mismatch");
  }
  return state.q + action;
}

bool TryForceAwareTactileRollout(
    const RobotRolloutState& state, const RolloutContext& context, double dt,
    TactileState* tactile_out) {
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
      state.tau.size() != static_cast<Eigen::Index>(model.nv) ||
      !state.q.allFinite() || !state.dq.allFinite() ||
      !state.tau.allFinite()) {
    return false;
  }

  const ContactForceProjectionConfig projection_config =
      context.contact_force_projection_config != nullptr
          ? *context.contact_force_projection_config
          : ContactForceProjectionConfig{};
  const ContactForceRolloutConfig force_rollout_config =
      context.contact_force_rollout_config != nullptr
          ? *context.contact_force_rollout_config
          : ContactForceRolloutConfig{};
  if (!projection_config.enabled ||
      !force_rollout_config.enable_force_projection_update) {
    return false;
  }

  const Eigen::VectorXd zero_acceleration =
      Eigen::VectorXd::Zero(static_cast<Eigen::Index>(model.nv));
  const Eigen::VectorXd tau_model =
      pinocchio::rnea(model, data, state.q, state.dq, zero_acceleration);
  if (tau_model.size() != state.tau.size() || !tau_model.allFinite()) {
    return false;
  }

  const Eigen::VectorXd tau_residual = state.tau - tau_model;
  ContactForceProjectionResult projection =
      ProjectContactForcesFromTorqueResidual(state, tau_residual, kinematics,
                                             projection_config);
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

bool HasValidStateVectors(const RobotRolloutState& state) {
  return state.tactile.valid && state.dq.size() == state.tau.size() &&
         state.q.allFinite() && state.dq.allFinite() && state.tau.allFinite();
}

}  // namespace

DeltaQReferenceRolloutModel::DeltaQReferenceRolloutModel(std::size_t joint_dim)
    : DeltaQReferenceRolloutModel(joint_dim,
                                  DeltaQReferenceRolloutConfig{}) {}

DeltaQReferenceRolloutModel::DeltaQReferenceRolloutModel(
    std::size_t joint_dim, DeltaQReferenceRolloutConfig config)
    : joint_dim_(joint_dim), config_(std::move(config)) {
  if (joint_dim_ == 0) {
    throw std::invalid_argument(
        "DeltaQReferenceRolloutModel: joint_dim must be nonzero");
  }
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
  if (!std::isfinite(dt) || dt <= 0.0) {
    throw std::invalid_argument(
        "DeltaQReferenceRolloutModel::Step: dt must be positive");
  }
  if (state.dq.size() != static_cast<Eigen::Index>(joint_dim_) ||
      state.tau.size() != state.dq.size() ||
      action.size() != static_cast<Eigen::Index>(joint_dim_)) {
    throw std::invalid_argument(
        "DeltaQReferenceRolloutModel::Step: dimension mismatch");
  }
  if (state.q.size() != action.size() &&
      !CanIntegrateWithPinocchio(state, action, context.contact_kinematics)) {
    throw std::invalid_argument(
        "DeltaQReferenceRolloutModel::Step: q and action dimension mismatch");
  }
  if (!state.q.allFinite() || !state.dq.allFinite() ||
      !state.tau.allFinite() || !action.allFinite()) {
    throw std::invalid_argument(
        "DeltaQReferenceRolloutModel::Step: state and action must be finite");
  }

  const ContactForceRolloutConfig torque_config =
      context.contact_force_rollout_config != nullptr
          ? *context.contact_force_rollout_config
          : ContactForceRolloutConfig{};

  next_state->q = IntegrateReference(state, action, context.contact_kinematics);
  next_state->dq = action / dt;
  // The input state's tau is measured joint torque at the current observation.
  // Future rollout states use this impedance-style predicted/commanded torque
  // proxy, so torque-residual force projection is measured-torque anchored for
  // the first step and then evolves as a force hypothesis.
  next_state->tau =
      ComputeRolloutPredictedTorque(action, state.dq, torque_config);
  next_state->tactile =
      state.tactile.valid ? state.tactile : InitialTactilePrediction(context);
  next_state->valid = HasValidStateVectors(*next_state);

  RobotRolloutState tactile_rollout_state = state;
  tactile_rollout_state.tactile = next_state->tactile;
  tactile_rollout_state.valid = HasValidStateVectors(tactile_rollout_state);

  TactileState force_predicted_tactile;
  if (TryForceAwareTactileRollout(tactile_rollout_state, context, dt,
                                  &force_predicted_tactile)) {
    next_state->tactile = force_predicted_tactile;
    next_state->valid = HasValidStateVectors(*next_state);
    return;
  }

  const TactileRolloutPolicy rollout_policy = config_.tactile_rollout_policy;
  if (rollout_policy == TactileRolloutPolicy::kForceAwareRequired) {
    next_state->valid = false;
    return;
  }

  if (ShouldTryKinematicPatchFallback(rollout_policy) &&
      context.contact_kinematics != nullptr &&
      IsValidContactKinematicsInput(tactile_rollout_state, action,
                                    *context.contact_kinematics)) {
    const GraspRolloutConfig rollout_config =
        context.grasp_rollout_config != nullptr ? *context.grasp_rollout_config
                                                : GraspRolloutConfig{};
    const auto motions = ComputeContactPointMotions(
        tactile_rollout_state, action, *context.contact_kinematics);
    *next_state =
        StepGraspTactilePatch(tactile_rollout_state, next_state->q,
                              next_state->dq, next_state->tau, motions, dt,
                              rollout_config);
    return;
  }

  next_state->valid = false;
}

}  // namespace mppi_core
