// Copyright 2026
//
// Licensed under the Apache License, Version 2.0.

#include "mppi_core/policies/jenga_grasp.hpp"

#include <stdexcept>
#include <utility>

#include <Eigen/Core>

namespace mppi_core {

JengaGraspConfig MakeDefaultJengaGraspConfig(std::size_t joint_dim) {
  if (joint_dim == 0) {
    throw std::invalid_argument(
        "MakeDefaultJengaGraspConfig: joint_dim is zero");
  }

  JengaGraspConfig config;
  config.mppi.horizon_steps = 20;
  config.mppi.num_rollouts = 128;
  config.mppi.action_dim = joint_dim;
  config.mppi.dt = 0.01;
  config.mppi.temperature = 1.0;
  config.mppi.random_seed = 1;
  config.mppi.action_lower_bound =
      Eigen::VectorXd::Constant(static_cast<Eigen::Index>(joint_dim), -0.015);
  config.mppi.action_upper_bound =
      Eigen::VectorXd::Constant(static_cast<Eigen::Index>(joint_dim), 0.015);
  config.mppi.action_noise_std =
      Eigen::VectorXd::Constant(static_cast<Eigen::Index>(joint_dim), 0.006);

  config.object = MakeJengaBlockObjectPrior();
  config.disturbances = MakeJengaGraspDisturbanceSet();
  config.grasp_stability_cost.closing_direction =
      Eigen::VectorXd::Ones(static_cast<Eigen::Index>(joint_dim));
  return config;
}

void JengaGrasp::Initialize(std::size_t joint_dim, JengaGraspConfig config) {
  if (joint_dim == 0) {
    throw std::invalid_argument("JengaGrasp::Initialize: joint_dim is zero");
  }
  if (config.mppi.action_dim == 0) {
    config.mppi.action_dim = joint_dim;
  }
  if (config.mppi.action_dim != joint_dim) {
    throw std::invalid_argument("JengaGrasp::Initialize: action_dim mismatch");
  }
  if (config.object.shape_type == ObjectShapeType::kUnknown) {
    config.object = MakeJengaBlockObjectPrior();
  }
  if (config.disturbances.empty()) {
    config.disturbances = MakeJengaGraspDisturbanceSet();
  }
  if (config.grasp_stability_cost.closing_direction.size() == 0) {
    config.grasp_stability_cost.closing_direction =
        Eigen::VectorXd::Ones(static_cast<Eigen::Index>(joint_dim));
  }

  object_ = config.object;
  disturbances_ = config.disturbances;
  ContactPredictionConfig prediction_config;
  prediction_config.friction_coefficient =
      config.grasp_stability_cost.friction_coefficient;
  prediction_config.supporting_contact_count =
      config.grasp_stability_cost.supporting_contact_count;
  prediction_config.gravity_tangential_load_weight =
      config.grasp_stability_cost.gravity_tangential_load_weight;
  prediction_config.motion_tangential_load_weight =
      config.grasp_stability_cost.motion_tangential_load_weight;
  prediction_config.slip_tangential_load_weight =
      config.grasp_stability_cost.slip_tangential_load_weight;
  prediction_config.slip_velocity_weight =
      config.grasp_stability_cost.slip_velocity_weight;
  prediction_config.closing_force_gain_n_per_rad =
      config.grasp_stability_cost.closing_force_gain_n_per_rad;
  prediction_config.opening_force_gain_n_per_rad =
      config.grasp_stability_cost.opening_force_gain_n_per_rad;
  prediction_config.force_proxy_max_n =
      config.grasp_stability_cost.force_proxy_max_n;
  prediction_config.contact_patch_force_per_node_n =
      config.grasp_stability_cost.contact_patch_force_per_node_n;
  prediction_config.slip_prediction_decay =
      config.grasp_stability_cost.slip_prediction_decay;
  prediction_config.slip_prediction_margin_gain_per_n =
      config.grasp_stability_cost.slip_prediction_margin_gain_per_n;
  prediction_config.centroid_slip_drift_gain_m_per_n =
      config.grasp_stability_cost.centroid_slip_drift_gain_m_per_n;
  prediction_config.closing_direction =
      config.grasp_stability_cost.closing_direction;
  model_ = std::make_shared<DeltaQReferenceRolloutModel>(
      joint_dim, std::move(prediction_config));

  optimizer_.Initialize(
      std::move(config.mppi), model_,
      std::make_shared<GraspStabilityCost>(config.grasp_stability_cost));
  initialized_ = true;
}

GraspCommand JengaGrasp::Update(const GraspObservation& observation) {
  if (!initialized_) {
    throw std::logic_error("JengaGrasp::Update: policy is not initialized");
  }

  GraspObservation policy_observation = observation;
  if (policy_observation.object == nullptr) {
    policy_observation.object = &object_;
  }
  if (policy_observation.tactile_disturbances.empty()) {
    policy_observation.tactile_disturbances = disturbances_;
  }
  return optimizer_.Update(policy_observation);
}

void JengaGrasp::Reset() {
  optimizer_.ResetNominalActions();
}

}  // namespace mppi_core
