// Copyright 2026
//
// Licensed under the Apache License, Version 2.0.

#include <gtest/gtest.h>

#include <cmath>

#include <Eigen/Core>
#include <yaml-cpp/yaml.h>

#include "mppi_core/config/grasp_config.hpp"
#include "mppi_core/config/mppi_config.hpp"
#include "mppi_core/contact/naritouch.hpp"
#include "mppi_core/grasp_types.hpp"
#include "mppi_core/model/delta_q_reference_rollout_model.hpp"

namespace {

constexpr double kTolerance = 1.0e-9;

mppi_core::RobotRolloutState MakeState(
    std::size_t dim, const mppi_core::NariTouchState& tactile) {
  mppi_core::RobotRolloutState state;
  state.q = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(dim));
  state.v = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(dim));
  state.tactile = tactile;
  state.tactile_initialized = true;
  return state;
}

}  // namespace

TEST(NariTouchFeatureTest, SlipRiskIncludesDerivedVelocityWhenWeighted) {
  mppi_core::NariTouchState tactile;
  tactile.slip_state = Eigen::Vector3d{0.03, 0.04, 0.0};
  tactile.slip_velocity_state = Eigen::Vector3d{0.0, 3.0, 4.0};

  EXPECT_NEAR(mppi_core::ComputeNariTouchSlipMagnitude(tactile), 0.05,
              kTolerance);
  EXPECT_NEAR(mppi_core::ComputeNariTouchSlipVelocityMagnitude(tactile), 5.0,
              kTolerance);
  EXPECT_NEAR(mppi_core::ComputeNariTouchSlipRisk(tactile, 0.2), 1.05,
              kTolerance);

  // Negative weights are treated as disabled rather than inverting slip risk.
  EXPECT_NEAR(mppi_core::ComputeNariTouchSlipRisk(tactile, -1.0), 0.05,
              kTolerance);
}

TEST(DeltaQReferenceRolloutModelTest,
     PredictsContactPatchNodeCountFromNormalForce) {
  mppi_core::ContactPredictionConfig config;
  config.friction_coefficient = 0.5;
  config.supporting_contact_count = 1.0;
  config.gravity_tangential_load_weight = 0.0;
  config.motion_tangential_load_weight = 0.0;
  config.slip_tangential_load_weight = 0.0;
  config.closing_force_gain_n_per_rad = 0.0;
  config.opening_force_gain_n_per_rad = 0.0;
  config.force_proxy_max_n = 5.0;
  config.contact_patch_force_per_node_n = 0.25;

  mppi_core::NariTouchState tactile;
  tactile.force_z = 0.9;
  tactile.contact_state = mppi_core::NariTouchContactState::kEnoughContacts;

  mppi_core::DeltaQReferenceRolloutModel model(2, config);
  const auto state = MakeState(2, tactile);
  const Eigen::VectorXd action = Eigen::VectorXd::Zero(2);
  mppi_core::RolloutContext context;
  mppi_core::RobotRolloutState next_state;

  model.Step(state, action, context, 0.01, &next_state);

  EXPECT_TRUE(next_state.contact.initialized);
  EXPECT_TRUE(next_state.contact.contact_valid);
  EXPECT_EQ(next_state.contact.contact_node_count, 4U);
  EXPECT_EQ(next_state.tactile.contactNodeCount(), 4U);
  EXPECT_EQ(next_state.tactile.contact_state,
            mppi_core::NariTouchContactState::kEnoughContacts);
  EXPECT_NEAR(mppi_core::ComputeNariTouchTotalNormalForceN(next_state.tactile),
              0.9, kTolerance);
}

TEST(DeltaQReferenceRolloutModelTest,
     FrictionDeficitIncreasesSlipAndDerivesVelocity) {
  mppi_core::ContactPredictionConfig config;
  config.friction_coefficient = 0.1;
  config.supporting_contact_count = 1.0;
  config.gravity_tangential_load_weight = 1.0;
  config.motion_tangential_load_weight = 0.0;
  config.slip_tangential_load_weight = 0.0;
  config.closing_force_gain_n_per_rad = 0.0;
  config.opening_force_gain_n_per_rad = 0.0;
  config.force_proxy_max_n = 5.0;
  config.contact_patch_force_per_node_n = 0.5;
  config.slip_prediction_decay = 1.0;
  config.slip_prediction_margin_gain_per_n = 0.5;

  mppi_core::NariTouchState tactile;
  tactile.force_z = 0.2;
  tactile.contact_state = mppi_core::NariTouchContactState::kFewContacts;
  tactile.slip_state = Eigen::Vector3d{0.01, 0.0, 0.0};

  const auto object = mppi_core::MakeJengaBlockObjectPrior();
  mppi_core::RolloutContext context;
  context.object = &object;

  mppi_core::DeltaQReferenceRolloutModel model(2, config);
  const auto state = MakeState(2, tactile);
  const Eigen::VectorXd action = Eigen::VectorXd::Zero(2);
  mppi_core::RobotRolloutState next_state;

  model.Step(state, action, context, 0.1, &next_state);

  EXPECT_TRUE(next_state.contact.initialized);
  EXPECT_GT(next_state.contact.slip_risk,
            mppi_core::ComputeNariTouchSlipMagnitude(tactile));
  EXPECT_GT(next_state.tactile.slip_state.x(), tactile.slip_state.x());
  EXPECT_GT(next_state.tactile.slip_velocity_state.x(), 0.0);
  EXPECT_GT(
      mppi_core::ComputeNariTouchSlipVelocityMagnitude(next_state.tactile),
      0.0);
}

TEST(GraspConfigTest, ParsesTactilePredictionAndSlipVelocityWeight) {
  const YAML::Node root = YAML::Load(R"(
grasp:
  tactile_prediction:
    force_per_node_n: 0.7
    slip_decay: 0.8
    slip_margin_gain_per_n: 0.3
    centroid_drift_gain_m_per_n: 0.0009
  slip_risk:
    velocity_weight: 0.05
  action:
    closing_direction: [1.0, -1.0]
)");

  const auto config = mppi_core::ParseGraspConfig(
      root["grasp"], 2, mppi_core::GraspStabilityCostConfig{});

  EXPECT_NEAR(config.contact_patch_force_per_node_n, 0.7, kTolerance);
  EXPECT_NEAR(config.slip_prediction_decay, 0.8, kTolerance);
  EXPECT_NEAR(config.slip_prediction_margin_gain_per_n, 0.3, kTolerance);
  EXPECT_NEAR(config.centroid_slip_drift_gain_m_per_n, 0.0009, kTolerance);
  EXPECT_NEAR(config.slip_velocity_weight, 0.05, kTolerance);
  ASSERT_EQ(config.closing_direction.size(), 2);
  EXPECT_NEAR(config.closing_direction[0], 1.0, kTolerance);
  EXPECT_NEAR(config.closing_direction[1], -1.0, kTolerance);
}

TEST(MPPIConfigTest, ParsesSamplingAndExpandsScalarActionParameters) {
  const YAML::Node root = YAML::Load(R"(
mppi:
  horizon_steps: 15
  dt: 0.02
  num_rollouts: 64
  temperature: 0.8
  random_seed: 42
  action:
    lower_bound: -0.003
    upper_bound: 0.004
    noise_std: 0.001
)");

  const auto config =
      mppi_core::ParseMPPIConfig(root["mppi"], 3, mppi_core::MPPIConfig{});

  EXPECT_EQ(config.horizon_steps, 15U);
  EXPECT_EQ(config.num_rollouts, 64U);
  EXPECT_EQ(config.action_dim, 3U);
  EXPECT_NEAR(config.dt, 0.02, kTolerance);
  EXPECT_NEAR(config.temperature, 0.8, kTolerance);
  EXPECT_EQ(config.random_seed, 42U);
  ASSERT_EQ(config.action_lower_bound.size(), 3);
  ASSERT_EQ(config.action_upper_bound.size(), 3);
  ASSERT_EQ(config.action_noise_std.size(), 3);
  for (Eigen::Index i = 0; i < 3; ++i) {
    EXPECT_NEAR(config.action_lower_bound[i], -0.003, kTolerance);
    EXPECT_NEAR(config.action_upper_bound[i], 0.004, kTolerance);
    EXPECT_NEAR(config.action_noise_std[i], 0.001, kTolerance);
  }
}

TEST(MPPIConfigTest, RejectsWrongSizedActionVectors) {
  const YAML::Node root = YAML::Load(R"(
mppi:
  action:
    lower_bound: [-0.1, -0.2]
)");

  EXPECT_THROW((void)mppi_core::ParseMPPIConfig(root["mppi"], 3,
                                                mppi_core::MPPIConfig{}),
               std::invalid_argument);
}
