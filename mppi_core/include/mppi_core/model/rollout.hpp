// Copyright 2026
//
// Licensed under the Apache License, Version 2.0.

#pragma once

#include <cstddef>

#include <Eigen/Core>

#include "mppi_core/grasp/grasp_state.hpp"
#include "mppi_core/grasp_types.hpp"

namespace mppi_core {

using RobotRolloutState = GraspState;

struct GraspRolloutConfig;
struct ContactForceCorrectionState;
struct ContactForceProjectionConfig;
struct ContactForceRolloutConfig;
struct PinocchioContactKinematicsContext;

struct RolloutContext {
  const TactileState* tactile{nullptr};
  const ObjectPrior* object{nullptr};
  const TactileDisturbanceSet* tactile_disturbances{nullptr};
  const RobotRolloutState* measured_state{nullptr};
  const RobotRolloutState* initial_reference_state{nullptr};
  const PinocchioContactKinematicsContext* contact_kinematics{nullptr};
  const GraspRolloutConfig* grasp_rollout_config{nullptr};
  const ContactForceProjectionConfig* contact_force_projection_config{nullptr};
  const ContactForceRolloutConfig* contact_force_rollout_config{nullptr};
  const ContactForceCorrectionState* contact_force_correction_state{nullptr};
  bool has_gravity_context{false};
  Eigen::Vector3d gravity_in_sensor_frame{Eigen::Vector3d::Zero()};
};

class RolloutModelBase {
 public:
  virtual ~RolloutModelBase() = default;

  virtual std::size_t actionDim() const = 0;

  virtual void Step(const RobotRolloutState& state,
                    const Eigen::Ref<const Eigen::VectorXd>& action,
                    const RolloutContext& context, double dt,
                    RobotRolloutState* next_state) const = 0;
};

}  // namespace mppi_core
