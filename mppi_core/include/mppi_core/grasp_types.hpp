// Copyright 2026
//
// Licensed under the Apache License, Version 2.0.

#pragma once

#include <Eigen/Core>

#include "mppi_core/robot_command.hpp"
#include "mppi_core/tactile/tactile_state.hpp"

namespace mppi_core {

struct GraspRolloutConfig;
struct ContactForceCorrectionState;
struct ContactForceProjectionConfig;
struct ContactForceRolloutConfig;
struct PinocchioContactKinematicsContext;

struct GraspObservation {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Eigen::VectorXd q_measured;
  Eigen::VectorXd v_measured;
  Eigen::VectorXd q_ref_current;
  Eigen::VectorXd v_ref_current;
  Eigen::VectorXd tau;

  TactileState tactile;
  const PinocchioContactKinematicsContext* contact_kinematics{nullptr};
  const GraspRolloutConfig* grasp_rollout_config{nullptr};
  const ContactForceProjectionConfig* contact_force_projection_config{nullptr};
  const ContactForceRolloutConfig* contact_force_rollout_config{nullptr};
  const ContactForceCorrectionState* contact_force_correction_state{nullptr};
  double time_s{0.0};
};

using GraspCommand = RobotCommand;

}  // namespace mppi_core
