// Copyright 2026
//
// Licensed under the Apache License, Version 2.0.

#pragma once

#include <cstddef>

#include <Eigen/Core>

#include "mppi_core/grasp_types.hpp"

namespace mppi_core {

struct ContactPredictionState {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  bool initialized{false};
  bool contact_valid{false};
  bool centroid_valid{false};
  std::size_t contact_node_count{0};
  double normal_force_n{0.0};
  double tangential_load_n{0.0};
  double friction_margin_n{0.0};
  double friction_pyramid_margin_n{0.0};
  double slip_risk{0.0};
  Eigen::Vector2d centroid_m = Eigen::Vector2d::Zero();
};

struct GraspState {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Eigen::VectorXd q;
  Eigen::VectorXd v;
  NariTouchState tactile;
  bool tactile_initialized{false};
  ContactPredictionState contact;

  void Resize(std::size_t nq, std::size_t nv) {
    q.setZero(static_cast<Eigen::Index>(nq));
    v.setZero(static_cast<Eigen::Index>(nv));
    tactile = NariTouchState{};
    tactile_initialized = false;
    contact = ContactPredictionState{};
  }
};

using RobotRolloutState = GraspState;

struct RolloutContext {
  const NariTouchState* tactile{nullptr};
  const ObjectPrior* object{nullptr};
  const TactileDisturbanceSet* tactile_disturbances{nullptr};
  const RobotRolloutState* measured_state{nullptr};
  const RobotRolloutState* initial_reference_state{nullptr};
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
