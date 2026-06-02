// Copyright 2026
//
// Licensed under the Apache License, Version 2.0.

#pragma once

#include <Eigen/Core>

#include "mppi_core/tactile/tactile_state.hpp"

namespace mppi_core {

struct GraspState {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  bool valid{false};

  Eigen::VectorXd q;
  Eigen::VectorXd dq;

  // Always available. For measured states this is measured joint torque; for
  // rollout states this is predicted/commanded torque from the rollout model.
  Eigen::VectorXd tau;

  // TactileState may be measured or predicted inside rollout.
  TactileState tactile;
};

inline GraspState MakeGraspState(const Eigen::VectorXd& q,
                                 const Eigen::VectorXd& dq,
                                 const Eigen::VectorXd& tau,
                                 const TactileState& tactile) {
  GraspState state;
  state.valid = tactile.valid && dq.size() == tau.size() && q.allFinite() &&
                dq.allFinite() && tau.allFinite();
  state.q = q;
  state.dq = dq;
  state.tau = tau;
  state.tactile = tactile;
  return state;
}

}  // namespace mppi_core
