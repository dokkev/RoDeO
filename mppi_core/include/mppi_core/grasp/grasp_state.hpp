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

  // Optional measured or commanded joint torque used by force-aware rollout
  // paths. GraspState remains generic and does not own sensor-specific logic.
  bool has_measured_tau{false};
  Eigen::VectorXd measured_tau;

  // TactileState may be measured or predicted inside rollout.
  TactileState tactile;
};

inline GraspState MakeGraspState(const Eigen::VectorXd& q,
                                 const Eigen::VectorXd& dq,
                                 const TactileState& tactile) {
  GraspState state;
  state.valid = tactile.valid && q.size() == dq.size();
  state.q = q;
  state.dq = dq;
  state.tactile = tactile;
  return state;
}

inline GraspState MakeGraspState(const Eigen::VectorXd& q,
                                 const Eigen::VectorXd& dq,
                                 const Eigen::VectorXd& measured_tau,
                                 const TactileState& tactile) {
  GraspState state = MakeGraspState(q, dq, tactile);
  state.has_measured_tau = measured_tau.size() == dq.size();
  state.measured_tau = measured_tau;
  state.valid = state.valid && state.has_measured_tau;
  return state;
}

}  // namespace mppi_core
