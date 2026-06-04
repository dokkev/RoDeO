// Copyright 2026
//
// Licensed under the Apache License, Version 2.0.

#pragma once

#include <cstddef>

#include <Eigen/Core>

#include "mppi_core/model/rollout.hpp"
#include "mppi_core/tactile/tactile_rollout_policy.hpp"

namespace mppi_core {

struct DeltaQReferenceRolloutConfig {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  TactileRolloutPolicy tactile_rollout_policy{
      TactileRolloutPolicy::kForceAwareRequired};
};

class DeltaQReferenceRolloutModel final : public RolloutModelBase {
 public:
  explicit DeltaQReferenceRolloutModel(std::size_t joint_dim);
  DeltaQReferenceRolloutModel(std::size_t joint_dim,
                              DeltaQReferenceRolloutConfig config);

  std::size_t actionDim() const override { return joint_dim_; }

  void Step(const RobotRolloutState& state,
            const Eigen::Ref<const Eigen::VectorXd>& action,
            const RolloutContext& context, double dt,
            RobotRolloutState* next_state) const override;

 private:
  std::size_t joint_dim_{0};
  DeltaQReferenceRolloutConfig config_;
};

}  // namespace mppi_core
