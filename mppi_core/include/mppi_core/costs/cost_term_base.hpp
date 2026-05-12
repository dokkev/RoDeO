// Copyright 2026
//
// Licensed under the Apache License, Version 2.0.

#pragma once

#include <cstddef>

#include <Eigen/Core>

#include "mppi_core/model/rollout.hpp"

namespace mppi_core {

struct CostContext {
  const RolloutContext* rollout{nullptr};
  std::size_t step_index{0};
  double time_s{0.0};
};

class CostTermBase {
 public:
  virtual ~CostTermBase() = default;

  virtual double Evaluate(const RobotRolloutState& state,
                          const Eigen::Ref<const Eigen::VectorXd>& action,
                          const CostContext& context) const = 0;
};

}  // namespace mppi_core
