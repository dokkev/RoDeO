// Copyright 2026
//
// Licensed under the Apache License, Version 2.0.

#pragma once

#include <cstddef>
#include <cstdint>

#include <Eigen/Core>

namespace mppi_core {

struct MPPIConfig {
  std::size_t horizon_steps{20};
  std::size_t num_rollouts{128};
  std::size_t action_dim{0};
  double dt{0.01};
  double temperature{1.0};
  std::uint32_t random_seed{1};

  Eigen::VectorXd action_lower_bound;
  Eigen::VectorXd action_upper_bound;
  Eigen::VectorXd action_noise_std;
};

}  // namespace mppi_core
