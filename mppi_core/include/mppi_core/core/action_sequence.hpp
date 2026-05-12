// Copyright 2026
//
// Licensed under the Apache License, Version 2.0.

#pragma once

#include <cstddef>

#include <Eigen/Core>

namespace mppi_core {

class ActionSequence {
 public:
  using Matrix = Eigen::MatrixXd;

  ActionSequence() = default;
  ActionSequence(std::size_t action_dim, std::size_t horizon_steps) {
    Resize(action_dim, horizon_steps);
  }

  void Resize(std::size_t action_dim, std::size_t horizon_steps);
  void SetZero();
  void ShiftAndRepeatLast();

  std::size_t actionDim() const {
    return static_cast<std::size_t>(values_.rows());
  }
  std::size_t horizonSteps() const {
    return static_cast<std::size_t>(values_.cols());
  }

  const Matrix& values() const { return values_; }
  Matrix& values() { return values_; }

  Eigen::VectorXd action(std::size_t step) const;
  Eigen::VectorXd firstAction() const;
  void setAction(std::size_t step, const Eigen::Ref<const Eigen::VectorXd>& u);

 private:
  Matrix values_;
};

}  // namespace mppi_core
