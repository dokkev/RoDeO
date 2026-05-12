// Copyright 2026
//
// Licensed under the Apache License, Version 2.0.

#include "mppi_core/core/action_sequence.hpp"

#include <stdexcept>

namespace mppi_core {

void ActionSequence::Resize(std::size_t action_dim, std::size_t horizon_steps) {
  values_.setZero(static_cast<Eigen::Index>(action_dim),
                  static_cast<Eigen::Index>(horizon_steps));
}

void ActionSequence::SetZero() {
  values_.setZero();
}

void ActionSequence::ShiftAndRepeatLast() {
  if (horizonSteps() <= 1) {
    return;
  }
  values_.leftCols(values_.cols() - 1) =
      values_.rightCols(values_.cols() - 1).eval();
  values_.rightCols(1) = values_.col(values_.cols() - 2);
}

Eigen::VectorXd ActionSequence::action(std::size_t step) const {
  if (step >= horizonSteps()) {
    throw std::out_of_range("ActionSequence::action: step out of range");
  }
  return values_.col(static_cast<Eigen::Index>(step));
}

Eigen::VectorXd ActionSequence::firstAction() const {
  if (horizonSteps() == 0) {
    return Eigen::VectorXd();
  }
  return values_.col(0);
}

void ActionSequence::setAction(std::size_t step,
                               const Eigen::Ref<const Eigen::VectorXd>& u) {
  if (step >= horizonSteps()) {
    throw std::out_of_range("ActionSequence::setAction: step out of range");
  }
  if (u.size() != values_.rows()) {
    throw std::invalid_argument(
        "ActionSequence::setAction: dimension mismatch");
  }
  values_.col(static_cast<Eigen::Index>(step)) = u;
}

}  // namespace mppi_core
