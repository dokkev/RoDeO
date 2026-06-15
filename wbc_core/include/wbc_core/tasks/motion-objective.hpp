//
// Copyright (c) 2026
//
// Per-tick motion objective view produced by a motion task.
//

#ifndef WBC_CORE_TASKS_MOTION_OBJECTIVE_HPP_
#define WBC_CORE_TASKS_MOTION_OBJECTIVE_HPP_

#include <string>
#include <utility>

#include "wbc_core/math/constraint-base.hpp"
#include "wbc_core/math/fwd.hpp"

namespace wbc {

struct MotionObjective {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  std::string name;
  const math::ConstraintBase* constraint{nullptr};
  unsigned int level{1};
  double weight{1.0};

  MotionObjective() = default;

  MotionObjective(std::string name_in,
                  const math::ConstraintBase* constraint_in,
                  unsigned int level_in, double weight_in)
      : name(std::move(name_in)),
        constraint(constraint_in),
        level(level_in),
        weight(weight_in) {}

  bool isValid() const { return constraint != nullptr; }
  bool isEquality() const { return constraint && constraint->isEquality(); }
  bool isInequality() const { return constraint && constraint->isInequality(); }
  bool isBound() const { return constraint && constraint->isBound(); }

  const math::Matrix& matrix() const { return constraint->matrix(); }
  const math::Vector& vector() const { return constraint->vector(); }
  const math::Vector& lowerBound() const { return constraint->lowerBound(); }
  const math::Vector& upperBound() const { return constraint->upperBound(); }
};

}  // namespace wbc

#endif  // WBC_CORE_TASKS_MOTION_OBJECTIVE_HPP_
