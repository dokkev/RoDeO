//
// Copyright (c) 2026
//
// Final-form IDHQP solve problem types.
//

#ifndef __wbc_controller_id_problem_hpp__
#define __wbc_controller_id_problem_hpp__

#include <algorithm>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "wbc_core/constraints/actuator-torque-limits.hpp"
#include "wbc_core/controller/id-hierarchy-policy.hpp"
#include "wbc_core/math/constraint-base.hpp"
#include "wbc_core/math/fwd.hpp"

namespace wbc {

struct IDRegularizationParams {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  double w_delta_qddot{1e-4};
  double w_lambda{1e-5};
};

struct MotionConstraintRef {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  std::string name;
  const math::ConstraintBase* constraint{nullptr};

  MotionConstraintRef() = default;

  MotionConstraintRef(std::string name_in,
                      const math::ConstraintBase* constraint_in)
      : name(std::move(name_in)), constraint(constraint_in) {}

  bool isValid() const { return constraint != nullptr; }
  bool isEquality() const { return constraint && constraint->isEquality(); }
  bool isInequality() const { return constraint && constraint->isInequality(); }
  bool isBound() const { return constraint && constraint->isBound(); }

  const math::Matrix& matrix() const { return constraint->matrix(); }
  const math::Vector& vector() const { return constraint->vector(); }
  const math::Vector& lowerBound() const { return constraint->lowerBound(); }
  const math::Vector& upperBound() const { return constraint->upperBound(); }
};

struct ContactConstraintData {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  std::string name;
  math::Matrix Jc;
  math::Vector Jcdot_qdot;
  math::Matrix T;
  math::Matrix Uf;
  math::Vector uf_lb;
  math::Vector uf_ub;

  int lambdaDim() const { return static_cast<int>(T.cols()); }
  int motionDim() const { return static_cast<int>(Jc.rows()); }
};

struct JointAccelerationTarget {
  std::string name{"joint-accel-objective"};
  const math::Vector* qddot_target{nullptr};
};

struct ObjectiveTerm {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Data = std::variant<JointAccelerationTarget, MotionConstraintRef>;

  unsigned int level{1};
  double weight{1.0};
  Data data;

  static ObjectiveTerm MakeJointAccelerationTarget(
      const JointAccelerationTarget& target, unsigned int level,
      double weight) {
    ObjectiveTerm ref;
    ref.level = level;
    ref.weight = weight;
    ref.data = target;
    return ref;
  }

  static ObjectiveTerm MakeMotionConstraint(const MotionConstraintRef& task,
                                            unsigned int level, double weight) {
    ObjectiveTerm ref;
    ref.level = level;
    ref.weight = weight;
    ref.data = task;
    return ref;
  }

  bool isJointAccelerationTarget() const {
    return std::holds_alternative<JointAccelerationTarget>(data);
  }

  bool isMotionConstraint() const {
    return std::holds_alternative<MotionConstraintRef>(data);
  }

  const JointAccelerationTarget& jointAccelerationTarget() const {
    return std::get<JointAccelerationTarget>(data);
  }

  const MotionConstraintRef& motionConstraint() const {
    return std::get<MotionConstraintRef>(data);
  }
};

struct IDProblem {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  // Per-cycle view consumed immediately by IDHQP::solve().
  // MotionConstraintRef entries point at task-owned constraints and are not
  // long-lived snapshots.
  const math::Vector* qddot_ref{nullptr};
  const math::Vector* h_ext{nullptr};
  constraints::ActuatorTorqueLimits torque_limits;

  std::vector<ObjectiveTerm> objectives;
  std::vector<ContactConstraintData> contacts;

  IDHierarchyPolicy hierarchy;
  IDRegularizationParams regularization;

  unsigned int maxObjectiveLevel() const {
    unsigned int max_level = hierarchy.physicsLevel;
    for (const auto& objective : objectives) {
      max_level = std::max(max_level, objective.level);
    }
    return max_level;
  }

  bool hasValidHierarchy() const {
    for (const auto& objective : objectives) {
      if (!hierarchy.isObjectiveLevelValid(objective.level)) {
        return false;
      }
    }
    return hierarchy.isValidFor(maxObjectiveLevel());
  }
};

}  // namespace wbc

#endif  // ifndef __wbc_controller_id_problem_hpp__
