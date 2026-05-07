//
// Copyright (c) 2026
//
// Final-form WBMC step input types.
//

#ifndef __wbc_controller_wbmc_step_input_hpp__
#define __wbc_controller_wbmc_step_input_hpp__

#include <algorithm>
#include <string>
#include <variant>
#include <vector>

#include "wbc_core/controller/wbmc-hierarchy-policy.hpp"
#include "wbc_core/math/fwd.hpp"

namespace tsid {

struct WBMCRegularizationParams {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  double w_delta_qddot{1e-4};
  double w_lambda{1e-5};
};

struct MotionObjective {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  enum class ConstraintKind {
    kEquality,
    kInequality,
  };

  std::string name;
  math::Matrix J;
  math::Vector a_des;
  math::Vector lower_bound;
  math::Vector upper_bound;
  ConstraintKind kind{ConstraintKind::kEquality};

  bool isEquality() const { return kind == ConstraintKind::kEquality; }
  bool isInequality() const { return kind == ConstraintKind::kInequality; }
};

struct ContactSnapshot {
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

struct JointAccelerationObjective {
  std::string name{"joint-accel-objective"};
  const math::Vector* qddot_target{nullptr};
};

struct SoftObjective {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Term = std::variant<JointAccelerationObjective, MotionObjective>;

  unsigned int level{1};
  double weight{1.0};
  Term term;

  static SoftObjective JointAcceleration(
      const JointAccelerationObjective& objective, unsigned int level,
      double weight) {
    SoftObjective ref;
    ref.level = level;
    ref.weight = weight;
    ref.term = objective;
    return ref;
  }

  static SoftObjective Motion(const MotionObjective& objective,
                              unsigned int level, double weight) {
    SoftObjective ref;
    ref.level = level;
    ref.weight = weight;
    ref.term = objective;
    return ref;
  }

  bool isJointAcceleration() const {
    return std::holds_alternative<JointAccelerationObjective>(term);
  }

  bool isMotion() const {
    return std::holds_alternative<MotionObjective>(term);
  }

  const JointAccelerationObjective& jointAcceleration() const {
    return std::get<JointAccelerationObjective>(term);
  }

  const MotionObjective& motion() const {
    return std::get<MotionObjective>(term);
  }
};

struct WBMCStepInput {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  const math::Vector* q{nullptr};
  const math::Vector* qdot{nullptr};
  const math::Vector* qddot_ref{nullptr};
  const math::Vector* tau_lb{nullptr};
  const math::Vector* tau_ub{nullptr};
  const math::Vector* h_ext{nullptr};

  std::vector<SoftObjective> objectives;
  std::vector<ContactSnapshot> contacts;

  WBMCHierarchyPolicy hierarchy;
  WBMCRegularizationParams regularization;

  bool hasState() const { return q != nullptr && qdot != nullptr; }

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

}  // namespace tsid

#endif  // ifndef __wbc_controller_wbmc_step_input_hpp__
