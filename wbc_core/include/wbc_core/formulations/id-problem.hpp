//
// Copyright (c) 2026
//
// Final-form IDHQP solve problem types.
//

#ifndef WBC_CORE_FORMULATIONS_ID_PROBLEM_HPP_
#define WBC_CORE_FORMULATIONS_ID_PROBLEM_HPP_

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include "wbc_core/contacts/contact-constraint-data.hpp"
#include "wbc_core/constraints/joint-torque-limits.hpp"
#include "wbc_core/math/fwd.hpp"
#include "wbc_core/tasks/motion-objective.hpp"

namespace wbc {

struct IDRegularizationParams {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  double w_delta_qddot{1e-4};
  double w_lambda{1e-5};
};

struct JointAccelerationObjective {
  std::string name{"joint-accel-objective"};
  const math::Vector* qddot_target{nullptr};
  unsigned int level{2};
  double weight{1.0};

  JointAccelerationObjective() = default;

  JointAccelerationObjective(std::string name_in,
                             const math::Vector* qddot_target_in,
                             unsigned int level_in, double weight_in)
      : name(std::move(name_in)),
        qddot_target(qddot_target_in),
        level(level_in),
        weight(weight_in) {}
};

struct IDProblem {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  // Per-cycle view consumed immediately by IDHQP::solve().
  // MotionObjective and ContactConstraintData come from task/contact layers.
  // Runtime assembly should provide qddot_ref; nullptr is treated as zero.
  const math::Vector* qddot_ref{nullptr};
  const math::Vector* h_ext{nullptr};
  constraints::JointTorqueLimits joint_torque_limits;

  std::vector<MotionObjective> motion_objectives;
  std::vector<JointAccelerationObjective> joint_acceleration_objectives;
  std::vector<ContactConstraintData> contacts;

  IDRegularizationParams regularization;

  unsigned int maxObjectiveLevel() const {
    unsigned int max_level = 0;
    for (const auto& objective : motion_objectives) {
      if (objective.isEquality()) {
        max_level = std::max(max_level, objective.level);
      }
    }
    for (const auto& objective : joint_acceleration_objectives) {
      max_level = std::max(max_level, objective.level);
    }
    return max_level;
  }
};

}  // namespace wbc

#endif  // WBC_CORE_FORMULATIONS_ID_PROBLEM_HPP_
