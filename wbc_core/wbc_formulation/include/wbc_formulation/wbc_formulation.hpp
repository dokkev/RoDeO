/**
 * @file wbc_core/wbc_formulation/include/wbc_formulation/wbc_formulation.hpp
 * @brief Doxygen documentation for wbc_formulation module.
 */
#pragma once

#include <cstddef>
#include <vector>

#include "wbc_formulation/contact.hpp"
#include "wbc_formulation/kinematic_constraint.hpp" // joint limits and kinematic constraints
#include "wbc_formulation/force_task.hpp"
#include "wbc_formulation/interface/task.hpp"

namespace wbc {

/**
 * @brief Runtime role of a motion task in the split IK/ID pipeline.
 */
enum class MotionTaskRole {
  kOperationalTask,  ///< tracked directly in ID-QP task-space cost
  kPostureTask,      ///< used by IK/posture reference generation
};

/**
 * @brief Runtime formulation bundle passed to WBC solvers each tick.
 *
 * @details
 * Members are non-owning pointers. Ownership remains in task/constraint
 * registries built by runtime config.
 */
struct WbcFormulation {
  using TaskList = std::vector<Task*>;
  using ContactConstraintList = std::vector<Contact*>;
  using ForceTaskList = std::vector<ForceTask*>;
  using KinematicConstraintList = std::vector<Constraint*>;

  // Motion tasks are split by runtime role:
  // - operational_tasks: ID-QP direct task-space tracking terms.
  // - posture_tasks: IK/posture bias references only.
  TaskList operational_tasks;
  TaskList posture_tasks;
  ContactConstraintList contact_constraints;
  ForceTaskList force_tasks;
  KinematicConstraintList kinematic_constraints;

  inline void Reserve(std::size_t operational_task_cap,
                      std::size_t posture_task_cap,
                      std::size_t contact_cap,
                      std::size_t force_task_cap,
                      std::size_t kin_cap = 5) {
    operational_tasks.reserve(operational_task_cap);
    posture_tasks.reserve(posture_task_cap);
    contact_constraints.reserve(contact_cap);
    force_tasks.reserve(force_task_cap);
    kinematic_constraints.reserve(kin_cap);
  }

  inline void Clear() noexcept {
    operational_tasks.clear();
    posture_tasks.clear();
    contact_constraints.clear();
    force_tasks.clear();
    kinematic_constraints.clear();
  }

  [[nodiscard]] inline bool Empty() const noexcept {
    return operational_tasks.empty() && posture_tasks.empty() &&
           contact_constraints.empty() && force_tasks.empty() &&
           kinematic_constraints.empty();
  }
};

} // namespace wbc
