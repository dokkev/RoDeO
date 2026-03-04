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

  // Contract:
  // - motion_tasks is already sorted by ascending priority.
  // - compiler/orchestrator is responsible for ordering before runtime.
  TaskList motion_tasks;
  ContactConstraintList contact_constraints;
  ForceTaskList force_tasks;
  KinematicConstraintList kinematic_constraints;

  inline void Reserve(std::size_t motion_task_cap, std::size_t contact_cap,
                      std::size_t force_task_cap, std::size_t kin_cap = 5) {
    motion_tasks.reserve(motion_task_cap);
    contact_constraints.reserve(contact_cap);
    force_tasks.reserve(force_task_cap);
    kinematic_constraints.reserve(kin_cap);
  }

  inline void Clear() noexcept {
    motion_tasks.clear();
    contact_constraints.clear();
    force_tasks.clear();
    kinematic_constraints.clear();
  }

  [[nodiscard]] inline bool Empty() const noexcept {
    return motion_tasks.empty() && contact_constraints.empty() &&
           force_tasks.empty() && kinematic_constraints.empty();
  }
};

} // namespace wbc
