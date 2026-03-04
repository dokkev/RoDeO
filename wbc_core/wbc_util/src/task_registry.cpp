/**
 * @file wbc_core/wbc_util/src/task_registry.cpp
 * @brief Doxygen documentation for task_registry module.
 */
#include "wbc_util/task_registry.hpp"

#include "wbc_formulation/force_task.hpp"
#include "wbc_formulation/interface/task.hpp"

namespace wbc {

void TaskRegistry::AddMotionTask(const std::string& name,
                                 std::unique_ptr<Task> task) {
  if (name.empty() || task == nullptr) {
    return;
  }
  motion_task_map_[name] = std::move(task);
}

Task* TaskRegistry::GetMotionTask(const std::string& name) const {
  const auto it = motion_task_map_.find(name);
  if (it == motion_task_map_.end()) {
    return nullptr;
  }
  return it->second.get();
}

void TaskRegistry::AddForceTask(const std::string& name,
                                std::unique_ptr<ForceTask> task) {
  if (name.empty() || task == nullptr) {
    return;
  }
  force_task_map_[name] = std::move(task);
}

ForceTask* TaskRegistry::GetForceTask(const std::string& name) const {
  const auto it = force_task_map_.find(name);
  if (it == force_task_map_.end()) {
    return nullptr;
  }
  return it->second.get();
}

void TaskRegistry::Clear() {
  motion_task_map_.clear();
  force_task_map_.clear();
}

} // namespace wbc
