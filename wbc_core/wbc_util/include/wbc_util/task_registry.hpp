/**
 * @file wbc_core/wbc_util/include/wbc_util/task_registry.hpp
 * @brief Doxygen documentation for task_registry module.
 */
#pragma once

#include <memory>
#include <string>
#include <unordered_map>

namespace wbc {
class Task;
class ForceTask;
} // namespace wbc

namespace wbc {

/**
 * @brief Ownership registry for motion/force tasks referenced by name.
 */
class TaskRegistry {
public:
  TaskRegistry() = default;
  ~TaskRegistry() = default;

  /**
   * @brief Add or replace a motion task entry.
   */
  void AddMotionTask(const std::string& name, std::unique_ptr<Task> task);
  /**
   * @brief Get motion task by name, or nullptr when absent.
   */
  Task* GetMotionTask(const std::string& name) const;

  /**
   * @brief Add or replace a force task entry.
   */
  void AddForceTask(const std::string& name, std::unique_ptr<ForceTask> task);
  /**
   * @brief Get force task by name, or nullptr when absent.
   */
  ForceTask* GetForceTask(const std::string& name) const;

  const std::unordered_map<std::string, std::unique_ptr<Task>>&
  GetMotionTasks() const {
    return motion_task_map_;
  }

  const std::unordered_map<std::string, std::unique_ptr<ForceTask>>&
  GetForceTasks() const {
    return force_task_map_;
  }

  void Clear();

private:
  std::unordered_map<std::string, std::unique_ptr<Task>> motion_task_map_;
  std::unordered_map<std::string, std::unique_ptr<ForceTask>> force_task_map_;
};

} // namespace wbc
