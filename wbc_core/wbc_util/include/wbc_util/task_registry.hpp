#pragma once

#include <memory>
#include <string>
#include <unordered_map>

namespace wbc {
class Task;
class ForceTask;
} // namespace wbc

namespace wbc {

class TaskRegistry {
public:
  TaskRegistry() = default;
  ~TaskRegistry() = default;

  void AddMotionTask(const std::string& name, std::unique_ptr<Task> task);
  Task* GetMotionTask(const std::string& name) const;

  void AddForceTask(const std::string& name, std::unique_ptr<ForceTask> task);
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
