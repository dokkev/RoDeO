//
// Copyright (c) 2017 CNRS
//

#include "wbc_core/tasks/task-base.hpp"

namespace wbc {
namespace tasks {
TaskBase::TaskBase(const std::string& name, RobotSystem& robot)
    : m_name(name), m_robot(robot) {}

const std::string& TaskBase::name() const { return m_name; }

void TaskBase::name(const std::string& name) { m_name = name; }

}  // namespace tasks
}  // namespace wbc
