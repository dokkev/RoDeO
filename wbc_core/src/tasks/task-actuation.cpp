//
// Copyright (c) 2017 CNRS
//

#include <wbc_core/tasks/task-actuation.hpp>

namespace wbc {
namespace tasks {

TaskActuation::TaskActuation(const std::string& name, RobotSystem& robot)
    : TaskBase(name, robot) {}

}  // namespace tasks
}  // namespace wbc
