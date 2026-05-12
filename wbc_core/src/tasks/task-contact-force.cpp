//
// Copyright (c) 2017 CNRS
//

#include "wbc_core/tasks/task-contact-force.hpp"

namespace wbc {
namespace tasks {

TaskContactForce::TaskContactForce(const std::string& name, RobotSystem& robot)
    : TaskBase(name, robot) {}

}  // namespace tasks
}  // namespace wbc
