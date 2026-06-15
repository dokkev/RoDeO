//
// Copyright (c) 2017 CNRS, 2026
//

#include "wbc_core/formulations/inverse-dynamics.hpp"

namespace wbc {

TaskLevel::TaskLevel(tasks::TaskBase& task, unsigned int priority)
    : task(task), priority(priority) {}

TaskLevelForce::TaskLevelForce(tasks::TaskContactForce& task,
                               unsigned int priority)
    : task(task), priority(priority) {}

InverseDynamicsBase::InverseDynamicsBase(
    const std::string& name, RobotSystem& robot, bool verbose)
    : m_name(name), m_robot(robot), m_verbose(verbose) {}

MeasuredForceLevel::MeasuredForceLevel(
    contacts::MeasuredForceBase& measuredForce)
    : measuredForce(measuredForce) {}
}  // namespace wbc
