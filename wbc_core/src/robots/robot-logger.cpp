//
// Copyright (c) 2026
//

#include "wbc_core/robots/robot-logger.hpp"

#include "wbc_core/robots/robot-system.hpp"

namespace wbc {
namespace robots {

void RobotLogger::Initialize(const RobotSystem& robot) {
  cmd.Initialize(robot);
  qddot_sol = math::Vector::Zero(robot.nv());
  q_cmd = math::Vector::Zero(robot.nq_joints());
  qdot_cmd = math::Vector::Zero(robot.nv_joints());
  tau_ff_cmd = math::Vector::Zero(robot.na());
  tau_fb_cmd = math::Vector::Zero(robot.na());
  tau_cmd = math::Vector::Zero(robot.na());
}

void RobotLogger::UpdateCommand(const RobotCommand& command) {
  cmd = command;
}

}  // namespace robots
}  // namespace wbc
