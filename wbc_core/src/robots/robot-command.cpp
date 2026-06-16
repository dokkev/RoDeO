//
// Copyright (c) 2026
//

#include "wbc_core/robots/robot-command.hpp"

#include "wbc_core/robots/robot-system.hpp"

namespace wbc {
namespace robots {

void RobotCommand::Initialize(const RobotSystem& robot) {
  Initialize(robot.nq_joints(), robot.nv_joints(), robot.na());
}

void RobotCommand::Initialize(Eigen::Index joint_dim) {
  Initialize(joint_dim, joint_dim, joint_dim);
}

void RobotCommand::Initialize(Eigen::Index nq_joints, Eigen::Index nv_joints,
                              Eigen::Index na) {
  q = math::Vector::Zero(nq_joints);
  qdot = math::Vector::Zero(nv_joints);
  tau = math::Vector::Zero(na);
}

}  // namespace robots
}  // namespace wbc
