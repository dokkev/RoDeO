// Copyright 2026

#ifndef CONTROL_ARCHITECTURE_STATE_MACHINE_STATE_UTIL_HPP_
#define CONTROL_ARCHITECTURE_STATE_MACHINE_STATE_UTIL_HPP_

#include <optional>
#include <string>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <yaml-cpp/yaml.h>

#include "wbc_core/robots/robot-system.hpp"
#include "wbc_core/tasks/task-joint-posture.hpp"
#include "wbc_core/trajectories/trajectory-base.hpp"

namespace wbc::state_util {

Eigen::VectorXd ReadOptionalVector(const YAML::Node& node,
                                   const std::string& key,
                                   Eigen::Index expected_size,
                                   const std::string& owner);

std::optional<Eigen::Vector3d> ReadOptionalVector3(const YAML::Node& node,
                                                   const std::string& key,
                                                   const std::string& owner);

std::optional<Eigen::Quaterniond> ReadOptionalQuaternion(
    const YAML::Node& node, const std::string& key,
    const std::string& owner);

Eigen::VectorXd CurrentJointPosition(const wbc::robots::RobotSystem& robot);

void SetJointPostureReference(wbc::tasks::TaskJointPosture& task,
                              wbc::trajectories::TrajectorySample& sample,
                              const Eigen::VectorXd& q,
                              const Eigen::VectorXd& qdot,
                              const Eigen::VectorXd& qddot);

void SetJointPostureReference(wbc::tasks::TaskJointPosture& task,
                              const Eigen::VectorXd& q,
                              const Eigen::VectorXd& qdot,
                              const Eigen::VectorXd& qddot);

}  // namespace wbc::state_util

#endif  // CONTROL_ARCHITECTURE_STATE_MACHINE_STATE_UTIL_HPP_
