// Copyright 2026

#include "control_architecture/state_machine/state_util.hpp"

#include <stdexcept>
#include <vector>

#include "wbc_core/trajectories/trajectory-base.hpp"

namespace wbc::state_util {

Eigen::VectorXd ReadOptionalVector(const YAML::Node& node,
                                   const std::string& key,
                                   Eigen::Index expected_size,
                                   const std::string& owner) {
  if (!node[key]) {
    return Eigen::VectorXd();
  }

  const auto values = node[key].as<std::vector<double>>();
  if (static_cast<Eigen::Index>(values.size()) != expected_size) {
    throw std::invalid_argument(owner + " parameter '" + key +
                                "' has wrong size");
  }

  Eigen::VectorXd out(expected_size);
  for (Eigen::Index i = 0; i < expected_size; ++i) {
    out(i) = values[static_cast<std::size_t>(i)];
  }
  return out;
}

std::optional<Eigen::Vector3d> ReadOptionalVector3(const YAML::Node& node,
                                                   const std::string& key,
                                                   const std::string& owner) {
  if (!node[key]) {
    return std::nullopt;
  }

  const auto values = node[key].as<std::vector<double>>();
  if (values.size() != 3U) {
    throw std::invalid_argument(owner + " parameter '" + key +
                                "' must have size 3");
  }

  return Eigen::Vector3d(values[0], values[1], values[2]);
}

std::optional<Eigen::Quaterniond> ReadOptionalQuaternion(
    const YAML::Node& node, const std::string& key,
    const std::string& owner) {
  if (!node[key]) {
    return std::nullopt;
  }

  const auto values = node[key].as<std::vector<double>>();
  if (values.size() != 4U) {
    throw std::invalid_argument(owner + " parameter '" + key +
                                "' must have size 4");
  }

  Eigen::Quaterniond quat(values[0], values[1], values[2], values[3]);
  if (quat.norm() < 1e-12) {
    throw std::invalid_argument(owner + " parameter '" + key +
                                "' has near-zero quaternion");
  }
  quat.normalize();
  return quat;
}

Eigen::VectorXd CurrentJointPosition(const wbc::robots::RobotSystem& robot) {
  return robot.jointState().q;
}

void SetJointPostureReference(wbc::tasks::TaskJointPosture& task,
                              wbc::trajectories::TrajectorySample& sample,
                              const Eigen::VectorXd& q,
                              const Eigen::VectorXd& qdot,
                              const Eigen::VectorXd& qddot) {
  if (sample.pos.size() != q.size() || sample.vel.size() != qdot.size()) {
    sample.resize(static_cast<unsigned int>(q.size()),
                  static_cast<unsigned int>(qdot.size()));
  }
  sample.setValue(q);
  sample.setDerivative(qdot);
  sample.setSecondDerivative(qddot);
  task.setReference(sample);
}

void SetJointPostureReference(wbc::tasks::TaskJointPosture& task,
                              const Eigen::VectorXd& q,
                              const Eigen::VectorXd& qdot,
                              const Eigen::VectorXd& qddot) {
  wbc::trajectories::TrajectorySample sample(
      static_cast<unsigned int>(q.size()),
      static_cast<unsigned int>(qdot.size()));
  SetJointPostureReference(task, sample, q, qdot, qddot);
}

}  // namespace wbc::state_util
