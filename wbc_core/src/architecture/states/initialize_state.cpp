//
// Copyright (c) 2026
//

#include "wbc_core/architecture/states/initialize_state.hpp"

namespace {

Eigen::VectorXd parseVectorOrScalar(const YAML::Node& node, int dim) {
  if (node.IsSequence()) {
    Eigen::VectorXd v(node.size());
    for (std::size_t i = 0; i < node.size(); ++i) {
      v(static_cast<int>(i)) = node[i].as<double>();
    }
    return v;
  }
  return Eigen::VectorXd::Constant(dim, node.as<double>());
}

}  // namespace

namespace wbc {

void InitializeState::SetParameters(const YAML::Node& node) {
  StateMachine::SetParameters(node);

  // Find the joint posture task
  for (auto& [name, task] : assigned_tasks_) {
    auto jt = std::dynamic_pointer_cast<tsid::tasks::TaskJointPosture>(task);
    if (jt) {
      jpos_task_ = jt;
      break;
    }
  }

  // Parse target joint positions
  const int na = robot_->na();
  if (node["target_jpos"]) {
    q_target_ = parseVectorOrScalar(node["target_jpos"], na);
  } else {
    q_target_ = Eigen::VectorXd::Zero(na);
  }

  zeros_ = Eigen::VectorXd::Zero(na);
}

void InitializeState::FirstVisit() {
  // Capture current joint positions as start
  const int na = robot_->na();
  const auto& q = sp_->nominal_jpos;
  if (q.size() >= na) {
    q_start_ = q.tail(na);
  } else {
    q_start_ = Eigen::VectorXd::Zero(na);
  }
}

void InitializeState::OneStep() {
  if (!jpos_task_) return;

  // Linear interpolation from q_start to q_target over duration
  double alpha = 0.0;
  if (duration_ > 0.0) {
    alpha = std::min(current_time_ / duration_, 1.0);
    // Smooth cosine ramp
    alpha = 0.5 * (1.0 - std::cos(alpha * M_PI));
  } else {
    alpha = 1.0;
  }

  if (!sample_initialized_) {
    const int na = q_target_.size();
    q_des_.resize(na);
    sample_ = tsid::trajectories::TrajectorySample(na);
    sample_initialized_ = true;
  }

  q_des_.noalias() = (1.0 - alpha) * q_start_ + alpha * q_target_;

  sample_.setValue(q_des_);
  sample_.setDerivative(zeros_);
  sample_.setSecondDerivative(zeros_);
  jpos_task_->setReference(sample_);
}

void InitializeState::LastVisit() {}

}  // namespace wbc
