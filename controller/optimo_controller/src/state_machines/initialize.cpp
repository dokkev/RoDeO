// Copyright 2024 Roboligent, Inc.
//
// Licensed under the Apache License, Version 2.0.

#include "optimo_controller/state_machines/initialize.hpp"

#include "control_architecture/state_machine/state_util.hpp"

namespace optimo_controller::state_machines {
namespace state_util = wbc::state_util;

InitializeState::InitializeState(wbc::StateId id, const std::string& name,
                                 const wbc::StateContext& ctx)
    : State(id, name, ctx) {}

void InitializeState::Configure(const YAML::Node& node) {
  State::Configure(node);
  if (node["task_name"]) {
    task_name_ = node["task_name"].as<std::string>();
  }

  target_q_ = state_util::ReadOptionalVector(
      node, "target_jpos", robot_->nq_actuated(), "InitializeState");
  target_qdot_ = state_util::ReadOptionalVector(
      node, "target_jvel", robot_->na(), "InitializeState");
  target_qddot_ = state_util::ReadOptionalVector(
      node, "target_jacc", robot_->na(), "InitializeState");
  ref_.resize(static_cast<unsigned int>(robot_->nq_actuated()),
              static_cast<unsigned int>(robot_->na()));
  task_ =
      RequireTask<wbc::tasks::TaskJointPosture>(task_name_, "InitializeState");
}

void InitializeState::OnEnter() {
  if (target_q_.size() != robot_->nq_actuated()) {
    target_q_ = state_util::CurrentJointPosition(*robot_);
  }
  if (target_qdot_.size() != robot_->na()) {
    target_qdot_ = Eigen::VectorXd::Zero(robot_->na());
  }
  if (target_qddot_.size() != robot_->na()) {
    target_qddot_ = Eigen::VectorXd::Zero(robot_->na());
  }
  ApplyReference();
}

void InitializeState::OnUpdate() {
  ApplyReference();
}

void InitializeState::OnExit() {}

void InitializeState::ApplyReference() {
  state_util::SetJointPostureReference(*task_, ref_, target_q_, target_qdot_,
                                       target_qddot_);
}

}  // namespace optimo_controller::state_machines
