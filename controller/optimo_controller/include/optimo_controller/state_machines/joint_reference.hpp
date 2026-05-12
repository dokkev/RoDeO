// Copyright 2024 Roboligent, Inc.
//
// Licensed under the Apache License, Version 2.0.

#pragma once

#include <memory>
#include <string>

#include <Eigen/Core>

#include "control_architecture/state_machine/state_machine.hpp"
#include "wbc_core/tasks/task-joint-posture.hpp"
#include "wbc_core/trajectories/trajectory-base.hpp"

namespace optimo_controller::state_machines {

class JointReferenceState final : public wbc::State {
 public:
  STATE_NAME("joint_reference");

  JointReferenceState(wbc::StateId id, const std::string& name,
                      const wbc::StateContext& ctx);

  void Configure(const YAML::Node& node) override;
  void OnEnter() override;
  void OnUpdate() override;
  void OnExit() override;

 private:
  void ApplyReference();

  std::string task_name_{"jpos_task"};
  std::shared_ptr<wbc::tasks::TaskJointPosture> task_;
  Eigen::VectorXd target_q_;
  Eigen::VectorXd target_qdot_;
  Eigen::VectorXd target_qddot_;
  wbc::trajectories::TrajectorySample ref_;
};

}  // namespace optimo_controller::state_machines
