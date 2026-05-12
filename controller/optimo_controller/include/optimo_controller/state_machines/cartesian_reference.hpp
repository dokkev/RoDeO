// Copyright 2024 Roboligent, Inc.
//
// Licensed under the Apache License, Version 2.0.

#pragma once

#include <memory>
#include <optional>
#include <string>

#include <Eigen/Core>
#include <pinocchio/spatial/se3.hpp>

#include "control_architecture/state_machine/state_machine.hpp"
#include "wbc_core/tasks/task-se3-equality.hpp"
#include "wbc_core/trajectories/trajectory-base.hpp"

namespace optimo_controller::state_machines {

class CartesianReferenceState final : public wbc::State {
 public:
  STATE_NAME("cartesian_reference");

  CartesianReferenceState(wbc::StateId id, const std::string& name,
                          const wbc::StateContext& ctx);

  void Configure(const YAML::Node& node) override;
  void OnEnter() override;
  void OnUpdate() override;
  void OnExit() override;

 private:
  void CaptureCurrentPose();
  void ApplyConfiguredPoseTarget();
  void ApplyReference();

  std::string task_name_{"ee_task"};
  std::shared_ptr<wbc::tasks::TaskSE3Equality> task_;
  wbc::trajectories::TrajectorySample ref_{12, 6};
  pinocchio::SE3 target_pose_{pinocchio::SE3::Identity()};
  std::optional<Eigen::Vector3d> target_translation_;
  std::optional<Eigen::Quaterniond> target_quat_;
  Eigen::Vector3d linear_velocity_{Eigen::Vector3d::Zero()};
  Eigen::Vector3d angular_velocity_{Eigen::Vector3d::Zero()};
};

}  // namespace optimo_controller::state_machines
