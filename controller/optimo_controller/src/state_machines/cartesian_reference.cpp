// Copyright 2024 Roboligent, Inc.
//
// Licensed under the Apache License, Version 2.0.

#include "optimo_controller/state_machines/cartesian_reference.hpp"

#include <cmath>
#include <stdexcept>

#include <Eigen/Geometry>

#include "control_architecture/state_machine/state_util.hpp"
#include "wbc_core/math/utils.hpp"
#include "wbc_core/robots/robot-system.hpp"
#include "wbc_core/trajectories/trajectory-base.hpp"

namespace optimo_controller::state_machines {
namespace state_util = wbc::state_util;

CartesianReferenceState::CartesianReferenceState(wbc::StateId id,
                                                 const std::string& name,
                                                 const wbc::StateContext& ctx)
    : State(id, name, ctx) {}

void CartesianReferenceState::Configure(const YAML::Node& node) {
  State::Configure(node);
  if (node["task_name"]) {
    task_name_ = node["task_name"].as<std::string>();
  }
  task_ = RequireTask<wbc::tasks::TaskSE3Equality>(
      task_name_, "CartesianReferenceState");

  if (node["dt"]) {
    throw std::invalid_argument(
        "CartesianReferenceState parameter 'dt' is not used; controller dt is "
        "passed by the FSM tick");
  }

  target_translation_ =
      state_util::ReadOptionalVector3(node, "target_pos",
                                      "CartesianReferenceState");
  if (!target_translation_) {
    target_translation_ = state_util::ReadOptionalVector3(
        node, "target_position", "CartesianReferenceState");
  }
  target_quat_ = state_util::ReadOptionalQuaternion(
      node, "target_quat", "CartesianReferenceState");
  if (!target_quat_) {
    target_quat_ = state_util::ReadOptionalQuaternion(
        node, "target_orientation", "CartesianReferenceState");
  }
  if (auto linear_velocity =
          state_util::ReadOptionalVector3(node, "linear_velocity",
                                          "CartesianReferenceState")) {
    linear_velocity_ = *linear_velocity;
  }
  if (auto angular_velocity =
          state_util::ReadOptionalVector3(node, "angular_velocity",
                                          "CartesianReferenceState")) {
    angular_velocity_ = *angular_velocity;
  }
}

void CartesianReferenceState::OnEnter() {
  CaptureCurrentPose();
  ApplyConfiguredPoseTarget();
  ApplyReference();
}

void CartesianReferenceState::OnUpdate() {
  if (linear_velocity_.squaredNorm() > 0.0 ||
      angular_velocity_.squaredNorm() > 0.0) {
    target_pose_.translation().noalias() += dt() * linear_velocity_;
    const double angle = angular_velocity_.norm() * dt();
    if (angle > 1e-12) {
      target_pose_.rotation() =
          target_pose_.rotation() *
          Eigen::AngleAxisd(angle, angular_velocity_.normalized())
              .toRotationMatrix();
    }
  }
  ApplyReference();
}

void CartesianReferenceState::OnExit() {}

void CartesianReferenceState::CaptureCurrentPose() {
  target_pose_ = robot_->framePosition(*data_, task_->frame_id());
}

void CartesianReferenceState::ApplyConfiguredPoseTarget() {
  if (target_translation_) {
    target_pose_.translation() = *target_translation_;
  }
  if (target_quat_) {
    target_pose_.rotation() = target_quat_->toRotationMatrix();
  }
}

void CartesianReferenceState::ApplyReference() {
  wbc::math::SE3ToVector(target_pose_, ref_.pos);
  ref_.vel.setZero();
  ref_.acc.setZero();
  ref_.vel.head<3>() = angular_velocity_;
  ref_.vel.tail<3>() = linear_velocity_;
  task_->setReference(ref_);
}

}  // namespace optimo_controller::state_machines
