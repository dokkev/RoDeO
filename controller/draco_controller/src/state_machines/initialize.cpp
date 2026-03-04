/**
 * @file draco_controller/src/state_machines/initialize.cpp
 * @brief Draco initialize state implementation.
 */
#include "draco_controller/state_machines/initialize.hpp"

#include <cstdio>
#include <vector>

#include "wbc_fsm/state_factory.hpp"
#include "wbc_robot_system/pinocchio_robot_system.hpp"

namespace wbc {

DracoInitialize::DracoInitialize(StateId state_id,
                                 const std::string& state_name,
                                 const StateMachineConfig& context)
    : StateMachine(state_id, state_name, context),
      target_jpos_(Eigen::VectorXd::Zero(
          context.robot != nullptr ? context.robot->GetJointPos().size() : 0)) {
  zeros_joint_.setZero(target_jpos_.size());
}

void DracoInitialize::FirstVisit() {
  fprintf(stderr, "[DracoInitialize] Enter '%s'.\n", name().c_str());
  jpos_task_ = GetMotionTask<JointTask>("jpos_task");
  com_task_ = GetMotionTask<ComTask>("com_task");

  if (jpos_task_ == nullptr && com_task_ == nullptr) {
    throw std::runtime_error(
        "[DracoInitialize] Requires at least one of {jpos_task, com_task}.");
  }

  const Eigen::VectorXd q_init = robot_->GetJointPos();
  if (jpos_task_ != nullptr) {
    if (target_jpos_.size() != q_init.size()) {
      target_jpos_ = q_init;
    }

    if (!joint_traj_handler_.SetTrajectory(q_init, target_jpos_, duration_)) {
      throw std::runtime_error(
          "[DracoInitialize] Failed to set joint initialization trajectory.");
    }
  }

  if (com_task_ != nullptr) {
    const Eigen::Vector3d com_init = robot_->GetComPosition();
    if (!has_target_com_) {
      target_com_ = com_init;
    }
    if (!com_traj_handler_.SetTrajectory(com_init, target_com_, duration_)) {
      throw std::runtime_error(
          "[DracoInitialize] Failed to set CoM initialization trajectory.");
    }
  }
}

void DracoInitialize::OneStep() {
  if (jpos_task_ != nullptr) {
    if (!joint_traj_handler_.IsFinished()) {
      joint_traj_handler_.Update(current_time_, jpos_task_);
    } else {
      jpos_task_->UpdateDesired(target_jpos_, zeros_joint_, zeros_joint_);
    }
  }

  if (com_task_ != nullptr) {
    if (!com_traj_handler_.IsFinished()) {
      com_traj_handler_.Update(current_time_, com_task_);
    } else {
      com_task_->UpdateDesired(target_com_, Eigen::Vector3d::Zero(),
                               Eigen::Vector3d::Zero());
    }
  }
}

void DracoInitialize::LastVisit() {
  fprintf(stderr, "[DracoInitialize] Exit '%s'.\n", name().c_str());
}

bool DracoInitialize::EndOfState() {
  const bool joint_done = (jpos_task_ == nullptr) || joint_traj_handler_.IsFinished();
  const bool com_done = (com_task_ == nullptr) || com_traj_handler_.IsFinished();
  return joint_done && com_done && StateMachine::EndOfState();
}

void DracoInitialize::SetParameters(const YAML::Node& node) {
  SetCommonParameters(node);

  const YAML::Node params = param::ResolveParamsNode(node);
  if (!params) {
    return;
  }
  if (!params["target_jpos"] && !params["target_com"]) {
    return;
  }

  if (params["target_jpos"]) {
    const std::vector<double> vec = params["target_jpos"].as<std::vector<double>>();
    if (target_jpos_.size() == 0 ||
        static_cast<int>(vec.size()) == target_jpos_.size()) {
      target_jpos_ =
          Eigen::Map<const Eigen::VectorXd>(vec.data(), static_cast<int>(vec.size()));
    } else {
      fprintf(stderr, "[DracoInitialize] Ignore target_jpos: dimension mismatch "
                      "(expected=%d, got=%zu).\n",
              static_cast<int>(target_jpos_.size()), vec.size());
    }
  }

  if (params["target_com"]) {
    const std::vector<double> vec = params["target_com"].as<std::vector<double>>();
    if (vec.size() == 3) {
      target_com_ = Eigen::Vector3d(vec[0], vec[1], vec[2]);
      has_target_com_ = true;
    } else {
      fprintf(stderr, "[DracoInitialize] Ignore target_com: dimension mismatch "
                      "(expected=3, got=%zu).\n", vec.size());
    }
  }
}

WBC_REGISTER_STATE(
    "draco_initialize",
    [](StateId id, const std::string& state_name,
       const StateMachineConfig& context) -> std::unique_ptr<StateMachine> {
      return std::make_unique<DracoInitialize>(id, state_name, context);
    });

} // namespace wbc
