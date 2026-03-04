/**
 * @file draco_controller/src/state_machines/balance.cpp
 * @brief Balance/home state implementation for Draco.
 */
#include "draco_controller/state_machines/balance.hpp"

#include <cstdio>
#include <stdexcept>
#include <vector>

#include "wbc_fsm/state_factory.hpp"
#include "wbc_robot_system/pinocchio_robot_system.hpp"

namespace wbc {

DracoBalance::DracoBalance(StateId state_id, const std::string& state_name,
                           const StateMachineConfig& context)
    : StateMachine(state_id, state_name, context),
      target_jpos_(Eigen::VectorXd::Zero(
          context.robot != nullptr ? context.robot->GetJointPos().size() : 0)) {
  zeros_joint_.setZero(target_jpos_.size());
}

void DracoBalance::FirstVisit() {
  fprintf(stderr, "[DracoBalance] Enter '%s'.\n", name().c_str());

  jpos_task_ = GetMotionTask<JointTask>("jpos_task");
  com_task_  = GetMotionTask<ComTask>("com_task");

  if (jpos_task_ == nullptr && com_task_ == nullptr) {
    throw std::runtime_error(
        "[DracoBalance] Requires at least one of {jpos_task, com_task}.");
  }

  if (jpos_task_ != nullptr) {
    const Eigen::VectorXd q_curr = robot_->GetJointPos();
    if (!has_target_jpos_) {
      target_jpos_ = q_curr;
    }
    if (target_jpos_.size() != q_curr.size()) {
      target_jpos_ = q_curr;
    }
    zeros_joint_.setZero(target_jpos_.size());

    if (!joint_traj_handler_.SetTrajectory(q_curr, target_jpos_, duration_)) {
      fprintf(stderr, "[DracoBalance] Failed to set joint trajectory — "
                      "duration=%.3f, fallback to direct hold.\n", duration_);
    }
  }

  if (com_task_ != nullptr) {
    const Eigen::Vector3d com_curr = robot_->GetComPosition();
    if (!has_target_com_) {
      target_com_ = com_curr;
    }

    if (!com_traj_handler_.SetTrajectory(com_curr, target_com_, duration_)) {
      fprintf(stderr, "[DracoBalance] Failed to set CoM trajectory — "
                      "duration=%.3f, fallback to direct hold.\n", duration_);
    }
  }
}

void DracoBalance::OneStep() {
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

void DracoBalance::LastVisit() {
  fprintf(stderr, "[DracoBalance] Exit '%s'.\n", name().c_str());
}

bool DracoBalance::EndOfState() { return StateMachine::EndOfState(); }

void DracoBalance::SetParameters(const YAML::Node& node) {
  SetCommonParameters(node);

  const YAML::Node params = param::ResolveParamsNode(node);
  if (!params) {
    return;
  }

  if (params["target_jpos"]) {
    const std::vector<double> vec = params["target_jpos"].as<std::vector<double>>();
    if (target_jpos_.size() == 0 ||
        static_cast<int>(vec.size()) == target_jpos_.size()) {
      target_jpos_ =
          Eigen::Map<const Eigen::VectorXd>(vec.data(), static_cast<int>(vec.size()));
      has_target_jpos_ = true;
    } else {
      fprintf(stderr, "[DracoBalance] Ignore target_jpos: dimension mismatch "
                      "(expected=%d, got=%zu).\n",
              static_cast<int>(target_jpos_.size()), vec.size());
    }
  }

  if (params["target_com"]) {
    const std::vector<double> vec = params["target_com"].as<std::vector<double>>();
    if (vec.size() == 3) {
      target_com_     = Eigen::Vector3d(vec[0], vec[1], vec[2]);
      has_target_com_ = true;
    } else {
      fprintf(stderr, "[DracoBalance] Ignore target_com: dimension mismatch "
                      "(expected=3, got=%zu).\n", vec.size());
    }
  }
}

WBC_REGISTER_STATE(
    "draco_balance",
    [](StateId id, const std::string& state_name,
       const StateMachineConfig& context) -> std::unique_ptr<StateMachine> {
      return std::make_unique<DracoBalance>(id, state_name, context);
    });

WBC_REGISTER_STATE(
    "draco_home",
    [](StateId id, const std::string& state_name,
       const StateMachineConfig& context) -> std::unique_ptr<StateMachine> {
      return std::make_unique<DracoBalance>(id, state_name, context);
    });

}  // namespace wbc
