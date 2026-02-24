#include "rpc_example/state_machines/initialize.hpp"

#include <iostream>
#include <vector>

#include "wbc_fsm/state_factory.hpp"
#include "wbc_robot_system/pinocchio_robot_system.hpp"

namespace wbc {

ExampleInitialize::ExampleInitialize(StateId state_id,
                                     const std::string& state_name,
                                     PinocchioRobotSystem* robot,
                                     TaskRegistry* task_reg,
                                     ConstraintRegistry* const_reg,
                                     StateProvider* state_provider)
    : StateMachine(state_id, state_name, robot, task_reg, const_reg,
                   state_provider),
      target_jpos_(Eigen::VectorXd::Zero(
          robot != nullptr ? robot->GetJointPos().size() : 0)) {}

void ExampleInitialize::FirstVisit() {
  std::cout << "[ExampleInitialize] Enter '" << name() << "'." << std::endl;
  jpos_task_ = GetTask<JointTask>("jpos_task");
  com_task_ = GetTask<ComTask>("com_task");

  if (jpos_task_ == nullptr && com_task_ == nullptr) {
    throw std::runtime_error(
        "[ExampleInitialize] Requires at least one of {jpos_task, com_task}.");
  }

  const Eigen::VectorXd q_init = robot_->GetJointPos();
  if (jpos_task_ != nullptr) {
    if (target_jpos_.size() != q_init.size()) {
      target_jpos_ = q_init;
    }

    if (!joint_traj_handler_.SetTrajectory(q_init, target_jpos_, duration_)) {
      throw std::runtime_error(
          "[ExampleInitialize] Failed to set joint initialization trajectory.");
    }
  }

  if (com_task_ != nullptr) {
    const Eigen::Vector3d com_init = robot_->GetRobotComPos();
    if (!has_target_com_) {
      target_com_ = com_init;
    }
    if (!com_traj_handler_.SetTrajectory(com_init, target_com_, duration_)) {
      throw std::runtime_error(
          "[ExampleInitialize] Failed to set CoM initialization trajectory.");
    }
  }
}

void ExampleInitialize::OneStep() {
  if (jpos_task_ != nullptr) {
    if (!joint_traj_handler_.IsFinished()) {
      joint_traj_handler_.Update(current_time_, jpos_task_);
    } else {
      const int dim = jpos_task_->Dim();
      jpos_task_->UpdateDesired(target_jpos_, Eigen::VectorXd::Zero(dim),
                                Eigen::VectorXd::Zero(dim));
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

void ExampleInitialize::LastVisit() {
  std::cout << "[ExampleInitialize] Exit '" << name() << "'." << std::endl;
}

bool ExampleInitialize::EndOfState() {
  const bool joint_done = (jpos_task_ == nullptr) || joint_traj_handler_.IsFinished();
  const bool com_done = (com_task_ == nullptr) || com_traj_handler_.IsFinished();
  return joint_done && com_done && StateMachine::EndOfState();
}

void ExampleInitialize::SetParameters(const YAML::Node& node) {
  StateMachine::SetParameters(node);

  const YAML::Node params = ResolveParamsNode(node);
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
      std::cerr << "[ExampleInitialize] Ignore target_jpos due to dimension "
                   "mismatch. expected="
                << target_jpos_.size() << ", got=" << vec.size() << std::endl;
    }
  }

  if (params["target_com"]) {
    const std::vector<double> vec = params["target_com"].as<std::vector<double>>();
    if (vec.size() == 3) {
      target_com_ = Eigen::Vector3d(vec[0], vec[1], vec[2]);
      has_target_com_ = true;
    } else {
      std::cerr << "[ExampleInitialize] Ignore target_com due to dimension "
                   "mismatch. expected=3, got="
                << vec.size() << std::endl;
    }
  }
}

WBC_REGISTER_STATE(
    "initialize",
    [](StateId id, const std::string& state_name,
       const StateBuildContext& context) -> std::unique_ptr<StateMachine> {
      return std::make_unique<ExampleInitialize>(
          id, state_name, context.robot, context.task_registry,
          context.constraint_registry, context.state_provider);
    });

} // namespace wbc
