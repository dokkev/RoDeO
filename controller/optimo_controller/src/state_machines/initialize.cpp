#include "optimo_controller/state_machines/initialize.hpp"

#include <iostream>
#include <vector>

#include "wbc_fsm/state_factory.hpp"
#include "wbc_robot_system/pinocchio_robot_system.hpp"

namespace wbc {

Initialize::Initialize(StateId state_id, const std::string& state_name,
                       PinocchioRobotSystem* robot, TaskRegistry* task_reg,
                       ConstraintRegistry* const_reg,
                       StateProvider* state_provider)
    : StateMachine(state_id, state_name, robot, task_reg, const_reg,
                   state_provider),
      target_jpos_(
          Eigen::VectorXd::Zero(robot != nullptr ? robot->GetJointPos().size()
                                                 : 0)) {}

void Initialize::FirstVisit() {
  std::cout << "[Initialize] Enter state '" << name() << "'." << std::endl;

  jpos_task_ = RequireTaskAs<JointTask>("jpos_task");

  const Eigen::VectorXd q_init = robot_->GetJointPos();
  if (target_jpos_.size() != q_init.size()) {
    target_jpos_ = q_init;
  }

  if (!traj_handler_.SetTrajectory(q_init, target_jpos_, duration_)) {
    throw std::runtime_error(
        "[Initialize] Failed to configure joint initialization trajectory.");
  }
}

void Initialize::OneStep() {
  if (jpos_task_ == nullptr) {
    return;
  }

  traj_handler_.Update(current_time_, jpos_task_);
}

void Initialize::LastVisit() {
  std::cout << "[Initialize] Exit state '" << name() << "'." << std::endl;
}

bool Initialize::EndOfState() {
  return traj_handler_.IsFinished() && StateMachine::EndOfState();
}

void Initialize::SetParameters(const YAML::Node& node) {
  StateMachine::SetParameters(node);

  const YAML::Node params = ResolveParamsNode(node);
  if (!params) {
    return;
  }
  if (params["target_jpos"]) {
    const std::vector<double> vec = params["target_jpos"].as<std::vector<double>>();
    if (target_jpos_.size() == 0 ||
        static_cast<int>(vec.size()) == target_jpos_.size()) {
      target_jpos_ =
          Eigen::Map<const Eigen::VectorXd>(vec.data(), static_cast<int>(vec.size()));
    } else {
      std::cerr << "[Initialize] Ignore target_jpos due to dimension mismatch. "
                << "expected=" << target_jpos_.size()
                << ", got=" << vec.size() << std::endl;
    }
  }
}

WBC_REGISTER_STATE(
    "initialize",
    [](StateId id, const std::string& state_name,
       const StateBuildContext& context) -> std::unique_ptr<StateMachine> {
      return std::make_unique<Initialize>(id, state_name, context.robot,
                                          context.task_registry,
                                          context.constraint_registry,
                                          context.state_provider);
    });

} // namespace wbc
