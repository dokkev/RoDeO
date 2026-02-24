#include "rpc_example/ExampleController.hpp"
#include "rpc_example/example_state_provider.hpp"

#include <stdexcept>

namespace wbc {

ExampleController::ExampleController(
    PinocchioRobotSystem* robot, const std::string& yaml_path,
    double control_dt)
    : engine_(
      BuildControlArchitecture(robot, yaml_path, control_dt,
                               std::make_unique<ExampleStateProvider>(control_dt),
                               std::make_unique<FSMHandler>())) {
  if (engine_ == nullptr) {
    throw std::runtime_error("[ExampleController] engine is null.");
  }
}

void ExampleController::Update(const Eigen::VectorXd& q,
                               const Eigen::VectorXd& qdot, double t,
                               double dt) {
  engine_->Update(q, qdot, t, dt);
}

void ExampleController::SetTeleopCommand(const TaskReference& command) {
  engine_->SetTeleopCommand(command);
}

void ExampleController::ClearTeleopCommand() {
  engine_->ClearTeleopCommand();
}

RobotCommand ExampleController::GetCommand() const {
  return engine_->GetCommand();
}

std::vector<std::pair<int, std::string>> ExampleController::GetStates() const {
  return engine_->GetStates();
}

void ExampleController::RequestState(int id) {
  engine_->RequestState(id);
}

bool ExampleController::RequestStateByName(const std::string& name) {
  return engine_->RequestStateByName(name);
}

int ExampleController::CurrentStateId() const {
  return engine_->CurrentStateId();
}

} // namespace wbc
