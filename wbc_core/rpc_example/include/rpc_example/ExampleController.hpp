#pragma once

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Dense>

#include "wbc_architecture/interface/control_architecture.hpp"

namespace wbc {

class PinocchioRobotSystem;

class ExampleController final {
public:
  ExampleController(PinocchioRobotSystem* robot, const std::string& yaml_path,
                    double control_dt = 0.001);

  ~ExampleController() = default;

  void Update(const Eigen::VectorXd& q, const Eigen::VectorXd& qdot, double t,
              double dt);
  void SetTeleopCommand(const TaskReference& command);
  void ClearTeleopCommand();

  [[nodiscard]] RobotCommand GetCommand() const;
  [[nodiscard]] std::vector<std::pair<int, std::string>> GetStates() const;
  void RequestState(int id);
  [[nodiscard]] bool RequestStateByName(const std::string& name);
  [[nodiscard]] int CurrentStateId() const;

private:
  std::unique_ptr<ControlArchitecture> engine_;
};

} // namespace wbc
