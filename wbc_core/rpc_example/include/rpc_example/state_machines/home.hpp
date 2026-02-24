#pragma once

#include <string>

#include "wbc_fsm/interface/state_machine.hpp"
#include "wbc_formulation/motion_task.hpp"
#include "wbc_trajectory/trajectory_handler.hpp"

namespace wbc {

class ExampleHome : public StateMachine {
public:
  ExampleHome(StateId state_id, const std::string& state_name,
              PinocchioRobotSystem* robot, TaskRegistry* task_reg,
              ConstraintRegistry* const_reg,
              StateProvider* state_provider = nullptr);
  ~ExampleHome() override = default;

  void FirstVisit() override;
  void OneStep() override;
  void LastVisit() override;
  bool EndOfState() override;
  void SetParameters(const YAML::Node& node) override;

private:
  JointTask* jpos_task_{nullptr};
  ComTask* com_task_{nullptr};
  VectorTrajectoryHandler joint_traj_handler_;
  VectorTrajectoryHandler com_traj_handler_;
  Eigen::VectorXd target_jpos_;
  Eigen::Vector3d target_com_{Eigen::Vector3d::Zero()};
  bool has_target_com_{false};
};

} // namespace wbc
