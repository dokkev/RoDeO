#pragma once

#include <string>

#include "wbc_fsm/interface/state_machine.hpp"
#include "wbc_formulation/motion_task.hpp"
#include "wbc_trajectory/trajectory_handler.hpp"

namespace wbc {

class Initialize : public StateMachine {
public:
  Initialize(StateId state_id, const std::string& state_name,
             PinocchioRobotSystem* robot, TaskRegistry* task_reg,
             ConstraintRegistry* const_reg,
             StateProvider* state_provider = nullptr);
  ~Initialize() override = default;

  void FirstVisit() override;
  void OneStep() override;
  void LastVisit() override;
  bool EndOfState() override;

  void SetParameters(const YAML::Node& node) override;

private:
  JointTask* jpos_task_{nullptr};
  VectorTrajectoryHandler traj_handler_;

  Eigen::VectorXd target_jpos_;
};

} // namespace wbc
