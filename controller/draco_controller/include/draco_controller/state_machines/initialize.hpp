/**
 * @file draco_controller/include/draco_controller/state_machines/initialize.hpp
 * @brief Draco initialize state with joint+CoM trajectories.
 */
#pragma once

#include <string>

#include "wbc_fsm/interface/state_machine.hpp"
#include "wbc_formulation/motion_task.hpp"
#include "wbc_trajectory/trajectory_handler.hpp"

namespace wbc {

/**
 * @brief Initialize state: trajectory from current to target joint+CoM pose.
 */
class DracoInitialize : public StateMachine {
public:
  DracoInitialize(StateId state_id, const std::string& state_name,
                  const StateMachineConfig& context);
  ~DracoInitialize() override = default;

  void FirstVisit() override;
  void OneStep() override;
  void LastVisit() override;
  bool EndOfState() override;
  void SetParameters(const YAML::Node& node) override;

private:
  JointTask* jpos_task_{nullptr};
  ComTask* com_task_{nullptr};
  JointTrajectoryHandler joint_traj_handler_;
  JointTrajectoryHandler com_traj_handler_;
  Eigen::VectorXd target_jpos_;
  Eigen::VectorXd zeros_joint_;
  Eigen::Vector3d target_com_{Eigen::Vector3d::Zero()};
  bool has_target_com_{false};
};

} // namespace wbc
