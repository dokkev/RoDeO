/**
 * @file draco_controller/include/draco_controller/state_machines/balance.hpp
 * @brief Balance/home state: trajectory to a target pose, then hold.
 */
#pragma once

#include <string>

#include <Eigen/Dense>

#include "wbc_formulation/motion_task.hpp"
#include "wbc_fsm/interface/state_machine.hpp"
#include "wbc_trajectory/trajectory_handler.hpp"

namespace wbc {

/**
 * @brief FSM state that moves to a target joint+CoM pose and holds it.
 *
 * Registration keys: "draco_balance", "draco_home"
 */
class DracoBalance : public StateMachine {
public:
  DracoBalance(StateId state_id, const std::string& state_name,
               const StateMachineConfig& context);
  ~DracoBalance() override = default;

  void FirstVisit() override;
  void OneStep() override;
  void LastVisit() override;
  bool EndOfState() override;
  void SetParameters(const YAML::Node& node) override;

private:
  JointTask* jpos_task_{nullptr};
  ComTask*   com_task_{nullptr};

  JointTrajectoryHandler joint_traj_handler_;
  JointTrajectoryHandler com_traj_handler_;

  Eigen::VectorXd target_jpos_;
  Eigen::VectorXd zeros_joint_;
  bool            has_target_jpos_{false};

  Eigen::Vector3d target_com_{Eigen::Vector3d::Zero()};
  bool            has_target_com_{false};
};

}  // namespace wbc
