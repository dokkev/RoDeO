/**
 * @file controller/optimo_controller/include/optimo_controller/state_machines/initialize.hpp
 * @brief Optimo initialization posture state.
 */
#pragma once

#include <string>

#include "wbc_formulation/motion_task.hpp"
#include "wbc_fsm/interface/state_machine.hpp"
#include "wbc_handlers/trajectory_handler.hpp"

namespace wbc {

/**
 * @brief Drives Optimo to a YAML-configured initial posture via min-jerk trajectory.
 *
 * YAML params (under `params:`):
 *   - `duration`:     trajectory duration in seconds
 *   - `target_jpos`:  target joint positions (must match robot DOF)
 *
 * Registration key: "initialize"
 */
class Initialize : public StateMachine {
public:
  Initialize(StateId state_id, const std::string& state_name,
             const StateMachineConfig& context);
  ~Initialize() override = default;

  void SetParameters(const YAML::Node& node) override;
  void FirstVisit() override;
  void OneStep() override;
  void LastVisit() override;
  bool EndOfState() override;

private:
  JointTask*              jpos_task_{nullptr};
  JointTrajectoryHandler  traj_;
  Eigen::VectorXd         q_curr_;
  Eigen::VectorXd         q_des_;
  Eigen::VectorXd         zeros_;
};

} // namespace wbc
