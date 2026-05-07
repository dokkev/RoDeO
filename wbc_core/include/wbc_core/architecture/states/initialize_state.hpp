//
// Copyright (c) 2026
//
// Initialize state: ramp joint positions from current to target over duration.
//

#ifndef WBC_CORE_ARCHITECTURE_STATES_INITIALIZE_STATE_HPP_
#define WBC_CORE_ARCHITECTURE_STATES_INITIALIZE_STATE_HPP_

#include "wbc_core/architecture/state_machine.hpp"
#include <wbc_core/trajectories/trajectory-euclidian.hpp>

namespace wbc {

class InitializeState : public StateMachine {
 public:
  using StateMachine::StateMachine;

  void SetParameters(const YAML::Node& node) override;
  void FirstVisit() override;
  void OneStep() override;
  void LastVisit() override;

 private:
  std::shared_ptr<tsid::tasks::TaskJointPosture> jpos_task_;
  Eigen::VectorXd q_start_;
  Eigen::VectorXd q_target_;
  Eigen::VectorXd zeros_;
  Eigen::VectorXd q_des_;
  tsid::trajectories::TrajectorySample sample_{0};
  bool sample_initialized_{false};
};

}  // namespace wbc

#endif
