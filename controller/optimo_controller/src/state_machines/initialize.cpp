/**
 * @file controller/optimo_controller/src/state_machines/initialize.cpp
 * @brief Optimo initialization posture state.
 */
#include "optimo_controller/state_machines/initialize.hpp"

#include <cstdio>

#include "wbc_fsm/state_factory.hpp"

namespace wbc {

Initialize::Initialize(StateId state_id, const std::string& state_name,
                       const StateMachineConfig& context)
    : StateMachine(state_id, state_name, context) {}

void Initialize::SetParameters(const YAML::Node& node) {
  SetCommonParameters(node);
  SetMotionTask("jpos_task", jpos_task_);
  zeros_.setZero(jpos_task_->Dim());
  q_des_ = ParseVectorParam(node, "target_jpos");
}

void Initialize::FirstVisit() {
  q_curr_ = robot_->GetJointPos();
  if (q_des_.size() != q_curr_.size()) q_des_ = q_curr_;

  if (!traj_.SetTrajectory(q_curr_, q_des_, duration_)) {
    fprintf(stderr, "[Initialize] SetTrajectory failed (duration=%.3f); holding current pose.\n",
            duration_);
  }
}

void Initialize::OneStep() {
  if (!traj_.IsFinished()) {
    traj_.Update(current_time_, jpos_task_);
    return;
  }
  jpos_task_->UpdateDesired(q_des_, zeros_, zeros_);
}

void Initialize::LastVisit() {}

bool Initialize::EndOfState() {
  return traj_.IsFinished() && StateMachine::EndOfState();
}

WBC_REGISTER_STATE(
    "initialize",
    [](StateId id, const std::string& state_name,
       const StateMachineConfig& context) -> std::unique_ptr<StateMachine> {
      return std::make_unique<Initialize>(id, state_name, context);
    });

} // namespace wbc
