//
// Copyright (c) 2026
//
// Joint teleop state: velocity integration with limits, watchdog, and
// RT-safe typed dispatch via UpdateCommand().
//

#ifndef WBC_CORE_ARCHITECTURE_STATES_JOINT_TELEOP_STATE_HPP_
#define WBC_CORE_ARCHITECTURE_STATES_JOINT_TELEOP_STATE_HPP_

#include "wbc_core/architecture/state_machine.hpp"
#include "wbc_core/handlers/joint_teleop_handler.hpp"
#include "wbc_core/utils/watchdog.hpp"

namespace wbc {

class JointTeleopState : public StateMachine {
 public:
  using StateMachine::StateMachine;

  void SetParameters(const YAML::Node& node) override;
  void FirstVisit() override;
  void OneStep() override;
  void LastVisit() override;
  void SetExternalInput(const TaskInput& input) override;

  bool EndOfState() const override { return false; }

  /// RT-safe typed dispatch — call before ControlArchitecture::Update().
  void UpdateCommand(const Eigen::Ref<const Eigen::VectorXd>& qdot_cmd,
                     int64_t vel_ts_ns,
                     const Eigen::Ref<const Eigen::VectorXd>& q_des,
                     int64_t pos_ts_ns);

 private:
  std::shared_ptr<tsid::tasks::TaskJointPosture> jpos_task_;
  JointTeleopHandler handler_;
  Eigen::VectorXd vel_limit_;
  Eigen::VectorXd current_qdot_;
  Watchdog watchdog_{0.2};
  int64_t prev_vel_ts_ns_{0};
  int64_t prev_pos_ts_ns_{0};

  // Pre-allocated buffers for hot path
  Eigen::VectorXd zero_acc_;
  tsid::trajectories::TrajectorySample jpos_sample_{0};
  bool jpos_buffers_initialized_{false};
};

}  // namespace wbc

#endif
