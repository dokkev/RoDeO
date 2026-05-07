//
// Copyright (c) 2026
//
// Cartesian teleop state: velocity-servo + absolute pose tracking with
// singularity avoidance and watchdog timeout.
//

#ifndef WBC_CORE_ARCHITECTURE_STATES_CARTESIAN_TELEOP_STATE_HPP_
#define WBC_CORE_ARCHITECTURE_STATES_CARTESIAN_TELEOP_STATE_HPP_

#include "wbc_core/architecture/state_machine.hpp"
#include "wbc_core/handlers/cartesian_velocity_teleop_handler.hpp"
#include "wbc_core/handlers/manipulability_handler.hpp"
#include "wbc_core/utils/watchdog.hpp"

namespace wbc {

class CartesianTeleopState : public StateMachine {
 public:
  using StateMachine::StateMachine;

  void SetParameters(const YAML::Node& node) override;
  void FirstVisit() override;
  void OneStep() override;
  void LastVisit() override;
  void SetExternalInput(const TaskInput& input) override;

  bool EndOfState() const override { return false; }

  /// RT-safe typed dispatch — call before ControlArchitecture::Update().
  void UpdateCommand(const Eigen::Vector3d& xdot,
                     const Eigen::Vector3d& wdot,
                     int64_t vel_ts_ns,
                     const Eigen::Vector3d& x_des = Eigen::Vector3d::Zero(),
                     const Eigen::Quaterniond& quat_des =
                         Eigen::Quaterniond::Identity(),
                     int64_t pose_ts_ns = 0);

  /// Diagnostic getters.
  const Eigen::Vector3d& PosDesired() const { return pos_des_; }
  const Eigen::Quaterniond& QuatDesired() const { return quat_des_; }

 private:
  std::shared_ptr<tsid::tasks::TaskSE3Equality> ee_pos_task_;
  std::shared_ptr<tsid::tasks::TaskSE3Equality> ee_ori_task_;
  std::shared_ptr<tsid::tasks::TaskJointPosture> jpos_task_;

  CartesianVelocityTeleopHandler ee_handler_;
  ManipulabilityHandler manip_handler_;
  ManipulabilityHandler::Config manip_config_;
  double preview_time_{0.02};
  Watchdog watchdog_{0.2};

  int64_t prev_vel_ts_ns_{0};
  int64_t prev_pose_ts_ns_{0};
  int64_t last_vel_ts_ns_{0};
  int64_t last_pose_ts_ns_{0};

  Eigen::Vector3d pos_des_{Eigen::Vector3d::Zero()};
  Eigen::Quaterniond quat_des_{Eigen::Quaterniond::Identity()};
  Eigen::Vector3d pose_cmd_pos_{Eigen::Vector3d::Zero()};
  Eigen::Quaterniond pose_cmd_quat_{Eigen::Quaterniond::Identity()};

  // Pre-allocated buffers for hot path (avoid heap alloc per tick)
  void setSE3Ref(tsid::tasks::TaskSE3Equality& task,
                 const Eigen::Vector3d& pos,
                 const Eigen::Quaterniond& quat,
                 const Eigen::Vector3d& linear_vel,
                 const Eigen::Vector3d& angular_vel);

  tsid::trajectories::TrajectorySample se3_sample_{12, 6};
  Eigen::Matrix<double, 12, 1> se3_val_;
  Eigen::Matrix<double, 6, 1> se3_twist_;
  Eigen::Matrix<double, 6, 1> se3_zero_acc_{Eigen::Matrix<double, 6, 1>::Zero()};

  // Posture section pre-allocated buffers
  Eigen::VectorXd q_active_;
  Eigen::VectorXd q_des_jpos_;
  Eigen::VectorXd zero_acc_jpos_;
  tsid::trajectories::TrajectorySample jpos_sample_{0};
  bool jpos_buffers_initialized_{false};
};

}  // namespace wbc

#endif
