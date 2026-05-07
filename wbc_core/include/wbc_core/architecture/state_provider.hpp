//
// Copyright (c) 2026
//
// Shared runtime context for the WBMC control architecture.
//

#ifndef WBC_CORE_ARCHITECTURE_STATE_PROVIDER_HPP_
#define WBC_CORE_ARCHITECTURE_STATE_PROVIDER_HPP_

#include <Eigen/Dense>
#include <optional>
#include <string>

namespace wbc {

/// External input for teleop / high-level commands.
struct TaskInput {
  std::optional<Eigen::Vector3d> x_des;
  std::optional<Eigen::Quaterniond> quat_des;
  std::optional<Eigen::VectorXd> q_des;
  std::optional<Eigen::Vector3d> com_pos_des;
  std::optional<Eigen::VectorXd> wrench_des;
  std::optional<double> traj_duration;
  std::string reference_frame;
};

/// Robot joint state measurement.
struct RobotJointState {
  Eigen::VectorXd q;
  Eigen::VectorXd qdot;
  Eigen::VectorXd tau;

  void Reset(Eigen::Index dof) {
    q.setZero(dof);
    qdot.setZero(dof);
    tau.setZero(dof);
  }
};

/// Robot base state (floating-base only).
struct RobotBaseState {
  Eigen::Vector3d pos{Eigen::Vector3d::Zero()};
  Eigen::Quaterniond quat{Eigen::Quaterniond::Identity()};
  Eigen::Vector3d lin_vel{Eigen::Vector3d::Zero()};
  Eigen::Vector3d ang_vel{Eigen::Vector3d::Zero()};
};

/// Shared context bus — read/written by ControlArchitecture, read by states.
struct StateProvider {
  double servo_dt{0.001};
  double current_time{0.0};
  uint64_t count{0};

  int state{-1};
  int prev_state{-1};

  Eigen::VectorXd nominal_jpos;
  Eigen::VectorXd nominal_jvel;
  bool is_floating_base{false};
  RobotBaseState base_state;

  void Initialize(Eigen::Index nq, Eigen::Index nv) {
    nominal_jpos = Eigen::VectorXd::Zero(nq);
    nominal_jvel = Eigen::VectorXd::Zero(nv);
    count = 0;
    current_time = 0.0;
    state = -1;
    prev_state = -1;
  }

  void Initialize(Eigen::Index nv) {
    Initialize(nv, nv);
  }

  void Update(double t, double dt, const Eigen::VectorXd& q,
              const Eigen::VectorXd& qdot) {
    servo_dt = dt;
    current_time = t;
    count++;
    nominal_jpos = q;
    nominal_jvel = qdot;
    prev_state = state;
  }
};

}  // namespace wbc

#endif  // WBC_CORE_ARCHITECTURE_STATE_PROVIDER_HPP_
