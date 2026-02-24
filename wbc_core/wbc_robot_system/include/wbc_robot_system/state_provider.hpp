#pragma once
#include <Eigen/Dense>
#include <Eigen/Geometry>

namespace wbc {

class StateProvider {
public:
  explicit StateProvider(double dt = 0.001)
      : servo_dt_(dt),
        current_time_(0.0),
        count_(0),
        state_(0),
        prev_state_(0),
        b_contact_(false),
        foot_step_index_(0),
        base_pos_(Eigen::Vector3d::Zero()),
        base_ori_(Eigen::Quaterniond::Identity()) {}
  virtual ~StateProvider() = default;

  // ---------------------------------------------------------------------------
  // 1. Common System Info
  // ---------------------------------------------------------------------------
  double servo_dt_;
  double current_time_;
  int count_; // Control tick count

  // ---------------------------------------------------------------------------
  // 2. Common Robot State
  // ---------------------------------------------------------------------------
  Eigen::VectorXd q_; 
  Eigen::VectorXd qdot_;
  Eigen::VectorXd nominal_jpos_; // 초기 자세 혹은 공칭 자세

  // ---------------------------------------------------------------------------
  // 3. FSM State Management
  // ---------------------------------------------------------------------------
  int state_;       // Current State ID
  int prev_state_;  // Previous State ID
  bool b_first_visit_ = true;

  // ---------------------------------------------------------------------------
  // 4. Shared Runtime Flags
  // ---------------------------------------------------------------------------
  bool b_contact_;
  int foot_step_index_;

  // ---------------------------------------------------------------------------
  // 5. Shared Estimated Base State
  // ---------------------------------------------------------------------------
  Eigen::Vector3d base_pos_;
  Eigen::Quaterniond base_ori_;
  Eigen::Matrix3d rot_world_local_{Eigen::Matrix3d::Identity()};
};

} // namespace wbc
