/**
 * @file wbc_handlers/include/wbc_handlers/manipulability_handler.hpp
 * @brief Manipulability-gradient singularity avoidance for null-space posture.
 *
 * Monitors the Yoshikawa manipulability index w = sqrt(det(J*J^T)) of the
 * end-effector Jacobian. When w drops below a threshold, computes an
 * avoidance velocity in the direction of the manipulability gradient ∂w/∂q.
 * The caller is responsible for applying this velocity. The WBC
 * task-priority structure naturally projects it into the null space of
 * higher-priority end-effector tasks.
 *
 * The gradient is computed via forward finite difference using a separate
 * pinocchio::Data scratch object (pre-allocated in Init, no RT heap alloc).
 * Gradient computation is amortized: one DOF per tick (round-robin).
 *
 * References:
 *   - T. Yoshikawa, "Manipulability of Robotic Mechanisms,"
 *     Int. J. Robotics Research, vol. 4, no. 2, 1985.
 */
#pragma once

#include <Eigen/Dense>

namespace wbc {

class PinocchioRobotSystem;

class ManipulabilityHandler {
public:
  struct Config {
    double step_size{0.5};         // [rad/s] max avoidance velocity magnitude
    double w_threshold{0.01};      // w below this activates avoidance
    double fd_epsilon{1e-4};       // finite difference step [rad]
  };

  ManipulabilityHandler() = default;
  ~ManipulabilityHandler() = default;

  /**
   * @brief Initialize handler. Call once in FirstVisit (non-RT).
   * @param robot   Robot system providing model + current state.
   * @param ee_frame_idx  Pinocchio frame index for the end-effector link.
   * @param config  Tuning parameters.
   */
  void Init(PinocchioRobotSystem* robot, int ee_frame_idx, const Config& config);

  /**
   * @brief Per-tick update (RT). Computes avoidance velocity when near singularity.
   * @param dt  Control time step [s].
   */
  void Update(double dt);

  double manipulability() const { return w_; }
  bool is_active() const { return w_ < config_.w_threshold; }
  const Eigen::VectorXd& avoidance_velocity() const { return qdot_avoid_; }

private:
  PinocchioRobotSystem* robot_{nullptr};
  int ee_frame_idx_{-1};
  int num_active_dof_{0};
  int num_float_dof_{0};
  Config config_;

  double w_{0.0};
  int fd_current_dof_{0};  // round-robin index for amortized gradient

  Eigen::VectorXd q_scratch_;   // perturbed q (num_q)
  Eigen::VectorXd gradient_;    // num_active_dof
  Eigen::VectorXd qdot_avoid_;  // num_active_dof
};

}  // namespace wbc
