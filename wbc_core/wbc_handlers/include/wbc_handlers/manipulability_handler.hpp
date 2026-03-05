/**
 * @file wbc_handlers/include/wbc_handlers/manipulability_handler.hpp
 * @brief SVD-based singularity avoidance for null-space posture.
 *
 * Monitors the minimum singular value of the end-effector Jacobian. When it
 * drops below a threshold, commands an avoidance velocity along the
 * corresponding right singular vector (the joint-space direction that is
 * "lost" at the singularity). The WBC task-priority structure naturally
 * projects this into the null space of higher-priority EE tasks.
 *
 * Unlike gradient-based approaches, the SVD method works correctly even at
 * exact singularity (where the manipulability gradient is zero due to the
 * cusp in √det(J·J^T)).
 */
#pragma once

#include <Eigen/Dense>
#include <Eigen/SVD>

namespace wbc {

class PinocchioRobotSystem;

class ManipulabilityHandler {
public:
  struct Config {
    double step_size{0.5};         // [rad/s] max avoidance velocity magnitude
    double w_threshold{0.01};      // σ_min below this activates avoidance
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
  double sigma_min() const { return sigma_min_; }
  bool is_active() const { return sigma_min_ < config_.w_threshold; }
  const Eigen::VectorXd& avoidance_velocity() const { return qdot_avoid_; }

private:
  PinocchioRobotSystem* robot_{nullptr};
  int ee_frame_idx_{-1};
  int num_active_dof_{0};
  int num_float_dof_{0};
  Config config_;

  double w_{0.0};
  double sigma_min_{0.0};

  Eigen::VectorXd qdot_avoid_;  // num_active_dof
};

}  // namespace wbc
