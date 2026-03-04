/**
 * @file optimo_controller/include/optimo_controller/manipulability_handler.hpp
 * @brief Manipulability-gradient singularity avoidance for null-space posture.
 *
 * Monitors the Yoshikawa manipulability index w = sqrt(det(J*J^T)) of the
 * end-effector Jacobian. When the minimum singular value σ_min drops below a
 * threshold, computes an avoidance velocity in the direction of the
 * manipulability gradient ∂w/∂q. The caller is responsible for applying
 * this velocity (e.g., via JointTeleopHandler::SetVelocity). The WBC
 * task-priority structure naturally projects it into the null space of
 * higher-priority end-effector tasks.
 *
 * The gradient is computed via forward finite difference using a separate
 * pinocchio::Data scratch object (pre-allocated in Init, no RT heap alloc).
 * Gradient recomputation is rate-limited to every gradient_interval ticks.
 *
 * References:
 *   - T. Yoshikawa, "Manipulability of Robotic Mechanisms,"
 *     Int. J. Robotics Research, vol. 4, no. 2, 1985.
 */
#pragma once

#include <memory>

#include <Eigen/Dense>
#include <pinocchio/multibody/fwd.hpp>

namespace wbc {

class PinocchioRobotSystem;

class ManipulabilityHandler {
public:
  struct Config {
    double step_size{0.5};         // [rad/s] max avoidance velocity magnitude
    double sigma_threshold{0.05};  // σ_min below this activates avoidance
    int gradient_interval{10};     // recompute gradient every N ticks
    double fd_epsilon{1e-4};       // finite difference step [rad]
  };

  ManipulabilityHandler() = default;
  ~ManipulabilityHandler();

  ManipulabilityHandler(const ManipulabilityHandler&) = delete;
  ManipulabilityHandler& operator=(const ManipulabilityHandler&) = delete;
  ManipulabilityHandler(ManipulabilityHandler&&) noexcept;
  ManipulabilityHandler& operator=(ManipulabilityHandler&&) noexcept;

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
  double min_singular_value() const { return sigma_min_; }
  bool is_active() const { return sigma_min_ < config_.sigma_threshold; }
  const Eigen::VectorXd& avoidance_velocity() const { return qdot_avoid_; }

private:
  double ComputeManipulability(const Eigen::VectorXd& q_full);
  void ComputeGradient();

  PinocchioRobotSystem* robot_{nullptr};
  int ee_frame_idx_{-1};
  int num_qdot_{0};
  int num_active_dof_{0};
  int num_float_dof_{0};
  Config config_;

  double w_{0.0};
  double sigma_min_{1.0};
  int tick_count_{0};

  std::unique_ptr<pinocchio::Data> data_scratch_;
  Eigen::MatrixXd J_;           // 6 x num_qdot
  Eigen::VectorXd q_scratch_;   // perturbed q (num_q)
  Eigen::VectorXd gradient_;    // num_active_dof
  Eigen::VectorXd qdot_avoid_;  // num_active_dof (scratch)
};

}  // namespace wbc
