/**
 * @file wbc_handlers/include/wbc_handlers/manipulability_handler.hpp
 * @brief Log-manipulability gradient posture bias for singularity avoidance.
 *
 * Computes the gradient of log(manipulability) via central finite differences
 * and commands a bias velocity along the gradient when sigma_min drops below
 * a threshold. The bias is injected as a low-priority posture velocity in the
 * weighted-QP IK, steering joints away from singular configurations.
 *
 * ## Jacobian convention (FillLinkJacobian output)
 * The robot wrapper reorders Pinocchio's [linear; angular] rows to
 * [angular(0:3); linear(3:6)], so:
 *   - topRows(3)    = angular velocity Jacobian  [rad/s per joint rad/s]
 *   - bottomRows(3) = linear  velocity Jacobian  [m/s   per joint rad/s]
 *
 * ## dt note
 * Update() does not integrate in time — it computes a bias *velocity*
 * (bias_qdot). The caller is responsible for integrating:
 *   q_des += bias_qdot * dt
 */
#pragma once

#include <Eigen/Dense>
#include <Eigen/SVD>

namespace wbc {

class PinocchioRobotSystem;

class ManipulabilityHandler {
public:
  struct Config {
    /// Activate gradient bias when σ_min < sigma_threshold.
    double sigma_threshold{0.08};

    /// Finite-difference step for central-difference gradient [rad].
    double fd_eps{1e-4};

    /// Floor to avoid log(0) in singular configurations.
    double sigma_eps{1e-6};

    /// Gain scaling the normalized gradient bias velocity [rad/s].
    double gain{0.15};

    /// Per-joint clamp on bias velocity magnitude [rad/s].
    double max_bias_qdot{0.2};

    /// If true, use full 6D Jacobian; if false, use linear-only (3D).
    bool use_full_jacobian{true};

    /// Angular rows of the Jacobian are scaled by this length [m] to reduce
    /// unit mismatch between angular [rad/s] and linear [m/s] contributions.
    double characteristic_length{0.2};
  };

  ManipulabilityHandler() = default;
  ~ManipulabilityHandler() = default;

  /**
   * @brief Initialize handler. Call once in FirstVisit (non-RT).
   * @param robot         Robot system providing model and current state.
   * @param ee_frame_idx  Pinocchio frame index for the end-effector link.
   * @param config        Tuning parameters.
   */
  void Init(PinocchioRobotSystem* robot, int ee_frame_idx, const Config& config);

  /**
   * @brief Per-tick update (RT). Computes bias_qdot when near singularity.
   *
   * Performs 2*n_active FD Jacobian evaluations per tick (each calls
   * UpdateRobotModel). At 7-DOF this is ~14 FK passes (~50 µs typical).
   *
   * Assumes the robot model is already synced to the current tick's state
   * (i.e., ControlArchitecture::Update() has already called UpdateRobotModel).
   *
   * @param dt  Control time step [s] — unused; bias is a velocity, not a step.
   */
  void Update(double dt);

  double sigma_min() const { return sigma_min_; }
  double logw()      const { return logw_; }
  bool   is_active() const { return is_active_; }

  const Eigen::VectorXd& grad_logw()  const { return grad_logw_; }
  const Eigen::VectorXd& bias_qdot()  const { return bias_qdot_; }

  /// Alias for backward compatibility with callers using avoidance_velocity().
  const Eigen::VectorXd& avoidance_velocity() const { return bias_qdot_; }

private:
  /// Task Jacobian at the robot's current internal state.
  Eigen::MatrixXd ComputeCurrentTaskJacobian() const;

  /// Task Jacobian after temporarily perturbing active joint positions to
  /// q_active, then restoring the original robot model state.
  Eigen::MatrixXd ComputeTaskJacobianAtQ(const Eigen::VectorXd& q_active);

  /// σ_min and log(manipulability) from a task Jacobian J.
  std::pair<double, double> ComputeMetrics(const Eigen::MatrixXd& J) const;

  static void ClampEach(Eigen::VectorXd& x, double abs_limit);

private:
  PinocchioRobotSystem* robot_{nullptr};
  int ee_frame_idx_{-1};
  Config config_;

  int num_active_dof_{0};
  int num_float_dof_{0};

  // Joint position limits [n_active × 2]: col 0 = lower, col 1 = upper.
  // Cached at Init() for FD boundary clamping.
  Eigen::MatrixXd joint_pos_limits_;

  double sigma_min_{0.0};
  double logw_{0.0};
  bool   is_active_{false};

  Eigen::VectorXd grad_logw_;
  Eigen::VectorXd bias_qdot_;
};

}  // namespace wbc
