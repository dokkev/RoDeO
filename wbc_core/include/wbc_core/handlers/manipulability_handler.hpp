/**
 * @file wbc_core/include/wbc_core/handlers/manipulability_handler.hpp
 * @brief Log-manipulability gradient posture bias for singularity avoidance.
 *
 * Computes the gradient of log(manipulability) via central finite differences
 * and commands a bias velocity along the gradient when sigma_min drops below
 * a threshold.
 */
#pragma once

#include <Eigen/Dense>
#include <Eigen/SVD>

#include <pinocchio/multibody/data.hpp>
#include <wbc_core/robots/robot-wrapper.hpp>

namespace wbc {

class ManipulabilityHandler {
public:
  struct Config {
    double sigma_threshold{0.08};
    double fd_eps{1e-4};
    double sigma_eps{1e-6};
    double gain{0.15};
    double max_bias_qdot{0.2};
    bool use_full_jacobian{true};
    double characteristic_length{0.2};
  };

  ManipulabilityHandler() = default;

  /// Initialize handler.
  /// @param robot  TSID RobotWrapper providing model.
  /// @param data   Pinocchio data (shared with formulation).
  /// @param ee_frame_idx  Pinocchio frame index for the end-effector link.
  /// @param config  Tuning parameters.
  void Init(tsid::robots::RobotWrapper* robot,
            pinocchio::Data* data,
            int ee_frame_idx,
            const Config& config);

  /// Per-tick update. q_current is the current full configuration (nq).
  void Update(const Eigen::VectorXd& q_current);

  double sigma_min() const { return sigma_min_; }
  double logw()      const { return logw_; }
  bool   is_active() const { return is_active_; }

  const Eigen::VectorXd& grad_logw()  const { return grad_logw_; }
  const Eigen::VectorXd& bias_qdot()  const { return bias_qdot_; }
  const Eigen::VectorXd& avoidance_velocity() const { return bias_qdot_; }

private:
  /// Compute task Jacobian at q_full into J_active_.
  void ComputeTaskJacobian(const Eigen::VectorXd& q_full);
  std::pair<double, double> ComputeMetrics() const;
  static void ClampEach(Eigen::VectorXd& x, double abs_limit);

  tsid::robots::RobotWrapper* robot_{nullptr};
  pinocchio::Data* data_{nullptr};
  pinocchio::Data fd_data_;  // separate Data for FD evaluations
  int ee_frame_idx_{-1};
  Config config_;

  int na_{0};
  int nv_{0};
  int nq_{0};
  bool is_fixed_base_{true};

  double sigma_min_{0.0};
  double logw_{0.0};
  bool   is_active_{false};

  Eigen::VectorXd grad_logw_;
  Eigen::VectorXd bias_qdot_;
  pinocchio::Data::Matrix6x jac_buf_;

  // Pre-allocated scratch buffers (RT-4)
  Eigen::VectorXd v_zero_;         ///< zero velocity for FK (size nv)
  Eigen::VectorXd q_scratch_;      ///< FD perturbation scratch (size nq)
  Eigen::MatrixXd J_active_;       ///< active-joint Jacobian (6 or 3 x na)
  mutable Eigen::JacobiSVD<Eigen::MatrixXd> svd_;  ///< SVD scratch
};

}  // namespace wbc
