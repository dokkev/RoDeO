/**
 * @file wbc_core/wbc_formulation/include/wbc_formulation/contact.hpp
 * @brief Doxygen documentation for contact module.
 */
#pragma once

#include <string>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include "wbc_formulation/constraint.hpp"

namespace wbc {

constexpr double kContactMinNormalForceZ = 0.001;

/**
 * @brief Common contact parameters.
 */
struct ContactConfig {
  double mu{0.0};
  Eigen::VectorXd kp;
  Eigen::VectorXd kd;
  double max_fz{kContactMinNormalForceZ};
};

/**
 * @brief Surface-contact parameters.
 */
struct SurfaceContactConfig : public ContactConfig {
  double foot_half_length{0.0};
  double foot_half_width{0.0};
};

/**
 * @brief Abstract contact interface for WBC constraints.
 */
class Contact : public Constraint {
public:
  /**
   * @brief Construct a contact object.
   * @param robot Robot model backend.
   * @param dim Contact wrench/constraint dimension.
   * @param target_link_idx Target frame index for this contact.
   * @param mu Friction coefficient.
   */
  Contact(PinocchioRobotSystem* robot, int dim, int target_link_idx, double mu);
  virtual ~Contact() = default;

  /** @brief Update friction/cone inequality matrices. */
  virtual void UpdateConeConstraint() = 0;
  void UpdateConstraint() final { UpdateConeConstraint(); }
  /** @brief Update operational-space acceleration command. */
  virtual void UpdateOpCommand() = 0;

  /** @brief Set contact parameters. */
  virtual void SetParameters(const ContactConfig& config) = 0;

  void SetMaxFz(double rf_z_max) { rf_z_max_ = rf_z_max; cone_dirty_ = true; }
  void SetDesiredPos(const Eigen::Vector3d& pos) { des_pos_ = pos; }
  void SetDesiredOri(const Eigen::Quaterniond& quat) { des_quat_ = quat; }

  double MaxFz() const { return rf_z_max_; }
  double Mu() const { return mu_; }

  const Eigen::MatrixXd& UfMatrix() const { return constraint_matrix_; }
  const Eigen::VectorXd& UfVector() const { return constraint_vector_; }
  const Eigen::VectorXd& OpCommand() const { return op_cmd_; }

protected:
  double mu_;
  double rf_z_max_;
  bool cone_dirty_{true};

  Eigen::Vector3d des_pos_;
  Eigen::Quaterniond des_quat_;
  Eigen::VectorXd kp_;
  Eigen::VectorXd kd_;
  Eigen::VectorXd op_cmd_;
};

} // namespace wbc
