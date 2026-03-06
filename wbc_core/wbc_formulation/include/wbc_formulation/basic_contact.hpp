/**
 * @file wbc_core/wbc_formulation/include/wbc_formulation/basic_contact.hpp
 * @brief Doxygen documentation for basic_contact module.
 */
#pragma once

#include "wbc_formulation/contact.hpp"
#include "wbc_robot_system/pinocchio_robot_system.hpp"

namespace wbc {
/**
 * @brief Point contact model (3D force).
 */
class PointContact : public Contact {
public:
  PointContact(PinocchioRobotSystem* robot, int target_link_idx, double mu);
  ~PointContact() override = default;

  void UpdateJacobian() override;
  void UpdateJacobianDotQdot() override;
  void UpdateConeConstraint() override;
  void UpdateOpCommand() override;
  void SetParameters(const ContactConfig& config) override;

private:
  // Pre-allocated 6xN scratch used by UpdateJacobian to avoid per-tick
  // heap allocation from GetLinkBodyJacobian() returning by value.
  Eigen::MatrixXd full_jac_scratch_;
};

/**
 * @brief Surface contact model (6D wrench with CoP/yaw constraints).
 */
class SurfaceContact : public Contact {
public:
  SurfaceContact(PinocchioRobotSystem* robot, int target_link_idx, double mu,
                 double foot_half_length, double foot_half_width);
  ~SurfaceContact() override = default;

  void UpdateJacobian() override;
  void UpdateJacobianDotQdot() override;
  void UpdateConeConstraint() override;
  void UpdateOpCommand() override;
  void SetParameters(const ContactConfig& config) override;
  void SetParameters(const SurfaceContactConfig& config);

  /** @brief Set contact half-length/half-width. */
  void SetFootHalfSize(double foot_half_length, double foot_half_width) {
    foot_half_length_ = foot_half_length;
    foot_half_width_ = foot_half_width;
    cone_dirty_ = true;
  }

private:
  double foot_half_length_;
  double foot_half_width_;
};

} // namespace wbc
