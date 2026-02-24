#pragma once

#include "wbc_formulation/friction_cone.hpp"
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
  }

private:
  double foot_half_length_;
  double foot_half_width_;
};

} // namespace wbc
