/**
 * @file wbc_util/include/wbc_util/actuator_interface.hpp
 * @brief Polymorphic actuator interface for WBC torque output.
 *
 * Decouples the controller from the physical actuator model:
 *   - SpringActuator: Spring compliance (motor+gear → spring → link)
 *   - DirectActuator: Passthrough (real hardware or direct-drive sim)
 */
#pragma once

#include <Eigen/Dense>

namespace wbc {

struct ActuatorCommand {
  Eigen::VectorXd q_des;    ///< Desired joint position from WBC
  Eigen::VectorXd qdot_des; ///< Desired joint velocity from WBC
  Eigen::VectorXd tau_ff;   ///< Feedforward torque from WBC
  Eigen::VectorXd q_link;   ///< Measured link-side position
  Eigen::VectorXd qdot_link;///< Measured link-side velocity
  double dt{0.001};
};

class ActuatorInterface {
public:
  virtual ~ActuatorInterface() = default;

  virtual void Reset(const Eigen::VectorXd& q_link) = 0;

  /// Compute the torque to send to the plant.
  virtual Eigen::VectorXd ProcessTorque(const ActuatorCommand& cmd) = 0;
};

/// Passthrough — tau_out = tau_ff. For real hardware.
class DirectActuator : public ActuatorInterface {
public:
  void Reset(const Eigen::VectorXd& /*q_link*/) override {}

  Eigen::VectorXd ProcessTorque(const ActuatorCommand& cmd) override {
    return cmd.tau_ff;
  }
};

/// Spring actuator — models motor+gear as position source, spring to link.
///   tau_out = k * (q_des - q_link) + d * (qdot_des - qdot_link) + tau_ff
class SpringActuator : public ActuatorInterface {
public:
  SpringActuator(const Eigen::VectorXd& stiffness,
                 const Eigen::VectorXd& damping)
    : k_(stiffness), d_(damping) {}

  void Reset(const Eigen::VectorXd& /*q_link*/) override {}

  Eigen::VectorXd ProcessTorque(const ActuatorCommand& cmd) override {
    return k_.cwiseProduct(cmd.q_des - cmd.q_link)
         + d_.cwiseProduct(cmd.qdot_des - cmd.qdot_link)
         + cmd.tau_ff;
  }

private:
  Eigen::VectorXd k_;  ///< Spring stiffness [Nm/rad]
  Eigen::VectorXd d_;  ///< Spring damping [Nm·s/rad]
};

} // namespace wbc
