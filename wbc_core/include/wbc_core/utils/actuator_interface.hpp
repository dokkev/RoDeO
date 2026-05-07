/**
 * @file wbc_core/include/wbc_core/utils/actuator_interface.hpp
 * @brief Polymorphic actuator interface for WBC torque output.
 */
#pragma once

#include <Eigen/Dense>

namespace wbc {

struct ActuatorCommand {
  Eigen::VectorXd q_des;
  Eigen::VectorXd qdot_des;
  Eigen::VectorXd tau_ff;
  Eigen::VectorXd q_link;
  Eigen::VectorXd qdot_link;
  double dt{0.001};
};

class ActuatorInterface {
public:
  virtual ~ActuatorInterface() = default;
  virtual void Reset(const Eigen::VectorXd& q_link) = 0;
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
  Eigen::VectorXd k_;
  Eigen::VectorXd d_;
};

} // namespace wbc
