/**
 * @file wbc_core/include/wbc_core/utils/actuator_interface.hpp
 * @brief Polymorphic actuator interface for WBC torque output.
 */
#pragma once

#include <Eigen/Dense>

namespace wbc {

/// Hardware-interface-local command used by actuator models.
///
/// Unlike `RobotCommand`, this struct may carry hardware-owned gain parameters
/// such as `kp` and `kd`.
struct ActuatorCommand {
  ActuatorCommand(const Eigen::Ref<const Eigen::VectorXd>& q_des_in,
                  const Eigen::Ref<const Eigen::VectorXd>& qdot_des_in,
                  const Eigen::Ref<const Eigen::VectorXd>& tau_ff_in,
                  const Eigen::Ref<const Eigen::VectorXd>& kp_in,
                  const Eigen::Ref<const Eigen::VectorXd>& kd_in,
                  const Eigen::Ref<const Eigen::VectorXd>& q_link_in,
                  const Eigen::Ref<const Eigen::VectorXd>& qdot_link_in,
                  double dt_in)
      : q_des(q_des_in),
        qdot_des(qdot_des_in),
        tau_ff(tau_ff_in),
        kp(kp_in),
        kd(kd_in),
        q_link(q_link_in),
        qdot_link(qdot_link_in),
        dt(dt_in) {}

  Eigen::Ref<const Eigen::VectorXd> q_des;
  Eigen::Ref<const Eigen::VectorXd> qdot_des;
  Eigen::Ref<const Eigen::VectorXd> tau_ff;
  Eigen::Ref<const Eigen::VectorXd> kp;
  Eigen::Ref<const Eigen::VectorXd> kd;
  Eigen::Ref<const Eigen::VectorXd> q_link;
  Eigen::Ref<const Eigen::VectorXd> qdot_link;
  double dt{0.001};
};

class ActuatorInterface {
 public:
  virtual ~ActuatorInterface() = default;
  virtual void Reset(const Eigen::VectorXd& q_link) = 0;
  virtual bool ProcessTorque(const ActuatorCommand& cmd,
                             Eigen::Ref<Eigen::VectorXd> tau_out) = 0;
};

/// Passthrough — tau_out = tau_ff. For real hardware.
class DirectActuator : public ActuatorInterface {
 public:
  void Reset(const Eigen::VectorXd& /*q_link*/) override {}
  bool ProcessTorque(const ActuatorCommand& cmd,
                     Eigen::Ref<Eigen::VectorXd> tau_out) override {
    if (tau_out.size() != cmd.tau_ff.size()) {
      return false;
    }
    tau_out = cmd.tau_ff;
    return true;
  }
};

/// Spring actuator — models motor+gear as position source, spring to link.
class SpringActuator : public ActuatorInterface {
 public:
  SpringActuator(const Eigen::VectorXd& stiffness,
                 const Eigen::VectorXd& damping)
      : k_(stiffness), d_(damping) {}

  void Reset(const Eigen::VectorXd& /*q_link*/) override {}

  bool ProcessTorque(const ActuatorCommand& cmd,
                     Eigen::Ref<Eigen::VectorXd> tau_out) override {
    const Eigen::Index n = k_.size();
    if (d_.size() != n || tau_out.size() != n || cmd.q_des.size() != n ||
        cmd.qdot_des.size() != n || cmd.tau_ff.size() != n ||
        cmd.kp.size() != n || cmd.kd.size() != n ||
        cmd.q_link.size() != n || cmd.qdot_link.size() != n) {
      return false;
    }
    tau_out = cmd.tau_ff;
    tau_out += k_.cwiseProduct(cmd.q_des - cmd.q_link);
    tau_out += d_.cwiseProduct(cmd.qdot_des - cmd.qdot_link);
    return tau_out.allFinite();
  }

 private:
  Eigen::VectorXd k_;
  Eigen::VectorXd d_;
};

}  // namespace wbc
