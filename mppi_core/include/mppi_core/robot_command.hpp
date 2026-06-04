// Copyright 2026
//
// Licensed under the Apache License, Version 2.0.

#pragma once

#include <cmath>
#include <stdexcept>

#include <Eigen/Core>

namespace mppi_core {

// Final robot/low-level-controller command packet. This is not the MPPI action:
// MPPI may sample delta_q_ref or delta_tau_ref internally, while the downstream
// controller interprets this packet as
//   tau_cmd = tau_ff + kp * (q_des - q) + kd * (qdot_des - qdot).
struct RobotCommand {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  bool valid{false};

  Eigen::VectorXd q_des;
  Eigen::VectorXd qdot_des;
  Eigen::VectorXd qddot_des;

  Eigen::VectorXd tau_ff;

  Eigen::VectorXd kp;
  Eigen::VectorXd kd;

  // Optional debug/introspection fields, not necessarily sent to hardware.
  Eigen::VectorXd delta_q_ref;
  Eigen::VectorXd delta_tau_ref;
  Eigen::VectorXd tau_raw;
  Eigen::VectorXd tau_limited;

  bool torque_saturated{false};
  bool torque_rate_limited{false};
  bool contact_required_but_missing{false};

  double stamp_sec{0.0};

  void Resize(int nq, int nv = -1) {
    if (nq < 0) {
      throw std::invalid_argument("RobotCommand::Resize: nq must be nonnegative");
    }
    if (nv < 0) {
      nv = nq;
    }
    if (nv < 0) {
      throw std::invalid_argument("RobotCommand::Resize: nv must be nonnegative");
    }

    q_des = Eigen::VectorXd::Zero(nq);
    qdot_des = Eigen::VectorXd::Zero(nv);
    qddot_des = Eigen::VectorXd::Zero(nv);
    tau_ff = Eigen::VectorXd::Zero(nv);
    kp = Eigen::VectorXd::Zero(nv);
    kd = Eigen::VectorXd::Zero(nv);

    delta_q_ref = Eigen::VectorXd::Zero(nv);
    delta_tau_ref = Eigen::VectorXd::Zero(nv);
    tau_raw = Eigen::VectorXd::Zero(nv);
    tau_limited = Eigen::VectorXd::Zero(nv);
  }

  bool HasValidDimensions() const {
    const Eigen::Index nq = q_des.size();
    const Eigen::Index nv = qdot_des.size();
    if (nq <= 0 || nv <= 0) {
      return false;
    }
    if (qddot_des.size() != nv || tau_ff.size() != nv || kp.size() != nv ||
        kd.size() != nv) {
      return false;
    }

    return OptionalVectorDimensionOk(delta_q_ref, nv) &&
           OptionalVectorDimensionOk(delta_tau_ref, nv) &&
           OptionalVectorDimensionOk(tau_raw, nv) &&
           OptionalVectorDimensionOk(tau_limited, nv);
  }

  bool AllFinite() const {
    return q_des.allFinite() && qdot_des.allFinite() && qddot_des.allFinite() &&
           tau_ff.allFinite() && kp.allFinite() && kd.allFinite() &&
           OptionalVectorFinite(delta_q_ref) &&
           OptionalVectorFinite(delta_tau_ref) && OptionalVectorFinite(tau_raw) &&
           OptionalVectorFinite(tau_limited) && std::isfinite(stamp_sec);
  }

  bool IsUsable() const { return valid && HasValidDimensions() && AllFinite(); }

 private:
  static bool OptionalVectorDimensionOk(const Eigen::VectorXd& value,
                                        Eigen::Index expected_size) {
    return value.size() == 0 || value.size() == expected_size;
  }

  static bool OptionalVectorFinite(const Eigen::VectorXd& value) {
    return value.size() == 0 || value.allFinite();
  }
};

inline RobotCommand MakeZeroHoldRobotCommand(
    const Eigen::VectorXd& q_current, const Eigen::VectorXd& qdot_current,
    const Eigen::VectorXd& kp_hold, const Eigen::VectorXd& kd_hold) {
  RobotCommand command;
  command.Resize(static_cast<int>(q_current.size()),
                 static_cast<int>(qdot_current.size()));
  command.q_des = q_current;
  command.qdot_des.setZero();
  command.kp = kp_hold;
  command.kd = kd_hold;
  command.valid = command.HasValidDimensions() && command.AllFinite() &&
                  qdot_current.allFinite();
  return command;
}

inline RobotCommand MakeInvalidRobotCommand(int nq, int nv = -1) {
  RobotCommand command;
  command.Resize(nq, nv);
  command.valid = false;
  return command;
}

}  // namespace mppi_core
