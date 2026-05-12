//
// Copyright (c) 2026
//
// Command adapter: convert IDHQP solver outputs into actuator-facing commands.
//

#ifndef __wbc_adapters_command_adapter_hpp__
#define __wbc_adapters_command_adapter_hpp__

#include <Eigen/Core>

#include "wbc_core/controller/id-solution.hpp"
#include "wbc_core/robots/robot-system.hpp"

namespace wbc {

struct LowLevelCommand {
  Eigen::VectorXd tau;
  Eigen::VectorXd q;
  Eigen::VectorXd qdot;
  Eigen::VectorXd kp;
  Eigen::VectorXd kd;

  void Initialize(Eigen::Index na) {
    tau = Eigen::VectorXd::Zero(na);
    q = Eigen::VectorXd::Zero(na);
    qdot = Eigen::VectorXd::Zero(na);
    kp = Eigen::VectorXd::Zero(na);
    kd = Eigen::VectorXd::Zero(na);
  }
};

class CommandAdapter {
 public:
  /// Map IDHQP solution state to a complete actuator-facing command payload.
  ///
  /// Contract:
  /// - Always consumes torque from `sol.tau_cmd`.
  /// - Always consumes integrated q/qdot helper channels from `sol.q_cmd` and
  ///   `sol.qdot_cmd`.
  /// - Leaves hardware gain channels (`kp`, `kd`) untouched. Those are hardware
  ///   output policy, not IDHQP solution data.
  ///
  /// Return value:
  /// - `true`: output command is valid.
  /// - `false`: output is invalid for this cycle (caller should keep previous
  ///   command as safe hold behavior).
  bool fromSolution(const wbc::IDSolution& sol,
                    const wbc::robots::RobotSystem& robot,
                    LowLevelCommand& cmd) const {
    const int na = robot.na();
    if (cmd.tau.size() != na || sol.tau_cmd.size() != na ||
        !sol.tau_cmd.allFinite()) {
      return false;
    }

    const int q_offset = robot.is_fixed_base() ? 0 : 7;
    const int v_offset = robot.is_fixed_base() ? 0 : 6;
    if (cmd.q.size() != na || cmd.qdot.size() != na ||
        sol.q_cmd.size() < q_offset + na ||
        sol.qdot_cmd.size() < v_offset + na ||
        !sol.q_cmd.segment(q_offset, na).allFinite() ||
        !sol.qdot_cmd.segment(v_offset, na).allFinite()) {
      return false;
    }

    cmd.tau = sol.tau_cmd;
    cmd.q = sol.q_cmd.segment(q_offset, na);
    cmd.qdot = sol.qdot_cmd.segment(v_offset, na);
    return true;
  }
};

}  // namespace wbc

#endif  // ifndef __wbc_adapters_command_adapter_hpp__
