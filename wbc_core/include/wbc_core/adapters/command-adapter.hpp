//
// Copyright (c) 2026
//
// Command adapter: convert WBMC solver outputs into actuator-facing commands.
//

#ifndef __wbc_adapters_command_adapter_hpp__
#define __wbc_adapters_command_adapter_hpp__

#include <Eigen/Core>

#include "wbc_core/controller/wbmc-solution.hpp"
#include "wbc_core/robots/robot-wrapper.hpp"

namespace wbc {

enum class CommandOutputMode {
  kTorqueWithIntegratedState,
  kTorqueOnly,
};

struct LowLevelCommand {
  Eigen::VectorXd tau;
  Eigen::VectorXd q;
  Eigen::VectorXd qdot;

  void Initialize(Eigen::Index na) {
    tau = Eigen::VectorXd::Zero(na);
    q = Eigen::VectorXd::Zero(na);
    qdot = Eigen::VectorXd::Zero(na);
  }
};

class CommandAdapter {
 public:
  void setOutputMode(CommandOutputMode mode) { mode_ = mode; }
  CommandOutputMode outputMode() const { return mode_; }

  /// Map WBMC solution to actuator-facing command.
  ///
  /// Contract:
  /// - Always consumes torque from `sol.tau` when dimensions are valid.
  /// - `kTorqueOnly` mode:
  ///   - only torque is updated,
  ///   - q/qdot helper channels are optional and left untouched.
  /// - `kTorqueWithIntegratedState` mode:
  ///   - requires full state dimensions (q, qdot, qddot_sol),
  ///   - updates helper q/qdot via explicit Euler integration.
  ///
  /// Return value:
  /// - `true`: output command is valid for the selected mode.
  /// - `false`: output is invalid for this cycle (caller should keep previous
  ///   command as safe hold behavior).
  bool fromSolution(const tsid::WBMCSolution& sol,
                    const tsid::robots::RobotWrapper& robot,
                    const Eigen::VectorXd& q,
                    const Eigen::VectorXd& qdot, double dt,
                    LowLevelCommand& cmd) const {
    const int na = robot.na();
    if (sol.tau.size() != na) {
      return false;
    }
    cmd.tau = sol.tau;

    if (mode_ == CommandOutputMode::kTorqueOnly) {
      // Helper q/qdot command is optional in torque-only mode.
      return true;
    }

    const int q_offset = robot.is_fixed_base() ? 0 : 7;
    const int v_offset = robot.is_fixed_base() ? 0 : 6;
    if (q.size() < q_offset + na || qdot.size() < v_offset + na ||
        sol.qddot_sol.size() < robot.nv()) {
      cmd.q.setZero(na);
      cmd.qdot.setZero(na);
      return false;
    }

    const auto q_act = q.segment(q_offset, na);
    const auto qdot_act = qdot.segment(v_offset, na);
    const auto qddot_act = sol.qddot_sol.tail(na);
    cmd.qdot = qdot_act + dt * qddot_act;
    cmd.q = q_act + dt * cmd.qdot;
    return true;
  }

 private:
  CommandOutputMode mode_{CommandOutputMode::kTorqueWithIntegratedState};
};

}  // namespace wbc

#endif  // ifndef __wbc_adapters_command_adapter_hpp__
