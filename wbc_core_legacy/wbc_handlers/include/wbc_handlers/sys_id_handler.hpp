/**
 * @file wbc_core/wbc_handlers/include/wbc_handlers/sys_id_handler.hpp
 * @brief SysID helper wrapper used by controller-level SysID FSM states.
 */
#pragma once

#include <string>

#include <Eigen/Dense>

#include "residual_compensator/sysid.hpp"

namespace wbc {

struct SysIdHandlerConfig {
  SysIDConfig sysid;
  bool abort_on_safety{true};
  bool hold_on_abort{true};
};

class SysIdHandler {
public:
  SysIdHandler() = default;

  void Configure(const SysIdHandlerConfig& config);

  void Initialize(int num_active,
                  const Eigen::Ref<const Eigen::VectorXd>& q_hold,
                  double start_time_sec);

  void Stop();

  void Update(double time_sec,
              double dt_sec,
              const Eigen::Ref<const Eigen::VectorXd>& q_meas,
              const Eigen::Ref<const Eigen::VectorXd>& qdot_meas,
              const Eigen::Ref<const Eigen::VectorXd>& tau_cmd,
              const Eigen::Ref<const Eigen::MatrixXd>& tau_limits,
              Eigen::Ref<Eigen::VectorXd> q_ref,
              Eigen::Ref<Eigen::VectorXd> qdot_ref,
              Eigen::Ref<Eigen::VectorXd> qddot_ref);

  bool IsActive() const { return sysid_.IsActive(); }
  bool IsFinished() const { return sysid_.IsFinished(); }
  bool IsAborted() const { return sysid_.IsAborted(); }
  SysIDPhase Phase() const { return sysid_.Phase(); }
  const std::string& LastReason() const { return sysid_.LastReason(); }

private:
  SysIdHandlerConfig config_;
  SysID sysid_;
};

}  // namespace wbc
