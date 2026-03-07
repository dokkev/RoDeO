/**
 * @file wbc_core/wbc_handlers/src/sys_id_handler.cpp
 * @brief SysID helper wrapper used by controller-level SysID FSM states.
 */
#include "wbc_handlers/sys_id_handler.hpp"

namespace wbc {

void SysIdHandler::Configure(const SysIdHandlerConfig& config) {
  config_ = config;
  sysid_.Configure(config_.sysid);
}

void SysIdHandler::Initialize(int num_active,
                              const Eigen::Ref<const Eigen::VectorXd>& q_hold,
                              double start_time_sec) {
  sysid_.Setup(num_active);
  sysid_.Configure(config_.sysid);
  sysid_.Reset(q_hold);
  sysid_.Start(start_time_sec);
}

void SysIdHandler::Stop() {
  sysid_.Stop();
}

void SysIdHandler::Update(double time_sec,
                          double dt_sec,
                          const Eigen::Ref<const Eigen::VectorXd>& q_meas,
                          const Eigen::Ref<const Eigen::VectorXd>& qdot_meas,
                          const Eigen::Ref<const Eigen::VectorXd>& tau_cmd,
                          const Eigen::Ref<const Eigen::MatrixXd>& tau_limits,
                          Eigen::Ref<Eigen::VectorXd> q_ref,
                          Eigen::Ref<Eigen::VectorXd> qdot_ref,
                          Eigen::Ref<Eigen::VectorXd> qddot_ref) {
  sysid_.Update(time_sec, dt_sec, q_ref, qdot_ref, qddot_ref);

  if (config_.abort_on_safety) {
    std::string reason;
    const bool safe =
        sysid_.CheckSafetyAndAbort(q_meas, qdot_meas, tau_cmd, tau_limits,
                                   &reason);
    (void)safe;
    (void)reason;
  }

  if (sysid_.IsAborted() && config_.hold_on_abort) {
    q_ref = q_meas;
    qdot_ref.setZero();
    qddot_ref.setZero();
  }
}

}  // namespace wbc
