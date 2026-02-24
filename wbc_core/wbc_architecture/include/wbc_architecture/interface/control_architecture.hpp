#pragma once

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include "wbc_fsm/fsm_handler.hpp"
#include "wbc_robot_system/pinocchio_robot_system.hpp"
#include "wbc_robot_system/state_provider.hpp"
#include "wbc_formulation/wbc_formulation.hpp"
#include "wbc_solver/wbic.hpp"
#include "wbc_architecture/wbc_compiled_config.hpp"

namespace wbc {

struct RobotCommand {
  Eigen::VectorXd tau;
  Eigen::VectorXd qdot;
  Eigen::VectorXd q;
};

struct TaskReference {
  // Optional frame in which x_des/quat_des are expressed.
  // If unset, ControlArchitecture defaults to robot_model.base_frame.
  std::optional<std::string> reference_frame;
  std::optional<Eigen::Vector3d> x_des;
  std::optional<Eigen::Quaterniond> quat_des;
  std::optional<Eigen::VectorXd> joint_pos;
};

class ControlArchitecture {
public:
  virtual ~ControlArchitecture() = default;
  ControlArchitecture(PinocchioRobotSystem* robot,
                      std::shared_ptr<WbcConfigCompiler> compiled_config,
                      std::unique_ptr<StateProvider> state_provider = nullptr,
                      std::unique_ptr<FSMHandler> fsm_handler = nullptr,
                      double control_dt = 0.001);

  virtual void Initialize();
  virtual void Update(const Eigen::VectorXd& q, const Eigen::VectorXd& qdot,
                      double t, double dt);
  void SetTeleopCommand(const TaskReference& cmd);
  void ClearTeleopCommand();
  [[nodiscard]] const std::optional<TaskReference>& GetTeleopCommand() const;
  virtual void Step();
  [[nodiscard]] virtual RobotCommand GetCommand() const { return cmd_; }
  virtual void RequestState(int id);
  virtual bool RequestStateByName(const std::string& name);
  [[nodiscard]] virtual int CurrentStateId() const;
  [[nodiscard]] virtual std::optional<int> FindStateIdByName(
      const std::string& name) const;
  [[nodiscard]] virtual std::vector<std::pair<int, std::string>> GetStates()
      const;

  void SetControlDt(double dt);
  double ControlDt() const { return control_dt_; }
  void SetHoldPreviousTorqueOnFailure(bool hold) {
    hold_prev_torque_on_fail_ = hold;
  }
  bool HoldPreviousTorqueOnFailure() const { return hold_prev_torque_on_fail_; }

  FSMHandler* fsm_handler() const { return fsm_handler_.get(); }
  StateProvider* state_provider() const { return sp_.get(); }
  WbcConfigCompiler* compiled_config() const { return compiled_.get(); }
  const WbcFormulation& formulation() const { return formulation_; }
  double CurrentTime() const { return current_time_; }
  double CurrentDt() const { return current_dt_; }

protected:
  virtual void ApplyDesiredsToTasks(WbcFormulation& f,
                                    const CompiledState* state);
  virtual void OnBeforeSolve(const WbcFormulation&) {}
  virtual void OnAfterSolve(const WbcFormulation&, RobotCommand&) {}

  bool EnsureCommandBuffers();
  void UpdateRobotModelFromJointState(const Eigen::VectorXd& q,
                                      const Eigen::VectorXd& qdot) const;
  void UpdateTaskAndConstraintStates(const WbcFormulation& formulation) const;
  void SetSafeCommand();
  void ApplyStateOverridesIfNeeded(int state_id, const CompiledState& state);

  PinocchioRobotSystem* robot_{nullptr};
  std::shared_ptr<WbcConfigCompiler> compiled_;
  std::unique_ptr<StateProvider> sp_;
  std::unique_ptr<FSMHandler> fsm_handler_;

  std::optional<TaskReference> teleop_cmd_;
  WbcFormulation formulation_;
  RobotCommand cmd_;

  std::unique_ptr<QPParams> qp_params_;
  std::unique_ptr<WBIC> solver_;

  Eigen::VectorXd wbc_qddot_cmd_;
  Eigen::VectorXd joint_trq_prev_;
  double current_time_{0.0};
  double current_dt_{0.001};
  bool warned_joint_des_dim_mismatch_{false};
  bool hold_prev_torque_on_fail_{true};
  int applied_state_id_{-1};
  bool fsm_initialized_{false};
  bool initialized_{false};
  double control_dt_{0.001};
};

// Build and initialize ControlArchitecture from YAML config.
// state_provider is required so robot-specific runtime state can be injected
// from the controller package.
std::unique_ptr<ControlArchitecture> BuildControlArchitecture(
    PinocchioRobotSystem* robot, const std::string& yaml_path,
    double control_dt, std::unique_ptr<StateProvider> state_provider,
    std::unique_ptr<FSMHandler> fsm_handler = nullptr);

} // namespace wbc
