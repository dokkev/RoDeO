/**
 * @file wbc_core/wbc_architecture/include/wbc_architecture/control_architecture.hpp
 * @brief Doxygen documentation for control_architecture module.
 */
#pragma once

#include <atomic>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include "wbc_architecture/config_compiler.hpp"
#include "wbc_architecture/control_architecture_config.hpp"
#include "wbc_architecture/control_buffers.hpp"
#include "wbc_architecture/runtime_config.hpp"
#include "wbc_formulation/wbc_formulation.hpp"
#include "wbc_robot_system/pinocchio_robot_system.hpp"
#include "wbc_solver/wbic.hpp"
#include "wbc_util/joint_pid.hpp"

namespace wbc {

struct RobotCommand {
  Eigen::VectorXd tau;     ///< total torque = tau_ff + tau_fb
  Eigen::VectorXd tau_ff;  ///< feedforward: WBIC solve + physics compensations
  Eigen::VectorXd tau_fb;  ///< feedback: joint PID on q_cmd/qdot_cmd tracking error
  Eigen::VectorXd qdot;
  Eigen::VectorXd q;
};

/**
 * @brief Runtime coordinator for WBC: model update, FSM, formulation, and solve.
 *
 * @details
 * Core responsibilities:
 * 1. Receive control-critical state inputs (`RobotJointState`, optional `RobotBaseState`).
 * 2. Update robot model deterministically for the current tick.
 * 3. Execute FSM transition/update and build state-dependent formulation.
 * 4. Run WBIC solver and expose `RobotCommand`.
 *
 * Data ownership model:
 * - Owns robot model (`robot_`), compiled runtime config (`runtime_config_`),
 *   state provider (`sp_`), and FSM handler (`fsm_handler_`).
 *
 * Recommended input path (memory + RT):
 * - Joint state: direct path into Update(...), no detour through StateProvider.
 * - Floating-base/contact estimates: estimator -> RT buffer -> update thread ->
 *   StateProvider, then Update(joint, base, ...).
 * - Task references: external caller writes to StateProvider::task_input;
 *   state machines read it in OneStep() and call task->UpdateDesired().
 */
class ControlArchitecture final {
public:
  ~ControlArchitecture() = default;

  explicit ControlArchitecture(ControlArchitectureConfig arch_config);

  /**
   * @brief One-time setup for runtime objects and FSM registration.
   *
   * @details
   * - Validates config/model dimensions.
   * - Determines floating-base mode and initializes base state context.
   * - Instantiates FSM states from config recipes.
   * - Creates solver and pre-allocates formulation/command buffers.
   */
  void Initialize();

  /**
   * @brief Update tick for fixed-base path.
   *
   * @param state Actuated joint state for current control tick.
   * @param t Current wall/monotonic time in seconds.
   * @param dt Tick duration in seconds.
   *
   * @details This is the shortest joint-critical path:
   * `hardware interface -> RobotJointState -> robot_->UpdateRobotModel -> Step`.
   */
  void Update(const RobotJointState& state, double t, double dt);

  /**
   * @brief Update tick for floating-base path with externally estimated base state.
   *
   * @param state Actuated joint state for current control tick.
   * @param base_state Estimated floating-base pose/velocity snapshot for this tick.
   * @param t Current wall/monotonic time in seconds.
   * @param dt Tick duration in seconds.
   *
   * @details
   * Intended flow:
   * `state estimator(non-RT) -> RT buffer -> update thread consume ->
   *  Update(joint, base, ...)`.
   */
  void Update(const RobotJointState& state, const RobotBaseState& base_state,
              double t, double dt);

  /**
   * @brief Execute one full control cycle after model/state update.
   *
   * @details
   * Sequence:
   * 1) update provider meta (`servo_dt_`, `current_time_`, `nominal_jpos_`)
   * 2) FSM consume-request + update
   * 3) build formulation for active state
   * 4) apply desired references
   * 5) update kinematics/constraints
   * 6) run solver and produce command / safe fallback
   */
  void Step();
  [[nodiscard]] const RobotCommand& GetCommand() const { return cmd_; }

  void SetControlDt(double dt);
  double ControlDt() const { return nominal_dt_; }
  void SetHoldPreviousTorqueOnFailure(bool hold) {
    hold_prev_torque_on_fail_ = hold;
  }
  bool HoldPreviousTorqueOnFailure() const { return hold_prev_torque_on_fail_; }

  FSMHandler* GetFsmHandler() const { return fsm_handler_.get(); }
  PinocchioRobotSystem* GetRobot() const { return robot_.get(); }
  RuntimeConfig* GetConfig() const { return runtime_config_.get(); }

  /**
   * @brief RT-safe read of the currently active state id (atomic load).
   * Returns -1 when no state is active.
   */
  [[nodiscard]] StateId GetCurrentStateId() const;

  /**
   * @brief Raw pointer to the active state, or nullptr when no state is active.
   * @note Non-RT use only — do not hold across ticks.
   */
  [[nodiscard]] StateMachine* GetCurrentState() const;

  /**
   * @brief Latch a state transition request by id (thread-safe).
   * Consumed by the RT thread on the next control tick.
   */
  void RequestState(StateId id);

  /**
   * @brief Latch a state transition request by registered name (thread-safe).
   * @return true if the name was found and the request was latched.
   */
  bool RequestState(const std::string& name);

  /**
   * @brief Forward external desired values to the currently active FSM state.
   *
   * @details Delegates to `StateMachine::SetExternalInput()` on the active state.
   *          States that do not support external input use the default no-op.
   *          Only teleop states override it to update their hold references.
   *
   * @note Non-RT API. Caller is responsible for RT safety (RealtimeBuffer later).
   */
  void SetExternalInput(const TaskInput& input);

private:
  // Step() sub-phase helpers — each handles one logical phase of the control tick.
  void StateProviderUpdate();
  void FsmUpdate();
  [[nodiscard]] const StateConfig* SyncActiveState();
  [[nodiscard]] bool SolverUpdate();

  void EnsureCommandBuffers();
  void InitializeStateProviderMode();
  void InitializeFsm();
  void InitializeSolver();
  void ReserveFormulationCapacity();
  void UpdateKinematics(const WbcFormulation& formulation) const;
  void ClampCommandLimits();
  void SetSafeCommand();
  std::unique_ptr<PinocchioRobotSystem> robot_;
  std::unique_ptr<RuntimeConfig>  runtime_config_;
  std::unique_ptr<ConfigCompiler> compiler_;  // freed after Initialize()
  std::unique_ptr<StateProvider> sp_;
  std::unique_ptr<FSMHandler> fsm_handler_;
  WbcFormulation formulation_;
  RobotCommand cmd_;

  std::unique_ptr<QPParams> qp_params_;
  std::unique_ptr<WBIC> solver_;
  const Eigen::Vector3d fixed_base_zero_vec_{Eigen::Vector3d::Zero()};
  const Eigen::Quaterniond fixed_base_identity_quat_{
      Eigen::Quaterniond::Identity()};

  ControlBuffers buffers_;
  double current_time_{0.0};
  double step_dt_{0.001};
  bool hold_prev_torque_on_fail_{true};
  bool enable_gravity_comp_{false};
  bool enable_coriolis_comp_{false};
  bool enable_inertia_comp_{false};
  JointPIDConfig pid_config_;
  JointPID pid_;
  bool pid_enabled_{false};
  int applied_state_id_{-1};
  const StateConfig* cached_state_{nullptr};
  bool is_floating_base_{false};
  bool fsm_initialized_{false};
  std::atomic<bool> initialized_{false};
  double nominal_dt_{0.001};
};

} // namespace wbc
