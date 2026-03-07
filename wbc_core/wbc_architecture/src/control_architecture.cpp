/**
 * @file wbc_core/wbc_architecture/src/control_architecture.cpp
 * @brief Doxygen documentation for control_architecture module.
 */
#include "wbc_architecture/control_architecture.hpp"

#include <algorithm>
#include <cassert>
#include <limits>
#include <stdexcept>

#include "wbc_formulation/constraint.hpp"
#include "wbc_formulation/contact.hpp"
#include "wbc_formulation/interface/task.hpp"
#include "wbc_formulation/kinematic_constraint.hpp"
#include "wbc_trajectory/math_util.hpp"
#include "wbc_util/step_count_guard.hpp"
#include "wbc_util/task_registry.hpp"

namespace wbc {

ControlArchitecture::ControlArchitecture(
    ControlArchitectureConfig arch_config)
    : robot_(std::move(arch_config.robot)),
      runtime_config_(std::move(arch_config.runtime_config)),
      compiler_(std::move(arch_config.compiler)),
      sp_(std::move(arch_config.state_provider)),
      fsm_handler_(std::move(arch_config.fsm_handler)) {
  if (robot_ == nullptr) {
    throw std::runtime_error("[ControlArchitecture] robot is null.");
  }
  if (runtime_config_ == nullptr) {
    throw std::runtime_error("[ControlArchitecture] runtime config is null.");
  }
  if (sp_ == nullptr) {
    sp_ = std::make_unique<StateProvider>(arch_config.control_dt);
  }
  if (fsm_handler_ == nullptr) {
    fsm_handler_ = std::make_unique<FSMHandler>();
  }

  pid_config_           = arch_config.joint_pid;
  friction_config_      = arch_config.friction_comp;
  observer_config_      = arch_config.momentum_observer;
  ik_method_            = arch_config.ik_method;
  enable_gravity_       = arch_config.enable_gravity;
  enable_coriolis_      = arch_config.enable_coriolis;
  enable_inertia_       = arch_config.enable_inertia;
  // Weight bounds: prefer task_pool YAML (RuntimeConfig), fall back to
  // controller section (ControlArchitectureConfig) for backward compat.
  weight_scheduler_.SetWeightBounds(
      runtime_config_->WeightMin().value_or(arch_config.weight_min),
      runtime_config_->WeightMax().value_or(arch_config.weight_max));
  SetControlDt(arch_config.control_dt);
}

void ControlArchitecture::Initialize() {
  runtime_config_->ValidateRobotDimensions();
  InitializeStateProviderMode();
  InitializeFsm();
  InitializeSolver();
  ReserveFormulationCapacity();

  // For weight-based QP: build the formulation once with ALL tasks,
  // register tasks with the scheduler, and set initial weights.
  if (ik_method_ == IKMethod::WEIGHTED_QP) {
    BuildFixedFormulation();
  }

  EnsureCommandBuffers();

  // Initialize logger with robot dimensions and task name mappings.
  logger_.Initialize(robot_->NumActiveDof(), robot_->NumQdot());
  if (const auto* reg = runtime_config_->taskRegistry()) {
    for (const auto& [name, task_ptr] : reg->GetMotionTasks()) {
      logger_.RegisterTaskName(task_ptr.get(), name);
    }
    for (const auto& [name, ft_ptr] : reg->GetForceTasks()) {
      logger_.RegisterForceTaskName(ft_ptr.get(), name);
    }
  }

  initialized_.store(true, std::memory_order_release);
}

void ControlArchitecture::Update(const RobotJointState& state, double t,
                                 double dt) {
  current_time_ = t;
  step_dt_ = (dt > 0.0) ? dt : nominal_dt_;
  const int n_active = robot_->NumActiveDof();
  if (state.q.size() != n_active || state.qdot.size() != n_active) {
    SetSafeCommand();
    return;
  }
  const auto t0 = enable_timing_ ? std::chrono::high_resolution_clock::now()
                                  : std::chrono::high_resolution_clock::time_point{};
  robot_->UpdateRobotModel(fixed_base_zero_vec_, fixed_base_identity_quat_,
                           fixed_base_zero_vec_, fixed_base_zero_vec_, state.q,
                           state.qdot, false);
  if (enable_timing_) {
    const auto t1 = std::chrono::high_resolution_clock::now();
    timing_stats_.robot_model_us =
        std::chrono::duration<double, std::micro>(t1 - t0).count();
  }
  Step();
}

void ControlArchitecture::Update(const RobotJointState& state,
                                 const RobotBaseState& base_state, double t,
                                 double dt) {
  current_time_ = t;
  step_dt_ = (dt > 0.0) ? dt : nominal_dt_;
  const int n_active = robot_->NumActiveDof();
  if (state.q.size() != n_active || state.qdot.size() != n_active) {
    SetSafeCommand();
    return;
  }
  sp_->base_state_.SetAndNormalize(base_state);

  const RobotBaseState& base = sp_->base_state_;
  const auto t0 = enable_timing_ ? std::chrono::high_resolution_clock::now()
                                  : std::chrono::high_resolution_clock::time_point{};
  robot_->UpdateRobotModel(base.pos, base.quat, base.lin_vel, base.ang_vel,
                           state.q, state.qdot, false);
  if (enable_timing_) {
    const auto t1 = std::chrono::high_resolution_clock::now();
    timing_stats_.robot_model_us =
        std::chrono::duration<double, std::micro>(t1 - t0).count();
  }
  Step();
}

void ControlArchitecture::Step() {
  util::StepCountGuard step_count_guard(&sp_->count_);

  if (!initialized_.load(std::memory_order_acquire)) {
    SetSafeCommand();
    return;
  }

  // 1. Sync state-provider meta fields used by FSM states and tasks.
  StateProviderUpdate();

  // 2. Consume any pending FSM transition request, then tick the FSM.
  FsmUpdate();

  // 3. Detect state change; rebuild formulation if state changed.
  const StateConfig* state = SyncActiveState();
  if (state == nullptr) {
    SetSafeCommand();
    return;
  }

  // 4. Update task/contact/constraint kinematics for this tick.
  {
    const auto t0_kin = enable_timing_ ? std::chrono::high_resolution_clock::now()
                                       : std::chrono::high_resolution_clock::time_point{};
    UpdateKinematics(formulation_);
    if (enable_timing_) {
      const auto t1_kin = std::chrono::high_resolution_clock::now();
      timing_stats_.kinematics_us =
          std::chrono::duration<double, std::micro>(t1_kin - t0_kin).count();
    }
  }

  // 5. Run solver; produce command or safe fallback on failure.
  if (!SolverUpdate()) {
    SetSafeCommand();
  }
}

void ControlArchitecture::StateProviderUpdate() {
  sp_->servo_dt_     = (step_dt_ > 0.0) ? step_dt_ : nominal_dt_;
  sp_->current_time_ = current_time_;
  const int num_active           = robot_->NumActiveDof();
  const Eigen::VectorXd& q_curr  = robot_->GetQRef();
  if (sp_->nominal_jpos_.size() == num_active &&
      q_curr.size() >= num_active) {
    sp_->nominal_jpos_ = q_curr.tail(num_active);
  }
}

void ControlArchitecture::FsmUpdate() {
  StateId requested_state_id = -1;
  if (fsm_handler_->ConsumeRequestedState(requested_state_id)) {
    fsm_handler_->ForceTransition(requested_state_id);
  }
  fsm_handler_->Update(current_time_);
}

const StateConfig* ControlArchitecture::SyncActiveState() {
  const StateId state_id = fsm_handler_->GetCurrentStateId();

  if (ik_method_ == IKMethod::WEIGHTED_QP) {
    // Weight-based QP: formulation is fixed (built once at init).
    // On state change, schedule weight ramps instead of rebuilding.
    if (state_id != applied_state_id_) {
      cached_state_ = runtime_config_->FindState(state_id);
      if (cached_state_ != nullptr) {
        const StateConfig& state = *cached_state_;
        runtime_config_->ApplyStateOverrides(state, WbcType::WBIC);

        // Per-state ramp duration (negative = use scheduler default).
        const double ramp_dur = (state.weight_ramp_duration >= 0.0)
            ? state.weight_ramp_duration
            : weight_scheduler_.GetRampDuration();

        // Zero-allocation: pass state arrays + default configs directly.
        weight_scheduler_.ScheduleTransition(
            state.motion, state.motion_cfg,
            runtime_config_->DefaultMotionTaskConfigs(),
            current_time_, ramp_dur);
        applied_state_id_ = state_id;
      } else {
        applied_state_id_ = -1;
      }
    }

    // Tick the scheduler every cycle to advance ramps.
    weight_scheduler_.Tick(current_time_);

    if (cached_state_ == nullptr) {
      applied_state_id_ = -1;
    }
    return cached_state_;
  }

  // Hierarchy mode: rebuild formulation on state change.
  if (state_id != applied_state_id_) {
    cached_state_ = runtime_config_->FindState(state_id);
    if (cached_state_ != nullptr) {
      const StateConfig& state = *cached_state_;
      runtime_config_->ApplyStateOverrides(state, WbcType::WBIC);
      runtime_config_->BuildFormulation(state, formulation_);
      applied_state_id_ = state_id;
    } else {
      applied_state_id_ = -1;
    }
  }
  if (cached_state_ == nullptr) {
    applied_state_id_ = -1;
  }
  return cached_state_;
}

bool ControlArchitecture::SolverUpdate() {
  using Clock = std::chrono::high_resolution_clock;
  const bool timing = enable_timing_;

  // --- Dynamics (lazy Pinocchio M, Minv, cori, grav) ---
  // When a compensation flag is disabled, pass zeros so the solver omits that term
  // from the torque equation: tau = M*qddot + cori + grav - Jc^T*rf.
  const auto t0 = timing ? Clock::now() : Clock::time_point{};
  const auto& cori = enable_coriolis_ ? robot_->GetCoriolisRef() : buffers_.zero_qdot;
  const auto& grav = enable_gravity_  ? robot_->GetGravityRef()  : buffers_.zero_qdot;
  solver_->UpdateSetting(robot_->GetMassMatrixRef(), robot_->GetMassMatrixInverseRef(),
                         cori, grav);
  const auto t1 = timing ? Clock::now() : Clock::time_point{};

  // --- FindConfiguration (null-space hierarchy + LLT) ---
  const bool found_cfg =
      solver_->FindConfiguration(formulation_, robot_->GetJointPos(), cmd_.q,
                                 cmd_.qdot, buffers_.wbc_qddot_cmd);
  if (!found_cfg) {
    return false;
  }
  const auto t2 = timing ? Clock::now() : Clock::time_point{};

  // --- MakeTorque (QP setup + ProxQP solve + torque recovery) ---
  if (!solver_->MakeTorque(formulation_, buffers_.wbc_qddot_cmd, cmd_.tau)) {
    return false;
  }

  // Inertia compensation: M*qddot is embedded in the solver output.
  // The QP needs M for its constraints/cost, so we can't zero it beforehand.
  // Instead, subtract the M*qddot contribution from the actuated torque.
  if (!enable_inertia_) {
    const int n_act = robot_->NumActiveDof();
    // For fixed-base fully-actuated: tau = M*qddot + cori + grav - Jc^T*rf
    // Subtract M_act * qddot (actuated block of mass matrix times full qddot).
    cmd_.tau.noalias() -= robot_->GetMassMatrixRef().bottomRows(n_act)
                          * solver_->GetWbicData()->corrected_wbc_qddot_cmd_;
  }
  const auto t3 = timing ? Clock::now() : Clock::time_point{};

  // --- Feedback: PID + clamping ---
  // WBIC output already includes full inverse dynamics: M*qddot + Ni_dyn^T*(cori+grav).
  const int n_active = robot_->NumActiveDof();

  // --- Adaptive feedforward compensators ---
  // Applied after WBIC solve, before PID feedback, so they augment the
  // model-based feedforward with learned friction/disturbance compensation.

  // Friction compensator: tau += f_c*sign(qdot) + f_v*qdot
  if (friction_comp_enabled_) {
    friction_comp_.Compute(cmd_.qdot, robot_->GetQdotRef().tail(n_active),
                           step_dt_, tau_fric_comp_);
    cmd_.tau += tau_fric_comp_;
  }

  // Momentum observer: tau -= tau_dist (cancel estimated disturbance)
  if (momentum_obs_enabled_) {
    momentum_obs_.Compute(
        robot_->GetMassMatrixRef().bottomRightCorner(n_active, n_active),
        robot_->GetCoriolisRef().tail(n_active),
        robot_->GetGravityRef().tail(n_active),
        robot_->GetQdotRef().tail(n_active),
        buffers_.joint_trq_prev,
        step_dt_, tau_dist_comp_);
    cmd_.tau -= tau_dist_comp_;
  }

  // Store feedforward (WBIC output + compensators) before adding feedback.
  cmd_.tau_ff = cmd_.tau;

  // Joint PID feedback on q_cmd / qdot_cmd tracking error.
  if (pid_enabled_) {
    pid_.Compute(cmd_.q, cmd_.qdot,
                 robot_->GetQRef().tail(n_active),
                 robot_->GetQdotRef().tail(n_active),
                 step_dt_, cmd_.tau_fb);
    cmd_.tau += cmd_.tau_fb;
  }

  buffers_.joint_trq_prev = cmd_.tau;

  // Enforce kinematic constraint limits on the final command.
  ClampCommandLimits();
  const auto t4 = timing ? Clock::now() : Clock::time_point{};

  if (timing) {
    timing_stats_.dynamics_us =
        std::chrono::duration<double, std::micro>(t1 - t0).count();
    timing_stats_.find_config_us =
        std::chrono::duration<double, std::micro>(t2 - t1).count();
    timing_stats_.make_torque_us =
        std::chrono::duration<double, std::micro>(t3 - t2).count();
    timing_stats_.feedback_us =
        std::chrono::duration<double, std::micro>(t4 - t3).count();
  }

  // Log tick data for offline analysis.
  if (logger_.enabled) {
    QpStateData qp_state;
    if (const WBICData* wbic_data = solver_->GetWbicData(); wbic_data != nullptr) {
      qp_state.solved = wbic_data->qp_solved_;
      qp_state.status = wbic_data->qp_status_;
      qp_state.iter = wbic_data->qp_iter_;
      qp_state.pri_res = wbic_data->qp_pri_res_;
      qp_state.dua_res = wbic_data->qp_dua_res_;
      qp_state.obj = wbic_data->qp_obj_;
      qp_state.setup_time_us = wbic_data->qp_setup_time_us_;
      qp_state.solve_time_us = wbic_data->qp_solve_time_us_;
    }

    logger_.LogTick(current_time_, applied_state_id_,
                    cmd_.q, cmd_.qdot, buffers_.wbc_qddot_cmd,
                    cmd_.tau_ff, cmd_.tau_fb, cmd_.tau,
                    robot_->GetQRef().tail(n_active),
                    robot_->GetQdotRef().tail(n_active),
                    robot_->GetGravityRef().tail(n_active),
                    formulation_, &qp_state);
  }

  return true;
}

void ControlArchitecture::SetExternalInput(const TaskInput& input) {
  StateMachine* state = fsm_handler_->GetCurrentState();
  if (state != nullptr) {
    state->SetExternalInput(input);
  }
}

StateId ControlArchitecture::GetCurrentStateId() const {
  return fsm_handler_->GetCurrentStateId();
}

StateMachine* ControlArchitecture::GetCurrentState() const {
  return fsm_handler_->GetCurrentState();
}

void ControlArchitecture::RequestState(StateId id) {
  fsm_handler_->RequestState(id);
}

bool ControlArchitecture::RequestState(const std::string& name) {
  return fsm_handler_->RequestStateByName(name);
}

void ControlArchitecture::SetControlDt(double dt) {
  if (dt <= 0.0) {
    throw std::runtime_error("[ControlArchitecture] control dt must be positive.");
  }
  nominal_dt_ = dt;
  sp_->servo_dt_ = dt;
}

bool ControlArchitecture::SetResidualDynamicsConfig(
    const FrictionCompensatorConfig& friction,
    const MomentumObserverConfig& observer,
    std::string* error_msg) {
  const int num_active = robot_->NumActiveDof();
  if (num_active <= 0) {
    if (error_msg != nullptr) {
      *error_msg = "invalid active dof";
    }
    return false;
  }

  const auto valid_size = [num_active](const Eigen::VectorXd& v) {
    return v.size() == 1 || v.size() == num_active;
  };
  const auto expand = [num_active](const Eigen::VectorXd& v) -> Eigen::VectorXd {
    if (v.size() == num_active) {
      return v;
    }
    return Eigen::VectorXd::Constant(num_active, v[0]);
  };

  if (!valid_size(friction.gamma_c) || !valid_size(friction.gamma_v) ||
      !valid_size(friction.max_f_c) || !valid_size(friction.max_f_v)) {
    if (error_msg != nullptr) {
      *error_msg = "friction vectors must be size 1 or num_active";
    }
    return false;
  }
  if (!valid_size(observer.K_o) || !valid_size(observer.max_tau_dist)) {
    if (error_msg != nullptr) {
      *error_msg = "observer vectors must be size 1 or num_active";
    }
    return false;
  }

  friction_config_ = friction;
  observer_config_ = observer;

  if (!friction_comp_.IsSetup()) {
    friction_comp_.Setup(num_active);
  }
  if (tau_fric_comp_.size() != num_active) {
    tau_fric_comp_.setZero(num_active);
  }
  friction_comp_.SetGains(expand(friction_config_.gamma_c),
                          expand(friction_config_.gamma_v));
  friction_comp_.SetLimits(expand(friction_config_.max_f_c),
                           expand(friction_config_.max_f_v));
  if (friction_config_.enabled) {
    friction_comp_.Reset();
  } else {
    tau_fric_comp_.setZero();
  }
  friction_comp_enabled_ = friction_config_.enabled;

  if (!momentum_obs_.IsSetup()) {
    momentum_obs_.Setup(num_active);
  }
  if (tau_dist_comp_.size() != num_active) {
    tau_dist_comp_.setZero(num_active);
  }
  momentum_obs_.SetGain(expand(observer_config_.K_o));
  momentum_obs_.SetLimit(expand(observer_config_.max_tau_dist));
  if (observer_config_.enabled) {
    momentum_obs_.Reset();
  } else {
    tau_dist_comp_.setZero();
  }
  momentum_obs_enabled_ = observer_config_.enabled;

  return true;
}

void ControlArchitecture::UpdateKinematics(
    const WbcFormulation& formulation) const {
  Eigen::Matrix3d world_R_local = Eigen::Matrix3d::Identity();
  if (is_floating_base_) {
    world_R_local = sp_->base_state_.rot_world_local;
  }

  for (Task* task : formulation.motion_tasks) {
    assert(task != nullptr);
    task->UpdateJacobian();
    task->UpdateJacobianDotQdot();
    task->UpdateOpCommand(world_R_local);
  }

  for (Contact* contact : formulation.contact_constraints) {
    assert(contact != nullptr);
    contact->UpdateJacobian();
    contact->UpdateJacobianDotQdot();
    contact->UpdateConstraint();
    contact->UpdateOpCommand();
  }

  for (Constraint* constraint : formulation.kinematic_constraints) {
    assert(constraint != nullptr);
    constraint->UpdateJacobian();
    constraint->UpdateJacobianDotQdot();
    constraint->UpdateConstraint();
  }
}

void ControlArchitecture::ClampCommandLimits() {
  const auto& pos_lim = robot_->SoftPositionLimits();
  const auto& vel_lim = robot_->SoftVelocityLimits();
  const auto& trq_lim = robot_->SoftTorqueLimits();
  for (int i = 0; i < cmd_.q.size(); ++i) {
    cmd_.q[i]    = std::clamp(cmd_.q[i],    pos_lim(i, 0), pos_lim(i, 1));
    cmd_.qdot[i] = std::clamp(cmd_.qdot[i], vel_lim(i, 0), vel_lim(i, 1));
    cmd_.tau[i]  = std::clamp(cmd_.tau[i],  trq_lim(i, 0), trq_lim(i, 1));
  }
}

void ControlArchitecture::SetSafeCommand() {
  const int n_active = robot_->NumActiveDof();
  if (cmd_.q.size() == n_active) {
    const Eigen::VectorXd& q_curr = robot_->GetQRef();
    if (q_curr.size() >= n_active) {
      cmd_.q = q_curr.tail(n_active);
    } else {
      cmd_.q.setZero();
    }
  }
  cmd_.qdot.setZero();
  cmd_.tau_fb.setZero();
  if (hold_prev_torque_on_fail_) {
    cmd_.tau    = buffers_.joint_trq_prev;
    cmd_.tau_ff = buffers_.joint_trq_prev;
  } else {
    cmd_.tau.setZero();
    cmd_.tau_ff.setZero();
  }
}

void ControlArchitecture::InitializeStateProviderMode() {
  is_floating_base_ = (robot_->NumFloatDof() > 0);
  sp_->is_floating_base_ = is_floating_base_;
  if (!is_floating_base_) {
    sp_->base_state_.Reset();
  }
}

void ControlArchitecture::InitializeFsm() {
  if (fsm_initialized_) {
    return;
  }
  if (compiler_ == nullptr) {
    throw std::runtime_error(
        "[ControlArchitecture] ConfigCompiler is null during InitializeFsm. "
        "Was ControlArchitecture constructed from a valid ControlArchitectureConfig?");
  }

  compiler_->InitializeFsm(*runtime_config_, *fsm_handler_, *sp_);
  compiler_.reset(); // free all recipe memory — no longer needed
  fsm_initialized_ = true;
}

void ControlArchitecture::InitializeSolver() {
  const std::vector<bool> act_qdot_list = runtime_config_->BuildActuationMask();
  if (act_qdot_list.empty()) {
    throw std::runtime_error("[ControlArchitecture] empty actuation mask.");
  }
  if (static_cast<int>(act_qdot_list.size()) != robot_->NumQdot()) {
    throw std::runtime_error("[ControlArchitecture] actuation mask size mismatch.");
  }

  const auto& reg = runtime_config_->Regularization();
  qp_params_ =
      std::make_unique<QPParams>(robot_->NumQdot(), runtime_config_->MaxContactDim());
  qp_params_->W_delta_qddot_.setConstant(reg.w_qddot);
  qp_params_->W_delta_rf_.setConstant(reg.w_rf);
  qp_params_->W_xc_ddot_.setConstant(reg.w_xc_ddot);
  qp_params_->W_f_dot_.setConstant(reg.w_f_dot);
  qp_params_->W_tau_.setConstant(reg.w_tau);
  qp_params_->W_tau_dot_.setConstant(reg.w_tau_dot);
  solver_ = std::make_unique<WBIC>(act_qdot_list, qp_params_.get());
  solver_->SetIKMethod(ik_method_);

  // Pre-allocate solver buffers to avoid first-tick heap allocations.
  {
    const int max_cdim = runtime_config_->MaxContactDim();
    // Upper bound: SurfaceContact has 18 Uf rows per 6 dim (3x), PointContact 6 per 3 (2x).
    const int max_uf = 3 * max_cdim;
    solver_->ReserveCapacity(max_cdim, max_uf);
  }

  // Apply per-constraint soft/hard toggle from YAML config.
  const auto& pos_sc = runtime_config_->SoftConfig("JointPosLimitConstraint");
  const auto& vel_sc = runtime_config_->SoftConfig("JointVelLimitConstraint");
  const auto& trq_sc = runtime_config_->SoftConfig("JointTrqLimitConstraint");
  solver_->soft_params_.pos   = pos_sc.is_soft;
  solver_->soft_params_.w_pos = pos_sc.weight;
  solver_->soft_params_.vel   = vel_sc.is_soft;
  solver_->soft_params_.w_vel = vel_sc.weight;
  solver_->soft_params_.trq   = trq_sc.is_soft;
  solver_->soft_params_.w_trq = trq_sc.weight;
}

void ControlArchitecture::ReserveFormulationCapacity() {
  std::size_t max_motion = 0;
  std::size_t max_contact = 0;
  std::size_t max_force = 0;
  std::size_t max_kin = runtime_config_->GlobalConstraints().size();
  for (const auto& [state_id, state] : runtime_config_->States()) {
    (void)state_id;
    max_motion = std::max(max_motion, state.motion.size());
    max_contact = std::max(max_contact, state.contacts.size());
    max_force = std::max(max_force, state.forces.size());
    max_kin = std::max(max_kin,
                       runtime_config_->GlobalConstraints().size() + state.kin.size());
  }
  formulation_.Reserve(max_motion, max_contact, max_force, max_kin);
}

void ControlArchitecture::BuildFixedFormulation() {
  // Build a single formulation containing ALL tasks from the pool.
  // This formulation is never rebuilt — only task weights change.
  formulation_.Clear();

  const auto* reg = runtime_config_->taskRegistry();
  for (const auto& [name, task_ptr] : reg->GetMotionTasks()) {
    formulation_.motion_tasks.push_back(task_ptr.get());
    // Register each task with the weight scheduler.
    weight_scheduler_.RegisterTask(task_ptr.get());
  }

  // Global constraints (always active).
  formulation_.kinematic_constraints.insert(
      formulation_.kinematic_constraints.end(),
      runtime_config_->GlobalConstraints().begin(),
      runtime_config_->GlobalConstraints().end());

  // Collect all contacts and force tasks across all states.
  // (For weighted QP, contacts/forces from all states are registered.)
  for (const auto& [state_id, state] : runtime_config_->States()) {
    (void)state_id;
    for (Contact* c : state.contacts) {
      if (std::find(formulation_.contact_constraints.begin(),
                    formulation_.contact_constraints.end(), c) ==
          formulation_.contact_constraints.end()) {
        formulation_.contact_constraints.push_back(c);
      }
    }
    for (ForceTask* ft : state.forces) {
      if (std::find(formulation_.force_tasks.begin(),
                    formulation_.force_tasks.end(), ft) ==
          formulation_.force_tasks.end()) {
        formulation_.force_tasks.push_back(ft);
      }
    }
    for (Constraint* k : state.kin) {
      if (std::find(formulation_.kinematic_constraints.begin(),
                    formulation_.kinematic_constraints.end(), k) ==
          formulation_.kinematic_constraints.end()) {
        formulation_.kinematic_constraints.push_back(k);
      }
    }
  }

  // Set all task weights to kMinWeight initially.
  // The first SyncActiveState will schedule the correct weights.
  for (Task* task : formulation_.motion_tasks) {
    task->SetWeight(Eigen::VectorXd::Constant(
        task->Dim(), TaskWeightScheduler::kMinWeight));
  }
}

void ControlArchitecture::EnsureCommandBuffers() {
  const int num_active = robot_->NumActiveDof();
  const int num_qdot = robot_->NumQdot();
  if (num_active <= 0 || num_qdot <= 0) {
    throw std::runtime_error(
        "[ControlArchitecture] invalid robot dof while preparing command buffers.");
  }

  if (cmd_.q.size() != num_active) {
    cmd_.q = Eigen::VectorXd::Zero(num_active);
  }
  if (cmd_.qdot.size() != num_active) {
    cmd_.qdot = Eigen::VectorXd::Zero(num_active);
  }
  if (cmd_.tau.size() != num_active) {
    cmd_.tau = Eigen::VectorXd::Zero(num_active);
  }
  if (cmd_.tau_ff.size() != num_active) {
    cmd_.tau_ff = Eigen::VectorXd::Zero(num_active);
  }
  if (cmd_.tau_fb.size() != num_active) {
    cmd_.tau_fb = Eigen::VectorXd::Zero(num_active);
  }
  if (buffers_.joint_trq_prev.size() != num_active) {
    buffers_.joint_trq_prev = Eigen::VectorXd::Zero(num_active);
  }
  if (buffers_.wbc_qddot_cmd.size() != num_qdot) {
    buffers_.wbc_qddot_cmd = Eigen::VectorXd::Zero(num_qdot);
  }
  if (buffers_.zero_qdot.size() != num_qdot) {
    buffers_.zero_qdot = Eigen::VectorXd::Zero(num_qdot);
  }
  if (sp_->nominal_jpos_.size() != num_active) {
    sp_->nominal_jpos_ = Eigen::VectorXd::Zero(num_active);
  }

  // Initialize joint PID once n_active is known.
  if (pid_config_.enabled && !pid_enabled_) {
    // Broadcast scalar (size 1) to all joints; pass per-joint vector as-is.
    auto expand = [num_active](const Eigen::VectorXd& v) -> Eigen::VectorXd {
      if (v.size() == num_active) return v;
      return Eigen::VectorXd::Constant(num_active, v[0]);
    };
    pid_.Setup(num_active);
    pid_.SetPositionGains(expand(pid_config_.kp_pos),
                          expand(pid_config_.ki_pos),
                          expand(pid_config_.kd_pos));
    pid_.SetVelocityGains(expand(pid_config_.kp_vel),
                          expand(pid_config_.ki_vel),
                          expand(pid_config_.kd_vel));
    pid_.SetPositionIntegralLimit(expand(pid_config_.pos_integral_limit));
    pid_.SetVelocityIntegralLimit(expand(pid_config_.vel_integral_limit));
    pid_enabled_ = true;
  }

  // Initialize adaptive friction compensator.
  if (friction_config_.enabled && !friction_comp_enabled_) {
    auto expand = [num_active](const Eigen::VectorXd& v) -> Eigen::VectorXd {
      if (v.size() == num_active) return v;
      return Eigen::VectorXd::Constant(num_active, v[0]);
    };
    friction_comp_.Setup(num_active);
    friction_comp_.SetGains(expand(friction_config_.gamma_c),
                            expand(friction_config_.gamma_v));
    friction_comp_.SetLimits(expand(friction_config_.max_f_c),
                             expand(friction_config_.max_f_v));
    tau_fric_comp_ = Eigen::VectorXd::Zero(num_active);
    friction_comp_enabled_ = true;
  }

  // Initialize momentum observer.
  if (observer_config_.enabled && !momentum_obs_enabled_) {
    auto expand = [num_active](const Eigen::VectorXd& v) -> Eigen::VectorXd {
      if (v.size() == num_active) return v;
      return Eigen::VectorXd::Constant(num_active, v[0]);
    };
    momentum_obs_.Setup(num_active);
    momentum_obs_.SetGain(expand(observer_config_.K_o));
    momentum_obs_.SetLimit(expand(observer_config_.max_tau_dist));
    tau_dist_comp_ = Eigen::VectorXd::Zero(num_active);
    momentum_obs_enabled_ = true;
  }
}

} // namespace wbc
