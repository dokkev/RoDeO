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

  enable_gravity_comp_  = arch_config.enable_gravity_comp;
  enable_coriolis_comp_ = arch_config.enable_coriolis_comp;
  enable_inertia_comp_  = arch_config.enable_inertia_comp;
  pid_config_           = arch_config.joint_pid;
  SetControlDt(arch_config.control_dt);
}

void ControlArchitecture::Initialize() {
  runtime_config_->ValidateRobotDimensions();
  InitializeStateProviderMode();
  InitializeFsm();
  InitializeSolver();
  ReserveFormulationCapacity();

  EnsureCommandBuffers();
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
  robot_->UpdateRobotModel(fixed_base_zero_vec_, fixed_base_identity_quat_,
                           fixed_base_zero_vec_, fixed_base_zero_vec_, state.q,
                           state.qdot, false);
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
  robot_->UpdateRobotModel(base.pos, base.quat, base.lin_vel, base.ang_vel,
                           state.q, state.qdot, false);
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
  UpdateKinematics(formulation_);

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
  solver_->UpdateSetting(robot_->GetMassMatrixRef(), robot_->GetMassMatrixInverseRef(),
                         robot_->GetCoriolisRef(), robot_->GetGravityRef());

  const bool found_cfg =
      solver_->FindConfiguration(formulation_, robot_->GetJointPos(), cmd_.q,
                                 cmd_.qdot, buffers_.wbc_qddot_cmd);
  if (!found_cfg) {
    return false;
  }
  if (!solver_->MakeTorque(formulation_, buffers_.wbc_qddot_cmd, cmd_.tau)) {
    return false;
  }

  const int n_active = robot_->NumActiveDof();
  if (enable_gravity_comp_)
    cmd_.tau += robot_->GetGravityRef().tail(n_active);
  if (enable_coriolis_comp_)
    cmd_.tau += robot_->GetCoriolisRef().tail(n_active);
  if (enable_inertia_comp_)
    cmd_.tau += (robot_->GetMassMatrixRef() * buffers_.wbc_qddot_cmd).tail(n_active);

  // Store feedforward (WBIC + physics) before adding feedback.
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
  for (Constraint* c : formulation_.kinematic_constraints) {
    if (auto* pos = dynamic_cast<JointPosLimitConstraint*>(c)) {
      const auto& lim = pos->EffectiveLimits();
      for (int i = 0; i < cmd_.q.size(); ++i) {
        cmd_.q[i] = std::clamp(cmd_.q[i], lim(i, 0), lim(i, 1));
      }
    } else if (auto* vel = dynamic_cast<JointVelLimitConstraint*>(c)) {
      const auto& lim = vel->EffectiveLimits();
      for (int i = 0; i < cmd_.qdot.size(); ++i) {
        cmd_.qdot[i] = std::clamp(cmd_.qdot[i], lim(i, 0), lim(i, 1));
      }
    } else if (auto* trq = dynamic_cast<JointTrqLimitConstraint*>(c)) {
      const auto& lim = trq->EffectiveLimits();
      for (int i = 0; i < cmd_.tau.size(); ++i) {
        cmd_.tau[i] = std::clamp(cmd_.tau[i], lim(i, 0), lim(i, 1));
      }
    }
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
}

} // namespace wbc
