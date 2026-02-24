#include "wbc_architecture/interface/control_architecture.hpp"

#include <algorithm>
#include <iostream>
#include <stdexcept>

#include "wbc_fsm/state_factory.hpp"
#include "wbc_formulation/constraint.hpp"
#include "wbc_formulation/friction_cone.hpp"
#include "wbc_formulation/interface/task.hpp"

namespace wbc {
namespace {

Eigen::Vector4d QuaternionToXyzw(const Eigen::Quaterniond& q) {
  const Eigen::Quaterniond normalized = q.normalized();
  // LinkOriTask expects desired quaternion packed as [x, y, z, w].
  return Eigen::Vector4d(normalized.x(), normalized.y(), normalized.z(),
                         normalized.w());
}

bool IsTeleopStateName(const std::string& state_name) {
  return state_name == "teleop" || state_name == "teleop_state" ||
         state_name.find("teleop") != std::string::npos;
}

std::string ResolveReferenceFrameName(
    const TaskReference& teleop, const WbcConfigCompiler* compiled) {
  if (teleop.reference_frame.has_value() &&
      !teleop.reference_frame->empty()) {
    return *teleop.reference_frame;
  }
  if (compiled != nullptr) {
    const std::string base_frame = compiled->BaseFrameName();
    if (!base_frame.empty()) {
      return base_frame;
    }
  }
  return "world";
}

Eigen::Isometry3d ResolveWorldIsoReferenceFrame(PinocchioRobotSystem* robot,
                                                const std::string& frame_name) {
  if (frame_name.empty() || frame_name == "world") {
    return Eigen::Isometry3d::Identity();
  }
  try {
    return robot->GetLinkIsometry(frame_name);
  } catch (const std::exception& e) {
    throw std::runtime_error("[ControlArchitecture] Invalid reference_frame '" +
                             frame_name + "': " + e.what());
  }
}

} // namespace

std::unique_ptr<ControlArchitecture> BuildControlArchitecture(
    PinocchioRobotSystem* robot, const std::string& yaml_path,
    double control_dt, std::unique_ptr<StateProvider> state_provider,
    std::unique_ptr<FSMHandler> fsm_handler) {
  if (robot == nullptr) {
    throw std::runtime_error("[BuildControlArchitecture] robot is null.");
  }
  if (yaml_path.empty()) {
    throw std::runtime_error("[BuildControlArchitecture] yaml_path is empty.");
  }
  if (control_dt <= 0.0) {
    throw std::runtime_error(
        "[BuildControlArchitecture] control_dt must be positive.");
  }
  if (state_provider == nullptr) {
    throw std::runtime_error(
        "[BuildControlArchitecture] state_provider is null.");
  }

  std::unique_ptr<WbcConfigCompiler> compiled =
      WbcConfigCompiler::Compile(robot, yaml_path);

  auto architecture = std::make_unique<ControlArchitecture>(
      robot, std::shared_ptr<WbcConfigCompiler>(std::move(compiled)),
      std::move(state_provider), std::move(fsm_handler), control_dt);
  architecture->Initialize();
  return architecture;
}

ControlArchitecture::ControlArchitecture(
    PinocchioRobotSystem* robot, std::shared_ptr<WbcConfigCompiler> compiled_config,
    std::unique_ptr<StateProvider> state_provider,
    std::unique_ptr<FSMHandler> fsm_handler, double control_dt)
    : robot_(robot),
      compiled_(std::move(compiled_config)),
      sp_(std::move(state_provider)),
      fsm_handler_(std::move(fsm_handler)) {
  if (robot_ == nullptr) {
    throw std::runtime_error("[ControlArchitecture] robot is null.");
  }
  if (compiled_ == nullptr) {
    throw std::runtime_error("[ControlArchitecture] compiled config is null.");
  }
  if (sp_ == nullptr) {
    sp_ = std::make_unique<StateProvider>(control_dt);
  }
  if (fsm_handler_ == nullptr) {
    fsm_handler_ = std::make_unique<FSMHandler>();
  }

  SetControlDt(control_dt);
}

void ControlArchitecture::Initialize() {
  compiled_->ValidateRobotDimensions();

  if (!fsm_initialized_) {
    if (fsm_handler_ == nullptr) {
      throw std::runtime_error("[ControlArchitecture] fsm handler is null.");
    }
    if (compiled_ == nullptr) {
      throw std::runtime_error("[ControlArchitecture] compiled config is null.");
    }

    StateFactory& factory = StateFactory::Instance();
    for (const auto& kv : compiled_->States()) {
      const CompiledState& recipe = kv.second;

      StateBuildContext context;
      context.robot = robot_;
      context.task_registry = compiled_->TaskRegistryPtr();
      context.constraint_registry = compiled_->ConstraintRegistryPtr();
      context.state_provider = sp_.get();
      context.params = recipe.params;

      std::unique_ptr<StateMachine> state =
          factory.Create(recipe.name, recipe.id, recipe.name, context);
      if (state == nullptr) {
        throw std::runtime_error("[ControlArchitecture] Unregistered state '" +
                                 recipe.name + "' (id=" +
                                 std::to_string(recipe.id) + ").");
      }

      // Keep state-level name-based lookup compatibility.
      for (const auto& entry : recipe.motion_by_name) {
        state->AssignTask(entry.first, entry.second);
      }
      for (const auto& entry : recipe.force_by_name) {
        state->AssignForceTask(entry.first, entry.second);
      }
      for (const auto& entry : recipe.contact_by_name) {
        state->AssignContact(entry.first, entry.second);
      }
      for (const auto& entry : recipe.kin_by_name) {
        state->AssignConstraint(entry.first, entry.second);
      }

      state->SetParameters(recipe.params);
      fsm_handler_->RegisterState(recipe.id, std::move(state));
    }

    const int start_id = compiled_->StartStateId();
    if (start_id < 0) {
      throw std::runtime_error(
          "[ControlArchitecture] Invalid start_state_id.");
    }
    fsm_handler_->SetStartState(start_id);
    if (fsm_handler_->GetCurrentStateId() != start_id) {
      throw std::runtime_error(
          "[ControlArchitecture] Failed to set start state.");
    }
    fsm_initialized_ = true;
  }

  const std::vector<bool> act_qdot_list = compiled_->BuildActuationMask();
  if (act_qdot_list.empty()) {
    throw std::runtime_error("[ControlArchitecture] empty actuation mask.");
  }
  if (static_cast<int>(act_qdot_list.size()) != robot_->NumQdot()) {
    throw std::runtime_error(
        "[ControlArchitecture] actuation mask size mismatch.");
  }

  const GlobalParams& g = compiled_->Globals();
  qp_params_ =
      std::make_unique<QPParams>(robot_->NumFloatDof(), compiled_->MaxContactDim());
  qp_params_->W_delta_qddot_.setConstant(g.w_qddot);
  qp_params_->W_delta_rf_.setConstant(g.w_rf);
  qp_params_->W_xc_ddot_.setConstant(g.w_xc_ddot);
  qp_params_->W_force_rate_of_change_.setConstant(g.w_force_rate_of_change);
  solver_ = std::make_unique<WBIC>(act_qdot_list, qp_params_.get());

  std::size_t max_motion = 0;
  std::size_t max_contact = 0;
  std::size_t max_force = 0;
  std::size_t max_kin = compiled_->GlobalConstraints().size();
  for (const auto& kv : compiled_->States()) {
    const CompiledState& state = kv.second;
    max_motion = std::max(max_motion, state.motion.size());
    max_contact = std::max(max_contact, state.contacts.size());
    max_force = std::max(max_force, state.forces.size());
    max_kin = std::max(max_kin, compiled_->GlobalConstraints().size() + state.kin.size());
  }
  formulation_.Reserve(max_motion, max_contact, max_force, max_kin);

  EnsureCommandBuffers();
  initialized_ = true;
}

void ControlArchitecture::SetTeleopCommand(const TaskReference& cmd) {
  teleop_cmd_ = cmd;
}

void ControlArchitecture::ClearTeleopCommand() { teleop_cmd_.reset(); }

const std::optional<TaskReference>&
ControlArchitecture::GetTeleopCommand() const {
  return teleop_cmd_;
}

void ControlArchitecture::Update(const Eigen::VectorXd& q,
                                 const Eigen::VectorXd& qdot, double t,
                                 double dt) {
  UpdateRobotModelFromJointState(q, qdot);
  current_time_ = t;
  current_dt_ = (dt > 0.0) ? dt : control_dt_;
  Step();
}

void ControlArchitecture::RequestState(int id) {
  if (fsm_handler_ == nullptr) {
    throw std::runtime_error(
        "[ControlArchitecture] fsm handler is not available.");
  }
  fsm_handler_->RequestState(id);
}

bool ControlArchitecture::RequestStateByName(const std::string& name) {
  const std::optional<int> state_id = FindStateIdByName(name);
  if (!state_id.has_value()) {
    return false;
  }
  RequestState(*state_id);
  return true;
}

int ControlArchitecture::CurrentStateId() const {
  if (fsm_handler_ == nullptr) {
    return -1;
  }
  return fsm_handler_->GetCurrentStateId();
}

std::optional<int> ControlArchitecture::FindStateIdByName(
    const std::string& name) const {
  if (name.empty() || compiled_ == nullptr) {
    return std::nullopt;
  }
  for (const auto& kv : compiled_->States()) {
    if (kv.second.name == name) {
      return kv.first;
    }
  }
  return std::nullopt;
}

std::vector<std::pair<int, std::string>> ControlArchitecture::GetStates()
    const {
  std::vector<std::pair<int, std::string>> states;
  if (compiled_ == nullptr) {
    return states;
  }

  const auto& compiled_states = compiled_->States();
  states.reserve(compiled_states.size());
  for (const auto& kv : compiled_states) {
    states.emplace_back(kv.first, kv.second.name);
  }
  std::sort(states.begin(), states.end(),
            [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; });
  return states;
}

void ControlArchitecture::Step() {
  if (!initialized_) {
    throw std::runtime_error(
        "[ControlArchitecture] Initialize() must be called before Step().");
  }
  if (!EnsureCommandBuffers()) {
    throw std::runtime_error("[ControlArchitecture] invalid robot dimensions.");
  }

  if (sp_ != nullptr) {
    sp_->servo_dt_ = (current_dt_ > 0.0) ? current_dt_ : control_dt_;
    sp_->current_time_ = current_time_;
    sp_->q_ = robot_->GetQ();
    sp_->qdot_ = robot_->GetQdot();
    sp_->nominal_jpos_ = robot_->GetJointPos();
  }

  if (fsm_handler_ == nullptr) {
    throw std::runtime_error("[ControlArchitecture] fsm handler is null.");
  }

  StateId requested_state_id = -1;
  if (fsm_handler_->ConsumeRequestedState(requested_state_id)) {
    fsm_handler_->ForceTransition(requested_state_id);
  }

  if (sp_ != nullptr) {
    sp_->prev_state_ = sp_->state_;
  }
  fsm_handler_->Update(current_time_);
  const int state_id = fsm_handler_->GetCurrentStateId();
  if (sp_ != nullptr) {
    sp_->state_ = state_id;
  }

  const CompiledState* state = compiled_->FindState(state_id);
  if (state == nullptr) {
    applied_state_id_ = -1;
    SetSafeCommand();
    if (sp_ != nullptr) {
      ++sp_->count_;
    }
    return;
  }

  ApplyStateOverridesIfNeeded(state_id, *state);
  compiled_->BuildFormulation(state_id, formulation_);
  ApplyDesiredsToTasks(formulation_, state);
  if (formulation_.motion_tasks.empty()) {
    SetSafeCommand();
    if (sp_ != nullptr) {
      ++sp_->count_;
    }
    return;
  }

  UpdateTaskAndConstraintStates(formulation_);

  OnBeforeSolve(formulation_);

  solver_->UpdateSetting(robot_->GetMassMatrixRef(), robot_->GetMassMatrixInverseRef(),
                         robot_->GetCoriolisRef(), robot_->GetGravityRef());

  const bool found_cfg =
      solver_->FindConfiguration(formulation_, robot_->GetJointPos(), cmd_.q,
                                 cmd_.qdot, wbc_qddot_cmd_);

  bool torque_ok = false;
  if (found_cfg) {
    torque_ok = solver_->MakeTorque(formulation_, wbc_qddot_cmd_, cmd_.tau);
  }

  if (!found_cfg || !torque_ok) {
    SetSafeCommand();
  } else {
    joint_trq_prev_ = cmd_.tau;
  }

  OnAfterSolve(formulation_, cmd_);

  if (sp_ != nullptr) {
    ++sp_->count_;
  }
}

void ControlArchitecture::SetControlDt(double dt) {
  if (dt <= 0.0) {
    throw std::runtime_error("[ControlArchitecture] control dt must be positive.");
  }
  control_dt_ = dt;
  if (sp_ != nullptr) {
    sp_->servo_dt_ = dt;
  }
}

void ControlArchitecture::ApplyDesiredsToTasks(WbcFormulation&,
                                               const CompiledState* state) {
  if (state == nullptr) {
    return;
  }

  const bool is_teleop_state = IsTeleopStateName(state->name);
  const TaskReference* teleop =
      (is_teleop_state && teleop_cmd_.has_value()) ? &(*teleop_cmd_) : nullptr;

  Eigen::Isometry3d world_iso_ref = Eigen::Isometry3d::Identity();
  if (teleop != nullptr &&
      (teleop->x_des.has_value() || teleop->quat_des.has_value())) {
    const std::string frame_name = ResolveReferenceFrameName(*teleop, compiled_.get());
    world_iso_ref = ResolveWorldIsoReferenceFrame(robot_, frame_name);
  }

  if (state->ee_pos != nullptr && teleop != nullptr &&
      teleop->x_des.has_value()) {
    const Eigen::Vector3d x_des_world =
        world_iso_ref.linear() * (*teleop->x_des) + world_iso_ref.translation();
    state->ee_pos->UpdateDesired(x_des_world, Eigen::Vector3d::Zero(),
                                 Eigen::Vector3d::Zero());
  }

  if (state->ee_ori != nullptr && teleop != nullptr &&
      teleop->quat_des.has_value()) {
    const Eigen::Quaterniond ref_quat(world_iso_ref.linear());
    const Eigen::Quaterniond quat_des_world =
        (ref_quat * teleop->quat_des->normalized()).normalized();
    const Eigen::Vector4d des_ori = QuaternionToXyzw(quat_des_world);
    state->ee_ori->UpdateDesired(des_ori, Eigen::Vector3d::Zero(),
                                 Eigen::Vector3d::Zero());
  }

  if (state->joint != nullptr && teleop != nullptr &&
      teleop->joint_pos.has_value()) {
    const int dim = state->joint->Dim();
    const Eigen::VectorXd& q_des = *teleop->joint_pos;
    if (q_des.size() == dim) {
      state->joint->UpdateDesired(q_des, Eigen::VectorXd::Zero(dim),
                                  Eigen::VectorXd::Zero(dim));
    } else if (!warned_joint_des_dim_mismatch_) {
      std::cerr
          << "[ControlArchitecture] Ignore teleop joint_pos due to dimension "
          << "mismatch. expected=" << dim << ", got=" << q_des.size()
          << std::endl;
      warned_joint_des_dim_mismatch_ = true;
    }
  }
}

bool ControlArchitecture::EnsureCommandBuffers() {
  if (robot_ == nullptr) {
    return false;
  }
  const int num_active = robot_->NumActiveDof();
  const int num_qdot = robot_->NumQdot();
  if (num_active <= 0 || num_qdot <= 0) {
    return false;
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
  if (joint_trq_prev_.size() != num_active) {
    joint_trq_prev_ = Eigen::VectorXd::Zero(num_active);
  }
  if (wbc_qddot_cmd_.size() != num_qdot) {
    wbc_qddot_cmd_ = Eigen::VectorXd::Zero(num_qdot);
  }
  return true;
}

void ControlArchitecture::UpdateRobotModelFromJointState(
    const Eigen::VectorXd& q, const Eigen::VectorXd& qdot) const {
  if (robot_ == nullptr) {
    throw std::runtime_error("[ControlArchitecture] robot is null.");
  }

  const int n_active = robot_->NumActiveDof();
  const int n_qdot = robot_->NumQdot();
  const int n_q = robot_->GetNumQ();
  const int n_float = robot_->NumFloatDof();

  if (q.size() != n_active || qdot.size() != n_active) {
    throw std::runtime_error(
        "[ControlArchitecture] update() expects actuated joint vectors with "
        "size NumActiveDof().");
  }

  if (n_float == 0) {
    robot_->UpdateRobotModel(Eigen::Vector3d::Zero(),
                             Eigen::Quaterniond::Identity(),
                             Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), q,
                             qdot, false);
    return;
  }

  if (n_q < 7 || n_qdot < 6) {
    throw std::runtime_error(
        "[ControlArchitecture] floating-base dimensions are invalid.");
  }

  const Eigen::VectorXd q_curr = robot_->GetQ();
  const Eigen::VectorXd qdot_curr = robot_->GetQdot();
  if (q_curr.size() != n_q || qdot_curr.size() != n_qdot) {
    throw std::runtime_error(
        "[ControlArchitecture] robot state size mismatch while preserving "
        "floating-base state.");
  }

  const Eigen::Vector3d base_pos = q_curr.segment<3>(0);
  Eigen::Quaterniond base_quat;
  base_quat.coeffs() = q_curr.segment<4>(3);
  base_quat.normalize();

  const Eigen::Matrix3d rot_w_base = base_quat.toRotationMatrix();
  // PinocchioRobotSystem::UpdateRobotModel() expects base velocities in world
  // frame. Internal qdot stores base twist in local(base) frame, so convert.
  const Eigen::Vector3d base_lin_vel_local = qdot_curr.segment<3>(0);
  const Eigen::Vector3d base_ang_vel_local = qdot_curr.segment<3>(3);
  const Eigen::Vector3d base_lin_vel = rot_w_base * base_lin_vel_local;
  const Eigen::Vector3d base_ang_vel = rot_w_base * base_ang_vel_local;

  robot_->UpdateRobotModel(base_pos, base_quat, base_lin_vel, base_ang_vel, q,
                           qdot, false);
}

void ControlArchitecture::UpdateTaskAndConstraintStates(
    const WbcFormulation& formulation) const {
  const Eigen::Matrix3d world_R_local =
      (sp_ != nullptr) ? sp_->rot_world_local_ : Eigen::Matrix3d::Identity();

  for (Task* task : formulation.motion_tasks) {
    if (task == nullptr) {
      continue;
    }
    task->UpdateJacobian();
    task->UpdateJacobianDotQdot();
    task->UpdateOpCommand(world_R_local);
  }

  for (Contact* contact : formulation.contact_constraints) {
    if (contact == nullptr) {
      continue;
    }
    contact->UpdateJacobian();
    contact->UpdateJacobianDotQdot();
    contact->UpdateConstraint();
    contact->UpdateOpCommand();
  }

  for (Constraint* constraint : formulation.kinematic_constraints) {
    if (constraint == nullptr) {
      continue;
    }
    constraint->UpdateJacobian();
    constraint->UpdateJacobianDotQdot();
    constraint->UpdateConstraint();
  }
}

void ControlArchitecture::SetSafeCommand() {
  if (cmd_.q.size() == robot_->NumActiveDof()) {
    cmd_.q = robot_->GetJointPos();
  }
  cmd_.qdot = Eigen::VectorXd::Zero(robot_->NumActiveDof());
  cmd_.tau = hold_prev_torque_on_fail_
                 ? joint_trq_prev_
                 : Eigen::VectorXd::Zero(robot_->NumActiveDof());
}

void ControlArchitecture::ApplyStateOverridesIfNeeded(
    int state_id, const CompiledState& state) {
  if (state_id == applied_state_id_) {
    return;
  }
  ApplyStateOverrides(state, WbcType::WBIC);
  applied_state_id_ = state_id;
}

} // namespace wbc
