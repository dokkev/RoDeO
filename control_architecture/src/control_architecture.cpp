//
// Copyright (c) 2026
//

#include "control_architecture/control_architecture.hpp"

#include <chrono>
#include <stdexcept>
#include <utility>

#include <pinocchio/algorithm/joint-configuration.hpp>

#include "control_architecture/runtime/runtime_assembler.hpp"
#include "control_architecture/runtime/state_machine_assembler.hpp"

namespace wbc {
namespace {

class ScopedPhaseTimer {
 public:
  ScopedPhaseTimer(bool enabled, double& destination)
      : enabled_(enabled), destination_(destination) {
    if (enabled_) {
      start_ = Clock::now();
    }
  }

  ~ScopedPhaseTimer() {
    if (!enabled_) {
      return;
    }
    destination_ =
        std::chrono::duration<double, std::micro>(Clock::now() - start_)
            .count();
  }

 private:
  using Clock = std::chrono::steady_clock;

  bool enabled_{false};
  double& destination_;
  Clock::time_point start_;
};

}  // namespace

// ─────────────────────────────────────────────────────────────────────────────
// Constructors
// ─────────────────────────────────────────────────────────────────────────────

ControlArchitecture::ControlArchitecture(
    RuntimeConfig config, std::shared_ptr<wbc::robots::RobotSystem> robot)
    : robot_(std::move(robot)), config_(std::move(config)) {
  if (!robot_) {
    throw std::invalid_argument("ControlArchitecture: robot is null");
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Initialize
// ─────────────────────────────────────────────────────────────────────────────

void ControlArchitecture::Initialize() {
  if (initialized_) return;

  registry_ = std::make_unique<wbc::IDProblemRegistry>(*robot_);
  solver_ = std::make_unique<wbc::IDHQP>(*robot_, config_.solver_type,
                                         config_.solver_qp_params);
  solver_->setTimingEnabled(timing_enabled_);

  robot_->computeAllTerms(solver_->data(), robot_->generalized_q(),
                          robot_->generalized_v());

  BindRegistry(config_, *registry_, *robot_, solver_->data());

  StateMachineAssembler::Assemble(config_, fsm_handler_, *robot_,
                                  solver_->data(), state_factory_);

  cmd_.Initialize(*robot_);
  logger_.Initialize(*robot_);

  initialized_ = true;
}

// ─────────────────────────────────────────────────────────────────────────────
// Update
// ─────────────────────────────────────────────────────────────────────────────

void ControlArchitecture::Update(const robots::RobotState& state, double dt) {
  if (state.base) {
    robot_->updateState(state.joint, *state.base);
  } else {
    robot_->updateState(state.joint);
  }
  if (!initialized_) Initialize();
  if (!command_initialized_) {
    InitializeCommandFromRobotState();
  }

  Step(dt);
}

void ControlArchitecture::setTimingEnabled(bool enabled) {
  timing_enabled_ = enabled;
  if (solver_) {
    solver_->setTimingEnabled(enabled);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Step: FSM tick → formulation solve
// ─────────────────────────────────────────────────────────────────────────────

void ControlArchitecture::Step(double dt) {
  const double current_time = robot_->time();
  UpdateModelTerms();
  UpdateStateMachine(current_time, dt);
  auto problem = BuildProblem(current_time);
  const auto& sol = SolveProblem(problem, dt);
  ApplySolution(sol, dt);
}

void ControlArchitecture::UpdateModelTerms() {
  ScopedPhaseTimer timer(timing_enabled_, timing_stats_.model_us);
  robot_->computeAllTerms(solver_->data(), robot_->generalized_q(),
                          robot_->generalized_v());
}

void ControlArchitecture::UpdateStateMachine(double current_time, double dt) {
  ScopedPhaseTimer timer(timing_enabled_, timing_stats_.fsm_us);
  fsm_handler_.Update(current_time, dt);
}

IDProblem ControlArchitecture::BuildProblem(double current_time) {
  ScopedPhaseTimer timer(timing_enabled_, timing_stats_.problem_us);
  static const std::vector<std::string> kEmptyNames;
  static const std::vector<double> kEmptyWeights;
  static const std::vector<int> kEmptyLevels;

  const StateConfig* sc = ActiveStateConfig();
  return registry_->buildProblem(
      current_time, robot_->generalized_q(), robot_->generalized_v(),
      solver_->data(),
      sc ? sc->task_names : kEmptyNames, sc ? sc->task_weights : kEmptyWeights,
      sc ? sc->task_levels : kEmptyLevels,
      sc ? sc->contact_names : kEmptyNames);
}

const IDSolution& ControlArchitecture::SolveProblem(const IDProblem& problem,
                                                    double dt) {
  ScopedPhaseTimer timer(timing_enabled_, timing_stats_.solve_us);
  return solver_->solve(problem, dt);
}

void ControlArchitecture::ApplySolution(const IDSolution& solution,
                                        double dt) {
  ScopedPhaseTimer timer(timing_enabled_, timing_stats_.output_us);
  if (!solution.success) {
    return;
  }

  const int nq_joints = robot_->nq_joints();
  const int nv_joints = robot_->nv_joints();
  const int na = robot_->na();
  const int q_offset = robot_->is_fixed_base() ? 0 : 7;
  const int v_offset = robot_->is_fixed_base() ? 0 : 6;

  if (solution.qddot_sol.size() != robot_->nv() ||
      !solution.qddot_sol.allFinite()) {
    return;
  }
  if (solution.tau_sol.size() != na || !solution.tau_sol.allFinite()) {
    return;
  }

  math::Vector qdot_cmd_full =
      robot_->generalized_v() + dt * solution.qddot_sol;
  math::Vector integrate_delta = dt * qdot_cmd_full;
  math::Vector q_cmd_full(robot_->nq());
  pinocchio::integrate(robot_->model(), robot_->generalized_q(),
                       integrate_delta, q_cmd_full);

  if (q_cmd_full.size() < q_offset + nq_joints ||
      qdot_cmd_full.size() < v_offset + nv_joints ||
      !q_cmd_full.segment(q_offset, nq_joints).allFinite() ||
      !qdot_cmd_full.segment(v_offset, nv_joints).allFinite()) {
    return;
  }

  logger_.qddot_sol = solution.qddot_sol;
  logger_.q_cmd = q_cmd_full.segment(q_offset, nq_joints);
  logger_.qdot_cmd = qdot_cmd_full.segment(v_offset, nv_joints);
  logger_.tau_ff_cmd = solution.tau_sol;
  logger_.tau_fb_cmd.setZero(na);
  logger_.tau_cmd = logger_.tau_ff_cmd + logger_.tau_fb_cmd;

  cmd_.q = logger_.q_cmd;
  cmd_.qdot = logger_.qdot_cmd;
  cmd_.tau = logger_.tau_cmd;
  logger_.UpdateCommand(cmd_);
  command_initialized_ = true;
}

void ControlArchitecture::InitializeCommandFromRobotState() {
  if (cmd_.q.size() != robot_->nq_joints() ||
      cmd_.qdot.size() != robot_->nv_joints() ||
      cmd_.tau.size() != robot_->na()) {
    cmd_.Initialize(*robot_);
  }
  if (logger_.qddot_sol.size() != robot_->nv() ||
      logger_.cmd.q.size() != robot_->nq_joints() ||
      logger_.cmd.qdot.size() != robot_->nv_joints() ||
      logger_.cmd.tau.size() != robot_->na() ||
      logger_.q_cmd.size() != robot_->nq_joints() ||
      logger_.qdot_cmd.size() != robot_->nv_joints() ||
      logger_.tau_cmd.size() != robot_->na()) {
    logger_.Initialize(*robot_);
  }

  if (!robot_->hasState()) {
    return;
  }

  cmd_.q = robot_->jointState().q;
  cmd_.qdot = robot_->jointState().qdot;
  cmd_.tau.setZero(robot_->na());

  logger_.qddot_sol.setZero(robot_->nv());
  logger_.q_cmd = cmd_.q;
  logger_.qdot_cmd = cmd_.qdot;
  logger_.tau_ff_cmd.setZero(robot_->na());
  logger_.tau_fb_cmd.setZero(robot_->na());
  logger_.tau_cmd = cmd_.tau;
  logger_.UpdateCommand(cmd_);
  command_initialized_ = true;
}

const StateConfig* ControlArchitecture::ActiveStateConfig() const {
  const auto* current_state = fsm_handler_.GetCurrentState();
  if (!current_state) {
    return nullptr;
  }

  auto state_it = config_.states.find(current_state->id());
  if (state_it == config_.states.end()) {
    return nullptr;
  }
  return &state_it->second;
}

}  // namespace wbc
