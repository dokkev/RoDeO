//
// Copyright (c) 2026
//

#include "control_architecture/control_architecture.hpp"

#include <chrono>
#include <stdexcept>
#include <utility>

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

  robot_->computeAllTerms(solver_->data(), robot_->q(), robot_->qdot());

  BindRegistry(config_, *registry_, *robot_, solver_->data());

  StateMachineAssembler::Assemble(config_, fsm_handler_, *robot_,
                                  solver_->data(), state_factory_);

  cmd_.Initialize(robot_->na());

  initialized_ = true;
}

// ─────────────────────────────────────────────────────────────────────────────
// Update
// ─────────────────────────────────────────────────────────────────────────────

void ControlArchitecture::Update(const robots::RobotState& state, double dt) {
  robot_->updateState(state);
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
  ApplySolution(sol);
}

void ControlArchitecture::UpdateModelTerms() {
  ScopedPhaseTimer timer(timing_enabled_, timing_stats_.model_us);
  robot_->computeAllTerms(solver_->data(), robot_->q(), robot_->qdot());
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
      current_time, robot_->q(), robot_->qdot(), solver_->data(),
      sc ? sc->task_names : kEmptyNames, sc ? sc->task_weights : kEmptyWeights,
      sc ? sc->task_levels : kEmptyLevels,
      sc ? sc->contact_names : kEmptyNames);
}

const IDSolution& ControlArchitecture::SolveProblem(const IDProblem& problem,
                                                    double dt) {
  ScopedPhaseTimer timer(timing_enabled_, timing_stats_.solve_us);
  return solver_->solve(problem, dt);
}

void ControlArchitecture::ApplySolution(const IDSolution& solution) {
  ScopedPhaseTimer timer(timing_enabled_, timing_stats_.output_us);
  if (!solution.success) {
    return;
  }

  if (command_adapter_.fromSolution(solution, *robot_, cmd_)) {
    command_initialized_ = true;
  }
}

void ControlArchitecture::InitializeCommandFromRobotState() {
  const int na = robot_->na();
  const int q_offset = robot_->is_fixed_base() ? 0 : 7;
  const int v_offset = robot_->is_fixed_base() ? 0 : 6;

  if (cmd_.tau.size() != na || cmd_.q.size() != na ||
      cmd_.qdot.size() != na || cmd_.kp.size() != na ||
      cmd_.kd.size() != na) {
    cmd_.Initialize(na);
  }

  if (!robot_->hasState() || robot_->q().size() < q_offset + na ||
      robot_->qdot().size() < v_offset + na) {
    return;
  }

  cmd_.q = robot_->q().segment(q_offset, na);
  cmd_.qdot = robot_->qdot().segment(v_offset, na);
  cmd_.tau.setZero(na);
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
