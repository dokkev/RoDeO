//
// Copyright (c) 2026
//

#include "wbc_core/architecture/control_architecture.hpp"

#include <chrono>
#include <utility>

#include <pinocchio/algorithm/joint-configuration.hpp>

#include "wbc_core/runtime/config_compiler.hpp"
#include "wbc_core/runtime/config_loader.hpp"
#include "wbc_core/runtime/config_validator.hpp"
#include "wbc_core/runtime/runtime_assembler.hpp"

namespace wbc {

// ─────────────────────────────────────────────────────────────────────────────
// Constructors
// ─────────────────────────────────────────────────────────────────────────────

ControlArchitecture::ControlArchitecture(
    const std::string& yaml_path,
    const std::string& urdf_path,
    const std::vector<std::string>& package_dirs) {
  yaml_path_ = yaml_path;
  yaml_config_ = YAML::LoadFile(yaml_path);

  // Read floating-base flag from YAML (default: false).
  bool floating_base = false;
  if (yaml_config_["robot_model"] &&
      yaml_config_["robot_model"]["is_floating_base"]) {
    floating_base =
        yaml_config_["robot_model"]["is_floating_base"].as<bool>();
  }

  if (floating_base) {
    robot_ = std::make_shared<tsid::robots::RobotWrapper>(
        urdf_path, package_dirs, pinocchio::JointModelFreeFlyer());
  } else {
    robot_ = std::make_shared<tsid::robots::RobotWrapper>(
        urdf_path, package_dirs, false);
  }
}

ControlArchitecture::ControlArchitecture(
    const YAML::Node& yaml_config,
    std::shared_ptr<tsid::robots::RobotWrapper> robot)
    : robot_(robot), yaml_config_(yaml_config) {}

// ─────────────────────────────────────────────────────────────────────────────
// Initialize
// ─────────────────────────────────────────────────────────────────────────────

void ControlArchitecture::Initialize() {
  if (initialized_) return;

  const int na = robot_->na();
  const int nv = robot_->nv();

  // Parse config -> compiled config -> validate -> assemble runtime.
  YAML::Node root = yaml_config_;
  if (!yaml_path_.empty()) {
    root = ConfigLoader::LoadFileAndResolve(yaml_path_);
  }
  CompiledConfig compiled_config = ConfigCompiler::Compile(root);
  ConfigValidator::Validate(compiled_config);
  config_ = RuntimeAssembler::Assemble(compiled_config, *robot_);

  // Create final-form runtime.
  registry_ = std::make_unique<tsid::WBMCRegistry>(*robot_);
  solver_ = std::make_unique<tsid::WBMC>(*robot_);
  solver_->setTimingEnabled(timing_enabled_);

  state_provider_.Initialize(robot_->nq(), nv);

  // Bootstrap current kinematics once so FSM states can query frame data.
  robot_->update(
      solver_->data(), pinocchio::neutral(robot_->model()),
      tsid::math::Vector::Zero(nv));

  // Build FSM
  RuntimeAssembler::InitializeFsm(
      config_, *registry_, fsm_handler_, state_provider_,
      *robot_, solver_->data());

  cmd_.Initialize(na);

  initialized_ = true;
}

// ─────────────────────────────────────────────────────────────────────────────
// Update
// ─────────────────────────────────────────────────────────────────────────────

void ControlArchitecture::Update(const RobotJointState& state, double t,
                                 double dt) {
  if (!initialized_) Initialize();

  current_time_ = t;
  dt_ = dt;

  state_provider_.Update(t, dt, state.q, state.qdot);

  Step();
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

void ControlArchitecture::Step() {
  using Clock = std::chrono::high_resolution_clock;
  auto t0 = Clock::now();

  const auto& q = state_provider_.nominal_jpos;
  const auto& v = state_provider_.nominal_jvel;

  // Keep solver data synchronized before state logic updates task references.
  robot_->update(solver_->data(), q, v);

  // 1. FSM update
  fsm_handler_.Update(current_time_);
  state_provider_.state = fsm_handler_.GetCurrentStateId();

  auto t1 = Clock::now();

  // 2. Build step input directly from the registry + current active state.
  static const std::vector<std::string> kEmptyNames;
  static const std::vector<double> kEmptyWeights;
  const auto* currentState = fsm_handler_.GetCurrentState();
  const StateConfig* sc = nullptr;
  if (currentState) {
    auto stateIt = config_.states.find(currentState->id());
    if (stateIt != config_.states.end()) {
      sc = &stateIt->second;
    }
  }

  auto input = registry_->buildStepInput(
      current_time_, q, v,
      sc ? sc->task_names : kEmptyNames,
      sc ? sc->task_weights : kEmptyWeights,
      sc ? sc->contact_names : kEmptyNames);

  auto t2 = Clock::now();

  // 3. Solve strict-priority WBMC core directly from step input.
  const auto& sol = solver_->solve(input);

  auto t3 = Clock::now();

  if (sol.success) {
    RobotCommand next_cmd = cmd_;
    if (command_adapter_.fromSolution(sol, *robot_, q, v, dt_, next_cmd)) {
      cmd_ = std::move(next_cmd);
    }
  }
  // Hold previous command when solve or command adaptation fails.

  if (timing_enabled_) {
    auto us = [](auto a, auto b) {
      return std::chrono::duration<double, std::micro>(b - a).count();
    };
    timing_stats_.find_config_us = us(t0, t1);
    timing_stats_.kinematics_us = us(t1, t2);
    timing_stats_.make_torque_us = us(t2, t3);
    timing_stats_.feedback_us = us(t3, Clock::now());
  }
}

}  // namespace wbc
