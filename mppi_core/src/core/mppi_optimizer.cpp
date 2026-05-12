// Copyright 2026
//
// Licensed under the Apache License, Version 2.0.

#include "mppi_core/core/mppi_optimizer.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace mppi_core {
namespace {

constexpr double kLargeCost = 1.0e30;

Eigen::VectorXd DefaultVector(std::size_t dim, double value) {
  return Eigen::VectorXd::Constant(static_cast<Eigen::Index>(dim), value);
}

void PrepareConfigVectors(MPPIConfig* config) {
  const auto dim = config->action_dim;
  if (config->action_lower_bound.size() == 0) {
    config->action_lower_bound =
        DefaultVector(dim, -std::numeric_limits<double>::infinity());
  }
  if (config->action_upper_bound.size() == 0) {
    config->action_upper_bound =
        DefaultVector(dim, std::numeric_limits<double>::infinity());
  }
  if (config->action_noise_std.size() == 0) {
    config->action_noise_std = DefaultVector(dim, 1.0);
  }
}

void CheckVectorDim(const Eigen::VectorXd& value, std::size_t dim,
                    const char* name) {
  if (value.size() != static_cast<Eigen::Index>(dim)) {
    throw std::invalid_argument(std::string("MPPIConfig: ") + name +
                                " dimension mismatch");
  }
}

bool IsFiniteOrInfinite(const Eigen::VectorXd& value) {
  for (Eigen::Index i = 0; i < value.size(); ++i) {
    if (std::isnan(value[i])) {
      return false;
    }
  }
  return true;
}

bool IsFiniteAndNonnegative(const Eigen::VectorXd& value) {
  for (Eigen::Index i = 0; i < value.size(); ++i) {
    if (!std::isfinite(value[i]) || value[i] < 0.0) {
      return false;
    }
  }
  return true;
}

void CheckConfig(const MPPIConfig& config) {
  if (config.horizon_steps == 0) {
    throw std::invalid_argument("MPPIConfig: horizon_steps must be nonzero");
  }
  if (config.num_rollouts == 0) {
    throw std::invalid_argument("MPPIConfig: num_rollouts must be nonzero");
  }
  if (config.action_dim == 0) {
    throw std::invalid_argument("MPPIConfig: action_dim must be nonzero");
  }
  if (!std::isfinite(config.dt) || config.dt <= 0.0) {
    throw std::invalid_argument("MPPIConfig: dt must be finite and positive");
  }
  if (!std::isfinite(config.temperature) || config.temperature <= 0.0) {
    throw std::invalid_argument(
        "MPPIConfig: temperature must be finite and positive");
  }
  CheckVectorDim(config.action_lower_bound, config.action_dim,
                 "action_lower_bound");
  CheckVectorDim(config.action_upper_bound, config.action_dim,
                 "action_upper_bound");
  CheckVectorDim(config.action_noise_std, config.action_dim,
                 "action_noise_std");

  if (!IsFiniteOrInfinite(config.action_lower_bound) ||
      !IsFiniteOrInfinite(config.action_upper_bound)) {
    throw std::invalid_argument("MPPIConfig: action bounds cannot contain NaN");
  }
  if (!IsFiniteAndNonnegative(config.action_noise_std)) {
    throw std::invalid_argument(
        "MPPIConfig: action_noise_std must be finite and nonnegative");
  }
  for (Eigen::Index i = 0; i < config.action_lower_bound.size(); ++i) {
    if (config.action_lower_bound[i] > config.action_upper_bound[i]) {
      throw std::invalid_argument(
          "MPPIConfig: action_lower_bound must be <= action_upper_bound");
    }
  }
}

double SanitizeCost(double cost) {
  if (!std::isfinite(cost)) {
    return kLargeCost;
  }
  return std::min(cost, kLargeCost);
}

}  // namespace

void MPPIOptimizer::Initialize(MPPIConfig config,
                               std::shared_ptr<const RolloutModelBase> model,
                               std::shared_ptr<const CostTermBase> cost_term) {
  if (!model) {
    throw std::invalid_argument("MPPIOptimizer::Initialize: model is null");
  }

  PrepareConfigVectors(&config);
  CheckConfig(config);
  if (model->actionDim() != config.action_dim) {
    throw std::invalid_argument(
        "MPPIOptimizer::Initialize: model action dimension mismatch");
  }

  config_ = std::move(config);
  model_ = std::move(model);
  cost_term_ = std::move(cost_term);
  rng_.seed(config_.random_seed);

  nominal_actions_.Resize(config_.action_dim, config_.horizon_steps);
  sampled_actions_.assign(
      config_.num_rollouts,
      ActionSequence(config_.action_dim, config_.horizon_steps));
  rollout_costs_.assign(config_.num_rollouts, 0.0);
  initialized_ = true;
}

GraspCommand MPPIOptimizer::Update(const GraspObservation& observation) {
  if (!initialized_) {
    throw std::logic_error(
        "MPPIOptimizer::Update: optimizer is not initialized");
  }

  if (!cost_term_) {
    GraspCommand command =
        MakeCommand(observation, nominal_actions_.firstAction());
    ShiftNominalTrajectory();
    return command;
  }

  SampleActionSequences();
  for (std::size_t i = 0; i < sampled_actions_.size(); ++i) {
    rollout_costs_[i] =
        SanitizeCost(EvaluateRollout(observation, sampled_actions_[i]));
  }
  UpdateNominalActionSequence();

  GraspCommand command =
      MakeCommand(observation, nominal_actions_.firstAction());
  ShiftNominalTrajectory();
  return command;
}

void MPPIOptimizer::ResetNominalActions() {
  nominal_actions_.SetZero();
}

void MPPIOptimizer::ShiftNominalTrajectory() {
  nominal_actions_.ShiftAndRepeatLast();
}

void MPPIOptimizer::SampleActionSequences() {
  std::normal_distribution<double> normal(0.0, 1.0);

  for (std::size_t rollout = 0; rollout < sampled_actions_.size(); ++rollout) {
    auto& sampled = sampled_actions_[rollout].values();
    sampled = nominal_actions_.values();

    if (rollout == 0) {
      continue;
    }

    for (Eigen::Index step = 0; step < sampled.cols(); ++step) {
      auto action = sampled.col(step);
      for (Eigen::Index dim = 0; dim < action.size(); ++dim) {
        action[dim] +=
            normal(rng_) * config_.action_noise_std[static_cast<int>(dim)];
      }
      for (Eigen::Index dim = 0; dim < action.size(); ++dim) {
        action[dim] = std::min(
            action[dim], config_.action_upper_bound[static_cast<int>(dim)]);
        action[dim] = std::max(
            action[dim], config_.action_lower_bound[static_cast<int>(dim)]);
      }
    }
  }
}

double MPPIOptimizer::EvaluateRollout(const GraspObservation& observation,
                                      const ActionSequence& actions) const {
  if (observation.q_ref_current.size() !=
      static_cast<Eigen::Index>(config_.action_dim)) {
    throw std::invalid_argument(
        "MPPIOptimizer::EvaluateRollout: q_ref_current dimension mismatch");
  }

  RobotRolloutState state;
  state.q = observation.q_ref_current;
  if (observation.v_ref_current.size() == state.q.size()) {
    state.v = observation.v_ref_current;
  } else {
    state.v = Eigen::VectorXd::Zero(state.q.size());
  }
  state.tactile = observation.tactile;
  state.tactile_initialized = true;

  RobotRolloutState measured_state;
  measured_state.q = observation.q_measured;
  measured_state.v = observation.v_measured;

  const RobotRolloutState initial_reference_state = state;
  RobotRolloutState next_state = state;
  RolloutContext rollout_context;
  rollout_context.tactile = &observation.tactile;
  rollout_context.object = observation.object;
  rollout_context.tactile_disturbances =
      observation.tactile_disturbances.empty()
          ? nullptr
          : &observation.tactile_disturbances;
  rollout_context.measured_state = &measured_state;
  rollout_context.initial_reference_state = &initial_reference_state;
  rollout_context.has_gravity_context = observation.has_gravity_context;
  rollout_context.gravity_in_sensor_frame = observation.gravity_in_sensor_frame;

  double cost = 0.0;
  for (std::size_t step = 0; step < actions.horizonSteps(); ++step) {
    const auto action = actions.values().col(static_cast<Eigen::Index>(step));
    model_->Step(state, action, rollout_context, config_.dt, &next_state);

    CostContext cost_context;
    cost_context.rollout = &rollout_context;
    cost_context.step_index = step;
    cost_context.time_s = observation.time_s + config_.dt * step;

    cost += cost_term_->Evaluate(next_state, action, cost_context);
    if (!std::isfinite(cost)) {
      return kLargeCost;
    }
    state = next_state;
  }
  return cost;
}

GraspCommand MPPIOptimizer::MakeCommand(
    const GraspObservation& observation,
    const Eigen::VectorXd& delta_q_ref) const {
  GraspCommand command;
  command.delta_q_ref = delta_q_ref;
  command.dt = config_.dt;

  if (observation.q_ref_current.size() == delta_q_ref.size()) {
    command.q_des = observation.q_ref_current + delta_q_ref;
  }
  if (config_.dt > 0.0) {
    command.v_des = delta_q_ref / config_.dt;
  }
  return command;
}

void MPPIOptimizer::UpdateNominalActionSequence() {
  const auto min_it =
      std::min_element(rollout_costs_.begin(), rollout_costs_.end());
  const double min_cost = *min_it;

  Eigen::MatrixXd weighted_actions = Eigen::MatrixXd::Zero(
      nominal_actions_.values().rows(), nominal_actions_.values().cols());
  double weight_sum = 0.0;

  for (std::size_t i = 0; i < sampled_actions_.size(); ++i) {
    const double weight =
        std::exp(-(rollout_costs_[i] - min_cost) / config_.temperature);
    weighted_actions += weight * sampled_actions_[i].values();
    weight_sum += weight;
  }

  if (weight_sum > 0.0) {
    nominal_actions_.values() = weighted_actions / weight_sum;
  }
}

}  // namespace mppi_core
