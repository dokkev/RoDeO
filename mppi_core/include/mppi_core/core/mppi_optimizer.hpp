// Copyright 2026
//
// Licensed under the Apache License, Version 2.0.

#pragma once

#include <memory>
#include <random>
#include <vector>

#include "mppi_core/core/action_sequence.hpp"
#include "mppi_core/core/mppi_config.hpp"
#include "mppi_core/costs/cost_term_base.hpp"
#include "mppi_core/grasp_types.hpp"
#include "mppi_core/model/rollout.hpp"

namespace mppi_core {

struct RolloutTrace {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  // states[0] is the initial rollout state. states[k + 1] is after actions[k].
  std::vector<RobotRolloutState> states;
  std::vector<Eigen::VectorXd> actions;
  std::vector<double> step_costs;
  double total_cost{0.0};
};

class MPPIOptimizer {
 public:
  MPPIOptimizer() = default;

  void Initialize(MPPIConfig config,
                  std::shared_ptr<const RolloutModelBase> model,
                  std::shared_ptr<const CostTermBase> cost_term);

  RobotCommand Update(const GraspObservation& observation);
  RolloutTrace PredictRollout(const GraspObservation& observation,
                              const ActionSequence& actions) const;
  RolloutTrace PredictNominalRollout(const GraspObservation& observation) const;
  void ResetNominalActions();
  void ShiftNominalTrajectory();

  const MPPIConfig& config() const { return config_; }
  const ActionSequence& nominalActionSequence() const {
    return nominal_actions_;
  }

 private:
  void SampleActionSequences();
  double EvaluateRollout(const GraspObservation& observation,
                         const ActionSequence& actions) const;
  RobotCommand MakeCommand(const GraspObservation& observation,
                           const Eigen::VectorXd& delta_q_ref) const;
  void UpdateNominalActionSequence();

  bool initialized_{false};
  MPPIConfig config_;
  std::shared_ptr<const RolloutModelBase> model_;
  std::shared_ptr<const CostTermBase> cost_term_;
  ActionSequence nominal_actions_;
  std::vector<ActionSequence> sampled_actions_;
  std::vector<double> rollout_costs_;
  std::mt19937 rng_;
};

}  // namespace mppi_core
