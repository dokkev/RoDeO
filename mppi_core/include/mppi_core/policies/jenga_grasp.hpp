// Copyright 2026
//
// Licensed under the Apache License, Version 2.0.

#pragma once

#include <memory>

#include "mppi_core/core/mppi_optimizer.hpp"
#include "mppi_core/costs/grasp_stability_cost.hpp"
#include "mppi_core/grasp_types.hpp"
#include "mppi_core/model/delta_q_reference_rollout_model.hpp"

namespace mppi_core {

struct JengaGraspConfig {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  MPPIConfig mppi;
  ObjectPrior object;
  TactileDisturbanceSet disturbances;
  GraspStabilityCostConfig grasp_stability_cost;
};

JengaGraspConfig MakeDefaultJengaGraspConfig(std::size_t joint_dim);

class JengaGrasp {
 public:
  JengaGrasp() = default;

  void Initialize(std::size_t joint_dim, JengaGraspConfig config);

  GraspCommand Update(const GraspObservation& observation);
  RolloutTrace PredictRollout(const GraspObservation& observation,
                              const ActionSequence& actions) const;
  RolloutTrace PredictNominalRollout(const GraspObservation& observation) const;
  void Reset();

  const MPPIOptimizer& optimizer() const { return optimizer_; }
  const ObjectPrior& object() const { return object_; }
  const TactileDisturbanceSet& disturbances() const { return disturbances_; }

 private:
  bool initialized_{false};
  ObjectPrior object_;
  TactileDisturbanceSet disturbances_;
  std::shared_ptr<const DeltaQReferenceRolloutModel> model_;
  MPPIOptimizer optimizer_;
};

}  // namespace mppi_core
