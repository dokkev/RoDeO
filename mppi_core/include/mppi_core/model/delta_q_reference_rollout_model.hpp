// Copyright 2026
//
// Licensed under the Apache License, Version 2.0.

#pragma once

#include <cstddef>

#include <Eigen/Core>

#include "mppi_core/model/rollout.hpp"

namespace mppi_core {

struct ContactPredictionConfig {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  double friction_coefficient{0.35};
  double supporting_contact_count{2.0};
  double gravity_tangential_load_weight{1.0};
  double motion_tangential_load_weight{0.0};
  double slip_tangential_load_weight{3.0};
  double slip_velocity_weight{0.0};
  double closing_force_gain_n_per_rad{1.0};
  double opening_force_gain_n_per_rad{1.0};
  double force_proxy_max_n{5.0};
  double contact_patch_force_per_node_n{0.4};
  double slip_prediction_decay{0.9};
  double slip_prediction_margin_gain_per_n{0.2};
  double centroid_slip_drift_gain_m_per_n{0.0005};
  double slip_velocity_decay{0.85};
  double slip_velocity_margin_gain_per_nps{0.2};
  double action_slip_damping_gain_per_rad{0.0};
  double max_slip_velocity{100.0};
  double centroid_velocity_decay{0.9};
  double centroid_velocity_slip_gain{0.001};
  double max_centroid_velocity_mps{0.05};
  Eigen::VectorXd closing_direction;
};

class DeltaQReferenceRolloutModel final : public RolloutModelBase {
 public:
  explicit DeltaQReferenceRolloutModel(std::size_t joint_dim);
  DeltaQReferenceRolloutModel(std::size_t joint_dim,
                              ContactPredictionConfig prediction_config);

  std::size_t actionDim() const override { return joint_dim_; }

  void Step(const RobotRolloutState& state,
            const Eigen::Ref<const Eigen::VectorXd>& action,
            const RolloutContext& context, double dt,
            RobotRolloutState* next_state) const override;

 private:
  std::size_t joint_dim_{0};
  ContactPredictionConfig prediction_config_;
};

}  // namespace mppi_core
