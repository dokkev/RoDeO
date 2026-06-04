// Copyright 2026
//
// Licensed under the Apache License, Version 2.0.

#pragma once

#include <cstddef>

#include <Eigen/Core>

#include "mppi_core/costs/cost_term_base.hpp"
#include "mppi_core/tactile/tactile_state.hpp"

namespace mppi_core {

struct GraspStabilityCostConfig {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  double force_min_n{0.5};
  double force_max_n{2.5};
  double force_under_weight{20.0};
  double force_over_weight{40.0};

  double slip_threshold{0.25};
  double slip_risk_weight{4.0};
  double slip_velocity_weight{0.0};

  bool contact_centroid_enabled{true};
  double centroid_boundary_weight{20.0};
  double centroid_x_min{-0.008};
  double centroid_x_max{0.008};
  double centroid_y_min{-0.008};
  double centroid_y_max{0.008};
  double contact_loss_weight{15.0};
  bool contact_patch_enabled{true};
  double contact_patch_target_node_count{6.0};
  double contact_patch_weight{2.0};

  double tracking_weight{5.0};
  double tracking_action_scale_weight{20.0};
  double action_smoothness_weight{1.0};
  double joint_limit_weight{10.0};
  Eigen::VectorXd joint_lower_bound;
  Eigen::VectorXd joint_upper_bound;
};

class GraspStabilityCost final : public CostTermBase {
 public:
  explicit GraspStabilityCost(GraspStabilityCostConfig config);

  double Evaluate(const RobotRolloutState& state,
                  const Eigen::Ref<const Eigen::VectorXd>& action,
                  const CostContext& context) const override;

 private:
  double TrackingGuardCost(const Eigen::Ref<const Eigen::VectorXd>& action,
                           const RolloutContext& rollout) const;
  double JointLimitCost(const RobotRolloutState& state,
                        const Eigen::Ref<const Eigen::VectorXd>& action) const;
  double ContactLocalCost(double normal_force_n, double slip_risk,
                          const Eigen::Vector2d& predicted_centroid_m,
                          bool centroid_valid,
                          std::size_t contact_support_count) const;

  GraspStabilityCostConfig config_;
};

}  // namespace mppi_core
