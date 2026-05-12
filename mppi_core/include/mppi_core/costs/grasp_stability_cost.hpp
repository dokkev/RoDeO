// Copyright 2026
//
// Licensed under the Apache License, Version 2.0.

#pragma once

#include <cstddef>

#include <Eigen/Core>

#include "mppi_core/contact/naritouch.hpp"
#include "mppi_core/costs/cost_term_base.hpp"

namespace mppi_core {

struct GraspStabilityCostConfig {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  double force_min_n{0.5};
  double force_max_n{2.5};
  double force_under_weight{20.0};
  double force_over_weight{40.0};
  bool use_object_weight_lower_bound{true};
  double object_force_safety_factor{1.5};
  double friction_coefficient{0.35};
  double friction_mu_min{0.2};
  double supporting_contact_count{2.0};

  double slip_threshold{0.25};
  double slip_risk_weight{4.0};
  double slip_velocity_weight{0.0};

  bool friction_margin_enabled{true};
  double friction_margin_weight{20.0};
  double required_force_weight{10.0};

  double gravity_tangential_load_weight{1.0};
  double motion_tangential_load_weight{0.0};
  double slip_tangential_load_weight{3.0};
  double force_spike_tangential_load_weight{5.0};

  double closing_force_gain_n_per_rad{1.0};
  double opening_force_gain_n_per_rad{1.0};
  double force_proxy_max_n{5.0};
  double contact_patch_force_per_node_n{0.4};
  double slip_prediction_decay{0.9};
  double slip_prediction_margin_gain_per_n{0.2};
  double centroid_slip_drift_gain_m_per_n{0.0005};

  bool contact_centroid_enabled{true};
  double centroid_boundary_weight{20.0};
  double centroid_x_min{-0.008};
  double centroid_x_max{0.008};
  double centroid_y_min{-0.008};
  double centroid_y_max{0.008};
  double contact_loss_weight{15.0};

  double tracking_weight{5.0};
  double tracking_action_scale_weight{20.0};
  double action_smoothness_weight{1.0};
  double joint_limit_weight{10.0};
  Eigen::VectorXd joint_lower_bound;
  Eigen::VectorXd joint_upper_bound;

  Eigen::VectorXd closing_direction;
};

class GraspStabilityCost final : public CostTermBase {
 public:
  explicit GraspStabilityCost(GraspStabilityCostConfig config);

  double Evaluate(const RobotRolloutState& state,
                  const Eigen::Ref<const Eigen::VectorXd>& action,
                  const CostContext& context) const override;

 private:
  double ClosingDelta(const Eigen::Ref<const Eigen::VectorXd>& action) const;
  double CumulativeClosingDelta(const RobotRolloutState& state,
                                const Eigen::Ref<const Eigen::VectorXd>& action,
                                const RolloutContext& rollout) const;
  double NormalForceProxyN(const NariTouchState& tactile,
                           const RobotRolloutState& state,
                           const Eigen::Ref<const Eigen::VectorXd>& action,
                           const RolloutContext& rollout) const;
  double TangentialLoadProxyN(double slip_risk,
                              const RolloutContext& rollout) const;
  double MinimumForceN(const RolloutContext& rollout) const;
  double TrackingGuardCost(const Eigen::Ref<const Eigen::VectorXd>& action,
                           const RolloutContext& rollout) const;
  double JointLimitCost(const RobotRolloutState& state,
                        const Eigen::Ref<const Eigen::VectorXd>& action) const;
  double ScenarioCost(double normal_force_proxy_n,
                      double tangential_load_proxy_n, double friction_margin_n,
                      double slip_risk,
                      const Eigen::Vector2d& predicted_centroid_m,
                      bool centroid_valid, double force_min_n) const;

  GraspStabilityCostConfig config_;
};

}  // namespace mppi_core
