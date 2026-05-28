// Copyright 2026
//
// Licensed under the Apache License, Version 2.0.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/StdVector>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/jacobian.hpp>

#include "mppi_core/grasp/grasp_contact_kinematics.hpp"
#include "mppi_core/grasp/grasp_state.hpp"

namespace mppi_core {

struct ContactForceProjectionConfig {
  bool enabled{true};

  double regularization{1.0e-6};
  double tactile_prior_weight{1.0};

  double friction_coefficient{0.8};

  // If true, negative normal force is clamped after solve.
  bool clamp_negative_normal_force{true};
};

struct ContactPointForce {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  std::size_t support_index{0};

  Eigen::Vector3d position_sensor_m{Eigen::Vector3d::Zero()};

  // Expressed in the tactile sensor frame after normal_axis_sign convention;
  // z is positive in the closing/compression direction used by rollout.
  Eigen::Vector3d force_sensor_n{Eigen::Vector3d::Zero()};

  double normal_force_n{0.0};
  Eigen::Vector2d tangential_force_n{Eigen::Vector2d::Zero()};

  // Positive means inside a linearized friction bound, negative means
  // tangential force exceeds mu * normal_force_n.
  double friction_margin{0.0};
};

struct ContactForceProjectionResult {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  bool valid{false};

  std::vector<ContactPointForce, Eigen::aligned_allocator<ContactPointForce>>
      contact_forces;

  double total_normal_force_n{0.0};
  Eigen::Vector2d net_tangential_force_n{Eigen::Vector2d::Zero()};
  double net_torsional_moment_nm{0.0};

  double total_friction_violation{0.0};

  // Diagnostics.
  double residual_norm{0.0};
};

inline bool IsFinitePositive(double x) {
  return std::isfinite(x) && x > 0.0;
}

inline double SafeNonnegative(double x) {
  return std::isfinite(x) ? std::max(0.0, x) : 0.0;
}

inline double ContactPointNormalForcePriorN(const TactileContactPoint& point,
                                            const TactileState& tactile,
                                            std::size_t active_count) {
  if (point.has_normal_force && IsFinitePositive(point.normal_force_n)) {
    return point.normal_force_n;
  }

  if (tactile.has_normal_force && IsFinitePositive(tactile.normal_force_n) &&
      active_count > 0) {
    return tactile.normal_force_n / static_cast<double>(active_count);
  }

  return 0.0;
}

inline void ApplyTactileNormalAxisConventionToJacobian(
    const PinocchioContactKinematicsContext& context,
    Eigen::Matrix<double, 3, Eigen::Dynamic>* point_jacobian_sensor) {
  if (point_jacobian_sensor == nullptr) {
    return;
  }
  // The projection solves forces in the same tactile convention used by
  // ContactPointMotion: x/y are tangent axes and positive z is closing.
  if (std::isfinite(context.normal_axis_sign) &&
      context.normal_axis_sign < 0.0) {
    point_jacobian_sensor->row(2) *= -1.0;
  }
}

inline ContactForceProjectionResult ProjectContactForcesFromTorqueResidual(
    const GraspState& state, const Eigen::VectorXd& tau_residual,
    const PinocchioContactKinematicsContext& context,
    const ContactForceProjectionConfig& config = {}) {
  ContactForceProjectionResult result;

  if (!config.enabled || !state.valid ||
      !IsValidContactKinematicsContext(context) ||
      state.q.size() != static_cast<Eigen::Index>(context.model->nq) ||
      tau_residual.size() != static_cast<Eigen::Index>(context.model->nv) ||
      !state.q.allFinite() || !tau_residual.allFinite() ||
      state.tactile.contact_points.empty()) {
    return result;
  }

  std::vector<const TactileContactPoint*> active_points;
  active_points.reserve(state.tactile.contact_points.size());
  for (const auto& point : state.tactile.contact_points) {
    if (point.active && point.position_sensor_m.allFinite()) {
      active_points.push_back(&point);
    }
  }
  if (active_points.empty()) {
    return result;
  }

  auto& data = *context.data;
  const auto& model = *context.model;

  pinocchio::computeJointJacobians(model, data, state.q);
  pinocchio::updateFramePlacements(model, data);

  Eigen::Matrix<double, 6, Eigen::Dynamic> frame_jacobian(6, model.nv);
  frame_jacobian.setZero();
  pinocchio::getFrameJacobian(model, data, context.sensor_frame_id,
                              pinocchio::LOCAL, frame_jacobian);

  const Eigen::Index contact_dim =
      static_cast<Eigen::Index>(3 * active_points.size());
  Eigen::MatrixXd torque_from_force(model.nv, contact_dim);
  torque_from_force.setZero();
  Eigen::VectorXd force_prior = Eigen::VectorXd::Zero(contact_dim);

  Eigen::Matrix<double, 3, Eigen::Dynamic> point_jacobian(3, model.nv);
  for (std::size_t i = 0; i < active_points.size(); ++i) {
    const auto& point = *active_points[i];
    if (!ComputeContactPointJacobianSensor(
            frame_jacobian, point.position_sensor_m, &point_jacobian)) {
      return ContactForceProjectionResult{};
    }
    ApplyTactileNormalAxisConventionToJacobian(context, &point_jacobian);

    const Eigen::Index col = static_cast<Eigen::Index>(3 * i);
    torque_from_force.block(0, col, model.nv, 3) = point_jacobian.transpose();
    force_prior.segment<3>(col) =
        Eigen::Vector3d{0.0, 0.0,
                        ContactPointNormalForcePriorN(point, state.tactile,
                                                      active_points.size())};
  }

  const double regularization = SafeNonnegative(config.regularization);
  const double prior_weight = SafeNonnegative(config.tactile_prior_weight);

  Eigen::MatrixXd lhs = torque_from_force.transpose() * torque_from_force;
  lhs.diagonal().array() += regularization + prior_weight;

  Eigen::VectorXd rhs =
      torque_from_force.transpose() * tau_residual + prior_weight * force_prior;

  Eigen::VectorXd solved_force = lhs.ldlt().solve(rhs);
  if (solved_force.size() != contact_dim || !solved_force.allFinite()) {
    return ContactForceProjectionResult{};
  }

  const double friction_mu = SafeNonnegative(config.friction_coefficient);
  if (config.clamp_negative_normal_force) {
    for (std::size_t i = 0; i < active_points.size(); ++i) {
      solved_force[static_cast<Eigen::Index>(3 * i + 2)] =
          std::max(0.0, solved_force[static_cast<Eigen::Index>(3 * i + 2)]);
    }
  }

  result.contact_forces.reserve(active_points.size());
  for (std::size_t i = 0; i < active_points.size(); ++i) {
    const auto& point = *active_points[i];
    const Eigen::Vector3d force =
        solved_force.segment<3>(static_cast<Eigen::Index>(3 * i));

    ContactPointForce point_force;
    point_force.support_index = point.support_index;
    point_force.position_sensor_m = point.position_sensor_m;
    point_force.force_sensor_n = force;
    point_force.normal_force_n = SafeNonnegative(force.z());
    point_force.tangential_force_n = force.head<2>();
    point_force.friction_margin = friction_mu * point_force.normal_force_n -
                                  point_force.tangential_force_n.norm();

    result.total_normal_force_n += point_force.normal_force_n;
    result.net_tangential_force_n += point_force.tangential_force_n;
    // Torsional moment is computed around the tactile sensor origin. If
    // patch-centered torsion is needed later, subtract the force-weighted
    // contact centroid before accumulating this cross product.
    result.net_torsional_moment_nm +=
        point.position_sensor_m.x() * point_force.tangential_force_n.y() -
        point.position_sensor_m.y() * point_force.tangential_force_n.x();
    result.total_friction_violation +=
        std::max(0.0, -point_force.friction_margin);
    result.contact_forces.push_back(point_force);
  }

  result.residual_norm =
      (torque_from_force * solved_force - tau_residual).norm();
  result.valid = std::isfinite(result.total_normal_force_n) &&
                 result.net_tangential_force_n.allFinite() &&
                 std::isfinite(result.net_torsional_moment_nm) &&
                 std::isfinite(result.total_friction_violation) &&
                 std::isfinite(result.residual_norm);

  return result;
}

}  // namespace mppi_core
