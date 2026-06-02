// Copyright 2026
//
// Licensed under the Apache License, Version 2.0.

#pragma once

#include <array>
#include <cstddef>
#include <stdexcept>

#include <Eigen/Core>

#include "mppi_core/tactile/tactile_state.hpp"

namespace mppi_core {

struct GraspRolloutConfig;
struct ContactForceCorrectionState;
struct ContactForceProjectionConfig;
struct ContactForceRolloutConfig;
struct PinocchioContactKinematicsContext;

enum class ObjectShapeType {
  kUnknown = 0,
  kBox = 1,
};

struct ObjectPrior {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  ObjectShapeType shape_type{ObjectShapeType::kUnknown};
  double mass_kg{0.0};
  Eigen::Vector3d dimensions_m{Eigen::Vector3d::Zero()};
  Eigen::Matrix3d inertia_kg_m2{Eigen::Matrix3d::Zero()};

  Eigen::Vector3d halfExtentsM() const { return 0.5 * dimensions_m; }
};

inline Eigen::Matrix3d BoxInertiaAboutCenterKgM2(
    double mass_kg, const Eigen::Vector3d& dimensions_m) {
  if (mass_kg <= 0.0) {
    throw std::invalid_argument(
        "BoxInertiaAboutCenterKgM2: mass must be positive");
  }
  if ((dimensions_m.array() <= 0.0).any()) {
    throw std::invalid_argument(
        "BoxInertiaAboutCenterKgM2: dimensions must be positive");
  }

  const double x = dimensions_m.x();
  const double y = dimensions_m.y();
  const double z = dimensions_m.z();
  Eigen::Matrix3d inertia = Eigen::Matrix3d::Zero();
  inertia(0, 0) = mass_kg * (y * y + z * z) / 12.0;
  inertia(1, 1) = mass_kg * (x * x + z * z) / 12.0;
  inertia(2, 2) = mass_kg * (x * x + y * y) / 12.0;
  return inertia;
}

inline ObjectPrior MakeBoxObjectPrior(double mass_kg,
                                      const Eigen::Vector3d& dimensions_m) {
  ObjectPrior prior;
  prior.shape_type = ObjectShapeType::kBox;
  prior.mass_kg = mass_kg;
  prior.dimensions_m = dimensions_m;
  prior.inertia_kg_m2 = BoxInertiaAboutCenterKgM2(mass_kg, dimensions_m);
  return prior;
}

inline ObjectPrior MakeJengaBlockObjectPrior() {
  return MakeBoxObjectPrior(0.1, Eigen::Vector3d{0.15, 0.05, 0.03});
}

inline constexpr std::size_t kMaxTactileDisturbanceScenarios = 4;

struct TactileDisturbance {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  double normal_force_delta_n{0.0};
  double slip_delta{0.0};
  Eigen::Vector2d centroid_delta_m{Eigen::Vector2d::Zero()};
};

struct TactileDisturbanceSet {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  std::array<TactileDisturbance, kMaxTactileDisturbanceScenarios> scenarios{};
  std::size_t count{0};

  bool empty() const { return count == 0; }

  void Clear() { count = 0; }

  void Add(const TactileDisturbance& disturbance) {
    if (count >= scenarios.size()) {
      throw std::out_of_range("TactileDisturbanceSet::Add: set is full");
    }
    scenarios[count++] = disturbance;
  }
};

inline TactileDisturbanceSet MakeJengaGraspDisturbanceSet() {
  TactileDisturbanceSet set;
  set.Add(TactileDisturbance{-0.6, 0.18, Eigen::Vector2d{0.0, 0.0025}});
  set.Add(TactileDisturbance{0.8, 0.0, Eigen::Vector2d{0.0, 0.0}});
  set.Add(TactileDisturbance{-0.3, 0.10, Eigen::Vector2d{0.002, 0.0}});
  return set;
}

struct GraspObservation {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Eigen::VectorXd q_measured;
  Eigen::VectorXd v_measured;
  Eigen::VectorXd q_ref_current;
  Eigen::VectorXd v_ref_current;
  Eigen::VectorXd tau;

  TactileState tactile;
  const PinocchioContactKinematicsContext* contact_kinematics{nullptr};
  const GraspRolloutConfig* grasp_rollout_config{nullptr};
  const ContactForceProjectionConfig* contact_force_projection_config{nullptr};
  const ContactForceRolloutConfig* contact_force_rollout_config{nullptr};
  const ContactForceCorrectionState* contact_force_correction_state{nullptr};
  bool has_gravity_context{false};
  Eigen::Vector3d gravity_in_sensor_frame{Eigen::Vector3d::Zero()};

  const ObjectPrior* object{nullptr};
  TactileDisturbanceSet tactile_disturbances;
  double time_s{0.0};
};

struct GraspCommand {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Eigen::VectorXd q_des;
  Eigen::VectorXd v_des;
  Eigen::VectorXd delta_q_ref;
  double dt{0.0};
};

}  // namespace mppi_core
