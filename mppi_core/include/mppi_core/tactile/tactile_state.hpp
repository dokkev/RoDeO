// Copyright 2026
//
// Licensed under the Apache License, Version 2.0.

#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/StdVector>

namespace mppi_core {

enum class ContactPresence : int {
  kUnknown = -1,
  kNoContact = 0,
  kLightContact = 1,
  kStableContact = 2,
};

struct TactileContactPoint {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  bool active{false};

  // Position of this support/contact point in the tactile sensor frame.
  // Convention:
  //   x/y lie on the tactile surface,
  //   z is the local tactile normal axis.
  Eigen::Vector3d position_sensor_m{Eigen::Vector3d::Zero()};

  // Optional local normal force measurement.
  bool has_normal_force{false};
  double normal_force_n{0.0};

  // Best-effort confidence of this support point, in [0, 1].
  double confidence{1.0};

  // Optional support index for debugging/logging.
  std::size_t support_index{0};
};

struct TactileState {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  // TactileState is used both for measured tactile observations and
  // predicted tactile/contact-patch state inside MPPI rollout.

  // Generic metadata.
  bool valid{false};
  double stamp_sec{0.0};
  std::string frame_name{};

  // Contact existence and quality.
  ContactPresence contact_presence{ContactPresence::kUnknown};
  double confidence{0.0};  // [0, 1], best-effort measurement confidence.

  // Normal support.
  bool has_normal_force{false};
  double normal_force_n{0.0};

  // Contact patch location and motion in tactile sensor frame.
  bool has_centroid{false};
  Eigen::Vector2d centroid_m{Eigen::Vector2d::Zero()};
  Eigen::Vector2d centroid_velocity_mps{Eigen::Vector2d::Zero()};

  // Translational shear / incipient slip state.
  bool has_shear{false};
  Eigen::Vector2d shear_displacement_m{Eigen::Vector2d::Zero()};

  bool has_shear_velocity{false};
  Eigen::Vector2d shear_velocity_mps{Eigen::Vector2d::Zero()};

  // Rotational shear / rotational incipient slip state.
  bool has_rotational_shear{false};
  double rotational_shear_rad{0.0};

  bool has_rotational_shear_velocity{false};
  double rotational_shear_velocity_radps{0.0};

  // Generic scalar slip scores.
  // These are dimensionless planning/control features.
  double slip_score{0.0};
  double slip_velocity_score{0.0};
  double incipient_slip_score{0.0};

  // Contact support proxy.
  std::size_t contact_support_count{0};
  std::size_t support_count{0};

  // Optional derived geometry quality terms.
  // Fraction of reduced contact units that are active. This is not the
  // physical contact area.
  double contact_area_proxy{0.0};
  double edge_risk{0.0};

  // Sparse contact support geometry.
  // MPPI rollout can use these points to compute per-contact-point
  // Pinocchio Jacobians.
  std::vector<TactileContactPoint,
              Eigen::aligned_allocator<TactileContactPoint>>
      contact_points{};

  bool hasContact() const {
    return contact_presence == ContactPresence::kLightContact ||
           contact_presence == ContactPresence::kStableContact;
  }

  bool stableContact() const {
    return contact_presence == ContactPresence::kStableContact;
  }

  std::size_t activeContactPointCount() const {
    std::size_t count = 0;
    for (const auto& point : contact_points) {
      if (point.active) {
        ++count;
      }
    }
    return count;
  }
};

}  // namespace mppi_core
