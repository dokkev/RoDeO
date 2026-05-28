// Copyright 2026
//
// Licensed under the Apache License, Version 2.0.

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>

#include <Eigen/Core>

namespace mppi_core {

enum class NariTouchContactState : int {
  kNoContact = 0,
  kFewContacts = 1,
  kEnoughContacts = 2,
};

inline constexpr std::size_t kNariTouchUnitCount = 8;

struct NariTouchUnitState {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  // Sensor-local unit center in meters, using tactile frame X/Y axes.
  Eigen::Vector2d position_m{Eigen::Vector2d::Zero()};

  bool contact{false};

  // Local center-of-pressure offset from position_m.
  Eigen::Vector2d cop{Eigen::Vector2d::Zero()};

  double normal_force{0.0};
};

inline std::array<Eigen::Vector2d, kNariTouchUnitCount>
NariTouchUnitPositionsM() {
  return {
      Eigen::Vector2d{-0.003, -0.009}, Eigen::Vector2d{-0.003, -0.003},
      Eigen::Vector2d{-0.003, 0.003},  Eigen::Vector2d{-0.003, 0.009},
      Eigen::Vector2d{0.003, -0.009},  Eigen::Vector2d{0.003, -0.003},
      Eigen::Vector2d{0.003, 0.003},   Eigen::Vector2d{0.003, 0.009},
  };
}

inline Eigen::Vector3d NariTouchUnitSensorPosition(
    const NariTouchUnitState& unit) {
  return Eigen::Vector3d{unit.position_m.x(), unit.position_m.y(), 0.0};
}

struct NariTouchState {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  NariTouchState() {
    const auto unit_positions = NariTouchUnitPositionsM();
    for (std::size_t i = 0; i < units.size(); ++i) {
      units[i].position_m = unit_positions[i];
    }
  }

  // NARI-specific slip/shear state.
  // Convention:
  //   x/y = translational shear displacement components in tactile frame.
  //   z   = rotational shear displacement around tactile normal.
  Eigen::Vector3d slip_state{Eigen::Vector3d::Zero()};

  // Time derivative of slip_state.
  Eigen::Vector3d slip_velocity_state{Eigen::Vector3d::Zero()};

  // Contact centroid velocity in tactile frame.
  Eigen::Vector2d centroid_velocity_mps{Eigen::Vector2d::Zero()};

  // Aggregate normal force fallback.
  double force_z{0.0};

  NariTouchContactState contact_state{NariTouchContactState::kNoContact};

  std::array<NariTouchUnitState, kNariTouchUnitCount> units{};

  bool hasContact() const {
    if (contact_state == NariTouchContactState::kFewContacts ||
        contact_state == NariTouchContactState::kEnoughContacts) {
      return true;
    }
    for (const auto& unit : units) {
      if (unit.contact) {
        return true;
      }
    }
    return false;
  }

  std::size_t contactUnitCount() const {
    std::size_t count = 0;
    for (const auto& unit : units) {
      if (unit.contact) {
        ++count;
      }
    }
    return count;
  }

  std::size_t contactNodeCount() const { return contactUnitCount(); }
};

using NariTouchNodeState = NariTouchUnitState;
inline constexpr std::size_t kNariTouchNodeCount = kNariTouchUnitCount;

inline std::array<Eigen::Vector2d, kNariTouchNodeCount>
NariTouchNodePositionsM() {
  return NariTouchUnitPositionsM();
}

inline Eigen::Vector3d NariTouchNodeSensorPosition(
    const NariTouchNodeState& node) {
  return NariTouchUnitSensorPosition(node);
}

constexpr int ToContactStateValue(NariTouchContactState state) {
  return static_cast<int>(state);
}

inline double ComputeNariTouchTotalNormalForceN(
    const NariTouchState& sensor) {
  double unit_force_sum = 0.0;
  for (const auto& unit : sensor.units) {
    if (std::isfinite(unit.normal_force) && unit.normal_force > 0.0) {
      unit_force_sum += unit.normal_force;
    }
  }
  if (unit_force_sum > 0.0) {
    return unit_force_sum;
  }

  if (!std::isfinite(sensor.force_z)) {
    return 0.0;
  }
  return std::max(0.0, sensor.force_z);
}

inline double ComputeNariTouchSlipMagnitude(const NariTouchState& sensor) {
  if (!sensor.slip_state.allFinite()) {
    return 0.0;
  }
  return sensor.slip_state.norm();
}

inline double ComputeNariTouchSlipVelocityMagnitude(
    const NariTouchState& sensor) {
  if (!sensor.slip_velocity_state.allFinite()) {
    return 0.0;
  }
  return sensor.slip_velocity_state.norm();
}

inline double ComputeNariTouchSlipRisk(const NariTouchState& sensor,
                                       double velocity_weight) {
  return ComputeNariTouchSlipMagnitude(sensor) +
         std::max(0.0, velocity_weight) *
             ComputeNariTouchSlipVelocityMagnitude(sensor);
}

inline bool ComputeNariTouchContactCentroidM(
    const NariTouchState& sensor, Eigen::Vector2d* centroid_m) {
  if (centroid_m == nullptr) {
    return false;
  }

  Eigen::Vector2d weighted_sum = Eigen::Vector2d::Zero();
  double weight_sum = 0.0;

  for (const auto& unit : sensor.units) {
    if (!unit.contact) {
      continue;
    }

    double weight = 1.0;
    if (std::isfinite(unit.normal_force) && unit.normal_force > 0.0) {
      weight = unit.normal_force;
    }

    Eigen::Vector2d contact_position_m = unit.position_m;
    if (unit.cop.allFinite()) {
      contact_position_m += unit.cop;
    }

    weighted_sum += weight * contact_position_m;
    weight_sum += weight;
  }

  if (weight_sum <= 0.0) {
    return false;
  }

  *centroid_m = weighted_sum / weight_sum;
  return true;
}

}  // namespace mppi_core
