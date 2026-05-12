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

inline constexpr std::size_t kNariTouchNodeCount = 8;

struct NariTouchNodeState {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  // Sensor-local node center in meters, using the tactile frame X/Y axes.
  Eigen::Vector2d position_m{Eigen::Vector2d::Zero()};
  bool contact{false};
  Eigen::Vector2d cop{Eigen::Vector2d::Zero()};
  double normal_force{0.0};
};

inline std::array<Eigen::Vector2d, kNariTouchNodeCount>
NariTouchNodePositionsM() {
  return {
      Eigen::Vector2d{-0.003, -0.009}, Eigen::Vector2d{-0.003, -0.003},
      Eigen::Vector2d{-0.003, 0.003},  Eigen::Vector2d{-0.003, 0.009},
      Eigen::Vector2d{0.003, -0.009},  Eigen::Vector2d{0.003, -0.003},
      Eigen::Vector2d{0.003, 0.003},   Eigen::Vector2d{0.003, 0.009},
  };
}

inline Eigen::Vector3d NariTouchNodeSensorPosition(
    const NariTouchNodeState& node) {
  return Eigen::Vector3d{node.position_m.x(), node.position_m.y(), 0.0};
}

struct NariTouchState {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  NariTouchState() {
    const auto node_positions = NariTouchNodePositionsM();
    for (std::size_t i = 0; i < nodes.size(); ++i) {
      nodes[i].position_m = node_positions[i];
    }
  }

  Eigen::Vector3d slip_state{Eigen::Vector3d::Zero()};
  Eigen::Vector3d slip_velocity_state{Eigen::Vector3d::Zero()};
  double force_z{0.0};
  NariTouchContactState contact_state{NariTouchContactState::kNoContact};
  std::array<NariTouchNodeState, kNariTouchNodeCount> nodes{};

  bool hasContact() const {
    if (contact_state == NariTouchContactState::kFewContacts ||
        contact_state == NariTouchContactState::kEnoughContacts) {
      return true;
    }
    for (const auto& node : nodes) {
      if (node.contact) {
        return true;
      }
    }
    return false;
  }

  std::size_t contactNodeCount() const {
    std::size_t count = 0;
    for (const auto& node : nodes) {
      if (node.contact) {
        ++count;
      }
    }
    return count;
  }
};

constexpr int ToContactStateValue(NariTouchContactState state) {
  return static_cast<int>(state);
}

inline double ComputeNariTouchTotalNormalForceN(const NariTouchState& sensor) {
  double node_force_sum = 0.0;
  for (const auto& node : sensor.nodes) {
    if (std::isfinite(node.normal_force) && node.normal_force > 0.0) {
      node_force_sum += node.normal_force;
    }
  }
  if (node_force_sum > 0.0) {
    return node_force_sum;
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

inline bool ComputeNariTouchContactCentroidM(const NariTouchState& sensor,
                                             Eigen::Vector2d* centroid_m) {
  if (centroid_m == nullptr) {
    return false;
  }

  Eigen::Vector2d weighted_sum = Eigen::Vector2d::Zero();
  double weight_sum = 0.0;

  for (const auto& node : sensor.nodes) {
    if (!node.contact) {
      continue;
    }

    double weight = 1.0;
    if (std::isfinite(node.normal_force) && node.normal_force > 0.0) {
      weight = node.normal_force;
    }

    Eigen::Vector2d contact_position_m = node.position_m;
    if (node.cop.allFinite()) {
      contact_position_m += node.cop;
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
