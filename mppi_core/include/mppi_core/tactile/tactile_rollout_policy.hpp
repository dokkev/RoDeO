// Copyright 2026
//
// Licensed under the Apache License, Version 2.0.

#pragma once

namespace mppi_core {

enum class TactileRolloutPolicy {
  // Contact-gated runtime mode. Force-aware tactile rollout must succeed;
  // otherwise the rollout step is invalid.
  kForceAwareRequired = 0,
  // Debug/ablation mode: use force-aware rollout first, then fall back to
  // kinematic contact-patch rollout. No heuristic proxy fallback exists.
  kForceThenKinematicFallback = 1,
};

}  // namespace mppi_core
