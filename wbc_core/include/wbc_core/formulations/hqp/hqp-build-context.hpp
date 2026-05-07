//
// Copyright (c) 2026
//
// Controller-agnostic context for HQP block construction.
// Populated by the controller each cycle, consumed by reusable blocks.
//

#ifndef __wbc_formulations_hqp_build_context_hpp__
#define __wbc_formulations_hqp_build_context_hpp__

#include <vector>

#include "wbc_core/math/fwd.hpp"

namespace tsid {

struct HQPBuildContext {
  // ── Robot dynamics (required by dynamics / torque-limit blocks) ──────
  const math::Matrix* M{nullptr};
  const math::Vector* h{nullptr};

  // ── Dimensions ────────────────────────────────────────────────────────
  int nv{0};          ///< Total velocity DOFs
  int na{0};          ///< Actuated DOFs
  int nvFloat{0};     ///< Floating-base DOFs (0 or 6)
  int lambdaDim{0};   ///< Total contact-force decision dimension

  /// Decision variable dimension. Always nv + lambdaDim.
  int qpDim() const { return nv + lambdaDim; }

  // ── Per-contact info (for Jc^T * T column assembly) ───────────────────
  struct ContactInfo {
    const math::Matrix* Jc{nullptr};
    const math::Matrix* T{nullptr};
    int lambdaOffset{0};
    int lambdaDim{0};
  };
  std::vector<ContactInfo> contactInfos;

  // ── Aggregated contact data (required by contact-aware blocks) ────────
  const math::Matrix* Jc{nullptr};
  const math::Vector* Jcdot_qdot{nullptr};
  const math::Matrix* Uf{nullptr};
  const math::Vector* uf_lb{nullptr};
  const math::Vector* uf_ub{nullptr};

  // ── Optional external generalized force contribution ──────────────────
  const math::Vector* h_ext{nullptr};

  // ── Optional torque bounds ────────────────────────────────────────────
  bool enableTorqueLimits{false};
  const math::Vector* tau_lb{nullptr};
  const math::Vector* tau_ub{nullptr};

  // ── Optional reference acceleration for delta formulations ────────────
  /// If non-null, blocks use delta form: x = [delta_qddot, lambda]
  /// Dynamics RHS becomes -(M * qddot_ref + h)
  /// Task RHS becomes a_des - J * qddot_ref
  const math::Vector* qddot_ref{nullptr};

  // ── Validation helpers ────────────────────────────────────────────────
  bool hasDynamics() const { return M != nullptr && h != nullptr; }
  bool hasContacts() const {
    return lambdaDim > 0 && Jc != nullptr && Jcdot_qdot != nullptr &&
           Uf != nullptr && uf_lb != nullptr && uf_ub != nullptr;
  }
  bool hasTorqueLimits() const {
    return enableTorqueLimits && tau_lb != nullptr && tau_ub != nullptr;
  }
  bool hasReferenceAcceleration() const { return qddot_ref != nullptr; }
};

}  // namespace tsid

#endif  // ifndef __wbc_formulations_hqp_build_context_hpp__
