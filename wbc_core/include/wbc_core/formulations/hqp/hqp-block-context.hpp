//
// Copyright (c) 2026
//
// Controller-agnostic context consumed by HQP blocks.
// Populated by the controller each cycle, consumed by reusable blocks.
//

#ifndef __wbc_formulations_hqp_block_context_hpp__
#define __wbc_formulations_hqp_block_context_hpp__

#include <vector>

#include "wbc_core/math/fwd.hpp"

namespace wbc {

struct HQPBlockContext {
  // Robot dynamics required by dynamics and torque-limit blocks.
  const math::Matrix* M{nullptr};
  const math::Vector* h{nullptr};

  // Decision dimensions.
  int nv{0};          ///< Total velocity DOFs
  int na{0};          ///< Actuated DOFs
  int nvFloat{0};     ///< Floating-base DOFs (0 or 6)
  int lambdaDim{0};   ///< Total contact-force decision dimension

  /// Decision variable dimension. Always nv + lambdaDim.
  int qpDim() const { return nv + lambdaDim; }

  // Per-contact info for Jc^T * T column assembly.
  struct ContactInfo {
    const math::Matrix* Jc{nullptr};
    const math::Matrix* T{nullptr};
    int lambdaOffset{0};
    int lambdaDim{0};
  };
  std::vector<ContactInfo> contactInfos;

  // Aggregated contact data required by contact-aware blocks.
  const math::Matrix* Jc{nullptr};
  const math::Vector* contact_motion_rhs{nullptr};
  const math::Matrix* Uf{nullptr};
  const math::Vector* uf_lb{nullptr};
  const math::Vector* uf_ub{nullptr};

  // Optional external generalized force contribution.
  const math::Vector* h_ext{nullptr};

  // Optional torque bounds.
  bool enableJointTorqueLimits{false};
  const math::Vector* tau_lb{nullptr};
  const math::Vector* tau_ub{nullptr};

  // Optional reference acceleration for delta formulations.
  /// If non-null, blocks use delta form: x = [delta_qddot, lambda]
  /// Dynamics RHS becomes -(M * qddot_ref + h)
  /// Task RHS becomes a_des - J * qddot_ref
  const math::Vector* qddot_ref{nullptr};

  // Validation helpers.
  bool hasDynamics() const { return M != nullptr && h != nullptr; }
  bool hasContacts() const {
    return lambdaDim > 0 && Jc != nullptr && contact_motion_rhs != nullptr &&
           Uf != nullptr && uf_lb != nullptr && uf_ub != nullptr;
  }
  bool hasJointTorqueLimits() const {
    return enableJointTorqueLimits && tau_lb != nullptr && tau_ub != nullptr;
  }
  bool hasReferenceAcceleration() const { return qddot_ref != nullptr; }
};

}  // namespace wbc

#endif  // ifndef __wbc_formulations_hqp_block_context_hpp__
