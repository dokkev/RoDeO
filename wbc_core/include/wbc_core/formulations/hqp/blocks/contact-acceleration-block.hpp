//
// Copyright (c) 2026
//
// Contact acceleration hard equality constraint block.
//
// Enforces stacked rigid-contact motion consistency:
//   Jc * qddot = -Jcdot_qdot
//
// For stationary rigid contacts, Jcdot_qdot is the usual drift term.
//
// If qddot_ref is provided (delta form):
//   Jc * delta_qddot = -(Jcdot_qdot + Jc * qddot_ref)
//

#ifndef __wbc_hqp_blocks_contact_acceleration_block_hpp__
#define __wbc_hqp_blocks_contact_acceleration_block_hpp__

#include <cassert>

#include "wbc_core/formulations/hqp/hqp-block-base.hpp"
#include "wbc_core/math/constraint-equality.hpp"

namespace tsid {

class ContactConsistencyConstraint : public HQPBlock {
 public:
  ContactConsistencyConstraint(unsigned int level = 0)
      : HQPBlock("contact-accel", level, 1.0) {
    m_constraint = std::make_shared<math::ConstraintEquality>(
        "contact-accel", 0, 0);
  }

  void build(const HQPBuildContext& ctx) override {
    if (!ctx.Jc || !ctx.Jcdot_qdot || ctx.Jc->rows() == 0) {
      cst()->resize(0, ctx.qpDim());
      return;
    }

    assert(ctx.Jc->cols() == ctx.nv);
    assert(ctx.Jcdot_qdot->size() == ctx.Jc->rows());

    const int nMotion = static_cast<int>(ctx.Jc->rows());
    cst()->resize(nMotion, ctx.qpDim());
    cst()->matrix().setZero();
    cst()->matrix().leftCols(ctx.nv) = *ctx.Jc;

    if (ctx.qddot_ref) {
      assert(ctx.qddot_ref->size() == ctx.nv);
      cst()->vector().noalias() =
          -(*ctx.Jcdot_qdot) - (*ctx.Jc) * (*ctx.qddot_ref);
    } else {
      cst()->vector() = -(*ctx.Jcdot_qdot);
    }
  }

 private:
  math::ConstraintEquality* cst() {
    assert(m_constraint);
    return static_cast<math::ConstraintEquality*>(m_constraint.get());
  }
};

}  // namespace tsid

#endif
