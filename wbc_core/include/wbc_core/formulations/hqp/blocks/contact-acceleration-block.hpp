//
// Copyright (c) 2026
//
// Contact acceleration hard equality constraint block.
//
// Enforces stacked rigid-contact motion consistency:
//   Jc * qddot = contact_motion_rhs
//
// If qddot_ref is provided (delta form):
//   Jc * delta_qddot = contact_motion_rhs - Jc * qddot_ref
//

#ifndef __wbc_hqp_blocks_contact_acceleration_block_hpp__
#define __wbc_hqp_blocks_contact_acceleration_block_hpp__

#include <cassert>

#include "wbc_core/formulations/hqp/hqp-block-base.hpp"
#include "wbc_core/math/constraints/constraint-equality.hpp"

namespace wbc {

class ContactConsistencyConstraint : public HQPBlock {
 public:
  ContactConsistencyConstraint(unsigned int level = 0)
      : HQPBlock("contact-accel", level, 1.0) {
    m_constraint = std::make_shared<math::ConstraintEquality>(
        "contact-accel", 0, 0);
  }

  void build(const HQPBlockContext& ctx) override {
    if (!ctx.Jc || !ctx.contact_motion_rhs || ctx.Jc->rows() == 0) {
      cst()->resize(0, ctx.qpDim());
      return;
    }

    assert(ctx.Jc->cols() == ctx.nv);
    assert(ctx.contact_motion_rhs->size() == ctx.Jc->rows());

    const int nMotion = static_cast<int>(ctx.Jc->rows());
    cst()->resize(nMotion, ctx.qpDim());
    cst()->matrix().setZero();
    cst()->matrix().leftCols(ctx.nv) = *ctx.Jc;

    if (ctx.qddot_ref) {
      assert(ctx.qddot_ref->size() == ctx.nv);
      cst()->vector().noalias() =
          *ctx.contact_motion_rhs - (*ctx.Jc) * (*ctx.qddot_ref);
    } else {
      cst()->vector() = *ctx.contact_motion_rhs;
    }
  }

 private:
  math::ConstraintEquality* cst() {
    assert(m_constraint);
    return static_cast<math::ConstraintEquality*>(m_constraint.get());
  }
};

}  // namespace wbc

#endif
