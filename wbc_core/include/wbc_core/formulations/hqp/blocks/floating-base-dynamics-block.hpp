//
// Copyright (c) 2026
//
// Floating-base dynamics equality constraint block.
//
// Sf * (M * qddot - Jc^T * T * lambda) = -Sf * h
//
// If qddot_ref is provided (delta form):
//   Sf * (M * delta_qddot - Jc^T * T * lambda) = -Sf * (M * qddot_ref + h)
//

#ifndef __wbc_hqp_blocks_floating_base_dynamics_block_hpp__
#define __wbc_hqp_blocks_floating_base_dynamics_block_hpp__

#include <cassert>

#include "wbc_core/formulations/hqp/hqp-block-base.hpp"
#include "wbc_core/math/constraints/constraint-equality.hpp"

namespace wbc {

class FloatingBaseDynamicsConstraint : public HQPBlock {
 public:
  FloatingBaseDynamicsConstraint(unsigned int level = 0)
      : HQPBlock("dynamics", level, 1.0) {
    m_constraint = std::make_shared<math::ConstraintEquality>(
        "dynamics", 0, 0);
  }

  void build(const HQPBlockContext& ctx) override {
    if (ctx.nvFloat == 0) {
      cst()->resize(0, ctx.qpDim());
      return;
    }

    assert(ctx.hasDynamics());
    assert(ctx.M->rows() >= ctx.nvFloat);
    assert(ctx.M->cols() == ctx.nv);
    assert(ctx.h->size() >= ctx.nvFloat);

    const auto& M = *ctx.M;
    const auto& h = *ctx.h;

    cst()->resize(ctx.nvFloat, ctx.qpDim());
    cst()->matrix().setZero();

    // Sf * M * qddot
    cst()->matrix().leftCols(ctx.nv) = M.topRows(ctx.nvFloat);

    // -Sf * Jc^T * T * lambda
    if (ctx.lambdaDim > 0) {
      int lambdaCol = ctx.nv;
      for (const auto& ci : ctx.contactInfos) {
        assert(ci.Jc != nullptr);
        assert(ci.T != nullptr);
        assert(ci.Jc->cols() == ctx.nv);
        assert(ci.T->cols() == ci.lambdaDim);
        cst()->matrix()
            .block(0, lambdaCol, ctx.nvFloat, ci.lambdaDim)
            .noalias() = -ci.Jc->transpose().topRows(ctx.nvFloat) * (*ci.T);
        lambdaCol += ci.lambdaDim;
      }
      assert(lambdaCol == ctx.nv + ctx.lambdaDim);
    }

    // RHS
    if (ctx.qddot_ref) {
      assert(ctx.qddot_ref->size() == ctx.nv);
      cst()->vector().noalias() =
          -(M.topRows(ctx.nvFloat) * (*ctx.qddot_ref));
      cst()->vector() -= h.head(ctx.nvFloat);
    } else {
      cst()->vector() = -h.head(ctx.nvFloat);
    }

    if (ctx.h_ext) {
      cst()->vector() += ctx.h_ext->head(ctx.nvFloat);
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
