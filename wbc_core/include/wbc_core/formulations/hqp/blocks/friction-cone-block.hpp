//
// Copyright (c) 2026
//
// Friction cone inequality constraint block.
//
// lb <= Uf * lambda <= ub
//

#ifndef __wbc_hqp_blocks_friction_cone_block_hpp__
#define __wbc_hqp_blocks_friction_cone_block_hpp__

#include <cassert>

#include "wbc_core/formulations/hqp/hqp-block-base.hpp"
#include "wbc_core/math/constraint-inequality.hpp"

namespace wbc {

class FrictionConeConstraint : public HQPBlock {
 public:
  FrictionConeConstraint(unsigned int level = 0)
      : HQPBlock("friction", level, 1.0) {
    m_constraint = std::make_shared<math::ConstraintInequality>(
        "friction", 0, 0);
  }

  void build(const HQPBuildContext& ctx) override {
    if (!ctx.Uf || !ctx.uf_lb || !ctx.uf_ub || ctx.lambdaDim == 0) {
      cst()->resize(0, ctx.qpDim());
      return;
    }

    assert(ctx.Uf->cols() == ctx.lambdaDim);
    assert(ctx.uf_lb->size() == ctx.Uf->rows());
    assert(ctx.uf_ub->size() == ctx.Uf->rows());

    const int nUf = static_cast<int>(ctx.Uf->rows());
    cst()->resize(nUf, ctx.qpDim());
    cst()->matrix().setZero();
    cst()->matrix().block(0, ctx.nv, nUf, ctx.lambdaDim) = *ctx.Uf;
    cst()->lowerBound() = *ctx.uf_lb;
    cst()->upperBound() = *ctx.uf_ub;
  }

 private:
  math::ConstraintInequality* cst() {
    assert(m_constraint);
    return static_cast<math::ConstraintInequality*>(m_constraint.get());
  }
};

}  // namespace wbc

#endif
