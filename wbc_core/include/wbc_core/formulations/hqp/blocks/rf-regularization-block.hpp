//
// Copyright (c) 2026
//
// Contact-force regularization block.
//
// ||lambda|| -> 0
//

#ifndef __wbc_hqp_blocks_rf_regularization_block_hpp__
#define __wbc_hqp_blocks_rf_regularization_block_hpp__

#include <cassert>

#include "wbc_core/formulations/hqp/hqp-block-base.hpp"
#include "wbc_core/math/constraints/constraint-equality.hpp"

namespace wbc {

class ContactForceRegularization : public HQPBlock {
 public:
  ContactForceRegularization(unsigned int level, double weight)
      : HQPBlock("rf-reg", level, weight) {
    m_constraint = std::make_shared<math::ConstraintEquality>(
        "rf-reg", 0, 0);
  }

  void build(const HQPBuildContext& ctx) override {
    auto* eq = cst();
    if (ctx.lambdaDim == 0) {
      eq->resize(0, ctx.qpDim());
      return;
    }

    eq->resize(ctx.lambdaDim, ctx.qpDim());
    eq->matrix().setZero();
    eq->matrix().block(0, ctx.nv, ctx.lambdaDim, ctx.lambdaDim).setIdentity();
    eq->vector().setZero();
  }

 private:
  math::ConstraintEquality* cst() {
    assert(m_constraint);
    return static_cast<math::ConstraintEquality*>(m_constraint.get());
  }
};

}  // namespace wbc

#endif
