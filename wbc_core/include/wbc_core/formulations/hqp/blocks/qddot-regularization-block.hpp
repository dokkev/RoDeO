//
// Copyright (c) 2026
//
// Acceleration minimization regularization block.
//
// ||qddot|| -> 0  (or ||delta_qddot|| -> 0)
//

#ifndef __wbc_hqp_blocks_qddot_regularization_block_hpp__
#define __wbc_hqp_blocks_qddot_regularization_block_hpp__

#include <cassert>

#include "wbc_core/formulations/hqp/hqp-block-base.hpp"
#include "wbc_core/math/constraints/constraint-equality.hpp"

namespace wbc {

class AccelerationRegularization : public HQPBlock {
 public:
  AccelerationRegularization(unsigned int level, double weight)
      : HQPBlock("qddot-reg", level, weight) {
    m_constraint = std::make_shared<math::ConstraintEquality>(
        "qddot-reg", 0, 0);
  }

  void build(const HQPBlockContext& ctx) override {
    assert(ctx.nv > 0);

    auto* eq = cst();
    eq->resize(ctx.nv, ctx.qpDim());
    eq->matrix().setZero();
    eq->matrix().leftCols(ctx.nv).setIdentity();
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
