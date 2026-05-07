//
// Copyright (c) 2026
//
// Full-joint acceleration bias block.
//

#ifndef __wbc_hqp_blocks_joint_accel_bias_block_hpp__
#define __wbc_hqp_blocks_joint_accel_bias_block_hpp__

#include <cassert>
#include <string>

#include "wbc_core/formulations/hqp/hqp-block-base.hpp"
#include "wbc_core/math/constraint-equality.hpp"

namespace tsid {

class JointAccelerationBias : public HQPBlock {
 public:
  JointAccelerationBias(const std::string& name, unsigned int level,
                        double weight)
      : HQPBlock(name, level, weight) {
    m_constraint = std::make_shared<math::ConstraintEquality>(name, 0, 0);
  }

  void setQddotBias(const math::Vector* qddot_bias) {
    m_qddot_bias = qddot_bias;
  }

  void build(const HQPBuildContext& ctx) override {
    assert(ctx.nv > 0);

    auto* eq = cst();
    eq->resize(ctx.nv, ctx.qpDim());
    eq->matrix().setZero();
    eq->matrix().leftCols(ctx.nv).setIdentity();
    eq->vector().setZero();

    if (!m_qddot_bias) {
      return;
    }

    assert(m_qddot_bias->size() == ctx.nv);
    eq->vector() = *m_qddot_bias;
    if (ctx.qddot_ref) {
      assert(ctx.qddot_ref->size() == ctx.nv);
      eq->vector() -= *ctx.qddot_ref;
    }
  }

 private:
  math::ConstraintEquality* cst() {
    assert(m_constraint);
    return static_cast<math::ConstraintEquality*>(m_constraint.get());
  }

  const math::Vector* m_qddot_bias{nullptr};
};

}  // namespace tsid

#endif  // ifndef __wbc_hqp_blocks_joint_accel_bias_block_hpp__
