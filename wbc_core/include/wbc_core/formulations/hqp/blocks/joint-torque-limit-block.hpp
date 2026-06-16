//
// Copyright (c) 2026
//
// Joint torque limit inequality constraint block.
//
// tau = M_joint * qddot + h_joint - Jc_joint^T * T * lambda
// tau_lb <= tau <= tau_ub
//
// If qddot_ref is provided (delta form):
//   constant includes M_joint * qddot_ref
//
// hFext convention (same as FloatingBaseDynamicsConstraint):
//   M * qddot + h = S^T * tau + Jc^T * T * lambda + hFext
//   => tau = M_joint * qddot + h_joint - Jc_joint^T * T * lambda - hFext_joint
//

#ifndef WBC_CORE_FORMULATIONS_HQP_BLOCKS_JOINT_TORQUE_LIMIT_BLOCK_HPP_
#define WBC_CORE_FORMULATIONS_HQP_BLOCKS_JOINT_TORQUE_LIMIT_BLOCK_HPP_

#include <cassert>

#include "wbc_core/formulations/hqp/hqp-block-base.hpp"
#include "wbc_core/math/constraints/constraint-inequality.hpp"

namespace wbc {

class JointTorqueLimitConstraint : public HQPBlock {
 public:
  JointTorqueLimitConstraint(unsigned int level = 0)
      : HQPBlock("joint-torque-limit", level, 1.0) {
    m_constraint = std::make_shared<math::ConstraintInequality>(
        "joint-torque-limit", 0, 0);
  }

  void build(const HQPBuildContext& ctx) override {
    if (!ctx.hasJointTorqueLimits()) {
      cst()->resize(0, ctx.qpDim());
      return;
    }

    assert(ctx.hasDynamics());
    assert(ctx.M->rows() == ctx.nv);
    assert(ctx.M->cols() == ctx.nv);
    assert(ctx.h->size() >= ctx.nv);
    assert(ctx.tau_lb->size() == ctx.na);
    assert(ctx.tau_ub->size() == ctx.na);
    assert(ctx.qpDim() >= ctx.nv);

    const auto& M = *ctx.M;
    const auto& h = *ctx.h;

    cst()->resize(ctx.na, ctx.qpDim());
    cst()->matrix().setZero();

    // M_joint * qddot
    cst()->matrix().leftCols(ctx.nv) = M.bottomRows(ctx.na);

    // -Jc_joint^T * T * lambda
    if (ctx.lambdaDim > 0) {
      int lambdaCol = ctx.nv;
      for (const auto& ci : ctx.contactInfos) {
        assert(ci.Jc != nullptr);
        assert(ci.T != nullptr);
        assert(ci.Jc->cols() == ctx.nv);
        assert(ci.T->cols() == ci.lambdaDim);
        cst()->matrix()
            .block(0, lambdaCol, ctx.na, ci.lambdaDim)
            .noalias() = -ci.Jc->transpose().bottomRows(ctx.na) * (*ci.T);
        lambdaCol += ci.lambdaDim;
      }
      assert(lambdaCol == ctx.nv + ctx.lambdaDim);
    }

    // Constant: h_joint (+ M_joint * qddot_ref if delta form)
    m_cTmp = h.tail(ctx.na);
    if (ctx.qddot_ref) {
      assert(ctx.qddot_ref->size() == ctx.nv);
      m_cTmp.noalias() += M.bottomRows(ctx.na) * (*ctx.qddot_ref);
    }
    if (ctx.h_ext) {
      m_cTmp -= ctx.h_ext->tail(ctx.na);
    }

    cst()->lowerBound() = *ctx.tau_lb - m_cTmp;
    cst()->upperBound() = *ctx.tau_ub - m_cTmp;
  }

  /// Pre-allocate scratch buffer for na-sized constant vector.
  void preallocate(int na) { m_cTmp.resize(na); }

 private:
  math::ConstraintInequality* cst() {
    assert(m_constraint);
    return static_cast<math::ConstraintInequality*>(m_constraint.get());
  }

  math::Vector m_cTmp;  ///< Scratch: joint torque limit constant (na)
};

}  // namespace wbc

#endif  // WBC_CORE_FORMULATIONS_HQP_BLOCKS_JOINT_TORQUE_LIMIT_BLOCK_HPP_
