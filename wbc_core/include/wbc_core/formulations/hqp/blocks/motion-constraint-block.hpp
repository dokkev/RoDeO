//
// Copyright (c) 2026
//
// Generic motion constraint block.
//
// J * qddot = a_des
//
// If qddot_ref is provided (delta form):
//   J * delta_qddot = a_des - J * qddot_ref
//
// This block is controller-agnostic: it does not know whether it
// represents an operational task, a bias task, or something else.
// The controller assigns it to the appropriate HQP level.
//

#ifndef __wbc_hqp_blocks_motion_constraint_block_hpp__
#define __wbc_hqp_blocks_motion_constraint_block_hpp__

#include <cassert>

#include "wbc_core/controller/id-problem.hpp"
#include "wbc_core/formulations/hqp/hqp-block-base.hpp"
#include "wbc_core/math/constraint-equality.hpp"
#include "wbc_core/math/constraint-inequality.hpp"

namespace wbc {

class MotionConstraintBlock : public HQPBlock {
 public:
  MotionConstraintBlock(const std::string& name, unsigned int level,
                        double weight)
      : HQPBlock(name, level, weight) {
    m_constraint = std::make_shared<math::ConstraintEquality>(name, 0, 0);
  }

  void setConstraintRef(const MotionConstraintRef* constraint_ref) {
    m_constraintRef = constraint_ref;
  }

  void build(const HQPBuildContext& ctx) override {
    assert(m_constraintRef != nullptr);
    build(*m_constraintRef, ctx);
  }

  void build(const MotionConstraintRef& constraint_ref,
             const HQPBuildContext& ctx) {
    assert(constraint_ref.isValid());
    const auto& J = constraint_ref.matrix();
    const int rows = static_cast<int>(J.rows());

    assert(ctx.qpDim() >= ctx.nv);
    assert(J.cols() == ctx.nv);

    if (constraint_ref.isEquality()) {
      assert(constraint_ref.vector().size() == rows);
      auto* eq = ensureEquality();
      eq->resize(rows, ctx.qpDim());
      eq->matrix().setZero();
      eq->matrix().leftCols(ctx.nv) = J;

      if (ctx.qddot_ref) {
        assert(ctx.qddot_ref->size() == ctx.nv);
        eq->vector().noalias() = constraint_ref.vector() - J * (*ctx.qddot_ref);
      } else {
        eq->vector() = constraint_ref.vector();
      }
      return;
    }

    assert(constraint_ref.isInequality() || constraint_ref.isBound());
    assert(constraint_ref.lowerBound().size() == rows);
    assert(constraint_ref.upperBound().size() == rows);
    auto* ineq = ensureInequality();
    ineq->resize(rows, ctx.qpDim());
    ineq->matrix().setZero();
    ineq->matrix().leftCols(ctx.nv) = J;

    if (ctx.qddot_ref) {
      assert(ctx.qddot_ref->size() == ctx.nv);
      const auto JqddotRef = J * (*ctx.qddot_ref);
      ineq->lowerBound() = constraint_ref.lowerBound() - JqddotRef;
      ineq->upperBound() = constraint_ref.upperBound() - JqddotRef;
    } else {
      ineq->lowerBound() = constraint_ref.lowerBound();
      ineq->upperBound() = constraint_ref.upperBound();
    }
  }

 private:
  math::ConstraintEquality* ensureEquality() {
    if (!m_constraint || !m_constraint->isEquality()) {
      m_constraint = std::make_shared<math::ConstraintEquality>(name(), 0, 0);
    }
    return static_cast<math::ConstraintEquality*>(m_constraint.get());
  }

  math::ConstraintInequality* ensureInequality() {
    if (!m_constraint || !m_constraint->isInequality()) {
      m_constraint = std::make_shared<math::ConstraintInequality>(name(), 0, 0);
    }
    return static_cast<math::ConstraintInequality*>(m_constraint.get());
  }

  const MotionConstraintRef* m_constraintRef{nullptr};
};

}  // namespace wbc

#endif  // ifndef __wbc_hqp_blocks_motion_constraint_block_hpp__
