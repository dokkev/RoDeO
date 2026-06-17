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

#include "wbc_core/formulations/hqp/hqp-block-base.hpp"
#include "wbc_core/math/constraints/constraint-equality.hpp"
#include "wbc_core/math/constraints/constraint-inequality.hpp"
#include "wbc_core/tasks/motion-objective.hpp"

namespace wbc {

class MotionConstraintBlock : public HQPBlock {
 public:
  MotionConstraintBlock(const std::string& name, unsigned int level,
                        double weight)
      : HQPBlock(name, level, weight) {
    m_constraint = std::make_shared<math::ConstraintEquality>(name, 0, 0);
  }

  void setObjective(const MotionObjective* objective) {
    m_objective = objective;
  }

  void build(const HQPBlockContext& ctx) override {
    assert(m_objective != nullptr);
    build(*m_objective, ctx);
  }

  void build(const MotionObjective& objective, const HQPBlockContext& ctx) {
    assert(objective.isValid());
    const auto& J = objective.matrix();
    const int rows = static_cast<int>(J.rows());

    assert(ctx.qpDim() >= ctx.nv);
    assert(J.cols() == ctx.nv);

    if (objective.isEquality()) {
      assert(objective.vector().size() == rows);
      auto* eq = ensureEquality();
      eq->resize(rows, ctx.qpDim());
      eq->matrix().setZero();
      eq->matrix().leftCols(ctx.nv) = J;

      if (ctx.qddot_ref) {
        assert(ctx.qddot_ref->size() == ctx.nv);
        eq->vector().noalias() = objective.vector() - J * (*ctx.qddot_ref);
      } else {
        eq->vector() = objective.vector();
      }
      return;
    }

    assert(objective.isInequality() || objective.isBound());
    assert(objective.lowerBound().size() == rows);
    assert(objective.upperBound().size() == rows);
    auto* ineq = ensureInequality();
    ineq->resize(rows, ctx.qpDim());
    ineq->matrix().setZero();
    ineq->matrix().leftCols(ctx.nv) = J;

    if (ctx.qddot_ref) {
      assert(ctx.qddot_ref->size() == ctx.nv);
      const auto JqddotRef = J * (*ctx.qddot_ref);
      ineq->lowerBound() = objective.lowerBound() - JqddotRef;
      ineq->upperBound() = objective.upperBound() - JqddotRef;
    } else {
      ineq->lowerBound() = objective.lowerBound();
      ineq->upperBound() = objective.upperBound();
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

  const MotionObjective* m_objective{nullptr};
};

}  // namespace wbc

#endif  // ifndef __wbc_hqp_blocks_motion_constraint_block_hpp__
