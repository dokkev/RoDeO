//
// Copyright (c) 2026
//
// Generic motion task block.
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

#ifndef __wbc_hqp_blocks_motion_task_block_hpp__
#define __wbc_hqp_blocks_motion_task_block_hpp__

#include <cassert>

#include "wbc_core/controller/wbmc-step-input.hpp"
#include "wbc_core/formulations/hqp/hqp-block-base.hpp"
#include "wbc_core/math/constraint-equality.hpp"
#include "wbc_core/math/constraint-inequality.hpp"

namespace tsid {

class MotionTask : public HQPBlock {
 public:
  MotionTask(const std::string& name, unsigned int level, double weight)
      : HQPBlock(name, level, weight) {
    m_constraint = std::make_shared<math::ConstraintEquality>(
        name, 0, 0);
  }

  void setTask(const MotionObjective* task) { m_task = task; }

  void build(const HQPBuildContext& ctx) override {
    assert(m_task != nullptr);
    build(*m_task, ctx);
  }

  void build(const MotionObjective& task, const HQPBuildContext& ctx) {
    const auto& J = task.J;
    const int rows = static_cast<int>(J.rows());

    assert(ctx.qpDim() >= ctx.nv);
    assert(J.cols() == ctx.nv);

    if (task.isEquality()) {
      assert(task.a_des.size() == rows);
      auto* eq = ensureEquality();
      eq->resize(rows, ctx.qpDim());
      eq->matrix().setZero();
      eq->matrix().leftCols(ctx.nv) = J;

      if (ctx.qddot_ref) {
        assert(ctx.qddot_ref->size() == ctx.nv);
        eq->vector().noalias() = task.a_des - J * (*ctx.qddot_ref);
      } else {
        eq->vector() = task.a_des;
      }
      return;
    }

    assert(task.lower_bound.size() == rows);
    assert(task.upper_bound.size() == rows);
    auto* ineq = ensureInequality();
    ineq->resize(rows, ctx.qpDim());
    ineq->matrix().setZero();
    ineq->matrix().leftCols(ctx.nv) = J;

    if (ctx.qddot_ref) {
      assert(ctx.qddot_ref->size() == ctx.nv);
      const auto JqddotRef = J * (*ctx.qddot_ref);
      ineq->lowerBound() = task.lower_bound - JqddotRef;
      ineq->upperBound() = task.upper_bound - JqddotRef;
    } else {
      ineq->lowerBound() = task.lower_bound;
      ineq->upperBound() = task.upper_bound;
    }
  }

 private:
  math::ConstraintEquality* ensureEquality() {
    if (!m_constraint || !m_constraint->isEquality()) {
      m_constraint = std::make_shared<math::ConstraintEquality>(
          name(), 0, 0);
    }
    return static_cast<math::ConstraintEquality*>(m_constraint.get());
  }

  math::ConstraintInequality* ensureInequality() {
    if (!m_constraint || !m_constraint->isInequality()) {
      m_constraint = std::make_shared<math::ConstraintInequality>(
          name(), 0, 0);
    }
    return static_cast<math::ConstraintInequality*>(m_constraint.get());
  }

  const MotionObjective* m_task{nullptr};
};

}  // namespace tsid

#endif
