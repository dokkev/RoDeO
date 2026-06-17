//
// Copyright (c) 2026
//
// Deterministic math validation tests for model-independent WBC core helpers.
//

#include <gtest/gtest.h>

#include <Eigen/SVD>

#include <wbc_core/formulations/hqp/blocks/contact-acceleration-block.hpp>
#include <wbc_core/formulations/hqp/blocks/floating-base-dynamics-block.hpp>
#include <wbc_core/formulations/hqp/blocks/friction-cone-block.hpp>
#include <wbc_core/formulations/hqp/blocks/joint-torque-limit-block.hpp>
#include <wbc_core/formulations/hqp/blocks/motion-constraint-block.hpp>
#include <wbc_core/math/constraints/constraint-bound.hpp>
#include <wbc_core/math/constraints/constraint-equality.hpp>
#include <wbc_core/math/constraints/constraint-inequality.hpp>
#include <wbc_core/math/linear_algebra/selection.hpp>
#include <wbc_core/math/linear_algebra/svd.hpp>

namespace
{

using wbc::HQPBuildContext;
using wbc::math::ConstraintBound;
using wbc::math::ConstraintEquality;
using wbc::math::ConstraintInequality;
using wbc::math::Matrix;
using wbc::math::Vector;
using wbc::math::VectorXi;

constexpr double kTol = 1e-10;

template<typename ActualDerived, typename ExpectedDerived>
void expectEigenNear(
  const Eigen::MatrixBase<ActualDerived> & actual,
  const Eigen::MatrixBase<ExpectedDerived> & expected,
  double tolerance = kTol)
{
  ASSERT_EQ(actual.rows(), expected.rows());
  ASSERT_EQ(actual.cols(), expected.cols());
  EXPECT_TRUE(actual.isApprox(expected, tolerance))
      << "actual:\n" << actual << "\nexpected:\n" << expected;
}

HQPBuildContext makeDecisionContext(int nv, int lambdaDim)
{
  HQPBuildContext ctx;
  ctx.nv = nv;
  ctx.na = nv;
  ctx.lambdaDim = lambdaDim;
  return ctx;
}

TEST(MathValidationTest, EqualityConstraintChecksLinearResidual) {
  Matrix A(2, 3);
  A << 1.0, 2.0, -1.0,
       0.0, -3.0, 4.0;
  Vector b(2);
  b << 0.5, 1.25;
  ConstraintEquality constraint("equality", A, b);

  Vector x(3);
  x << 0.5, 0.25, 0.5;

  ASSERT_TRUE((A * x).isApprox(b, kTol));
  EXPECT_TRUE(constraint.isSatisfied(x, kTol));

  x(2) += 1e-4;
  EXPECT_FALSE(constraint.isSatisfied(x, kTol));
}

TEST(MathValidationTest, InequalityAndBoundConstraintsCheckBothSides) {
  Matrix A(2, 2);
  A << 1.0, -2.0,
       -1.0, 1.0;
  Vector lower(2);
  lower << -1.0, -0.5;
  Vector upper(2);
  upper << 2.0, 1.0;
  ConstraintInequality inequality("ineq", A, lower, upper);

  Vector x(2);
  x << 0.0, 0.4;
  EXPECT_TRUE(inequality.isSatisfied(x, kTol));

  x << -1.0, 1.0;
  EXPECT_FALSE(inequality.isSatisfied(x, kTol));

  Vector lb(3);
  lb << -2.0, -1.0, 0.5;
  Vector ub(3);
  ub << 2.0, 3.0, 1.5;
  ConstraintBound bound("bound", lb, ub);

  Vector y(3);
  y << -2.0, 1.25, 1.5;
  EXPECT_TRUE(bound.isSatisfied(y, kTol));

  y(2) = 1.5001;
  EXPECT_FALSE(bound.isSatisfied(y, kTol));
}

TEST(MathValidationTest, WeightedSelectionMatrixUsesMaskOffsetAndWeights) {
  Vector mask(4);
  mask << 1.0, 0.0, 1.0, 1.0;
  Vector weights(4);
  weights << 2.0, 10.0, -3.0, 0.5;

  VectorXi active_indices;
  Matrix selection;
  wbc::math::buildWeightedSelectionMatrix(mask, 6, 1, weights, active_indices,
                                          selection);

  VectorXi expected_indices(3);
  expected_indices << 0, 2, 3;
  Matrix expected = Matrix::Zero(3, 6);
  expected(0, 1) = 2.0;
  expected(1, 3) = -3.0;
  expected(2, 4) = 0.5;

  expectEigenNear(active_indices.cast<double>(), expected_indices.cast<double>());
  expectEigenNear(selection, expected);
}

TEST(MathValidationTest, DampedSvdSolveMatchesDiagonalClosedForm) {
  Matrix A = Matrix::Zero(2, 2);
  A(0, 0) = 2.0;
  A(1, 1) = 0.5;
  Vector b(2);
  b << 4.0, 1.0;

  Vector undamped(2);
  wbc::math::solveDamped(A, b, undamped, 0.0);
  Vector expected_undamped(2);
  expected_undamped << 2.0, 2.0;
  expectEigenNear(undamped, expected_undamped);

  Vector damped(2);
  wbc::math::solveDamped(A, b, damped, 1.0);
  Vector expected_damped(2);
  expected_damped << 1.6, 0.4;
  expectEigenNear(damped, expected_damped);
}

TEST(MathValidationTest, PseudoInverseSatisfiesMoorePenroseProjection) {
  Matrix A(2, 3);
  A << 1.0, 2.0, 0.0,
       0.0, 1.0, 1.0;
  Matrix A_pinv(3, 2);

  wbc::math::pseudoInverse(A, A_pinv, 1e-12);

  expectEigenNear(A * A_pinv * A, A);
  expectEigenNear(A_pinv * A * A_pinv, A_pinv);
}

TEST(MathValidationTest, MotionEqualityBlockBuildsDeltaFormRhs) {
  Matrix J(2, 3);
  J << 1.0, 2.0, 0.0,
       0.0, -1.0, 3.0;
  Vector target(2);
  target << 0.5, -1.0;
  ConstraintEquality task_constraint("task", J, target);
  wbc::MotionObjective objective("task", &task_constraint, 1u, 1.0);

  Vector qddot_ref(3);
  qddot_ref << 0.1, 0.2, -0.3;
  HQPBuildContext ctx = makeDecisionContext(3, 2);
  ctx.qddot_ref = &qddot_ref;

  wbc::MotionConstraintBlock block("task", 1u, 1.0);
  block.build(objective, ctx);

  const auto & constraint = *block.constraint();
  ASSERT_TRUE(constraint.isEquality());

  Matrix expected_matrix = Matrix::Zero(2, 5);
  expected_matrix.leftCols(3) = J;
  const Vector expected_rhs = target - J * qddot_ref;

  expectEigenNear(constraint.matrix(), expected_matrix);
  expectEigenNear(constraint.vector(), expected_rhs);
}

TEST(MathValidationTest, MotionInequalityBlockOffsetsBoundsByReference) {
  Matrix A(1, 3);
  A << 2.0, -1.0, 0.5;
  Vector lower(1);
  lower << -0.5;
  Vector upper(1);
  upper << 1.5;
  ConstraintInequality task_constraint("limit", A, lower, upper);
  wbc::MotionObjective objective("limit", &task_constraint, 0u, 1.0);

  Vector qddot_ref(3);
  qddot_ref << 0.25, -0.5, 0.75;
  HQPBuildContext ctx = makeDecisionContext(3, 1);
  ctx.qddot_ref = &qddot_ref;

  wbc::MotionConstraintBlock block("limit", 0u, 1.0);
  block.build(objective, ctx);

  const auto & constraint = *block.constraint();
  ASSERT_TRUE(constraint.isInequality());

  Matrix expected_matrix = Matrix::Zero(1, 4);
  expected_matrix.leftCols(3) = A;
  const Vector shift = A * qddot_ref;

  expectEigenNear(constraint.matrix(), expected_matrix);
  expectEigenNear(constraint.lowerBound(), lower - shift);
  expectEigenNear(constraint.upperBound(), upper - shift);
}

TEST(MathValidationTest, ContactAccelerationBlockBuildsStackedDeltaConstraint) {
  Matrix Jc(2, 3);
  Jc << 1.0, 0.0, -1.0,
        0.5, 2.0, 0.0;
  Vector contact_rhs(2);
  contact_rhs << 0.25, -0.75;
  Vector qddot_ref(3);
  qddot_ref << -0.2, 0.1, 0.4;

  HQPBuildContext ctx = makeDecisionContext(3, 1);
  ctx.Jc = &Jc;
  ctx.contact_motion_rhs = &contact_rhs;
  ctx.qddot_ref = &qddot_ref;

  wbc::ContactConsistencyConstraint block;
  block.build(ctx);

  const auto & constraint = *block.constraint();
  ASSERT_TRUE(constraint.isEquality());

  Matrix expected_matrix = Matrix::Zero(2, 4);
  expected_matrix.leftCols(3) = Jc;
  const Vector expected_rhs = contact_rhs - Jc * qddot_ref;

  expectEigenNear(constraint.matrix(), expected_matrix);
  expectEigenNear(constraint.vector(), expected_rhs);
}

TEST(MathValidationTest, FrictionConeBlockPlacesUfAtLambdaColumns) {
  Matrix Uf(3, 2);
  Uf << 1.0, 0.0,
        -1.0, 0.2,
        0.0, 1.0;
  Vector lower(3);
  lower << 0.0, -2.0, 0.5;
  Vector upper(3);
  upper << 4.0, 2.0, 3.0;

  HQPBuildContext ctx = makeDecisionContext(3, 2);
  ctx.Uf = &Uf;
  ctx.uf_lb = &lower;
  ctx.uf_ub = &upper;

  wbc::FrictionConeConstraint block;
  block.build(ctx);

  const auto & constraint = *block.constraint();
  ASSERT_TRUE(constraint.isInequality());

  Matrix expected_matrix = Matrix::Zero(3, 5);
  expected_matrix.block(0, 3, 3, 2) = Uf;

  expectEigenNear(constraint.matrix(), expected_matrix);
  expectEigenNear(constraint.lowerBound(), lower);
  expectEigenNear(constraint.upperBound(), upper);
}

TEST(MathValidationTest, FloatingBaseDynamicsBlockUsesContactAndDeltaSigns) {
  Matrix M(4, 4);
  M << 2.0, 0.1, -0.3, 0.2,
       0.1, 3.0, 0.4, -0.5,
       -0.3, 0.4, 4.0, 0.6,
       0.2, -0.5, 0.6, 5.0;
  Vector h(4);
  h << 1.0, 2.0, 3.0, 4.0;
  Vector qddot_ref(4);
  qddot_ref << 0.1, -0.2, 0.3, -0.4;
  Vector h_ext(4);
  h_ext << 0.5, -0.25, 0.75, -1.0;
  Matrix Jc(3, 4);
  Jc << 1.0, 0.0, 2.0, -1.0,
        0.0, 1.0, -2.0, 0.0,
        1.0, 1.0, 0.0, 0.0;
  Matrix T(3, 2);
  T << 1.0, 0.5,
       -0.25, 1.0,
       2.0, -1.0;

  HQPBuildContext ctx = makeDecisionContext(4, 2);
  ctx.na = 2;
  ctx.nvFloat = 2;
  ctx.M = &M;
  ctx.h = &h;
  ctx.qddot_ref = &qddot_ref;
  ctx.h_ext = &h_ext;
  ctx.contactInfos.push_back({&Jc, &T, 0, 2});

  wbc::FloatingBaseDynamicsConstraint block;
  block.build(ctx);

  const auto & constraint = *block.constraint();
  ASSERT_TRUE(constraint.isEquality());

  Matrix expected_matrix = Matrix::Zero(2, 6);
  expected_matrix.leftCols(4) = M.topRows(2);
  expected_matrix.block(0, 4, 2, 2).noalias() =
    -Jc.transpose().topRows(2) * T;
  const Vector expected_rhs =
    -(M.topRows(2) * qddot_ref) - h.head(2) + h_ext.head(2);

  expectEigenNear(constraint.matrix(), expected_matrix);
  expectEigenNear(constraint.vector(), expected_rhs);
}

TEST(MathValidationTest, JointTorqueLimitBlockBuildsTorqueInequality) {
  Matrix M(4, 4);
  M << 2.0, 0.1, -0.3, 0.2,
       0.1, 3.0, 0.4, -0.5,
       -0.3, 0.4, 4.0, 0.6,
       0.2, -0.5, 0.6, 5.0;
  Vector h(4);
  h << 1.0, 2.0, 3.0, 4.0;
  Vector qddot_ref(4);
  qddot_ref << 0.1, -0.2, 0.3, -0.4;
  Vector h_ext(4);
  h_ext << 0.5, -0.25, 0.75, -1.0;
  Matrix Jc(3, 4);
  Jc << 1.0, 0.0, 2.0, -1.0,
        0.0, 1.0, -2.0, 0.0,
        1.0, 1.0, 0.0, 0.0;
  Matrix T(3, 2);
  T << 1.0, 0.5,
       -0.25, 1.0,
       2.0, -1.0;
  Vector tau_lb(2);
  tau_lb << -5.0, -4.0;
  Vector tau_ub(2);
  tau_ub << 6.0, 8.0;

  HQPBuildContext ctx = makeDecisionContext(4, 2);
  ctx.na = 2;
  ctx.nvFloat = 2;
  ctx.M = &M;
  ctx.h = &h;
  ctx.qddot_ref = &qddot_ref;
  ctx.h_ext = &h_ext;
  ctx.enableJointTorqueLimits = true;
  ctx.tau_lb = &tau_lb;
  ctx.tau_ub = &tau_ub;
  ctx.contactInfos.push_back({&Jc, &T, 0, 2});

  wbc::JointTorqueLimitConstraint block;
  block.build(ctx);

  const auto & constraint = *block.constraint();
  ASSERT_TRUE(constraint.isInequality());

  Matrix expected_matrix = Matrix::Zero(2, 6);
  expected_matrix.leftCols(4) = M.bottomRows(2);
  expected_matrix.block(0, 4, 2, 2).noalias() =
    -Jc.transpose().bottomRows(2) * T;
  const Vector constant =
    h.tail(2) + M.bottomRows(2) * qddot_ref - h_ext.tail(2);

  expectEigenNear(constraint.matrix(), expected_matrix);
  expectEigenNear(constraint.lowerBound(), tau_lb - constant);
  expectEigenNear(constraint.upperBound(), tau_ub - constant);
}

}  // namespace
