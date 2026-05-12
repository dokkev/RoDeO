//
// Copyright (c) 2026
//
// Smoke tests for optional 2-level numerical solver backends.
//

#include <gtest/gtest.h>

#include <memory>
#include <string>

#include <wbc_core/math/constraint-bound.hpp>
#include <wbc_core/math/constraint-equality.hpp>
#include <wbc_core/solvers/solver-HQP-cascade.hpp>
#include <wbc_core/solvers/solver-HQP-factory.hpp>
#include <wbc_core/solvers/solver-HQP-factory.hxx>

namespace {

using wbc::math::ConstraintBound;
using wbc::math::ConstraintEquality;
using wbc::math::Matrix;
using wbc::math::Vector;
using wbc::solvers::HQP_STATUS_OPTIMAL;
using wbc::solvers::HQPData;
using wbc::solvers::SolverHQP;
using wbc::solvers::SolverHQPBase;
using wbc::solvers::SolverHQPFactory;

HQPData makeBoundedTargetProblem() {
  HQPData problem;
  problem.resize(2);

  auto bounds = std::make_shared<ConstraintBound>("bounds", 2);
  bounds->lowerBound() = Vector::Constant(2, -5.0);
  bounds->upperBound() = Vector::Constant(2, 5.0);
  problem[0].push_back(wbc::solvers::make_pair<double>(
      1.0, std::static_pointer_cast<wbc::math::ConstraintBase>(bounds)));

  auto target = std::make_shared<ConstraintEquality>("target", 2, 2);
  target->matrix() = Matrix::Identity(2, 2);
  target->vector() << 1.25, -0.5;
  problem[1].push_back(wbc::solvers::make_pair<double>(
      1.0, std::static_pointer_cast<wbc::math::ConstraintBase>(target)));

  return problem;
}

std::shared_ptr<wbc::math::ConstraintBase> makeEqualityTarget(
    const std::string& name, const Matrix& matrix, const Vector& target) {
  auto equality =
      std::make_shared<ConstraintEquality>(name, matrix.rows(), matrix.cols());
  equality->matrix() = matrix;
  equality->vector() = target;
  return std::static_pointer_cast<wbc::math::ConstraintBase>(equality);
}

HQPData makeConflictingThreeLevelProblem() {
  HQPData problem;
  problem.resize(3);

  Matrix highPriority = Matrix::Zero(1, 2);
  highPriority(0, 0) = 1.0;
  Vector highTarget(1);
  highTarget << 1.0;
  problem[1].push_back(wbc::solvers::make_pair<double>(
      1.0, makeEqualityTarget("keep_x0_at_one", highPriority, highTarget)));

  Matrix lowPriority = Matrix::Identity(2, 2);
  Vector lowTarget(2);
  lowTarget << -3.0, 2.0;
  problem[2].push_back(wbc::solvers::make_pair<double>(
      1.0, makeEqualityTarget("prefer_full_target", lowPriority, lowTarget)));

  return problem;
}

void expectSolvesBoundedTarget(SolverHQPBase& solver, double tolerance = 1e-6) {
  const auto& output = solver.solve(makeBoundedTargetProblem());
  ASSERT_EQ(output.status, HQP_STATUS_OPTIMAL);
  ASSERT_EQ(output.x.size(), 2);
  EXPECT_NEAR(output.x(0), 1.25, tolerance);
  EXPECT_NEAR(output.x(1), -0.5, tolerance);
}

void expectFactorySolvesBoundedTarget(SolverHQP solverType,
                                      const std::string& solverName,
                                      double tolerance = 1e-6) {
  std::unique_ptr<SolverHQPBase> solver(
      SolverHQPFactory::createNewSolver(solverType, solverName));
  ASSERT_NE(solver, nullptr);

  expectSolvesBoundedTarget(*solver, tolerance);
}

}  // namespace

TEST(SolverBaselineTest, CascadeUsesSelectedInnerBackend) {
  wbc::solvers::SolverHQPCascade solver("cascade",
                                        wbc::solvers::SOLVER_HQP_PROXQP);
  expectSolvesBoundedTarget(solver, 2e-6);
}

TEST(SolverBaselineTest,
     CascadePreservesHigherPriorityObjectiveAcrossConflictingLowerLevel) {
  wbc::solvers::SolverHQPCascade solver("cascade",
                                        wbc::solvers::SOLVER_HQP_PROXQP);
  wbc::solvers::SolverQPParams qp_params;
  qp_params.max_iter = 200u;
  qp_params.eps_abs = 1e-9;
  qp_params.eps_rel = 1e-9;
  solver.setQPParams(qp_params);

  const auto& output = solver.solve(makeConflictingThreeLevelProblem());

  ASSERT_EQ(output.status, HQP_STATUS_OPTIMAL);
  ASSERT_EQ(output.x.size(), 2);
  EXPECT_NEAR(output.x(0), 1.0, 1e-5);
  EXPECT_NEAR(output.x(1), 2.0, 1e-5);
}

#ifdef TSID_WITH_PROXSUITE
TEST(SolverBaselineTest, ProxQPSolvesTwoLevelProblem) {
  expectFactorySolvesBoundedTarget(wbc::solvers::SOLVER_HQP_PROXQP, "proxqp",
                                   2e-6);
}
#endif

TEST(SolverBaselineTest, EiquadprogSolvesTwoLevelProblem) {
  expectFactorySolvesBoundedTarget(wbc::solvers::SOLVER_HQP_EIQUADPROG,
                                   "eiquadprog");
}

TEST(SolverBaselineTest, EiquadprogFastSolvesTwoLevelProblem) {
  expectFactorySolvesBoundedTarget(wbc::solvers::SOLVER_HQP_EIQUADPROG_FAST,
                                   "eiquadprog-fast");
}

TEST(SolverBaselineTest, EiquadprogRtSolvesTwoLevelProblem) {
  std::unique_ptr<SolverHQPBase> solver(
      SolverHQPFactory::createNewSolver<2, 0, 2>(
          wbc::solvers::SOLVER_HQP_EIQUADPROG_RT, "eiquadprog-rt"));
  ASSERT_NE(solver, nullptr);

  expectSolvesBoundedTarget(*solver);
}
