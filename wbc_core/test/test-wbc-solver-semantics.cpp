//
// Copyright (c) 2026
//
// Semantic tests for the strict-level delta-form IDHQP core.
//

#include <gtest/gtest.h>

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/rnea.hpp>

#include <wbc_core/controller/id-hqp.hpp>
#include <wbc_core/controller/base/id-problem-registry.hpp>
#include <wbc_core/math/constraints/constraint-equality.hpp>
#include <wbc_core/math/constraints/constraint-inequality.hpp>
#include <wbc_core/robots/robot-system.hpp>
#include <wbc_core/tasks/task-motion.hpp>

namespace {

using wbc::IDHQP;
using wbc::IDProblem;
using wbc::IDSolution;
using wbc::math::Matrix;
using wbc::math::Vector;
using wbc::robots::BaseState;
using wbc::robots::JointState;
using wbc::robots::RobotSystem;

std::shared_ptr<wbc::math::ConstraintBase> makeSingleDoFConstraint(
    const std::string& name, int nv, int dof, double target) {
  auto constraint =
      std::make_shared<wbc::math::ConstraintEquality>(name, 1, nv);
  constraint->matrix().setZero();
  constraint->matrix()(0, dof) = 1.0;
  constraint->vector() = Vector::Constant(1, target);
  return constraint;
}

std::shared_ptr<wbc::math::ConstraintBase> makeSingleDoFInequality(
    const std::string& name, int nv, int dof, double lower, double upper) {
  auto constraint =
      std::make_shared<wbc::math::ConstraintInequality>(name, 1, nv);
  constraint->matrix().setZero();
  constraint->matrix()(0, dof) = 1.0;
  constraint->lowerBound() = Vector::Constant(1, lower);
  constraint->upperBound() = Vector::Constant(1, upper);
  return constraint;
}

wbc::ContactConstraintData makeConstraintContact(const std::string& name,
                                                 int nv, int dof,
                                                 double motion_rhs = 0.0) {
  wbc::ContactConstraintData contact;
  contact.name = name;
  contact.Jc = Matrix::Zero(1, nv);
  contact.Jc(0, dof) = 1.0;
  contact.motion_rhs = Vector::Constant(1, motion_rhs);
  contact.T = Matrix::Zero(1, 1);
  contact.Uf = Matrix::Identity(1, 1);
  contact.uf_lb = Vector::Zero(1);
  contact.uf_ub = Vector::Constant(1, 100.0);
  return contact;
}

wbc::ContactConstraintData makeForceOnlyContact(const std::string& name, int nv,
                                                double lambda_lb,
                                                double lambda_ub) {
  wbc::ContactConstraintData contact;
  contact.name = name;
  contact.Jc = Matrix::Zero(0, nv);
  contact.motion_rhs = Vector::Zero(0);
  contact.T = Matrix::Zero(0, 1);
  contact.Uf = Matrix::Identity(1, 1);
  contact.uf_lb = Vector::Constant(1, lambda_lb);
  contact.uf_ub = Vector::Constant(1, lambda_ub);
  return contact;
}

wbc::ContactConstraintData makeSingleLambdaContact(const std::string& name,
                                                   int nv, int dof,
                                                   double motion_rhs,
                                                   double lambda_lb,
                                                   double lambda_ub) {
  wbc::ContactConstraintData contact;
  contact.name = name;
  contact.Jc = Matrix::Zero(1, nv);
  contact.Jc(0, dof) = 1.0;
  contact.motion_rhs = Vector::Constant(1, motion_rhs);
  contact.T = Matrix::Identity(1, 1);
  contact.Uf = Matrix::Identity(1, 1);
  contact.uf_lb = Vector::Constant(1, lambda_lb);
  contact.uf_ub = Vector::Constant(1, lambda_ub);
  return contact;
}

class MockMotionTask final : public wbc::tasks::TaskMotion {
 public:
  MockMotionTask(const std::string& name, RobotSystem& robot)
      : TaskMotion(name, robot),
        m_nv(robot.nv()),
        m_constraint(name, 1, robot.nv()) {
    m_constraint.matrix().setZero();
    m_constraint.vector().setZero();
    m_aDes = Vector::Zero(1);
  }

  void setSingleDoFTarget(int dof, double value) {
    m_constraint.resize(1, m_nv);
    m_constraint.matrix().setZero();
    m_constraint.matrix()(0, dof) = 1.0;
    m_constraint.vector().setConstant(value);
    m_aDes(0) = value;
  }

  int dim() const override { return static_cast<int>(m_constraint.rows()); }

  const wbc::math::ConstraintBase& compute(double, ConstRefVector,
                                           ConstRefVector, Data&) override {
    return m_constraint;
  }

  const wbc::math::ConstraintBase& getConstraint() const override {
    return m_constraint;
  }

  const Vector& getDesiredAcceleration() const override { return m_aDes; }

 private:
  int m_nv;
  wbc::math::ConstraintEquality m_constraint;
  Vector m_aDes;
};

class WBCSolverSemanticsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const std::vector<std::string> packageDirs{TSID_MODEL_DIR};
    const std::string urdfFile =
        std::string(TSID_MODEL_DIR) + "/romeo/urdf/romeo.urdf";

    robot = std::make_unique<RobotSystem>(urdfFile, packageDirs);
    solver =
        std::make_unique<IDHQP>(*robot, wbc::solvers::SOLVER_HQP_EIQUADPROG);
    registry = std::make_unique<wbc::IDProblemRegistry>(*robot);

    q = pinocchio::neutral(robot->model());
    qdot = Vector::Zero(robot->nv_joints());
  }

  static constexpr double kTol = 1e-7;
  static constexpr double kDt = 0.001;

  IDProblem makeProblem() {
    JointState joint;
    joint.q = q;
    joint.qdot = qdot;
    joint.tau = Vector::Zero(robot->na());
    robot->updateState(joint);
    IDProblem problem;
    return problem;
  }

  wbc::MotionObjective makeSingleDoFObjective(const std::string& name, int dof,
                                              double target,
                                              unsigned int level,
                                              double weight) {
    motion_constraints.push_back(
        makeSingleDoFConstraint(name, robot->nv(), dof, target));
    return wbc::MotionObjective{name, motion_constraints.back().get(), level,
                                weight};
  }

  std::unique_ptr<RobotSystem> robot;
  std::unique_ptr<IDHQP> solver;
  std::unique_ptr<wbc::IDProblemRegistry> registry;
  std::vector<std::shared_ptr<wbc::math::ConstraintBase>> motion_constraints;
  Vector q;
  Vector qdot;
};

TEST_F(WBCSolverSemanticsTest, ZeroNominalBaselineProducesZeroCorrection) {
  auto problem = makeProblem();

  const IDSolution& solution = solver->solve(problem, kDt);

  ASSERT_TRUE(solution.success);
  EXPECT_TRUE(solution.qddot_ref.isZero(kTol));
  EXPECT_TRUE(solution.delta_qddot_sol.isZero(kTol));
  EXPECT_TRUE(solution.qddot_sol.isZero(kTol));
  EXPECT_EQ(solution.lambda_sol.size(), 0);
}

TEST_F(WBCSolverSemanticsTest, ExternalNominalCentersDeltaSolve) {
  auto problem = makeProblem();
  Vector qddot_ref = Vector::LinSpaced(robot->nv(), -0.5, 0.5);
  problem.qddot_ref = &qddot_ref;

  const IDSolution& solution = solver->solve(problem, kDt);

  ASSERT_TRUE(solution.success);
  EXPECT_TRUE(solution.qddot_ref.isApprox(qddot_ref, kTol));
  EXPECT_TRUE(solution.delta_qddot_sol.isZero(kTol));
  EXPECT_TRUE(solution.qddot_sol.isApprox(qddot_ref, kTol));
}

TEST_F(WBCSolverSemanticsTest,
       DisabledRegistryReferenceSolvesPureAccelerationProblem) {
  MockMotionTask task("mock-task", *robot);
  task.setSingleDoFTarget(0, 0.75);
  registry->registerTask(task, 1u, 1.0);

  Vector ignored_reference = Vector::Constant(robot->nv(), 10.0);
  registry->setReferenceAcceleration(ignored_reference);
  registry->setReferenceAccelerationEnabled(false);

  IDProblem problem = registry->buildProblem(0.0, q, qdot);

  ASSERT_NE(problem.qddot_ref, nullptr);
  EXPECT_TRUE(problem.qddot_ref->isZero(kTol));

  const IDSolution& solution = solver->solve(problem, kDt);

  ASSERT_TRUE(solution.success);
  EXPECT_TRUE(solution.qddot_ref.isZero(kTol));
  EXPECT_TRUE(solution.delta_qddot_sol.isApprox(solution.qddot_sol, kTol));
  EXPECT_NEAR(solution.qddot_sol(0), 0.75, kTol);
}

TEST_F(WBCSolverSemanticsTest, ContactFreeCycleClearsPreviousLambdaState) {
  auto withContact = makeProblem();
  withContact.contacts.push_back(
      makeForceOnlyContact("contact", robot->nv(), 1.0, 5.0));

  const IDSolution& withContactSol = solver->solve(withContact, kDt);
  ASSERT_TRUE(withContactSol.success);
  ASSERT_EQ(withContactSol.lambda_sol.size(), 1);
  EXPECT_NEAR(withContactSol.lambda_sol(0), 1.0, 1e-6);

  auto withoutContact = makeProblem();
  const IDSolution& withoutContactSol = solver->solve(withoutContact, kDt);
  ASSERT_TRUE(withoutContactSol.success);
  EXPECT_EQ(withoutContactSol.lambda_sol.size(), 0);
}

TEST_F(WBCSolverSemanticsTest, BetterNominalReducesCorrectionNorm) {
  auto baseline = makeProblem();
  baseline.motion_objectives.push_back(
      makeSingleDoFObjective("operational", 0, 1.25, 1u, 1.0));

  auto goodNominal = baseline;
  Vector qddot_ref = Vector::Zero(robot->nv());
  qddot_ref(0) = 1.25;
  goodNominal.qddot_ref = &qddot_ref;

  const IDSolution& baselineSol = solver->solve(baseline, kDt);
  ASSERT_TRUE(baselineSol.success);
  const double baselineDeltaNorm = baselineSol.delta_qddot_sol.norm();

  const IDSolution& goodNominalSol = solver->solve(goodNominal, kDt);
  ASSERT_TRUE(goodNominalSol.success);
  const double goodDeltaNorm = goodNominalSol.delta_qddot_sol.norm();

  EXPECT_LT(goodDeltaNorm, baselineDeltaNorm);
  EXPECT_NEAR(goodNominalSol.delta_qddot_sol(0), 0.0, kTol);
  EXPECT_NEAR(goodNominalSol.qddot_sol(0), 1.25, kTol);
}

TEST_F(WBCSolverSemanticsTest, InvalidHierarchyFailureResetsSolutionSafely) {
  auto seedInput = makeProblem();
  seedInput.motion_objectives.push_back(
      makeSingleDoFObjective("seed-task", 0, 1.0, 1u, 1.0));
  seedInput.contacts.push_back(
      makeForceOnlyContact("contact", robot->nv(), 1.0, 5.0));
  const IDSolution& seedSol = solver->solve(seedInput, kDt);
  ASSERT_TRUE(seedSol.success);
  ASSERT_EQ(seedSol.lambda_sol.size(), 1);

  auto badInput = makeProblem();
  Vector qddot_ref = Vector::LinSpaced(robot->nv(), -0.5, 0.5);
  badInput.qddot_ref = &qddot_ref;
  badInput.motion_objectives.push_back(
      makeSingleDoFObjective("bad-task", 0, 1.0, 0u, 1.0));

  const IDSolution& badSol = solver->solve(badInput, kDt);
  EXPECT_FALSE(badSol.success);
  EXPECT_TRUE(badSol.qddot_ref.isApprox(qddot_ref, kTol));
  EXPECT_TRUE(badSol.delta_qddot_sol.isZero(kTol));
  EXPECT_TRUE(badSol.qddot_sol.isApprox(qddot_ref, kTol));
  EXPECT_EQ(badSol.lambda_sol.size(), 0);
  EXPECT_TRUE(badSol.tau_sol.isZero(kTol));
}

TEST_F(WBCSolverSemanticsTest,
       InequalityMotionObjectiveIsHardFeasibilityConstraint) {
  auto problem = makeProblem();
  problem.motion_objectives.push_back(
      makeSingleDoFObjective("soft-task", 0, 1.0, 1u, 1.0));
  motion_constraints.push_back(
      makeSingleDoFInequality("hard-bound", robot->nv(), 0, -0.25, 0.25));
  problem.motion_objectives.push_back(
      wbc::MotionObjective{"hard-bound", motion_constraints.back().get(), 9u,
                           1000.0});

  const IDSolution& solution = solver->solve(problem, kDt);

  ASSERT_TRUE(solution.success);
  EXPECT_NEAR(solution.qddot_sol(0), 0.25, kTol);
}

TEST_F(WBCSolverSemanticsTest,
       OperationalLayerDominatesBiasAndBiasActsInRemainingSubspace) {
  auto problem = makeProblem();
  problem.motion_objectives.push_back(
      makeSingleDoFObjective("operational", 0, 1.25, 1u, 1.0));

  Vector qddot_bias = Vector::Constant(robot->nv(), 2.0);
  qddot_bias(0) = -4.0;
  problem.joint_acceleration_objectives.emplace_back("joint-bias",
                                                     &qddot_bias, 2u, 1.0);

  const IDSolution& solution = solver->solve(problem, kDt);

  ASSERT_TRUE(solution.success);
  EXPECT_NEAR(solution.qddot_sol(0), 1.25, kTol);
  EXPECT_NEAR(solution.delta_qddot_sol(0), 1.25, kTol);
  ASSERT_GT(robot->nv(), 1);
  EXPECT_NEAR(solution.qddot_sol(1), 2.0, kTol);
  EXPECT_NEAR(solution.delta_qddot_sol(1), 2.0, kTol);
}

TEST_F(WBCSolverSemanticsTest,
       BiasChangesSolutionFamilyWithoutBreakingOperationalTask) {
  auto noBias = makeProblem();
  noBias.motion_objectives.push_back(
      makeSingleDoFObjective("operational", 0, 1.25, 1u, 1.0));

  auto withBias = noBias;
  Vector qddot_bias = Vector::Zero(robot->nv());
  qddot_bias(1) = 2.0;
  withBias.joint_acceleration_objectives.emplace_back("joint-bias",
                                                      &qddot_bias, 2u, 1.0);

  const IDSolution& noBiasSol = solver->solve(noBias, kDt);
  ASSERT_TRUE(noBiasSol.success);
  const double noBiasDoF1 = noBiasSol.qddot_sol(1);

  const IDSolution& withBiasSol = solver->solve(withBias, kDt);
  ASSERT_TRUE(withBiasSol.success);

  EXPECT_NEAR(noBiasSol.qddot_sol(0), 1.25, kTol);
  EXPECT_NEAR(withBiasSol.qddot_sol(0), 1.25, kTol);
  EXPECT_NEAR(withBiasSol.qddot_sol(1), 2.0, kTol);
  EXPECT_GT(std::abs(withBiasSol.qddot_sol(1) - noBiasDoF1), 1e-4);
}

TEST_F(WBCSolverSemanticsTest, ContactConsistencyBeatsOperationalTask) {
  auto problem = makeProblem();
  problem.motion_objectives.push_back(
      makeSingleDoFObjective("operational", 0, 1.25, 1u, 1.0));
  problem.contacts.push_back(makeConstraintContact("contact", robot->nv(), 0));

  const IDSolution& solution = solver->solve(problem, kDt);

  ASSERT_TRUE(solution.success);
  EXPECT_NEAR(solution.qddot_sol(0), 0.0, kTol);
  EXPECT_NEAR(solution.delta_qddot_sol(0), 0.0, kTol);
}

TEST_F(WBCSolverSemanticsTest, ContactMotionRhsIsEnforcedDirectly) {
  auto problem = makeProblem();
  problem.motion_objectives.push_back(
      makeSingleDoFObjective("operational", 0, 0.0, 1u, 1.0));
  problem.contacts.push_back(
      makeConstraintContact("contact", robot->nv(), 0, 0.4));

  const IDSolution& solution = solver->solve(problem, kDt);

  ASSERT_TRUE(solution.success);
  EXPECT_NEAR(solution.qddot_sol(0), 0.4, kTol);
  const auto& contact = problem.contacts.front();
  const Vector residual = contact.Jc * solution.qddot_sol - contact.motion_rhs;
  EXPECT_LT(residual.norm(), 1e-7);
}

TEST_F(WBCSolverSemanticsTest, SupportContactConsistencyResidualIsNearZero) {
  auto problem = makeProblem();
  problem.motion_objectives.push_back(
      makeSingleDoFObjective("operational", 0, 1.25, 1u, 1.0));
  problem.contacts.push_back(makeConstraintContact("contact", robot->nv(), 0));

  const IDSolution& solution = solver->solve(problem, kDt);

  ASSERT_TRUE(solution.success);
  ASSERT_EQ(problem.contacts.size(), 1u);
  const auto& contact = problem.contacts.front();
  const Vector residual = contact.Jc * solution.qddot_sol - contact.motion_rhs;
  EXPECT_LT(residual.norm(), 1e-7);
}

TEST_F(WBCSolverSemanticsTest,
       ContactOnOffTransitionKeepsUnconstrainedTaskStable) {
  auto noContact = makeProblem();
  noContact.motion_objectives.push_back(
      makeSingleDoFObjective("task-dof1", 1, 0.5, 1u, 1.0));

  auto withContact = noContact;
  withContact.contacts.push_back(
      makeConstraintContact("contact", robot->nv(), 0));

  IDHQP contact_solver(*robot, wbc::solvers::SOLVER_HQP_EIQUADPROG);
  const IDSolution& noContactSol = solver->solve(noContact, kDt);
  ASSERT_TRUE(noContactSol.success);

  const IDSolution& withContactSol = contact_solver.solve(withContact, kDt);
  ASSERT_TRUE(withContactSol.success);

  // Contact toggling should leave unconstrained task-space behavior unchanged.
  EXPECT_NEAR(noContactSol.qddot_sol(1), 0.5, kTol);
  EXPECT_NEAR(withContactSol.qddot_sol(1), 0.5, kTol);
  EXPECT_LT(std::abs(withContactSol.qddot_sol(1) - noContactSol.qddot_sol(1)),
            1e-7);
}

TEST_F(WBCSolverSemanticsTest,
       RegularizationNeverOverridesOperationalOrFeasibilityLevels) {
  auto problem = makeProblem();
  problem.motion_objectives.push_back(
      makeSingleDoFObjective("operational", 0, 1.25, 1u, 1.0));
  problem.contacts.push_back(makeConstraintContact("contact", robot->nv(), 0));
  problem.regularization.w_delta_qddot = 1e12;
  problem.regularization.w_lambda = 1e12;

  const IDSolution& solution = solver->solve(problem, kDt);

  ASSERT_TRUE(solution.success);
  // Level 0 contact consistency should still dominate Level 3 regularization.
  EXPECT_NEAR(solution.qddot_sol(0), 0.0, kTol);
  EXPECT_NEAR(solution.delta_qddot_sol(0), 0.0, kTol);
}

TEST_F(WBCSolverSemanticsTest, JointTorqueBoundsRemainActiveAgainstBias) {
  auto problem = makeProblem();
  pinocchio::Data data(robot->model());
  Vector gravity_tau =
      pinocchio::rnea(robot->model(), data, q, qdot, Vector::Zero(robot->nv()));
  problem.joint_torque_limits.lower = &gravity_tau;
  problem.joint_torque_limits.upper = &gravity_tau;

  Vector qddot_bias = Vector::Constant(robot->nv(), 5.0);
  problem.joint_acceleration_objectives.emplace_back("joint-bias",
                                                     &qddot_bias, 2u, 1.0);

  const IDSolution& solution = solver->solve(problem, kDt);

  ASSERT_TRUE(solution.success);
  EXPECT_TRUE(solution.tau_sol.isApprox(gravity_tau, 1e-5));
  EXPECT_LT(solution.qddot_sol.norm(), 1e-5);
}

TEST_F(WBCSolverSemanticsTest,
       BadNominalIsCorrectedBackIntoFeasibleTorqueRegion) {
  auto problem = makeProblem();
  pinocchio::Data data(robot->model());
  Vector gravity_tau =
      pinocchio::rnea(robot->model(), data, q, qdot, Vector::Zero(robot->nv()));
  problem.joint_torque_limits.lower = &gravity_tau;
  problem.joint_torque_limits.upper = &gravity_tau;

  Vector badReference = Vector::LinSpaced(robot->nv(), -50.0, 50.0);
  problem.qddot_ref = &badReference;

  const IDSolution& solution = solver->solve(problem, kDt);

  ASSERT_TRUE(solution.success);
  EXPECT_TRUE(solution.tau_sol.isApprox(gravity_tau, 1e-5));
  EXPECT_LT(solution.qddot_sol.norm(), 1e-4);
  EXPECT_GT(solution.delta_qddot_sol.norm(), 1.0);
}

TEST_F(WBCSolverSemanticsTest, ContactForceContributesToRecoveredTorque) {
  auto problem = makeProblem();
  const int dof = robot->nv() - 1;
  const double lambda = 2.0;
  problem.contacts.push_back(
      makeSingleLambdaContact("contact", robot->nv(), dof, 0.0, lambda,
                              lambda));

  const IDSolution& solution = solver->solve(problem, kDt);

  ASSERT_TRUE(solution.success);
  ASSERT_EQ(solution.lambda_sol.size(), 1);
  EXPECT_NEAR(solution.lambda_sol(0), lambda, 1e-6);

  pinocchio::Data data(robot->model());
  Vector expected_tau =
      pinocchio::rnea(robot->model(), data, q, qdot, solution.qddot_sol);
  const Vector contact_tau =
      problem.contacts.front().Jc.transpose() * problem.contacts.front().T *
      solution.lambda_sol;
  expected_tau -= contact_tau;
  EXPECT_TRUE(solution.tau_sol.isApprox(expected_tau.tail(robot->na()), 1e-5));
}

TEST_F(WBCSolverSemanticsTest,
       JointTorqueLimitIncludesContactForceContribution) {
  auto problem = makeProblem();
  const int dof = robot->nv() - 1;
  const double lambda = 2.0;
  const double qddot_target = 0.15;
  problem.contacts.push_back(makeSingleLambdaContact(
      "contact", robot->nv(), dof, qddot_target, lambda, lambda));

  Vector qddot_expected = Vector::Zero(robot->nv());
  qddot_expected(dof) = qddot_target;

  pinocchio::Data data(robot->model());
  Vector tau_bound =
      pinocchio::rnea(robot->model(), data, q, qdot, qddot_expected);
  const Vector contact_tau =
      problem.contacts.front().Jc.transpose() * problem.contacts.front().T *
      Vector::Constant(1, lambda);
  tau_bound -= contact_tau;
  Vector tau_bound_actuated = tau_bound.tail(robot->na());
  problem.joint_torque_limits.lower = &tau_bound_actuated;
  problem.joint_torque_limits.upper = &tau_bound_actuated;

  Vector qddot_bias = Vector::Constant(robot->nv(), -3.0);
  problem.joint_acceleration_objectives.emplace_back("joint-bias",
                                                     &qddot_bias, 2u, 1.0);

  const IDSolution& solution = solver->solve(problem, kDt);

  ASSERT_TRUE(solution.success);
  EXPECT_NEAR(solution.qddot_sol(dof), qddot_target, 1e-6);
  EXPECT_NEAR(solution.lambda_sol(0), lambda, 1e-6);
  EXPECT_TRUE(solution.tau_sol.isApprox(tau_bound_actuated, 1e-5));
}

TEST_F(WBCSolverSemanticsTest,
       FloatingBaseDynamicsBalancesContactForceDecision) {
  const std::vector<std::string> packageDirs{TSID_MODEL_DIR};
  const std::string urdfFile =
      std::string(TSID_MODEL_DIR) + "/romeo/urdf/romeo.urdf";
  RobotSystem floatingRobot(urdfFile, packageDirs,
                            pinocchio::JointModelFreeFlyer());
  IDHQP floatingSolver(floatingRobot, wbc::solvers::SOLVER_HQP_EIQUADPROG);

  const Vector q_joints = pinocchio::neutral(floatingRobot.model())
                              .tail(floatingRobot.nq_joints());
  const Vector qdot_joints = Vector::Zero(floatingRobot.nv_joints());
  JointState joint;
  joint.q = q_joints;
  joint.qdot = qdot_joints;
  joint.tau = Vector::Zero(floatingRobot.na());
  floatingRobot.updateState(joint, BaseState{});

  IDProblem problem;
  problem.contacts.push_back(
      makeSingleLambdaContact("base-contact", floatingRobot.nv(), 2, 0.0,
                              -1e5, 1e5));

  const IDSolution& solution = floatingSolver.solve(problem, kDt);

  ASSERT_TRUE(solution.success);
  ASSERT_EQ(solution.lambda_sol.size(), 1);

  pinocchio::Data data(floatingRobot.model());
  floatingRobot.computeAllTerms(data, floatingRobot.generalized_q(),
                                floatingRobot.generalized_v());
  const auto& contact = problem.contacts.front();
  const Vector residual =
      floatingRobot.mass(data).topRows(6) * solution.qddot_sol -
      (contact.Jc.transpose() * contact.T * solution.lambda_sol).head(6) +
      floatingRobot.nonLinearEffects(data).head(6);
  EXPECT_LT(residual.norm(), 1e-5);
}

TEST_F(WBCSolverSemanticsTest, SupportContactChoosesMinimumNormFeasibleLambda) {
  auto problem = makeProblem();
  problem.contacts.push_back(
      makeForceOnlyContact("contact", robot->nv(), 1.0, 5.0));

  const IDSolution& solution = solver->solve(problem, kDt);

  ASSERT_TRUE(solution.success);
  ASSERT_EQ(solution.lambda_sol.size(), 1);
  EXPECT_NEAR(solution.lambda_sol(0), 1.0, 1e-6);
}

TEST_F(WBCSolverSemanticsTest, UnconstrainedSupportForceFallsBackToZeroLambda) {
  auto problem = makeProblem();
  problem.contacts.push_back(
      makeForceOnlyContact("contact", robot->nv(), 0.0, 5.0));

  const IDSolution& solution = solver->solve(problem, kDt);

  ASSERT_TRUE(solution.success);
  ASSERT_EQ(solution.lambda_sol.size(), 1);
  EXPECT_NEAR(solution.lambda_sol(0), 0.0, 1e-6);
}

TEST_F(WBCSolverSemanticsTest, RegistryBuildsProblemFromRegisteredRuntimeData) {
  MockMotionTask task("mock-task", *robot);
  task.setSingleDoFTarget(0, 0.75);
  registry->registerTask(task, 1u, 2.0);

  IDProblem problem = registry->buildProblem(0.0, q, qdot);
  ASSERT_EQ(problem.motion_objectives.size(), 1u);
  EXPECT_EQ(problem.motion_objectives[0].name, "mock-task");
  EXPECT_DOUBLE_EQ(problem.motion_objectives[0].weight, 2.0);
  EXPECT_NEAR(problem.motion_objectives[0].matrix()(0, 0), 1.0,
              kTol);
  EXPECT_NEAR(problem.motion_objectives[0].vector()(0), 0.75, kTol);

  const IDSolution& solution = solver->solve(problem, kDt);
  ASSERT_TRUE(solution.success);
  EXPECT_NEAR(solution.qddot_sol(0), 0.75, kTol);
}

TEST_F(WBCSolverSemanticsTest,
       RegistryPreservesExplicitTaskLevels) {
  MockMotionTask operationalTask("operational-task", *robot);
  operationalTask.setSingleDoFTarget(0, 0.5);
  registry->registerTask(operationalTask, 1u, 2.0);

  MockMotionTask biasTask("bias-task", *robot);
  biasTask.setSingleDoFTarget(0, -0.5);
  registry->registerTask(biasTask, 2u, 3.0);

  const std::vector<std::string> activeTasks{"operational-task", "bias-task"};
  const std::vector<double> taskWeights{-1.0, -1.0};
  const std::vector<std::string> activeContacts;

  IDProblem problem = registry->buildProblem(0.0, q, qdot, activeTasks,
                                             taskWeights, activeContacts);

  ASSERT_EQ(problem.motion_objectives.size(), 2u);
  EXPECT_EQ(problem.motion_objectives[0].name, "operational-task");
  EXPECT_EQ(problem.motion_objectives[0].level, 1u);
  EXPECT_EQ(problem.motion_objectives[1].name, "bias-task");
  EXPECT_EQ(problem.motion_objectives[1].level, 2u);
}

TEST_F(WBCSolverSemanticsTest, RegistryThrowsOnUnknownActiveTaskName) {
  MockMotionTask task("known-task", *robot);
  task.setSingleDoFTarget(0, 0.5);
  registry->registerTask(task, 1u, 1.0);

  const std::vector<std::string> activeTasks{"missing-task"};
  const std::vector<double> taskWeights{-1.0};
  const std::vector<std::string> activeContacts;

  EXPECT_THROW(registry->buildProblem(0.0, q, qdot, activeTasks, taskWeights,
                                      activeContacts),
               std::invalid_argument);
}

TEST_F(WBCSolverSemanticsTest, RegistryThrowsOnUnknownActiveContactName) {
  const std::vector<std::string> activeTasks;
  const std::vector<double> taskWeights;
  const std::vector<std::string> activeContacts{"missing-contact"};

  EXPECT_THROW(registry->buildProblem(0.0, q, qdot, activeTasks, taskWeights,
                                      activeContacts),
               std::invalid_argument);
}

}  // namespace
