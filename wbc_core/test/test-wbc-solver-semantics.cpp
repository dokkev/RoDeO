//
// Copyright (c) 2026
//
// Semantic tests for the strict-level delta-form WBMC core.
//

#include <gtest/gtest.h>

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/rnea.hpp>

#include <wbc_core/controller/wbmc.hpp>
#include <wbc_core/controller/wbmc-registry.hpp>
#include <wbc_core/math/constraint-equality.hpp>
#include <wbc_core/robots/robot-wrapper.hpp>
#include <wbc_core/tasks/task-motion.hpp>

namespace {

using tsid::WBMCStepInput;
using tsid::WBMC;
using tsid::WBMCSolution;
using tsid::robots::RobotWrapper;
using tsid::math::Matrix;
using tsid::math::Vector;

tsid::MotionObjective makeSingleDoFTask(const std::string& name, int nv,
                                        int dof, double target) {
  tsid::MotionObjective task;
  task.name = name;
  task.J = Matrix::Zero(1, nv);
  task.J(0, dof) = 1.0;
  task.a_des = Vector::Constant(1, target);
  return task;
}

tsid::ContactSnapshot makeConstraintContact(const std::string& name, int nv,
                                            int dof) {
  tsid::ContactSnapshot contact;
  contact.name = name;
  contact.Jc = Matrix::Zero(1, nv);
  contact.Jc(0, dof) = 1.0;
  contact.Jcdot_qdot = Vector::Zero(1);
  contact.T = Matrix::Zero(1, 1);
  contact.Uf = Matrix::Identity(1, 1);
  contact.uf_lb = Vector::Zero(1);
  contact.uf_ub = Vector::Constant(1, 100.0);
  return contact;
}

tsid::ContactSnapshot makeForceOnlyContact(const std::string& name, int nv,
                                           double lambda_lb,
                                           double lambda_ub) {
  tsid::ContactSnapshot contact;
  contact.name = name;
  contact.Jc = Matrix::Zero(0, nv);
  contact.Jcdot_qdot = Vector::Zero(0);
  contact.T = Matrix::Zero(0, 1);
  contact.Uf = Matrix::Identity(1, 1);
  contact.uf_lb = Vector::Constant(1, lambda_lb);
  contact.uf_ub = Vector::Constant(1, lambda_ub);
  return contact;
}

class MockMotionTask final : public tsid::tasks::TaskMotion {
 public:
  MockMotionTask(const std::string& name, RobotWrapper& robot)
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

  const tsid::math::ConstraintBase& compute(
      double, ConstRefVector, ConstRefVector, Data&) override {
    return m_constraint;
  }

  const tsid::math::ConstraintBase& getConstraint() const override {
    return m_constraint;
  }

  const Vector& getDesiredAcceleration() const override { return m_aDes; }

 private:
  int m_nv;
  tsid::math::ConstraintEquality m_constraint;
  Vector m_aDes;
};

class WBCSolverSemanticsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const std::vector<std::string> packageDirs{TSID_MODEL_DIR};
    const std::string urdfFile =
        std::string(TSID_MODEL_DIR) + "/romeo/urdf/romeo.urdf";

    robot = std::make_unique<RobotWrapper>(urdfFile, packageDirs);
    solver = std::make_unique<WBMC>(*robot);
    registry = std::make_unique<tsid::WBMCRegistry>(*robot);

    q = pinocchio::neutral(robot->model());
    qdot = Vector::Zero(robot->nv());
  }

  static constexpr double kTol = 1e-7;

  WBMCStepInput makeInput() {
    WBMCStepInput input;
    input.q = &q;
    input.qdot = &qdot;
    return input;
  }

  std::unique_ptr<RobotWrapper> robot;
  std::unique_ptr<WBMC> solver;
  std::unique_ptr<tsid::WBMCRegistry> registry;
  Vector q;
  Vector qdot;
};

TEST_F(WBCSolverSemanticsTest, ZeroNominalBaselineProducesZeroCorrection) {
  auto input = makeInput();

  const WBMCSolution& solution = solver->solve(input);

  ASSERT_TRUE(solution.success);
  EXPECT_TRUE(solution.qddot_ref.isZero(kTol));
  EXPECT_TRUE(solution.delta_qddot.isZero(kTol));
  EXPECT_TRUE(solution.qddot_sol.isZero(kTol));
  EXPECT_EQ(solution.lambda.size(), 0);
}

TEST_F(WBCSolverSemanticsTest, ExternalNominalCentersDeltaSolve) {
  auto input = makeInput();
  Vector qddot_ref = Vector::LinSpaced(robot->nv(), -0.5, 0.5);
  input.qddot_ref = &qddot_ref;

  const WBMCSolution& solution = solver->solve(input);

  ASSERT_TRUE(solution.success);
  EXPECT_TRUE(solution.qddot_ref.isApprox(qddot_ref, kTol));
  EXPECT_TRUE(solution.delta_qddot.isZero(kTol));
  EXPECT_TRUE(solution.qddot_sol.isApprox(qddot_ref, kTol));
}

TEST_F(WBCSolverSemanticsTest, ContactFreeCycleClearsPreviousLambdaState) {
  auto withContact = makeInput();
  withContact.contacts.push_back(
      makeForceOnlyContact("contact", robot->nv(), 1.0, 5.0));

  const WBMCSolution& withContactSol = solver->solve(withContact);
  ASSERT_TRUE(withContactSol.success);
  ASSERT_EQ(withContactSol.lambda.size(), 1);
  EXPECT_NEAR(withContactSol.lambda(0), 1.0, 1e-6);

  auto withoutContact = makeInput();
  const WBMCSolution& withoutContactSol = solver->solve(withoutContact);
  ASSERT_TRUE(withoutContactSol.success);
  EXPECT_EQ(withoutContactSol.lambda.size(), 0);
}

TEST_F(WBCSolverSemanticsTest, BetterNominalReducesCorrectionNorm) {
  auto baseline = makeInput();
  baseline.objectives.push_back(tsid::SoftObjective::Motion(
      makeSingleDoFTask("operational", robot->nv(), 0, 1.25), 1u, 1.0));

  auto goodNominal = baseline;
  Vector qddot_ref = Vector::Zero(robot->nv());
  qddot_ref(0) = 1.25;
  goodNominal.qddot_ref = &qddot_ref;

  const WBMCSolution& baselineSol = solver->solve(baseline);
  ASSERT_TRUE(baselineSol.success);
  const double baselineDeltaNorm = baselineSol.delta_qddot.norm();

  const WBMCSolution& goodNominalSol = solver->solve(goodNominal);
  ASSERT_TRUE(goodNominalSol.success);
  const double goodDeltaNorm = goodNominalSol.delta_qddot.norm();

  EXPECT_LT(goodDeltaNorm, baselineDeltaNorm);
  EXPECT_NEAR(goodNominalSol.delta_qddot(0), 0.0, kTol);
  EXPECT_NEAR(goodNominalSol.qddot_sol(0), 1.25, kTol);
}

TEST_F(WBCSolverSemanticsTest, InvalidHierarchyFailureResetsSolutionSafely) {
  auto seedInput = makeInput();
  seedInput.objectives.push_back(tsid::SoftObjective::Motion(
      makeSingleDoFTask("seed-task", robot->nv(), 0, 1.0), 1u, 1.0));
  seedInput.contacts.push_back(
      makeForceOnlyContact("contact", robot->nv(), 1.0, 5.0));
  const WBMCSolution& seedSol = solver->solve(seedInput);
  ASSERT_TRUE(seedSol.success);
  ASSERT_EQ(seedSol.lambda.size(), 1);

  auto badInput = makeInput();
  Vector qddot_ref = Vector::LinSpaced(robot->nv(), -0.5, 0.5);
  badInput.qddot_ref = &qddot_ref;
  badInput.objectives.push_back(tsid::SoftObjective::Motion(
      makeSingleDoFTask("bad-task", robot->nv(), 0, 1.0), 0u, 1.0));

  const WBMCSolution& badSol = solver->solve(badInput);
  EXPECT_FALSE(badSol.success);
  EXPECT_TRUE(badSol.qddot_ref.isApprox(qddot_ref, kTol));
  EXPECT_TRUE(badSol.delta_qddot.isZero(kTol));
  EXPECT_TRUE(badSol.qddot_sol.isApprox(qddot_ref, kTol));
  EXPECT_EQ(badSol.lambda.size(), 0);
  EXPECT_TRUE(badSol.tau.isZero(kTol));
}

TEST_F(WBCSolverSemanticsTest,
       OperationalLayerDominatesBiasAndBiasActsInRemainingSubspace) {
  auto input = makeInput();
  input.objectives.push_back(tsid::SoftObjective::Motion(
      makeSingleDoFTask("operational", robot->nv(), 0, 1.25), 1u, 1.0));

  Vector qddot_bias = Vector::Constant(robot->nv(), 2.0);
  qddot_bias(0) = -4.0;
  input.objectives.push_back(tsid::SoftObjective::JointAcceleration(
      tsid::JointAccelerationObjective{"joint-bias", &qddot_bias}, 2u, 1.0));

  const WBMCSolution& solution = solver->solve(input);

  ASSERT_TRUE(solution.success);
  EXPECT_NEAR(solution.qddot_sol(0), 1.25, kTol);
  EXPECT_NEAR(solution.delta_qddot(0), 1.25, kTol);
  ASSERT_GT(robot->nv(), 1);
  EXPECT_NEAR(solution.qddot_sol(1), 2.0, kTol);
  EXPECT_NEAR(solution.delta_qddot(1), 2.0, kTol);
}

TEST_F(WBCSolverSemanticsTest,
       BiasChangesSolutionFamilyWithoutBreakingOperationalTask) {
  auto noBias = makeInput();
  noBias.objectives.push_back(tsid::SoftObjective::Motion(
      makeSingleDoFTask("operational", robot->nv(), 0, 1.25), 1u, 1.0));

  auto withBias = noBias;
  Vector qddot_bias = Vector::Zero(robot->nv());
  qddot_bias(1) = 2.0;
  withBias.objectives.push_back(tsid::SoftObjective::JointAcceleration(
      tsid::JointAccelerationObjective{"joint-bias", &qddot_bias}, 2u, 1.0));

  const WBMCSolution& noBiasSol = solver->solve(noBias);
  ASSERT_TRUE(noBiasSol.success);
  const double noBiasDoF1 = noBiasSol.qddot_sol(1);

  const WBMCSolution& withBiasSol = solver->solve(withBias);
  ASSERT_TRUE(withBiasSol.success);

  EXPECT_NEAR(noBiasSol.qddot_sol(0), 1.25, kTol);
  EXPECT_NEAR(withBiasSol.qddot_sol(0), 1.25, kTol);
  EXPECT_NEAR(withBiasSol.qddot_sol(1), 2.0, kTol);
  EXPECT_GT(std::abs(withBiasSol.qddot_sol(1) - noBiasDoF1), 1e-4);
}

TEST_F(WBCSolverSemanticsTest, ContactConsistencyBeatsOperationalTask) {
  auto input = makeInput();
  input.objectives.push_back(tsid::SoftObjective::Motion(
      makeSingleDoFTask("operational", robot->nv(), 0, 1.25), 1u, 1.0));
  input.contacts.push_back(makeConstraintContact("contact", robot->nv(), 0));

  const WBMCSolution& solution = solver->solve(input);

  ASSERT_TRUE(solution.success);
  EXPECT_NEAR(solution.qddot_sol(0), 0.0, kTol);
  EXPECT_NEAR(solution.delta_qddot(0), 0.0, kTol);
}

TEST_F(WBCSolverSemanticsTest, SupportContactConsistencyResidualIsNearZero) {
  auto input = makeInput();
  input.objectives.push_back(tsid::SoftObjective::Motion(
      makeSingleDoFTask("operational", robot->nv(), 0, 1.25), 1u, 1.0));
  input.contacts.push_back(makeConstraintContact("contact", robot->nv(), 0));

  const WBMCSolution& solution = solver->solve(input);

  ASSERT_TRUE(solution.success);
  ASSERT_EQ(input.contacts.size(), 1u);
  const auto& contact = input.contacts.front();
  const Vector residual =
      contact.Jc * solution.qddot_sol + contact.Jcdot_qdot;
  EXPECT_LT(residual.norm(), 1e-7);
}

TEST_F(WBCSolverSemanticsTest,
       ContactOnOffTransitionKeepsUnconstrainedTaskStable) {
  auto noContact = makeInput();
  noContact.objectives.push_back(tsid::SoftObjective::Motion(
      makeSingleDoFTask("task-dof1", robot->nv(), 1, 0.5), 1u, 1.0));

  auto withContact = noContact;
  withContact.contacts.push_back(makeConstraintContact("contact", robot->nv(), 0));

  WBMC withContactSolver(*robot);
  const WBMCSolution& noContactSol = solver->solve(noContact);
  ASSERT_TRUE(noContactSol.success);

  const WBMCSolution& withContactSol = withContactSolver.solve(withContact);
  ASSERT_TRUE(withContactSol.success);

  // Contact toggling should leave unconstrained task-space behavior unchanged.
  EXPECT_NEAR(noContactSol.qddot_sol(1), 0.5, kTol);
  EXPECT_NEAR(withContactSol.qddot_sol(1), 0.5, kTol);
  EXPECT_LT(
      std::abs(withContactSol.qddot_sol(1) - noContactSol.qddot_sol(1)),
      1e-7);
}

TEST_F(WBCSolverSemanticsTest,
       RegularizationNeverOverridesOperationalOrFeasibilityLevels) {
  auto input = makeInput();
  input.objectives.push_back(tsid::SoftObjective::Motion(
      makeSingleDoFTask("operational", robot->nv(), 0, 1.25), 1u, 1.0));
  input.contacts.push_back(makeConstraintContact("contact", robot->nv(), 0));
  input.regularization.w_delta_qddot = 1e12;
  input.regularization.w_lambda = 1e12;

  const WBMCSolution& solution = solver->solve(input);

  ASSERT_TRUE(solution.success);
  // Level 0 contact consistency should still dominate Level 3 regularization.
  EXPECT_NEAR(solution.qddot_sol(0), 0.0, kTol);
  EXPECT_NEAR(solution.delta_qddot(0), 0.0, kTol);
}

TEST_F(WBCSolverSemanticsTest, TorqueBoundsRemainActiveAgainstBias) {
  auto input = makeInput();
  pinocchio::Data data(robot->model());
  Vector gravity_tau = pinocchio::rnea(robot->model(), data, q, qdot,
                                       Vector::Zero(robot->nv()));
  input.tau_lb = &gravity_tau;
  input.tau_ub = &gravity_tau;

  Vector qddot_bias = Vector::Constant(robot->nv(), 5.0);
  input.objectives.push_back(tsid::SoftObjective::JointAcceleration(
      tsid::JointAccelerationObjective{"joint-bias", &qddot_bias}, 2u, 1.0));

  const WBMCSolution& solution = solver->solve(input);

  ASSERT_TRUE(solution.success);
  EXPECT_TRUE(solution.tau.isApprox(gravity_tau, 1e-5));
  EXPECT_LT(solution.qddot_sol.norm(), 1e-5);
}

TEST_F(WBCSolverSemanticsTest,
       BadNominalIsCorrectedBackIntoFeasibleTorqueRegion) {
  auto input = makeInput();
  pinocchio::Data data(robot->model());
  Vector gravity_tau = pinocchio::rnea(robot->model(), data, q, qdot,
                                       Vector::Zero(robot->nv()));
  input.tau_lb = &gravity_tau;
  input.tau_ub = &gravity_tau;

  Vector badReference = Vector::LinSpaced(robot->nv(), -50.0, 50.0);
  input.qddot_ref = &badReference;

  const WBMCSolution& solution = solver->solve(input);

  ASSERT_TRUE(solution.success);
  EXPECT_TRUE(solution.tau.isApprox(gravity_tau, 1e-5));
  EXPECT_LT(solution.qddot_sol.norm(), 1e-4);
  EXPECT_GT(solution.delta_qddot.norm(), 1.0);
}

TEST_F(WBCSolverSemanticsTest, SupportContactChoosesMinimumNormFeasibleLambda) {
  auto input = makeInput();
  input.contacts.push_back(
      makeForceOnlyContact("contact", robot->nv(), 1.0, 5.0));

  const WBMCSolution& solution = solver->solve(input);

  ASSERT_TRUE(solution.success);
  ASSERT_EQ(solution.lambda.size(), 1);
  EXPECT_NEAR(solution.lambda(0), 1.0, 1e-6);
}

TEST_F(WBCSolverSemanticsTest, UnconstrainedSupportForceFallsBackToZeroLambda) {
  auto input = makeInput();
  input.contacts.push_back(makeForceOnlyContact("contact", robot->nv(), 0.0, 5.0));

  const WBMCSolution& solution = solver->solve(input);

  ASSERT_TRUE(solution.success);
  ASSERT_EQ(solution.lambda.size(), 1);
  EXPECT_NEAR(solution.lambda(0), 0.0, 1e-6);
}

TEST_F(WBCSolverSemanticsTest, RegistryBuildsStepInputFromRegisteredRuntimeData) {
  MockMotionTask task("mock-task", *robot);
  task.setSingleDoFTarget(0, 0.75);
  registry->addOperationalTask(task, 2.0);

  WBMCStepInput input = registry->buildStepInput(0.0, q, qdot);
  ASSERT_EQ(input.objectives.size(), 1u);
  ASSERT_TRUE(input.objectives[0].isMotion());
  EXPECT_EQ(input.objectives[0].motion().name, "mock-task");
  EXPECT_DOUBLE_EQ(input.objectives[0].weight, 2.0);
  EXPECT_NEAR(input.objectives[0].motion().J(0, 0), 1.0, kTol);
  EXPECT_NEAR(input.objectives[0].motion().a_des(0), 0.75, kTol);

  const WBMCSolution& solution = solver->solve(input);
  ASSERT_TRUE(solution.success);
  EXPECT_NEAR(solution.qddot_sol(0), 0.75, kTol);
}

TEST_F(WBCSolverSemanticsTest,
       RegistryPreservesOperationalAndBiasRoleSemantics) {
  MockMotionTask operationalTask("operational-task", *robot);
  operationalTask.setSingleDoFTarget(0, 0.5);
  registry->addOperationalTask(operationalTask, 2.0);

  MockMotionTask biasTask("bias-task", *robot);
  biasTask.setSingleDoFTarget(0, -0.5);
  registry->addTaskSpaceBias(biasTask, 3.0);

  const std::vector<std::string> activeTasks{"operational-task", "bias-task"};
  const std::vector<double> taskWeights{-1.0, -1.0};
  const std::vector<std::string> activeContacts;

  WBMCStepInput input = registry->buildStepInput(
      0.0, q, qdot, activeTasks, taskWeights, activeContacts);

  ASSERT_EQ(input.objectives.size(), 2u);
  ASSERT_TRUE(input.objectives[0].isMotion());
  ASSERT_TRUE(input.objectives[1].isMotion());
  EXPECT_EQ(input.objectives[0].motion().name, "operational-task");
  EXPECT_EQ(input.objectives[0].level, 1u);
  EXPECT_EQ(input.objectives[1].motion().name, "bias-task");
  EXPECT_EQ(input.objectives[1].level, 2u);
}

TEST_F(WBCSolverSemanticsTest, RegistryThrowsOnUnknownActiveTaskName) {
  MockMotionTask task("known-task", *robot);
  task.setSingleDoFTarget(0, 0.5);
  registry->addOperationalTask(task, 1.0);

  const std::vector<std::string> activeTasks{"missing-task"};
  const std::vector<double> taskWeights{-1.0};
  const std::vector<std::string> activeContacts;

  EXPECT_THROW(
      registry->buildStepInput(0.0, q, qdot, activeTasks, taskWeights,
                                activeContacts),
      std::invalid_argument);
}

TEST_F(WBCSolverSemanticsTest, RegistryThrowsOnUnknownActiveContactName) {
  const std::vector<std::string> activeTasks;
  const std::vector<double> taskWeights;
  const std::vector<std::string> activeContacts{"missing-contact"};

  EXPECT_THROW(
      registry->buildStepInput(0.0, q, qdot, activeTasks, taskWeights,
                                activeContacts),
      std::invalid_argument);
}

}  // namespace
