//
// Copyright (c) 2026
//
// Public IDHQP facade tests.
//

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <vector>

#include <pinocchio/algorithm/joint-configuration.hpp>

#include <wbc_core/controller/id-hqp.hpp>
#include <wbc_core/controller/id-problem-registry.hpp>
#include <wbc_core/math/constraint-equality.hpp>
#include <wbc_core/robots/robot-system.hpp>
#include <wbc_core/tasks/task-motion.hpp>

namespace {

using wbc::IDHQP;
using wbc::IDProblem;
using wbc::IDSolution;
using wbc::math::Matrix;
using wbc::math::Vector;
using wbc::robots::RobotSystem;

std::shared_ptr<wbc::math::ConstraintBase> makeSingleDoFConstraint(
    const std::string& name, int nv, int dof, double a_des) {
  auto constraint =
      std::make_shared<wbc::math::ConstraintEquality>(name, 1, nv);
  constraint->matrix().setZero();
  constraint->matrix()(0, dof) = 1.0;
  constraint->vector() = Vector::Constant(1, a_des);
  return constraint;
}

wbc::ContactConstraintData makeForceOnlyContact(const std::string& name, int nv,
                                                double uf_lb, double uf_ub) {
  wbc::ContactConstraintData contact;
  contact.name = name;
  contact.Jc = Matrix::Zero(0, nv);
  contact.Jcdot_qdot = Vector::Zero(0);
  contact.T = Matrix::Zero(0, 1);
  contact.Uf = Matrix::Identity(1, 1);
  contact.uf_lb = Vector::Constant(1, uf_lb);
  contact.uf_ub = Vector::Constant(1, uf_ub);
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
    m_a_des = Vector::Zero(1);
  }

  void setSingleDoFTarget(int dof, double value) {
    m_constraint.resize(1, m_nv);
    m_constraint.matrix().setZero();
    m_constraint.matrix()(0, dof) = 1.0;
    m_constraint.vector().setConstant(value);
    m_a_des(0) = value;
  }

  int dim() const override { return static_cast<int>(m_constraint.rows()); }

  const wbc::math::ConstraintBase& compute(double, ConstRefVector,
                                           ConstRefVector, Data&) override {
    return m_constraint;
  }

  const wbc::math::ConstraintBase& getConstraint() const override {
    return m_constraint;
  }

  const Vector& getDesiredAcceleration() const override { return m_a_des; }

 private:
  int m_nv;
  wbc::math::ConstraintEquality m_constraint;
  Vector m_a_des;
};

class PublicIDHQPTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const std::vector<std::string> package_dirs{TSID_MODEL_DIR};
    const std::string urdf_file =
        std::string(TSID_MODEL_DIR) + "/romeo/urdf/romeo.urdf";

    robot = std::make_unique<RobotSystem>(urdf_file, package_dirs);
    id_hqp = std::make_unique<IDHQP>(*robot);
    registry = std::make_unique<wbc::IDProblemRegistry>(*robot);

    q = pinocchio::neutral(robot->model());
    qdot = Vector::Zero(robot->nv());
  }

  IDProblem makeProblem() {
    robot->updateState(q, qdot);
    IDProblem problem;
    return problem;
  }

  wbc::ObjectiveTerm makeSingleDoFObjective(const std::string& name, int dof,
                                            double a_des, unsigned int level,
                                            double weight) {
    motion_constraints.push_back(
        makeSingleDoFConstraint(name, robot->nv(), dof, a_des));
    return wbc::ObjectiveTerm::MakeMotionConstraint(
        wbc::MotionConstraintRef{name, motion_constraints.back().get()}, level,
        weight);
  }

  static constexpr double kTol = 1e-5;
  static constexpr double kDt = 0.001;

  std::unique_ptr<RobotSystem> robot;
  std::unique_ptr<IDHQP> id_hqp;
  std::unique_ptr<wbc::IDProblemRegistry> registry;
  std::vector<std::shared_ptr<wbc::math::ConstraintBase>> motion_constraints;
  Vector q;
  Vector qdot;
};

TEST_F(PublicIDHQPTest, SolveZeroNominalProblem) {
  auto problem = makeProblem();

  const IDSolution& solution = id_hqp->solve(problem, kDt);

  ASSERT_TRUE(solution.success);
  EXPECT_TRUE(solution.qddot_ref.isZero(kTol));
  EXPECT_TRUE(solution.delta_qddot.isZero(kTol));
  EXPECT_TRUE(solution.qddot_sol.isZero(kTol));
  EXPECT_TRUE(solution.qdot_cmd.isApprox(qdot, kTol));
  EXPECT_TRUE(solution.q_cmd.isApprox(q, kTol));
  EXPECT_EQ(solution.lambda_sol.size(), 0);
}

TEST_F(PublicIDHQPTest, PublicIDHQPSolvesNominalCenteredHierarchy) {
  auto problem = makeProblem();
  const double dt = 0.002;
  problem.objectives.push_back(
      makeSingleDoFObjective("operational", 0, 1.25, 1u, 1.0));

  Vector qddot_bias = Vector::Constant(robot->nv(), 2.0);
  qddot_bias(0) = -4.0;
  problem.objectives.push_back(wbc::ObjectiveTerm::MakeJointAccelerationTarget(
      wbc::JointAccelerationTarget{"joint-bias", &qddot_bias}, 2u, 1.0));

  const IDSolution& solution = id_hqp->solve(problem, dt);

  ASSERT_TRUE(solution.success);
  EXPECT_NEAR(solution.qddot_sol(0), 1.25, kTol);
  ASSERT_GT(robot->nv(), 1);
  EXPECT_NEAR(solution.qddot_sol(1), 2.0, kTol);
  const Vector expected_qdot = qdot + dt * solution.qddot_sol;
  Vector expected_delta = dt * expected_qdot;
  Vector expected_q(robot->nq());
  pinocchio::integrate(robot->model(), q, expected_delta, expected_q);
  EXPECT_TRUE(solution.qdot_cmd.isApprox(expected_qdot, kTol));
  EXPECT_TRUE(solution.q_cmd.isApprox(expected_q, kTol));
}

TEST_F(PublicIDHQPTest, PublicIDHQPChoosesMinimumNormFeasibleLambda) {
  auto problem = makeProblem();
  problem.contacts.push_back(
      makeForceOnlyContact("contact", robot->nv(), 1.0, 5.0));

  const IDSolution& solution = id_hqp->solve(problem, kDt);

  ASSERT_TRUE(solution.success);
  ASSERT_EQ(solution.lambda_sol.size(), 1);
  EXPECT_NEAR(solution.lambda_sol(0), 1.0, kTol);
}

TEST_F(PublicIDHQPTest, RegistryBuildsProblemForPublicIDHQP) {
  MockMotionTask task("mock-task", *robot);
  task.setSingleDoFTarget(0, 0.75);
  registry->addOperationalTask(task, 2.0);

  IDProblem problem = registry->buildProblem(0.0, q, qdot);
  ASSERT_EQ(problem.objectives.size(), 1u);
  ASSERT_TRUE(problem.objectives[0].isMotionConstraint());
  EXPECT_EQ(problem.objectives[0].motionConstraint().name, "mock-task");
  EXPECT_DOUBLE_EQ(problem.objectives[0].weight, 2.0);
  EXPECT_NEAR(problem.objectives[0].motionConstraint().matrix()(0, 0), 1.0,
              kTol);
  EXPECT_NEAR(problem.objectives[0].motionConstraint().vector()(0), 0.75, kTol);

  const IDSolution& solution = id_hqp->solve(problem, kDt);
  ASSERT_TRUE(solution.success);
  EXPECT_NEAR(solution.qddot_sol(0), 0.75, kTol);
}

}  // namespace
