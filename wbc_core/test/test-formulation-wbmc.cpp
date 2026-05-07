//
// Copyright (c) 2026
//
// Public WBMC facade tests.
//

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <vector>

#include <pinocchio/algorithm/joint-configuration.hpp>

#include <wbc_core/controller/wbmc.hpp>
#include <wbc_core/controller/wbmc-registry.hpp>
#include <wbc_core/math/constraint-equality.hpp>
#include <wbc_core/robots/robot-wrapper.hpp>
#include <wbc_core/tasks/task-motion.hpp>

namespace {

using tsid::WBMC;
using tsid::WBMCStepInput;
using tsid::WBMCSolution;
using tsid::math::Matrix;
using tsid::math::Vector;
using tsid::robots::RobotWrapper;

tsid::MotionObjective makeSingleDoFTask(const std::string& name, int nv,
                                        int dof, double a_des) {
  tsid::MotionObjective task;
  task.name = name;
  task.J = Matrix::Zero(1, nv);
  task.J(0, dof) = 1.0;
  task.a_des = Vector::Constant(1, a_des);
  return task;
}

tsid::ContactSnapshot makeForceOnlyContact(const std::string& name, int nv,
                                           double uf_lb, double uf_ub) {
  tsid::ContactSnapshot contact;
  contact.name = name;
  contact.Jc = Matrix::Zero(0, nv);
  contact.Jcdot_qdot = Vector::Zero(0);
  contact.T = Matrix::Zero(0, 1);
  contact.Uf = Matrix::Identity(1, 1);
  contact.uf_lb = Vector::Constant(1, uf_lb);
  contact.uf_ub = Vector::Constant(1, uf_ub);
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

  const tsid::math::ConstraintBase& compute(
      double, ConstRefVector, ConstRefVector, Data&) override {
    return m_constraint;
  }

  const tsid::math::ConstraintBase& getConstraint() const override {
    return m_constraint;
  }

  const Vector& getDesiredAcceleration() const override { return m_a_des; }

 private:
  int m_nv;
  tsid::math::ConstraintEquality m_constraint;
  Vector m_a_des;
};

class PublicWBMCTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const std::vector<std::string> package_dirs{TSID_MODEL_DIR};
    const std::string urdf_file =
        std::string(TSID_MODEL_DIR) + "/romeo/urdf/romeo.urdf";

    robot = std::make_unique<RobotWrapper>(urdf_file, package_dirs);
    wbmc = std::make_unique<WBMC>(*robot);
    registry = std::make_unique<tsid::WBMCRegistry>(*robot);

    q = pinocchio::neutral(robot->model());
    qdot = Vector::Zero(robot->nv());
  }

  WBMCStepInput makeInput() {
    WBMCStepInput input;
    input.q = &q;
    input.qdot = &qdot;
    return input;
  }

  static constexpr double kTol = 1e-7;

  std::unique_ptr<RobotWrapper> robot;
  std::unique_ptr<WBMC> wbmc;
  std::unique_ptr<tsid::WBMCRegistry> registry;
  Vector q;
  Vector qdot;
};

TEST_F(PublicWBMCTest, SolveZeroNominalStepInput) {
  auto input = makeInput();

  const WBMCSolution& solution = wbmc->solve(input);

  ASSERT_TRUE(solution.success);
  EXPECT_TRUE(solution.qddot_ref.isZero(kTol));
  EXPECT_TRUE(solution.delta_qddot.isZero(kTol));
  EXPECT_TRUE(solution.qddot_sol.isZero(kTol));
  EXPECT_EQ(solution.lambda.size(), 0);
}

TEST_F(PublicWBMCTest, PublicWBMCSolvesNominalCenteredHierarchy) {
  auto input = makeInput();
  input.objectives.push_back(tsid::SoftObjective::Motion(
      makeSingleDoFTask("operational", robot->nv(), 0, 1.25), 1u, 1.0));

  Vector qddot_bias = Vector::Constant(robot->nv(), 2.0);
  qddot_bias(0) = -4.0;
  input.objectives.push_back(tsid::SoftObjective::JointAcceleration(
      tsid::JointAccelerationObjective{"joint-bias", &qddot_bias}, 2u, 1.0));

  const WBMCSolution& solution = wbmc->solve(input);

  ASSERT_TRUE(solution.success);
  EXPECT_NEAR(solution.qddot_sol(0), 1.25, kTol);
  ASSERT_GT(robot->nv(), 1);
  EXPECT_NEAR(solution.qddot_sol(1), 2.0, kTol);
}

TEST_F(PublicWBMCTest, PublicWBMCChoosesMinimumNormFeasibleLambda) {
  auto input = makeInput();
  input.contacts.push_back(makeForceOnlyContact("contact", robot->nv(), 1.0, 5.0));

  const WBMCSolution& solution = wbmc->solve(input);

  ASSERT_TRUE(solution.success);
  ASSERT_EQ(solution.lambda.size(), 1);
  EXPECT_NEAR(solution.lambda(0), 1.0, 1e-6);
}

TEST_F(PublicWBMCTest, RegistryBuildsStepInputForPublicWBMC) {
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

  const WBMCSolution& solution = wbmc->solve(input);
  ASSERT_TRUE(solution.success);
  EXPECT_NEAR(solution.qddot_sol(0), 0.75, kTol);
}

}  // namespace
