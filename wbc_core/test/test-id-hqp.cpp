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
#include <wbc_core/controller/base/id-problem-registry.hpp>
#include <wbc_core/math/constraints/constraint-equality.hpp>
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
  contact.motion_rhs = Vector::Zero(0);
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
    qdot = Vector::Zero(robot->nv_joints());
  }

  JointState makeJointState(const Vector& q_in, const Vector& qdot_in,
                            const Vector& tau_in) const {
    JointState joint;
    joint.q = q_in;
    joint.qdot = qdot_in;
    joint.tau = tau_in;
    return joint;
  }

  IDProblem makeProblem() {
    robot->updateState(makeJointState(q, qdot, Vector::Zero(robot->na())));
    IDProblem problem;
    return problem;
  }

  wbc::MotionObjective makeSingleDoFObjective(const std::string& name, int dof,
                                              double a_des, unsigned int level,
                                              double weight) {
    motion_constraints.push_back(
        makeSingleDoFConstraint(name, robot->nv(), dof, a_des));
    return wbc::MotionObjective{name, motion_constraints.back().get(), level,
                                weight};
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

TEST_F(PublicIDHQPTest, RobotSystemStoresActuatedTauAndSystemTime) {
  const Vector tau = Vector::LinSpaced(robot->na(), -0.5, 0.5);
  const JointState joint = makeJointState(q, qdot, tau);

  robot->setTime(1.25);
  robot->updateState(joint);

  EXPECT_TRUE(robot->jointState().q.isApprox(q));
  EXPECT_TRUE(robot->jointState().qdot.isApprox(qdot));
  EXPECT_TRUE(robot->tau_actuated().isApprox(tau));
  EXPECT_TRUE(robot->generalized_actuation_force().isApprox(tau));
  EXPECT_TRUE(robot->generalized_q().isApprox(q));
  EXPECT_TRUE(robot->generalized_v().isApprox(qdot));
  EXPECT_TRUE(robot->state().joint.tau.isApprox(tau));
  EXPECT_DOUBLE_EQ(robot->time(), 1.25);

  const Vector next_qdot = Vector::Constant(robot->nv_joints(), 0.1);
  const Vector next_tau = Vector::Constant(robot->na(), 0.2);

  robot->updateState(makeJointState(q, next_qdot, next_tau));

  EXPECT_TRUE(robot->generalized_v().isApprox(next_qdot));
  EXPECT_TRUE(robot->tau_actuated().isApprox(next_tau));
  EXPECT_TRUE(robot->generalized_actuation_force().isApprox(next_tau));
  EXPECT_DOUBLE_EQ(robot->time(), 1.25);
}

TEST_F(PublicIDHQPTest, RobotSystemPacksFloatingBaseState) {
  const std::vector<std::string> package_dirs{TSID_MODEL_DIR};
  const std::string urdf_file =
      std::string(TSID_MODEL_DIR) + "/romeo/urdf/romeo.urdf";
  RobotSystem floating_robot(urdf_file, package_dirs,
                             pinocchio::JointModelFreeFlyer());

  const Vector q_joints =
      Vector::LinSpaced(floating_robot.nq_joints(), -0.2, 0.2);
  const Vector qdot_joints =
      Vector::LinSpaced(floating_robot.nv_joints(), -0.1, 0.1);
  const Vector tau = Vector::LinSpaced(floating_robot.na(), -1.0, 1.0);

  BaseState base;
  base.pose_world_base =
      pinocchio::SE3(Eigen::Matrix3d::Identity(),
                     Eigen::Vector3d(0.3, -0.2, 0.7));
  Eigen::Matrix<double, 6, 1> base_twist;
  base_twist << 0.1, 0.2, 0.3, -0.1, -0.2, -0.3;
  base.twist_world_base = pinocchio::Motion(base_twist);

  JointState joint;
  joint.q = q_joints;
  joint.qdot = qdot_joints;
  joint.tau = tau;
  floating_robot.setTime(2.5);
  floating_robot.updateState(joint, base);

  ASSERT_EQ(floating_robot.generalized_q().size(), floating_robot.nq());
  ASSERT_EQ(floating_robot.generalized_v().size(), floating_robot.nv());
  EXPECT_TRUE(floating_robot.jointState().q.isApprox(q_joints));
  EXPECT_TRUE(floating_robot.baseState().pose_world_base.isApprox(
      base.pose_world_base));
  EXPECT_TRUE(floating_robot.generalized_q().head<3>().isApprox(
      base.pose_world_base.translation()));
  EXPECT_TRUE(floating_robot.generalized_q().segment<4>(3).isApprox(
      Vector::Unit(4, 3)));
  EXPECT_TRUE(floating_robot.generalized_q()
                  .tail(floating_robot.nq_joints())
                  .isApprox(q_joints));
  EXPECT_TRUE(floating_robot.generalized_v().head<6>().isApprox(
      base.twist_world_base.toVector()));
  EXPECT_TRUE(floating_robot.generalized_v()
                  .tail(floating_robot.nv_joints())
                  .isApprox(qdot_joints));
  EXPECT_TRUE(floating_robot.tau_actuated().isApprox(tau));
  EXPECT_TRUE(floating_robot.generalized_actuation_force()
                  .head<6>()
                  .isApprox(Vector::Zero(6)));
  EXPECT_TRUE(floating_robot.generalized_actuation_force()
                  .tail(floating_robot.na())
                  .isApprox(tau));
  EXPECT_DOUBLE_EQ(floating_robot.time(), 2.5);
}

TEST_F(PublicIDHQPTest, SolveZeroNominalProblem) {
  auto problem = makeProblem();

  const IDSolution& solution = id_hqp->solve(problem, kDt);

  ASSERT_TRUE(solution.success);
  EXPECT_TRUE(solution.qddot_ref.isZero(kTol));
  EXPECT_TRUE(solution.delta_qddot_sol.isZero(kTol));
  EXPECT_TRUE(solution.qddot_sol.isZero(kTol));
  EXPECT_EQ(solution.lambda_sol.size(), 0);
  EXPECT_EQ(solution.tau_sol.size(), robot->na());
}

TEST_F(PublicIDHQPTest, PublicIDHQPSolvesNominalCenteredHierarchy) {
  auto problem = makeProblem();
  const double dt = 0.002;
  problem.motion_objectives.push_back(
      makeSingleDoFObjective("operational", 0, 1.25, 1u, 1.0));

  Vector qddot_bias = Vector::Constant(robot->nv(), 2.0);
  qddot_bias(0) = -4.0;
  problem.joint_acceleration_objectives.emplace_back("joint-bias", &qddot_bias,
                                                     2u, 1.0);

  const IDSolution& solution = id_hqp->solve(problem, dt);

  ASSERT_TRUE(solution.success);
  EXPECT_NEAR(solution.qddot_sol(0), 1.25, kTol);
  ASSERT_GT(robot->nv(), 1);
  EXPECT_NEAR(solution.qddot_sol(1), 2.0, kTol);
  EXPECT_TRUE(solution.tau_sol.allFinite());
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
  registry->registerTask(task, 1u, 2.0);

  IDProblem problem = registry->buildProblem(0.0, q, qdot);
  ASSERT_EQ(problem.motion_objectives.size(), 1u);
  EXPECT_EQ(problem.motion_objectives[0].name, "mock-task");
  EXPECT_DOUBLE_EQ(problem.motion_objectives[0].weight, 2.0);
  EXPECT_NEAR(problem.motion_objectives[0].matrix()(0, 0), 1.0,
              kTol);
  EXPECT_NEAR(problem.motion_objectives[0].vector()(0), 0.75, kTol);

  const IDSolution& solution = id_hqp->solve(problem, kDt);
  ASSERT_TRUE(solution.success);
  EXPECT_NEAR(solution.qddot_sol(0), 0.75, kTol);
}

}  // namespace
