//
// Copyright (c) 2026
//
// RobotSystem state contract tests.
//

#include <gtest/gtest.h>

#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Geometry>

#include <wbc_core/robots/robot-system.hpp>

namespace {

using wbc::math::Vector;
using wbc::robots::BaseState;
using wbc::robots::GeneralizedState;
using wbc::robots::JointState;
using wbc::robots::RobotSystem;

std::string romeoUrdf() {
  return std::string(TSID_MODEL_DIR) + "/romeo/urdf/romeo.urdf";
}

std::vector<std::string> packageDirs() { return {TSID_MODEL_DIR}; }

RobotSystem makeFixedBaseRobot() {
  return RobotSystem(romeoUrdf(), packageDirs());
}

RobotSystem makeFloatingBaseRobot() {
  return RobotSystem(romeoUrdf(), packageDirs(),
                     pinocchio::JointModelFreeFlyer());
}

JointState makeJointState(const RobotSystem& robot) {
  JointState joint;
  joint.q = Vector::Zero(robot.nq_joints());
  joint.qdot = Vector::Zero(robot.na());
  joint.tau = Vector::Zero(robot.na());
  return joint;
}

BaseState makeBaseState() {
  BaseState base;
  base.pose_world_base = pinocchio::SE3::Identity();
  base.twist_world_base = pinocchio::Motion::Zero();
  return base;
}

TEST(RobotSystemStateTest, FixedBaseDimensionsMatchActuatedState) {
  RobotSystem robot = makeFixedBaseRobot();

  EXPECT_TRUE(robot.is_fixed_base());
  EXPECT_EQ(robot.nq(), robot.nv());
  EXPECT_EQ(robot.nq(), robot.nq_joints());
  EXPECT_EQ(robot.nv(), robot.nv_joints());
  EXPECT_EQ(robot.nv(), robot.na());
  EXPECT_EQ(robot.nv_joints(), robot.na());
}

TEST(RobotSystemStateTest, FixedBaseUpdatePacksJointStateAsGeneralizedState) {
  RobotSystem robot = makeFixedBaseRobot();
  JointState joint = makeJointState(robot);
  joint.q = Vector::LinSpaced(robot.nq_joints(), -0.2, 0.2);
  joint.qdot = Vector::LinSpaced(robot.na(), -0.1, 0.1);
  joint.tau = Vector::LinSpaced(robot.na(), -1.0, 1.0);

  ASSERT_TRUE(robot.isValidJointState(joint));
  robot.updateState(joint);

  EXPECT_EQ(robot.state().joint.q.size(), robot.nq_joints());
  EXPECT_EQ(robot.state().joint.qdot.size(), robot.na());
  EXPECT_EQ(robot.state().joint.tau.size(), robot.na());
  EXPECT_EQ(robot.generalized_q().size(), robot.nq());
  EXPECT_EQ(robot.generalized_v().size(), robot.nv());

  EXPECT_TRUE(robot.generalized_q().isApprox(joint.q));
  EXPECT_TRUE(robot.generalized_v().isApprox(joint.qdot));
  EXPECT_TRUE(robot.tau_actuated().isApprox(joint.tau));
  EXPECT_TRUE(robot.generalized_actuation_force().isApprox(joint.tau));
}

TEST(RobotSystemStateTest, FixedBaseGeneralizedUpdateUnpacksJointState) {
  RobotSystem robot = makeFixedBaseRobot();

  GeneralizedState generalized;
  generalized.q = Vector::LinSpaced(robot.nq(), -0.2, 0.2);
  generalized.v = Vector::LinSpaced(robot.nv(), -0.1, 0.1);
  const Vector tau = Vector::LinSpaced(robot.na(), -1.0, 1.0);

  ASSERT_TRUE(robot.isValidGeneralizedState(generalized));
  robot.setTime(1.5);
  robot.updateState(generalized, tau);

  EXPECT_TRUE(robot.generalized_q().isApprox(generalized.q));
  EXPECT_TRUE(robot.generalized_v().isApprox(generalized.v));
  EXPECT_TRUE(robot.state().joint.q.isApprox(generalized.q));
  EXPECT_TRUE(robot.state().joint.qdot.isApprox(generalized.v));
  EXPECT_TRUE(robot.tau_actuated().isApprox(tau));
  EXPECT_FALSE(robot.state().base.has_value());
  EXPECT_DOUBLE_EQ(robot.time(), 1.5);
}

TEST(RobotSystemStateTest, FloatingBaseDimensionsIncludeFreeFlyer) {
  RobotSystem robot = makeFloatingBaseRobot();

  EXPECT_FALSE(robot.is_fixed_base());
  EXPECT_EQ(robot.nq(), 7 + robot.nq_joints());
  EXPECT_EQ(robot.nv(), 6 + robot.na());
  EXPECT_EQ(robot.nv_joints(), robot.na());
}

TEST(RobotSystemStateTest, FloatingBaseUpdatePacksBaseAndJointState) {
  RobotSystem robot = makeFloatingBaseRobot();
  JointState joint = makeJointState(robot);
  joint.q = Vector::Random(robot.nq_joints());
  joint.qdot = Vector::Random(robot.na());
  joint.tau = Vector::Random(robot.na());

  const Eigen::Vector3d base_translation(0.3, -0.2, 0.7);
  const Eigen::Quaterniond base_quat(
      Eigen::AngleAxisd(0.4, Eigen::Vector3d::UnitZ()) *
      Eigen::AngleAxisd(-0.2, Eigen::Vector3d::UnitY()));
  Eigen::Matrix<double, 6, 1> base_twist;
  base_twist << 0.1, 0.2, 0.3, -0.1, -0.2, -0.3;

  BaseState base;
  base.pose_world_base =
      pinocchio::SE3(base_quat.toRotationMatrix(), base_translation);
  base.twist_world_base = pinocchio::Motion(base_twist);

  ASSERT_TRUE(robot.isValidJointState(joint));
  ASSERT_TRUE(robot.isValidBaseState(base));
  robot.updateState(joint, base);

  EXPECT_EQ(robot.state().joint.q.size(), robot.nq_joints());
  EXPECT_EQ(robot.state().joint.qdot.size(), robot.na());
  EXPECT_EQ(robot.state().joint.tau.size(), robot.na());
  EXPECT_EQ(robot.generalized_q().size(), robot.nq());
  EXPECT_EQ(robot.generalized_v().size(), robot.nv());

  Vector expected_base_pose(7);
  expected_base_pose.head<3>() = base_translation;
  expected_base_pose.segment<4>(3) << base_quat.x(), base_quat.y(),
      base_quat.z(), base_quat.w();

  EXPECT_TRUE(robot.generalized_q().head<7>().isApprox(expected_base_pose));
  EXPECT_TRUE(robot.generalized_q().tail(robot.nq_joints()).isApprox(joint.q));
  EXPECT_TRUE(robot.generalized_v().head<6>().isApprox(base_twist));
  EXPECT_TRUE(robot.generalized_v().tail(robot.na()).isApprox(joint.qdot));
  EXPECT_TRUE(
      robot.generalized_actuation_force().head<6>().isApprox(Vector::Zero(6)));
  EXPECT_TRUE(
      robot.generalized_actuation_force().tail(robot.na()).isApprox(joint.tau));
}

TEST(RobotSystemStateTest,
     FloatingBaseGeneralizedUpdateUnpacksBaseAndJointState) {
  RobotSystem robot = makeFloatingBaseRobot();

  const Eigen::Vector3d base_translation(-0.4, 0.2, 0.9);
  const Eigen::Quaterniond base_quat(
      Eigen::AngleAxisd(-0.3, Eigen::Vector3d::UnitX()) *
      Eigen::AngleAxisd(0.5, Eigen::Vector3d::UnitZ()));
  Eigen::Matrix<double, 6, 1> base_twist;
  base_twist << -0.2, 0.4, -0.6, 0.1, -0.3, 0.5;

  const Vector q_joints = Vector::Random(robot.nq_joints());
  const Vector qdot_joints = Vector::Random(robot.na());
  const Vector tau = Vector::Random(robot.na());

  GeneralizedState generalized;
  generalized.q.setZero(robot.nq());
  generalized.v.setZero(robot.nv());
  generalized.q.head<3>() = base_translation;
  generalized.q.segment<4>(3) << base_quat.x(), base_quat.y(), base_quat.z(),
      base_quat.w();
  generalized.q.tail(robot.nq_joints()) = q_joints;
  generalized.v.head<6>() = base_twist;
  generalized.v.tail(robot.na()) = qdot_joints;

  ASSERT_TRUE(robot.isValidGeneralizedState(generalized));
  robot.setTime(2.75);
  robot.updateState(generalized, tau);

  EXPECT_TRUE(robot.generalized_q().isApprox(generalized.q));
  EXPECT_TRUE(robot.generalized_v().isApprox(generalized.v));
  EXPECT_TRUE(robot.state().joint.q.isApprox(q_joints));
  EXPECT_TRUE(robot.state().joint.qdot.isApprox(qdot_joints));
  EXPECT_TRUE(robot.tau_actuated().isApprox(tau));
  EXPECT_TRUE(robot.baseState().pose_world_base.translation().isApprox(
      base_translation));
  EXPECT_TRUE(robot.baseState().pose_world_base.rotation().isApprox(
      base_quat.toRotationMatrix()));
  EXPECT_TRUE(robot.baseState().twist_world_base.toVector().isApprox(
      base_twist));
  EXPECT_DOUBLE_EQ(robot.time(), 2.75);
}

TEST(RobotSystemStateTest, RejectsFixedBaseUpdateOnFloatingBaseRobot) {
  RobotSystem robot = makeFloatingBaseRobot();
  const JointState joint = makeJointState(robot);

  EXPECT_THROW(robot.updateState(joint), std::invalid_argument);
}

TEST(RobotSystemStateTest, RejectsFloatingBaseUpdateOnFixedBaseRobot) {
  RobotSystem robot = makeFixedBaseRobot();
  const JointState joint = makeJointState(robot);
  const BaseState base = makeBaseState();

  EXPECT_THROW(robot.updateState(joint, base), std::invalid_argument);
}

TEST(RobotSystemStateTest, RejectsWrongJointPositionSize) {
  RobotSystem robot = makeFixedBaseRobot();
  JointState joint = makeJointState(robot);
  joint.q = Vector::Zero(robot.nq_joints() + 1);

  EXPECT_FALSE(robot.isValidJointState(joint));
  EXPECT_THROW(robot.updateState(joint), std::invalid_argument);
}

TEST(RobotSystemStateTest, RejectsWrongJointVelocitySize) {
  RobotSystem robot = makeFixedBaseRobot();
  JointState joint = makeJointState(robot);
  joint.qdot = Vector::Zero(robot.na() + 1);

  EXPECT_FALSE(robot.isValidJointState(joint));
  EXPECT_THROW(robot.updateState(joint), std::invalid_argument);
}

TEST(RobotSystemStateTest, RejectsWrongTorqueSize) {
  RobotSystem robot = makeFixedBaseRobot();
  JointState joint = makeJointState(robot);
  joint.tau = Vector::Zero(robot.na() + 1);

  EXPECT_FALSE(robot.isValidJointState(joint));
  EXPECT_THROW(robot.updateState(joint), std::invalid_argument);
}

TEST(RobotSystemStateTest, RejectsNonFiniteJointState) {
  RobotSystem robot = makeFixedBaseRobot();
  JointState joint = makeJointState(robot);
  joint.q(0) = std::numeric_limits<double>::quiet_NaN();

  EXPECT_FALSE(robot.isValidJointState(joint));
  EXPECT_THROW(robot.updateState(joint), std::invalid_argument);
}

TEST(RobotSystemStateTest, RejectsNonFiniteBaseState) {
  RobotSystem robot = makeFloatingBaseRobot();
  const JointState joint = makeJointState(robot);
  BaseState base = makeBaseState();
  base.pose_world_base.translation()(0) =
      std::numeric_limits<double>::quiet_NaN();

  EXPECT_FALSE(robot.isValidBaseState(base));
  EXPECT_THROW(robot.updateState(joint, base), std::invalid_argument);
}

TEST(RobotSystemStateTest, RejectsWrongGeneralizedStateSize) {
  RobotSystem robot = makeFixedBaseRobot();
  GeneralizedState generalized;
  generalized.q = Vector::Zero(robot.nq() + 1);
  generalized.v = Vector::Zero(robot.nv());

  EXPECT_FALSE(robot.isValidGeneralizedState(generalized));
  EXPECT_THROW(robot.updateState(generalized), std::invalid_argument);
}

TEST(RobotSystemStateTest, RejectsNonFiniteGeneralizedState) {
  RobotSystem robot = makeFixedBaseRobot();
  GeneralizedState generalized;
  generalized.q = Vector::Zero(robot.nq());
  generalized.v = Vector::Zero(robot.nv());
  generalized.v(0) = std::numeric_limits<double>::infinity();

  EXPECT_FALSE(robot.isValidGeneralizedState(generalized));
  EXPECT_THROW(robot.updateState(generalized), std::invalid_argument);
}

TEST(RobotSystemStateTest, RejectsNonFiniteSystemTime) {
  RobotSystem robot = makeFixedBaseRobot();

  EXPECT_THROW(robot.setTime(std::numeric_limits<double>::quiet_NaN()),
               std::invalid_argument);
}

}  // namespace
