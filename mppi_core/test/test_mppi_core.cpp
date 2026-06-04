// Copyright 2026
//
// Licensed under the Apache License, Version 2.0.

#include <gtest/gtest.h>

#include <cmath>
#include <limits>

#include <Eigen/Core>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/multibody/frame.hpp>
#include <pinocchio/multibody/joint/joint-prismatic.hpp>
#include <pinocchio/multibody/joint/joint-revolute.hpp>
#include <pinocchio/multibody/joint/joint-revolute-unbounded.hpp>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/spatial/se3.hpp>
#include <yaml-cpp/yaml.h>

#include "mppi_core/config/grasp_config.hpp"
#include "mppi_core/config/mppi_config.hpp"
#include "mppi_core/core/mppi_optimizer.hpp"
#include "mppi_core/costs/grasp_stability_cost.hpp"
#include "mppi_core/grasp/contact_force_correction.hpp"
#include "mppi_core/grasp/contact_force_projection.hpp"
#include "mppi_core/grasp/contact_force_rollout.hpp"
#include "mppi_core/grasp/grasp_contact_kinematics.hpp"
#include "mppi_core/grasp/grasp_rollout.hpp"
#include "mppi_core/grasp/grasp_state.hpp"
#include "mppi_core/grasp_types.hpp"
#include "mppi_core/model/delta_q_reference_rollout_model.hpp"
#include "mppi_core/robot_command.hpp"
#include "mppi_core/tactile/nari_touch_adapter.hpp"

namespace {

constexpr double kTolerance = 1.0e-9;

mppi_core::RobotRolloutState MakeState(std::size_t dim,
                                       const mppi_core::TactileState& tactile) {
  const Eigen::VectorXd zero =
      Eigen::VectorXd::Zero(static_cast<Eigen::Index>(dim));
  return mppi_core::MakeGraspState(zero, zero, zero, tactile);
}

mppi_core::TactileState ToTestTactileState(
    const mppi_core::NariTouchState& nari, double slip_velocity_weight = 0.0) {
  mppi_core::NariTouchAdapterConfig config;
  config.slip_velocity_weight = slip_velocity_weight;
  return mppi_core::ConvertNariTouchToTactileState(nari, "test_tactile", 1.0,
                                                   config);
}

struct TestPinocchioSensorModel {
  pinocchio::Model model;
  pinocchio::FrameIndex sensor_frame_id{0};
};

TestPinocchioSensorModel MakeSingleRevoluteZSensorModel() {
  TestPinocchioSensorModel out;
  const auto joint_id = out.model.addJoint(
      0, pinocchio::JointModelRZ(), pinocchio::SE3::Identity(), "finger_rz");
  out.sensor_frame_id = out.model.addFrame(
      pinocchio::Frame("tactile_sensor", joint_id, 0,
                       pinocchio::SE3::Identity(), pinocchio::OP_FRAME));
  return out;
}

TestPinocchioSensorModel MakeSinglePrismaticZSensorModel() {
  TestPinocchioSensorModel out;
  const auto joint_id = out.model.addJoint(
      0, pinocchio::JointModelPZ(), pinocchio::SE3::Identity(), "finger_pz");
  out.sensor_frame_id = out.model.addFrame(
      pinocchio::Frame("tactile_sensor", joint_id, 0,
                       pinocchio::SE3::Identity(), pinocchio::OP_FRAME));
  return out;
}

TestPinocchioSensorModel MakeSingleUnboundedRevoluteZSensorModel() {
  TestPinocchioSensorModel out;
  const auto joint_id = out.model.addJoint(
      0, pinocchio::JointModelRUBZ(), pinocchio::SE3::Identity(),
      "finger_rubz");
  out.sensor_frame_id = out.model.addFrame(
      pinocchio::Frame("tactile_sensor", joint_id, 0,
                       pinocchio::SE3::Identity(), pinocchio::OP_FRAME));
  return out;
}

mppi_core::GraspState MakeContactKinematicsState(
    const pinocchio::Model& model, const mppi_core::TactileState& tactile) {
  return mppi_core::MakeGraspState(pinocchio::neutral(model),
                                   Eigen::VectorXd::Zero(model.nv),
                                   Eigen::VectorXd::Zero(model.nv), tactile);
}

Eigen::Vector3d WorldPointPosition(const pinocchio::Model& model,
                                   pinocchio::Data* data,
                                   pinocchio::FrameIndex sensor_frame_id,
                                   const Eigen::VectorXd& q,
                                   const Eigen::Vector3d& point_sensor_m) {
  pinocchio::forwardKinematics(model, *data, q);
  pinocchio::updateFramePlacements(model, *data);
  return data->oMf[sensor_frame_id].act(point_sensor_m);
}

}  // namespace

TEST(TactileStateTest, ContactPointsDefaultEmptyAndCountActiveOnly) {
  mppi_core::TactileState tactile;
  EXPECT_TRUE(tactile.contact_points.empty());
  EXPECT_EQ(tactile.activeContactPointCount(), 0U);

  mppi_core::TactileContactPoint inactive;
  inactive.active = false;
  mppi_core::TactileContactPoint active;
  active.active = true;
  tactile.contact_points.push_back(inactive);
  tactile.contact_points.push_back(active);
  tactile.contact_points.push_back(inactive);

  EXPECT_EQ(tactile.activeContactPointCount(), 1U);
}

TEST(RobotCommandTest, ResizeAndValidationSupportNqNvSplit) {
  mppi_core::RobotCommand command;
  command.Resize(7, 6);
  command.valid = true;

  EXPECT_TRUE(command.HasValidDimensions());
  EXPECT_TRUE(command.AllFinite());
  EXPECT_TRUE(command.IsUsable());
  EXPECT_EQ(command.q_des.size(), 7);
  EXPECT_EQ(command.qdot_des.size(), 6);
  EXPECT_EQ(command.qddot_des.size(), 6);
  EXPECT_EQ(command.tau_ff.size(), 6);
  EXPECT_EQ(command.kp.size(), 6);
  EXPECT_EQ(command.kd.size(), 6);
  EXPECT_EQ(command.delta_q_ref.size(), 6);
  EXPECT_EQ(command.delta_tau_ref.size(), 6);
  EXPECT_EQ(command.tau_raw.size(), 6);
  EXPECT_EQ(command.tau_limited.size(), 6);

  command.tau_raw = Eigen::VectorXd::Zero(5);
  EXPECT_FALSE(command.HasValidDimensions());
  EXPECT_FALSE(command.IsUsable());
}

TEST(RobotCommandTest, ZeroHoldAndInvalidHelpersSetUsability) {
  const Eigen::VectorXd q_current = Eigen::VectorXd::Constant(2, 0.3);
  const Eigen::VectorXd qdot_current = Eigen::VectorXd::Constant(2, 0.4);
  const Eigen::VectorXd kp_hold = Eigen::VectorXd::Constant(2, 20.0);
  const Eigen::VectorXd kd_hold = Eigen::VectorXd::Constant(2, 2.0);

  const auto hold = mppi_core::MakeZeroHoldRobotCommand(
      q_current, qdot_current, kp_hold, kd_hold);

  EXPECT_TRUE(hold.valid);
  EXPECT_TRUE(hold.IsUsable());
  EXPECT_NEAR(hold.q_des[0], 0.3, kTolerance);
  EXPECT_NEAR(hold.qdot_des.norm(), 0.0, kTolerance);
  EXPECT_NEAR(hold.qddot_des.norm(), 0.0, kTolerance);
  EXPECT_NEAR(hold.tau_ff.norm(), 0.0, kTolerance);
  EXPECT_NEAR(hold.kp[0], 20.0, kTolerance);
  EXPECT_NEAR(hold.kd[0], 2.0, kTolerance);

  const auto invalid = mppi_core::MakeInvalidRobotCommand(3, 2);
  EXPECT_FALSE(invalid.valid);
  EXPECT_TRUE(invalid.HasValidDimensions());
  EXPECT_TRUE(invalid.AllFinite());
  EXPECT_FALSE(invalid.IsUsable());
}

TEST(GraspStateTest, MakeGraspStateValidatesDimensionsAndTactileValidity) {
  mppi_core::TactileState tactile;
  tactile.valid = true;
  tactile.contact_presence = mppi_core::ContactPresence::kStableContact;
  tactile.has_normal_force = true;
  tactile.normal_force_n = 1.0;

  const Eigen::VectorXd q = Eigen::VectorXd::Constant(2, 0.25);
  const Eigen::VectorXd dq = Eigen::VectorXd::Constant(2, -0.5);
  const Eigen::VectorXd tau = Eigen::VectorXd::Constant(2, 1.25);

  const auto valid_state = mppi_core::MakeGraspState(q, dq, tau, tactile);
  EXPECT_TRUE(valid_state.valid);
  EXPECT_EQ(valid_state.q.size(), 2);
  EXPECT_EQ(valid_state.dq.size(), 2);
  EXPECT_EQ(valid_state.tau.size(), 2);
  EXPECT_NEAR(valid_state.q[0], 0.25, kTolerance);
  EXPECT_NEAR(valid_state.dq[1], -0.5, kTolerance);
  EXPECT_NEAR(valid_state.tau[0], 1.25, kTolerance);
  EXPECT_TRUE(valid_state.tactile.valid);
  EXPECT_NEAR(valid_state.tactile.normal_force_n, 1.0, kTolerance);

  tactile.valid = false;
  const auto invalid_tactile_state =
      mppi_core::MakeGraspState(q, dq, tau, tactile);
  EXPECT_FALSE(invalid_tactile_state.valid);

  tactile.valid = true;
  const Eigen::VectorXd mismatched_dq = Eigen::VectorXd::Zero(3);
  const auto mismatched_state =
      mppi_core::MakeGraspState(q, mismatched_dq, tau, tactile);
  EXPECT_FALSE(mismatched_state.valid);
  EXPECT_EQ(mismatched_state.q.size(), 2);
  EXPECT_EQ(mismatched_state.dq.size(), 3);

  const Eigen::VectorXd wrong_tau = Eigen::VectorXd::Zero(3);
  const auto bad_torque_state =
      mppi_core::MakeGraspState(q, dq, wrong_tau, tactile);
  EXPECT_FALSE(bad_torque_state.valid);

  const Eigen::VectorXd q_nq_not_nv = Eigen::VectorXd::Constant(3, 0.2);
  const auto pinocchio_shaped_state =
      mppi_core::MakeGraspState(q_nq_not_nv, dq, tau, tactile);
  EXPECT_TRUE(pinocchio_shaped_state.valid);
  EXPECT_EQ(pinocchio_shaped_state.q.size(), 3);
  EXPECT_EQ(pinocchio_shaped_state.dq.size(), 2);
  EXPECT_EQ(pinocchio_shaped_state.tau.size(), 2);
}

TEST(GraspStateTest, MakeGraspStateRejectsNonFiniteJointVectors) {
  mppi_core::TactileState tactile;
  tactile.valid = true;
  tactile.contact_presence = mppi_core::ContactPresence::kStableContact;

  Eigen::VectorXd q = Eigen::VectorXd::Zero(2);
  Eigen::VectorXd dq = Eigen::VectorXd::Zero(2);
  Eigen::VectorXd tau = Eigen::VectorXd::Zero(2);

  q[0] = std::numeric_limits<double>::quiet_NaN();
  EXPECT_FALSE(mppi_core::MakeGraspState(q, dq, tau, tactile).valid);

  q[0] = 0.0;
  dq[1] = std::numeric_limits<double>::infinity();
  EXPECT_FALSE(mppi_core::MakeGraspState(q, dq, tau, tactile).valid);

  dq[1] = 0.0;
  tau[0] = std::numeric_limits<double>::quiet_NaN();
  const auto bad_torque_state =
      mppi_core::MakeGraspState(q, dq, tau, tactile);
  EXPECT_FALSE(bad_torque_state.valid);
}

TEST(GraspContactKinematicsTest, EmptyOrInactiveContactPointsReturnEmpty) {
  const auto sensor_model = MakeSingleRevoluteZSensorModel();
  pinocchio::Data data(sensor_model.model);

  mppi_core::TactileState tactile;
  tactile.valid = true;
  tactile.contact_presence = mppi_core::ContactPresence::kStableContact;

  const auto state = MakeContactKinematicsState(sensor_model.model, tactile);
  const Eigen::VectorXd delta_q = Eigen::VectorXd::Constant(1, 0.1);
  mppi_core::PinocchioContactKinematicsContext context;
  context.model = &sensor_model.model;
  context.data = &data;
  context.sensor_frame_id = sensor_model.sensor_frame_id;

  EXPECT_TRUE(
      mppi_core::ComputeContactPointMotions(state, delta_q, context).empty());

  mppi_core::TactileContactPoint inactive;
  inactive.active = false;
  inactive.position_sensor_m = Eigen::Vector3d{1.0, 0.0, 0.0};
  tactile.contact_points.push_back(inactive);
  const auto inactive_state =
      MakeContactKinematicsState(sensor_model.model, tactile);
  EXPECT_TRUE(
      mppi_core::ComputeContactPointMotions(inactive_state, delta_q, context)
          .empty());
}

TEST(GraspContactKinematicsTest, ActiveContactPointProducesSensorFrameMotion) {
  const auto sensor_model = MakeSingleRevoluteZSensorModel();
  pinocchio::Data data(sensor_model.model);

  mppi_core::TactileState tactile;
  tactile.valid = true;
  tactile.contact_presence = mppi_core::ContactPresence::kStableContact;
  mppi_core::TactileContactPoint point;
  point.active = true;
  point.support_index = 7;
  point.position_sensor_m = Eigen::Vector3d{1.0, 0.0, 0.0};
  tactile.contact_points.push_back(point);

  const auto state = MakeContactKinematicsState(sensor_model.model, tactile);
  const Eigen::VectorXd delta_q = Eigen::VectorXd::Constant(1, 0.25);
  mppi_core::PinocchioContactKinematicsContext context;
  context.model = &sensor_model.model;
  context.data = &data;
  context.sensor_frame_id = sensor_model.sensor_frame_id;

  const auto motions =
      mppi_core::ComputeContactPointMotions(state, delta_q, context);

  ASSERT_EQ(motions.size(), 1U);
  EXPECT_EQ(motions[0].support_index, 7U);
  EXPECT_NEAR(motions[0].position_sensor_m.x(), 1.0, kTolerance);
  EXPECT_NEAR(motions[0].position_sensor_m.y(), 0.0, kTolerance);
  EXPECT_NEAR(motions[0].position_sensor_m.z(), 0.0, kTolerance);
  EXPECT_NEAR(motions[0].delta_position_sensor_m.x(), 0.0, kTolerance);
  EXPECT_NEAR(motions[0].delta_position_sensor_m.y(), 0.25, kTolerance);
  EXPECT_NEAR(motions[0].delta_position_sensor_m.z(), 0.0, kTolerance);
}

TEST(GraspContactKinematicsTest, PointJacobianMatchesFiniteDifference) {
  const auto sensor_model = MakeSingleRevoluteZSensorModel();
  pinocchio::Data data(sensor_model.model);
  pinocchio::Data finite_difference_data(sensor_model.model);

  mppi_core::TactileState tactile;
  tactile.valid = true;
  tactile.contact_presence = mppi_core::ContactPresence::kStableContact;
  mppi_core::TactileContactPoint point;
  point.active = true;
  point.position_sensor_m = Eigen::Vector3d{0.7, -0.2, 0.1};
  tactile.contact_points.push_back(point);

  auto state = MakeContactKinematicsState(sensor_model.model, tactile);
  state.q[0] = 0.35;
  const double eps = 1.0e-6;
  const Eigen::VectorXd delta_q_tangent = Eigen::VectorXd::Constant(1, 0.7);
  mppi_core::PinocchioContactKinematicsContext context;
  context.model = &sensor_model.model;
  context.data = &data;
  context.sensor_frame_id = sensor_model.sensor_frame_id;

  const auto motions =
      mppi_core::ComputeContactPointMotions(state, delta_q_tangent, context);

  ASSERT_EQ(motions.size(), 1U);
  pinocchio::forwardKinematics(sensor_model.model, finite_difference_data,
                               state.q);
  pinocchio::updateFramePlacements(sensor_model.model, finite_difference_data);
  const Eigen::Matrix3d current_sensor_rotation =
      finite_difference_data.oMf[sensor_model.sensor_frame_id].rotation();
  const Eigen::Vector3d point_world_before = WorldPointPosition(
      sensor_model.model, &finite_difference_data, sensor_model.sensor_frame_id,
      state.q, point.position_sensor_m);
  const Eigen::VectorXd q_next =
      pinocchio::integrate(sensor_model.model, state.q, eps * delta_q_tangent);
  const Eigen::Vector3d point_world_after = WorldPointPosition(
      sensor_model.model, &finite_difference_data, sensor_model.sensor_frame_id,
      q_next, point.position_sensor_m);
  const Eigen::Vector3d finite_difference_delta_sensor =
      current_sensor_rotation.transpose() *
      (point_world_after - point_world_before);

  EXPECT_NEAR((eps * motions[0].delta_position_sensor_m -
               finite_difference_delta_sensor)
                  .norm(),
              0.0, 1.0e-9);
}

TEST(GraspContactKinematicsTest,
     NormalAxisSignKeepsPositiveZAsClosingConvention) {
  const auto sensor_model = MakeSinglePrismaticZSensorModel();
  pinocchio::Data data(sensor_model.model);

  mppi_core::TactileState tactile;
  tactile.valid = true;
  tactile.contact_presence = mppi_core::ContactPresence::kStableContact;
  mppi_core::TactileContactPoint point;
  point.active = true;
  point.position_sensor_m = Eigen::Vector3d::Zero();
  tactile.contact_points.push_back(point);

  const auto state = MakeContactKinematicsState(sensor_model.model, tactile);
  const Eigen::VectorXd delta_q_tangent = Eigen::VectorXd::Constant(1, 0.01);
  mppi_core::PinocchioContactKinematicsContext context;
  context.model = &sensor_model.model;
  context.data = &data;
  context.sensor_frame_id = sensor_model.sensor_frame_id;

  const auto closing_positive =
      mppi_core::ComputeContactPointMotions(state, delta_q_tangent, context);
  ASSERT_EQ(closing_positive.size(), 1U);
  EXPECT_GT(closing_positive[0].delta_position_sensor_m.z(), 0.0);

  const Eigen::VectorXd opening_delta_q_tangent =
      Eigen::VectorXd::Constant(1, -0.01);
  const auto opening_negative = mppi_core::ComputeContactPointMotions(
      state, opening_delta_q_tangent, context);
  ASSERT_EQ(opening_negative.size(), 1U);
  EXPECT_LT(opening_negative[0].delta_position_sensor_m.z(), 0.0);

  context.normal_axis_sign = -1.0;
  const auto closing_flipped =
      mppi_core::ComputeContactPointMotions(state, delta_q_tangent, context);
  ASSERT_EQ(closing_flipped.size(), 1U);
  EXPECT_LT(closing_flipped[0].delta_position_sensor_m.z(), 0.0);
}

TEST(GraspContactKinematicsTest, ZeroDeltaQProducesZeroMotion) {
  const auto sensor_model = MakeSingleRevoluteZSensorModel();
  pinocchio::Data data(sensor_model.model);

  mppi_core::TactileState tactile;
  tactile.valid = true;
  tactile.contact_presence = mppi_core::ContactPresence::kStableContact;
  mppi_core::TactileContactPoint point;
  point.active = true;
  point.position_sensor_m = Eigen::Vector3d{1.0, 0.0, 0.0};
  tactile.contact_points.push_back(point);

  const auto state = MakeContactKinematicsState(sensor_model.model, tactile);
  const Eigen::VectorXd delta_q = Eigen::VectorXd::Zero(1);
  mppi_core::PinocchioContactKinematicsContext context;
  context.model = &sensor_model.model;
  context.data = &data;
  context.sensor_frame_id = sensor_model.sensor_frame_id;

  const auto motions =
      mppi_core::ComputeContactPointMotions(state, delta_q, context);

  ASSERT_EQ(motions.size(), 1U);
  EXPECT_NEAR(motions[0].delta_position_sensor_m.norm(), 0.0, kTolerance);
}

TEST(GraspContactKinematicsTest, InvalidDimensionsReturnEmpty) {
  const auto sensor_model = MakeSingleRevoluteZSensorModel();
  pinocchio::Data data(sensor_model.model);

  mppi_core::TactileState tactile;
  tactile.valid = true;
  tactile.contact_presence = mppi_core::ContactPresence::kStableContact;
  mppi_core::TactileContactPoint point;
  point.active = true;
  point.position_sensor_m = Eigen::Vector3d{1.0, 0.0, 0.0};
  tactile.contact_points.push_back(point);

  const auto state = mppi_core::MakeGraspState(
      Eigen::VectorXd::Zero(2), Eigen::VectorXd::Zero(2),
      Eigen::VectorXd::Zero(2), tactile);
  const Eigen::VectorXd delta_q = Eigen::VectorXd::Zero(1);
  mppi_core::PinocchioContactKinematicsContext context;
  context.model = &sensor_model.model;
  context.data = &data;
  context.sensor_frame_id = sensor_model.sensor_frame_id;

  EXPECT_TRUE(
      mppi_core::ComputeContactPointMotions(state, delta_q, context).empty());

  const auto valid_state =
      MakeContactKinematicsState(sensor_model.model, tactile);
  const Eigen::VectorXd wrong_delta_q = Eigen::VectorXd::Zero(2);
  EXPECT_TRUE(
      mppi_core::ComputeContactPointMotions(valid_state, wrong_delta_q, context)
          .empty());
}

TEST(ContactForceProjectionTest, EmptyOrInvalidInputsReturnInvalidResult) {
  const auto sensor_model = MakeSinglePrismaticZSensorModel();
  pinocchio::Data data(sensor_model.model);

  mppi_core::TactileState tactile;
  tactile.valid = true;
  tactile.contact_presence = mppi_core::ContactPresence::kStableContact;
  const auto state = MakeContactKinematicsState(sensor_model.model, tactile);

  mppi_core::PinocchioContactKinematicsContext context;
  context.model = &sensor_model.model;
  context.data = &data;
  context.sensor_frame_id = sensor_model.sensor_frame_id;

  const Eigen::VectorXd tau_residual = Eigen::VectorXd::Constant(1, 1.0);
  const auto no_contacts = mppi_core::ProjectContactForcesFromTorqueResidual(
      state, tau_residual, context);
  EXPECT_FALSE(no_contacts.valid);
  EXPECT_TRUE(no_contacts.contact_forces.empty());

  const Eigen::VectorXd wrong_tau = Eigen::VectorXd::Zero(2);
  const auto wrong_dimension =
      mppi_core::ProjectContactForcesFromTorqueResidual(state, wrong_tau,
                                                        context);
  EXPECT_FALSE(wrong_dimension.valid);
}

TEST(ContactForceProjectionTest, PrismaticNormalResidualProjectsToNormalForce) {
  const auto sensor_model = MakeSinglePrismaticZSensorModel();
  pinocchio::Data data(sensor_model.model);

  mppi_core::TactileState tactile;
  tactile.valid = true;
  tactile.contact_presence = mppi_core::ContactPresence::kStableContact;
  mppi_core::TactileContactPoint point;
  point.active = true;
  point.support_index = 4;
  point.position_sensor_m = Eigen::Vector3d::Zero();
  tactile.contact_points.push_back(point);

  const auto state = MakeContactKinematicsState(sensor_model.model, tactile);
  mppi_core::PinocchioContactKinematicsContext context;
  context.model = &sensor_model.model;
  context.data = &data;
  context.sensor_frame_id = sensor_model.sensor_frame_id;

  mppi_core::ContactForceProjectionConfig config;
  config.regularization = 1.0e-9;
  config.tactile_prior_weight = 0.0;

  const Eigen::VectorXd tau_residual = Eigen::VectorXd::Constant(1, 2.0);
  const auto projection = mppi_core::ProjectContactForcesFromTorqueResidual(
      state, tau_residual, context, config);

  ASSERT_TRUE(projection.valid);
  ASSERT_EQ(projection.contact_forces.size(), 1U);
  EXPECT_EQ(projection.contact_forces[0].support_index, 4U);
  EXPECT_NEAR(projection.contact_forces[0].normal_force_n, 2.0, 1.0e-6);
  EXPECT_NEAR(projection.total_normal_force_n, 2.0, 1.0e-6);
  EXPECT_NEAR(projection.residual_norm, 0.0, 1.0e-6);
}

TEST(ContactForceProjectionTest, TactilePriorRegularizesNormalForce) {
  const auto sensor_model = MakeSinglePrismaticZSensorModel();
  pinocchio::Data data(sensor_model.model);

  mppi_core::TactileState tactile;
  tactile.valid = true;
  tactile.contact_presence = mppi_core::ContactPresence::kStableContact;
  mppi_core::TactileContactPoint point;
  point.active = true;
  point.position_sensor_m = Eigen::Vector3d::Zero();
  point.has_normal_force = true;
  point.normal_force_n = 1.5;
  tactile.contact_points.push_back(point);

  const auto state = MakeContactKinematicsState(sensor_model.model, tactile);
  mppi_core::PinocchioContactKinematicsContext context;
  context.model = &sensor_model.model;
  context.data = &data;
  context.sensor_frame_id = sensor_model.sensor_frame_id;

  mppi_core::ContactForceProjectionConfig config;
  config.regularization = 0.0;
  config.tactile_prior_weight = 1000.0;

  const Eigen::VectorXd tau_residual = Eigen::VectorXd::Zero(1);
  const auto projection = mppi_core::ProjectContactForcesFromTorqueResidual(
      state, tau_residual, context, config);

  ASSERT_TRUE(projection.valid);
  ASSERT_EQ(projection.contact_forces.size(), 1U);
  EXPECT_NEAR(projection.contact_forces[0].normal_force_n,
              1000.0 * 1.5 / 1001.0, 1.0e-6);
}

TEST(ContactForceProjectionTest, NegativeNormalForceCanBeClamped) {
  const auto sensor_model = MakeSinglePrismaticZSensorModel();
  pinocchio::Data data(sensor_model.model);

  mppi_core::TactileState tactile;
  tactile.valid = true;
  tactile.contact_presence = mppi_core::ContactPresence::kStableContact;
  mppi_core::TactileContactPoint point;
  point.active = true;
  point.position_sensor_m = Eigen::Vector3d::Zero();
  tactile.contact_points.push_back(point);

  const auto state = MakeContactKinematicsState(sensor_model.model, tactile);
  mppi_core::PinocchioContactKinematicsContext context;
  context.model = &sensor_model.model;
  context.data = &data;
  context.sensor_frame_id = sensor_model.sensor_frame_id;

  mppi_core::ContactForceProjectionConfig config;
  config.regularization = 1.0e-9;
  config.tactile_prior_weight = 0.0;
  config.clamp_negative_normal_force = true;

  const Eigen::VectorXd tau_residual = Eigen::VectorXd::Constant(1, -1.0);
  const auto projection = mppi_core::ProjectContactForcesFromTorqueResidual(
      state, tau_residual, context, config);

  ASSERT_TRUE(projection.valid);
  ASSERT_EQ(projection.contact_forces.size(), 1U);
  EXPECT_NEAR(projection.contact_forces[0].normal_force_n, 0.0, kTolerance);
  EXPECT_NEAR(projection.total_normal_force_n, 0.0, kTolerance);
}

TEST(ContactForceProjectionTest,
     RevoluteTorqueResidualProjectsToTangentialForceAndTorsion) {
  const auto sensor_model = MakeSingleRevoluteZSensorModel();
  pinocchio::Data data(sensor_model.model);

  mppi_core::TactileState tactile;
  tactile.valid = true;
  tactile.contact_presence = mppi_core::ContactPresence::kStableContact;
  mppi_core::TactileContactPoint point;
  point.active = true;
  point.position_sensor_m = Eigen::Vector3d{1.0, 0.0, 0.0};
  tactile.contact_points.push_back(point);

  const auto state = MakeContactKinematicsState(sensor_model.model, tactile);
  mppi_core::PinocchioContactKinematicsContext context;
  context.model = &sensor_model.model;
  context.data = &data;
  context.sensor_frame_id = sensor_model.sensor_frame_id;

  mppi_core::ContactForceProjectionConfig config;
  config.regularization = 1.0e-9;
  config.tactile_prior_weight = 0.0;

  const Eigen::VectorXd tau_residual = Eigen::VectorXd::Constant(1, 2.0);
  const auto projection = mppi_core::ProjectContactForcesFromTorqueResidual(
      state, tau_residual, context, config);

  ASSERT_TRUE(projection.valid);
  ASSERT_EQ(projection.contact_forces.size(), 1U);
  EXPECT_NEAR(projection.contact_forces[0].tangential_force_n.x(), 0.0,
              kTolerance);
  EXPECT_NEAR(projection.contact_forces[0].tangential_force_n.y(), 2.0, 1.0e-6);
  EXPECT_NEAR(projection.net_tangential_force_n.y(), 2.0, 1.0e-6);
  EXPECT_NEAR(projection.net_torsional_moment_nm, 2.0, 1.0e-6);
  EXPECT_GT(projection.total_friction_violation, 0.0);
}

TEST(ContactForceRolloutTest, ProjectedForceUpdatesPredictedTactileState) {
  mppi_core::TactileState tactile;
  tactile.valid = true;
  tactile.contact_presence = mppi_core::ContactPresence::kStableContact;
  tactile.has_normal_force = true;
  tactile.normal_force_n = 1.0;
  tactile.has_shear = true;
  tactile.shear_displacement_m = Eigen::Vector2d::Zero();
  tactile.has_rotational_shear = true;
  tactile.rotational_shear_rad = 0.0;
  tactile.confidence = 1.0;
  tactile.contact_support_count = 2;
  tactile.support_count = 2;

  mppi_core::TactileContactPoint first;
  first.active = true;
  first.support_index = 0;
  mppi_core::TactileContactPoint second;
  second.active = true;
  second.support_index = 1;
  tactile.contact_points.push_back(first);
  tactile.contact_points.push_back(second);

  mppi_core::ContactForceProjectionResult projection;
  projection.valid = true;
  projection.total_normal_force_n = 4.0;
  projection.net_tangential_force_n = Eigen::Vector2d{2.0, 0.0};
  projection.net_torsional_moment_nm = 0.5;

  mppi_core::ContactPointForce first_force;
  first_force.support_index = 0;
  first_force.normal_force_n = 2.0;
  mppi_core::ContactPointForce second_force;
  second_force.support_index = 1;
  second_force.normal_force_n = 2.0;
  projection.contact_forces.push_back(first_force);
  projection.contact_forces.push_back(second_force);

  mppi_core::ContactForceRolloutConfig config;
  config.force_lowpass_alpha = 1.0;
  config.shear_force_gain_m_per_n_s = 0.1;
  config.rotational_shear_gain_rad_per_nm_s = 1.0;
  config.min_stable_support_count = 2;
  config.shear_ref_m = 1.0;
  config.rotational_shear_ref_rad = 1.0;

  mppi_core::StepTactileStateFromProjectedForce(projection, 0.1, config,
                                                &tactile);

  EXPECT_EQ(tactile.contact_presence,
            mppi_core::ContactPresence::kStableContact);
  EXPECT_NEAR(tactile.normal_force_n, 4.0, kTolerance);
  ASSERT_TRUE(tactile.has_shear);
  EXPECT_NEAR(tactile.shear_displacement_m.x(), 0.02, kTolerance);
  ASSERT_TRUE(tactile.has_rotational_shear);
  EXPECT_NEAR(tactile.rotational_shear_rad, 0.05, kTolerance);
  EXPECT_NEAR(tactile.incipient_slip_score, 0.07, kTolerance);
  EXPECT_EQ(tactile.contact_support_count, 2U);
  EXPECT_EQ(tactile.activeContactPointCount(), 2U);
  EXPECT_NEAR(tactile.contact_points[0].normal_force_n, 2.0, kTolerance);
}

TEST(ContactForceRolloutTest,
     ZeroProjectedNormalForceLosesContactAndDeactivatesPoints) {
  mppi_core::TactileState tactile;
  tactile.valid = true;
  tactile.contact_presence = mppi_core::ContactPresence::kStableContact;
  tactile.has_normal_force = true;
  tactile.normal_force_n = 1.0;
  tactile.confidence = 1.0;
  tactile.contact_support_count = 1;
  tactile.support_count = 1;

  mppi_core::TactileContactPoint point;
  point.active = true;
  point.support_index = 0;
  tactile.contact_points.push_back(point);

  mppi_core::ContactForceProjectionResult projection;
  projection.valid = true;
  projection.total_normal_force_n = 0.0;
  projection.total_friction_violation = 2.0;
  mppi_core::ContactPointForce force;
  force.support_index = 0;
  force.normal_force_n = 0.0;
  projection.contact_forces.push_back(force);

  mppi_core::ContactForceRolloutConfig config;
  config.force_lowpass_alpha = 1.0;
  config.negative_normal_confidence_decay = 0.5;
  config.friction_violation_confidence_decay = 0.1;

  mppi_core::StepTactileStateFromProjectedForce(projection, 0.1, config,
                                                &tactile);

  EXPECT_NEAR(tactile.normal_force_n, 0.0, kTolerance);
  EXPECT_LT(tactile.confidence, 1.0);
  EXPECT_EQ(tactile.contact_presence, mppi_core::ContactPresence::kNoContact);
  EXPECT_EQ(tactile.contact_support_count, 0U);
  EXPECT_EQ(tactile.activeContactPointCount(), 0U);
}

TEST(ContactForceCorrectionTest, UpdatesAndClampsNormalForceBias) {
  mppi_core::TactileState measured;
  measured.valid = true;
  measured.contact_presence = mppi_core::ContactPresence::kStableContact;

  mppi_core::ContactForceCorrectionState state;
  mppi_core::ContactForceCorrectionConfig config;
  config.bias_update_rate = 0.1;
  config.max_abs_bias_n = 0.5;

  mppi_core::UpdateContactForceCorrection(1.0, 2.0, measured, config, &state);
  EXPECT_NEAR(state.normal_force_bias_n, 0.1, kTolerance);

  mppi_core::UpdateContactForceCorrection(0.0, 10.0, measured, config, &state);
  EXPECT_NEAR(state.normal_force_bias_n, 0.5, kTolerance);
  EXPECT_NEAR(mppi_core::ApplyContactForceCorrection(1.0, state), 1.5,
              kTolerance);
}

TEST(ContactForceCorrectionTest, SkipsUpdateWithoutValidMeasuredContact) {
  mppi_core::TactileState measured;
  measured.valid = true;
  measured.contact_presence = mppi_core::ContactPresence::kNoContact;

  mppi_core::ContactForceCorrectionState state;
  state.normal_force_bias_n = 0.25;
  mppi_core::ContactForceCorrectionConfig config;
  config.update_only_in_contact = true;

  mppi_core::UpdateContactForceCorrection(1.0, 3.0, measured, config, &state);
  EXPECT_NEAR(state.normal_force_bias_n, 0.25, kTolerance);

  config.enabled = false;
  measured.contact_presence = mppi_core::ContactPresence::kStableContact;
  mppi_core::UpdateContactForceCorrection(1.0, 3.0, measured, config, &state);
  EXPECT_NEAR(state.normal_force_bias_n, 0.25, kTolerance);
}

TEST(NariTouchStateTest, DefaultConstructorInitializesUnitPositions) {
  const mppi_core::NariTouchState tactile;
  const auto positions = mppi_core::NariTouchUnitPositionsM();

  ASSERT_EQ(tactile.units.size(), positions.size());
  for (std::size_t i = 0; i < tactile.units.size(); ++i) {
    EXPECT_NEAR(tactile.units[i].position_m.x(), positions[i].x(), kTolerance);
    EXPECT_NEAR(tactile.units[i].position_m.y(), positions[i].y(), kTolerance);
  }
}

TEST(NariTouchStateTest, ContactPresenceUsesContactStateAndUnits) {
  mppi_core::NariTouchState tactile;
  EXPECT_FALSE(tactile.hasContact());

  tactile.contact_state = mppi_core::NariTouchContactState::kFewContacts;
  EXPECT_TRUE(tactile.hasContact());

  tactile.contact_state = mppi_core::NariTouchContactState::kEnoughContacts;
  EXPECT_TRUE(tactile.hasContact());

  tactile.contact_state = mppi_core::NariTouchContactState::kNoContact;
  tactile.units[2].contact = true;
  tactile.units[4].contact = true;
  EXPECT_TRUE(tactile.hasContact());
  EXPECT_EQ(tactile.contactUnitCount(), 2U);
  EXPECT_EQ(tactile.contactNodeCount(), 2U);
}

TEST(NariTouchStateTest, TotalForceSumsPositiveFiniteUnitForces) {
  mppi_core::NariTouchState tactile;
  tactile.force_z = 10.0;
  tactile.units[0].normal_force = 0.25;
  tactile.units[1].normal_force = -1.0;
  tactile.units[2].normal_force = std::numeric_limits<double>::infinity();
  tactile.units[3].normal_force = 0.75;

  EXPECT_NEAR(mppi_core::ComputeNariTouchTotalNormalForceN(tactile), 1.0,
              kTolerance);
}

TEST(NariTouchStateTest, TotalForceFallsBackToAggregateForce) {
  mppi_core::NariTouchState tactile;
  tactile.force_z = 1.2;
  EXPECT_NEAR(mppi_core::ComputeNariTouchTotalNormalForceN(tactile), 1.2,
              kTolerance);

  tactile.force_z = -1.0;
  EXPECT_NEAR(mppi_core::ComputeNariTouchTotalNormalForceN(tactile), 0.0,
              kTolerance);

  tactile.force_z = std::numeric_limits<double>::quiet_NaN();
  EXPECT_NEAR(mppi_core::ComputeNariTouchTotalNormalForceN(tactile), 0.0,
              kTolerance);
}

TEST(NariTouchStateTest, ContactCentroidIsForceWeightedAndUsesFiniteCop) {
  mppi_core::NariTouchState tactile;
  tactile.units[0].contact = true;
  tactile.units[0].position_m = Eigen::Vector2d{0.0, 0.0};
  tactile.units[0].cop = Eigen::Vector2d{0.001, 0.0};
  tactile.units[0].normal_force = 1.0;

  tactile.units[1].contact = true;
  tactile.units[1].position_m = Eigen::Vector2d{0.004, 0.0};
  tactile.units[1].cop =
      Eigen::Vector2d{std::numeric_limits<double>::quiet_NaN(), 0.002};
  tactile.units[1].normal_force = 3.0;

  Eigen::Vector2d centroid = Eigen::Vector2d::Zero();
  ASSERT_TRUE(mppi_core::ComputeNariTouchContactCentroidM(tactile, &centroid));
  EXPECT_NEAR(centroid.x(), 0.00325, kTolerance);
  EXPECT_NEAR(centroid.y(), 0.0, kTolerance);
}

TEST(NariTouchFeatureTest, SlipRiskIncludesDerivedVelocityWhenWeighted) {
  mppi_core::NariTouchState tactile;
  tactile.slip_state = Eigen::Vector3d{0.03, 0.04, 0.0};
  tactile.slip_velocity_state = Eigen::Vector3d{0.0, 3.0, 4.0};

  EXPECT_NEAR(mppi_core::ComputeNariTouchSlipMagnitude(tactile), 0.05,
              kTolerance);
  EXPECT_NEAR(mppi_core::ComputeNariTouchSlipVelocityMagnitude(tactile), 5.0,
              kTolerance);
  EXPECT_NEAR(mppi_core::ComputeNariTouchSlipRisk(tactile, 0.2), 1.05,
              kTolerance);

  // Negative weights are treated as disabled rather than inverting slip risk.
  EXPECT_NEAR(mppi_core::ComputeNariTouchSlipRisk(tactile, -1.0), 0.05,
              kTolerance);
}

TEST(NariTouchFeatureTest, NonFiniteSlipVectorsProduceZeroScores) {
  mppi_core::NariTouchState tactile;
  tactile.slip_state =
      Eigen::Vector3d{0.01, std::numeric_limits<double>::quiet_NaN(), 0.0};
  tactile.slip_velocity_state =
      Eigen::Vector3d{0.0, 1.0, std::numeric_limits<double>::infinity()};

  EXPECT_NEAR(mppi_core::ComputeNariTouchSlipMagnitude(tactile), 0.0,
              kTolerance);
  EXPECT_NEAR(mppi_core::ComputeNariTouchSlipVelocityMagnitude(tactile), 0.0,
              kTolerance);
  EXPECT_NEAR(mppi_core::ComputeNariTouchSlipRisk(tactile, 1.0), 0.0,
              kTolerance);
}

TEST(NariTouchAdapterTest, ContactStatesMapToGenericContactPresence) {
  EXPECT_EQ(mppi_core::ToContactPresence(
                mppi_core::NariTouchContactState::kNoContact),
            mppi_core::ContactPresence::kNoContact);
  EXPECT_EQ(mppi_core::ToContactPresence(
                mppi_core::NariTouchContactState::kFewContacts),
            mppi_core::ContactPresence::kLightContact);
  EXPECT_EQ(mppi_core::ToContactPresence(
                mppi_core::NariTouchContactState::kEnoughContacts),
            mppi_core::ContactPresence::kStableContact);
}

TEST(NariTouchAdapterTest, ConvertsForcesCentroidShearAndSupportCounts) {
  mppi_core::NariTouchState nari;
  nari.contact_state = mppi_core::NariTouchContactState::kEnoughContacts;
  nari.slip_state = Eigen::Vector3d{0.01, -0.02, 0.3};
  nari.slip_velocity_state = Eigen::Vector3d{0.4, -0.5, 0.6};
  nari.centroid_velocity_mps = Eigen::Vector2d{0.7, -0.8};
  nari.units[0].contact = true;
  nari.units[0].position_m = Eigen::Vector2d{0.0, 0.0};
  nari.units[0].normal_force = 0.25;
  nari.units[1].contact = true;
  nari.units[1].position_m = Eigen::Vector2d{0.002, 0.0};
  nari.units[1].normal_force = 0.75;

  const auto tactile = ToTestTactileState(nari, 0.5);

  EXPECT_TRUE(tactile.valid);
  EXPECT_EQ(tactile.frame_name, "test_tactile");
  EXPECT_NEAR(tactile.stamp_sec, 1.0, kTolerance);
  EXPECT_EQ(tactile.contact_presence,
            mppi_core::ContactPresence::kStableContact);
  EXPECT_TRUE(tactile.has_normal_force);
  EXPECT_NEAR(tactile.normal_force_n, 1.0, kTolerance);
  ASSERT_TRUE(tactile.has_centroid);
  EXPECT_NEAR(tactile.centroid_m.x(), 0.0015, kTolerance);
  EXPECT_NEAR(tactile.centroid_velocity_mps.x(), 0.7, kTolerance);
  EXPECT_TRUE(tactile.has_shear);
  EXPECT_NEAR(tactile.shear_displacement_m.x(), 0.01, kTolerance);
  EXPECT_NEAR(tactile.shear_displacement_m.y(), -0.02, kTolerance);
  EXPECT_TRUE(tactile.has_rotational_shear);
  EXPECT_NEAR(tactile.rotational_shear_rad, 0.3, kTolerance);
  EXPECT_TRUE(tactile.has_shear_velocity);
  EXPECT_NEAR(tactile.shear_velocity_mps.x(), 0.4, kTolerance);
  EXPECT_NEAR(tactile.shear_velocity_mps.y(), -0.5, kTolerance);
  EXPECT_TRUE(tactile.has_rotational_shear_velocity);
  EXPECT_NEAR(tactile.rotational_shear_velocity_radps, 0.6, kTolerance);
  EXPECT_EQ(tactile.contact_support_count, 2U);
  EXPECT_EQ(tactile.support_count, mppi_core::kNariTouchUnitCount);
  EXPECT_NEAR(tactile.contact_area_proxy, 0.25, kTolerance);
  EXPECT_EQ(tactile.contact_points.size(), 2U);
  EXPECT_EQ(tactile.activeContactPointCount(), 2U);
}

TEST(NariTouchAdapterTest, ActiveUnitsProduceSparseContactPoints) {
  mppi_core::NariTouchState nari;
  nari.units[0].contact = true;
  nari.units[0].position_m = Eigen::Vector2d{0.001, 0.002};
  nari.units[0].cop = Eigen::Vector2d{0.0005, -0.00025};
  nari.units[0].normal_force = 0.4;

  nari.units[1].contact = false;
  nari.units[1].normal_force = 9.0;

  nari.units[3].contact = true;
  nari.units[3].position_m = Eigen::Vector2d{0.003, -0.004};
  nari.units[3].cop =
      Eigen::Vector2d{std::numeric_limits<double>::quiet_NaN(), 0.001};
  nari.units[3].normal_force = -0.1;

  nari.units[7].contact = true;
  nari.units[7].position_m = Eigen::Vector2d{-0.002, 0.005};
  nari.units[7].cop = Eigen::Vector2d{0.0, 0.001};
  nari.units[7].normal_force = std::numeric_limits<double>::infinity();

  const auto tactile = ToTestTactileState(nari);

  ASSERT_EQ(tactile.contact_points.size(), 3U);
  EXPECT_EQ(tactile.activeContactPointCount(), 3U);
  EXPECT_EQ(tactile.contact_support_count, tactile.activeContactPointCount());
  EXPECT_EQ(tactile.support_count, mppi_core::kNariTouchUnitCount);
  EXPECT_NEAR(tactile.contact_area_proxy, 3.0 / 8.0, kTolerance);

  const auto& first = tactile.contact_points[0];
  EXPECT_TRUE(first.active);
  EXPECT_EQ(first.support_index, 0U);
  EXPECT_NEAR(first.position_sensor_m.x(), 0.0015, kTolerance);
  EXPECT_NEAR(first.position_sensor_m.y(), 0.00175, kTolerance);
  EXPECT_NEAR(first.position_sensor_m.z(), 0.0, kTolerance);
  EXPECT_TRUE(first.has_normal_force);
  EXPECT_NEAR(first.normal_force_n, 0.4, kTolerance);
  EXPECT_NEAR(first.confidence, 1.0, kTolerance);

  const auto& second = tactile.contact_points[1];
  EXPECT_EQ(second.support_index, 3U);
  EXPECT_NEAR(second.position_sensor_m.x(), 0.003, kTolerance);
  EXPECT_NEAR(second.position_sensor_m.y(), -0.004, kTolerance);
  EXPECT_NEAR(second.position_sensor_m.z(), 0.0, kTolerance);
  EXPECT_FALSE(second.has_normal_force);
  EXPECT_NEAR(second.normal_force_n, 0.0, kTolerance);

  const auto& third = tactile.contact_points[2];
  EXPECT_EQ(third.support_index, 7U);
  EXPECT_NEAR(third.position_sensor_m.x(), -0.002, kTolerance);
  EXPECT_NEAR(third.position_sensor_m.y(), 0.006, kTolerance);
  EXPECT_NEAR(third.position_sensor_m.z(), 0.0, kTolerance);
  EXPECT_FALSE(third.has_normal_force);
}

TEST(NariTouchAdapterTest, ShearAndVelocityValidityAreIndependent) {
  mppi_core::NariTouchState nari;
  nari.slip_state =
      Eigen::Vector3d{0.01, -0.02, std::numeric_limits<double>::quiet_NaN()};
  nari.slip_velocity_state =
      Eigen::Vector3d{0.3, -0.4, std::numeric_limits<double>::infinity()};

  const auto tactile = ToTestTactileState(nari, 0.5);

  EXPECT_TRUE(tactile.has_shear);
  EXPECT_NEAR(tactile.shear_displacement_m.x(), 0.01, kTolerance);
  EXPECT_NEAR(tactile.shear_displacement_m.y(), -0.02, kTolerance);
  EXPECT_FALSE(tactile.has_rotational_shear);
  EXPECT_TRUE(tactile.has_shear_velocity);
  EXPECT_NEAR(tactile.shear_velocity_mps.x(), 0.3, kTolerance);
  EXPECT_NEAR(tactile.shear_velocity_mps.y(), -0.4, kTolerance);
  EXPECT_FALSE(tactile.has_rotational_shear_velocity);
  EXPECT_NEAR(tactile.slip_score, std::sqrt(0.01 * 0.01 + 0.02 * 0.02),
              kTolerance);
  EXPECT_NEAR(tactile.slip_velocity_score, std::sqrt(0.3 * 0.3 + 0.4 * 0.4),
              kTolerance);
}

TEST(NariTouchAdapterTest, ConfidenceAndEdgeRiskReflectContactQuality) {
  mppi_core::NariTouchState no_contact;
  const auto no_contact_tactile = ToTestTactileState(no_contact);

  mppi_core::NariTouchState stable;
  stable.contact_state = mppi_core::NariTouchContactState::kEnoughContacts;
  stable.units[3].contact = true;
  stable.units[3].position_m = Eigen::Vector2d{0.0, 0.0};
  stable.units[3].normal_force = 1.0;
  const auto stable_tactile = ToTestTactileState(stable);

  EXPECT_GT(stable_tactile.confidence, no_contact_tactile.confidence);

  mppi_core::NariTouchAdapterConfig config;
  mppi_core::TactileState center;
  center.has_centroid = true;
  center.centroid_m = Eigen::Vector2d::Zero();
  EXPECT_NEAR(mppi_core::ComputeNariTouchEdgeRisk(center, config), 0.0,
              kTolerance);

  center.centroid_m = Eigen::Vector2d{config.reduced_half_width_x_m, 0.0};
  EXPECT_NEAR(mppi_core::ComputeNariTouchEdgeRisk(center, config), 1.0,
              kTolerance);
}

TEST(GraspRolloutTest, TangentialMotionUpdatesCentroidAndShear) {
  mppi_core::TactileState tactile;
  tactile.valid = true;
  tactile.contact_presence = mppi_core::ContactPresence::kStableContact;
  tactile.normal_force_n = 1.0;
  tactile.has_centroid = true;
  tactile.centroid_m = Eigen::Vector2d::Zero();
  tactile.has_shear = true;
  tactile.shear_displacement_m = Eigen::Vector2d::Zero();
  tactile.has_rotational_shear = true;
  tactile.rotational_shear_rad = 0.0;
  tactile.contact_support_count = 4;
  tactile.support_count = 8;
  tactile.confidence = 1.0;

  mppi_core::ContactPointMotion first;
  first.position_sensor_m = Eigen::Vector3d{-0.001, 0.0, 0.0};
  first.delta_position_sensor_m = Eigen::Vector3d{0.001, -0.0005, 0.0};
  mppi_core::ContactPointMotion second;
  second.position_sensor_m = Eigen::Vector3d{0.001, 0.0, 0.0};
  second.delta_position_sensor_m = first.delta_position_sensor_m;

  mppi_core::GraspRolloutConfig config;
  config.tangential_confidence_loss_per_m = 0.0;
  config.edge_confidence_loss_gain = 0.0;

  const auto next = mppi_core::StepTactileContactPatch(
      tactile, std::vector<mppi_core::ContactPointMotion>{first, second}, 0.01,
      config);

  ASSERT_TRUE(next.has_centroid);
  EXPECT_NEAR(next.centroid_m.x(), 0.001, kTolerance);
  EXPECT_NEAR(next.centroid_m.y(), -0.0005, kTolerance);
  ASSERT_TRUE(next.has_shear);
  EXPECT_NEAR(next.shear_displacement_m.x(), 0.001, kTolerance);
  EXPECT_NEAR(next.shear_displacement_m.y(), -0.0005, kTolerance);
  ASSERT_TRUE(next.has_rotational_shear);
  EXPECT_NEAR(next.rotational_shear_rad, 0.0, kTolerance);
  EXPECT_NEAR(next.normal_force_n, 1.0, kTolerance);
  EXPECT_EQ(next.contact_presence, mppi_core::ContactPresence::kStableContact);
}

TEST(GraspRolloutTest, TangentialMotionAlongShearIncreasesSlipScore) {
  mppi_core::TactileState tactile;
  tactile.valid = true;
  tactile.contact_presence = mppi_core::ContactPresence::kStableContact;
  tactile.normal_force_n = 1.0;
  tactile.has_centroid = true;
  tactile.centroid_m = Eigen::Vector2d::Zero();
  tactile.has_shear = true;
  tactile.shear_displacement_m = Eigen::Vector2d{1.0e-3, 0.0};
  tactile.contact_support_count = 2;
  tactile.support_count = 2;
  tactile.confidence = 1.0;

  mppi_core::ContactPointMotion motion;
  motion.position_sensor_m = Eigen::Vector3d::Zero();
  motion.delta_position_sensor_m = Eigen::Vector3d{1.0e-3, 0.0, 0.0};

  mppi_core::GraspRolloutConfig config;
  config.min_stable_support_count = 2;
  config.tangential_confidence_loss_per_m = 0.0;
  config.edge_confidence_loss_gain = 0.0;
  config.shear_ref_m = 1.0e-3;

  const auto next = mppi_core::StepTactileContactPatch(
      tactile, std::vector<mppi_core::ContactPointMotion>{motion}, 0.01,
      config);

  EXPECT_GT(next.shear_displacement_m.norm(),
            tactile.shear_displacement_m.norm());
  EXPECT_GT(next.incipient_slip_score, 1.0);
}

TEST(GraspRolloutTest, TangentialMotionOppositeShearReducesSlipScore) {
  mppi_core::TactileState tactile;
  tactile.valid = true;
  tactile.contact_presence = mppi_core::ContactPresence::kStableContact;
  tactile.normal_force_n = 1.0;
  tactile.has_centroid = true;
  tactile.centroid_m = Eigen::Vector2d::Zero();
  tactile.has_shear = true;
  tactile.shear_displacement_m = Eigen::Vector2d{2.0e-3, 0.0};
  tactile.contact_support_count = 2;
  tactile.support_count = 2;
  tactile.confidence = 1.0;

  mppi_core::ContactPointMotion motion;
  motion.position_sensor_m = Eigen::Vector3d::Zero();
  motion.delta_position_sensor_m = Eigen::Vector3d{-1.0e-3, 0.0, 0.0};

  mppi_core::GraspRolloutConfig config;
  config.min_stable_support_count = 2;
  config.tangential_confidence_loss_per_m = 0.0;
  config.edge_confidence_loss_gain = 0.0;
  config.shear_ref_m = 1.0e-3;

  const auto next = mppi_core::StepTactileContactPatch(
      tactile, std::vector<mppi_core::ContactPointMotion>{motion}, 0.01,
      config);

  EXPECT_LT(next.shear_displacement_m.norm(),
            tactile.shear_displacement_m.norm());
  EXPECT_LT(next.incipient_slip_score, 2.0);
}

TEST(GraspRolloutTest, RelativeTangentialMotionUpdatesRotationalShear) {
  mppi_core::TactileState tactile;
  tactile.valid = true;
  tactile.contact_presence = mppi_core::ContactPresence::kStableContact;
  tactile.normal_force_n = 1.0;
  tactile.has_centroid = true;
  tactile.centroid_m = Eigen::Vector2d::Zero();
  tactile.has_rotational_shear = true;
  tactile.rotational_shear_rad = 0.0;
  tactile.contact_support_count = 4;
  tactile.support_count = 8;
  tactile.confidence = 1.0;

  mppi_core::ContactPointMotion left;
  left.position_sensor_m = Eigen::Vector3d{-0.001, 0.0, 0.0};
  left.delta_position_sensor_m = Eigen::Vector3d{0.0, -0.0001, 0.0};
  mppi_core::ContactPointMotion right;
  right.position_sensor_m = Eigen::Vector3d{0.001, 0.0, 0.0};
  right.delta_position_sensor_m = Eigen::Vector3d{0.0, 0.0001, 0.0};

  mppi_core::GraspRolloutConfig config;
  config.tangential_confidence_loss_per_m = 0.0;
  config.edge_confidence_loss_gain = 0.0;

  const auto next = mppi_core::StepTactileContactPatch(
      tactile, std::vector<mppi_core::ContactPointMotion>{left, right}, 0.01,
      config);

  ASSERT_TRUE(next.has_centroid);
  EXPECT_NEAR(next.centroid_m.norm(), 0.0, kTolerance);
  ASSERT_TRUE(next.has_rotational_shear);
  EXPECT_NEAR(next.rotational_shear_rad, 0.1, kTolerance);
}

TEST(GraspRolloutTest,
     SeparatingMotionReducesNormalForceConfidenceAndContactQuality) {
  mppi_core::TactileState tactile;
  tactile.valid = true;
  tactile.contact_presence = mppi_core::ContactPresence::kStableContact;
  tactile.normal_force_n = 1.0;
  tactile.has_centroid = true;
  tactile.centroid_m = Eigen::Vector2d::Zero();
  tactile.contact_support_count = 4;
  tactile.support_count = 8;
  tactile.confidence = 1.0;

  mppi_core::ContactPointMotion motion;
  motion.position_sensor_m = Eigen::Vector3d::Zero();
  motion.delta_position_sensor_m = Eigen::Vector3d{0.0, 0.0, -0.002};

  mppi_core::GraspRolloutConfig config;
  config.normal_force_gain_n_per_m = 100.0;
  config.separating_confidence_loss_per_m = 200.0;

  const auto next = mppi_core::StepTactileContactPatch(
      tactile, std::vector<mppi_core::ContactPointMotion>{motion}, 0.01,
      config);

  EXPECT_NEAR(next.normal_force_n, 0.8, kTolerance);
  EXPECT_NEAR(next.confidence, 0.6, kTolerance);
  EXPECT_EQ(next.contact_presence, mppi_core::ContactPresence::kStableContact);
}

TEST(GraspRolloutTest, ClosingMotionIncreasesNormalForceAndConfidence) {
  mppi_core::TactileState tactile;
  tactile.valid = true;
  tactile.contact_presence = mppi_core::ContactPresence::kLightContact;
  tactile.normal_force_n = 0.5;
  tactile.has_centroid = true;
  tactile.centroid_m = Eigen::Vector2d::Zero();
  tactile.contact_support_count = 2;
  tactile.support_count = 2;
  tactile.confidence = 0.5;

  mppi_core::ContactPointMotion motion;
  motion.position_sensor_m = Eigen::Vector3d::Zero();
  motion.delta_position_sensor_m = Eigen::Vector3d{0.0, 0.0, 0.002};

  mppi_core::GraspRolloutConfig config;
  config.min_stable_support_count = 2;
  config.normal_force_gain_n_per_m = 100.0;
  config.closing_confidence_gain_per_m = 20.0;

  const auto next = mppi_core::StepTactileContactPatch(
      tactile, std::vector<mppi_core::ContactPointMotion>{motion}, 0.01,
      config);

  EXPECT_NEAR(next.normal_force_n, 0.7, kTolerance);
  EXPECT_NEAR(next.confidence, 0.54, kTolerance);
  EXPECT_EQ(next.contact_presence, mppi_core::ContactPresence::kStableContact);
}

TEST(GraspRolloutTest,
     EdgeMigrationReducesConfidenceAndKeepsEdgeRiskDiagnostic) {
  mppi_core::TactileState tactile;
  tactile.valid = true;
  tactile.contact_presence = mppi_core::ContactPresence::kStableContact;
  tactile.normal_force_n = 1.0;
  tactile.has_centroid = true;
  tactile.centroid_m = Eigen::Vector2d{0.0029, 0.0};
  tactile.contact_support_count = 4;
  tactile.support_count = 8;
  tactile.confidence = 1.0;

  mppi_core::ContactPointMotion motion;
  motion.position_sensor_m = Eigen::Vector3d{0.0029, 0.0, 0.0};
  motion.delta_position_sensor_m = Eigen::Vector3d{0.001, 0.0, 0.0};

  mppi_core::GraspRolloutConfig config;
  config.tangential_confidence_loss_per_m = 0.0;
  config.edge_confidence_loss_gain = 0.5;

  const auto next = mppi_core::StepTactileContactPatch(
      tactile, std::vector<mppi_core::ContactPointMotion>{motion}, 0.01,
      config);

  EXPECT_GT(next.edge_risk, 1.0);
  EXPECT_LT(next.confidence, tactile.confidence);
}

TEST(GraspRolloutTest, CenteringMotionReducesEdgeRiskAndMaintainsConfidence) {
  mppi_core::TactileState tactile;
  tactile.valid = true;
  tactile.contact_presence = mppi_core::ContactPresence::kStableContact;
  tactile.normal_force_n = 1.0;
  tactile.has_centroid = true;
  tactile.centroid_m = Eigen::Vector2d{0.0027, 0.0};
  tactile.contact_support_count = 4;
  tactile.support_count = 8;
  tactile.confidence = 0.8;

  mppi_core::ContactPointMotion motion;
  motion.position_sensor_m = Eigen::Vector3d{0.0027, 0.0, 0.0};
  motion.delta_position_sensor_m = Eigen::Vector3d{-0.001, 0.0, 0.0};

  mppi_core::GraspRolloutConfig config;
  config.tangential_confidence_loss_per_m = 0.0;
  config.edge_confidence_loss_gain = 0.5;

  const double initial_edge_risk =
      mppi_core::ComputeTactilePatchEdgeRisk(tactile.centroid_m, config);
  const auto next = mppi_core::StepTactileContactPatch(
      tactile, std::vector<mppi_core::ContactPointMotion>{motion}, 0.01,
      config);

  EXPECT_LT(next.edge_risk, initial_edge_risk);
  EXPECT_NEAR(next.confidence, tactile.confidence, kTolerance);
}

TEST(GraspRolloutTest, LargeSeparatingMotionCanLoseContactAndDeactivatePoints) {
  mppi_core::TactileState tactile;
  tactile.valid = true;
  tactile.contact_presence = mppi_core::ContactPresence::kStableContact;
  tactile.normal_force_n = 0.5;
  tactile.has_centroid = true;
  tactile.centroid_m = Eigen::Vector2d::Zero();
  tactile.contact_support_count = 1;
  tactile.support_count = 1;
  tactile.confidence = 1.0;
  mppi_core::TactileContactPoint point;
  point.active = true;
  point.position_sensor_m = Eigen::Vector3d::Zero();
  tactile.contact_points.push_back(point);

  mppi_core::ContactPointMotion motion;
  motion.position_sensor_m = Eigen::Vector3d::Zero();
  motion.delta_position_sensor_m = Eigen::Vector3d{0.0, 0.0, -0.01};

  mppi_core::GraspRolloutConfig config;
  config.min_stable_support_count = 1;
  config.normal_force_gain_n_per_m = 100.0;
  config.separating_confidence_loss_per_m = 200.0;

  const auto next = mppi_core::StepTactileContactPatch(
      tactile, std::vector<mppi_core::ContactPointMotion>{motion}, 0.01,
      config);

  EXPECT_NEAR(next.normal_force_n, 0.0, kTolerance);
  EXPECT_NEAR(next.confidence, 0.0, kTolerance);
  EXPECT_EQ(next.contact_presence, mppi_core::ContactPresence::kNoContact);
  EXPECT_EQ(next.contact_support_count, 0U);
  EXPECT_EQ(next.activeContactPointCount(), 0U);
}

TEST(GraspRolloutTest, InvalidInputsDoNotProduceNonFiniteState) {
  mppi_core::TactileState tactile;
  tactile.valid = true;
  tactile.contact_presence = mppi_core::ContactPresence::kStableContact;
  tactile.normal_force_n = 1.0;
  tactile.has_centroid = true;
  tactile.centroid_m = Eigen::Vector2d::Zero();
  tactile.has_shear = true;
  tactile.shear_displacement_m = Eigen::Vector2d::Zero();
  tactile.contact_support_count = 4;
  tactile.support_count = 8;
  tactile.confidence = 0.7;

  mppi_core::ContactPointMotion invalid;
  invalid.position_sensor_m =
      Eigen::Vector3d{std::numeric_limits<double>::quiet_NaN(), 0.0, 0.0};
  invalid.delta_position_sensor_m = Eigen::Vector3d{1.0, 0.0, 0.0};

  const auto next = mppi_core::StepTactileContactPatch(
      tactile, std::vector<mppi_core::ContactPointMotion>{invalid}, 0.01);

  EXPECT_TRUE(next.centroid_m.allFinite());
  EXPECT_TRUE(next.shear_displacement_m.allFinite());
  EXPECT_TRUE(std::isfinite(next.normal_force_n));
  EXPECT_TRUE(std::isfinite(next.confidence));

  mppi_core::ContactPointMotion valid;
  valid.position_sensor_m = Eigen::Vector3d::Zero();
  valid.delta_position_sensor_m = Eigen::Vector3d{0.001, 0.0, 0.0};
  const auto zero_dt = mppi_core::StepTactileContactPatch(
      tactile, std::vector<mppi_core::ContactPointMotion>{valid}, 0.0);
  EXPECT_NEAR(zero_dt.centroid_m.x(), 0.0, kTolerance);
  EXPECT_NEAR(zero_dt.shear_displacement_m.x(), 0.0, kTolerance);
}

TEST(GraspRolloutTest, RefreshUsesNormalizedPredictedSlipScore) {
  mppi_core::TactileState tactile;
  tactile.valid = true;
  tactile.contact_presence = mppi_core::ContactPresence::kStableContact;
  tactile.has_normal_force = true;
  tactile.normal_force_n = 1.0;
  tactile.has_shear = true;
  tactile.shear_displacement_m = Eigen::Vector2d{1.0e-3, 0.0};
  tactile.has_rotational_shear = true;
  tactile.rotational_shear_rad = 2.0e-2;
  tactile.contact_support_count = 4;
  tactile.support_count = 8;
  tactile.contact_area_proxy = 0.5;
  tactile.confidence = 1.0;

  mppi_core::GraspRolloutConfig config;
  config.shear_ref_m = 1.0e-3;
  config.rotational_shear_ref_rad = 2.0e-2;

  mppi_core::RefreshPredictedTactileFields(&tactile, config);

  EXPECT_EQ(tactile.contact_presence,
            mppi_core::ContactPresence::kStableContact);
  EXPECT_NEAR(tactile.slip_score, 2.0, kTolerance);
  EXPECT_NEAR(tactile.incipient_slip_score, 2.0, kTolerance);
}

TEST(GraspRolloutTest, RefreshMarksContactPointsInactiveOnContactLoss) {
  mppi_core::TactileState tactile;
  tactile.valid = true;
  tactile.contact_presence = mppi_core::ContactPresence::kStableContact;
  tactile.has_normal_force = true;
  tactile.normal_force_n = 0.0;
  tactile.contact_support_count = 2;
  tactile.support_count = 8;
  tactile.contact_area_proxy = 0.25;
  tactile.confidence = 1.0;

  mppi_core::TactileContactPoint first;
  first.active = true;
  first.support_index = 0;
  mppi_core::TactileContactPoint second;
  second.active = true;
  second.support_index = 1;
  tactile.contact_points.push_back(first);
  tactile.contact_points.push_back(second);

  mppi_core::GraspRolloutConfig config;
  mppi_core::RefreshPredictedTactileFields(&tactile, config);

  EXPECT_EQ(tactile.contact_presence, mppi_core::ContactPresence::kNoContact);
  EXPECT_EQ(tactile.contact_support_count, 0U);
  EXPECT_EQ(tactile.contact_points.size(), 2U);
  EXPECT_EQ(tactile.activeContactPointCount(), 0U);
  EXPECT_FALSE(tactile.contact_points[0].active);
  EXPECT_FALSE(tactile.contact_points[1].active);
}

TEST(GraspRolloutTest, StepGraspTactilePatchUpdatesRobotAndTactileState) {
  mppi_core::TactileState tactile;
  tactile.valid = true;
  tactile.contact_presence = mppi_core::ContactPresence::kStableContact;
  tactile.normal_force_n = 1.0;
  tactile.has_centroid = true;
  tactile.centroid_m = Eigen::Vector2d::Zero();
  tactile.contact_support_count = 4;
  tactile.support_count = 8;
  tactile.confidence = 1.0;

  const Eigen::VectorXd q = Eigen::VectorXd::Zero(2);
  const Eigen::VectorXd dq = Eigen::VectorXd::Zero(2);
  const Eigen::VectorXd tau = Eigen::VectorXd::Zero(2);
  const auto state = mppi_core::MakeGraspState(q, dq, tau, tactile);

  mppi_core::ContactPointMotion motion;
  motion.position_sensor_m = Eigen::Vector3d::Zero();
  motion.delta_position_sensor_m = Eigen::Vector3d{0.001, 0.0, 0.0};

  const Eigen::VectorXd next_q = Eigen::VectorXd::Constant(2, 0.2);
  const Eigen::VectorXd next_dq = Eigen::VectorXd::Constant(2, 0.3);
  const Eigen::VectorXd next_tau = Eigen::VectorXd::Constant(2, 0.4);
  const auto next = mppi_core::StepGraspTactilePatch(
      state, next_q, next_dq, next_tau,
      std::vector<mppi_core::ContactPointMotion>{motion}, 0.01);

  EXPECT_TRUE(next.valid);
  EXPECT_NEAR(next.q[0], 0.2, kTolerance);
  EXPECT_NEAR(next.dq[1], 0.3, kTolerance);
  EXPECT_NEAR(next.tau[0], 0.4, kTolerance);
  ASSERT_TRUE(next.tactile.has_centroid);
  EXPECT_GT(next.tactile.centroid_m.x(), 0.0);
}

TEST(DeltaQReferenceRolloutModelTest, StateTauDrivesForceAwareTactileRollout) {
  const auto sensor_model = MakeSinglePrismaticZSensorModel();
  pinocchio::Data data(sensor_model.model);

  mppi_core::TactileState tactile;
  tactile.valid = true;
  tactile.contact_presence = mppi_core::ContactPresence::kStableContact;
  tactile.has_normal_force = true;
  tactile.normal_force_n = 0.5;
  tactile.contact_support_count = 1;
  tactile.support_count = 1;
  tactile.confidence = 1.0;
  mppi_core::TactileContactPoint point;
  point.active = true;
  point.support_index = 0;
  point.position_sensor_m = Eigen::Vector3d::Zero();
  tactile.contact_points.push_back(point);

  mppi_core::PinocchioContactKinematicsContext kinematics;
  kinematics.model = &sensor_model.model;
  kinematics.data = &data;
  kinematics.sensor_frame_id = sensor_model.sensor_frame_id;

  mppi_core::ContactForceProjectionConfig projection_config;
  projection_config.regularization = 1.0e-9;
  projection_config.tactile_prior_weight = 0.0;
  mppi_core::ContactForceRolloutConfig force_rollout_config;
  force_rollout_config.force_lowpass_alpha = 1.0;
  force_rollout_config.min_stable_support_count = 1;

  mppi_core::ContactForceCorrectionState correction;
  correction.normal_force_bias_n = 0.5;

  mppi_core::RolloutContext context;
  context.tactile = &tactile;
  context.contact_kinematics = &kinematics;
  context.contact_force_projection_config = &projection_config;
  context.contact_force_rollout_config = &force_rollout_config;
  context.contact_force_correction_state = &correction;

  mppi_core::DeltaQReferenceRolloutModel model(1);
  const Eigen::VectorXd q = Eigen::VectorXd::Zero(1);
  const Eigen::VectorXd dq = Eigen::VectorXd::Zero(1);
  const Eigen::VectorXd tau = Eigen::VectorXd::Constant(1, 1.0);
  const auto state = mppi_core::MakeGraspState(q, dq, tau, tactile);
  const Eigen::VectorXd action = Eigen::VectorXd::Zero(1);
  mppi_core::RobotRolloutState next_state;

  model.Step(state, action, context, 0.1, &next_state);

  EXPECT_TRUE(next_state.valid);
  EXPECT_NEAR(next_state.tactile.normal_force_n, 1.5, 1.0e-6);
  EXPECT_EQ(next_state.tactile.contact_presence,
            mppi_core::ContactPresence::kStableContact);
  EXPECT_EQ(next_state.tactile.contact_support_count, 1U);
  EXPECT_NEAR(next_state.q[0], 0.0, kTolerance);
  EXPECT_NEAR(next_state.tau[0], 0.0, kTolerance);
}

TEST(DeltaQReferenceRolloutModelTest, RejectsNonFiniteAction) {
  mppi_core::TactileState tactile;
  tactile.valid = true;
  tactile.contact_presence = mppi_core::ContactPresence::kStableContact;
  tactile.has_normal_force = true;
  tactile.normal_force_n = 1.0;

  const auto state = mppi_core::MakeGraspState(
      Eigen::VectorXd::Zero(1), Eigen::VectorXd::Zero(1),
      Eigen::VectorXd::Zero(1), tactile);
  Eigen::VectorXd action = Eigen::VectorXd::Zero(1);
  action[0] = std::numeric_limits<double>::quiet_NaN();

  mppi_core::DeltaQReferenceRolloutConfig config;
  config.tactile_rollout_policy =
      mppi_core::TactileRolloutPolicy::kForceThenKinematicFallback;
  mppi_core::DeltaQReferenceRolloutModel model(1, config);
  mppi_core::RolloutContext context;
  mppi_core::RobotRolloutState next_state;

  EXPECT_THROW(model.Step(state, action, context, 0.01, &next_state),
               std::invalid_argument);
}

TEST(DeltaQReferenceRolloutModelTest, RolloutTorqueModelAssignsFutureTau) {
  const auto sensor_model = MakeSinglePrismaticZSensorModel();
  pinocchio::Data data(sensor_model.model);

  mppi_core::TactileState tactile;
  tactile.valid = true;
  tactile.contact_presence = mppi_core::ContactPresence::kStableContact;
  tactile.has_normal_force = true;
  tactile.normal_force_n = 0.0;
  tactile.contact_support_count = 1;
  tactile.support_count = 1;
  tactile.confidence = 1.0;
  mppi_core::TactileContactPoint point;
  point.active = true;
  point.support_index = 0;
  point.position_sensor_m = Eigen::Vector3d::Zero();
  tactile.contact_points.push_back(point);

  mppi_core::PinocchioContactKinematicsContext kinematics;
  kinematics.model = &sensor_model.model;
  kinematics.data = &data;
  kinematics.sensor_frame_id = sensor_model.sensor_frame_id;
  mppi_core::ContactForceProjectionConfig projection_config;
  mppi_core::ContactForceRolloutConfig force_rollout_config;
  force_rollout_config.rollout_torque_stiffness_nm_per_rad = 100.0;
  force_rollout_config.rollout_torque_damping_nms_per_rad = 0.5;

  mppi_core::RolloutContext context;
  context.tactile = &tactile;
  context.contact_kinematics = &kinematics;
  context.contact_force_projection_config = &projection_config;
  context.contact_force_rollout_config = &force_rollout_config;

  mppi_core::DeltaQReferenceRolloutModel model(1);
  const Eigen::VectorXd q = Eigen::VectorXd::Zero(1);
  const Eigen::VectorXd dq = Eigen::VectorXd::Constant(1, 0.4);
  const Eigen::VectorXd tau = Eigen::VectorXd::Zero(1);
  const auto state = mppi_core::MakeGraspState(q, dq, tau, tactile);
  const Eigen::VectorXd action = Eigen::VectorXd::Constant(1, 0.02);
  mppi_core::RobotRolloutState next_state;

  model.Step(state, action, context, 0.1, &next_state);

  EXPECT_TRUE(next_state.valid);
  EXPECT_NEAR(next_state.q[0], 0.02, kTolerance);
  EXPECT_NEAR(next_state.dq[0], 0.2, kTolerance);
  EXPECT_NEAR(next_state.tau[0], 100.0 * 0.02 - 0.5 * 0.4, kTolerance);
}

TEST(DeltaQReferenceRolloutModelTest, PinocchioRolloutAllowsNqDifferentFromNv) {
  const auto sensor_model = MakeSingleUnboundedRevoluteZSensorModel();
  ASSERT_EQ(sensor_model.model.nq, 2);
  ASSERT_EQ(sensor_model.model.nv, 1);
  pinocchio::Data data(sensor_model.model);

  mppi_core::TactileState tactile;
  tactile.valid = true;
  tactile.contact_presence = mppi_core::ContactPresence::kStableContact;
  tactile.has_normal_force = true;
  tactile.normal_force_n = 1.0;
  tactile.has_centroid = true;
  tactile.centroid_m = Eigen::Vector2d::Zero();
  tactile.contact_support_count = 1;
  tactile.support_count = 1;
  tactile.confidence = 1.0;
  mppi_core::TactileContactPoint point;
  point.active = true;
  point.position_sensor_m = Eigen::Vector3d{1.0, 0.0, 0.0};
  tactile.contact_points.push_back(point);

  mppi_core::PinocchioContactKinematicsContext kinematics;
  kinematics.model = &sensor_model.model;
  kinematics.data = &data;
  kinematics.sensor_frame_id = sensor_model.sensor_frame_id;
  mppi_core::ContactForceProjectionConfig projection_config;
  mppi_core::ContactForceRolloutConfig force_rollout_config;
  force_rollout_config.min_stable_support_count = 1;

  mppi_core::RolloutContext context;
  context.tactile = &tactile;
  context.contact_kinematics = &kinematics;
  context.contact_force_projection_config = &projection_config;
  context.contact_force_rollout_config = &force_rollout_config;

  mppi_core::DeltaQReferenceRolloutConfig model_config;
  model_config.tactile_rollout_policy =
      mppi_core::TactileRolloutPolicy::kForceThenKinematicFallback;
  mppi_core::DeltaQReferenceRolloutModel model(1, model_config);
  const auto state = MakeContactKinematicsState(sensor_model.model, tactile);
  const Eigen::VectorXd action = Eigen::VectorXd::Constant(1, 0.2);
  mppi_core::RobotRolloutState next_state;

  model.Step(state, action, context, 0.1, &next_state);

  EXPECT_TRUE(next_state.valid);
  EXPECT_EQ(next_state.q.size(), 2);
  EXPECT_EQ(next_state.dq.size(), 1);
  EXPECT_EQ(next_state.tau.size(), 1);
  EXPECT_TRUE(next_state.q.allFinite());
  EXPECT_NEAR(next_state.dq[0], 2.0, kTolerance);
  EXPECT_NEAR(next_state.tau[0], 0.2, kTolerance);
}

TEST(DeltaQReferenceRolloutModelTest,
     ForceAwareFailureFallsBackToKinematicPatchRollout) {
  const auto sensor_model = MakeSinglePrismaticZSensorModel();
  pinocchio::Data data(sensor_model.model);

  mppi_core::TactileState tactile;
  tactile.valid = true;
  tactile.contact_presence = mppi_core::ContactPresence::kStableContact;
  tactile.has_normal_force = true;
  tactile.normal_force_n = 0.5;
  tactile.has_centroid = true;
  tactile.centroid_m = Eigen::Vector2d::Zero();
  tactile.contact_support_count = 1;
  tactile.support_count = 1;
  tactile.confidence = 1.0;
  mppi_core::TactileContactPoint point;
  point.active = true;
  point.position_sensor_m = Eigen::Vector3d::Zero();
  tactile.contact_points.push_back(point);

  mppi_core::PinocchioContactKinematicsContext kinematics;
  kinematics.model = &sensor_model.model;
  kinematics.data = &data;
  kinematics.sensor_frame_id = sensor_model.sensor_frame_id;

  mppi_core::ContactForceProjectionConfig projection_config;
  projection_config.enabled = false;
  mppi_core::GraspRolloutConfig rollout_config;
  rollout_config.min_stable_support_count = 1;
  rollout_config.normal_force_gain_n_per_m = 100.0;

  mppi_core::RolloutContext context;
  context.tactile = &tactile;
  context.contact_kinematics = &kinematics;
  context.grasp_rollout_config = &rollout_config;
  context.contact_force_projection_config = &projection_config;

  mppi_core::DeltaQReferenceRolloutConfig model_config;
  model_config.tactile_rollout_policy =
      mppi_core::TactileRolloutPolicy::kForceThenKinematicFallback;
  mppi_core::DeltaQReferenceRolloutModel model(1, model_config);
  const auto state = MakeContactKinematicsState(sensor_model.model, tactile);
  const Eigen::VectorXd action = Eigen::VectorXd::Constant(1, 0.01);
  mppi_core::RobotRolloutState next_state;

  model.Step(state, action, context, 0.1, &next_state);

  EXPECT_TRUE(next_state.valid);
  EXPECT_NEAR(next_state.tactile.normal_force_n, 1.5, 1.0e-6);
  EXPECT_EQ(next_state.tactile.contact_presence,
            mppi_core::ContactPresence::kStableContact);
}

TEST(DeltaQReferenceRolloutModelTest,
     RequireForceAwareMarksForceProjectionFailureInvalid) {
  const auto sensor_model = MakeSinglePrismaticZSensorModel();
  pinocchio::Data data(sensor_model.model);

  mppi_core::TactileState tactile;
  tactile.valid = true;
  tactile.contact_presence = mppi_core::ContactPresence::kStableContact;
  tactile.has_normal_force = true;
  tactile.normal_force_n = 0.5;
  tactile.has_centroid = true;
  tactile.centroid_m = Eigen::Vector2d::Zero();
  tactile.contact_support_count = 1;
  tactile.support_count = 1;
  tactile.confidence = 1.0;
  mppi_core::TactileContactPoint point;
  point.active = true;
  point.position_sensor_m = Eigen::Vector3d::Zero();
  tactile.contact_points.push_back(point);

  mppi_core::PinocchioContactKinematicsContext kinematics;
  kinematics.model = &sensor_model.model;
  kinematics.data = &data;
  kinematics.sensor_frame_id = sensor_model.sensor_frame_id;

  mppi_core::ContactForceProjectionConfig projection_config;
  projection_config.enabled = false;
  mppi_core::GraspRolloutConfig rollout_config;
  rollout_config.min_stable_support_count = 1;
  rollout_config.normal_force_gain_n_per_m = 100.0;

  mppi_core::RolloutContext context;
  context.tactile = &tactile;
  context.contact_kinematics = &kinematics;
  context.grasp_rollout_config = &rollout_config;
  context.contact_force_projection_config = &projection_config;

  mppi_core::DeltaQReferenceRolloutConfig config;
  config.tactile_rollout_policy =
      mppi_core::TactileRolloutPolicy::kForceAwareRequired;
  mppi_core::DeltaQReferenceRolloutModel model(1, config);
  const auto state = MakeContactKinematicsState(sensor_model.model, tactile);
  const Eigen::VectorXd action = Eigen::VectorXd::Constant(1, 0.01);
  mppi_core::RobotRolloutState next_state;

  model.Step(state, action, context, 0.1, &next_state);

  EXPECT_FALSE(next_state.valid);
  EXPECT_TRUE(next_state.q.allFinite());
  EXPECT_TRUE(next_state.dq.allFinite());
  EXPECT_TRUE(next_state.tau.allFinite());
}

TEST(DeltaQReferenceRolloutModelTest,
     RequireForceAwareRequiresActiveTactileContact) {
  mppi_core::TactileState tactile;
  tactile.valid = true;
  tactile.contact_presence = mppi_core::ContactPresence::kNoContact;
  tactile.has_normal_force = true;
  tactile.normal_force_n = 0.0;
  tactile.contact_support_count = 0;
  tactile.support_count = 1;
  tactile.confidence = 1.0;

  mppi_core::DeltaQReferenceRolloutConfig config;
  config.tactile_rollout_policy =
      mppi_core::TactileRolloutPolicy::kForceAwareRequired;
  mppi_core::DeltaQReferenceRolloutModel model(1, config);
  const auto state = MakeState(1, tactile);
  const Eigen::VectorXd action = Eigen::VectorXd::Zero(1);
  mppi_core::RolloutContext context;
  context.tactile = &tactile;
  mppi_core::RobotRolloutState next_state;

  model.Step(state, action, context, 0.1, &next_state);

  EXPECT_FALSE(next_state.valid);
}

TEST(DeltaQReferenceRolloutModelTest,
     ForceThenKinematicFallbackRejectsMissingKinematics) {
  mppi_core::TactileState tactile;
  tactile.valid = true;
  tactile.contact_presence = mppi_core::ContactPresence::kStableContact;
  tactile.has_normal_force = true;
  tactile.normal_force_n = 1.0;
  tactile.contact_support_count = 1;
  tactile.support_count = 1;
  tactile.confidence = 1.0;

  mppi_core::DeltaQReferenceRolloutConfig config;
  config.tactile_rollout_policy =
      mppi_core::TactileRolloutPolicy::kForceThenKinematicFallback;
  mppi_core::DeltaQReferenceRolloutModel model(1, config);
  const auto state = MakeState(1, tactile);
  const Eigen::VectorXd action = Eigen::VectorXd::Constant(1, 0.01);
  mppi_core::RolloutContext context;
  context.tactile = &tactile;
  mppi_core::RobotRolloutState next_state;

  model.Step(state, action, context, 0.1, &next_state);

  EXPECT_FALSE(next_state.valid);
}

TEST(DeltaQReferenceRolloutModelTest,
     PinocchioContactMotionsDriveTactilePatchRollout) {
  const auto sensor_model = MakeSingleRevoluteZSensorModel();
  pinocchio::Data data(sensor_model.model);

  mppi_core::TactileState tactile;
  tactile.valid = true;
  tactile.contact_presence = mppi_core::ContactPresence::kStableContact;
  tactile.has_normal_force = true;
  tactile.normal_force_n = 1.0;
  tactile.has_centroid = true;
  tactile.centroid_m = Eigen::Vector2d::Zero();
  tactile.has_shear = true;
  tactile.shear_displacement_m = Eigen::Vector2d{0.0, 0.01};
  tactile.contact_support_count = 1;
  tactile.support_count = 1;
  tactile.confidence = 1.0;
  mppi_core::TactileContactPoint point;
  point.active = true;
  point.position_sensor_m = Eigen::Vector3d{1.0, 0.0, 0.0};
  tactile.contact_points.push_back(point);

  mppi_core::PinocchioContactKinematicsContext kinematics;
  kinematics.model = &sensor_model.model;
  kinematics.data = &data;
  kinematics.sensor_frame_id = sensor_model.sensor_frame_id;
  mppi_core::GraspRolloutConfig rollout_config;
  rollout_config.min_stable_support_count = 1;
  rollout_config.tangential_confidence_loss_per_m = 0.0;
  rollout_config.edge_confidence_loss_gain = 0.0;
  rollout_config.shear_ref_m = 1.0;
  mppi_core::ContactForceProjectionConfig projection_config;
  projection_config.enabled = false;

  mppi_core::RolloutContext context;
  context.tactile = &tactile;
  context.contact_kinematics = &kinematics;
  context.grasp_rollout_config = &rollout_config;
  context.contact_force_projection_config = &projection_config;

  mppi_core::DeltaQReferenceRolloutConfig model_config;
  model_config.tactile_rollout_policy =
      mppi_core::TactileRolloutPolicy::kForceThenKinematicFallback;
  mppi_core::DeltaQReferenceRolloutModel model(1, model_config);
  const auto state = MakeState(1, tactile);
  const Eigen::VectorXd action = Eigen::VectorXd::Constant(1, 0.005);
  mppi_core::RobotRolloutState next_state;

  model.Step(state, action, context, 0.1, &next_state);

  EXPECT_TRUE(next_state.valid);
  ASSERT_TRUE(next_state.tactile.has_shear);
  EXPECT_GT(next_state.tactile.shear_displacement_m.y(),
            tactile.shear_displacement_m.y());
  EXPECT_GT(next_state.tactile.slip_score, tactile.shear_displacement_m.norm());
  EXPECT_GT(next_state.tactile.centroid_m.y(), tactile.centroid_m.y());
}

TEST(DeltaQReferenceRolloutModelTest,
     OpposingPinocchioTangentialMotionReducesPredictedShear) {
  const auto sensor_model = MakeSingleRevoluteZSensorModel();
  pinocchio::Data data(sensor_model.model);

  mppi_core::TactileState tactile;
  tactile.valid = true;
  tactile.contact_presence = mppi_core::ContactPresence::kStableContact;
  tactile.has_normal_force = true;
  tactile.normal_force_n = 1.0;
  tactile.has_centroid = true;
  tactile.centroid_m = Eigen::Vector2d::Zero();
  tactile.has_shear = true;
  tactile.shear_displacement_m = Eigen::Vector2d{0.0, 0.01};
  tactile.contact_support_count = 1;
  tactile.support_count = 1;
  tactile.confidence = 1.0;
  mppi_core::TactileContactPoint point;
  point.active = true;
  point.position_sensor_m = Eigen::Vector3d{1.0, 0.0, 0.0};
  tactile.contact_points.push_back(point);

  mppi_core::PinocchioContactKinematicsContext kinematics;
  kinematics.model = &sensor_model.model;
  kinematics.data = &data;
  kinematics.sensor_frame_id = sensor_model.sensor_frame_id;
  mppi_core::GraspRolloutConfig rollout_config;
  rollout_config.min_stable_support_count = 1;
  rollout_config.tangential_confidence_loss_per_m = 0.0;
  rollout_config.edge_confidence_loss_gain = 0.0;
  rollout_config.shear_ref_m = 1.0;
  mppi_core::ContactForceProjectionConfig projection_config;
  projection_config.enabled = false;

  mppi_core::RolloutContext context;
  context.tactile = &tactile;
  context.contact_kinematics = &kinematics;
  context.grasp_rollout_config = &rollout_config;
  context.contact_force_projection_config = &projection_config;

  mppi_core::DeltaQReferenceRolloutConfig model_config;
  model_config.tactile_rollout_policy =
      mppi_core::TactileRolloutPolicy::kForceThenKinematicFallback;
  mppi_core::DeltaQReferenceRolloutModel model(1, model_config);
  const auto state = MakeState(1, tactile);
  const Eigen::VectorXd action = Eigen::VectorXd::Constant(1, -0.005);
  mppi_core::RobotRolloutState next_state;

  model.Step(state, action, context, 0.1, &next_state);

  ASSERT_TRUE(next_state.tactile.has_shear);
  EXPECT_LT(next_state.tactile.shear_displacement_m.norm(),
            tactile.shear_displacement_m.norm());
}

TEST(DeltaQReferenceRolloutModelTest,
     ZeroPinocchioActionLeavesPatchGeometryUnchanged) {
  const auto sensor_model = MakeSingleRevoluteZSensorModel();
  pinocchio::Data data(sensor_model.model);

  mppi_core::TactileState tactile;
  tactile.valid = true;
  tactile.contact_presence = mppi_core::ContactPresence::kStableContact;
  tactile.has_normal_force = true;
  tactile.normal_force_n = 1.0;
  tactile.has_centroid = true;
  tactile.centroid_m = Eigen::Vector2d{0.001, 0.0};
  tactile.has_shear = true;
  tactile.shear_displacement_m = Eigen::Vector2d{0.0, 0.01};
  tactile.contact_support_count = 1;
  tactile.support_count = 1;
  tactile.confidence = 0.8;
  mppi_core::TactileContactPoint point;
  point.active = true;
  point.position_sensor_m = Eigen::Vector3d{1.0, 0.0, 0.0};
  tactile.contact_points.push_back(point);

  mppi_core::PinocchioContactKinematicsContext kinematics;
  kinematics.model = &sensor_model.model;
  kinematics.data = &data;
  kinematics.sensor_frame_id = sensor_model.sensor_frame_id;
  mppi_core::GraspRolloutConfig rollout_config;
  rollout_config.min_stable_support_count = 1;
  rollout_config.tangential_confidence_loss_per_m = 0.0;
  rollout_config.edge_confidence_loss_gain = 0.0;
  rollout_config.shear_ref_m = 1.0;
  mppi_core::ContactForceProjectionConfig projection_config;
  projection_config.enabled = false;

  mppi_core::RolloutContext context;
  context.tactile = &tactile;
  context.contact_kinematics = &kinematics;
  context.grasp_rollout_config = &rollout_config;
  context.contact_force_projection_config = &projection_config;

  mppi_core::DeltaQReferenceRolloutConfig model_config;
  model_config.tactile_rollout_policy =
      mppi_core::TactileRolloutPolicy::kForceThenKinematicFallback;
  mppi_core::DeltaQReferenceRolloutModel model(1, model_config);
  const auto state = MakeState(1, tactile);
  const Eigen::VectorXd action = Eigen::VectorXd::Zero(1);
  mppi_core::RobotRolloutState next_state;

  model.Step(state, action, context, 0.1, &next_state);

  ASSERT_TRUE(next_state.tactile.has_centroid);
  ASSERT_TRUE(next_state.tactile.has_shear);
  EXPECT_NEAR(next_state.tactile.centroid_m.x(), tactile.centroid_m.x(),
              kTolerance);
  EXPECT_NEAR(next_state.tactile.centroid_m.y(), tactile.centroid_m.y(),
              kTolerance);
  EXPECT_NEAR(next_state.tactile.shear_displacement_m.x(),
              tactile.shear_displacement_m.x(), kTolerance);
  EXPECT_NEAR(next_state.tactile.shear_displacement_m.y(),
              tactile.shear_displacement_m.y(), kTolerance);
  EXPECT_NEAR(next_state.tactile.normal_force_n, tactile.normal_force_n,
              kTolerance);
  EXPECT_NEAR(next_state.tactile.confidence, tactile.confidence, kTolerance);
}

TEST(DeltaQReferenceRolloutModelTest,
     ForceAwareRequiredMarksNoActiveContactPointsInvalid) {
  const auto sensor_model = MakeSingleRevoluteZSensorModel();
  pinocchio::Data data(sensor_model.model);

  mppi_core::TactileState tactile;
  tactile.valid = true;
  tactile.contact_presence = mppi_core::ContactPresence::kStableContact;
  tactile.has_normal_force = true;
  tactile.normal_force_n = 1.0;
  tactile.has_centroid = true;
  tactile.centroid_m = Eigen::Vector2d::Zero();
  tactile.has_shear = true;
  tactile.shear_displacement_m = Eigen::Vector2d{0.0, 0.01};
  tactile.contact_support_count = 1;
  tactile.support_count = 1;
  tactile.confidence = 1.0;
  mppi_core::TactileContactPoint point;
  point.active = false;
  point.position_sensor_m = Eigen::Vector3d{1.0, 0.0, 0.0};
  tactile.contact_points.push_back(point);

  mppi_core::PinocchioContactKinematicsContext kinematics;
  kinematics.model = &sensor_model.model;
  kinematics.data = &data;
  kinematics.sensor_frame_id = sensor_model.sensor_frame_id;
  mppi_core::GraspRolloutConfig rollout_config;
  rollout_config.min_stable_support_count = 1;
  rollout_config.shear_ref_m = 1.0;

  mppi_core::RolloutContext context;
  context.tactile = &tactile;
  context.contact_kinematics = &kinematics;
  context.grasp_rollout_config = &rollout_config;

  mppi_core::DeltaQReferenceRolloutModel model(1);
  const auto state = MakeState(1, tactile);
  const Eigen::VectorXd action = Eigen::VectorXd::Constant(1, 0.005);
  mppi_core::RobotRolloutState next_state;

  model.Step(state, action, context, 0.1, &next_state);

  EXPECT_FALSE(next_state.valid);
  EXPECT_EQ(next_state.tactile.activeContactPointCount(), 0U);
}

TEST(GraspStabilityCostTest, PenalizesSmallPredictedContactPatch) {
  mppi_core::GraspStabilityCostConfig config;
  config.force_min_n = 0.0;
  config.force_max_n = 10.0;
  config.force_under_weight = 0.0;
  config.force_over_weight = 0.0;
  config.slip_risk_weight = 0.0;
  config.contact_centroid_enabled = true;
  config.centroid_boundary_weight = 0.0;
  config.contact_loss_weight = 0.0;
  config.tracking_weight = 0.0;
  config.tracking_action_scale_weight = 0.0;
  config.action_smoothness_weight = 0.0;
  config.joint_limit_weight = 0.0;
  config.contact_patch_enabled = true;
  config.contact_patch_target_node_count = 6.0;
  config.contact_patch_weight = 1.0;

  mppi_core::GraspStabilityCost cost(config);

  mppi_core::TactileState tactile;
  tactile.valid = true;
  tactile.contact_presence = mppi_core::ContactPresence::kStableContact;
  tactile.has_normal_force = true;
  tactile.normal_force_n = 1.0;
  tactile.has_centroid = true;
  tactile.support_count = 8;
  mppi_core::RolloutContext rollout;
  rollout.tactile = &tactile;
  mppi_core::CostContext context;
  context.rollout = &rollout;
  const Eigen::VectorXd action = Eigen::VectorXd::Zero(1);

  auto low_patch_state = MakeState(1, tactile);
  low_patch_state.tactile.contact_support_count = 2;

  auto wide_patch_state = low_patch_state;
  wide_patch_state.tactile.contact_support_count = 6;

  const double low_patch_cost = cost.Evaluate(low_patch_state, action, context);
  const double wide_patch_cost =
      cost.Evaluate(wide_patch_state, action, context);

  EXPECT_GT(low_patch_cost, wide_patch_cost);
  EXPECT_NEAR(wide_patch_cost, 0.0, kTolerance);
}

TEST(GraspStabilityCostTest, DisablingCentroidCostDoesNotAddContactLossCost) {
  mppi_core::GraspStabilityCostConfig config;
  config.force_min_n = 0.0;
  config.force_max_n = 10.0;
  config.force_under_weight = 0.0;
  config.force_over_weight = 0.0;
  config.slip_risk_weight = 0.0;
  config.contact_centroid_enabled = false;
  config.centroid_boundary_weight = 0.0;
  config.contact_loss_weight = 10.0;
  config.contact_patch_enabled = false;
  config.tracking_weight = 0.0;
  config.tracking_action_scale_weight = 0.0;
  config.action_smoothness_weight = 0.0;
  config.joint_limit_weight = 0.0;

  mppi_core::GraspStabilityCost cost(config);

  mppi_core::TactileState tactile;
  tactile.valid = true;
  tactile.contact_presence = mppi_core::ContactPresence::kStableContact;
  tactile.has_normal_force = true;
  tactile.normal_force_n = 1.0;
  tactile.has_centroid = false;
  tactile.contact_support_count = 1;
  tactile.support_count = 1;

  auto state = MakeState(1, tactile);
  mppi_core::RolloutContext rollout;
  rollout.tactile = &tactile;
  mppi_core::CostContext context;
  context.rollout = &rollout;
  const Eigen::VectorXd action = Eigen::VectorXd::Zero(1);

  EXPECT_NEAR(cost.Evaluate(state, action, context), 0.0, kTolerance);
}

TEST(GraspConfigTest, ParsesContactLocalCostAndForceRolloutConfig) {
  const YAML::Node root = YAML::Load(R"(
grasp:
  tactile_prediction:
    rollout_policy: force_then_kinematic_fallback
  contact_force_rollout:
    enable_force_projection_update: true
    force_lowpass_alpha: 0.7
    max_predicted_normal_force_n: 12.0
    shear_force_gain_m_per_n_s: 0.0002
    rotational_shear_gain_rad_per_nm_s: 0.03
    friction_violation_confidence_decay: 0.4
    negative_normal_confidence_decay: 0.6
    min_stable_support_count: 2
    min_contact_confidence: 0.02
    shear_ref_m: 0.004
    rotational_shear_ref_rad: 0.05
    rollout_torque_stiffness_nm_per_rad: 3.0
    rollout_torque_damping_nms_per_rad: 0.2
  slip_risk:
    velocity_weight: 0.05
  contact_patch:
    enabled: true
    target_node_count: 7
    weight: 3.5
)");

  const auto cost_config = mppi_core::ParseGraspConfig(
      root["grasp"], 2, mppi_core::GraspStabilityCostConfig{});
  const auto rollout_config = mppi_core::ParseDeltaQReferenceRolloutConfig(
      root["grasp"], mppi_core::DeltaQReferenceRolloutConfig{});
  const auto force_rollout_config = mppi_core::ParseContactForceRolloutConfig(
      root["grasp"], mppi_core::ContactForceRolloutConfig{});

  EXPECT_NEAR(cost_config.slip_velocity_weight, 0.05, kTolerance);
  EXPECT_TRUE(cost_config.contact_patch_enabled);
  EXPECT_NEAR(cost_config.contact_patch_target_node_count, 7.0, kTolerance);
  EXPECT_NEAR(cost_config.contact_patch_weight, 3.5, kTolerance);
  EXPECT_EQ(rollout_config.tactile_rollout_policy,
            mppi_core::TactileRolloutPolicy::kForceThenKinematicFallback);
  EXPECT_TRUE(force_rollout_config.enable_force_projection_update);
  EXPECT_NEAR(force_rollout_config.force_lowpass_alpha, 0.7, kTolerance);
  EXPECT_NEAR(force_rollout_config.max_predicted_normal_force_n, 12.0,
              kTolerance);
  EXPECT_NEAR(force_rollout_config.shear_force_gain_m_per_n_s, 0.0002,
              kTolerance);
  EXPECT_NEAR(force_rollout_config.rotational_shear_gain_rad_per_nm_s, 0.03,
              kTolerance);
  EXPECT_NEAR(force_rollout_config.friction_violation_confidence_decay, 0.4,
              kTolerance);
  EXPECT_NEAR(force_rollout_config.negative_normal_confidence_decay, 0.6,
              kTolerance);
  EXPECT_EQ(force_rollout_config.min_stable_support_count, 2U);
  EXPECT_NEAR(force_rollout_config.min_contact_confidence, 0.02, kTolerance);
  EXPECT_NEAR(force_rollout_config.shear_ref_m, 0.004, kTolerance);
  EXPECT_NEAR(force_rollout_config.rotational_shear_ref_rad, 0.05, kTolerance);
  EXPECT_NEAR(force_rollout_config.rollout_torque_stiffness_nm_per_rad, 3.0,
              kTolerance);
  EXPECT_NEAR(force_rollout_config.rollout_torque_damping_nms_per_rad, 0.2,
              kTolerance);
}

TEST(MPPIConfigTest, ParsesSamplingAndExpandsScalarActionParameters) {
  const YAML::Node root = YAML::Load(R"(
mppi:
  horizon_steps: 15
  dt: 0.02
  num_rollouts: 64
  temperature: 0.8
  random_seed: 42
  action:
    lower_bound: -0.003
    upper_bound: 0.004
    noise_std: 0.001
)");

  const auto config =
      mppi_core::ParseMPPIConfig(root["mppi"], 3, mppi_core::MPPIConfig{});

  EXPECT_EQ(config.horizon_steps, 15U);
  EXPECT_EQ(config.num_rollouts, 64U);
  EXPECT_EQ(config.action_dim, 3U);
  EXPECT_NEAR(config.dt, 0.02, kTolerance);
  EXPECT_NEAR(config.temperature, 0.8, kTolerance);
  EXPECT_EQ(config.random_seed, 42U);
  ASSERT_EQ(config.action_lower_bound.size(), 3);
  ASSERT_EQ(config.action_upper_bound.size(), 3);
  ASSERT_EQ(config.action_noise_std.size(), 3);
  for (Eigen::Index i = 0; i < 3; ++i) {
    EXPECT_NEAR(config.action_lower_bound[i], -0.003, kTolerance);
    EXPECT_NEAR(config.action_upper_bound[i], 0.004, kTolerance);
    EXPECT_NEAR(config.action_noise_std[i], 0.001, kTolerance);
  }
}

TEST(MPPIConfigTest, RejectsWrongSizedActionVectors) {
  const YAML::Node root = YAML::Load(R"(
mppi:
  action:
    lower_bound: [-0.1, -0.2]
)");

  EXPECT_THROW((void)mppi_core::ParseMPPIConfig(root["mppi"], 3,
                                                mppi_core::MPPIConfig{}),
               std::invalid_argument);
}

TEST(MPPIOptimizerTest, PredictRolloutRecordsActionsStatesAndStepCosts) {
  const auto sensor_model = MakeSinglePrismaticZSensorModel();
  pinocchio::Data data(sensor_model.model);

  mppi_core::MPPIConfig config;
  config.horizon_steps = 2;
  config.num_rollouts = 1;
  config.action_dim = 1;
  config.dt = 0.1;
  config.temperature = 1.0;
  config.action_lower_bound = Eigen::VectorXd::Constant(1, -1.0);
  config.action_upper_bound = Eigen::VectorXd::Constant(1, 1.0);
  config.action_noise_std = Eigen::VectorXd::Zero(1);

  auto model = std::make_shared<mppi_core::DeltaQReferenceRolloutModel>(1);
  mppi_core::MPPIOptimizer optimizer;
  optimizer.Initialize(config, model, nullptr);

  mppi_core::TactileState tactile;
  tactile.valid = true;
  tactile.contact_presence = mppi_core::ContactPresence::kStableContact;
  tactile.has_normal_force = true;
  tactile.normal_force_n = 1.0;
  tactile.contact_support_count = 1;
  tactile.support_count = 1;
  tactile.confidence = 1.0;
  mppi_core::TactileContactPoint point;
  point.active = true;
  point.support_index = 0;
  point.position_sensor_m = Eigen::Vector3d::Zero();
  tactile.contact_points.push_back(point);

  mppi_core::PinocchioContactKinematicsContext kinematics;
  kinematics.model = &sensor_model.model;
  kinematics.data = &data;
  kinematics.sensor_frame_id = sensor_model.sensor_frame_id;
  mppi_core::ContactForceProjectionConfig projection_config;
  mppi_core::ContactForceRolloutConfig force_rollout_config;
  force_rollout_config.min_stable_support_count = 1;

  mppi_core::GraspObservation observation;
  observation.q_ref_current = Eigen::VectorXd::Constant(1, 0.5);
  observation.v_ref_current = Eigen::VectorXd::Zero(1);
  observation.q_measured = observation.q_ref_current;
  observation.v_measured = observation.v_ref_current;
  observation.tau = Eigen::VectorXd::Zero(1);
  observation.tactile = tactile;
  observation.contact_kinematics = &kinematics;
  observation.contact_force_projection_config = &projection_config;
  observation.contact_force_rollout_config = &force_rollout_config;

  mppi_core::ActionSequence actions(1, 2);
  actions.setAction(0, Eigen::VectorXd::Constant(1, 0.1));
  actions.setAction(1, Eigen::VectorXd::Constant(1, -0.05));

  const auto trace = optimizer.PredictRollout(observation, actions);

  ASSERT_EQ(trace.actions.size(), 2U);
  ASSERT_EQ(trace.step_costs.size(), 2U);
  ASSERT_EQ(trace.states.size(), 3U);
  EXPECT_NEAR(trace.total_cost, 0.0, kTolerance);
  EXPECT_NEAR(trace.states[0].q[0], 0.5, kTolerance);
  EXPECT_NEAR(trace.states[1].q[0], 0.6, kTolerance);
  EXPECT_NEAR(trace.states[2].q[0], 0.55, kTolerance);
  EXPECT_NEAR(trace.actions[0][0], 0.1, kTolerance);
  EXPECT_NEAR(trace.actions[1][0], -0.05, kTolerance);
  EXPECT_TRUE(trace.states[1].valid);
  EXPECT_TRUE(trace.states[1].tactile.valid);
}

TEST(MPPIOptimizerTest, InvalidRolloutStepGetsLargeCostAndStopsTrace) {
  mppi_core::MPPIConfig config;
  config.horizon_steps = 2;
  config.num_rollouts = 1;
  config.action_dim = 1;
  config.dt = 0.1;
  config.temperature = 1.0;
  config.action_lower_bound = Eigen::VectorXd::Constant(1, -1.0);
  config.action_upper_bound = Eigen::VectorXd::Constant(1, 1.0);
  config.action_noise_std = Eigen::VectorXd::Zero(1);

  mppi_core::DeltaQReferenceRolloutConfig prediction_config;
  prediction_config.tactile_rollout_policy =
      mppi_core::TactileRolloutPolicy::kForceAwareRequired;
  auto model = std::make_shared<mppi_core::DeltaQReferenceRolloutModel>(
      1, prediction_config);
  mppi_core::MPPIOptimizer optimizer;
  optimizer.Initialize(config, model, nullptr);

  mppi_core::TactileState tactile;
  tactile.valid = true;
  tactile.contact_presence = mppi_core::ContactPresence::kNoContact;
  tactile.has_normal_force = true;
  tactile.normal_force_n = 0.0;

  mppi_core::GraspObservation observation;
  observation.q_ref_current = Eigen::VectorXd::Zero(1);
  observation.v_ref_current = Eigen::VectorXd::Zero(1);
  observation.q_measured = observation.q_ref_current;
  observation.v_measured = observation.v_ref_current;
  observation.tau = Eigen::VectorXd::Zero(1);
  observation.tactile = tactile;

  mppi_core::ActionSequence actions(1, 2);
  actions.setAction(0, Eigen::VectorXd::Constant(1, 0.1));
  actions.setAction(1, Eigen::VectorXd::Constant(1, 0.1));

  const auto trace = optimizer.PredictRollout(observation, actions);

  ASSERT_EQ(trace.states.size(), 2U);
  ASSERT_EQ(trace.actions.size(), 1U);
  ASSERT_EQ(trace.step_costs.size(), 1U);
  EXPECT_FALSE(trace.states[1].valid);
  EXPECT_NEAR(trace.step_costs[0], 1.0e30, 0.0);
  EXPECT_NEAR(trace.total_cost, 1.0e30, 0.0);
}

TEST(MPPIOptimizerTest, AllInvalidRolloutsReturnHoldCommand) {
  mppi_core::MPPIConfig config;
  config.horizon_steps = 2;
  config.num_rollouts = 8;
  config.action_dim = 1;
  config.dt = 0.1;
  config.temperature = 1.0;
  config.random_seed = 7;
  config.action_lower_bound = Eigen::VectorXd::Constant(1, -1.0);
  config.action_upper_bound = Eigen::VectorXd::Constant(1, 1.0);
  config.action_noise_std = Eigen::VectorXd::Constant(1, 0.5);

  auto model = std::make_shared<mppi_core::DeltaQReferenceRolloutModel>(1);
  auto cost = std::make_shared<mppi_core::GraspStabilityCost>(
      mppi_core::GraspStabilityCostConfig{});
  mppi_core::MPPIOptimizer optimizer;
  optimizer.Initialize(config, model, cost);

  mppi_core::TactileState tactile;
  tactile.valid = true;
  tactile.contact_presence = mppi_core::ContactPresence::kNoContact;
  tactile.has_normal_force = true;
  tactile.normal_force_n = 0.0;

  mppi_core::GraspObservation observation;
  observation.q_ref_current = Eigen::VectorXd::Constant(1, 0.25);
  observation.v_ref_current = Eigen::VectorXd::Zero(1);
  observation.q_measured = observation.q_ref_current;
  observation.v_measured = observation.v_ref_current;
  observation.tau = Eigen::VectorXd::Zero(1);
  observation.tactile = tactile;
  observation.time_s = 12.34;

  const auto command = optimizer.Update(observation);

  ASSERT_EQ(command.delta_q_ref.size(), 1);
  ASSERT_EQ(command.q_des.size(), 1);
  ASSERT_EQ(command.qdot_des.size(), 1);
  ASSERT_EQ(command.qddot_des.size(), 1);
  ASSERT_EQ(command.tau_ff.size(), 1);
  ASSERT_EQ(command.kp.size(), 1);
  ASSERT_EQ(command.kd.size(), 1);
  EXPECT_TRUE(command.valid);
  EXPECT_TRUE(command.IsUsable());
  EXPECT_NEAR(command.delta_q_ref[0], 0.0, kTolerance);
  EXPECT_NEAR(command.q_des[0], 0.25, kTolerance);
  EXPECT_NEAR(command.qdot_des[0], 0.0, kTolerance);
  EXPECT_NEAR(command.qddot_des[0], 0.0, kTolerance);
  EXPECT_NEAR(command.tau_ff[0], 0.0, kTolerance);
  EXPECT_NEAR(command.kp[0], 0.0, kTolerance);
  EXPECT_NEAR(command.kd[0], 0.0, kTolerance);
  EXPECT_NEAR(command.stamp_sec, 12.34, kTolerance);
}

TEST(MPPIOptimizerTest, PredictRolloutAllowsPinocchioNqDifferentFromNv) {
  const auto sensor_model = MakeSingleUnboundedRevoluteZSensorModel();
  pinocchio::Data data(sensor_model.model);

  mppi_core::MPPIConfig config;
  config.horizon_steps = 1;
  config.num_rollouts = 1;
  config.action_dim = sensor_model.model.nv;
  config.dt = 0.1;
  config.temperature = 1.0;
  config.action_lower_bound =
      Eigen::VectorXd::Constant(sensor_model.model.nv, -1.0);
  config.action_upper_bound =
      Eigen::VectorXd::Constant(sensor_model.model.nv, 1.0);
  config.action_noise_std = Eigen::VectorXd::Zero(sensor_model.model.nv);

  auto model = std::make_shared<mppi_core::DeltaQReferenceRolloutModel>(
      sensor_model.model.nv);
  mppi_core::MPPIOptimizer optimizer;
  optimizer.Initialize(config, model, nullptr);

  mppi_core::TactileState tactile;
  tactile.valid = true;
  tactile.contact_presence = mppi_core::ContactPresence::kStableContact;
  tactile.has_normal_force = true;
  tactile.normal_force_n = 1.0;
  tactile.contact_support_count = 1;
  tactile.support_count = 1;
  tactile.confidence = 1.0;
  mppi_core::TactileContactPoint point;
  point.active = true;
  point.support_index = 0;
  point.position_sensor_m = Eigen::Vector3d{1.0, 0.0, 0.0};
  tactile.contact_points.push_back(point);

  mppi_core::ContactForceProjectionConfig projection_config;
  mppi_core::ContactForceRolloutConfig force_rollout_config;
  force_rollout_config.min_stable_support_count = 1;

  mppi_core::PinocchioContactKinematicsContext kinematics;
  kinematics.model = &sensor_model.model;
  kinematics.data = &data;
  kinematics.sensor_frame_id = sensor_model.sensor_frame_id;

  mppi_core::GraspObservation observation;
  observation.q_ref_current = pinocchio::neutral(sensor_model.model);
  observation.v_ref_current = Eigen::VectorXd::Zero(sensor_model.model.nv);
  observation.q_measured = observation.q_ref_current;
  observation.v_measured = observation.v_ref_current;
  observation.tau = Eigen::VectorXd::Zero(sensor_model.model.nv);
  observation.tactile = tactile;
  observation.contact_kinematics = &kinematics;
  observation.contact_force_projection_config = &projection_config;
  observation.contact_force_rollout_config = &force_rollout_config;

  mppi_core::ActionSequence actions(sensor_model.model.nv, 1);
  actions.setAction(0, Eigen::VectorXd::Constant(sensor_model.model.nv, 0.2));

  const auto trace = optimizer.PredictRollout(observation, actions);

  ASSERT_EQ(trace.states.size(), 2U);
  EXPECT_TRUE(trace.states[1].valid);
  EXPECT_EQ(trace.states[1].q.size(), sensor_model.model.nq);
  EXPECT_EQ(trace.states[1].dq.size(), sensor_model.model.nv);
  EXPECT_EQ(trace.states[1].tau.size(), sensor_model.model.nv);
  EXPECT_TRUE(trace.states[1].q.allFinite());
}
