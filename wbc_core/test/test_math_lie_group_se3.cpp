//
// Copyright (c) 2026
//
// Tests for SE(3) vector conversion helpers.
//

#include <gtest/gtest.h>

#include <Eigen/Geometry>

#include <wbc_core/math/lie_group/se3.hpp>

namespace
{

TEST(SE3ConversionTest, VectorRoundTripPreservesTransform) {
  const Eigen::Matrix3d rotation =
    (Eigen::AngleAxisd(0.4, Eigen::Vector3d::UnitZ()) *
    Eigen::AngleAxisd(-0.2, Eigen::Vector3d::UnitY()) *
    Eigen::AngleAxisd(0.1, Eigen::Vector3d::UnitX()))
    .toRotationMatrix();
  const Eigen::Vector3d translation(0.3, -0.2, 0.7);
  const pinocchio::SE3 input(rotation, translation);

  wbc::math::Vector vector(12);
  wbc::math::se3ToVector(input, vector);

  pinocchio::SE3 recovered;
  wbc::math::vectorToSE3(vector, recovered);

  EXPECT_TRUE(recovered.translation().isApprox(input.translation(), 1e-12));
  EXPECT_TRUE(recovered.rotation().isApprox(input.rotation(), 1e-12));
}

}  // namespace
