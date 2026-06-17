#include "wbc_core/math/lie_group/se3.hpp"

#include <pinocchio/macros.hpp>

namespace wbc::math {

void se3ToXyzQuat(const pinocchio::SE3& transform, RefVector xyz_quat) {
  PINOCCHIO_CHECK_INPUT_ARGUMENT(
      xyz_quat.size() == 7, "The size of the xyz_quat vector needs to equal 7");
  xyz_quat.head<3>() = transform.translation();
  xyz_quat.tail<4>() = Eigen::Quaterniond(transform.rotation()).coeffs();
}

void se3ToVector(const pinocchio::SE3& transform, RefVector vector) {
  PINOCCHIO_CHECK_INPUT_ARGUMENT(
      vector.size() == 12, "The size of the vector needs to equal 12");
  vector.head<3>() = transform.translation();
  typedef Eigen::Matrix<double, 9, 1> Vector9;
  vector.tail<9>() = Eigen::Map<const Vector9>(&transform.rotation()(0), 9);
}

void vectorToSE3(ConstRefVector vector, pinocchio::SE3& transform) {
  PINOCCHIO_CHECK_INPUT_ARGUMENT(vector.size() == 12,
                                 "vector needs to contain 12 rows");
  transform.translation(vector.head<3>());
  typedef Eigen::Matrix<double, 3, 3> Matrix3;
  transform.rotation(Eigen::Map<const Matrix3>(vector.data() + 3, 3, 3));
}

void poseError(const pinocchio::SE3& current,
               const pinocchio::SE3& desired,
               pinocchio::Motion& error) {
  const pinocchio::SE3 transform_error = current.actInv(desired);
  error.linear() = transform_error.translation();
  error.angular() = pinocchio::log3(transform_error.rotation());
}

}  // namespace wbc::math
