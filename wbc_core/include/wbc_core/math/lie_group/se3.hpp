#pragma once

#include "wbc_core/math/fwd.hpp"

#include <pinocchio/spatial/explog.hpp>
#include <pinocchio/spatial/se3.hpp>

namespace wbc::math {

void se3ToXyzQuat(const pinocchio::SE3& transform, RefVector xyz_quat);
void se3ToVector(const pinocchio::SE3& transform, RefVector vector);
void vectorToSE3(ConstRefVector vector, pinocchio::SE3& transform);

/**
 * Computes the task-space pose error used by SE(3) motion tasks.
 *
 * The error is expressed in the current frame. Its linear part is the
 * translation of current.actInv(desired), and its angular part is
 * log3(current.rotation().transpose() * desired.rotation()). This is not the
 * coupled SE(3) logarithm; use pinocchio::log6 directly when that convention is
 * required.
 */
void poseError(const pinocchio::SE3& current,
               const pinocchio::SE3& desired,
               pinocchio::Motion& error);

}  // namespace wbc::math
