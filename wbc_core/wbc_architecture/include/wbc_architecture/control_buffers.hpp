/**
 * @file wbc_core/wbc_architecture/include/wbc_architecture/control_buffers.hpp
 * @brief Internal pre-allocated buffers for control loop runtime.
 */
#pragma once

#include <Eigen/Dense>

namespace wbc {

/**
 * @brief Pre-allocated scratch buffers used in ControlArchitecture hot path.
 *
 * @details
 * These vectors are resized once in initialization and reused every tick to
 * avoid dynamic allocations in the real-time loop.
 *
 * Task-reference scratch buffers (joint_ref_zeros, com_ref_*) have moved into
 * TaskReference, which owns its own pre-allocated fields.
 */
struct ControlBuffers {
  Eigen::VectorXd wbc_qddot_cmd;   ///< solver output (size = num_qdot)
  Eigen::VectorXd joint_trq_prev;  ///< hold-torque on failure (size = num_active)
  Eigen::VectorXd zero_qdot;       ///< pre-allocated zeros (size = num_qdot), used for disabled compensation
};

} // namespace wbc
