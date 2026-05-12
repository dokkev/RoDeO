//
// Copyright (c) 2026
//
// Solution bundle for delta-form IDHQP.
//

#ifndef __wbc_controller_id_solution_hpp__
#define __wbc_controller_id_solution_hpp__

#include "wbc_core/math/fwd.hpp"

namespace wbc {

struct IDSolution {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  math::Vector qddot_ref;    ///< Reference acceleration center.
  math::Vector delta_qddot;  ///< HQP correction around the reference acceleration.
  math::Vector qddot_sol;    ///< Solved generalized acceleration.
  math::Vector qdot_cmd;     ///< Integrated generalized velocity command.
  math::Vector q_cmd;        ///< Integrated generalized configuration command.
  math::Vector lambda_sol;   ///< Solved contact reaction forces.
  math::Vector tau_cmd;      ///< Solved actuator torque command.
  bool success{false};
};

}  // namespace wbc

#endif  // ifndef __wbc_controller_id_solution_hpp__
