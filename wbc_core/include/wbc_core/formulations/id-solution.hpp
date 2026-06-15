//
// Copyright (c) 2026
//
// Solution bundle for delta-form IDHQP.
//

#ifndef WBC_CORE_FORMULATIONS_ID_SOLUTION_HPP_
#define WBC_CORE_FORMULATIONS_ID_SOLUTION_HPP_

#include "wbc_core/math/fwd.hpp"

namespace wbc {

struct IDSolution {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  math::Vector qddot_ref;        ///< Reference acceleration center.
  math::Vector delta_qddot_sol;  ///< Solved HQP acceleration correction.
  math::Vector qddot_sol;        ///< qddot_ref + delta_qddot_sol.
  math::Vector lambda_sol;       ///< Solved contact reaction forces.
  math::Vector tau_sol;          ///< Solved model torque from ID recovery.
  bool success{false};
};

}  // namespace wbc

#endif  // WBC_CORE_FORMULATIONS_ID_SOLUTION_HPP_
