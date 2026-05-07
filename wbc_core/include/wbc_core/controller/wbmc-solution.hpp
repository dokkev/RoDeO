//
// Copyright (c) 2026
//
// Solution bundle for delta-form WBMC.
//

#ifndef __wbc_controller_wbmc_solution_hpp__
#define __wbc_controller_wbmc_solution_hpp__

#include "wbc_core/math/fwd.hpp"

namespace tsid {

struct WBMCSolution {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  math::Vector qddot_ref;     ///< Reference acceleration center.
  math::Vector delta_qddot;   ///< HQP correction around the reference acceleration.
  math::Vector qddot_sol;     ///< Solved acceleration = qddot_ref + delta_qddot.
  math::Vector lambda;        ///< Contact force decision variable.
  math::Vector tau;           ///< Actuator torques.
  bool success{false};
};

}  // namespace tsid

#endif  // ifndef __wbc_controller_wbmc_solution_hpp__
