//
// Copyright (c) 2026
//
// Legacy staged-IK kinematic reference.
//

#ifndef __wbc_formulations_ik_kinematic_reference_hpp__
#define __wbc_formulations_ik_kinematic_reference_hpp__

#include "wbc_core/math/fwd.hpp"

namespace wbc {

struct IKKinematicReference {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  math::Vector jposRef;
  math::Vector jvelRef;
};

}  // namespace wbc

#endif  // ifndef __wbc_formulations_ik_kinematic_reference_hpp__
