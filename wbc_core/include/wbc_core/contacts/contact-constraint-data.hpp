//
// Copyright (c) 2026
//
// Per-tick contact constraint snapshot consumed by inverse dynamics.
//

#ifndef WBC_CORE_CONTACTS_CONTACT_CONSTRAINT_DATA_HPP_
#define WBC_CORE_CONTACTS_CONTACT_CONSTRAINT_DATA_HPP_

#include <string>

#include "wbc_core/math/fwd.hpp"

namespace wbc {

struct ContactConstraintData {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  std::string name;
  math::Matrix Jc;
  /// RHS for the hard contact motion constraint: Jc * qddot = motion_rhs.
  math::Vector motion_rhs;
  math::Matrix T;
  math::Matrix Uf;
  math::Vector uf_lb;
  math::Vector uf_ub;

  int lambdaDim() const { return static_cast<int>(T.cols()); }
  int motionDim() const { return static_cast<int>(Jc.rows()); }
};

}  // namespace wbc

#endif  // WBC_CORE_CONTACTS_CONTACT_CONSTRAINT_DATA_HPP_
