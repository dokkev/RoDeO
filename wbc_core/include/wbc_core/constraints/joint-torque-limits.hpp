//
// Copyright (c) 2026
//
// Problem-facing joint torque limit primitive.
//

#ifndef WBC_CORE_CONSTRAINTS_JOINT_TORQUE_LIMITS_HPP_
#define WBC_CORE_CONSTRAINTS_JOINT_TORQUE_LIMITS_HPP_

#include "wbc_core/math/fwd.hpp"

namespace wbc {
namespace constraints {

struct JointTorqueLimits {
  const math::Vector* lower{nullptr};
  const math::Vector* upper{nullptr};

  bool enabled() const { return lower != nullptr && upper != nullptr; }
};

}  // namespace constraints
}  // namespace wbc

#endif  // WBC_CORE_CONSTRAINTS_JOINT_TORQUE_LIMITS_HPP_
