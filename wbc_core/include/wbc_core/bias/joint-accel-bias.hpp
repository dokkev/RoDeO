//
// Copyright (c) 2026
//
// Dedicated full-joint acceleration bias for WBMC.
//

#ifndef __wbc_bias_joint_accel_bias_hpp__
#define __wbc_bias_joint_accel_bias_hpp__

#include <string>

#include "wbc_core/math/fwd.hpp"

namespace tsid {
namespace bias {

struct JointAccelBias {
  std::string name{"joint-accel-bias"};
  const math::Vector* qddot_bias{nullptr};
  double weight{1.0};
  unsigned int level{2};
};

}  // namespace bias
}  // namespace tsid

#endif  // ifndef __wbc_bias_joint_accel_bias_hpp__
