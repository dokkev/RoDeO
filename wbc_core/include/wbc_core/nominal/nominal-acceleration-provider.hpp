//
// Copyright (c) 2026
//
// External nominal acceleration interface for final-form IDHQP.
//

#ifndef __wbc_nominal_nominal_acceleration_provider_hpp__
#define __wbc_nominal_nominal_acceleration_provider_hpp__

#include "wbc_core/math/fwd.hpp"

namespace wbc {
namespace nominal {

struct NominalAccelerationContext {
  double time{0.0};
  const math::Vector* q{nullptr};
  const math::Vector* v{nullptr};
  int nv{0};
  int lambdaDim{0};
};

class NominalAccelerationProvider {
 public:
  virtual ~NominalAccelerationProvider() = default;

  virtual bool compute(const NominalAccelerationContext& context,
                       math::Vector& qddotNominal) = 0;
};

}  // namespace nominal
}  // namespace wbc

#endif  // ifndef __wbc_nominal_nominal_acceleration_provider_hpp__
