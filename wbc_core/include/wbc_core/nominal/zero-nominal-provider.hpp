//
// Copyright (c) 2026
//
// Zero nominal acceleration fallback provider.
//

#ifndef __wbc_nominal_zero_nominal_provider_hpp__
#define __wbc_nominal_zero_nominal_provider_hpp__

#include "wbc_core/nominal/nominal-acceleration-provider.hpp"

namespace tsid {
namespace nominal {

class ZeroNominalProvider final : public NominalAccelerationProvider {
 public:
  bool compute(const NominalAccelerationContext& context,
               math::Vector& qddotNominal) override {
    qddotNominal.setZero(context.nv);
    return true;
  }
};

}  // namespace nominal
}  // namespace tsid

#endif  // ifndef __wbc_nominal_zero_nominal_provider_hpp__
