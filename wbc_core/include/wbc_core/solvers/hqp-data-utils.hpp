//
// Copyright (c) 2026
//
// Shared helpers for assembling and inspecting solver HQPData.
//

#ifndef WBC_CORE_SOLVERS_HQP_DATA_UTILS_HPP_
#define WBC_CORE_SOLVERS_HQP_DATA_UTILS_HPP_

#include <memory>

#include "wbc_core/math/fwd.hpp"
#include "wbc_core/solvers/fwd.hpp"

namespace wbc::solvers::hqp {

constexpr unsigned int kLevel0 = 0u;

struct Dimensions {
  unsigned int variables{0};
  unsigned int equalities{0};
  unsigned int inequalities{0};
};

void ensureLevel(HQPData& hqp_data, unsigned int level);
void resizeData(HQPData& hqp_data, unsigned int num_levels);
void addTerm(ConstraintLevel& level, double weight,
             const std::shared_ptr<math::ConstraintBase>& constraint);
void addTerm(HQPData& hqp_data, unsigned int level, double weight,
             const std::shared_ptr<math::ConstraintBase>& constraint);
Dimensions dimensions(const HQPData& hqp_data);

}  // namespace wbc::solvers::hqp

#endif  // WBC_CORE_SOLVERS_HQP_DATA_UTILS_HPP_
