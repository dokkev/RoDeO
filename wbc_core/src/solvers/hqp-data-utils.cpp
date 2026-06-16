#include "wbc_core/solvers/hqp-data-utils.hpp"

#include "wbc_core/math/constraints/constraint-base.hpp"

namespace wbc::solvers::hqp {

void ensureLevel(HQPData& hqp_data, unsigned int level) {
  if (hqp_data.size() <= level) {
    hqp_data.resize(level + 1);
  }
}

void resizeData(HQPData& hqp_data, unsigned int num_levels) {
  hqp_data.clear();
  hqp_data.resize(num_levels);
}

void addTerm(ConstraintLevel& level, double weight,
             const std::shared_ptr<math::ConstraintBase>& constraint) {
  level.emplace_back(weight, constraint);
}

void addTerm(HQPData& hqp_data, unsigned int level, double weight,
             const std::shared_ptr<math::ConstraintBase>& constraint) {
  ensureLevel(hqp_data, level);
  addTerm(hqp_data[level], weight, constraint);
}

Dimensions dimensions(const HQPData& hqp_data) {
  Dimensions out;

  for (const auto& level : hqp_data) {
    for (const auto& pair : level) {
      if (!pair.second) {
        continue;
      }
      if (out.variables == 0) {
        out.variables = pair.second->cols();
      }
      if (pair.second->isEquality()) {
        out.equalities += pair.second->rows();
      } else {
        out.inequalities += pair.second->rows();
      }
    }
  }

  return out;
}

}  // namespace wbc::solvers::hqp
