//
// Copyright (c) 2026
//
// Abstract base for reusable HQP formulation blocks.
//
// A block owns a ConstraintBase object and updates its matrices each cycle
// via build(). Controllers compose blocks into HQPData at desired levels.
//

#ifndef __wbc_formulations_hqp_block_base_hpp__
#define __wbc_formulations_hqp_block_base_hpp__

#include <memory>
#include <string>
#include <utility>

#include "wbc_core/formulations/hqp/hqp-block-context.hpp"
#include "wbc_core/math/fwd.hpp"

namespace wbc {

class HQPBlock {
 public:
  virtual ~HQPBlock() = default;

  HQPBlock(const HQPBlock&) = delete;
  HQPBlock& operator=(const HQPBlock&) = delete;
  HQPBlock(HQPBlock&&) = default;
  HQPBlock& operator=(HQPBlock&&) = default;

  const std::string& name() const { return m_name; }
  unsigned int level() const { return m_level; }
  double weight() const { return m_weight; }
  void setWeight(double w) { m_weight = w; }

  const std::shared_ptr<math::ConstraintBase>& constraint() const {
    return m_constraint;
  }
  std::shared_ptr<math::ConstraintBase>& constraint() {
    return m_constraint;
  }

  /// Updates the owned constraint object in-place from the current context.
  virtual void build(const HQPBlockContext& ctx) = 0;

 protected:
  HQPBlock(std::string name, unsigned int level, double weight)
      : m_name(std::move(name)), m_level(level), m_weight(weight) {}

  std::string m_name;
  unsigned int m_level{0};
  double m_weight{1.0};
  std::shared_ptr<math::ConstraintBase> m_constraint;
};

}  // namespace wbc

#endif  // ifndef __wbc_formulations_hqp_block_base_hpp__
