/**
 * @file wbc_core/wbc_util/src/constraint_registry.cpp
 * @brief Doxygen documentation for constraint_registry module.
 */
#include "wbc_util/constraint_registry.hpp"

#include "wbc_formulation/constraint.hpp"
#include "wbc_formulation/contact.hpp"

namespace wbc {

void ConstraintRegistry::AddConstraint(const std::string& name,
                                       std::unique_ptr<Constraint> constraint) {
  if (name.empty() || constraint == nullptr) {
    return;
  }
  constraint_map_[name] = std::move(constraint);
}

Constraint* ConstraintRegistry::GetConstraint(const std::string& name) const {
  const auto it = constraint_map_.find(name);
  if (it == constraint_map_.end()) {
    return nullptr;
  }
  return it->second.get();
}

Contact* ConstraintRegistry::GetContact(const std::string& name) const {
  Constraint* constraint = GetConstraint(name);
  if (constraint == nullptr) {
    return nullptr;
  }
  return dynamic_cast<Contact*>(constraint);
}

void ConstraintRegistry::Clear() { constraint_map_.clear(); }

} // namespace wbc
