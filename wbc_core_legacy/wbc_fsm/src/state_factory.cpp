/**
 * @file wbc_core/wbc_fsm/src/state_factory.cpp
 * @brief Doxygen documentation for state_factory module.
 */
#include "wbc_fsm/state_factory.hpp"

#include <iostream>

namespace wbc {

StateFactory& StateFactory::Instance() {
  static StateFactory factory;
  return factory;
}

bool StateFactory::Register(const std::string& key, StateCreator creator) {
  if (key.empty() || !creator) {
    std::cerr << "[StateFactory] Reject invalid registration." << std::endl;
    return false;
  }
  const auto [it, inserted] = creators_.try_emplace(key, std::move(creator));
  (void)it;
  if (!inserted) {
    std::cerr << "[StateFactory] Reject duplicate registration for key '"
              << key << "'." << std::endl;
    return false;
  }
  return true;
}

bool StateFactory::Has(const std::string& key) const {
  return creators_.find(key) != creators_.end();
}

std::unique_ptr<StateMachine> StateFactory::Create(
    const std::string& key, StateId id, const std::string& state_name,
    const StateMachineConfig& context) const {
  const auto it = creators_.find(key);
  if (it == creators_.end()) {
    return nullptr;
  }
  return it->second(id, state_name, context);
}

} // namespace wbc
