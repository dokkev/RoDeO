//
// Copyright (c) 2026
//

#include "control_architecture/state_machine/state_factory.hpp"

#include <stdexcept>
#include <utility>

namespace wbc {

void StateFactory::Register(const std::string& key, Creator creator) {
  if (key.empty()) {
    throw std::invalid_argument("StateFactory::Register: key is empty");
  }
  if (!creator) {
    throw std::invalid_argument("StateFactory::Register: creator for '" + key +
                                "' is empty");
  }
  if (!creators_.emplace(key, std::move(creator)).second) {
    throw std::invalid_argument("StateFactory::Register: duplicate key '" +
                                key + "'");
  }
}

bool StateFactory::Contains(const std::string& key) const {
  return creators_.find(key) != creators_.end();
}

std::unique_ptr<State> StateFactory::Create(
    const std::string& key, StateId id, const std::string& name,
    const StateContext& context) const {
  const auto it = creators_.find(key);
  if (it == creators_.end()) {
    throw std::invalid_argument("StateFactory::Create: unknown key '" + key +
                                "'");
  }
  return it->second(id, name, context);
}

}  // namespace wbc
