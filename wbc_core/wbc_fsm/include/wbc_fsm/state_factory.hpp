#pragma once

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>

#include <yaml-cpp/yaml.h>

#include "wbc_fsm/interface/state_machine.hpp"

namespace wbc {

struct StateBuildContext {
  PinocchioRobotSystem* robot{nullptr};
  TaskRegistry* task_registry{nullptr};
  ConstraintRegistry* constraint_registry{nullptr};
  StateProvider* state_provider{nullptr};
  YAML::Node params;
  void* user_data{nullptr};
};

using StateCreator = std::function<std::unique_ptr<StateMachine>(
    StateId id, const std::string& state_name, const StateBuildContext& context)>;

class StateFactory {
public:
  static StateFactory& Instance();

  // Register creator for a state key. Duplicate key registration is rejected.
  // The caller must ensure registration happens at static init or startup.
  bool Register(const std::string& key, StateCreator creator);
  bool Has(const std::string& key) const;

  // Create state instance using pre-registered creator.
  // Returns nullptr when the key is unknown or creator returns nullptr.
  std::unique_ptr<StateMachine> Create(const std::string& key, StateId id,
                                       const std::string& state_name,
                                       const StateBuildContext& context) const;

private:
  StateFactory() = default;

  std::unordered_map<std::string, StateCreator> creators_;
};

// Generic registration macro. CREATOR must be a lambda/functor compatible with
// StateCreator.
#define WBC_DETAIL_CONCAT_INNER(a, b) a##b
#define WBC_DETAIL_CONCAT(a, b) WBC_DETAIL_CONCAT_INNER(a, b)
#define WBC_REGISTER_STATE(KEY, CREATOR)                                     \
  namespace {                                                                 \
  const bool WBC_DETAIL_CONCAT(k_registered_state_, __COUNTER__) = []() {    \
    ::wbc::StateFactory::Instance().Register((KEY), (CREATOR));              \
    return true;                                                              \
  }();                                                                        \
  } // namespace

} // namespace wbc
