/**
 * @file wbc_core/wbc_fsm/include/wbc_fsm/state_factory.hpp
 * @brief Doxygen documentation for state_factory module.
 */
#pragma once

#include <cstdlib>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>

#include "wbc_fsm/interface/state_machine.hpp"

namespace wbc {

/**
 * @brief Creator callback signature for a concrete state implementation.
 */
using StateCreator = std::function<std::unique_ptr<StateMachine>(
    StateId id, const std::string& state_name, const StateMachineConfig& context)>;

/**
 * @brief Global registry for mapping YAML state keys to constructors.
 */
class StateFactory {
public:
  /**
   * @brief Singleton accessor.
   */
  static StateFactory& Instance();

  /**
   * @brief Register creator for a state key.
   * @return false when key/creator is invalid or key already exists.
   */
  bool Register(const std::string& key, StateCreator creator);
  /**
   * @brief Check whether a key has a registered creator.
   */
  bool Has(const std::string& key) const;

  /**
   * @brief Create a state instance from a pre-registered key.
   * @return New state instance, or nullptr on unknown key/failed creator.
   */
  std::unique_ptr<StateMachine> Create(const std::string& key, StateId id,
                                       const std::string& state_name,
                                       const StateMachineConfig& context) const;

private:
  StateFactory() = default;

  std::unordered_map<std::string, StateCreator> creators_;
};

// Generic registration macro. CREATOR must be a lambda/functor compatible with
// StateCreator.
//
// @note Requires shared-library (.so/.dylib) linkage. Static archives suppress
//       anonymous-namespace symbols and will silently skip registration.
//
// Aborts on registration failure (empty key, null creator, or duplicate key)
// since any such failure is always a programming error detectable at startup.
#define WBC_DETAIL_CONCAT_INNER(a, b) a##b
#define WBC_DETAIL_CONCAT(a, b) WBC_DETAIL_CONCAT_INNER(a, b)
#define WBC_REGISTER_STATE(KEY, CREATOR)                                      \
  namespace {                                                                  \
  const bool WBC_DETAIL_CONCAT(k_registered_state_, __COUNTER__) = []() {    \
    if (!::wbc::StateFactory::Instance().Register((KEY), (CREATOR))) {        \
      std::abort();                                                            \
    }                                                                          \
    return true;                                                               \
  }();                                                                         \
  } // namespace

} // namespace wbc
