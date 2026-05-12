//
// Copyright (c) 2026
//
// StateFactory: pluggable factory for FSM state construction.
//

#ifndef CONTROL_ARCHITECTURE_STATE_MACHINE_STATE_FACTORY_HPP_
#define CONTROL_ARCHITECTURE_STATE_MACHINE_STATE_FACTORY_HPP_

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>

#include "control_architecture/state_machine/state_machine.hpp"

namespace wbc {

class StateFactory {
 public:
  using Creator = std::function<std::unique_ptr<State>(
      StateId, const std::string&, const StateContext&)>;

  void Register(const std::string& key, Creator creator);

  template <typename StateT>
  void Register(const std::string& key) {
    Register(key, [](StateId id, const std::string& name,
                     const StateContext& context) {
      return std::make_unique<StateT>(id, name, context);
    });
  }

  template <typename StateT>
  void Register() {
    Register<StateT>(StateT::kName);
  }

  bool Contains(const std::string& key) const;

  std::unique_ptr<State> Create(const std::string& key, StateId id,
                                const std::string& name,
                                const StateContext& context) const;

 private:
  std::unordered_map<std::string, Creator> creators_;
};

}  // namespace wbc

#endif  // CONTROL_ARCHITECTURE_STATE_MACHINE_STATE_FACTORY_HPP_
