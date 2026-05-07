//
// Copyright (c) 2026
//

#include "wbc_core/runtime/config_validator.hpp"

#include <stdexcept>
#include <string>
#include <unordered_set>

namespace wbc {

namespace {

template <typename SetT>
void throwOnDuplicateInsert(const std::string& kind,
                            const std::string& key,
                            SetT& set) {
  if (!set.insert(key).second) {
    throw std::invalid_argument(
        "ConfigValidator::Validate: duplicate " + kind + " '" + key + "'");
  }
}

}  // namespace

void ConfigValidator::Validate(const CompiledConfig& compiled_config) {
  std::unordered_set<std::string> task_names;
  task_names.reserve(compiled_config.task_pool.size());
  for (const auto& task : compiled_config.task_pool) {
    if (task.name.empty()) {
      throw std::invalid_argument(
          "ConfigValidator::Validate: task name must not be empty");
    }
    throwOnDuplicateInsert("task name", task.name, task_names);
  }

  std::unordered_set<std::string> contact_names;
  contact_names.reserve(compiled_config.contact_pool.size());
  for (const auto& contact : compiled_config.contact_pool) {
    if (contact.name.empty()) {
      throw std::invalid_argument(
          "ConfigValidator::Validate: contact name must not be empty");
    }
    throwOnDuplicateInsert("contact name", contact.name, contact_names);
  }

  std::unordered_set<StateId> state_ids;
  state_ids.reserve(compiled_config.states.size());
  for (const auto& state : compiled_config.states) {
    if (state.id < 0) {
      throw std::invalid_argument(
          "ConfigValidator::Validate: state id must be non-negative");
    }
    if (!state_ids.insert(state.id).second) {
      throw std::invalid_argument(
          "ConfigValidator::Validate: duplicate state id '" +
          std::to_string(state.id) + "'");
    }
    if (state.name.empty()) {
      throw std::invalid_argument(
          "ConfigValidator::Validate: state name must not be empty");
    }
    if (state.type.empty()) {
      throw std::invalid_argument(
          "ConfigValidator::Validate: state '" + state.name +
          "' has empty type");
    }

    std::unordered_set<std::string> state_task_names;
    state_task_names.reserve(state.tasks.size());
    for (const auto& task_sel : state.tasks) {
      if (!task_names.count(task_sel.name)) {
        throw std::invalid_argument(
            "ConfigValidator::Validate: state '" + state.name +
            "' references unknown task '" + task_sel.name + "'");
      }
      throwOnDuplicateInsert("task selection in state '" + state.name + "'",
                             task_sel.name, state_task_names);
    }

    std::unordered_set<std::string> state_contact_names;
    state_contact_names.reserve(state.contacts.size());
    for (const auto& contact_sel : state.contacts) {
      if (!contact_names.count(contact_sel.name)) {
        throw std::invalid_argument(
            "ConfigValidator::Validate: state '" + state.name +
            "' references unknown contact '" + contact_sel.name + "'");
      }
      throwOnDuplicateInsert(
          "contact selection in state '" + state.name + "'",
          contact_sel.name, state_contact_names);
    }
  }
}

}  // namespace wbc
