//
// Copyright (c) 2026
//

#include "wbc_core/runtime/config_compiler.hpp"

#include <stdexcept>

namespace wbc {

namespace {

TaskTypeSpec parseTaskTypeSpec(const std::string& type,
                               const std::string& task_name) {
  if (type == "JointTask") {
    return TaskTypeSpec::kJointTask;
  }
  if (type == "LinkPosTask") {
    return TaskTypeSpec::kLinkPosTask;
  }
  if (type == "LinkOriTask") {
    return TaskTypeSpec::kLinkOriTask;
  }
  if (type == "ComTask") {
    return TaskTypeSpec::kComTask;
  }
  if (type == "ForceTask") {
    return TaskTypeSpec::kForceTask;
  }
  throw std::invalid_argument(
      "ConfigCompiler::parseTaskPool: task '" + task_name +
      "' uses unknown type '" + type + "'");
}

unsigned int parseTaskLevel(const YAML::Node& item,
                            const std::string& task_name) {
  if (item["priority"]) {
    return item["priority"].as<unsigned int>();
  }
  if (!item["role"]) {
    return 1u;
  }

  const auto role = item["role"].as<std::string>();
  if (role == "operational_task") {
    return 1u;
  }
  if (role == "bias_task" || role == "posture_task") {
    return 2u;
  }
  throw std::invalid_argument(
      "ConfigCompiler::parseTaskPool: task '" + task_name +
      "' uses unknown role '" + role + "'");
}

ContactTypeSpec parseContactTypeSpec(const std::string& type,
                                     const std::string& contact_name) {
  if (type == "SurfaceContact") {
    return ContactTypeSpec::kSurfaceContact;
  }
  if (type == "PointContact") {
    return ContactTypeSpec::kPointContact;
  }
  throw std::invalid_argument(
      "ConfigCompiler::parseContactPool: contact '" + contact_name +
      "' uses unknown type '" + type + "'");
}

GlobalConstraintTypeSpec parseConstraintTypeSpec(const std::string& type) {
  if (type == "JointTrqLimitConstraint") {
    return GlobalConstraintTypeSpec::kJointTrqLimitConstraint;
  }
  throw std::invalid_argument(
      "ConfigCompiler::parseGlobalConstraints: unknown constraint type '" +
      type + "'");
}

ScalarOrVectorSpec parseScalarOrVectorSpec(const YAML::Node& node) {
  ScalarOrVectorSpec out;
  if (!node) {
    return out;
  }
  out.has_value = true;
  if (node.IsSequence()) {
    out.is_vector = true;
    out.values.reserve(node.size());
    for (std::size_t i = 0; i < node.size(); ++i) {
      out.values.push_back(node[i].as<double>());
    }
  } else {
    out.scalar = node.as<double>();
  }
  return out;
}

}  // namespace

CompiledConfig ConfigCompiler::Compile(const YAML::Node& root) {
  CompiledConfig compiled_config;

  if (root["contact_pool"]) {
    parseContactPool(root["contact_pool"], compiled_config);
  }
  if (root["global_constraints"]) {
    parseGlobalConstraints(root["global_constraints"], compiled_config);
  }
  if (root["task_pool"]) {
    parseTaskPool(root["task_pool"], compiled_config);
  }
  if (root["regularization"]) {
    parseRegularization(root["regularization"], compiled_config);
  }
  if (root["controller"]) {
    parseController(root["controller"], compiled_config);
  }
  if (root["state_machine"]) {
    parseStateMachine(root["state_machine"], compiled_config);
  }

  return compiled_config;
}

void ConfigCompiler::parseTaskPool(const YAML::Node& node,
                                   CompiledConfig& compiled_config) {
  for (const auto& item : node) {
    if (!item["name"] || !item["type"]) {
      throw std::invalid_argument(
          "ConfigCompiler::parseTaskPool: task requires name and type");
    }

    TaskSpec task;
    task.name = item["name"].as<std::string>();
    task.type = parseTaskTypeSpec(item["type"].as<std::string>(), task.name);
    task.level = parseTaskLevel(item, task.name);
    if (item["weight"]) {
      task.weight = item["weight"].as<double>();
    }
    if (item["kp_ik"]) {
      task.kp_ik = item["kp_ik"].as<double>();
    }
    if (item["target_frame"]) {
      task.target_frame = item["target_frame"].as<std::string>();
    }
    task.kp = parseScalarOrVectorSpec(item["kp"]);
    task.kd = parseScalarOrVectorSpec(item["kd"]);
    compiled_config.task_pool.push_back(std::move(task));
  }
}

void ConfigCompiler::parseContactPool(const YAML::Node& node,
                                      CompiledConfig& compiled_config) {
  for (const auto& item : node) {
    if (!item["name"] || !item["type"] || !item["target_frame"]) {
      throw std::invalid_argument(
          "ConfigCompiler::parseContactPool: contact requires name, type, "
          "and target_frame");
    }

    ContactSpec contact;
    contact.name = item["name"].as<std::string>();
    contact.type = parseContactTypeSpec(item["type"].as<std::string>(),
                                        contact.name);
    contact.target_frame = item["target_frame"].as<std::string>();
    if (item["mu"]) {
      contact.mu = item["mu"].as<double>();
    }
    if (item["force_reg_weight"]) {
      contact.force_reg_weight = item["force_reg_weight"].as<double>();
    }
    if (item["fMin"]) {
      contact.fMin = item["fMin"].as<double>();
    }
    if (item["fMax"]) {
      contact.fMax = item["fMax"].as<double>();
    }
    if (item["kp_contact"]) {
      contact.kp_contact = item["kp_contact"].as<double>();
    }
    if (item["sole_thickness"]) {
      contact.sole_thickness = item["sole_thickness"].as<double>();
    }
    if (item["foot_half_length"]) {
      contact.has_foot_half_length = true;
      contact.foot_half_length = item["foot_half_length"].as<double>();
    }
    if (item["foot_half_width"]) {
      contact.has_foot_half_width = true;
      contact.foot_half_width = item["foot_half_width"].as<double>();
    }

    compiled_config.contact_pool.push_back(std::move(contact));
  }
}

void ConfigCompiler::parseStateMachine(const YAML::Node& node,
                                       CompiledConfig& compiled_config) {
  for (const auto& stateNode : node) {
    // State nodes may select active tasks/contacts, but must not redefine
    // solver hierarchy semantics owned by WBMC policy.
    if (stateNode["solver_hierarchy"] || stateNode["hierarchy_policy"] ||
        stateNode["hqp_hierarchy"] || stateNode["physics_level"] ||
        stateNode["operational_level"] || stateNode["bias_level"] ||
        stateNode["regularization_level"]) {
      throw std::invalid_argument(
          "ConfigCompiler::parseStateMachine: state-level solver "
          "hierarchy override is not allowed");
    }
    if (stateNode["task_priorities"]) {
      throw std::invalid_argument(
          "ConfigCompiler::parseStateMachine: 'task_priorities' is "
          "removed. Use per-task 'weight' entries in tasks/active_tasks.");
    }
    if (stateNode["task_hierarchy"]) {
      throw std::invalid_argument(
          "ConfigCompiler::parseStateMachine: 'task_hierarchy' is "
          "removed. Use tasks/active_tasks/task_selection.");
    }
    if (stateNode["contact_constraints"]) {
      throw std::invalid_argument(
          "ConfigCompiler::parseStateMachine: 'contact_constraints' is "
          "removed. Use 'contacts'.");
    }

    StateSpec state;
    state.id = stateNode["id"].as<StateId>();
    state.name = stateNode["name"].as<std::string>();
    state.type =
        stateNode["type"] ? stateNode["type"].as<std::string>() : state.name;

    if (stateNode["params"]) {
      state.params_yaml = YAML::Dump(stateNode["params"]);
    }

    // Preferred keys: tasks / active_tasks / task_selection.
    const YAML::Node taskSetNode =
        stateNode["tasks"] ? stateNode["tasks"]
                           : (stateNode["active_tasks"]
                                  ? stateNode["active_tasks"]
                                  : stateNode["task_selection"]);
    if (taskSetNode) {
      for (const auto& taskEntry : taskSetNode) {
        StateTaskSelection selection;
        if (taskEntry.IsScalar()) {
          selection.name = taskEntry.as<std::string>();
        } else {
          selection.name = taskEntry["name"].as<std::string>();
          if (taskEntry["weight"]) {
            selection.weight = taskEntry["weight"].as<double>();
          }
        }
        state.tasks.push_back(std::move(selection));
      }
    }

    // Preferred key: contacts.
    const YAML::Node contactSetNode = stateNode["contacts"];
    if (contactSetNode) {
      for (const auto& contactEntry : contactSetNode) {
        StateContactSelection selection;
        if (contactEntry.IsScalar()) {
          selection.name = contactEntry.as<std::string>();
        } else {
          selection.name = contactEntry["name"].as<std::string>();
        }
        state.contacts.push_back(std::move(selection));
      }
    }

    compiled_config.states.push_back(std::move(state));
  }
}

void ConfigCompiler::parseRegularization(const YAML::Node& node,
                                         CompiledConfig& compiled_config) {
  if (node["w_delta_qddot"]) {
    compiled_config.regularization.w_delta_qddot =
        node["w_delta_qddot"].as<double>();
  } else if (node["w_qddot"]) {
    compiled_config.regularization.w_delta_qddot =
        node["w_qddot"].as<double>();
  }
  if (node["w_xc_ddot"]) {
    compiled_config.regularization.w_xc_ddot = node["w_xc_ddot"].as<double>();
  }
}

void ConfigCompiler::parseController(const YAML::Node& node,
                                     CompiledConfig& compiled_config) {
  if (node["kp_acc"]) {
    compiled_config.controller.kp_acc = node["kp_acc"].as<double>();
  }
  if (node["kd_acc"]) {
    compiled_config.controller.kd_acc = node["kd_acc"].as<double>();
  }
  if (node["dt"]) {
    compiled_config.controller.dt = node["dt"].as<double>();
  }
}

void ConfigCompiler::parseGlobalConstraints(const YAML::Node& node,
                                            CompiledConfig& compiled_config) {
  for (auto it = node.begin(); it != node.end(); ++it) {
    const std::string type = it->first.as<std::string>();
    // Joint position/velocity limits are handled downstream via clamp.
    if (type == "JointPosLimitConstraint" ||
        type == "JointVelLimitConstraint") {
      continue;
    }
    GlobalConstraintSpec constraint;
    constraint.type = parseConstraintTypeSpec(type);
    const auto& val = it->second;
    if (val["enabled"]) {
      constraint.enabled = val["enabled"].as<bool>();
    }
    if (val["scale"]) {
      constraint.scale = val["scale"].as<double>();
    }
    if (val["is_soft"]) {
      constraint.is_soft = val["is_soft"].as<bool>();
    }
    if (val["soft_weight"]) {
      constraint.soft_weight = val["soft_weight"].as<double>();
    }
    compiled_config.global_constraints.push_back(std::move(constraint));
  }
}

}  // namespace wbc
