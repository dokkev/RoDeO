//
// Copyright (c) 2026
//

#include "control_architecture/runtime/config_compiler.hpp"

#include <cmath>
#include <stdexcept>

#include "control_architecture/runtime/yaml_parser.hpp"

namespace wbc {

namespace {

TaskTypeSpec parseTaskTypeSpec(const std::string& type,
                               const std::string& task_name) {
  if (type == "JointTask") {
    return TaskTypeSpec::kJointTask;
  }
  if (type == "SE3Task") {
    return TaskTypeSpec::kSE3Task;
  }
  if (type == "ComTask") {
    return TaskTypeSpec::kComTask;
  }
  if (type == "ForceTask") {
    return TaskTypeSpec::kForceTask;
  }
  throw std::invalid_argument("ConfigCompiler::parseTaskPool: task '" +
                              task_name + "' uses unknown type '" + type + "'");
}

unsigned int parseTaskLevel(const YAML::Node& item,
                            const std::string& task_name) {
  if (item["level"]) {
    return item["level"].as<unsigned int>();
  }
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
  if (role == "bias_task") {
    return 2u;
  }
  throw std::invalid_argument("ConfigCompiler::parseTaskPool: task '" +
                              task_name + "' uses unknown role '" + role + "'");
}

ContactTypeSpec parseContactTypeSpec(const std::string& type,
                                     const std::string& contact_name) {
  if (type == "SurfaceContact") {
    return ContactTypeSpec::kSurfaceContact;
  }
  if (type == "PointContact") {
    return ContactTypeSpec::kPointContact;
  }
  throw std::invalid_argument("ConfigCompiler::parseContactPool: contact '" +
                              contact_name + "' uses unknown type '" + type +
                              "'");
}

ConstraintTypeSpec parseConstraintTypeSpec(const std::string& type) {
  if (type == "JointTorque") {
    return ConstraintTypeSpec::kJointTorque;
  }
  throw std::invalid_argument(
      "ConfigCompiler::parseConstraints: unknown constraint type '" + type +
      "'");
}

solvers::SolverHQP parseSolverTypeSpec(const std::string& type) {
  if (type == "SOLVER_HQP_EIQUADPROG") {
    return solvers::SOLVER_HQP_EIQUADPROG;
  }
  if (type == "SOLVER_HQP_EIQUADPROG_FAST") {
    return solvers::SOLVER_HQP_EIQUADPROG_FAST;
  }
  if (type == "SOLVER_HQP_EIQUADPROG_RT") {
    return solvers::SOLVER_HQP_EIQUADPROG_RT;
  }
#ifdef TSID_QPMAD_FOUND
  if (type == "SOLVER_HQP_QPMAD") {
    return solvers::SOLVER_HQP_QPMAD;
  }
#endif
#ifdef TSID_WITH_PROXSUITE
  if (type == "SOLVER_HQP_PROXQP") {
    return solvers::SOLVER_HQP_PROXQP;
  }
#endif
#ifdef TSID_WITH_OSQP
  if (type == "SOLVER_HQP_OSQP") {
    return solvers::SOLVER_HQP_OSQP;
  }
#endif
#ifdef QPOASES_FOUND
  if (type == "SOLVER_HQP_OASES") {
    return solvers::SOLVER_HQP_OASES;
  }
#endif
  throw std::invalid_argument(
      "ConfigCompiler::parseSolver: unknown or unavailable SolverHQP '" + type +
      "'");
}

void parseSolverQPParams(const YAML::Node& node,
                         solvers::SolverQPParams& params) {
  if (!node) {
    return;
  }
  if (!node.IsMap()) {
    throw std::invalid_argument(
        "ConfigCompiler::parseSolver: qp_params must be a map");
  }

  if (node["max_iter"]) {
    params.max_iter = node["max_iter"].as<unsigned int>();
  }
  if (node["max_time"]) {
    params.max_time = node["max_time"].as<double>();
  }
  if (node["warm_start"]) {
    params.warm_start = node["warm_start"].as<bool>();
  }
  if (node["verbose"]) {
    params.verbose = node["verbose"].as<bool>();
  }
  if (node["rho"]) {
    params.rho = node["rho"].as<double>();
  }
  if (node["mu_eq"]) {
    params.mu_eq = node["mu_eq"].as<double>();
  }
  if (node["mu_ineq"]) {
    params.mu_ineq = node["mu_ineq"].as<double>();
  }
  if (node["eps_abs"]) {
    params.eps_abs = node["eps_abs"].as<double>();
  }
  if (node["eps_rel"]) {
    params.eps_rel = node["eps_rel"].as<double>();
  }
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

YAML::Node findMapField(const YAML::Node& map, const std::string& field) {
  if (!map.IsMap()) {
    return YAML::Node();
  }
  for (const auto& item : map) {
    if (item.first.as<std::string>() == field) {
      return item.second;
    }
  }
  return YAML::Node();
}

bool hasYamlValue(const YAML::Node& node) {
  return node.IsDefined() && !node.IsNull() &&
         node.Type() != YAML::NodeType::Undefined;
}

YAML::Node lifecycleField(const YAML::Node& state_node,
                          const std::string& field) {
  YAML::Node state_field = findMapField(state_node, field);
  if (hasYamlValue(state_field)) {
    return state_field;
  }
  YAML::Node params = findMapField(state_node, "params");
  if (hasYamlValue(params)) {
    return findMapField(params, field);
  }
  return YAML::Node();
}

StateLifecycle parseStateLifecycle(const YAML::Node& state_node) {
  StateLifecycle lifecycle;
  if (auto node = lifecycleField(state_node, "duration"); hasYamlValue(node)) {
    lifecycle.duration = node.as<double>();
  }
  if (auto node = lifecycleField(state_node, "wait_time"); hasYamlValue(node)) {
    lifecycle.wait_time = node.as<double>();
  }
  if (auto node = lifecycleField(state_node, "next_state_id");
      hasYamlValue(node)) {
    lifecycle.next_state_id = node.as<StateId>();
  }
  if (auto node = lifecycleField(state_node, "stay_here"); hasYamlValue(node)) {
    lifecycle.stay_here = node.as<bool>();
  }
  if (auto node = lifecycleField(state_node, "b_stay_here");
      hasYamlValue(node)) {
    lifecycle.stay_here = node.as<bool>();
  }
  return lifecycle;
}

StateTaskSelection parseStateTaskSelection(const YAML::Node& task_entry,
                                           const std::string& state_name) {
  const std::string context =
      "ConfigCompiler::parseStateMachine: state '" + state_name + "'";
  StateTaskSelection selection;

  if (task_entry.IsScalar()) {
    selection.name = task_entry.as<std::string>();
  } else if (task_entry.IsMap()) {
    YamlParser::RequireField(task_entry, "name", context + " task entry");
    selection.name = task_entry["name"].as<std::string>();
    if (task_entry["weight"]) {
      selection.weight = task_entry["weight"].as<double>();
      if (!std::isfinite(selection.weight)) {
        throw std::invalid_argument(context + " task '" + selection.name +
                                    "' must use a finite weight");
      }
    }
    if (task_entry["level"]) {
      selection.level = task_entry["level"].as<int>();
      if (selection.level <= 0) {
        throw std::invalid_argument(context + " task '" + selection.name +
                                    "' must use level > 0");
      }
    }
  } else {
    throw std::invalid_argument(context +
                                " task selection must be a string or map");
  }

  if (selection.name.empty()) {
    throw std::invalid_argument(context + " task selection name is empty");
  }
  return selection;
}

StateContactSelection parseStateContactSelection(
    const YAML::Node& contact_entry, const std::string& state_name) {
  const std::string context =
      "ConfigCompiler::parseStateMachine: state '" + state_name + "'";
  StateContactSelection selection;

  if (contact_entry.IsScalar()) {
    selection.name = contact_entry.as<std::string>();
  } else if (contact_entry.IsMap()) {
    YamlParser::RequireField(contact_entry, "name", context + " contact entry");
    selection.name = contact_entry["name"].as<std::string>();
  } else {
    throw std::invalid_argument(context +
                                " contact selection must be a string or map");
  }

  if (selection.name.empty()) {
    throw std::invalid_argument(context + " contact selection name is empty");
  }
  return selection;
}

}  // namespace

CompiledConfig ConfigCompiler::Compile(const YAML::Node& root) {
  CompiledConfig compiled_config;

  if (root["contact_pool"]) {
    parseContactPool(root["contact_pool"], compiled_config);
  }
  if (root["global_constraints"]) {
    throw std::invalid_argument(
        "ConfigCompiler::Compile: 'global_constraints' is removed. Use "
        "'constraints'.");
  }
  if (root["constraints"]) {
    parseConstraints(root["constraints"], compiled_config);
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
  if (root["solver"]) {
    parseSolver(root["solver"], compiled_config);
  }
  parseDebug(root, compiled_config);
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
    if (item["target_frame"]) {
      task.target_frame = item["target_frame"].as<std::string>();
    }
    if (item["position"]) {
      task.use_position = item["position"].as<bool>();
    }
    if (item["orientation"]) {
      task.use_orientation = item["orientation"].as<bool>();
    }
    if (task.type == TaskTypeSpec::kSE3Task && !task.use_position &&
        !task.use_orientation) {
      throw std::invalid_argument("ConfigCompiler::parseTaskPool: SE3Task '" +
                                  task.name +
                                  "' must enable position or orientation");
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
    contact.type =
        parseContactTypeSpec(item["type"].as<std::string>(), contact.name);
    contact.target_frame = item["target_frame"].as<std::string>();
    if (item["mu"]) {
      contact.mu = item["mu"].as<double>();
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
    // solver hierarchy semantics owned by IDHQP policy.
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

    const std::string context = "ConfigCompiler::parseStateMachine";
    YamlParser::RequireField(stateNode, "id", context);
    YamlParser::RejectField(stateNode, "type", context,
                            "Use 'name' as the state factory key.");
    YamlParser::RejectField(stateNode, "implementation", context,
                            "Use 'name' as the state factory key.");
    YamlParser::RequireField(stateNode, "name", context);

    StateSpec state;
    state.id = YamlParser::RequiredAs<StateId>(stateNode, "id", context);
    state.name =
        YamlParser::RequiredAs<std::string>(stateNode, "name", context);
    state.lifecycle = parseStateLifecycle(stateNode);

    if (stateNode["params"]) {
      state.params = stateNode["params"];
    }

    // Preferred keys: tasks / active_tasks / task_selection.
    const YAML::Node taskSetNode =
        stateNode["tasks"]
            ? stateNode["tasks"]
            : (stateNode["active_tasks"] ? stateNode["active_tasks"]
                                         : stateNode["task_selection"]);
    if (taskSetNode) {
      for (const auto& taskEntry : taskSetNode) {
        state.tasks.push_back(parseStateTaskSelection(taskEntry, state.name));
      }
    }

    // Preferred key: contacts.
    const YAML::Node contactSetNode = stateNode["contacts"];
    if (contactSetNode) {
      for (const auto& contactEntry : contactSetNode) {
        state.contacts.push_back(
            parseStateContactSelection(contactEntry, state.name));
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
    compiled_config.regularization.w_delta_qddot = node["w_qddot"].as<double>();
  }
  if (node["w_lambda"]) {
    compiled_config.regularization.w_lambda = node["w_lambda"].as<double>();
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
  if (node["qddot_ref"]) {
    compiled_config.controller.qddot_ref = node["qddot_ref"].as<bool>();
  }
}

void ConfigCompiler::parseDebug(const YAML::Node& root,
                                CompiledConfig& compiled_config) {
  if (root["debug_mode"]) {
    compiled_config.debug.enabled = root["debug_mode"].as<bool>();
  }
  if (root["debug_print_interval"]) {
    compiled_config.debug.print_interval =
        root["debug_print_interval"].as<double>();
  }
  if (!root["debug"]) {
    return;
  }
  const YAML::Node& node = root["debug"];
  if (node["enabled"]) {
    compiled_config.debug.enabled = node["enabled"].as<bool>();
  }
  if (node["print_interval"]) {
    compiled_config.debug.print_interval = node["print_interval"].as<double>();
  }
}

void ConfigCompiler::parseSolver(const YAML::Node& node,
                                 CompiledConfig& compiled_config) {
  if (node.IsScalar()) {
    compiled_config.solver.type = parseSolverTypeSpec(node.as<std::string>());
    return;
  }

  if (!node.IsMap()) {
    throw std::invalid_argument(
        "ConfigCompiler::parseSolver: solver must be a SolverHQP enum name or "
        "a map");
  }

  if (!node["type"]) {
    throw std::invalid_argument(
        "ConfigCompiler::parseSolver: solver map requires type");
  }
  compiled_config.solver.type =
      parseSolverTypeSpec(node["type"].as<std::string>());
  parseSolverQPParams(node["qp_params"], compiled_config.solver.qp_params);
}

void ConfigCompiler::parseConstraints(const YAML::Node& node,
                                      CompiledConfig& compiled_config) {
  for (auto it = node.begin(); it != node.end(); ++it) {
    const std::string type = it->first.as<std::string>();
    // Joint position/velocity limits are handled downstream via clamp.
    if (type == "JointPosLimitConstraint" ||
        type == "JointVelLimitConstraint") {
      continue;
    }
    ConstraintSpec constraint;
    constraint.type = parseConstraintTypeSpec(type);
    const auto& val = it->second;
    if (val["enabled"]) {
      constraint.enabled = val["enabled"].as<bool>();
    }
    if (val["scale"]) {
      constraint.scale = val["scale"].as<double>();
    }
    compiled_config.constraints.push_back(std::move(constraint));
  }
}

}  // namespace wbc
