#include "wbc_architecture/wbc_compiled_config.hpp"

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

#include "wbc_formulation/basic_contact.hpp"
#include "wbc_formulation/kinematic_constraint.hpp"
#include "wbc_robot_system/pinocchio_robot_system.hpp"
#include "wbc_formulation/force_task.hpp"
#include "wbc_formulation/motion_task.hpp"

namespace wbc {
namespace {

YAML::Node ResolveSolverParamsNode(const YAML::Node& config_node,
                                   const std::string& main_yaml_path) {
  if (config_node["solver_params_yaml"]) {
    const std::string raw_path =
        config_node["solver_params_yaml"].as<std::string>();
    if (raw_path.empty()) {
      throw std::runtime_error(
          "[WbcConfigCompiler] solver_params_yaml is empty.");
    }
    std::filesystem::path params_path(raw_path);
    if (params_path.is_relative()) {
      params_path =
          std::filesystem::path(main_yaml_path).parent_path() / params_path;
    }

    YAML::Node external = YAML::LoadFile(params_path.string());
    return external["solver_params"] ? external["solver_params"] : external;
  }
  return config_node["solver_params"];
}

std::optional<int> ParseOptionalNonNegativeInt(const YAML::Node& node,
                                               const char* key) {
  if (!node || !node[key]) {
    return std::nullopt;
  }
  const int value = node[key].as<int>();
  if (value < 0) {
    throw std::runtime_error(std::string("[WbcConfigCompiler] '") + key +
                             "' must be non-negative.");
  }
  return value;
}

} // namespace

Eigen::VectorXd WbcConfigCompiler::ParseVectorOrScalar(
    const YAML::Node& node, int dim, const std::string& object_name,
    const std::string& field_name) {
  if (dim <= 0) {
    throw std::runtime_error("[WbcConfigCompiler] Invalid non-positive dimension "
                             "for '" +
                             field_name + "' in '" + object_name + "'.");
  }
  if (node.IsScalar()) {
    return Eigen::VectorXd::Constant(dim, node.as<double>());
  }
  if (!node.IsSequence()) {
    throw std::runtime_error("[WbcConfigCompiler] Field '" + field_name +
                             "' in '" + object_name +
                             "' must be scalar or sequence.");
  }

  const std::vector<double> values = node.as<std::vector<double>>();
  if (values.empty()) {
    throw std::runtime_error("[WbcConfigCompiler] Field '" + field_name +
                             "' in '" + object_name + "' is empty.");
  }
  if (values.size() == 1) {
    return Eigen::VectorXd::Constant(dim, values.front());
  }
  if (static_cast<int>(values.size()) != dim) {
    throw std::runtime_error("[WbcConfigCompiler] Field '" + field_name +
                             "' in '" + object_name +
                             "' has wrong dimension. expected=" +
                             std::to_string(dim) + ", got=" +
                             std::to_string(values.size()));
  }
  return Eigen::Map<const Eigen::VectorXd>(values.data(),
                                           static_cast<int>(values.size()));
}

YAML::Node WbcConfigCompiler::ResolveParamsNode(const YAML::Node& node) {
  if (!node) {
    return YAML::Node();
  }
  return node["params"] ? node["params"] : node;
}

std::unique_ptr<WbcConfigCompiler> WbcConfigCompiler::Compile(
    PinocchioRobotSystem* robot, const std::string& yaml_path) {
  auto compiled = std::unique_ptr<WbcConfigCompiler>(new WbcConfigCompiler(robot));
  compiled->LoadYaml(yaml_path);
  return compiled;
}

WbcConfigCompiler::WbcConfigCompiler(PinocchioRobotSystem* robot) : robot_(robot) {
  if (robot_ == nullptr) {
    throw std::runtime_error("[WbcConfigCompiler] Robot pointer is null.");
  }
}

void WbcConfigCompiler::LoadYaml(const std::string& yaml_path) {
  if (yaml_path.empty()) {
    throw std::runtime_error("[WbcConfigCompiler] YAML path is empty.");
  }

  config_node_ = YAML::LoadFile(yaml_path);
  task_registry_ = std::make_unique<TaskRegistry>();
  constraint_registry_ = std::make_unique<ConstraintRegistry>();
  states_.clear();
  global_constraints_.clear();

  globals_ = GlobalParams{};
  solver_start_state_id_ = -1;
  solver_max_contact_dim_ = -1;
  expected_num_qdot_.reset();
  expected_num_active_dof_.reset();
  expected_num_float_dof_.reset();

  const YAML::Node robot_model = config_node_["robot_model"];
  if (robot_model) {
    if (robot_model["urdf_path"]) {
      std::cerr
          << "[WbcConfigCompiler] 'robot_model.urdf_path' is ignored. Robot "
             "dimensions are determined from PinocchioRobotSystem."
          << std::endl;
    }
    expected_num_qdot_ =
        ParseOptionalNonNegativeInt(robot_model, "expected_num_qdot");
    expected_num_active_dof_ =
        ParseOptionalNonNegativeInt(robot_model, "expected_num_active_dof");
    expected_num_float_dof_ =
        ParseOptionalNonNegativeInt(robot_model, "expected_num_float_dof");
  }

  ParseSolverParams(ResolveSolverParamsNode(config_node_, yaml_path));

  if (config_node_["contact_pool"]) {
    ParseConstraintPool(config_node_["contact_pool"]);
  }
  if (config_node_["global_constraints"]) {
    ParseGlobalConstraints(config_node_["global_constraints"]);
  }
  if (config_node_["task_pool"]) {
    ParseTaskPool(config_node_["task_pool"]);
  }

  if (!config_node_["state_machine"]) {
    throw std::runtime_error(
        "[WbcConfigCompiler] Missing required 'state_machine' section.");
  }
  CompileStateMachine(config_node_["state_machine"]);
}

void WbcConfigCompiler::ParseSolverParams(const YAML::Node& solver_params) {
  if (solver_params) {
    globals_.w_qddot = solver_params["w_qddot"].as<double>(globals_.w_qddot);
    globals_.w_rf = solver_params["w_rf"].as<double>(globals_.w_rf);
    globals_.w_tau = solver_params["w_tau"].as<double>(globals_.w_tau);

    // Preferred explicit fields, with legacy fallback to w_tau.
    globals_.w_xc_ddot =
        solver_params["w_xc_ddot"].as<double>(globals_.w_tau);
    if (solver_params["w_force_rate_of_change"]) {
      globals_.w_force_rate_of_change =
          solver_params["w_force_rate_of_change"].as<double>();
    } else {
      globals_.w_force_rate_of_change =
          solver_params["w_force_rate"].as<double>(globals_.w_tau);
    }
    solver_start_state_id_ =
        solver_params["start_state_id"].as<int>(solver_start_state_id_);
    solver_max_contact_dim_ =
        solver_params["max_contact_dim"].as<int>(solver_max_contact_dim_);
  }

  std::cout << "[WbcConfigCompiler] Solver weights: "
            << "w_qddot=" << globals_.w_qddot
            << ", w_rf=" << globals_.w_rf
            << ", w_xc_ddot=" << globals_.w_xc_ddot
            << ", w_force_rate_of_change=" << globals_.w_force_rate_of_change
            << std::endl;
}

int WbcConfigCompiler::ResolveConfiguredStartStateId() const {
  if (config_node_["start_state_id"]) {
    return config_node_["start_state_id"].as<int>();
  }
  if (solver_start_state_id_ >= 0) {
    return solver_start_state_id_;
  }
  return -1;
}

void WbcConfigCompiler::ParseConstraintPool(const YAML::Node& node) {
  for (const auto& item : node) {
    if (!item["name"] || !item["type"]) {
      throw std::runtime_error(
          "[WbcConfigCompiler] contact_pool entry must include 'name' and "
          "'type'.");
    }

    const std::string name = item["name"].as<std::string>();
    const std::string type = item["type"].as<std::string>();

    if (constraint_registry_->GetConstraint(name) != nullptr) {
      throw std::runtime_error(
          "[WbcConfigCompiler] Duplicate constraint name: " + name);
    }

    std::unique_ptr<Constraint> constraint;
    if (type == "SurfaceContact") {
      if (!item["link_name"]) {
        throw std::runtime_error("[WbcConfigCompiler] SurfaceContact '" + name +
                                 "' requires 'link_name'.");
      }
      const int link_idx =
          robot_->GetFrameIndex(item["link_name"].as<std::string>());
      constraint = std::make_unique<SurfaceContact>(
          robot_, link_idx, item["mu"].as<double>(0.3),
          item["foot_half_length"].as<double>(0.1),
          item["foot_half_width"].as<double>(0.1));
    } else if (type == "PointContact") {
      if (!item["link_name"]) {
        throw std::runtime_error("[WbcConfigCompiler] PointContact '" + name +
                                 "' requires 'link_name'.");
      }
      const int link_idx =
          robot_->GetFrameIndex(item["link_name"].as<std::string>());
      constraint =
          std::make_unique<PointContact>(robot_, link_idx, item["mu"].as<double>(0.3));
    } else if (type == "JointPosLimitConstraint") {
      throw std::runtime_error(
          "[WbcConfigCompiler] 'JointPosLimitConstraint' is no longer allowed "
          "in contact_pool. Move it to global_constraints.");
    } else {
      throw std::runtime_error("[WbcConfigCompiler] Unsupported contact_pool type: " +
                               type);
    }

    constraint_registry_->AddConstraint(name, std::move(constraint));
  }
}

void WbcConfigCompiler::ParseGlobalConstraints(const YAML::Node& node) {
  for (const auto& item : node) {
    if (!item["name"] || !item["type"]) {
      throw std::runtime_error(
          "[WbcConfigCompiler] global_constraints entry must include 'name' and "
          "'type'.");
    }

    const std::string name = item["name"].as<std::string>();
    const std::string type = item["type"].as<std::string>();

    if (constraint_registry_->GetConstraint(name) != nullptr) {
      throw std::runtime_error(
          "[WbcConfigCompiler] Duplicate constraint name: " + name);
    }

    std::unique_ptr<Constraint> constraint;
    if (type == "JointPosLimitConstraint") {
      constraint = std::make_unique<JointPosLimitConstraint>(
          robot_, item["dt"].as<double>(0.001));
    } else if (type == "JointVelLimitConstraint") {
      constraint = std::make_unique<JointVelLimitConstraint>(
          robot_, item["dt"].as<double>(0.001));
    } else if (type == "JointTrqLimitConstraint") {
      constraint = std::make_unique<JointTrqLimitConstraint>(robot_);
    } else {
      throw std::runtime_error(
          "[WbcConfigCompiler] Unsupported global constraint type: " + type);
    }

    Constraint* ptr = constraint.get();
    constraint_registry_->AddConstraint(name, std::move(constraint));
    global_constraints_.push_back(ptr);
  }
}

void WbcConfigCompiler::ParseTaskPool(const YAML::Node& node) {
  for (const auto& item : node) {
    if (!item["name"] || !item["type"]) {
      throw std::runtime_error(
          "[WbcConfigCompiler] task_pool entry must include 'name' and 'type'.");
    }

    const std::string name = item["name"].as<std::string>();
    const std::string type = item["type"].as<std::string>();

    if (type == "ForceTask") {
      if (task_registry_->GetForceTask(name) != nullptr) {
        throw std::runtime_error(
            "[WbcConfigCompiler] Duplicate force task name: " + name);
      }
      if (!item["contact_name"]) {
        throw std::runtime_error("[WbcConfigCompiler] ForceTask '" + name +
                                 "' requires 'contact_name'.");
      }

      const std::string contact_name = item["contact_name"].as<std::string>();
      Contact* contact = constraint_registry_->GetContact(contact_name);
      if (contact == nullptr) {
        throw std::runtime_error("[WbcConfigCompiler] ForceTask '" + name +
                                 "' references missing contact '" + contact_name +
                                 "'.");
      }

      auto force_task = std::make_unique<ForceTask>(robot_, contact);
      if (item["weight"]) {
        ForceTaskConfig cfg;
        cfg.weight =
            ParseVectorOrScalar(item["weight"], force_task->Dim(), name, "weight");
        force_task->SetParameters(cfg);
      }

      task_registry_->AddForceTask(name, std::move(force_task));
      continue;
    }

    if (task_registry_->GetMotionTask(name) != nullptr) {
      throw std::runtime_error(
          "[WbcConfigCompiler] Duplicate motion task name: " + name);
    }

    std::unique_ptr<Task> task;
    if (type == "LinkPosTask") {
      if (!item["link_name"]) {
        throw std::runtime_error("[WbcConfigCompiler] LinkPosTask '" + name +
                                 "' requires 'link_name'.");
      }
      const int idx = robot_->GetFrameIndex(item["link_name"].as<std::string>());
      task = std::make_unique<LinkPosTask>(robot_, idx);
    } else if (type == "LinkOriTask") {
      if (!item["link_name"]) {
        throw std::runtime_error("[WbcConfigCompiler] LinkOriTask '" + name +
                                 "' requires 'link_name'.");
      }
      const int idx = robot_->GetFrameIndex(item["link_name"].as<std::string>());
      task = std::make_unique<LinkOriTask>(robot_, idx);
    } else if (type == "JointTask") {
      task = std::make_unique<JointTask>(robot_);
    } else if (type == "ComTask") {
      task = std::make_unique<ComTask>(robot_);
    } else {
      throw std::runtime_error("[WbcConfigCompiler] Unsupported task type: " + type);
    }

    TaskConfig base_cfg;
    base_cfg.kp = task->Kp();
    base_cfg.kd = task->Kd();
    base_cfg.ki = task->Ki();
    base_cfg.weight = task->Weight();
    base_cfg.kp_ik = task->KpIK();

    if (item["kp"]) {
      base_cfg.kp = ParseVectorOrScalar(item["kp"], task->Dim(), name, "kp");
    }
    if (item["kd"]) {
      base_cfg.kd = ParseVectorOrScalar(item["kd"], task->Dim(), name, "kd");
    }
    if (item["ki"]) {
      base_cfg.ki = ParseVectorOrScalar(item["ki"], task->Dim(), name, "ki");
    }
    if (item["weight"]) {
      base_cfg.weight =
          ParseVectorOrScalar(item["weight"], task->Dim(), name, "weight");
    }
    if (item["kp_ik"]) {
      base_cfg.kp_ik =
          ParseVectorOrScalar(item["kp_ik"], task->Dim(), name, "kp_ik");
    }

    // Base task settings are fixed at pool compile time.
    task->SetKp(base_cfg.kp);
    task->SetKd(base_cfg.kd);
    task->SetKi(base_cfg.ki);
    task->SetWeight(base_cfg.weight);
    task->SetKpIK(base_cfg.kp_ik);

    task_registry_->AddMotionTask(name, std::move(task));
  }
}

void WbcConfigCompiler::CompileStateMachine(const YAML::Node& node) {
  if (!node.IsSequence()) {
    throw std::runtime_error(
        "[WbcConfigCompiler] state_machine must be a YAML sequence.");
  }

  int validation_error_count = 0;
  std::vector<std::pair<int, int>> transition_edges;

  for (const auto& entry : node) {
    if (!entry["id"] || !entry["name"]) {
      std::cerr << "[WbcConfigCompiler] Skip state entry without 'id' or 'name'."
                << std::endl;
      ++validation_error_count;
      continue;
    }

    const int state_id = entry["id"].as<int>();
    if (states_.find(state_id) != states_.end()) {
      std::cerr << "[WbcConfigCompiler] Duplicate state id: " << state_id
                << std::endl;
      ++validation_error_count;
      continue;
    }

    CompiledState state;
    try {
      CompileState(entry, state);
    } catch (const std::exception& e) {
      std::cerr << "[WbcConfigCompiler] State compile failed (id=" << state_id
                << "): " << e.what() << std::endl;
      ++validation_error_count;
      continue;
    }

    const YAML::Node params = ResolveParamsNode(entry);
    if (params && params["next_state_id"]) {
      transition_edges.emplace_back(state_id, params["next_state_id"].as<int>());
    }

    states_[state_id] = std::move(state);
  }

  for (const auto& edge : transition_edges) {
    if (states_.find(edge.second) == states_.end()) {
      std::cerr << "[WbcConfigCompiler] transition error: state id=" << edge.first
                << " references missing next_state_id=" << edge.second
                << std::endl;
      ++validation_error_count;
    }
  }

  const int start_state = StartStateId();
  if (start_state >= 0 && states_.find(start_state) == states_.end()) {
    std::cerr << "[WbcConfigCompiler] start_state_id=" << start_state
              << " does not exist in state_machine." << std::endl;
    ++validation_error_count;
  }

  if (validation_error_count > 0) {
    throw std::runtime_error("[WbcConfigCompiler] Invalid state_machine contract: " +
                             std::to_string(validation_error_count) +
                             " unresolved references.");
  }
}

void WbcConfigCompiler::CompileState(const YAML::Node& state_node,
                                     CompiledState& out) {
  out.id = state_node["id"].as<int>();
  out.name = state_node["name"].as<std::string>();
  out.params = state_node["params"] ? state_node["params"] : YAML::Node();

  if (state_node["contact_constraints"]) {
    for (const auto& n : state_node["contact_constraints"]) {
      if (!n["name"]) {
        throw std::runtime_error("contact_constraints entry without 'name'.");
      }
      const std::string name = n["name"].as<std::string>();
      Contact* contact = constraint_registry_->GetContact(name);
      if (contact == nullptr) {
        throw std::runtime_error("missing contact '" + name + "'.");
      }
      out.contacts.push_back(contact);
      out.contact_by_name[name] = contact;
    }
  }

  if (state_node["kinematic_constraints"]) {
    for (const auto& n : state_node["kinematic_constraints"]) {
      if (!n["name"]) {
        throw std::runtime_error("kinematic_constraints entry without 'name'.");
      }
      const std::string name = n["name"].as<std::string>();
      Constraint* constraint = constraint_registry_->GetConstraint(name);
      if (constraint == nullptr) {
        throw std::runtime_error("missing kinematic constraint '" + name + "'.");
      }
      out.kin.push_back(constraint);
      out.kin_by_name[name] = constraint;
    }
  }

  if (state_node["force_tasks"]) {
    for (const auto& n : state_node["force_tasks"]) {
      if (!n["name"]) {
        throw std::runtime_error("force_tasks entry without 'name'.");
      }
      const std::string name = n["name"].as<std::string>();
      ForceTask* force_task = task_registry_->GetForceTask(name);
      if (force_task == nullptr) {
        throw std::runtime_error("missing force task '" + name + "'.");
      }

      const bool has_weight_override = static_cast<bool>(n["weight"]);
      if (has_weight_override) {
        auto cfg = std::make_unique<ForceTaskConfig>();
        cfg->weight = ParseVectorOrScalar(n["weight"], force_task->Dim(),
                                          out.name + ":" + name, "weight");
        out.force_cfg.push_back(cfg.get());
        out.owned_force_cfg.push_back(std::move(cfg));
      } else {
        out.force_cfg.push_back(nullptr);
      }
      out.forces.push_back(force_task);
      out.force_by_name[name] = force_task;
      if (out.ee_force == nullptr) {
        out.ee_force = force_task;
      }
      if (name == "ee_force" || name.find("ee") != std::string::npos) {
        out.ee_force = force_task;
      }
    }
  }

  struct OrderedEntry {
    int priority{99};
    YAML::Node node;
  };
  std::vector<OrderedEntry> ordered_motion_entries;

  if (state_node["motion_tasks"]) {
    for (const auto& n : state_node["motion_tasks"]) {
      ordered_motion_entries.push_back({n["priority"].as<int>(99), n});
    }
  }
  std::sort(ordered_motion_entries.begin(), ordered_motion_entries.end(),
            [](const OrderedEntry& a, const OrderedEntry& b) {
              return a.priority < b.priority;
            });

  for (const auto& entry : ordered_motion_entries) {
    const YAML::Node n = entry.node;
    if (!n["name"]) {
      throw std::runtime_error("motion_tasks entry without 'name'.");
    }

    const std::string task_name = n["name"].as<std::string>();
    Task* motion_task = task_registry_->GetMotionTask(task_name);
    if (motion_task == nullptr) {
      throw std::runtime_error("missing motion task '" + task_name + "'.");
    }

    const bool has_override = static_cast<bool>(n["kp"]) ||
                              static_cast<bool>(n["kd"]) ||
                              static_cast<bool>(n["ki"]) ||
                              static_cast<bool>(n["weight"]) ||
                              static_cast<bool>(n["kp_ik"]);

    if (has_override) {
      auto cfg = std::make_unique<TaskConfig>();
      cfg->kp = motion_task->Kp();
      cfg->kd = motion_task->Kd();
      cfg->ki = motion_task->Ki();
      cfg->weight = motion_task->Weight();
      cfg->kp_ik = motion_task->KpIK();

      const std::string scoped_name = out.name + ":" + task_name;
      if (n["kp"]) {
        cfg->kp =
            ParseVectorOrScalar(n["kp"], motion_task->Dim(), scoped_name, "kp");
      }
      if (n["kd"]) {
        cfg->kd =
            ParseVectorOrScalar(n["kd"], motion_task->Dim(), scoped_name, "kd");
      }
      if (n["ki"]) {
        cfg->ki =
            ParseVectorOrScalar(n["ki"], motion_task->Dim(), scoped_name, "ki");
      }
      if (n["weight"]) {
        cfg->weight = ParseVectorOrScalar(n["weight"], motion_task->Dim(),
                                          scoped_name, "weight");
      }
      if (n["kp_ik"]) {
        cfg->kp_ik = ParseVectorOrScalar(n["kp_ik"], motion_task->Dim(),
                                         scoped_name, "kp_ik");
      }

      out.motion_cfg.push_back(cfg.get());
      out.owned_task_cfg.push_back(std::move(cfg));
    } else {
      out.motion_cfg.push_back(nullptr);
    }

    out.motion.push_back(motion_task);
    out.motion_by_name[task_name] = motion_task;
    if (out.ee_pos == nullptr && task_name == "ee_pos") {
      out.ee_pos = motion_task;
    }
    if (out.ee_ori == nullptr && task_name == "ee_ori") {
      out.ee_ori = motion_task;
    }
    if (out.joint == nullptr &&
        (task_name == "jpos_task" || task_name == "joint_task")) {
      out.joint = motion_task;
    }
  }

  // Fallback auto-discovery when canonical names are not used.
  if (out.ee_pos == nullptr || out.ee_ori == nullptr || out.joint == nullptr) {
    for (Task* task : out.motion) {
      if (task == nullptr) {
        continue;
      }
      if (out.ee_pos == nullptr && dynamic_cast<LinkPosTask*>(task) != nullptr) {
        out.ee_pos = task;
      }
      if (out.ee_ori == nullptr && dynamic_cast<LinkOriTask*>(task) != nullptr) {
        out.ee_ori = task;
      }
      if (out.joint == nullptr && dynamic_cast<JointTask*>(task) != nullptr) {
        out.joint = task;
      }
    }
  }

  if (out.motion_cfg.size() != out.motion.size()) {
    throw std::runtime_error("motion override vector size mismatch.");
  }
  if (out.force_cfg.size() != out.forces.size()) {
    throw std::runtime_error("force override vector size mismatch.");
  }
  if (out.motion.empty()) {
    throw std::runtime_error("state '" + out.name +
                             "' has no motion_tasks. WBIC requires at least "
                             "one motion task in each state.");
  }
}

const CompiledState& WbcConfigCompiler::State(int state_id) const {
  const auto it = states_.find(state_id);
  if (it == states_.end()) {
    throw std::runtime_error("[WbcConfigCompiler] Unknown state id: " +
                             std::to_string(state_id));
  }
  return it->second;
}

const CompiledState* WbcConfigCompiler::FindState(int state_id) const {
  const auto it = states_.find(state_id);
  if (it == states_.end()) {
    return nullptr;
  }
  return &it->second;
}

void WbcConfigCompiler::BuildFormulation(int state_id, WbcFormulation& out) const {
  const CompiledState& state = State(state_id);

  out.Clear();
  out.motion_tasks = state.motion;
  out.contact_constraints = state.contacts;
  out.force_tasks = state.forces;

  out.kinematic_constraints = global_constraints_;
  out.kinematic_constraints.insert(out.kinematic_constraints.end(),
                                   state.kin.begin(), state.kin.end());
}

std::vector<bool> WbcConfigCompiler::BuildActuationMask() const {
  const int num_qdot = robot_->NumQdot();
  std::vector<bool> mask(static_cast<std::size_t>(num_qdot), true);

  const YAML::Node robot_model = config_node_["robot_model"];
  if (robot_model && robot_model["unactuated_qdot_indices"]) {
    const YAML::Node indices = robot_model["unactuated_qdot_indices"];
    if (!indices.IsSequence()) {
      throw std::runtime_error(
          "[WbcConfigCompiler] robot_model.unactuated_qdot_indices must be a "
          "YAML sequence.");
    }
    for (const auto& idx_node : indices) {
      const int idx = idx_node.as<int>();
      if (idx < 0 || idx >= num_qdot) {
        throw std::runtime_error(
            "[WbcConfigCompiler] robot_model.unactuated_qdot_indices contains "
            "out-of-range index " +
            std::to_string(idx) + " for NumQdot()=" +
            std::to_string(num_qdot) + ".");
      }
      mask[static_cast<std::size_t>(idx)] = false;
    }
    return mask;
  }

  bool floating_base = false;
  if (robot_model && robot_model["floating_base"]) {
    floating_base = robot_model["floating_base"].as<bool>(false);
  } else {
    floating_base = (robot_->NumFloatDof() > 0);
  }

  if (floating_base) {
    const int num_virtual = std::min(6, num_qdot);
    for (int i = 0; i < num_virtual; ++i) {
      mask[static_cast<std::size_t>(i)] = false;
    }
  }
  return mask;
}

int WbcConfigCompiler::MaxContactDim() const {
  if (solver_max_contact_dim_ > 0) {
    return solver_max_contact_dim_;
  }

  int estimated_dim = 0;
  const auto& constraints = constraint_registry_->GetConstraints();
  for (const auto& kv : constraints) {
    const Contact* contact = dynamic_cast<Contact*>(kv.second.get());
    if (contact != nullptr) {
      estimated_dim += contact->Dim();
    }
  }
  return estimated_dim > 0 ? estimated_dim : 24;
}

int WbcConfigCompiler::StartStateId() const {
  const int configured_start = ResolveConfiguredStartStateId();
  if (configured_start >= 0) {
    return configured_start;
  }

  const YAML::Node state_machine = config_node_["state_machine"];
  if (!state_machine || !state_machine.IsSequence() || state_machine.size() == 0) {
    return -1;
  }
  const YAML::Node first = state_machine[0];
  if (!first["id"]) {
    return -1;
  }
  return first["id"].as<int>();
}

std::string WbcConfigCompiler::BaseFrameName() const {
  const YAML::Node robot_model = config_node_["robot_model"];
  if (!robot_model) {
    return "";
  }
  return robot_model["base_frame"].as<std::string>("");
}

std::string WbcConfigCompiler::EndEffectorFrameName() const {
  const YAML::Node robot_model = config_node_["robot_model"];
  if (!robot_model) {
    return "";
  }
  return robot_model["end_effector_frame"].as<std::string>("");
}

void WbcConfigCompiler::ValidateRobotDimensions() const {
  if (robot_ == nullptr) {
    throw std::runtime_error("[WbcConfigCompiler] Robot pointer is null.");
  }

  if (expected_num_qdot_.has_value() &&
      robot_->NumQdot() != *expected_num_qdot_) {
    throw std::runtime_error(
        "[WbcConfigCompiler] expected_num_qdot mismatch. expected=" +
        std::to_string(*expected_num_qdot_) +
        ", actual=" + std::to_string(robot_->NumQdot()));
  }

  if (expected_num_active_dof_.has_value() &&
      robot_->NumActiveDof() != *expected_num_active_dof_) {
    throw std::runtime_error(
        "[WbcConfigCompiler] expected_num_active_dof mismatch. expected=" +
        std::to_string(*expected_num_active_dof_) +
        ", actual=" + std::to_string(robot_->NumActiveDof()));
  }

  if (expected_num_float_dof_.has_value() &&
      robot_->NumFloatDof() != *expected_num_float_dof_) {
    throw std::runtime_error(
        "[WbcConfigCompiler] expected_num_float_dof mismatch. expected=" +
        std::to_string(*expected_num_float_dof_) +
        ", actual=" + std::to_string(robot_->NumFloatDof()));
  }
}

} // namespace wbc
