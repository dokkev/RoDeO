/**
 * @file wbc_core/wbc_architecture/src/config_compiler.cpp
 * @brief One-time YAML parsing and FSM construction for WBC startup.
 */
#include "wbc_architecture/config_compiler.hpp"

#include <filesystem>
#include <stdexcept>
#include <utility>
#include <vector>

#include <rclcpp/rclcpp.hpp>

#include "wbc_architecture/runtime_config.hpp"
#include "wbc_formulation/basic_contact.hpp"
#include "wbc_formulation/force_task.hpp"
#include "wbc_formulation/kinematic_constraint.hpp"
#include "wbc_formulation/motion_task.hpp"
#include "wbc_fsm/fsm_handler.hpp"
#include "wbc_fsm/state_factory.hpp"
#include "wbc_robot_system/pinocchio_robot_system.hpp"
#include "wbc_robot_system/state_provider.hpp"
#include "wbc_util/ros_path_utils.hpp"
#include "wbc_util/yaml_parser.hpp"

namespace wbc {
namespace {

std::pair<std::string, std::string> ParseConstraintNameType(
    const YAML::Node& item, const char* section_name) {
  if (!item["name"] || !item["type"]) {
    throw std::runtime_error(std::string("[ConfigCompiler] ") + section_name +
                             " entry must include 'name' and 'type'.");
  }
  return {item["name"].as<std::string>(), item["type"].as<std::string>()};
}

} // namespace

// ---------------------------------------------------------------------------
// Phase 1: Compile
// ---------------------------------------------------------------------------

std::unique_ptr<ConfigCompiler> ConfigCompiler::Compile(
    PinocchioRobotSystem* robot, const std::string& yaml_path) {
  if (robot == nullptr) {
    throw std::runtime_error("[ConfigCompiler] Robot pointer is null.");
  }
  auto compiler = std::unique_ptr<ConfigCompiler>(new ConfigCompiler());
  compiler->robot_ = robot;
  compiler->runtime_config_ =
      std::unique_ptr<RuntimeConfig>(new RuntimeConfig());
  compiler->LoadYaml(robot, yaml_path);
  return compiler;
}

std::unique_ptr<RuntimeConfig> ConfigCompiler::TakeConfig() {
  return std::move(runtime_config_);
}

// ---------------------------------------------------------------------------
// Phase 2: InitializeFsm
// ---------------------------------------------------------------------------

void ConfigCompiler::InitializeFsm(RuntimeConfig& config,
                                    FSMHandler& fsm_handler,
                                    StateProvider& state_provider) {
  StateFactory& factory = StateFactory::Instance();

  for (auto& [state_id_key, recipe] : state_recipes_) {
    if (state_id_key != recipe.id) {
      throw std::runtime_error(
          "[ConfigCompiler] state id key/value mismatch in recipe map.");
    }

    StateMachineConfig context;
    context.robot              = robot_;
    context.task_registry      = config.taskRegistry();
    context.constraint_registry = config.constraintRegistry();
    context.state_provider     = &state_provider;
    context.params             = recipe.params;

    std::unique_ptr<StateMachine> state =
        factory.Create(recipe.type, recipe.id, recipe.name, context);
    if (state == nullptr) {
      throw std::runtime_error(
          "[ConfigCompiler] Failed to create state type '" + recipe.type +
          "' for instance '" + recipe.name + "' (id=" +
          std::to_string(recipe.id) +
          "). Ensure the state type is registered and its creator returns non-null.");
    }

    state->AssignFromRecipe(recipe);
    state->SetParameters(recipe.params);
    fsm_handler.RegisterState(recipe.id, std::move(state));
  }

  // Rebuild the sorted state catalog once after all states are registered.
  fsm_handler.FinalizeStates();

  const StateId start_id = config.StartStateId();
  if (start_id < 0) {
    throw std::runtime_error("[ConfigCompiler] Invalid start_state_id.");
  }
  if (!fsm_handler.SetStartState(start_id)) {
    throw std::runtime_error(
        "[ConfigCompiler] Failed to set start state: id=" +
        std::to_string(start_id));
  }

  // Free all recipe memory — no longer needed after FSM construction.
  state_recipes_.clear();
}

// ---------------------------------------------------------------------------
// LoadYaml
// ---------------------------------------------------------------------------

void ConfigCompiler::LoadYaml(PinocchioRobotSystem* robot,
                               const std::string& yaml_path) {
  if (yaml_path.empty()) {
    throw std::runtime_error("[ConfigCompiler] YAML path is empty.");
  }
  const YAML::Node root = param::LoadYamlFile(yaml_path);
  LoadYaml(robot, yaml_path, root);
}

void ConfigCompiler::LoadYaml(PinocchioRobotSystem* robot,
                               const std::string& yaml_path,
                               const YAML::Node& root) {
  if (yaml_path.empty()) {
    throw std::runtime_error("[ConfigCompiler] YAML path is empty.");
  }
  if (robot == nullptr) {
    throw std::runtime_error("[ConfigCompiler] Robot pointer is null.");
  }

  RuntimeConfig& cfg = *runtime_config_;
  cfg.num_qdot_      = robot->NumQdot();
  cfg.num_active_dof_ = robot->NumActiveDof();
  cfg.num_float_dof_  = robot->NumFloatDof();
  cfg.task_registry_       = std::make_unique<TaskRegistry>();
  cfg.constraint_registry_ = std::make_unique<ConstraintRegistry>();

  cfg.configured_start_state_id_ = param::ParseTopLevelStartStateId(root);
  cfg.robot_model_hints_         = param::ParseRobotModelHints(root);
  cfg.regularization_            = param::ParseRegularization(root, yaml_path);
  cfg.max_contact_dim_           = param::ParseMaxContactDim(root, yaml_path);

  const auto logger = rclcpp::get_logger("config_compiler");
  const auto& reg = cfg.regularization_;
  RCLCPP_INFO(
      logger,
      "Regularization: w_qddot=%.6g, w_rf=%.6g, w_tau=%.6g, w_tau_dot=%.6g, "
      "w_xc_ddot=%.6g, w_f_dot=%.6g",
      reg.w_qddot, reg.w_rf, reg.w_tau, reg.w_tau_dot,
      reg.w_xc_ddot, reg.w_f_dot);

  const std::string yaml_dir =
      std::filesystem::path(yaml_path).parent_path().string();

  if (root["contact_pool"]) {
    ParseConstraintPool(root["contact_pool"], robot);
  }
  if (root["global_constraints"]) {
    ParseGlobalConstraints(root["global_constraints"], robot);
  }

  // task_pool: inline or external file via task_pool_yaml.
  // Weight ratio guard (weight_min/weight_max) can live alongside task_pool
  // or under `controller:` in the main config (backward compat via ControlArchitectureConfig).
  auto parse_weight_bounds = [&cfg](const YAML::Node& node) {
    if (node["weight_min"])
      cfg.weight_min_ = node["weight_min"].as<double>();
    if (node["weight_max"])
      cfg.weight_max_ = node["weight_max"].as<double>();
  };

  if (root["task_pool"]) {
    ParseTaskPool(root["task_pool"], robot);
    parse_weight_bounds(root);
  } else if (root["task_pool_yaml"]) {
    const std::string task_pool_path =
        yaml_dir + "/" + root["task_pool_yaml"].as<std::string>();
    const YAML::Node task_root = param::LoadYamlFile(task_pool_path);
    if (!task_root["task_pool"]) {
      throw std::runtime_error(
          "[ConfigCompiler] task_pool_yaml '" + task_pool_path +
          "' does not contain 'task_pool' key.");
    }
    ParseTaskPool(task_root["task_pool"], robot);
    parse_weight_bounds(task_root);
  }

  // state_machine: inline or external file via state_machine_yaml.
  YAML::Node state_machine;
  if (root["state_machine"]) {
    state_machine = root["state_machine"];
  } else if (root["state_machine_yaml"]) {
    const std::string sm_path =
        yaml_dir + "/" + root["state_machine_yaml"].as<std::string>();
    const YAML::Node sm_root = param::LoadYamlFile(sm_path);
    if (!sm_root["state_machine"]) {
      throw std::runtime_error(
          "[ConfigCompiler] state_machine_yaml '" + sm_path +
          "' does not contain 'state_machine' key.");
    }
    state_machine = sm_root["state_machine"];
  } else {
    throw std::runtime_error(
        "[ConfigCompiler] Missing required 'state_machine' or "
        "'state_machine_yaml' section.");
  }

  if (state_machine.IsSequence() && state_machine.size() > 0 &&
      state_machine[0]["id"]) {
    cfg.first_state_id_ = state_machine[0]["id"].as<StateId>();
  }

  ParseStateMachine(state_machine);
}

// ---------------------------------------------------------------------------
// ParseConstraintPool
// ---------------------------------------------------------------------------

void ConfigCompiler::ParseConstraintPool(const YAML::Node& node,
                                          PinocchioRobotSystem* robot) {
  RuntimeConfig& cfg = *runtime_config_;
  for (const auto& item : node) {
    const auto [name, type] = ParseConstraintNameType(item, "contact_pool");

    if (cfg.constraint_registry_->GetConstraint(name) != nullptr) {
      throw std::runtime_error(
          "[ConfigCompiler] Duplicate constraint name: " + name);
    }

    std::unique_ptr<Constraint> constraint;
    if (type == "SurfaceContact") {
      if (!item["target_frame"]) {
        throw std::runtime_error("[ConfigCompiler] SurfaceContact '" + name +
                                 "' requires 'target_frame'.");
      }
      const int link_idx =
          robot->GetFrameIndex(item["target_frame"].as<std::string>());
      constraint = std::make_unique<SurfaceContact>(
          robot, link_idx, item["mu"].as<double>(0.3),
          item["foot_half_length"].as<double>(0.1),
          item["foot_half_width"].as<double>(0.1));
    } else if (type == "PointContact") {
      if (!item["target_frame"]) {
        throw std::runtime_error("[ConfigCompiler] PointContact '" + name +
                                 "' requires 'target_frame'.");
      }
      const int link_idx =
          robot->GetFrameIndex(item["target_frame"].as<std::string>());
      constraint = std::make_unique<PointContact>(
          robot, link_idx, item["mu"].as<double>(0.3));
    } else if (type == "JointPosLimitConstraint") {
      throw std::runtime_error(
          "[ConfigCompiler] 'JointPosLimitConstraint' is no longer allowed "
          "in contact_pool. Move it to global_constraints.");
    } else {
      throw std::runtime_error(
          "[ConfigCompiler] Unsupported contact_pool type: " + type);
    }

    cfg.constraint_registry_->AddConstraint(name, std::move(constraint));
  }
}

// ---------------------------------------------------------------------------
// ParseGlobalConstraints
// ---------------------------------------------------------------------------

void ConfigCompiler::ParseGlobalConstraints(const YAML::Node& node,
                                             PinocchioRobotSystem* robot) {
  RuntimeConfig& cfg = *runtime_config_;
  const auto logger = rclcpp::get_logger("config_compiler");

  // Helper lambda: create constraint from type string + optional config node.
  auto make_constraint =
      [&](const std::string& type,
          const YAML::Node& entry) -> std::unique_ptr<Constraint> {
    // Check enabled flag (defaults to true).
    if (entry && !entry["enabled"].as<bool>(true)) {
      RCLCPP_INFO(logger, "[ConfigCompiler] global constraint '%s' disabled.",
                  type.c_str());
      return nullptr;
    }

    const double dt = (entry && entry["dt"]) ? entry["dt"].as<double>() : 0.001;

    if (type == "JointPosLimitConstraint") {
      return std::make_unique<JointPosLimitConstraint>(robot, dt);
    } else if (type == "JointVelLimitConstraint") {
      return std::make_unique<JointVelLimitConstraint>(robot, dt);
    } else if (type == "JointTrqLimitConstraint") {
      return std::make_unique<JointTrqLimitConstraint>(robot);
    }
    throw std::runtime_error(
        "[ConfigCompiler] Unsupported global constraint type: " + type);
  };

  if (node.IsMap()) {
    // Map format: constraint type as key, with optional 'scale' per entry.
    //   global_constraints:
    //     JointPosLimitConstraint: { enabled: true, scale: 0.9 }
    //     JointVelLimitConstraint: { enabled: true, scale: 0.8 }
    static const std::vector<std::string> kConstraintTypes = {
        "JointPosLimitConstraint",
        "JointVelLimitConstraint",
        "JointTrqLimitConstraint",
    };

    const int n = robot->NumActiveDof();
    const Eigen::MatrixXd& urdf_pos = robot->JointPosLimits();
    const Eigen::MatrixXd& urdf_vel = robot->JointVelLimits();
    const Eigen::MatrixXd& urdf_trq = robot->JointTrqLimits();

    for (const auto& type : kConstraintTypes) {
      if (!node[type]) continue;

      auto constraint = make_constraint(type, node[type]);
      if (!constraint) continue;

      // Apply per-constraint URDF scaling.
      const double scale = node[type]["scale"].as<double>(1.0);
      if (scale != 1.0) {
        if (auto* pos_c = dynamic_cast<JointPosLimitConstraint*>(constraint.get())) {
          Eigen::MatrixXd limits(n, 2);
          for (int i = 0; i < n; ++i) {
            const double mid  = (urdf_pos(i, 0) + urdf_pos(i, 1)) * 0.5;
            const double half = (urdf_pos(i, 1) - urdf_pos(i, 0)) * 0.5;
            limits(i, 0) = mid - half * scale;
            limits(i, 1) = mid + half * scale;
          }
          pos_c->SetCustomLimits(limits);
          robot->SetSoftPositionLimits(limits);
        } else if (auto* vel_c = dynamic_cast<JointVelLimitConstraint*>(constraint.get())) {
          Eigen::MatrixXd limits(n, 2);
          for (int i = 0; i < n; ++i) {
            limits(i, 0) = urdf_vel(i, 0) * scale;
            limits(i, 1) = urdf_vel(i, 1) * scale;
          }
          vel_c->SetCustomLimits(limits);
          robot->SetSoftVelocityLimits(limits);
        } else if (auto* trq_c = dynamic_cast<JointTrqLimitConstraint*>(constraint.get())) {
          Eigen::MatrixXd limits(n, 2);
          for (int i = 0; i < n; ++i) {
            limits(i, 0) = urdf_trq(i, 0) * scale;
            limits(i, 1) = urdf_trq(i, 1) * scale;
          }
          trq_c->SetCustomLimits(limits);
          robot->SetSoftTorqueLimits(limits);
        }
        RCLCPP_INFO(logger, "[ConfigCompiler] %s scale=%.2f", type.c_str(), scale);
      }

      // Parse soft constraint toggle.
      const bool is_soft = node[type]["is_soft"].as<bool>(false);
      const double soft_weight = node[type]["soft_weight"].as<double>(1e5);
      if (is_soft) {
        cfg.soft_constraint_cfg_[type] = {true, soft_weight};
        RCLCPP_INFO(logger, "[ConfigCompiler] %s is_soft=true, w=%.0e",
                    type.c_str(), soft_weight);
      }

      Constraint* ptr = constraint.get();
      cfg.constraint_registry_->AddConstraint(type, std::move(constraint));
      cfg.global_constraints_.push_back(ptr);
    }
  } else if (node.IsSequence()) {
    // Legacy format: list of {name, type, enabled?} entries.
    for (const auto& item : node) {
      const auto [name, type] =
          ParseConstraintNameType(item, "global_constraints");

      auto constraint = make_constraint(type, item);
      if (!constraint) continue;

      if (cfg.constraint_registry_->GetConstraint(name) != nullptr) {
        throw std::runtime_error(
            "[ConfigCompiler] Duplicate constraint name: " + name);
      }

      // Parse soft constraint toggle (sequence format).
      const bool is_soft = item["is_soft"].as<bool>(false);
      const double soft_weight = item["soft_weight"].as<double>(1e5);
      if (is_soft) {
        cfg.soft_constraint_cfg_[type] = {true, soft_weight};
        RCLCPP_INFO(logger, "[ConfigCompiler] %s is_soft=true, w=%.0e",
                    type.c_str(), soft_weight);
      }

      Constraint* ptr = constraint.get();
      cfg.constraint_registry_->AddConstraint(name, std::move(constraint));
      cfg.global_constraints_.push_back(ptr);
    }
  } else {
    throw std::runtime_error(
        "[ConfigCompiler] global_constraints must be a map or sequence.");
  }
}

// ---------------------------------------------------------------------------
// ParseTaskPool
// ---------------------------------------------------------------------------

void ConfigCompiler::ParseTaskPool(const YAML::Node& node,
                                    PinocchioRobotSystem* robot) {
  RuntimeConfig& cfg = *runtime_config_;
  auto parse_task_role = [](const std::string& role_text,
                            const std::string& task_name) -> MotionTaskRole {
    if (role_text == "operational_task") {
      return MotionTaskRole::kOperationalTask;
    }
    if (role_text == "posture_task") {
      return MotionTaskRole::kPostureTask;
    }
    throw std::runtime_error(
        "[ConfigCompiler] Unsupported motion-task role '" + role_text +
        "' for task '" + task_name +
        "'. Use 'operational_task' or 'posture_task'.");
  };

  for (const auto& item : node) {
    if (!item["name"] || !item["type"]) {
      throw std::runtime_error(
          "[ConfigCompiler] task_pool entry must include 'name' and 'type'.");
    }

    const std::string name = item["name"].as<std::string>();
    const std::string type = item["type"].as<std::string>();

    if (type == "ForceTask") {
      if (cfg.task_registry_->GetForceTask(name) != nullptr) {
        throw std::runtime_error(
            "[ConfigCompiler] Duplicate force task name: " + name);
      }
      if (!item["contact_name"]) {
        throw std::runtime_error("[ConfigCompiler] ForceTask '" + name +
                                 "' requires 'contact_name'.");
      }

      const std::string contact_name = item["contact_name"].as<std::string>();
      Contact* contact = cfg.constraint_registry_->GetContact(contact_name);
      if (contact == nullptr) {
        throw std::runtime_error("[ConfigCompiler] ForceTask '" + name +
                                 "' references missing contact '" +
                                 contact_name + "'.");
      }

      auto force_task = std::make_unique<ForceTask>(robot, contact);
      ForceTaskConfig base_cfg;
      base_cfg.weight = force_task->Weight();
      if (item["weight"]) {
        base_cfg.weight = ParseVectorOrScalar(item["weight"], force_task->Dim(),
                                              name, "weight");
      }
      force_task->SetParameters(base_cfg);

      ForceTask* force_task_ptr = force_task.get();
      cfg.task_registry_->AddForceTask(name, std::move(force_task));
      cfg.default_force_task_cfg_[force_task_ptr] = base_cfg;
      continue;
    }

    if (cfg.task_registry_->GetMotionTask(name) != nullptr) {
      throw std::runtime_error(
          "[ConfigCompiler] Duplicate motion task name: " + name);
    }

    std::unique_ptr<Task> task;
    if (type == "LinkPosTask") {
      if (!item["target_frame"]) {
        throw std::runtime_error("[ConfigCompiler] LinkPosTask '" + name +
                                 "' requires 'target_frame'.");
      }
      const int idx = robot->GetFrameIndex(item["target_frame"].as<std::string>());
      task = std::make_unique<LinkPosTask>(robot, idx);
    } else if (type == "LinkOriTask") {
      if (!item["target_frame"]) {
        throw std::runtime_error("[ConfigCompiler] LinkOriTask '" + name +
                                 "' requires 'target_frame'.");
      }
      const int idx = robot->GetFrameIndex(item["target_frame"].as<std::string>());
      task = std::make_unique<LinkOriTask>(robot, idx);
    } else if (type == "JointTask") {
      task = std::make_unique<JointTask>(robot);
    } else if (type == "ComTask") {
      task = std::make_unique<ComTask>(robot);
    } else {
      throw std::runtime_error("[ConfigCompiler] Unsupported task type: " + type);
    }

    // Wire optional reference_frame for link-space tasks.
    // LinkPosTask / LinkOriTask: per-task reference_frame, else default to base_frame.
    // JointTask / ComTask: reference_frame is not applicable.
    if (type == "LinkPosTask" || type == "LinkOriTask") {
      if (item["reference_frame"]) {
        task->SetReferenceFrameIdx(
            robot->GetFrameIndex(item["reference_frame"].as<std::string>()));
      } else if (!cfg.robot_model_hints_.base_frame_name.empty()) {
        task->SetReferenceFrameIdx(
            robot->GetFrameIndex(cfg.robot_model_hints_.base_frame_name));
      }
    }

    TaskConfig base_cfg = TaskConfig::FromTask(*task);

    if (item["kp"]) {
      base_cfg.kp = ParseVectorOrScalar(item["kp"], task->Dim(), name, "kp");
      task->SetKp(base_cfg.kp);
    }
    if (item["kd"]) {
      base_cfg.kd = ParseVectorOrScalar(item["kd"], task->Dim(), name, "kd");
      task->SetKd(base_cfg.kd);
    }
    if (item["ki"]) {
      base_cfg.ki = ParseVectorOrScalar(item["ki"], task->Dim(), name, "ki");
      task->SetKi(base_cfg.ki);
    }
    if (item["weight"]) {
      base_cfg.weight =
          ParseVectorOrScalar(item["weight"], task->Dim(), name, "weight");
    } else {
      // Default weight: 1.0 for all dimensions (ensures valid pool default).
      base_cfg.weight = Eigen::VectorXd::Ones(task->Dim());
    }
    task->SetWeight(base_cfg.weight);
    if (item["kp_ik"]) {
      base_cfg.kp_ik =
          ParseVectorOrScalar(item["kp_ik"], task->Dim(), name, "kp_ik");
      task->SetKpIK(base_cfg.kp_ik);
    }

    Task* task_ptr = task.get();
    // Safe default keeps legacy behavior unless role is explicitly provided.
    MotionTaskRole task_role = MotionTaskRole::kPostureTask;
    if (item["role"]) {
      task_role = parse_task_role(item["role"].as<std::string>(), name);
    }

    cfg.task_registry_->AddMotionTask(name, std::move(task));
    cfg.default_motion_task_cfg_[task_ptr] = base_cfg;
    cfg.motion_task_roles_[task_ptr] = task_role;
  }
}

// ---------------------------------------------------------------------------
// ParseStateMachine
// ---------------------------------------------------------------------------

void ConfigCompiler::ParseStateMachine(const YAML::Node& node) {
  if (!node.IsSequence()) {
    throw std::runtime_error(
        "[ConfigCompiler] state_machine must be a YAML sequence.");
  }

  RuntimeConfig& cfg = *runtime_config_;
  const auto logger  = rclcpp::get_logger("config_compiler");
  int validation_error_count = 0;
  std::vector<std::pair<StateId, StateId>> transition_edges;

  for (const auto& entry : node) {
    if (!entry["id"] || !entry["name"]) {
      RCLCPP_ERROR(logger,
                   "[ConfigCompiler] Skip state entry without 'id' or 'name'.");
      ++validation_error_count;
      continue;
    }

    const StateId state_id = entry["id"].as<StateId>();
    if (cfg.states_.find(state_id) != cfg.states_.end()) {
      RCLCPP_ERROR(logger, "[ConfigCompiler] Duplicate state id: %d", state_id);
      ++validation_error_count;
      continue;
    }

    StateConfig state;
    StateRecipe recipe;
    try {
      ParseState(entry, state, recipe);
    } catch (const std::exception& e) {
      RCLCPP_ERROR(logger,
                   "[ConfigCompiler] State parse failed (id=%d): %s",
                   state_id, e.what());
      ++validation_error_count;
      continue;
    }

    const YAML::Node params = param::ResolveParamsNode(entry);
    if (params && params["next_state_id"]) {
      transition_edges.emplace_back(state_id,
                                    params["next_state_id"].as<StateId>());
    }

    cfg.states_[state_id]       = std::move(state);
    state_recipes_[state_id]    = std::move(recipe);
  }

  for (const auto& [from_state, to_state] : transition_edges) {
    if (cfg.states_.find(to_state) == cfg.states_.end()) {
      RCLCPP_ERROR(
          logger,
          "[ConfigCompiler] transition error: state id=%d references missing "
          "next_state_id=%d",
          from_state, to_state);
      ++validation_error_count;
    }
  }

  const StateId start_state = cfg.StartStateId();
  if (start_state >= 0 &&
      cfg.states_.find(start_state) == cfg.states_.end()) {
    RCLCPP_ERROR(
        logger,
        "[ConfigCompiler] start_state_id=%d does not exist in state_machine.",
        start_state);
    ++validation_error_count;
  }

  if (validation_error_count > 0) {
    throw std::runtime_error(
        "[ConfigCompiler] Invalid state_machine contract: " +
        std::to_string(validation_error_count) + " unresolved references.");
  }
}

// ---------------------------------------------------------------------------
// ParseState
// ---------------------------------------------------------------------------

void ConfigCompiler::ParseState(const YAML::Node& state_node,
                                 StateConfig& out_config,
                                 StateRecipe& out_recipe) {
  RuntimeConfig& cfg = *runtime_config_;

  out_config.id   = state_node["id"].as<StateId>();
  out_config.name = state_node["name"].as<std::string>();
  if (state_node["type"]) {
    out_config.type = state_node["type"].as<std::string>();
  } else {
    out_config.type = out_config.name;
  }
  if (out_config.type.empty()) {
    throw std::runtime_error("state entry has empty 'type'.");
  }
  out_recipe.id     = out_config.id;
  out_recipe.name   = out_config.name;
  out_recipe.type   = out_config.type;
  out_recipe.params = state_node["params"] ? state_node["params"] : YAML::Node();

  // Per-state weight ramp duration (optional, overrides global default).
  if (out_recipe.params["weight_ramp_duration"]) {
    out_config.weight_ramp_duration =
        out_recipe.params["weight_ramp_duration"].as<double>();
  }

  if (state_node["contact_constraints"]) {
    for (const auto& n : state_node["contact_constraints"]) {
      if (!n["name"]) {
        throw std::runtime_error("contact_constraints entry without 'name'.");
      }
      const std::string name = n["name"].as<std::string>();
      Contact* contact = cfg.constraint_registry_->GetContact(name);
      if (contact == nullptr) {
        throw std::runtime_error("missing contact '" + name + "'.");
      }
      out_config.contacts.push_back(contact);
      out_recipe.contact_by_name[name] = contact;
    }
  }

  if (state_node["kinematic_constraints"]) {
    for (const auto& n : state_node["kinematic_constraints"]) {
      if (!n["name"]) {
        throw std::runtime_error("kinematic_constraints entry without 'name'.");
      }
      const std::string name = n["name"].as<std::string>();
      Constraint* constraint = cfg.constraint_registry_->GetConstraint(name);
      if (constraint == nullptr) {
        throw std::runtime_error("missing kinematic constraint '" + name + "'.");
      }
      out_config.kin.push_back(constraint);
      out_recipe.kin_by_name[name] = constraint;
    }
  }

  if (state_node["force_tasks"]) {
    ForceTask* first_force_task   = nullptr;
    ForceTask* explicit_ee_force  = nullptr;

    for (const auto& n : state_node["force_tasks"]) {
      if (!n["name"]) {
        throw std::runtime_error("force_tasks entry without 'name'.");
      }
      const std::string name = n["name"].as<std::string>();
      ForceTask* force_task = cfg.task_registry_->GetForceTask(name);
      if (force_task == nullptr) {
        throw std::runtime_error("missing force task '" + name + "'.");
      }

      const bool has_weight_override = static_cast<bool>(n["weight"]);
      if (has_weight_override) {
        auto override_cfg = std::make_unique<ForceTaskConfig>();
        override_cfg->weight = ParseVectorOrScalar(
            n["weight"], force_task->Dim(),
            out_config.name + ":" + name, "weight");
        out_config.force_cfg.push_back(override_cfg.get());
        out_config.owned_force_cfg.push_back(std::move(override_cfg));
      } else {
        out_config.force_cfg.push_back(nullptr);
      }
      out_config.forces.push_back(force_task);
      out_recipe.force_by_name[name] = force_task;

      if (first_force_task == nullptr) {
        first_force_task = force_task;
      }
      if (name == "ee_force") {
        explicit_ee_force = force_task;
      }
    }

    // Selection priority:
    // 1) exact "ee_force" task name
    // 2) first registered force task as fallback
    out_config.ee_force =
        (explicit_ee_force != nullptr) ? explicit_ee_force : first_force_task;
  }

  // Task ordering: insertion order from YAML (no priority sorting).
  // Weight-based QP uses task weights to determine relative importance.
  if (!state_node["task_hierarchy"]) {
    throw std::runtime_error("state '" + out_config.name +
                             "' has no task_hierarchy.");
  }

  for (const auto& n : state_node["task_hierarchy"]) {
    if (!n["name"]) {
      throw std::runtime_error("task_hierarchy entry without 'name'.");
    }

    const std::string task_name = n["name"].as<std::string>();
    Task* motion_task = cfg.task_registry_->GetMotionTask(task_name);
    if (motion_task == nullptr) {
      throw std::runtime_error("missing motion task '" + task_name + "'.");
    }

    const bool has_override = static_cast<bool>(n["kp"])   ||
                              static_cast<bool>(n["kd"])   ||
                              static_cast<bool>(n["ki"])   ||
                              static_cast<bool>(n["weight"]) ||
                              static_cast<bool>(n["kp_ik"]);

    if (has_override) {
      auto override_cfg = std::make_unique<TaskConfig>();
      override_cfg->kp     = motion_task->Kp();
      override_cfg->kd     = motion_task->Kd();
      override_cfg->ki     = motion_task->Ki();
      override_cfg->weight = motion_task->Weight();
      override_cfg->kp_ik  = motion_task->KpIK();

      const std::string scoped_name = out_config.name + ":" + task_name;
      if (n["kp"])
        override_cfg->kp = ParseVectorOrScalar(n["kp"], motion_task->Dim(),
                                               scoped_name, "kp");
      if (n["kd"])
        override_cfg->kd = ParseVectorOrScalar(n["kd"], motion_task->Dim(),
                                               scoped_name, "kd");
      if (n["ki"])
        override_cfg->ki = ParseVectorOrScalar(n["ki"], motion_task->Dim(),
                                               scoped_name, "ki");
      if (n["weight"])
        override_cfg->weight = ParseVectorOrScalar(n["weight"], motion_task->Dim(),
                                                   scoped_name, "weight");
      if (n["kp_ik"])
        override_cfg->kp_ik = ParseVectorOrScalar(n["kp_ik"], motion_task->Dim(),
                                                   scoped_name, "kp_ik");

      out_config.motion_cfg.push_back(override_cfg.get());
      out_config.owned_task_cfg.push_back(std::move(override_cfg));
    } else {
      out_config.motion_cfg.push_back(nullptr);
    }

    out_config.motion.push_back(motion_task);
    out_recipe.motion_by_name[task_name] = motion_task;

    if (out_config.ee_pos == nullptr && task_name == "ee_pos")
      out_config.ee_pos = motion_task;
    if (out_config.ee_ori == nullptr && task_name == "ee_ori")
      out_config.ee_ori = motion_task;
    if (out_config.com == nullptr &&
        (task_name == "com" || task_name == "com_task"))
      out_config.com = motion_task;
    if (out_config.joint == nullptr &&
        (task_name == "jpos_task" || task_name == "joint_task"))
      out_config.joint = motion_task;
  }

  // Fallback auto-discovery when canonical names are not used.
  if (out_config.ee_pos == nullptr || out_config.ee_ori == nullptr ||
      out_config.com == nullptr    || out_config.joint == nullptr) {
    for (Task* task : out_config.motion) {
      if (task == nullptr) continue;
      if (out_config.ee_pos == nullptr &&
          dynamic_cast<LinkPosTask*>(task) != nullptr)
        out_config.ee_pos = task;
      if (out_config.ee_ori == nullptr &&
          dynamic_cast<LinkOriTask*>(task) != nullptr)
        out_config.ee_ori = task;
      if (out_config.com == nullptr &&
          dynamic_cast<ComTask*>(task) != nullptr)
        out_config.com = task;
      if (out_config.joint == nullptr &&
          dynamic_cast<JointTask*>(task) != nullptr)
        out_config.joint = task;
    }
  }

  if (out_config.motion_cfg.size() != out_config.motion.size()) {
    throw std::runtime_error("motion override vector size mismatch.");
  }
  if (out_config.force_cfg.size() != out_config.forces.size()) {
    throw std::runtime_error("force override vector size mismatch.");
  }
  if (out_config.motion.empty()) {
    throw std::runtime_error("state '" + out_config.name +
                             "' has no task_hierarchy. WBIC requires at least "
                             "one motion task in each state.");
  }
}

// ---------------------------------------------------------------------------
// ParseVectorOrScalar
// ---------------------------------------------------------------------------

Eigen::VectorXd ConfigCompiler::ParseVectorOrScalar(
    const YAML::Node& node, int dim, const std::string& object_name,
    const std::string& field_name) {
  if (dim <= 0) {
    throw std::runtime_error(
        "[ConfigCompiler] Invalid non-positive dimension for '" + field_name +
        "' in '" + object_name + "'.");
  }
  if (node.IsScalar()) {
    return Eigen::VectorXd::Constant(dim, node.as<double>());
  }
  if (!node.IsSequence()) {
    throw std::runtime_error("[ConfigCompiler] Field '" + field_name +
                             "' in '" + object_name +
                             "' must be scalar or sequence.");
  }

  const std::vector<double> values = node.as<std::vector<double>>();
  if (values.empty()) {
    throw std::runtime_error("[ConfigCompiler] Field '" + field_name +
                             "' in '" + object_name + "' is empty.");
  }
  if (values.size() == 1) {
    return Eigen::VectorXd::Constant(dim, values.front());
  }
  if (static_cast<int>(values.size()) != dim) {
    throw std::runtime_error(
        "[ConfigCompiler] Field '" + field_name + "' in '" + object_name +
        "' has wrong dimension. expected=" + std::to_string(dim) +
        ", got=" + std::to_string(values.size()));
  }
  return Eigen::Map<const Eigen::VectorXd>(values.data(),
                                           static_cast<int>(values.size()));
}

} // namespace wbc
