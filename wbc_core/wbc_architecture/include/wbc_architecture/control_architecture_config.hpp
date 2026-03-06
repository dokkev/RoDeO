/**
 * @file wbc_core/wbc_architecture/include/wbc_architecture/control_architecture_config.hpp
 * @brief Doxygen documentation for control_architecture_config module.
 */
#pragma once

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>  // for YAML sequence parsing in FromYaml

#include <Eigen/Dense>

#include "wbc_architecture/config_compiler.hpp"
#include "wbc_architecture/runtime_config.hpp"
#include "wbc_fsm/fsm_handler.hpp"
#include "wbc_robot_system/pinocchio_robot_system.hpp"
#include "wbc_robot_system/state_provider.hpp"
#include "wbc_solver/wbic.hpp"
#include "wbc_util/joint_pid.hpp"
#include "wbc_util/ros_path_utils.hpp"
#include "wbc_util/yaml_parser.hpp"

namespace wbc {

/**
 * @brief Non-RT assembly bundle for constructing ControlArchitecture.
 *
 * @details
 * Contains all heavy/once-only objects prepared from YAML:
 * - robot model instance
 * - compiled runtime config
 * - runtime providers/handlers
 * - nominal control period
 */
class ControlArchitectureConfig {
public:
  static ControlArchitectureConfig FromYaml(const std::string& yaml_path,
                                            const double control_dt) {
    if (control_dt <= 0.0) {
      throw std::runtime_error(
          "[ControlArchitectureConfig] control_dt must be positive.");
    }

    const param::WbcDefinition definition = param::ParseDefinition(yaml_path);
    const std::string resolved_urdf_path = path::ResolveRelativePath(
        definition.robot_model.urdf_path, definition.yaml_path);
    const std::string package_root = path::ResolveUrdfPackageRoot(
        definition.robot_model.urdf_path, resolved_urdf_path);

    ControlArchitectureConfig config;
    config.robot = std::make_unique<PinocchioRobotSystem>(
        resolved_urdf_path, package_root,
        !definition.robot_model.is_floating_base, false);
    config.compiler = ConfigCompiler::Compile(config.robot.get(),
                                               definition.yaml_path);
    config.runtime_config = config.compiler->TakeConfig();
    config.state_provider = std::make_unique<StateProvider>(control_dt);
    config.control_dt = control_dt;

    // controller: physics compensation flags + optional PID feedback.
    const YAML::Node& ctrl = definition.root["controller"];
    if (ctrl) {
      if (ctrl["enable_gravity_compensation"])
        config.enable_gravity = ctrl["enable_gravity_compensation"].as<bool>();
      if (ctrl["enable_coriolis_compensation"])
        config.enable_coriolis = ctrl["enable_coriolis_compensation"].as<bool>();
      if (ctrl["enable_inertia_compensation"])
        config.enable_inertia = ctrl["enable_inertia_compensation"].as<bool>();

      if (ctrl["ik_method"]) {
        const std::string method = ctrl["ik_method"].as<std::string>();
        if (method == "weighted_qp")
          config.ik_method = IKMethod::WEIGHTED_QP;
        else if (method == "hierarchy")
          config.ik_method = IKMethod::HIERARCHY;
        else
          throw std::runtime_error(
              "[ControlArchitectureConfig] Unknown ik_method: '" + method +
              "'. Use 'hierarchy' or 'weighted_qp'.");
      }

      if (ctrl["weight_min"])
        config.weight_min = ctrl["weight_min"].as<double>();
      if (ctrl["weight_max"])
        config.weight_max = ctrl["weight_max"].as<double>();

      // joint_pid: optional joint-level feedback controller.
      const YAML::Node& pid_node = ctrl["joint_pid"];
      if (pid_node) {
        config.joint_pid.enabled =
            pid_node["enabled"] && pid_node["enabled"].as<bool>();

        // Parse a scalar or sequence YAML node into an Eigen column vector.
        auto parse_gain = [](const YAML::Node& n) -> Eigen::VectorXd {
          if (!n) return Eigen::VectorXd::Zero(1);
          if (n.IsSequence()) {
            const auto v = n.as<std::vector<double>>();
            return Eigen::Map<const Eigen::VectorXd>(v.data(), v.size());
          }
          return Eigen::VectorXd::Constant(1, n.as<double>());
        };

        if (pid_node["gains_yaml"]) {
          // Load per-joint gains by joint name from a separate YAML file.
          const std::string gains_path = path::ResolveRelativePath(
              pid_node["gains_yaml"].as<std::string>(), definition.yaml_path);
          const YAML::Node gains_root = param::LoadYamlFile(gains_path);

          const int n = config.robot->NumActiveDof();

          // Read defaults (zero if absent).
          auto def_scalar = [&](const char* key) -> double {
            return (gains_root["default"] && gains_root["default"][key])
                       ? gains_root["default"][key].as<double>() : 0.0;
          };
          config.joint_pid.kp_pos = Eigen::VectorXd::Constant(n, def_scalar("kp_pos"));
          config.joint_pid.ki_pos = Eigen::VectorXd::Constant(n, def_scalar("ki_pos"));
          config.joint_pid.kd_pos = Eigen::VectorXd::Constant(n, def_scalar("kd_pos"));
          config.joint_pid.kp_vel = Eigen::VectorXd::Constant(n, def_scalar("kp_vel"));
          config.joint_pid.ki_vel = Eigen::VectorXd::Constant(n, def_scalar("ki_vel"));
          config.joint_pid.kd_vel = Eigen::VectorXd::Constant(n, def_scalar("kd_vel"));

          if (gains_root["joints"]) {
            for (const auto& entry : gains_root["joints"]) {
              const std::string jname = entry.first.as<std::string>();
              try {
                const int idx = config.robot->GetActuatedIndex(jname);
                auto set_if = [&](const char* key, Eigen::VectorXd& vec) {
                  if (entry.second[key]) vec[idx] = entry.second[key].as<double>();
                };
                set_if("kp_pos", config.joint_pid.kp_pos);
                set_if("ki_pos", config.joint_pid.ki_pos);
                set_if("kd_pos", config.joint_pid.kd_pos);
                set_if("kp_vel", config.joint_pid.kp_vel);
                set_if("ki_vel", config.joint_pid.ki_vel);
                set_if("kd_vel", config.joint_pid.kd_vel);
              } catch (const std::exception&) {
                // Unknown joint name in gains file — skip silently.
              }
            }
          }
        } else {
          // Inline scalar/vector gains.
          config.joint_pid.kp_pos = parse_gain(pid_node["kp_pos"]);
          config.joint_pid.ki_pos = parse_gain(pid_node["ki_pos"]);
          config.joint_pid.kd_pos = parse_gain(pid_node["kd_pos"]);
          config.joint_pid.kp_vel = parse_gain(pid_node["kp_vel"]);
          config.joint_pid.ki_vel = parse_gain(pid_node["ki_vel"]);
          config.joint_pid.kd_vel = parse_gain(pid_node["kd_vel"]);
        }

        if (pid_node["pos_integral_limit"])
          config.joint_pid.pos_integral_limit = parse_gain(pid_node["pos_integral_limit"]);
        if (pid_node["vel_integral_limit"])
          config.joint_pid.vel_integral_limit = parse_gain(pid_node["vel_integral_limit"]);
      }
    }

    return config;
  }

  std::unique_ptr<PinocchioRobotSystem> robot;
  std::unique_ptr<RuntimeConfig>        runtime_config;
  std::unique_ptr<ConfigCompiler>       compiler;   // alive until ControlArchitecture::Initialize()
  std::unique_ptr<StateProvider>        state_provider;
  std::unique_ptr<FSMHandler>           fsm_handler;
  double control_dt{0.001};
  JointPIDConfig joint_pid;

  // IK solver method: HIERARCHY (null-space projection) or WEIGHTED_QP (weight-based).
  IKMethod ik_method{IKMethod::WEIGHTED_QP};

  // Weight Ratio Guard: clamp all task weights to [weight_min, weight_max].
  double weight_min{1e-6};
  double weight_max{1e4};

  // Physics compensation toggles (all true by default — full inverse dynamics).
  // When disabled, the corresponding term is zeroed before passing to the solver.
  bool enable_gravity{true};
  bool enable_coriolis{true};
  bool enable_inertia{true};
};

}  // namespace wbc
