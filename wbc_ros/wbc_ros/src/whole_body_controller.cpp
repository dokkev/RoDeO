// Copyright 2026
//
// Licensed under the Apache License, Version 2.0.

#include "wbc_ros/whole_body_controller.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <utility>

#include <Eigen/Dense>
#include <hardware_interface/types/hardware_interface_type_values.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pluginlib/class_list_macros.hpp>
#include <rclcpp/rclcpp.hpp>

#include "control_architecture/runtime/runtime_loader.hpp"

namespace wbc_ros {
namespace {

using wbc::robots::RobotSystem;

constexpr double kDefaultControlFrequencyHz = 1000.0;
constexpr std::array<const char*, 3> kStateInterfaces = {
    hardware_interface::HW_IF_POSITION, hardware_interface::HW_IF_VELOCITY,
    hardware_interface::HW_IF_EFFORT};
constexpr const char* kCommandInterfaceKp = "kp";
constexpr const char* kCommandInterfaceKd = "kd";

std::vector<std::string> DefaultCommandInterfaces() {
  return {hardware_interface::HW_IF_POSITION, hardware_interface::HW_IF_VELOCITY,
          hardware_interface::HW_IF_EFFORT};
}

std::vector<std::string> StateInterfaces() {
  return {kStateInterfaces.begin(), kStateInterfaces.end()};
}

Eigen::VectorXd ToJointVector(const std::vector<double>& values,
                              std::size_t joint_count,
                              const std::string& parameter_name) {
  if (values.empty()) {
    return Eigen::VectorXd::Zero(static_cast<Eigen::Index>(joint_count));
  }
  if (values.size() != joint_count) {
    throw std::invalid_argument("parameter '" + parameter_name +
                                "' must have one value per joint");
  }

  Eigen::VectorXd out(static_cast<Eigen::Index>(joint_count));
  for (std::size_t i = 0; i < joint_count; ++i) {
    out(static_cast<Eigen::Index>(i)) = values[i];
  }
  return out;
}

bool VectorHasSizeAndFinite(const Eigen::VectorXd& value,
                            std::size_t expected_size) {
  return value.size() == static_cast<Eigen::Index>(expected_size) &&
         value.allFinite();
}

bool CommandHasSizeAndFinite(const wbc::LowLevelCommand& cmd,
                             std::size_t joint_count) {
  return VectorHasSizeAndFinite(cmd.q, joint_count) &&
         VectorHasSizeAndFinite(cmd.qdot, joint_count) &&
         VectorHasSizeAndFinite(cmd.tau, joint_count) &&
         VectorHasSizeAndFinite(cmd.kp, joint_count) &&
         VectorHasSizeAndFinite(cmd.kd, joint_count);
}

bool IsSupportedCommandInterface(const std::string& interface) {
  return interface == hardware_interface::HW_IF_POSITION ||
         interface == hardware_interface::HW_IF_VELOCITY ||
         interface == hardware_interface::HW_IF_EFFORT ||
         interface == kCommandInterfaceKp || interface == kCommandInterfaceKd;
}

const Eigen::VectorXd* CommandVectorForInterface(
    const wbc::LowLevelCommand& cmd, const std::string& interface) {
  if (interface == hardware_interface::HW_IF_POSITION) {
    return &cmd.q;
  }
  if (interface == hardware_interface::HW_IF_VELOCITY) {
    return &cmd.qdot;
  }
  if (interface == hardware_interface::HW_IF_EFFORT) {
    return &cmd.tau;
  }
  if (interface == kCommandInterfaceKp) {
    return &cmd.kp;
  }
  if (interface == kCommandInterfaceKd) {
    return &cmd.kd;
  }
  return nullptr;
}

void ApplyConfiguredGains(const Eigen::VectorXd& kp, const Eigen::VectorXd& kd,
                          wbc::LowLevelCommand& cmd) {
  cmd.kp = kp;
  cmd.kd = kd;
}

}  // namespace

WholeBodyController::~WholeBodyController() = default;

controller_interface::CallbackReturn WholeBodyController::on_init() {
  command_interface_names_ = DefaultCommandInterfaces();
  auto_declare<std::vector<std::string>>("joints", {});
  auto_declare<std::string>("wbc_yaml_path", "");
  auto_declare<double>("control_frequency", kDefaultControlFrequencyHz);
  auto_declare<bool>("is_simulation", false);
  auto_declare<std::string>("actuator_model", "");
  auto_declare<std::vector<double>>("spring_stiffness", {});
  auto_declare<std::vector<double>>("spring_damping", {});
  auto_declare<std::vector<std::string>>("command_interfaces",
                                         DefaultCommandInterfaces());
  auto_declare<std::vector<double>>("command_kp", {});
  auto_declare<std::vector<double>>("command_kd", {});
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::InterfaceConfiguration
WholeBodyController::command_interface_configuration() const {
  return JointInterfaceConfiguration(command_interface_names_);
}

controller_interface::InterfaceConfiguration
WholeBodyController::state_interface_configuration() const {
  return JointInterfaceConfiguration(StateInterfaces());
}

controller_interface::InterfaceConfiguration
WholeBodyController::JointInterfaceConfiguration(
    const std::vector<std::string>& interface_names) const {
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  config.names.reserve(joint_count_ * interface_names.size());
  for (const auto& interface : interface_names) {
    for (const auto& joint : joints_) {
      config.names.push_back(joint + "/" + interface);
    }
  }
  return config;
}

controller_interface::CallbackReturn WholeBodyController::on_configure(
    const rclcpp_lifecycle::State&) {
  joints_ = get_node()->get_parameter("joints").as_string_array();
  joint_count_ = joints_.size();
  const auto yaml_path = get_node()->get_parameter("wbc_yaml_path").as_string();
  const double control_frequency_hz =
      get_node()->get_parameter("control_frequency").as_double();

  if (!std::isfinite(control_frequency_hz) || control_frequency_hz <= 0.0) {
    RCLCPP_ERROR(get_node()->get_logger(),
                 "[WholeBodyController] control_frequency must be finite and "
                 "> 0");
    return controller_interface::CallbackReturn::ERROR;
  }
  control_dt_ = 1.0 / control_frequency_hz;

  if (joints_.empty()) {
    RCLCPP_ERROR(get_node()->get_logger(),
                 "[WholeBodyController] parameter 'joints' is empty");
    return controller_interface::CallbackReturn::ERROR;
  }

  if (yaml_path.empty()) {
    RCLCPP_ERROR(get_node()->get_logger(),
                 "[WholeBodyController] parameter 'wbc_yaml_path' is empty");
    return controller_interface::CallbackReturn::ERROR;
  }

  if (!ConfigureCommandInterfaces() || !ConfigureCommandGains()) {
    return controller_interface::CallbackReturn::ERROR;
  }

  if (ConfigureRuntime(yaml_path) !=
      controller_interface::CallbackReturn::SUCCESS) {
    return controller_interface::CallbackReturn::ERROR;
  }

  if (!ConfigureActuator()) {
    return controller_interface::CallbackReturn::ERROR;
  }
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn WholeBodyController::ConfigureRuntime(
    const std::string& yaml_path) {
  try {
    auto runtime = wbc::RuntimeLoader::LoadFromYamlFile(yaml_path);
    robot_ = std::move(runtime.robot);
    if (!robot_) {
      throw std::runtime_error("RuntimeLoader returned a null RobotSystem");
    }

    if (!robot_->is_fixed_base()) {
      throw std::runtime_error(
          "WholeBodyController currently requires a fixed-base RobotSystem. "
          "Floating-base support needs an explicit base-state estimator path.");
    }

    if (static_cast<std::size_t>(robot_->na()) != joint_count_) {
      throw std::runtime_error(
          "joint count mismatch: ROS joints=" + std::to_string(joint_count_) +
          ", RobotSystem na=" + std::to_string(robot_->na()));
    }

    ctrl_arch_ = std::make_unique<wbc::ControlArchitecture>(
        std::move(runtime.config), robot_);

    control_profile_ = CreateControlProfile();
    if (control_profile_) {
      control_profile_->RegisterStates(*ctrl_arch_->stateFactory());
      control_profile_->Configure(*ctrl_arch_);
    } else if (!ctrl_arch_->config()->states.empty()) {
      throw std::runtime_error(
          "RobotControlProfile is required because the WBC YAML defines FSM "
          "states. Derive from wbc_ros::WholeBodyController and override "
          "CreateControlProfile().");
    }

    const auto* config = ctrl_arch_->config();
    debug_mode_ = config->debug_enabled;
    debug_print_interval_s_ = config->debug_print_interval;
    ctrl_arch_->setTimingEnabled(debug_mode_);
    ctrl_arch_->Initialize();

    if (config->dt > 0.0 && std::isfinite(config->dt)) {
      control_dt_ = config->dt;
    }

    LogAvailableStates();
  } catch (const std::exception& e) {
    control_profile_.reset();
    ctrl_arch_.reset();
    robot_.reset();
    RCLCPP_ERROR(get_node()->get_logger(),
                 "[WholeBodyController] failed to build WBC runtime: %s",
                 e.what());
    return controller_interface::CallbackReturn::ERROR;
  }

  return controller_interface::CallbackReturn::SUCCESS;
}

std::unique_ptr<wbc::RobotControlProfile>
WholeBodyController::CreateControlProfile() {
  return nullptr;
}

bool WholeBodyController::ConfigureCommandInterfaces() {
  command_interface_names_ =
      get_node()->get_parameter("command_interfaces").as_string_array();
  if (command_interface_names_.empty()) {
    command_interface_names_ = DefaultCommandInterfaces();
  }

  for (const auto& interface : command_interface_names_) {
    if (!IsSupportedCommandInterface(interface)) {
      RCLCPP_ERROR(get_node()->get_logger(),
                   "[WholeBodyController] unsupported command interface '%s'",
                   interface.c_str());
      return false;
    }
  }
  return true;
}

bool WholeBodyController::ConfigureCommandGains() {
  try {
    command_kp_ = ToJointVector(
        get_node()->get_parameter("command_kp").as_double_array(),
        joint_count_, "command_kp");
    command_kd_ = ToJointVector(
        get_node()->get_parameter("command_kd").as_double_array(),
        joint_count_, "command_kd");
    return true;
  } catch (const std::exception& e) {
    RCLCPP_ERROR(get_node()->get_logger(),
                 "[WholeBodyController] invalid command gains: %s", e.what());
    return false;
  }
}

bool WholeBodyController::ConfigureActuator() {
  const bool is_simulation =
      get_node()->get_parameter("is_simulation").as_bool();
  std::string actuator_model =
      get_node()->get_parameter("actuator_model").as_string();
  if (actuator_model.empty()) {
    actuator_model = is_simulation ? "spring" : "direct";
  }

  if (actuator_model == "direct") {
    actuator_ = std::make_unique<wbc::DirectActuator>();
    RCLCPP_INFO(get_node()->get_logger(),
                "[WholeBodyController] actuator model: direct torque");
    return true;
  }

  if (actuator_model == "spring") {
    try {
      const auto stiffness = ToJointVector(
          get_node()->get_parameter("spring_stiffness").as_double_array(),
          joint_count_, "spring_stiffness");
      const auto damping = ToJointVector(
          get_node()->get_parameter("spring_damping").as_double_array(),
          joint_count_, "spring_damping");
      actuator_ = std::make_unique<wbc::SpringActuator>(stiffness, damping);
      RCLCPP_INFO(get_node()->get_logger(),
                  "[WholeBodyController] actuator model: spring");
      return true;
    } catch (const std::exception& e) {
      RCLCPP_ERROR(get_node()->get_logger(),
                   "[WholeBodyController] invalid actuator parameters: %s",
                   e.what());
      return false;
    }
  }

  RCLCPP_ERROR(get_node()->get_logger(),
               "[WholeBodyController] unsupported actuator_model '%s'",
               actuator_model.c_str());
  return false;
}

void WholeBodyController::LogAvailableStates() const {
  std::string state_list;
  for (const auto& [id, state] : ctrl_arch_->fsmHandler()->states()) {
    if (!state_list.empty()) {
      state_list += ", ";
    }
    state_list += std::to_string(id) + ":" + state->name();
  }
  RCLCPP_INFO(get_node()->get_logger(),
              "[WholeBodyController] available states: [%s]",
              state_list.c_str());
}

controller_interface::CallbackReturn WholeBodyController::on_activate(
    const rclcpp_lifecycle::State&) {
  const std::size_t expected_state = joint_count_ * kStateInterfaces.size();
  const std::size_t expected_command =
      joint_count_ * command_interface_names_.size();
  if (state_interfaces_.size() < expected_state ||
      command_interfaces_.size() < expected_command) {
    RCLCPP_ERROR(get_node()->get_logger(),
                 "[WholeBodyController] missing interfaces. expected "
                 "state>=%zu command>=%zu, got state=%zu command=%zu",
                 expected_state, expected_command, state_interfaces_.size(),
                 command_interfaces_.size());
    return controller_interface::CallbackReturn::ERROR;
  }

  robot_state_.q = pinocchio::neutral(robot_->model());
  robot_state_.qdot = Eigen::VectorXd::Zero(robot_->nv());
  robot_state_.time = 0.0;
  output_cmd_.Initialize(static_cast<Eigen::Index>(joint_count_));
  safe_cmd_.Initialize(static_cast<Eigen::Index>(joint_count_));
  ApplyConfiguredGains(command_kp_, command_kd_, output_cmd_);
  ApplyConfiguredGains(command_kp_, command_kd_, safe_cmd_);
  runtime_faulted_ = false;

  Eigen::VectorXd q0(joint_count_);
  for (std::size_t i = 0; i < joint_count_; ++i) {
    const double q_measured =
        state_interfaces_[InterfaceIndex(kPositionBlock, i, joint_count_)]
            .get_value();
    if (!std::isfinite(q_measured)) {
      RCLCPP_ERROR(get_node()->get_logger(),
                   "[WholeBodyController] invalid initial position for joint "
                   "%s",
                   joints_[i].c_str());
      return controller_interface::CallbackReturn::ERROR;
    }
    q0[static_cast<Eigen::Index>(i)] = q_measured;
  }

  output_cmd_.q = q0;
  output_cmd_.qdot.setZero();
  output_cmd_.tau.setZero();
  safe_cmd_ = output_cmd_;

  if (!WriteJointCommand(safe_cmd_)) {
    RCLCPP_ERROR(get_node()->get_logger(),
                 "[WholeBodyController] failed to write initial command");
    return controller_interface::CallbackReturn::ERROR;
  }

  actuator_->Reset(q0);
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn WholeBodyController::on_deactivate(
    const rclcpp_lifecycle::State&) {
  for (std::size_t block = 0; block < command_interface_names_.size();
       ++block) {
    const auto& interface = command_interface_names_[block];
    if (interface != hardware_interface::HW_IF_VELOCITY &&
        interface != hardware_interface::HW_IF_EFFORT &&
        interface != kCommandInterfaceKp && interface != kCommandInterfaceKd) {
      continue;
    }

    for (std::size_t i = 0; i < joint_count_; ++i) {
      (void)command_interfaces_[InterfaceIndex(block, i, joint_count_)]
          .set_value(0.0);
    }
  }
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::return_type WholeBodyController::update(
    const rclcpp::Time& time, const rclcpp::Duration& period) {
  if (!ctrl_arch_) {
    return controller_interface::return_type::OK;
  }

  if (runtime_faulted_) {
    WriteSafeCommand();
    return controller_interface::return_type::ERROR;
  }

  const double time_sec = time.seconds();
  try {
    double dt = period.seconds();
    if (!std::isfinite(dt) || dt <= 0.0) {
      dt = control_dt_;
    }

    if (!ReadRobotState(time_sec)) {
      return HandleRuntimeFault("invalid robot state sample");
    }

    ctrl_arch_->Update(robot_state_, dt);

    if (!PrepareOutputCommand(ctrl_arch_->command(), dt)) {
      return HandleRuntimeFault("invalid WBC command");
    }
    if (!WriteJointCommand(output_cmd_)) {
      return HandleRuntimeFault("failed to write WBC command");
    }

    safe_cmd_.q = output_cmd_.q;
    safe_cmd_.qdot.setZero();
    safe_cmd_.tau.setZero();
    ApplyConfiguredGains(command_kp_, command_kd_, safe_cmd_);
    UpdateDebugStats(time_sec);

    return controller_interface::return_type::OK;
  } catch (const std::exception& e) {
    return HandleRuntimeFault(e.what());
  } catch (...) {
    return HandleRuntimeFault("unknown runtime exception");
  }
}

bool WholeBodyController::PrepareOutputCommand(const wbc::LowLevelCommand& cmd,
                                               double dt) {
  if (!CommandHasSizeAndFinite(cmd, joint_count_) ||
      !CommandHasSizeAndFinite(output_cmd_, joint_count_)) {
    return false;
  }

  output_cmd_.q = cmd.q;
  output_cmd_.qdot = cmd.qdot;
  output_cmd_.tau = cmd.tau;
  ApplyConfiguredGains(command_kp_, command_kd_, output_cmd_);

  if (actuator_) {
    wbc::ActuatorCommand act_cmd(output_cmd_.q, output_cmd_.qdot,
                                 output_cmd_.tau, output_cmd_.kp,
                                 output_cmd_.kd,
                                 robot_->q().tail(robot_->nq_actuated()),
                                 robot_->qdot().tail(robot_->na()), dt);
    return actuator_->ProcessTorque(act_cmd, output_cmd_.tau) &&
           output_cmd_.tau.allFinite();
  }

  return output_cmd_.tau.allFinite();
}

void WholeBodyController::UpdateDebugStats(double time_sec) {
  const auto& ts = ctrl_arch_->timingStats();
  const double total_us =
      ts.model_us + ts.fsm_us + ts.problem_us + ts.solve_us + ts.output_us;
  max_tick_us_ = std::max(max_tick_us_, total_us);

  if (debug_mode_ &&
      (time_sec - last_debug_print_time_) >= debug_print_interval_s_) {
    RCLCPP_INFO(get_node()->get_logger(),
                "[WBC] total=%.0f us peak=%.0f us "
                "model=%.0f fsm=%.0f build=%.0f solve=%.0f out=%.0f",
                total_us, max_tick_us_, ts.model_us, ts.fsm_us, ts.problem_us,
                ts.solve_us, ts.output_us);
    last_debug_print_time_ = time_sec;
    max_tick_us_ = 0.0;
  }
}

bool WholeBodyController::ReadRobotState(double time_sec) {
  if (!std::isfinite(time_sec) || robot_state_.q.size() != robot_->nq() ||
      robot_state_.qdot.size() != robot_->nv()) {
    return false;
  }

  const int q_offset = robot_->is_fixed_base() ? 0 : 7;
  const int v_offset = robot_->is_fixed_base() ? 0 : 6;
  for (std::size_t i = 0; i < joint_count_; ++i) {
    const double q =
        state_interfaces_[InterfaceIndex(kPositionBlock, i, joint_count_)]
            .get_value();
    const double qdot =
        state_interfaces_[InterfaceIndex(kVelocityBlock, i, joint_count_)]
            .get_value();
    if (!std::isfinite(q) || !std::isfinite(qdot)) {
      return false;
    }
    robot_state_.q[q_offset + static_cast<int>(i)] = q;
    robot_state_.qdot[v_offset + static_cast<int>(i)] = qdot;
  }
  robot_state_.time = time_sec;
  return true;
}

bool WholeBodyController::WriteJointCommand(const wbc::LowLevelCommand& cmd) {
  if (!CommandHasSizeAndFinite(cmd, joint_count_)) {
    return false;
  }

  if (command_interfaces_.size() <
      joint_count_ * command_interface_names_.size()) {
    return false;
  }

  for (std::size_t block = 0; block < command_interface_names_.size();
       ++block) {
    const auto* values =
        CommandVectorForInterface(cmd, command_interface_names_[block]);
    if (!values ||
        values->size() != static_cast<Eigen::Index>(joint_count_)) {
      return false;
    }

    for (std::size_t i = 0; i < joint_count_; ++i) {
      (void)command_interfaces_[InterfaceIndex(block, i, joint_count_)]
          .set_value((*values)[static_cast<Eigen::Index>(i)]);
    }
  }
  return true;
}

bool WholeBodyController::WriteSafeCommand() {
  return WriteJointCommand(safe_cmd_);
}

controller_interface::return_type WholeBodyController::HandleRuntimeFault(
    const char* message) {
  if (!runtime_faulted_) {
    RCLCPP_ERROR(get_node()->get_logger(),
                 "[WholeBodyController] runtime fault: %s", message);
  }
  runtime_faulted_ = true;
  WriteSafeCommand();
  return controller_interface::return_type::ERROR;
}

}  // namespace wbc_ros

PLUGINLIB_EXPORT_CLASS(wbc_ros::WholeBodyController,
                       controller_interface::ControllerInterface)
