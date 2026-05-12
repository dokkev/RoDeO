#include "joint_impedance_controller/joint_impedance_controller.hpp"

#include <memory>
#include <string>
#include <vector>
#include <cmath>
#include <limits>

#include "controller_interface/helpers.hpp"
#include "hardware_interface/loaned_command_interface.hpp"
#include "hardware_interface/types/hardware_interface_type_values.hpp"
#include "rclcpp/logging.hpp"
#include "rclcpp/qos.hpp"

namespace joint_impedance_controller
{

namespace
{
void set_command_interface_value(
  hardware_interface::LoanedCommandInterface & command_interface,
  double value)
{
  (void)command_interface.set_value(value);
}

double read_state_interface_value(
  const hardware_interface::LoanedStateInterface & state_interface)
{
  return state_interface.get_optional().value_or(std::numeric_limits<double>::quiet_NaN());
}

bool command_is_finite(const CmdType & commands, size_t index)
{
  return std::isfinite(commands.position[index]) &&
         std::isfinite(commands.velocity[index]) &&
         std::isfinite(commands.stiffness[index]) &&
         std::isfinite(commands.damping[index]) &&
         std::isfinite(commands.effort_ff[index]);
}
}  // namespace

std::shared_ptr<CmdType> JointImpedanceController::make_hold_command() const
{
  auto command = std::make_shared<CmdType>();
  const size_t num_joints = joint_names_.size();

  command->position.resize(num_joints, 0.0);
  command->velocity.resize(num_joints, 0.0);
  command->stiffness.resize(num_joints, 0.0);
  command->damping.resize(num_joints, 0.0);
  command->effort_ff.resize(num_joints, 0.0);

  for (size_t i = 0; i < num_joints; ++i) {
    if (i < positions_.size() && std::isfinite(positions_[i])) {
      command->position[i] = positions_[i];
    }
  }

  return command;
}

JointImpedanceController::JointImpedanceController()
: controller_interface::ControllerInterface()
{
}

controller_interface::CallbackReturn JointImpedanceController::on_init()
{
  try
  {
    param_listener_ = std::make_shared<ParamListener>(get_node());
  }
  catch (const std::exception & e)
  {
    fprintf(stderr, "Exception thrown during init stage with message: %s \n", e.what());
    return controller_interface::CallbackReturn::ERROR;
  }

  return controller_interface::CallbackReturn::SUCCESS;
}

////////////////////////////////////////////////////////////////////////

controller_interface::CallbackReturn JointImpedanceController::on_configure(
  const rclcpp_lifecycle::State & /*previous_state*/)
{
  if (!param_listener_)
  {
    RCLCPP_ERROR(get_node()->get_logger(), "Parameter listener was not initialized");
    return controller_interface::CallbackReturn::ERROR;
  }

  params_ = param_listener_->get_params();

  if (params_.joints.empty()) 
  {
    RCLCPP_ERROR(get_node()->get_logger(), "'joints' parameter was empty");
    return controller_interface::CallbackReturn::ERROR;
  }

  joint_names_ = params_.joints;
  const size_t num_joints = joint_names_.size();

  // Initialize state vectors
  positions_.resize(num_joints, 0.0);
  velocities_.resize(num_joints, 0.0);
  efforts_.resize(num_joints, 0.0);
  position_errors_.resize(num_joints, 0.0);
  velocity_errors_.resize(num_joints, 0.0);
  feedback_efforts_.resize(num_joints, 0.0);
  desired_efforts_.resize(num_joints, 0.0);

  // Create command subscriber with VOLATILE QoS
  auto qos = rclcpp::QoS(rclcpp::KeepLast(10));
  qos.durability(RMW_QOS_POLICY_DURABILITY_VOLATILE);
  
  joints_command_subscriber_ = get_node()->create_subscription<CmdType>(
    "~/commands", qos,
    [this](const CmdType::SharedPtr msg) { rt_command_ptr_.writeFromNonRT(msg); });

  // Create state publisher
  publisher_ = get_node()->create_publisher<StateMsg>(
    "~/controller_state", rclcpp::SystemDefaultsQoS());

  state_publisher_ = std::make_unique<realtime_tools::RealtimePublisher<StateMsg>>(publisher_);

  // Pre-allocate message fields to avoid runtime allocations
  state_publisher_->lock();
  auto& msg = state_publisher_->msg_;
  msg.joint_names = joint_names_;
  msg.position_desired.resize(num_joints, 0.0);
  msg.velocity_desired.resize(num_joints, 0.0);
  msg.position_actual.resize(num_joints, 0.0);
  msg.velocity_actual.resize(num_joints, 0.0);
  msg.position_error.resize(num_joints, 0.0);
  msg.velocity_error.resize(num_joints, 0.0);
  msg.stiffness.resize(num_joints, 0.0);
  msg.damping.resize(num_joints, 0.0);
  msg.effort_ff.resize(num_joints, 0.0);
  msg.effort_fb.resize(num_joints, 0.0);
  msg.effort_desired.resize(num_joints, 0.0);
  msg.effort_actual.resize(num_joints, 0.0);
  state_publisher_->unlock();

  RCLCPP_INFO(
    get_node()->get_logger(),
    "Configured successfully. compute_impedance_torque=%s",
    params_.compute_impedance_torque ? "true" : "false");
  return controller_interface::CallbackReturn::SUCCESS;
}

////////////////////////////////////////////////////////////////////////

controller_interface::CallbackReturn JointImpedanceController::on_activate(
  const rclcpp_lifecycle::State & /*previous_state*/)
{
  const auto num_joints = joint_names_.size();
  read_state_interfaces();

  // Clear and reserve interface vectors (prevents reallocation)
  position_command_interfaces_.clear();
  velocity_command_interfaces_.clear();
  effort_command_interfaces_.clear();
  stiffness_command_interfaces_.clear();
  damping_command_interfaces_.clear();

  position_command_interfaces_.reserve(num_joints);
  velocity_command_interfaces_.reserve(num_joints);
  effort_command_interfaces_.reserve(num_joints);
  stiffness_command_interfaces_.reserve(num_joints);
  damping_command_interfaces_.reserve(num_joints);

  // Gather command interfaces in order
  for (const auto & joint_name : joint_names_)
  {
    const std::string pos_name = joint_name + "/" + hardware_interface::HW_IF_POSITION;
    const std::string vel_name = joint_name + "/" + hardware_interface::HW_IF_VELOCITY;
    const std::string eff_name = joint_name + "/" + hardware_interface::HW_IF_EFFORT;
    const std::string stiff_name = joint_name + "/stiffness";
    const std::string damp_name = joint_name + "/damping";

    for (auto & command_interface : command_interfaces_)
    {
      const auto& iface_name = command_interface.get_name();
      
      if (iface_name == pos_name) {
        position_command_interfaces_.emplace_back(command_interface);
      } else if (iface_name == vel_name) {
        velocity_command_interfaces_.emplace_back(command_interface);
      } else if (iface_name == eff_name) {
        effort_command_interfaces_.emplace_back(command_interface);
      } else if (iface_name == stiff_name) {
        stiffness_command_interfaces_.emplace_back(command_interface);
      } else if (iface_name == damp_name) {
        damping_command_interfaces_.emplace_back(command_interface);
      }
    }
  }

  // Validate all interfaces were found
  if (position_command_interfaces_.size() != num_joints ||
      velocity_command_interfaces_.size() != num_joints ||
      effort_command_interfaces_.size() != num_joints ||
      stiffness_command_interfaces_.size() != num_joints ||
      damping_command_interfaces_.size() != num_joints)
  {
    RCLCPP_FATAL(get_node()->get_logger(), "Not all command interfaces found!");
    return controller_interface::CallbackReturn::ERROR;
  }

  // Reset command buffer
  rt_command_ptr_ = realtime_tools::RealtimeBuffer<std::shared_ptr<CmdType>>(make_hold_command());
  std::fill(position_errors_.begin(), position_errors_.end(), 0.0);
  std::fill(velocity_errors_.begin(), velocity_errors_.end(), 0.0);
  std::fill(feedback_efforts_.begin(), feedback_efforts_.end(), 0.0);
  std::fill(desired_efforts_.begin(), desired_efforts_.end(), 0.0);
  state_publish_counter_ = 0;

  RCLCPP_INFO(get_node()->get_logger(), "Activated successfully");
  return controller_interface::CallbackReturn::SUCCESS;
}

////////////////////////////////////////////////////////////////////////

controller_interface::CallbackReturn JointImpedanceController::on_deactivate(
  const rclcpp_lifecycle::State & /*previous_state*/)
{
  rt_command_ptr_ = realtime_tools::RealtimeBuffer<std::shared_ptr<CmdType>>(nullptr);
  std::fill(position_errors_.begin(), position_errors_.end(), 0.0);
  std::fill(velocity_errors_.begin(), velocity_errors_.end(), 0.0);
  std::fill(feedback_efforts_.begin(), feedback_efforts_.end(), 0.0);
  std::fill(desired_efforts_.begin(), desired_efforts_.end(), 0.0);
  state_publish_counter_ = 0;
  
  // Zero all commands
  const size_t num_joints = joint_names_.size();
  for (size_t i = 0; i < num_joints; ++i)
  {
    set_command_interface_value(position_command_interfaces_[i].get(), 0.0);
    set_command_interface_value(velocity_command_interfaces_[i].get(), 0.0);
    set_command_interface_value(effort_command_interfaces_[i].get(), 0.0);
    set_command_interface_value(stiffness_command_interfaces_[i].get(), 0.0);
    set_command_interface_value(damping_command_interfaces_[i].get(), 0.0);
  }
  
  return controller_interface::CallbackReturn::SUCCESS;
}

////////////////////////////////////////////////////////////////////////

void JointImpedanceController::read_state_interfaces()
{
  const size_t num_joints = joint_names_.size();
  for (size_t i = 0; i < num_joints; ++i)
  {
    positions_[i] = read_state_interface_value(state_interfaces_[i * 3]);
    velocities_[i] = read_state_interface_value(state_interfaces_[i * 3 + 1]);
    efforts_[i] = read_state_interface_value(state_interfaces_[i * 3 + 2]);
  }
}

////////////////////////////////////////////////////////////////////////

controller_interface::return_type JointImpedanceController::update(
  const rclcpp::Time & time, const rclcpp::Duration & /*period*/)
{
  // Read state interfaces
  read_state_interfaces();

  // Get commands from realtime buffer
  auto impedance_commands = rt_command_ptr_.readFromRT();

  if (!impedance_commands || !(*impedance_commands))
  {
    return controller_interface::return_type::OK;
  }

  const auto& commands = **impedance_commands;
  const size_t num_joints = joint_names_.size();

  // Validate command size
  if (commands.position.size() != num_joints ||
      commands.velocity.size() != num_joints ||
      commands.stiffness.size() != num_joints ||
      commands.damping.size() != num_joints ||
      commands.effort_ff.size() != num_joints)
  {
    RCLCPP_ERROR_THROTTLE(
      get_node()->get_logger(), *(get_node()->get_clock()), 1000,
      "Command size mismatch. Expected %zu joints for position/velocity/stiffness/damping/effort_ff.",
      num_joints);
    return controller_interface::return_type::ERROR;
  }

  for (size_t i = 0; i < num_joints; ++i) {
    if (!command_is_finite(commands, i)) {
      RCLCPP_ERROR_THROTTLE(
        get_node()->get_logger(), *(get_node()->get_clock()), 1000,
        "Received non-finite command values for joint index %zu.", i);
      return controller_interface::return_type::ERROR;
    }
  }

  // Write commands to hardware
  bool missing_feedback_for_pd = false;
  for (size_t i = 0; i < num_joints; ++i)
  {
    const bool state_valid = std::isfinite(positions_[i]) && std::isfinite(velocities_[i]);
    const bool apply_pd = params_.compute_impedance_torque && state_valid;

    if (state_valid) {
      position_errors_[i] = commands.position[i] - positions_[i];
      velocity_errors_[i] = commands.velocity[i] - velocities_[i];
    } else {
      position_errors_[i] = std::numeric_limits<double>::quiet_NaN();
      velocity_errors_[i] = std::numeric_limits<double>::quiet_NaN();
    }

    feedback_efforts_[i] = apply_pd ?
      commands.stiffness[i] * position_errors_[i] +
      commands.damping[i] * velocity_errors_[i] :
      0.0;
    desired_efforts_[i] = commands.effort_ff[i] + feedback_efforts_[i];

    if (params_.compute_impedance_torque && !state_valid) {
      missing_feedback_for_pd = true;
    }
    
    set_command_interface_value(position_command_interfaces_[i].get(), commands.position[i]);
    set_command_interface_value(velocity_command_interfaces_[i].get(), commands.velocity[i]);
    set_command_interface_value(
      effort_command_interfaces_[i].get(),
      params_.compute_impedance_torque ? desired_efforts_[i] : commands.effort_ff[i]);

    set_command_interface_value(
      stiffness_command_interfaces_[i].get(),
      commands.stiffness[i]);
    set_command_interface_value(
      damping_command_interfaces_[i].get(),
      commands.damping[i]);
  }

  if (missing_feedback_for_pd) {
    RCLCPP_WARN_THROTTLE(
      get_node()->get_logger(), *(get_node()->get_clock()), 1000,
      "Skipping controller-side impedance torque on joints without finite state feedback; sending effort_ff only.");
  }

  // Publish controller debug state at a reduced rate to lower update-stage jitter.
  if ((state_publish_counter_++ % state_publish_divisor_) == 0) {
    publish_state(time, commands);
  }

  return controller_interface::return_type::OK;
}

controller_interface::InterfaceConfiguration
JointImpedanceController::command_interface_configuration() const
{
  controller_interface::InterfaceConfiguration command_interfaces_config;
  command_interfaces_config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  
  command_interfaces_config.names.clear();
  for (const auto & joint : joint_names_)
  {
    command_interfaces_config.names.push_back(joint + "/" + hardware_interface::HW_IF_POSITION);
    command_interfaces_config.names.push_back(joint + "/" + hardware_interface::HW_IF_VELOCITY);
    command_interfaces_config.names.push_back(joint + "/" + hardware_interface::HW_IF_EFFORT);
    command_interfaces_config.names.push_back(joint + "/stiffness");
    command_interfaces_config.names.push_back(joint + "/damping");
  }

  return command_interfaces_config;
}

controller_interface::InterfaceConfiguration
JointImpedanceController::state_interface_configuration() const
{
  controller_interface::InterfaceConfiguration state_interfaces_config;
  state_interfaces_config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  state_interfaces_config.names.clear();
  
  // Add state interfaces for the joints
  for (const auto & joint : joint_names_)
  {
    state_interfaces_config.names.push_back(joint + "/" + hardware_interface::HW_IF_POSITION);
    state_interfaces_config.names.push_back(joint + "/" + hardware_interface::HW_IF_VELOCITY);
    state_interfaces_config.names.push_back(joint + "/" + hardware_interface::HW_IF_EFFORT);
  }
  
  return state_interfaces_config;
}

////////////////////////////////////////////////////////////////////////

void JointImpedanceController::publish_state(const rclcpp::Time & time, const CmdType& command)
{
  if (!state_publisher_->trylock()) {
    return;
  }

  auto& msg = state_publisher_->msg_;
  msg.header.stamp = time;

  // Assign actual states
  msg.position_actual = positions_;
  msg.velocity_actual = velocities_;
  msg.effort_actual = efforts_;
  
  // Assign desired states
  msg.position_desired = command.position;
  msg.velocity_desired = command.velocity;
  msg.stiffness = command.stiffness;
  msg.damping = command.damping;
  msg.effort_ff = command.effort_ff;
  msg.position_error = position_errors_;
  msg.velocity_error = velocity_errors_;
  msg.effort_fb = feedback_efforts_;
  msg.effort_desired = desired_efforts_;

  state_publisher_->unlockAndPublish();
}

////////////////////////////////////////////////////////////////////////

}  // namespace joint_impedance_controller

#include "pluginlib/class_list_macros.hpp"

PLUGINLIB_EXPORT_CLASS(
  joint_impedance_controller::JointImpedanceController, controller_interface::ControllerInterface)
