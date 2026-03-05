/**
 * @file controller/optimo_controller/src/optimo_controller.cpp
 * @brief Doxygen documentation for optimo_controller module.
 */
#include "optimo_controller/optimo_controller.hpp"

#include <cmath>

#include <Eigen/Dense>
#include <hardware_interface/types/hardware_interface_type_values.hpp>
#include <pluginlib/class_list_macros.hpp>
#include <rclcpp/rclcpp.hpp>

#include "optimo_controller/state_machines/cartesian_teleop.hpp"
#include "optimo_controller/state_machines/joint_teleop.hpp"

namespace optimo_controller
{
////////////////////////////////////////////////////////////////////////

OptimoController::~OptimoController() = default;

////////////////////////////////////////////////////////////////////////

controller_interface::CallbackReturn OptimoController::on_init()
{
  auto_declare<std::vector<std::string>>("joints", {});
  auto_declare<std::string>(
    "wbc_yaml_path", "package://optimo_controller/config/optimo_wbc.yaml");
  auto_declare<double>("control_frequency", 1000.0);
  return controller_interface::CallbackReturn::SUCCESS;
}

////////////////////////////////////////////////////////////////////////

controller_interface::InterfaceConfiguration
OptimoController::command_interface_configuration() const
{
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;

  config.names.reserve(joint_count_ * kInterfacesPerJoint);
  for (const auto & joint : joints_)
  {
    config.names.push_back(joint + "/" + hardware_interface::HW_IF_POSITION);
  }
  for (const auto & joint : joints_)
  {
    config.names.push_back(joint + "/" + hardware_interface::HW_IF_VELOCITY);
  }
  for (const auto & joint : joints_)
  {
    config.names.push_back(joint + "/" + hardware_interface::HW_IF_EFFORT);
  }

  return config;
}

////////////////////////////////////////////////////////////////////////

controller_interface::InterfaceConfiguration
OptimoController::state_interface_configuration() const
{
  return command_interface_configuration();
}

////////////////////////////////////////////////////////////////////////

controller_interface::CallbackReturn
OptimoController::on_configure(const rclcpp_lifecycle::State & /*previous_state*/)
{
  joints_ = get_node()->get_parameter("joints").as_string_array();
  joint_count_ = joints_.size();
  wbc_yaml_path_ = get_node()->get_parameter("wbc_yaml_path").as_string();
  control_frequency_hz_ = get_node()->get_parameter("control_frequency").as_double();
  if (!std::isfinite(control_frequency_hz_) || control_frequency_hz_ <= 0.0)
  {
    RCLCPP_ERROR(
      get_node()->get_logger(),
      "[OptimoController] parameter 'control_frequency' must be finite and > 0. got=%.6f",
      control_frequency_hz_);
    return controller_interface::CallbackReturn::ERROR;
  }
  control_dt_ = 1.0 / control_frequency_hz_;
  if (joints_.empty())
  {
    RCLCPP_ERROR(get_node()->get_logger(), "[OptimoController] parameter 'joints' is empty.");
    return controller_interface::CallbackReturn::ERROR;
  }

  try
  {
    auto arch_config =
      wbc::ControlArchitectureConfig::FromYaml(wbc_yaml_path_, control_dt_);
    arch_config.state_provider = std::make_unique<wbc::StateProvider>(control_dt_);
    ctrl_arch_ = std::make_unique<wbc::ControlArchitecture>(std::move(arch_config));
    ctrl_arch_->Initialize();
    ctrl_arch_->logger_.enabled = true;

    // Cache typed state pointers (non-RT, configure phase only).
    // Pointers remain valid for the controller's lifetime: states are owned
    // by FSMHandler and never moved or destroyed after Initialize().
    auto * fsm = ctrl_arch_->GetFsmHandler();
    if (const auto id = fsm->FindStateIdByName("joint_teleop")) {
      joint_teleop_state_ = dynamic_cast<wbc::JointTeleop *>(fsm->FindStateById(*id));
    }
    if (const auto id = fsm->FindStateIdByName("cartesian_teleop")) {
      cartesian_teleop_state_ = dynamic_cast<wbc::CartesianTeleop *>(fsm->FindStateById(*id));
    }
  }
  catch (const std::exception & e)
  {
    ctrl_arch_.reset();
    RCLCPP_ERROR(
      get_node()->get_logger(), "[OptimoController] failed to build control architecture: %s",
      e.what());
    return controller_interface::CallbackReturn::ERROR;
  }

  if (joint_teleop_state_ == nullptr) {
    RCLCPP_ERROR(get_node()->get_logger(),
      "[OptimoController] required state 'joint_teleop' not found in WBC config.");
    return controller_interface::CallbackReturn::ERROR;
  }
  if (cartesian_teleop_state_ == nullptr) {
    RCLCPP_ERROR(get_node()->get_logger(),
      "[OptimoController] required state 'cartesian_teleop' not found in WBC config.");
    return controller_interface::CallbackReturn::ERROR;
  }

  // Pre-size joint buffers so readFromRT() in update() never sees an empty vector.
  // (non-RT configure phase — heap alloc acceptable here.)
  {
    const std::vector<double> zeros(joint_count_, 0.0);
    qdot_des_buf_.writeFromNonRT(JointVelRef{zeros, 0});
    q_des_buf_.writeFromNonRT(JointPosRef{zeros, 0});
  }

  // Joint velocity subscriber — Float64MultiArray: [qdot_0 .. qdot_n] [rad/s]
  joint_vel_sub_ =
    get_node()->create_subscription<std_msgs::msg::Float64MultiArray>(
      "~/joint_vel_cmd",
      rclcpp::SensorDataQoS(),
      [this](std_msgs::msg::Float64MultiArray::ConstSharedPtr msg) {
        if (msg->data.size() != joint_count_) { return; }
        qdot_des_buf_.writeFromNonRT(JointVelRef{msg->data, get_node()->now().nanoseconds()});
      });

  // EE velocity subscriber — TwistStamped: linear [m/s] + angular [rad/s], world frame
  ee_vel_sub_ =
    get_node()->create_subscription<geometry_msgs::msg::TwistStamped>(
      "~/ee_vel_cmd",
      rclcpp::SensorDataQoS(),
      [this](geometry_msgs::msg::TwistStamped::ConstSharedPtr msg) {
        xdot_des_buf_.writeFromNonRT(EEVelRef{
          {msg->twist.linear.x,  msg->twist.linear.y,  msg->twist.linear.z},
          {msg->twist.angular.x, msg->twist.angular.y, msg->twist.angular.z},
          rclcpp::Time(msg->header.stamp).nanoseconds()});
      });

  // Joint position subscriber — Float64MultiArray: [q_0 .. q_n] [rad]
  joint_pos_sub_ =
    get_node()->create_subscription<std_msgs::msg::Float64MultiArray>(
      "~/joint_pos_cmd",
      rclcpp::SensorDataQoS(),
      [this](std_msgs::msg::Float64MultiArray::ConstSharedPtr msg) {
        if (msg->data.size() != joint_count_) { return; }
        q_des_buf_.writeFromNonRT(JointPosRef{msg->data, get_node()->now().nanoseconds()});
      });

  // EE pose subscriber — PoseStamped: position [m] + orientation [quat], world frame
  ee_pos_sub_ =
    get_node()->create_subscription<geometry_msgs::msg::PoseStamped>(
      "~/ee_pose_cmd",
      rclcpp::SensorDataQoS(),
      [this](geometry_msgs::msg::PoseStamped::ConstSharedPtr msg) {
        x_des_buf_.writeFromNonRT(EEPoseRef{
          {msg->pose.position.x, msg->pose.position.y, msg->pose.position.z},
          Eigen::Quaterniond(
            msg->pose.orientation.w,
            msg->pose.orientation.x,
            msg->pose.orientation.y,
            msg->pose.orientation.z),
          rclcpp::Time(msg->header.stamp).nanoseconds()});
      });

  // State transition service — call with state_name or state_id.
  // Uses atomic RequestState internally; consumed on the next control tick.
  set_state_srv_ = get_node()->create_service<wbc_msgs::srv::TransitionState>(
    "~/set_state",
    [this](const wbc_msgs::srv::TransitionState::Request::SharedPtr req,
           wbc_msgs::srv::TransitionState::Response::SharedPtr res) {
      if (!req->state_name.empty()) {
        if (ctrl_arch_->RequestState(req->state_name)) {
          res->success = true;
          res->message = "Transition requested: " + req->state_name;
        } else {
          res->success = false;
          res->message = "Unknown state name: " + req->state_name;
        }
      } else {
        ctrl_arch_->RequestState(req->state_id);
        res->success = true;
        res->message = "Transition requested: id=" + std::to_string(req->state_id);
      }
    });

  // Log available states for discoverability.
  {
    const auto& states = ctrl_arch_->GetFsmHandler()->GetStates();
    std::string state_list;
    for (const auto& [id, name] : states) {
      if (!state_list.empty()) state_list += ", ";
      state_list += std::to_string(id) + ":" + name;
    }
    RCLCPP_INFO(get_node()->get_logger(),
      "[OptimoController] Available states: [%s]", state_list.c_str());
  }

  // RT-safe WBC state publisher for monitoring (PlotJuggler, custom viz, etc.)
  {
    auto pub = get_node()->create_publisher<wbc_msgs::msg::WbcState>(
      "~/wbc_state", rclcpp::SensorDataQoS());
    rt_wbc_pub_ = std::make_shared<
      realtime_tools::RealtimePublisher<wbc_msgs::msg::WbcState>>(pub);

    // Pre-allocate message vectors so publish path does no heap alloc.
    auto& msg = rt_wbc_pub_->msg_;
    msg.q_des.resize(joint_count_, 0.0);
    msg.qdot_des.resize(joint_count_, 0.0);
    msg.q_curr.resize(joint_count_, 0.0);
    msg.qdot_curr.resize(joint_count_, 0.0);
    msg.q_cmd.resize(joint_count_, 0.0);
    msg.qdot_cmd.resize(joint_count_, 0.0);
    msg.qddot_cmd.resize(joint_count_, 0.0);
    msg.tau_ff.resize(joint_count_, 0.0);
    msg.tau_fb.resize(joint_count_, 0.0);
    msg.tau.resize(joint_count_, 0.0);
    msg.gravity.resize(joint_count_, 0.0);
  }

  return controller_interface::CallbackReturn::SUCCESS;
}

////////////////////////////////////////////////////////////////////////

controller_interface::CallbackReturn
OptimoController::on_activate(const rclcpp_lifecycle::State & /*previous_state*/)
{
  const std::size_t expected_interfaces = joint_count_ * kInterfacesPerJoint;
  if (state_interfaces_.size() < expected_interfaces ||
    command_interfaces_.size() < expected_interfaces)
  {
    RCLCPP_ERROR(
      get_node()->get_logger(),
      "[OptimoController] missing interfaces. expected>=%zu (state=%zu, command=%zu)",
      expected_interfaces, state_interfaces_.size(), command_interfaces_.size());
    return controller_interface::CallbackReturn::ERROR;
  }

  robot_joint_state_.Reset(static_cast<Eigen::Index>(joint_count_));

  for (auto & cmd_if : command_interfaces_)
  {
    (void)cmd_if.set_value(0.0);
  }
  return controller_interface::CallbackReturn::SUCCESS;
}

////////////////////////////////////////////////////////////////////////

controller_interface::CallbackReturn
OptimoController::on_deactivate(const rclcpp_lifecycle::State & /*previous_state*/)
{
  for (auto & cmd_if : command_interfaces_)
  {
    (void)cmd_if.set_value(0.0);
  }
  return controller_interface::CallbackReturn::SUCCESS;
}

////////////////////////////////////////////////////////////////////////

controller_interface::return_type OptimoController::update(
  const rclcpp::Time & time, const rclcpp::Duration & period)
{
  if (!ctrl_arch_) return controller_interface::return_type::OK;

  /**
   * @brief Direct joint path (control-critical, RT primary).
   *
   * Hardware interface layout:
   * - state_interfaces_[0 .. n-1]       : joint position
   * - state_interfaces_[n .. 2n-1]      : joint velocity
   * - state_interfaces_[2n .. 3n-1]     : measured effort
   *
   * Memory behavior:
   * - `robot_joint_state_` is pre-allocated in on_activate().
   * - Per tick this section only overwrites scalar entries (no resize/new).
   */
  // Route teleop commands to the active state.
  // active_state_id_ is refreshed after each ctrl_arch_->Update() so FSM
  // auto-transitions are observed before the next tick's dispatch.
  // No dynamic_cast or atomic read in the hot path.
  if (active_state_id_ == joint_teleop_state_->id()) {
    const auto* qdot_des = qdot_des_buf_.readFromRT();
    const auto* q_des = q_des_buf_.readFromRT();
    joint_teleop_state_->UpdateCommand(
      Eigen::Map<const Eigen::VectorXd>(qdot_des->qdot.data(), joint_count_),
      qdot_des->ts_ns,
      Eigen::Map<const Eigen::VectorXd>(q_des->q.data(), joint_count_),
      q_des->ts_ns);
  }

  if (active_state_id_ == cartesian_teleop_state_->id()) {
    const auto* xdot_des = xdot_des_buf_.readFromRT();
    const auto* x_des = x_des_buf_.readFromRT();
    cartesian_teleop_state_->UpdateCommand(
      xdot_des->xdot, xdot_des->wdot, xdot_des->ts_ns,
      x_des->x, x_des->w, x_des->ts_ns);
  }

  ctrl_arch_->Update(ReadJointState(), time.seconds(), control_dt_);
  // Refresh after FSM ran so auto-transitions are captured for the next tick.
  active_state_id_ = ctrl_arch_->GetCurrentStateId();

  // Write command to hardware interfaces.
  // Layout: [pos x n] [vel x n] [effort x n]
  //
  // Command ordering matches command_interface_configuration():
  // - [0 .. n-1]       : desired joint position
  // - [n .. 2n-1]      : desired joint velocity
  // - [2n .. 3n-1]     : desired joint torque

  WriteJointCommand(ctrl_arch_->GetCommand());

  PublishWbcState(time);

  return controller_interface::return_type::OK;
}

////////////////////////////////////////////////////////////////////////

void OptimoController::PublishWbcState(const rclcpp::Time& time)
{
  if (!ctrl_arch_->logger_.HasNewData()) return;

  if (rt_wbc_pub_ && rt_wbc_pub_->trylock()) {
    const auto& src = ctrl_arch_->logger_.GetStateData();
    auto& msg = rt_wbc_pub_->msg_;

    msg.header.stamp = time;
    msg.state_id = src.state_id;

    const auto n = joint_count_;
    auto copy = [&](auto& dst, const auto& s, std::size_t count) {
      for (std::size_t i = 0; i < count; ++i) dst[i] = s[i];
    };

    copy(msg.q_des,     src.q_des,     n);
    copy(msg.qdot_des,  src.qdot_des,  n);
    copy(msg.q_curr,    src.q_curr,    n);
    copy(msg.qdot_curr, src.qdot_curr, n);
    copy(msg.q_cmd,     src.q_cmd,     n);
    copy(msg.qdot_cmd,  src.qdot_cmd,  n);
    copy(msg.qddot_cmd, src.qddot_cmd, n);
    copy(msg.tau_ff,    src.tau_ff,    n);
    copy(msg.tau_fb,    src.tau_fb,    n);
    copy(msg.tau,       src.tau,       n);
    copy(msg.gravity,   src.gravity,   n);

    msg.tasks.resize(src.tasks.size());
    for (std::size_t t = 0; t < src.tasks.size(); ++t) {
      const auto& ts = src.tasks[t];
      auto& td = msg.tasks[t];
      td.name     = ts.name;
      td.dim      = ts.dim;
      td.priority = ts.priority;
      td.x_des    = ts.x_des;
      td.xdot_des = ts.xdot_des;
      td.x_curr   = ts.x_curr;
      td.x_err    = ts.x_err;
      td.op_cmd   = ts.op_cmd;
      td.kp       = ts.kp;
      td.kd       = ts.kd;
    }

    rt_wbc_pub_->unlockAndPublish();
    ctrl_arch_->logger_.ClearNewData();
  }
}

////////////////////////////////////////////////////////////////////////

const wbc::RobotJointState & OptimoController::ReadJointState()
{
  for (std::size_t i = 0; i < joint_count_; ++i)
  {
    robot_joint_state_.q[i] =
      state_interfaces_[InterfaceIndex(kPositionBlock, i, joint_count_)].get_value();
    robot_joint_state_.qdot[i] =
      state_interfaces_[InterfaceIndex(kVelocityBlock, i, joint_count_)].get_value();
    robot_joint_state_.tau[i] =
      state_interfaces_[InterfaceIndex(kEffortBlock, i, joint_count_)].get_value();
  }
  return robot_joint_state_;
}

////////////////////////////////////////////////////////////////////////

void OptimoController::WriteJointCommand(const wbc::RobotCommand & cmd)
{
  for (std::size_t i = 0; i < joint_count_; ++i)
  {
    (void)command_interfaces_[InterfaceIndex(kPositionBlock, i, joint_count_)].set_value(cmd.q[i]);
    (void)command_interfaces_[InterfaceIndex(kVelocityBlock, i, joint_count_)].set_value(cmd.qdot[i]);
    (void)command_interfaces_[InterfaceIndex(kEffortBlock, i, joint_count_)].set_value(cmd.tau[i]);
  }
}

}  // namespace optimo_controller

PLUGINLIB_EXPORT_CLASS(
  optimo_controller::OptimoController, controller_interface::ControllerInterface)
