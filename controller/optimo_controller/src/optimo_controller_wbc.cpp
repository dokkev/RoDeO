/**
 * @file controller/optimo_controller/src/optimo_controller.cpp
 * @brief Doxygen documentation for optimo_controller module.
 */
#include "optimo_controller/optimo_controller.hpp"

#include <algorithm>
#include <cmath>

#include <Eigen/Dense>
#include <hardware_interface/types/hardware_interface_type_values.hpp>
#include <pluginlib/class_list_macros.hpp>
#include <rclcpp/rclcpp.hpp>
#include <yaml-cpp/yaml.h>

// Roboligent SDK for gravity compensation bootstrap.
#ifdef HAS_ROBOLIGENT_SDK
#include <rl/common/RobotConfiguration.h>
#include <rl/model/Model.h>
#include <rl/util/Math.h>
#endif

#include "optimo_controller/state_machines/cartesian_teleop.hpp"
#include "optimo_controller/state_machines/joint_teleop.hpp"
#include "wbc_formulation/interface/task.hpp"
#include "wbc_util/ros_path_utils.hpp"
#include "wbc_util/task_registry.hpp"

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
  auto_declare<bool>("is_simulation", false);
  auto_declare<std::string>(
    "joint_dynamics_yaml", "package://optimo_description/config/joint_dynamics.yaml");
  auto_declare<int>("robot_index", 0);
  return controller_interface::CallbackReturn::SUCCESS;
}

////////////////////////////////////////////////////////////////////////

controller_interface::InterfaceConfiguration
OptimoController::command_interface_configuration() const
{
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;

  config.names.reserve(joint_count_ * kInterfacesPerJoint + 1);
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

  // IMPORTANT: exact name must match `ros2 control list_hardware_interfaces`
  config.names.push_back("/model_safety_error");

  return config;
}

////////////////////////////////////////////////////////////////////////

controller_interface::InterfaceConfiguration
OptimoController::state_interface_configuration() const
{
  // Explicitly list state interfaces — decoupled from command_interface_configuration()
  // so future command-only additions (e.g. model_safety_error) don't silently bleed here.
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  config.names.reserve(joint_count_ * kInterfacesPerJoint);
  for (const auto & joint : joints_)
    config.names.push_back(joint + "/" + hardware_interface::HW_IF_POSITION);
  for (const auto & joint : joints_)
    config.names.push_back(joint + "/" + hardware_interface::HW_IF_VELOCITY);
  for (const auto & joint : joints_)
    config.names.push_back(joint + "/" + hardware_interface::HW_IF_EFFORT);
  return config;
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
    if (const auto id = fsm->FindStateIdByName("safe_command")) {
      safe_command_state_id_ = *id;
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
  if (!safe_command_state_id_.has_value()) {
    RCLCPP_ERROR(get_node()->get_logger(),
      "[OptimoController] required state 'safe_command' not found in WBC config.");
    return controller_interface::CallbackReturn::ERROR;
  }

  // Build actuator interface (spring for sim, passthrough for real HW).
  {
    const bool is_sim = get_node()->get_parameter("is_simulation").as_bool();
    if (is_sim) {
      // SEA stiffness [Nm/rad] and damping [Nm·s/rad] per joint (from hardware spec).
      Eigen::VectorXd stiffness(7);
      stiffness << 966.4, 947.6, 509.3, 404.1, 484.3, 479.2, 455.6;
      Eigen::VectorXd damping(7);
      damping << 10.0, 10.0, 5.0, 5.0, 3.0, 3.0, 3.0;

      actuator_ = std::make_unique<wbc::SpringActuator>(stiffness, damping);
      RCLCPP_INFO(get_node()->get_logger(),
        "[OptimoController] Simulation mode: spring actuator enabled");
    } else {
      actuator_ = std::make_unique<wbc::DirectActuator>();
      RCLCPP_INFO(get_node()->get_logger(),
        "[OptimoController] Hardware mode: direct passthrough");
    }
  }

#ifdef HAS_ROBOLIGENT_SDK
  // Roboligent Model for gravity compensation bootstrap (real hardware only).
  // The hardware interface's check_for_controller() requires non-zero torques
  // before enabling motors. The WBC may output zero torques on early ticks
  // (e.g. QP solver not yet converged). The roboligent model provides gravity
  // comp as a fallback so the hardware can enable.
  {
    const bool is_sim = get_node()->get_parameter("is_simulation").as_bool();
    if (!is_sim) {
      try {
        robot_index_ = get_node()->get_parameter("robot_index").as_int();
        const std::string config_path =
          "/home/optimo/CODE/optimo_wbc_ws/src/optimo_ros/optimo_api/resources/master"
          + std::to_string(robot_index_) + "/OR7_config.yml";
        auto rconfig = std::make_shared<roboligent::RobotConfiguration>(
          config_path, "right_arm");
        rl_model_ = std::make_shared<roboligent::Model>(rconfig);
        rl_pos_deg_.resize(joint_count_, 0.0);
        rl_vel_deg_.resize(joint_count_, 0.0);
        rl_trq_ref_.resize(joint_count_, 0);
        rl_model_first_run_ = true;
        RCLCPP_INFO(get_node()->get_logger(),
          "[OptimoController] Roboligent model loaded for gravity comp bootstrap (robot_index=%d)",
          robot_index_);
      } catch (const std::exception& e) {
        RCLCPP_WARN(get_node()->get_logger(),
          "[OptimoController] Failed to load roboligent model: %s. "
          "Hardware enabling may require manual torque command.", e.what());
        rl_model_.reset();
      }
    }
  }
#endif

  // Pre-size / pre-initialize all command buffers.
  // (non-RT configure phase — heap alloc acceptable here.)
  {
    const std::vector<double> zeros(joint_count_, 0.0);
    qdot_des_buf_.writeFromNonRT(JointVelRef{zeros, 0});
    q_des_buf_.writeFromNonRT(JointPosRef{zeros, 0});
  }
  // Cartesian velocity buffer: zero linear/angular.
  xdot_des_buf_.writeFromNonRT(EEVelRef{{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, 0});

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

  // Task gains service — scalar kp/kd per task name (broadcast per task dim).
  task_gain_srv_ = get_node()->create_service<wbc_msgs::srv::TaskGainService>(
    "~/set_task_gains",
    [this](const wbc_msgs::srv::TaskGainService::Request::SharedPtr req,
           wbc_msgs::srv::TaskGainService::Response::SharedPtr res) {
      if (req->task_names.empty()) {
        res->success = false;
        res->message = "task_names is empty";
        return;
      }
      if (req->kp.size() != req->task_names.size() ||
          req->kd.size() != req->task_names.size()) {
        res->success = false;
        res->message = "kp/kd size must match task_names size";
        return;
      }

      auto* reg = ctrl_arch_->GetConfig()->taskRegistry();
      for (const auto& name : req->task_names) {
        if (reg->GetMotionTask(name) == nullptr) {
          res->success = false;
          res->message = "unknown motion task: " + name;
          return;
        }
      }

      TaskGainUpdate upd;
      upd.task_names = req->task_names;
      upd.kp = req->kp;
      upd.kd = req->kd;
      upd.ts_ns = get_node()->now().nanoseconds();
      task_gain_update_buf_.writeFromNonRT(upd);

      res->success = true;
      res->message = "task gain update latched";
    });

  // Task weights service — scalar weight per task name (broadcast per task dim).
  task_weight_srv_ = get_node()->create_service<wbc_msgs::srv::TaskWeightService>(
    "~/set_task_weights",
    [this](const wbc_msgs::srv::TaskWeightService::Request::SharedPtr req,
           wbc_msgs::srv::TaskWeightService::Response::SharedPtr res) {
      if (req->task_names.empty()) {
        res->success = false;
        res->message = "task_names is empty";
        return;
      }
      if (req->weight.size() != req->task_names.size()) {
        res->success = false;
        res->message = "weight size must match task_names size";
        return;
      }

      auto* reg = ctrl_arch_->GetConfig()->taskRegistry();
      for (const auto& name : req->task_names) {
        if (reg->GetMotionTask(name) == nullptr) {
          res->success = false;
          res->message = "unknown motion task: " + name;
          return;
        }
      }

      TaskWeightUpdate upd;
      upd.task_names = req->task_names;
      upd.weight = req->weight;
      upd.ts_ns = get_node()->now().nanoseconds();
      task_weight_update_buf_.writeFromNonRT(upd);

      res->success = true;
      res->message = "task weight update latched";
    });

  // Residual dynamics service — friction/observer parameter update.
  residual_dyn_srv_ =
    get_node()->create_service<wbc_msgs::srv::ResidualDynamicsService>(
      "~/set_residual_dynamics",
      [this](const wbc_msgs::srv::ResidualDynamicsService::Request::SharedPtr req,
             wbc_msgs::srv::ResidualDynamicsService::Response::SharedPtr res) {
        const auto valid = [this](const std::vector<double>& v) {
          return v.size() == 1 || v.size() == joint_count_;
        };

        if (req->friction_enabled) {
          if (!valid(req->gamma_c) || !valid(req->gamma_v) ||
              !valid(req->max_f_c) || !valid(req->max_f_v)) {
            res->success = false;
            res->message = "friction vectors must be size 1 or num_joints";
            return;
          }
        }
        if (req->observer_enabled) {
          if (!valid(req->k_o) || !valid(req->max_tau_dist)) {
            res->success = false;
            res->message = "observer vectors must be size 1 or num_joints";
            return;
          }
        }

        ResidualDynamicsUpdate upd;
        upd.friction_enabled = req->friction_enabled;
        upd.gamma_c = req->gamma_c;
        upd.gamma_v = req->gamma_v;
        upd.max_f_c = req->max_f_c;
        upd.max_f_v = req->max_f_v;
        upd.observer_enabled = req->observer_enabled;
        upd.k_o = req->k_o;
        upd.max_tau_dist = req->max_tau_dist;
        upd.ts_ns = get_node()->now().nanoseconds();
        residual_update_buf_.writeFromNonRT(upd);

        res->success = true;
        res->message = "residual dynamics update latched";
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

  // Pre-populate tuned maps and msg.tasks using task registry (non-RT configure phase).
  // Goals:
  //   1. Maps get all valid task keys inserted now → runtime updates are guaranteed
  //      lookups (no new key insertion, no heap alloc in RT loop).
  //   2. msg.tasks inner vectors are pre-allocated to max_dim → PublishWbcState's
  //      copy() lambda never allocates (dst capacity ≥ actual task dim always).
  //   3. reapply_scratch_ is pre-sized to max task dim → ReapplyTunedTaskParams
  //      avoids per-call VectorXd::Constant heap allocation.
  {
    auto* reg = ctrl_arch_->GetConfig()->taskRegistry();
    if (reg) {
      int max_dim = 0;
      for (const auto& [name, task] : reg->GetMotionTasks()) {
        // NaN sentinel: "not yet set by user". ReapplyTunedTaskParams skips NaN entries.
        tuned_task_kp_[name]     = std::nan("");
        tuned_task_kd_[name]     = std::nan("");
        tuned_task_weight_[name] = std::nan("");
        max_dim = std::max(max_dim, task->Dim());
      }
      reapply_scratch_.resize(max_dim);

      if (rt_wbc_pub_) {
        // Pre-allocate msg.tasks inner vectors to max_dim (not per-task dim) because
        // the logger fills tasks in WbcFormulation iteration order (operational_tasks
        // then posture_tasks), which differs from unordered_map iteration order here.
        // Using max_dim guarantees copy() in PublishWbcState never truncates regardless
        // of which task lands at which slot. Subscribers must use td.dim for valid count.
        auto& tasks_msg = rt_wbc_pub_->msg_.tasks;
        tasks_msg.resize(reg->GetMotionTasks().size());
        for (auto& td : tasks_msg) {
          td.x_des.resize(max_dim, 0.0);
          td.xdot_des.resize(max_dim, 0.0);
          td.x_curr.resize(max_dim, 0.0);
          td.x_err.resize(max_dim, 0.0);
          td.op_cmd.resize(max_dim, 0.0);
          td.kp.resize(max_dim, 0.0);
          td.kd.resize(max_dim, 0.0);
          td.weight.resize(max_dim, 0.0);
        }
      }
    }
  }

  return controller_interface::CallbackReturn::SUCCESS;
}

////////////////////////////////////////////////////////////////////////

controller_interface::CallbackReturn
OptimoController::on_activate(const rclcpp_lifecycle::State & /*previous_state*/)
{
  const std::size_t expected_state_interfaces  = joint_count_ * kInterfacesPerJoint;
  const std::size_t expected_command_interfaces = joint_count_ * kInterfacesPerJoint + 1; // +1 for model_safety_error
  if (state_interfaces_.size() < expected_state_interfaces ||
    command_interfaces_.size() < expected_command_interfaces)
  {
    RCLCPP_ERROR(
      get_node()->get_logger(),
      "[OptimoController] missing interfaces. expected state>=%zu command>=%zu (got state=%zu command=%zu)",
      expected_state_interfaces, expected_command_interfaces,
      state_interfaces_.size(), command_interfaces_.size());
    return controller_interface::CallbackReturn::ERROR;
  }

  robot_joint_state_.Reset(static_cast<Eigen::Index>(joint_count_));

  // Read initial joint positions and reset actuator state (zero spring deflection).
  {
    Eigen::VectorXd q0(joint_count_);
    for (std::size_t i = 0; i < joint_count_; ++i) {
      q0[static_cast<Eigen::Index>(i)] =
        state_interfaces_[InterfaceIndex(kPositionBlock, i, joint_count_)].get_value();
    }
    actuator_->Reset(q0);
  }

  // Position: hold current hardware positions (not zero — that could swing joints).
  // Velocity and effort: zero. model_safety_error: 0 (not in safe state on activate).
  // Layout: [0, n) = position, [n, 2n) = velocity, [2n, 3n) = effort, [3n] = safety.
  for (std::size_t i = 0; i < joint_count_; ++i) {
    (void)command_interfaces_[i].set_value(state_interfaces_[i].get_value());
  }
  for (std::size_t i = joint_count_; i < command_interfaces_.size(); ++i) {
    (void)command_interfaces_[i].set_value(0.0);
  }
  (void)command_interfaces_[ModelSafetyErrorCmdIndex()].set_value(0.0);

  // Sync active_state_id_ before the first update() tick.
  // Without this, the first tick dispatches against the header-default (-1),
  // which mismatches joint_teleop/cartesian_teleop IDs and causes a missed dispatch.
  active_state_id_ = ctrl_arch_->GetCurrentStateId();
  last_tuned_state_id_ = active_state_id_;

  return controller_interface::CallbackReturn::SUCCESS;
}

////////////////////////////////////////////////////////////////////////

controller_interface::CallbackReturn
OptimoController::on_deactivate(const rclcpp_lifecycle::State & /*previous_state*/)
{
  // Keep last position command (do not zero — position=0 rad can swing joints).
  // Zero velocity, effort, and safety error signal.
  for (std::size_t i = joint_count_; i < command_interfaces_.size(); ++i) {
    (void)command_interfaces_[i].set_value(0.0);
  }
  (void)command_interfaces_[ModelSafetyErrorCmdIndex()].set_value(0.0);
  return controller_interface::CallbackReturn::SUCCESS;
}

////////////////////////////////////////////////////////////////////////

controller_interface::return_type OptimoController::update(
  const rclcpp::Time & time, const rclcpp::Duration & period)
{
  if (!ctrl_arch_) return controller_interface::return_type::OK;

  // Consume pending non-RT service updates on the control thread.
  ApplyPendingRuntimeUpdates();

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
    cartesian_teleop_state_->UpdateCommand(
      xdot_des->xdot, xdot_des->wdot, xdot_des->ts_ns);
  }

  ctrl_arch_->Update(ReadJointState(), time.seconds(), control_dt_);
  // Refresh after FSM ran so auto-transitions are captured for the next tick.
  active_state_id_ = ctrl_arch_->GetCurrentStateId();

  // Write command to hardware interfaces.
  // Layout: [pos x n] [vel x n] [effort x n] [model_safety_error]
  //
  // Command ordering matches command_interface_configuration():
  // - [0 .. n-1]       : desired joint position
  // - [n .. 2n-1]      : desired joint velocity
  // - [2n .. 3n-1]     : desired joint torque
  // - [3n]             : model_safety_error (written below, after WriteJointCommand)

  auto cmd = ctrl_arch_->GetCommand();
  {
    wbc::ActuatorCommand act_cmd;
    act_cmd.q_des = cmd.q;
    act_cmd.qdot_des = cmd.qdot;
    act_cmd.tau_ff = cmd.tau;
    act_cmd.q_link = robot_joint_state_.q;
    act_cmd.qdot_link = robot_joint_state_.qdot;
    act_cmd.dt = control_dt_;
    cmd.tau = actuator_->ProcessTorque(act_cmd);
  }

#ifdef HAS_ROBOLIGENT_SDK
  // Replicate the original effort controller's idle behavior exactly:
  //   torque = get_joint_fallback_impedance_torque() + get_model_torque() (gravity)
  // This is what AbstractCallback::get_torque() does with IdleCallback.
  if (rl_model_) {
    // Convert joint state: ROS (rad, Nm) → roboligent (deg, mNm).
    for (std::size_t i = 0; i < joint_count_; ++i) {
      rl_pos_deg_[i] = robot_joint_state_.q[i] * roboligent::RAD2DEG;
      rl_vel_deg_[i] = robot_joint_state_.qdot[i] * roboligent::RAD2DEG;
    }
    // Pass previous torque command (rl_trq_ref_) as 4th arg, matching original controller.
    rl_model_->update(rl_pos_deg_, rl_vel_deg_,
                      std::vector<double>(joint_count_, 0.0), rl_trq_ref_);

    // First-run setup — matches original effort controller exactly.
    if (rl_model_first_run_) {
      rl_model_first_run_ = false;
      rl_model_->reset_fallback();
      rl_model_->enable_fallback_defaults(true);
      rl_model_->enable_fallback_dragging(true);
      rl_model_->enable_elbow_fallback_dragging(true);
      rl_model_->enable_j1j7_sg_compensation(false);
      rl_model_->enable_j2_sg_compensation(false);
      rl_model_->enable_j4_sg_compensation(false);
      rl_model_->enable_j6_sg_compensation(false);
      rl_model_->set_angle_j6_sg_compensation(0);
    }

    // IdleCallback::calculate_torque = get_joint_fallback_impedance_torque()
    // + AbstractCallback::safety_gravity_compensation = get_model_torque()
    const auto impedance_torque = rl_model_->get_joint_fallback_impedance_torque();
    const auto model_torque = rl_model_->get_model_torque();
    for (std::size_t i = 0; i < joint_count_ && i < model_torque.size(); ++i) {
      rl_trq_ref_[i] = impedance_torque[i] + model_torque[i];
      cmd.tau[i] = static_cast<double>(rl_trq_ref_[i]) * roboligent::MILLI2UNIT;
    }

  }
#endif

  WriteJointCommand(cmd);

  // Signal hardware to disable if FSM entered safe_command OR roboligent model detected error.
  {
    double safety_error = 0.0;
    if (safe_command_state_id_.has_value() &&
        active_state_id_ == *safe_command_state_id_) {
      safety_error = 1.0;
    }
#ifdef HAS_ROBOLIGENT_SDK
    if (rl_model_ && rl_model_->get_safety_error()) {
      safety_error = 1.0;
    }
#endif
    (void)command_interfaces_[ModelSafetyErrorCmdIndex()].set_value(safety_error);
  }

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
      const std::size_t safe_count =
          std::min({count, dst.size(), s.size()});
      for (std::size_t i = 0; i < safe_count; ++i) {
        dst[i] = s[i];
      }
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

    msg.joint_pos_err_norm = src.joint_pos_err_norm;
    msg.joint_vel_err_norm = src.joint_vel_err_norm;
    msg.joint_pos_err_max = src.joint_pos_err_max;
    msg.joint_vel_err_max = src.joint_vel_err_max;
    msg.tau_fb_norm = src.tau_fb_norm;

    msg.qp_solved = src.qp_solved;
    msg.qp_status = src.qp_status;
    msg.qp_iter = src.qp_iter;
    msg.qp_pri_res = src.qp_pri_res;
    msg.qp_dua_res = src.qp_dua_res;
    msg.qp_obj = src.qp_obj;
    msg.qp_setup_time_us = src.qp_setup_time_us;
    msg.qp_solve_time_us = src.qp_solve_time_us;

    // msg.tasks inner vectors are pre-allocated to max_dim at configure time.
    // name/dim/priority must be copied from src (not pre-populated) because the logger
    // fills tasks in WbcFormulation order, which differs from unordered_map order used
    // during configure. Task names are short (<16 chars) — SSO, no heap alloc on assign.
    const std::size_t n_tasks = std::min(msg.tasks.size(), src.tasks.size());
    for (std::size_t t = 0; t < n_tasks; ++t) {
      const auto& ts = src.tasks[t];
      auto& td = msg.tasks[t];
      td.name       = ts.name;  // needed: configure order ≠ formulation order
      td.dim        = ts.dim;
      td.priority   = ts.priority;
      td.x_err_norm = ts.x_err_norm;
      // Vectors: element-wise copy only (no resize, no realloc if sizes match).
      copy(td.x_des,    ts.x_des,    ts.x_des.size());
      copy(td.xdot_des, ts.xdot_des, ts.xdot_des.size());
      copy(td.x_curr,   ts.x_curr,   ts.x_curr.size());
      copy(td.x_err,    ts.x_err,    ts.x_err.size());
      copy(td.op_cmd,   ts.op_cmd,   ts.op_cmd.size());
      copy(td.kp,       ts.kp,       ts.kp.size());
      copy(td.kd,       ts.kd,       ts.kd.size());
      copy(td.weight,   ts.weight,   ts.weight.size());
    }

    rt_wbc_pub_->unlockAndPublish();
    ctrl_arch_->logger_.ClearNewData();
  }
}

void OptimoController::ApplyPendingRuntimeUpdates()
{
  bool task_update_arrived = false;

  if (const auto* upd = task_gain_update_buf_.readFromRT();
      upd != nullptr && upd->ts_ns > last_task_gain_update_ts_) {
    last_task_gain_update_ts_ = upd->ts_ns;
    for (std::size_t i = 0; i < upd->task_names.size(); ++i) {
      tuned_task_kp_[upd->task_names[i]] = upd->kp[i];
      tuned_task_kd_[upd->task_names[i]] = upd->kd[i];
    }
    task_update_arrived = true;
  }

  if (const auto* upd = task_weight_update_buf_.readFromRT();
      upd != nullptr && upd->ts_ns > last_task_weight_update_ts_) {
    last_task_weight_update_ts_ = upd->ts_ns;
    for (std::size_t i = 0; i < upd->task_names.size(); ++i) {
      tuned_task_weight_[upd->task_names[i]] = upd->weight[i];
    }
    task_update_arrived = true;
  }

  if (const auto* upd = residual_update_buf_.readFromRT();
      upd != nullptr && upd->ts_ns > last_residual_update_ts_) {
    last_residual_update_ts_ = upd->ts_ns;

    auto to_eigen = [](const std::vector<double>& src, double def) -> Eigen::VectorXd {
      if (src.empty()) {
        return Eigen::VectorXd::Constant(1, def);
      }
      Eigen::VectorXd v(static_cast<Eigen::Index>(src.size()));
      for (std::size_t i = 0; i < src.size(); ++i) {
        v[static_cast<Eigen::Index>(i)] = src[i];
      }
      return v;
    };

    wbc::FrictionCompensatorConfig fric;
    fric.enabled = upd->friction_enabled;
    fric.gamma_c = to_eigen(upd->gamma_c, 0.0);
    fric.gamma_v = to_eigen(upd->gamma_v, 0.0);
    fric.max_f_c = to_eigen(upd->max_f_c, 10.0);
    fric.max_f_v = to_eigen(upd->max_f_v, 5.0);

    wbc::MomentumObserverConfig obs;
    obs.enabled = upd->observer_enabled;
    obs.K_o = to_eigen(upd->k_o, 50.0);
    obs.max_tau_uncertainty = to_eigen(upd->max_tau_dist, 50.0);

    // SetResidualDynamicsConfig is called from the RT thread.
    // Service callback already validates sizes, so failure here is a programming error.
    // std::string err omitted — string construction is not RT-safe. Pass nullptr.
    (void)ctrl_arch_->SetResidualDynamicsConfig(fric, obs, nullptr);
  }

  // Reapply tuned task params when a new request arrives, or when state changed.
  if (task_update_arrived || active_state_id_ != last_tuned_state_id_) {
    ReapplyTunedTaskParams();
    last_tuned_state_id_ = active_state_id_;
  }
}

void OptimoController::ReapplyTunedTaskParams()
{
  auto* reg = ctrl_arch_->GetConfig()->taskRegistry();
  if (reg == nullptr) {
    return;
  }

  // NaN sentinel: key pre-inserted at configure time but not yet set by user — skip.
  // reapply_scratch_ is pre-sized to max task dim; resize() is a no-op if dim unchanged.
  for (const auto& [name, kp] : tuned_task_kp_) {
    if (std::isnan(kp)) continue;
    auto* task = reg->GetMotionTask(name);
    if (task == nullptr) continue;
    reapply_scratch_.resize(task->Dim());
    reapply_scratch_.setConstant(kp);
    task->SetKp(reapply_scratch_);
  }
  for (const auto& [name, kd] : tuned_task_kd_) {
    if (std::isnan(kd)) continue;
    auto* task = reg->GetMotionTask(name);
    if (task == nullptr) continue;
    reapply_scratch_.resize(task->Dim());
    reapply_scratch_.setConstant(kd);
    task->SetKd(reapply_scratch_);
  }
  for (const auto& [name, w] : tuned_task_weight_) {
    if (std::isnan(w)) continue;
    auto* task = reg->GetMotionTask(name);
    if (task == nullptr) continue;
    reapply_scratch_.resize(task->Dim());
    reapply_scratch_.setConstant(w);
    task->SetWeight(reapply_scratch_);
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
