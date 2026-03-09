// Copyright 2024 Roboligent, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef OPTIMO_CONTROLLER_ROS_WRAPPER_SRC_OPTIMO_CONTROLLERS_INCLUDE_OPTIMO_CONTROLLERS_OPTIMO_EFFORT_CONTROLLER_H_
#define OPTIMO_CONTROLLER_ROS_WRAPPER_SRC_OPTIMO_CONTROLLERS_INCLUDE_OPTIMO_CONTROLLERS_OPTIMO_EFFORT_CONTROLLER_H_

#include <string>

#include <rl/controller/AbstractCallback.h>
#include <rl/model/Model.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>

#include <controller_interface/controller_interface.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <optimo_msgs/msg/pose_elbow.hpp>
#include <optimo_msgs/srv/generic_cb.hpp>
#include <optimo_msgs/srv/moveit_cb.hpp>
#include <optimo_msgs/srv/play_traj_cb.hpp>
#include <optimo_msgs/srv/pos_cb.hpp>
#include <optimo_msgs/srv/servo_cb.hpp>
#include <optimo_msgs/srv/string_cb.hpp>
#include <rclcpp/duration.hpp>
#include <rclcpp/time.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <std_srvs/srv/trigger.hpp>

#include "optimo_api/MoveItInterface.h"
#include "optimo_api/ROSShared.h"

using CallbackReturn = rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;

namespace optimo_ros
{

/**
 * @brief Optimo's torque (effort) controller, under the ROS control framework.
 *
 */
class OptimoEffortController : public controller_interface::ControllerInterface
{
public:
  /**
   * @brief Does nothing.
   *
   * @return CallbackReturn success
   */
  CallbackReturn on_configure(const rclcpp_lifecycle::State &) override;

  /**
   * @brief Initializes the robot model.
   *
   * @return CallbackReturn Success if the roboligent::model is successfully initialized.
   */
  CallbackReturn on_init() override;

  /**
   * @brief The command interface is joint effort.
   *
   * @return controller_interface::InterfaceConfiguration
   */
  controller_interface::InterfaceConfiguration command_interface_configuration() const override;

  /**
   * @brief The state interfaces are joint position, velocity, and effort.
   *
   * @return controller_interface::InterfaceConfiguration
   */
  controller_interface::InterfaceConfiguration state_interface_configuration() const override;

  /**
   * @brief Populates the command torque using current_task's callback.
   *
   * @return controller_interface::return_type
   */
  controller_interface::return_type update(const rclcpp::Time &, const rclcpp::Duration &) override;

private:
  /**
   * @brief Checks on the current task, and if complete, starts the next one in the queue.
   *
   * @details See optimo::Controller::is_task_configured and update_task
   */
  void update_task();

  /**
   * @brief Removes the completed task from the queue, and resets model's settings to a default.
   *
   */
  void complete_task();

  /**
   * @brief Service Callback for free motion.
   *
   * @param request
   * @param response
   */
  void free_motion_cb(
    const std::shared_ptr<optimo_msgs::srv::GenericCb::Request> request,
    const std::shared_ptr<optimo_msgs::srv::GenericCb::Response> response);

  /**
   * @brief Service Callback for move home.
   *
   * @param request
   * @param response
   */
  void move_home_cb(
    const std::shared_ptr<optimo_msgs::srv::PosCb::Request> request,
    const std::shared_ptr<optimo_msgs::srv::PosCb::Response> response);

  /**
   * @brief Service Callback for stopping motion.
   *
   * @param response
   */
  void stop_cb(
    const std::shared_ptr<std_srvs::srv::Trigger::Request>,
    const std::shared_ptr<std_srvs::srv::Trigger::Response> response);

  /**
   * @brief Service Callback for teaching trajectories.
   *
   * @param request
   * @param response
   */
  void teach_traj_cb(
    const std::shared_ptr<optimo_msgs::srv::StringCb::Request> request,
    const std::shared_ptr<optimo_msgs::srv::StringCb::Response> response);

  /**
   * @brief Service Callback for trajectory playing
   *
   * @param request
   * @param response
   */
  void play_traj_cb(
    const std::shared_ptr<optimo_msgs::srv::PlayTrajCb::Request> request,
    const std::shared_ptr<optimo_msgs::srv::PlayTrajCb::Response> response);

  /**
   * @brief Service Callback for using MoveIt for trajectory planning.
   *
   * @param request
   * @param response
   */
  void moveit_cb(
    const std::shared_ptr<optimo_msgs::srv::MoveitCb::Request> request,
    const std::shared_ptr<optimo_msgs::srv::MoveitCb::Response> response);

  /**
   * @brief Service Callback for using MoveIt Servo.
   *
   * @param request
   * @param response
   */
  void moveit_servo_cb(
    const std::shared_ptr<optimo_msgs::srv::GenericCb::Request> request,
    const std::shared_ptr<optimo_msgs::srv::GenericCb::Response> response);

  /**
   * @brief Service Callback for using servoing without MoveIt.
   *
   * @param request
   * @param response
   */
  void servo_cb(
    const std::shared_ptr<optimo_msgs::srv::ServoCb::Request> request,
    const std::shared_ptr<optimo_msgs::srv::ServoCb::Response> response);

  /**
   * @brief Service Callback for getting force feedback from servoing.
   *
   * @param request
   * @param response
   */
  void servo_fb_cb(
    const std::shared_ptr<optimo_msgs::srv::StringCb::Request> request,
    const std::shared_ptr<optimo_msgs::srv::StringCb::Response> response);

  /**
   * @brief Service Callback for enabling load calculation. Returns false if the load does not
   * change within the 1 second it should be calculated, indicating that something has gone wrong.
   *
   * @param request
   * @param response
   */
  void enable_load_calculation_cb(
    const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
    const std::shared_ptr<std_srvs::srv::Trigger::Response> response);

  /**
   * @brief Helper to add a task to the queue, and blocks until the task is complete.
   *
   * @param task task to queue
   */
  bool queue_task(const std::shared_ptr<optimo::Task> task);

  // Controller adjacent parameters
  std::vector<double> pos;
  std::vector<double> vel;
  std::vector<int> eff;
  bool communication_enabled;
  bool first_run;
  std::shared_ptr<roboligent::Model> model;
  optimo::TaskQueueObject current_task;
  optimo::TaskQueue task_queue;
  std::shared_ptr<MoveItInterface> moveit_iface;
  std::vector<int> torque_command;
  std::map<int, std::unique_ptr<roboligent::AbstractCallback>> cb_list;
  int robot_index;
  std::string robot_prefix;

  // Tools for running services concurrently
  std::mutex task_mtx;
  std::shared_ptr<rclcpp::Executor> exec;
  std::thread exec_thread;
  rclcpp::CallbackGroup::SharedPtr cb_group;

  // service for swapping callbacks
  rclcpp::Service<optimo_msgs::srv::GenericCb>::SharedPtr free_motion_srv;
  rclcpp::Service<optimo_msgs::srv::PosCb>::SharedPtr move_home_srv;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr stop_srv;
  rclcpp::Service<optimo_msgs::srv::StringCb>::SharedPtr teach_traj_srv;
  rclcpp::Service<optimo_msgs::srv::PlayTrajCb>::SharedPtr play_traj_srv;
  rclcpp::Service<optimo_msgs::srv::MoveitCb>::SharedPtr moveit_srv;
  rclcpp::Service<optimo_msgs::srv::GenericCb>::SharedPtr moveit_servo_srv;
  rclcpp::Service<optimo_msgs::srv::ServoCb>::SharedPtr servo_srv;
  rclcpp::Service<optimo_msgs::srv::StringCb>::SharedPtr servo_fb_srv;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr enable_load_calculation_srv;

  Eigen::VectorXd measured_torque_;  // [Nm] measured from the effort interface
  std::vector<int> effort_idx_;      // cached indices into state_interfaces_

  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
};
}  // namespace optimo_ros

#endif  // OPTIMO_CONTROLLER_ROS_WRAPPER_SRC_OPTIMO_CONTROLLERS_INCLUDE_OPTIMO_CONTROLLERS_OPTIMO_EFFORT_CONTROLLER_H_