// Copyright 2026
//
// Licensed under the Apache License, Version 2.0.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "ament_index_cpp/get_package_share_directory.hpp"
#include "geometry_msgs/msg/pose.hpp"
#include "geometry_msgs/msg/pose_array.hpp"
#include "mppi_core/config/grasp_config.hpp"
#include "mppi_core/config/mppi_config.hpp"
#include "mppi_core/policies/jenga_grasp.hpp"
#include "mppi_core/tactile/nari_touch_adapter.hpp"
#include "pinocchio/spatial/se3.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sdr_grasp_msgs/msg/tactile.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "wbc_core/robots/robot-system.hpp"
#include "wbc_msgs/msg/impedance_commands.hpp"

namespace {

using namespace std::chrono_literals;

std::vector<std::string> DefaultCommandJointNames() {
  return {"joint1", "joint2", "joint3", "joint4",
          "joint5", "joint6", "joint7", "joint8"};
}

std::vector<std::string> DefaultControlledJointNames() {
  return {"joint3", "joint4", "joint5", "joint6"};
}

std::vector<std::string> DefaultTactileFrameNames(std::size_t count) {
  std::vector<std::string> names = {
      "thumb_distal_tactile", "index_distal_tactile", "middle_distal_tactile"};
  names.resize(count);
  return names;
}

std::string DefaultUrdfPath() {
  return "package://plato_description/urdf/plato_naritouch.urdf";
}

std::vector<std::filesystem::path> PackageShareCandidates(
    const std::string& package_name) {
  std::vector<std::filesystem::path> candidates;
  const char* prefixes = std::getenv("AMENT_PREFIX_PATH");
  if (prefixes != nullptr) {
    std::string remaining(prefixes);
    while (!remaining.empty()) {
      const std::size_t separator = remaining.find(':');
      const std::string prefix = remaining.substr(
          0, separator == std::string::npos ? std::string::npos : separator);
      if (!prefix.empty()) {
        const std::filesystem::path prefix_path(prefix);
        const auto marker = prefix_path / "share" / "ament_index" /
                            "resource_index" / "packages" / package_name;
        const auto share_dir = prefix_path / "share" / package_name;
        if (std::filesystem::exists(marker) &&
            std::filesystem::exists(share_dir)) {
          candidates.push_back(share_dir);
        }
      }
      if (separator == std::string::npos) {
        break;
      }
      remaining.erase(0, separator + 1);
    }
  }

  const std::filesystem::path indexed_share =
      ament_index_cpp::get_package_share_directory(package_name);
  if (std::find(candidates.begin(), candidates.end(), indexed_share) ==
      candidates.end()) {
    candidates.push_back(indexed_share);
  }
  return candidates;
}

std::string ResolvePackageUri(const std::string& uri) {
  constexpr const char kPackagePrefix[] = "package://";
  if (uri.rfind(kPackagePrefix, 0) != 0U) {
    return uri;
  }

  const std::string suffix = uri.substr(sizeof(kPackagePrefix) - 1U);
  const std::size_t slash = suffix.find('/');
  const std::string package_name =
      (slash == std::string::npos) ? suffix : suffix.substr(0, slash);
  const std::string relative =
      (slash == std::string::npos) ? std::string() : suffix.substr(slash + 1U);
  if (package_name.empty()) {
    throw std::runtime_error("Invalid package URI: " + uri);
  }

  const auto candidates = PackageShareCandidates(package_name);
  if (relative.empty()) {
    return candidates.front().string();
  }

  for (const auto& share_dir : candidates) {
    const auto resolved = share_dir / relative;
    if (std::filesystem::exists(resolved)) {
      return resolved.string();
    }
  }

  throw std::runtime_error("Package URI target does not exist: " + uri);
}

std::vector<double> ToStdVector(const Eigen::VectorXd& value) {
  std::vector<double> out(static_cast<std::size_t>(value.size()), 0.0);
  for (Eigen::Index i = 0; i < value.size(); ++i) {
    out[static_cast<std::size_t>(i)] = value[i];
  }
  return out;
}

Eigen::Vector3d ClampVectorNorm(const Eigen::Vector3d& value, double max_norm) {
  if (!value.allFinite()) {
    return Eigen::Vector3d::Zero();
  }
  if (!std::isfinite(max_norm) || max_norm <= 0.0) {
    return value;
  }

  const double norm = value.norm();
  if (norm <= max_norm || norm <= 0.0) {
    return value;
  }
  return value * (max_norm / norm);
}

Eigen::Vector2d ClampVectorNorm(const Eigen::Vector2d& value, double max_norm) {
  if (!value.allFinite()) {
    return Eigen::Vector2d::Zero();
  }
  if (!std::isfinite(max_norm) || max_norm <= 0.0) {
    return value;
  }

  const double norm = value.norm();
  if (norm <= max_norm || norm <= 0.0) {
    return value;
  }
  return value * (max_norm / norm);
}

Eigen::VectorXd ReadVectorParameter(rclcpp::Node* node, const std::string& name,
                                    std::size_t size, double fill_value,
                                    bool allow_empty = false) {
  const std::vector<double> default_values;
  const auto values =
      node->declare_parameter<std::vector<double>>(name, default_values);
  if (values.empty()) {
    if (allow_empty) {
      return Eigen::VectorXd();
    }
    return Eigen::VectorXd::Constant(static_cast<Eigen::Index>(size),
                                     fill_value);
  }
  if (values.size() == 1 && size > 1) {
    return Eigen::VectorXd::Constant(static_cast<Eigen::Index>(size),
                                     values.front());
  }
  if (values.size() != size) {
    throw std::invalid_argument("Parameter '" + name +
                                "' must be empty, scalar, or joint-sized");
  }

  Eigen::VectorXd out(static_cast<Eigen::Index>(size));
  for (std::size_t i = 0; i < size; ++i) {
    out[static_cast<Eigen::Index>(i)] = values[i];
  }
  return out;
}

Eigen::VectorXd ReadVectorParameter(rclcpp::Node* node, const std::string& name,
                                    std::size_t size,
                                    const Eigen::VectorXd& default_value) {
  if (default_value.size() != 0 &&
      default_value.size() != static_cast<Eigen::Index>(size)) {
    throw std::invalid_argument("Default for parameter '" + name +
                                "' has wrong size");
  }

  const auto values = node->declare_parameter<std::vector<double>>(
      name, ToStdVector(default_value));
  if (values.empty()) {
    return default_value;
  }
  if (values.size() == 1 && size > 1) {
    return Eigen::VectorXd::Constant(static_cast<Eigen::Index>(size),
                                     values.front());
  }
  if (values.size() != size) {
    throw std::invalid_argument("Parameter '" + name +
                                "' must be scalar or joint-sized");
  }

  Eigen::VectorXd out(static_cast<Eigen::Index>(size));
  for (std::size_t i = 0; i < size; ++i) {
    out[static_cast<Eigen::Index>(i)] = values[i];
  }
  return out;
}

std::size_t FindJointIndex(const sensor_msgs::msg::JointState& msg,
                           const std::string& joint_name) {
  const auto it = std::find(msg.name.begin(), msg.name.end(), joint_name);
  if (it == msg.name.end()) {
    return msg.name.size();
  }
  return static_cast<std::size_t>(std::distance(msg.name.begin(), it));
}

std::size_t FindNameIndex(const std::vector<std::string>& names,
                          const std::string& name) {
  const auto it = std::find(names.begin(), names.end(), name);
  if (it == names.end()) {
    return names.size();
  }
  return static_cast<std::size_t>(std::distance(names.begin(), it));
}

std::vector<std::size_t> BuildNameIndexMap(
    const std::vector<std::string>& subset,
    const std::vector<std::string>& full_set, const char* subset_label,
    const char* full_set_label) {
  std::vector<std::size_t> indices;
  indices.reserve(subset.size());
  for (const auto& name : subset) {
    const std::size_t index = FindNameIndex(full_set, name);
    if (index >= full_set.size()) {
      throw std::invalid_argument(std::string("'") + subset_label +
                                  "' entry '" + name + "' is not in '" +
                                  full_set_label + "'");
    }
    indices.push_back(index);
  }
  return indices;
}

mppi_core::NariTouchContactState ConvertContactState(int state) {
  switch (state) {
    case sdr_grasp_msgs::msg::Tactile::ENOUGH_CONTACTS:
      return mppi_core::NariTouchContactState::kEnoughContacts;
    case sdr_grasp_msgs::msg::Tactile::FEW_CONTACTS:
      return mppi_core::NariTouchContactState::kFewContacts;
    default:
      return mppi_core::NariTouchContactState::kNoContact;
  }
}

geometry_msgs::msg::Pose ToPoseMsg(const pinocchio::SE3& placement) {
  geometry_msgs::msg::Pose pose;
  pose.position.x = placement.translation().x();
  pose.position.y = placement.translation().y();
  pose.position.z = placement.translation().z();

  Eigen::Quaterniond quat(placement.rotation());
  quat.normalize();
  pose.orientation.x = quat.x();
  pose.orientation.y = quat.y();
  pose.orientation.z = quat.z();
  pose.orientation.w = quat.w();
  return pose;
}

mppi_core::NariTouchState ConvertTactile(
    const sdr_grasp_msgs::msg::Tactile& msg, double cop_to_m_scale) {
  mppi_core::NariTouchState tactile;
  tactile.slip_state =
      Eigen::Vector3d{msg.shear_displacement.x, msg.shear_displacement.y,
                      msg.shear_displacement.theta};
  tactile.force_z = msg.force.z;
  tactile.contact_state = ConvertContactState(msg.contact_state);

  const std::size_t count = std::min(tactile.units.size(), msg.units.size());
  for (std::size_t i = 0; i < count; ++i) {
    const auto& unit = msg.units[i];
    auto& state_unit = tactile.units[i];
    state_unit.contact = unit.contact;
    state_unit.cop = Eigen::Vector2d{unit.cop.x * cop_to_m_scale,
                                     unit.cop.y * cop_to_m_scale};
    state_unit.normal_force = unit.normal_force;
  }
  return tactile;
}

}  // namespace

class JengaGraspMppiNode final : public rclcpp::Node {
 public:
  JengaGraspMppiNode() : Node("jenga_grasp_mppi_node") {
    command_joint_names_ = declare_parameter<std::vector<std::string>>(
        "command_joint_names", DefaultCommandJointNames());
    if (command_joint_names_.empty()) {
      throw std::invalid_argument("'command_joint_names' parameter is empty");
    }
    command_joint_dim_ = command_joint_names_.size();

    joint_names_ = declare_parameter<std::vector<std::string>>(
        "joint_names", DefaultControlledJointNames());
    if (joint_names_.empty()) {
      throw std::invalid_argument("'joint_names' parameter is empty");
    }
    joint_dim_ = joint_names_.size();
    controlled_command_indices_ =
        BuildNameIndexMap(joint_names_, command_joint_names_, "joint_names",
                          "command_joint_names");

    const auto fixed_joint_names = declare_parameter<std::vector<std::string>>(
        "fixed_joint_names", std::vector<std::string>{});
    const auto fixed_joint_positions = declare_parameter<std::vector<double>>(
        "fixed_joint_positions", std::vector<double>{});
    if (fixed_joint_names.size() != fixed_joint_positions.size()) {
      throw std::invalid_argument(
          "'fixed_joint_names' and 'fixed_joint_positions' must have the same "
          "size");
    }
    const auto fixed_joint_command_indices =
        BuildNameIndexMap(fixed_joint_names, command_joint_names_,
                          "fixed_joint_names", "command_joint_names");
    fixed_joint_positions_.setConstant(
        static_cast<Eigen::Index>(command_joint_dim_),
        std::numeric_limits<double>::quiet_NaN());
    for (std::size_t i = 0; i < fixed_joint_command_indices.size(); ++i) {
      fixed_joint_positions_[static_cast<Eigen::Index>(
          fixed_joint_command_indices[i])] = fixed_joint_positions[i];
    }

    joint_state_topic_ = declare_parameter<std::string>("joint_state_topic",
                                                        "/plato2/joint_states");
    command_topic_ = declare_parameter<std::string>(
        "command_topic", "/plato2/joint_impedance_controller/commands");
    tactile_topics_ = declare_parameter<std::vector<std::string>>(
        "tactile_topics",
        {"tactile_0/tactile_states", "tactile_1/tactile_states"});
    if (tactile_topics_.empty()) {
      throw std::invalid_argument("'tactile_topics' parameter is empty");
    }
    tactile_frame_names_ = declare_parameter<std::vector<std::string>>(
        "tactile_frame_names",
        DefaultTactileFrameNames(tactile_topics_.size()));
    if (tactile_frame_names_.size() != tactile_topics_.size()) {
      throw std::invalid_argument(
          "'tactile_frame_names' must match 'tactile_topics' size");
    }

    control_rate_hz_ = declare_parameter<double>("control_rate_hz", 100.0);
    tactile_timeout_s_ = declare_parameter<double>("tactile_timeout_s", 0.25);
    joint_state_timeout_s_ =
        declare_parameter<double>("joint_state_timeout_s", 0.25);
    cop_to_m_scale_ = declare_parameter<double>("cop_to_m_scale", 1.0e-3);
    slip_velocity_filter_alpha_ =
        declare_parameter<double>("slip_velocity_filter_alpha", 0.25);
    slip_velocity_max_norm_ =
        declare_parameter<double>("slip_velocity_max_norm", 100.0);
    centroid_velocity_filter_alpha_ =
        declare_parameter<double>("centroid_velocity_filter_alpha", 0.25);
    centroid_velocity_max_norm_mps_ =
        declare_parameter<double>("centroid_velocity_max_norm_mps", 0.2);
    if (slip_velocity_filter_alpha_ < 0.0 ||
        slip_velocity_filter_alpha_ > 1.0) {
      throw std::invalid_argument(
          "'slip_velocity_filter_alpha' must be in [0, 1]");
    }
    if (centroid_velocity_filter_alpha_ < 0.0 ||
        centroid_velocity_filter_alpha_ > 1.0) {
      throw std::invalid_argument(
          "'centroid_velocity_filter_alpha' must be in [0, 1]");
    }
    if (!std::isfinite(slip_velocity_max_norm_) ||
        slip_velocity_max_norm_ <= 0.0) {
      throw std::invalid_argument("'slip_velocity_max_norm' must be positive");
    }
    if (!std::isfinite(centroid_velocity_max_norm_mps_) ||
        centroid_velocity_max_norm_mps_ <= 0.0) {
      throw std::invalid_argument(
          "'centroid_velocity_max_norm_mps' must be positive");
    }
    publish_hold_without_tactile_ =
        declare_parameter<bool>("publish_hold_without_tactile", true);
    activation_requires_all_tactile_enough_contact_ = declare_parameter<bool>(
        "activation_requires_all_tactile_enough_contact", true);
    activation_contact_state_threshold_ =
        declare_parameter<int>("activation_contact_state_threshold", 2);
    if (activation_contact_state_threshold_ < 0 ||
        activation_contact_state_threshold_ >
            static_cast<int>(
                mppi_core::NariTouchContactState::kEnoughContacts)) {
      throw std::invalid_argument(
          "'activation_contact_state_threshold' must be in [0, 2]");
    }

    stiffness_ =
        ReadVectorParameter(this, "stiffness", command_joint_dim_, 4.0);
    damping_ = ReadVectorParameter(this, "damping", command_joint_dim_, 1.0);
    effort_ff_ =
        ReadVectorParameter(this, "effort_ff", command_joint_dim_, 0.0);
    q_lower_bound_ =
        ReadVectorParameter(this, "q_lower_bound", joint_dim_,
                            -std::numeric_limits<double>::infinity(), true);
    q_upper_bound_ =
        ReadVectorParameter(this, "q_upper_bound", joint_dim_,
                            std::numeric_limits<double>::infinity(), true);

    q_measured_.setZero(static_cast<Eigen::Index>(joint_dim_));
    v_measured_.setZero(static_cast<Eigen::Index>(joint_dim_));
    tau_measured_.setZero(static_cast<Eigen::Index>(joint_dim_));
    q_ref_.setZero(static_cast<Eigen::Index>(joint_dim_));
    command_q_measured_.setZero(static_cast<Eigen::Index>(command_joint_dim_));
    command_v_measured_.setZero(static_cast<Eigen::Index>(command_joint_dim_));
    command_tau_measured_.setZero(
        static_cast<Eigen::Index>(command_joint_dim_));
    command_q_ref_.setZero(static_cast<Eigen::Index>(command_joint_dim_));

    InitializeRobotSystem();
    InitializePolicy();

    tactile_states_.resize(tactile_topics_.size());
    tactile_received_.assign(tactile_topics_.size(), false);
    last_tactile_time_.assign(tactile_topics_.size(), rclcpp::Time(0, 0));
    previous_tactile_slip_states_.assign(tactile_topics_.size(),
                                         Eigen::Vector3d::Zero());
    filtered_slip_velocity_states_.assign(tactile_topics_.size(),
                                          Eigen::Vector3d::Zero());
    previous_tactile_centroids_.assign(tactile_topics_.size(),
                                       Eigen::Vector2d::Zero());
    filtered_centroid_velocity_mps_.assign(tactile_topics_.size(),
                                           Eigen::Vector2d::Zero());
    previous_tactile_sample_time_.assign(tactile_topics_.size(),
                                         rclcpp::Time(0, 0));
    slip_velocity_initialized_.assign(tactile_topics_.size(), false);
    centroid_velocity_initialized_.assign(tactile_topics_.size(), false);

    joint_state_sub_ = create_subscription<sensor_msgs::msg::JointState>(
        joint_state_topic_, rclcpp::SensorDataQoS(),
        std::bind(&JengaGraspMppiNode::JointStateCallback, this,
                  std::placeholders::_1));

    tactile_subs_.reserve(tactile_topics_.size());
    for (std::size_t i = 0; i < tactile_topics_.size(); ++i) {
      tactile_subs_.push_back(create_subscription<sdr_grasp_msgs::msg::Tactile>(
          tactile_topics_[i], rclcpp::SensorDataQoS(),
          [this, i](const sdr_grasp_msgs::msg::Tactile::SharedPtr msg) {
            TactileCallback(i, msg);
          }));
    }

    rclcpp::QoS command_qos(rclcpp::KeepLast(10));
    command_qos.durability(RMW_QOS_POLICY_DURABILITY_VOLATILE);
    command_pub_ = create_publisher<wbc_msgs::msg::ImpedanceCommands>(
        command_topic_, command_qos);

    const double safe_rate_hz = std::max(1.0, control_rate_hz_);
    update_timer_ = create_wall_timer(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::duration<double>(1.0 / safe_rate_hz)),
        std::bind(&JengaGraspMppiNode::Update, this));

    RCLCPP_INFO(get_logger(),
                "Started MPPI -> impedance pipeline: joint_state='%s', "
                "command='%s', controlled_joints=%zu/%zu, tactile_topics=%zu, "
                "rate=%.1f Hz",
                joint_state_topic_.c_str(), command_topic_.c_str(), joint_dim_,
                command_joint_dim_, tactile_topics_.size(), safe_rate_hz);
  }

 private:
  void InitializeRobotSystem() {
    urdf_path_ = declare_parameter<std::string>("urdf_path", DefaultUrdfPath());
    publish_tactile_frame_poses_ =
        declare_parameter<bool>("publish_tactile_frame_poses", true);
    tactile_frame_pose_topic_ = declare_parameter<std::string>(
        "tactile_frame_pose_topic", "~/tactile_frame_poses");
    tactile_pose_frame_id_ =
        declare_parameter<std::string>("tactile_pose_frame_id", "base_link");

    if (urdf_path_.empty()) {
      RCLCPP_WARN(
          get_logger(),
          "RobotSystem tactile frame tracking disabled: set 'urdf_path' to "
          "resolve tactile frames from URDF");
      return;
    }

    resolved_urdf_path_ = ResolvePackageUri(urdf_path_);
    robot_system_ = std::make_unique<wbc::robots::RobotSystem>(
        resolved_urdf_path_, std::vector<std::string>{}, false);
    robot_data_ = std::make_unique<wbc::robots::RobotSystem::Data>(
        robot_system_->model());

    robot_q_ = robot_system_->q();
    robot_v_ = Eigen::VectorXd::Zero(robot_system_->nv());
    command_joint_q_indices_.resize(command_joint_names_.size());

    for (std::size_t i = 0; i < command_joint_names_.size(); ++i) {
      const auto joint_id =
          robot_system_->model().getJointId(command_joint_names_[i]);
      if (joint_id >= robot_system_->model().joints.size()) {
        throw std::invalid_argument("URDF does not contain joint '" +
                                    command_joint_names_[i] + "'");
      }
      const auto& joint = robot_system_->model().joints[joint_id];
      if (joint.nq() != 1) {
        throw std::invalid_argument("Joint '" + command_joint_names_[i] +
                                    "' is not a scalar joint in Pinocchio");
      }
      command_joint_q_indices_[i] = joint.idx_q();
    }

    tactile_frame_ids_.resize(tactile_frame_names_.size());
    tactile_frame_placements_.resize(tactile_frame_names_.size(),
                                     pinocchio::SE3::Identity());
    tactile_frame_pose_valid_.assign(tactile_frame_names_.size(), false);
    for (std::size_t i = 0; i < tactile_frame_names_.size(); ++i) {
      if (tactile_frame_names_[i].empty()) {
        throw std::invalid_argument("Empty tactile frame name at index " +
                                    std::to_string(i));
      }

      const auto frame_id =
          robot_system_->model().getFrameId(tactile_frame_names_[i]);
      if (frame_id >= robot_system_->model().frames.size()) {
        throw std::invalid_argument("URDF does not contain tactile frame '" +
                                    tactile_frame_names_[i] + "'");
      }
      tactile_frame_ids_[i] = frame_id;
      RCLCPP_INFO(get_logger(), "Mapped tactile topic '%s' to URDF frame '%s'",
                  tactile_topics_[i].c_str(), tactile_frame_names_[i].c_str());
    }

    if (publish_tactile_frame_poses_) {
      tactile_frame_pose_pub_ = create_publisher<geometry_msgs::msg::PoseArray>(
          tactile_frame_pose_topic_, rclcpp::QoS(10));
    }
    robot_system_enabled_ = true;
    RCLCPP_INFO(get_logger(), "Loaded RobotSystem model '%s' with nq=%d, nv=%d",
                resolved_urdf_path_.c_str(), robot_system_->nq(),
                robot_system_->nv());
  }

  void InitializePolicy() {
    auto config = mppi_core::MakeDefaultJengaGraspConfig(joint_dim_);

    std::string default_mppi_yaml;
    try {
      default_mppi_yaml =
          ament_index_cpp::get_package_share_directory("mppi_core") +
          "/config/mppi.yaml";
    } catch (const std::exception&) {
      default_mppi_yaml.clear();
    }

    auto mppi_yaml_path =
        declare_parameter<std::string>("mppi_yaml_path", default_mppi_yaml);
    if (mppi_yaml_path.empty()) {
      mppi_yaml_path = default_mppi_yaml;
    }
    if (!mppi_yaml_path.empty()) {
      config.mppi = mppi_core::LoadMPPIConfigFromYamlFile(
          mppi_yaml_path, joint_dim_, config.mppi);
    }

    config.mppi.horizon_steps = static_cast<std::size_t>(declare_parameter<int>(
        "mppi.horizon_steps", static_cast<int>(config.mppi.horizon_steps)));
    config.mppi.num_rollouts = static_cast<std::size_t>(declare_parameter<int>(
        "mppi.num_rollouts", static_cast<int>(config.mppi.num_rollouts)));
    config.mppi.dt = declare_parameter<double>("mppi.dt", config.mppi.dt);
    config.mppi.temperature =
        declare_parameter<double>("mppi.temperature", config.mppi.temperature);
    config.mppi.random_seed = static_cast<std::uint32_t>(declare_parameter<int>(
        "mppi.random_seed", static_cast<int>(config.mppi.random_seed)));
    config.mppi.action_lower_bound =
        ReadVectorParameter(this, "mppi.action_lower_bound", joint_dim_,
                            config.mppi.action_lower_bound);
    config.mppi.action_upper_bound =
        ReadVectorParameter(this, "mppi.action_upper_bound", joint_dim_,
                            config.mppi.action_upper_bound);
    config.mppi.action_noise_std =
        ReadVectorParameter(this, "mppi.action_noise_std", joint_dim_,
                            config.mppi.action_noise_std);
    if (q_lower_bound_.size() == static_cast<Eigen::Index>(joint_dim_) &&
        q_upper_bound_.size() == static_cast<Eigen::Index>(joint_dim_)) {
      config.grasp_stability_cost.joint_lower_bound = q_lower_bound_;
      config.grasp_stability_cost.joint_upper_bound = q_upper_bound_;
    }

    std::string default_cost_yaml;
    try {
      default_cost_yaml =
          ament_index_cpp::get_package_share_directory("mppi_core") +
          "/config/grasp.yaml";
    } catch (const std::exception&) {
      default_cost_yaml.clear();
    }

    auto cost_yaml_path =
        declare_parameter<std::string>("cost_yaml_path", default_cost_yaml);
    if (cost_yaml_path.empty()) {
      cost_yaml_path = default_cost_yaml;
    }
    if (!cost_yaml_path.empty()) {
      config.grasp_stability_cost = mppi_core::LoadGraspConfigFromYamlFile(
          cost_yaml_path, joint_dim_, config.grasp_stability_cost);
    }

    tactile_adapter_config_.slip_velocity_weight =
        config.grasp_stability_cost.slip_velocity_weight;
    policy_.Initialize(joint_dim_, std::move(config));
  }

  void JointStateCallback(const sensor_msgs::msg::JointState::SharedPtr msg) {
    if (!msg) {
      return;
    }

    Eigen::VectorXd command_q(static_cast<Eigen::Index>(command_joint_dim_));
    Eigen::VectorXd command_v =
        Eigen::VectorXd::Zero(static_cast<Eigen::Index>(command_joint_dim_));
    Eigen::VectorXd command_tau =
        Eigen::VectorXd::Zero(static_cast<Eigen::Index>(command_joint_dim_));

    for (std::size_t i = 0; i < command_joint_dim_; ++i) {
      const std::size_t msg_index =
          FindJointIndex(*msg, command_joint_names_[i]);
      if (msg_index >= msg->name.size() || msg_index >= msg->position.size()) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
                             "JointState is missing joint '%s'",
                             command_joint_names_[i].c_str());
        return;
      }
      command_q[static_cast<Eigen::Index>(i)] = msg->position[msg_index];
      if (msg_index < msg->velocity.size()) {
        command_v[static_cast<Eigen::Index>(i)] = msg->velocity[msg_index];
      }
      if (msg_index < msg->effort.size()) {
        command_tau[static_cast<Eigen::Index>(i)] = msg->effort[msg_index];
      }
    }

    command_q_measured_ = command_q;
    command_v_measured_ = command_v;
    command_tau_measured_ = command_tau;
    Eigen::VectorXd q(static_cast<Eigen::Index>(joint_dim_));
    Eigen::VectorXd v =
        Eigen::VectorXd::Zero(static_cast<Eigen::Index>(joint_dim_));
    Eigen::VectorXd tau =
        Eigen::VectorXd::Zero(static_cast<Eigen::Index>(joint_dim_));
    for (std::size_t i = 0; i < joint_dim_; ++i) {
      const Eigen::Index command_index =
          static_cast<Eigen::Index>(controlled_command_indices_[i]);
      q[static_cast<Eigen::Index>(i)] = command_q[command_index];
      v[static_cast<Eigen::Index>(i)] = command_v[command_index];
      tau[static_cast<Eigen::Index>(i)] = command_tau[command_index];
    }

    q_measured_ = q;
    v_measured_ = v;
    tau_measured_ = tau;
    last_joint_state_time_ = now();
    have_joint_state_ = true;
    UpdateRobotSystemConfiguration(command_q_measured_);

    if (!q_ref_initialized_) {
      command_q_ref_ = command_q_measured_;
      ApplyFixedJointReferences(&command_q_ref_);
      q_ref_ = ExtractControlled(command_q_ref_);
      q_ref_initialized_ = true;
    }
  }

  void TactileCallback(std::size_t index,
                       const sdr_grasp_msgs::msg::Tactile::SharedPtr msg) {
    if (!msg || index >= tactile_states_.size()) {
      return;
    }
    auto tactile = ConvertTactile(*msg, cop_to_m_scale_);
    UpdateTactileDerivativeFeatures(index, TactileSampleTime(*msg), &tactile);
    tactile_states_[index] = tactile;
    tactile_received_[index] = true;
    last_tactile_time_[index] = now();
  }

  rclcpp::Time TactileSampleTime(const sdr_grasp_msgs::msg::Tactile& msg) {
    if (msg.header.stamp.sec == 0 && msg.header.stamp.nanosec == 0U) {
      return now();
    }
    return rclcpp::Time(msg.header.stamp);
  }

  void UpdateTactileDerivativeFeatures(
      std::size_t index, const rclcpp::Time& sample_time,
      mppi_core::NariTouchState* tactile) {
    if (tactile == nullptr || index >= previous_tactile_slip_states_.size()) {
      return;
    }

    if (!tactile->hasContact()) {
      tactile->slip_velocity_state.setZero();
      previous_tactile_slip_states_[index] = tactile->slip_state;
      filtered_slip_velocity_states_[index].setZero();
      previous_tactile_centroids_[index].setZero();
      filtered_centroid_velocity_mps_[index].setZero();
      previous_tactile_sample_time_[index] = sample_time;
      slip_velocity_initialized_[index] = true;
      centroid_velocity_initialized_[index] = false;
      return;
    }

    Eigen::Vector3d velocity = Eigen::Vector3d::Zero();
    Eigen::Vector2d centroid = Eigen::Vector2d::Zero();
    const bool centroid_valid =
        mppi_core::ComputeNariTouchContactCentroidM(*tactile, &centroid);
    Eigen::Vector2d centroid_velocity = Eigen::Vector2d::Zero();
    if (slip_velocity_initialized_[index]) {
      const double dt =
          (sample_time - previous_tactile_sample_time_[index]).seconds();
      if (std::isfinite(dt) && dt > 1.0e-4) {
        velocity =
            (tactile->slip_state - previous_tactile_slip_states_[index]) / dt;
        velocity = ClampVectorNorm(velocity, slip_velocity_max_norm_);
        velocity = slip_velocity_filter_alpha_ * velocity +
                   (1.0 - slip_velocity_filter_alpha_) *
                       filtered_slip_velocity_states_[index];
        if (centroid_valid && centroid_velocity_initialized_[index]) {
          centroid_velocity =
              (centroid - previous_tactile_centroids_[index]) / dt;
          centroid_velocity = ClampVectorNorm(
              centroid_velocity, centroid_velocity_max_norm_mps_);
          centroid_velocity =
              centroid_velocity_filter_alpha_ * centroid_velocity +
              (1.0 - centroid_velocity_filter_alpha_) *
                  filtered_centroid_velocity_mps_[index];
        }
      }
    }

    tactile->slip_velocity_state = velocity;
    tactile->centroid_velocity_mps = centroid_velocity;
    previous_tactile_slip_states_[index] = tactile->slip_state;
    filtered_slip_velocity_states_[index] = velocity;
    if (centroid_valid) {
      previous_tactile_centroids_[index] = centroid;
      filtered_centroid_velocity_mps_[index] = centroid_velocity;
      centroid_velocity_initialized_[index] = true;
    } else {
      filtered_centroid_velocity_mps_[index].setZero();
      centroid_velocity_initialized_[index] = false;
    }
    previous_tactile_sample_time_[index] = sample_time;
    slip_velocity_initialized_[index] = true;
  }

  bool JointStateFresh() const {
    return have_joint_state_ &&
           (now() - last_joint_state_time_).seconds() <= joint_state_timeout_s_;
  }

  bool BuildControlTactile(mppi_core::NariTouchState* tactile,
                           std::size_t* selected_index) const {
    if (tactile == nullptr) {
      return false;
    }

    bool any_fresh = false;
    std::size_t best_index = 0;
    double best_score = -std::numeric_limits<double>::infinity();

    for (std::size_t i = 0; i < tactile_states_.size(); ++i) {
      if (!tactile_received_[i]) {
        continue;
      }
      if ((now() - last_tactile_time_[i]).seconds() > tactile_timeout_s_) {
        continue;
      }

      any_fresh = true;
      const auto& sensor = tactile_states_[i];
      const double score =
          1000.0 * static_cast<double>(ToContactRank(sensor.contact_state)) +
          mppi_core::ComputeNariTouchTotalNormalForceN(sensor);
      if (score > best_score) {
        best_score = score;
        best_index = i;
      }
    }

    if (!any_fresh) {
      return false;
    }
    *tactile = tactile_states_[best_index];
    if (selected_index != nullptr) {
      *selected_index = best_index;
    }
    return true;
  }

  bool AllTactileActivationContactsReady() const {
    if (tactile_states_.empty()) {
      return false;
    }

    for (std::size_t i = 0; i < tactile_states_.size(); ++i) {
      if (!tactile_received_[i]) {
        return false;
      }
      if ((now() - last_tactile_time_[i]).seconds() > tactile_timeout_s_) {
        return false;
      }
      if (ToContactRank(tactile_states_[i].contact_state) <
          activation_contact_state_threshold_) {
        return false;
      }
    }
    return true;
  }

  static int ToContactRank(mppi_core::NariTouchContactState state) {
    return static_cast<int>(state);
  }

  Eigen::VectorXd ExtractControlled(
      const Eigen::VectorXd& command_vector) const {
    Eigen::VectorXd out(static_cast<Eigen::Index>(joint_dim_));
    for (std::size_t i = 0; i < joint_dim_; ++i) {
      out[static_cast<Eigen::Index>(i)] =
          command_vector[static_cast<Eigen::Index>(
              controlled_command_indices_[i])];
    }
    return out;
  }

  void ScatterControlled(const Eigen::VectorXd& controlled_vector,
                         Eigen::VectorXd* command_vector) const {
    if (command_vector == nullptr) {
      return;
    }
    for (std::size_t i = 0; i < joint_dim_; ++i) {
      (*command_vector)[static_cast<Eigen::Index>(
          controlled_command_indices_[i])] =
          controlled_vector[static_cast<Eigen::Index>(i)];
    }
  }

  void ApplyFixedJointReferences(Eigen::VectorXd* position,
                                 Eigen::VectorXd* velocity = nullptr) const {
    if (position == nullptr) {
      return;
    }
    for (Eigen::Index i = 0; i < fixed_joint_positions_.size(); ++i) {
      if (!std::isfinite(fixed_joint_positions_[i])) {
        continue;
      }
      (*position)[i] = fixed_joint_positions_[i];
      if (velocity != nullptr && i < velocity->size()) {
        (*velocity)[i] = 0.0;
      }
    }
  }

  void UpdateRobotSystemConfiguration(const Eigen::VectorXd& q) {
    if (!robot_system_enabled_) {
      return;
    }
    if (q.size() !=
        static_cast<Eigen::Index>(command_joint_q_indices_.size())) {
      return;
    }

    robot_q_ = robot_system_->q();
    for (std::size_t i = 0; i < command_joint_q_indices_.size(); ++i) {
      robot_q_[command_joint_q_indices_[i]] = q[static_cast<Eigen::Index>(i)];
    }
    robot_v_.setZero(robot_system_->nv());
    robot_q_valid_ = true;
  }

  void UpdateTactileFramePlacements() {
    if (!robot_system_enabled_ || !robot_q_valid_ || robot_data_ == nullptr) {
      return;
    }

    robot_system_->updateState(robot_q_, robot_v_, now().seconds());
    robot_system_->computeAllTerms(*robot_data_, robot_q_, robot_v_);

    for (std::size_t i = 0; i < tactile_frame_ids_.size(); ++i) {
      tactile_frame_placements_[i] =
          robot_system_->framePosition(*robot_data_, tactile_frame_ids_[i]);
      tactile_frame_pose_valid_[i] = true;
    }
    PublishTactileFramePoses();
  }

  void PublishTactileFramePoses() {
    if (!publish_tactile_frame_poses_ || !tactile_frame_pose_pub_) {
      return;
    }

    geometry_msgs::msg::PoseArray msg;
    msg.header.stamp = now();
    msg.header.frame_id = tactile_pose_frame_id_;
    msg.poses.reserve(tactile_frame_placements_.size());
    for (std::size_t i = 0; i < tactile_frame_placements_.size(); ++i) {
      if (i < tactile_frame_pose_valid_.size() &&
          tactile_frame_pose_valid_[i]) {
        msg.poses.push_back(ToPoseMsg(tactile_frame_placements_[i]));
      }
    }
    tactile_frame_pose_pub_->publish(msg);
  }

  void MaybeLogSelectedTactile(std::size_t selected_index) {
    if (selected_index == active_tactile_index_) {
      return;
    }
    active_tactile_index_ = selected_index;

    if (robot_system_enabled_ &&
        selected_index < tactile_frame_placements_.size() &&
        selected_index < tactile_frame_pose_valid_.size() &&
        tactile_frame_pose_valid_[selected_index]) {
      const auto& translation =
          tactile_frame_placements_[selected_index].translation();
      RCLCPP_INFO(get_logger(),
                  "Using tactile topic '%s' at frame '%s' "
                  "(%.3f, %.3f, %.3f in %s)",
                  tactile_topics_[selected_index].c_str(),
                  tactile_frame_names_[selected_index].c_str(), translation.x(),
                  translation.y(), translation.z(),
                  tactile_pose_frame_id_.c_str());
      return;
    }

    RCLCPP_INFO(get_logger(), "Using tactile topic '%s' at frame '%s'",
                tactile_topics_[selected_index].c_str(),
                tactile_frame_names_[selected_index].c_str());
  }

  void Update() {
    if (!JointStateFresh()) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
                           "Waiting for fresh joint state");
      return;
    }

    if (!q_ref_initialized_) {
      q_ref_ = q_measured_;
      q_ref_initialized_ = true;
    }

    UpdateTactileFramePlacements();

    mppi_core::NariTouchState tactile;
    std::size_t selected_tactile_index = 0;
    const bool tactile_ready =
        BuildControlTactile(&tactile, &selected_tactile_index);
    if (!tactile_ready) {
      if (publish_hold_without_tactile_) {
        PublishCommand(command_q_ref_,
                       Eigen::VectorXd::Zero(command_q_ref_.size()));
      } else {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
                             "Waiting for fresh tactile state");
      }
      return;
    }
    MaybeLogSelectedTactile(selected_tactile_index);

    if (activation_requires_all_tactile_enough_contact_ && !mppi_active_) {
      if (!AllTactileActivationContactsReady()) {
        RCLCPP_WARN_THROTTLE(
            get_logger(), *get_clock(), 1000,
            "Waiting to activate MPPI: all tactile topics must be fresh with "
            "contact_state >= %d",
            activation_contact_state_threshold_);
        if (publish_hold_without_tactile_) {
          PublishCommand(command_q_ref_,
                         Eigen::VectorXd::Zero(command_q_ref_.size()));
        }
        return;
      }

      mppi_active_ = true;
      RCLCPP_INFO(get_logger(),
                  "Activated MPPI: all %zu tactile topics reached "
                  "contact_state >= %d",
                  tactile_topics_.size(), activation_contact_state_threshold_);
    }

    mppi_core::GraspObservation observation;
    observation.q_measured = q_measured_;
    observation.v_measured = v_measured_;
    observation.q_ref_current = q_ref_;
    observation.v_ref_current =
        Eigen::VectorXd::Zero(static_cast<Eigen::Index>(joint_dim_));
    observation.tau = tau_measured_;
    const std::string tactile_frame =
        selected_tactile_index < tactile_frame_names_.size()
            ? tactile_frame_names_[selected_tactile_index]
            : std::string{};
    const double tactile_stamp =
        selected_tactile_index < last_tactile_time_.size()
            ? last_tactile_time_[selected_tactile_index].seconds()
            : now().seconds();
    observation.tactile = mppi_core::ConvertNariTouchToTactileState(
        tactile, tactile_frame, tactile_stamp, tactile_adapter_config_);
    observation.time_s = now().seconds();

    const Eigen::VectorXd old_q_ref = q_ref_;
    const auto command = policy_.Update(observation);
    if (command.q_des.size() != static_cast<Eigen::Index>(joint_dim_) ||
        command.v_des.size() != static_cast<Eigen::Index>(joint_dim_) ||
        command.delta_q_ref.size() != static_cast<Eigen::Index>(joint_dim_)) {
      RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 1000,
                            "MPPI command dimension mismatch");
      return;
    }

    q_ref_ = command.q_des;
    ClampQRef();
    Eigen::VectorXd v_ff = Eigen::VectorXd::Zero(q_ref_.size());
    if (command.dt > 0.0) {
      v_ff = (q_ref_ - old_q_ref) / command.dt;
    }
    Eigen::VectorXd command_v_ff =
        Eigen::VectorXd::Zero(static_cast<Eigen::Index>(command_joint_dim_));
    ScatterControlled(q_ref_, &command_q_ref_);
    ScatterControlled(v_ff, &command_v_ff);
    ApplyFixedJointReferences(&command_q_ref_, &command_v_ff);
    PublishCommand(command_q_ref_, command_v_ff);
  }

  void ClampQRef() {
    if (q_lower_bound_.size() == static_cast<Eigen::Index>(joint_dim_)) {
      q_ref_ = q_ref_.cwiseMax(q_lower_bound_);
    }
    if (q_upper_bound_.size() == static_cast<Eigen::Index>(joint_dim_)) {
      q_ref_ = q_ref_.cwiseMin(q_upper_bound_);
    }
  }

  void PublishCommand(const Eigen::VectorXd& q_ref,
                      const Eigen::VectorXd& v_ff) {
    wbc_msgs::msg::ImpedanceCommands msg;
    msg.position = ToStdVector(q_ref);
    msg.velocity = ToStdVector(v_ff);
    msg.stiffness = ToStdVector(stiffness_);
    msg.damping = ToStdVector(damping_);
    msg.effort_ff = ToStdVector(effort_ff_);
    command_pub_->publish(msg);
  }

  std::vector<std::string> command_joint_names_;
  std::vector<std::string> joint_names_;
  std::vector<std::size_t> controlled_command_indices_;
  std::vector<std::string> tactile_topics_;
  std::vector<std::string> tactile_frame_names_;
  std::string joint_state_topic_;
  std::string command_topic_;
  std::string urdf_path_;
  std::string resolved_urdf_path_;
  std::string tactile_frame_pose_topic_;
  std::string tactile_pose_frame_id_;
  std::size_t command_joint_dim_{0};
  std::size_t joint_dim_{0};

  double control_rate_hz_{100.0};
  double tactile_timeout_s_{0.25};
  double joint_state_timeout_s_{0.25};
  double cop_to_m_scale_{1.0e-3};
  double slip_velocity_filter_alpha_{0.25};
  double slip_velocity_max_norm_{100.0};
  double centroid_velocity_filter_alpha_{0.25};
  double centroid_velocity_max_norm_mps_{0.2};
  bool publish_hold_without_tactile_{true};
  bool publish_tactile_frame_poses_{true};
  bool activation_requires_all_tactile_enough_contact_{true};
  bool mppi_active_{false};
  int activation_contact_state_threshold_{2};

  Eigen::VectorXd q_measured_;
  Eigen::VectorXd v_measured_;
  Eigen::VectorXd tau_measured_;
  Eigen::VectorXd q_ref_;
  Eigen::VectorXd command_q_measured_;
  Eigen::VectorXd command_v_measured_;
  Eigen::VectorXd command_tau_measured_;
  Eigen::VectorXd command_q_ref_;
  Eigen::VectorXd fixed_joint_positions_;
  Eigen::VectorXd q_lower_bound_;
  Eigen::VectorXd q_upper_bound_;
  Eigen::VectorXd stiffness_;
  Eigen::VectorXd damping_;
  Eigen::VectorXd effort_ff_;

  bool have_joint_state_{false};
  bool q_ref_initialized_{false};
  rclcpp::Time last_joint_state_time_{0, 0, RCL_ROS_TIME};

  std::vector<mppi_core::NariTouchState> tactile_states_;
  std::vector<bool> tactile_received_;
  std::vector<rclcpp::Time> last_tactile_time_;
  std::vector<Eigen::Vector3d> previous_tactile_slip_states_;
  std::vector<Eigen::Vector3d> filtered_slip_velocity_states_;
  std::vector<Eigen::Vector2d> previous_tactile_centroids_;
  std::vector<Eigen::Vector2d> filtered_centroid_velocity_mps_;
  std::vector<rclcpp::Time> previous_tactile_sample_time_;
  std::vector<bool> slip_velocity_initialized_;
  std::vector<bool> centroid_velocity_initialized_;
  mppi_core::NariTouchAdapterConfig tactile_adapter_config_;

  bool robot_system_enabled_{false};
  bool robot_q_valid_{false};
  std::unique_ptr<wbc::robots::RobotSystem> robot_system_;
  std::unique_ptr<wbc::robots::RobotSystem::Data> robot_data_;
  Eigen::VectorXd robot_q_;
  Eigen::VectorXd robot_v_;
  std::vector<Eigen::Index> command_joint_q_indices_;
  std::vector<pinocchio::FrameIndex> tactile_frame_ids_;
  std::vector<pinocchio::SE3> tactile_frame_placements_;
  std::vector<bool> tactile_frame_pose_valid_;
  std::size_t active_tactile_index_{std::numeric_limits<std::size_t>::max()};

  mppi_core::JengaGrasp policy_;

  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr
      joint_state_sub_;
  std::vector<rclcpp::Subscription<sdr_grasp_msgs::msg::Tactile>::SharedPtr>
      tactile_subs_;
  rclcpp::Publisher<wbc_msgs::msg::ImpedanceCommands>::SharedPtr command_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr
      tactile_frame_pose_pub_;
  rclcpp::TimerBase::SharedPtr update_timer_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<JengaGraspMppiNode>());
  rclcpp::shutdown();
  return 0;
}
