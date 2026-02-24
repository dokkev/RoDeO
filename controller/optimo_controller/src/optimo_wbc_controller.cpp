#include "optimo_controller/optimo_wbc_controller.hpp"

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <hardware_interface/types/hardware_interface_type_values.hpp>
#include <pluginlib/class_list_macros.hpp>
#include <rclcpp/rclcpp.hpp>

#include <cmath>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <utility>

#include "optimo_controller/optimo_state_provider.hpp"

namespace optimo_controller {
namespace {

bool IsFinite(double value) {
  return std::isfinite(value);
}

} // namespace

controller_interface::CallbackReturn OptimoWbcController::on_init() {
  try {
    auto_declare<std::vector<std::string>>("joints", {});
    auto_declare<std::string>(
        "urdf_path", "package://optimo_description/urdf/optimo.urdf");
    auto_declare<std::string>("wbc_yaml_path",
                              "package://optimo_controller/config/optimo_wbc.yaml");
    auto_declare<std::string>("package_root", "");
    auto_declare<bool>("fixed_base", true);
  } catch (const std::exception& e) {
    RCLCPP_ERROR(get_node()->get_logger(),
                 "[OptimoWbcController] on_init failed: %s", e.what());
    return controller_interface::CallbackReturn::ERROR;
  }

  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::InterfaceConfiguration
OptimoWbcController::command_interface_configuration() const {
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;

  std::vector<std::string> joints = joints_;
  if (joints.empty() && get_node() != nullptr) {
    joints = get_node()->get_parameter("joints").as_string_array();
  }

  config.names.reserve(joints.size() * 3U);
  for (const auto& joint : joints) {
    config.names.push_back(joint + "/" + hardware_interface::HW_IF_POSITION);
  }
  for (const auto& joint : joints) {
    config.names.push_back(joint + "/" + hardware_interface::HW_IF_VELOCITY);
  }
  for (const auto& joint : joints) {
    config.names.push_back(joint + "/" + hardware_interface::HW_IF_EFFORT);
  }
  return config;
}

controller_interface::InterfaceConfiguration
OptimoWbcController::state_interface_configuration() const {
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;

  std::vector<std::string> joints = joints_;
  if (joints.empty() && get_node() != nullptr) {
    joints = get_node()->get_parameter("joints").as_string_array();
  }

  config.names.reserve(joints.size() * 3U);
  for (const auto& joint : joints) {
    config.names.push_back(joint + "/" + hardware_interface::HW_IF_POSITION);
  }
  for (const auto& joint : joints) {
    config.names.push_back(joint + "/" + hardware_interface::HW_IF_VELOCITY);
  }
  for (const auto& joint : joints) {
    config.names.push_back(joint + "/" + hardware_interface::HW_IF_EFFORT);
  }
  return config;
}

controller_interface::CallbackReturn
OptimoWbcController::on_configure(const rclcpp_lifecycle::State& /*previous_state*/) {
  try {
    joints_ = get_node()->get_parameter("joints").as_string_array();
    urdf_path_ = get_node()->get_parameter("urdf_path").as_string();
    wbc_yaml_path_ = get_node()->get_parameter("wbc_yaml_path").as_string();
    package_root_ = get_node()->get_parameter("package_root").as_string();
    fixed_base_ = get_node()->get_parameter("fixed_base").as_bool();
  } catch (const std::exception& e) {
    RCLCPP_ERROR(get_node()->get_logger(),
                 "[OptimoWbcController] failed to read parameters: %s",
                 e.what());
    return controller_interface::CallbackReturn::ERROR;
  }

  if (joints_.empty()) {
    RCLCPP_ERROR(get_node()->get_logger(),
                 "[OptimoWbcController] 'joints' parameter is empty.");
    return controller_interface::CallbackReturn::ERROR;
  }

  const unsigned int update_rate_hz = get_update_rate();
  if (update_rate_hz == 0U) {
    RCLCPP_ERROR(get_node()->get_logger(),
                 "[OptimoWbcController] update_rate must be positive.");
    return controller_interface::CallbackReturn::ERROR;
  }
  fallback_dt_ = 1.0 / static_cast<double>(update_rate_hz);

  try {
    urdf_path_ = ResolvePackageUri(urdf_path_);
    wbc_yaml_path_ = ResolvePackageUri(wbc_yaml_path_);

    if (package_root_.empty()) {
      package_root_ = ResolvePackageRootForUri(urdf_path_);
    } else {
      package_root_ = ResolvePackageUri(package_root_);
    }
  } catch (const std::exception& e) {
    RCLCPP_ERROR(get_node()->get_logger(),
                 "[OptimoWbcController] Failed to resolve package URIs: %s",
                 e.what());
    return controller_interface::CallbackReturn::ERROR;
  }

  try {
    robot_ = std::make_unique<wbc::PinocchioRobotSystem>(
        urdf_path_, package_root_, fixed_base_, false);
    wbc_arch_ = wbc::BuildControlArchitecture(
        robot_.get(), wbc_yaml_path_, fallback_dt_,
        std::make_unique<wbc::OptimoStateProvider>(fallback_dt_), nullptr);
  } catch (const std::exception& e) {
    RCLCPP_ERROR(get_node()->get_logger(),
                 "[OptimoWbcController] Initialization failed: %s", e.what());
    return controller_interface::CallbackReturn::ERROR;
  }

  const int active_dof = robot_->NumActiveDof();
  if (active_dof != static_cast<int>(joints_.size())) {
    RCLCPP_ERROR(
        get_node()->get_logger(),
        "[OptimoWbcController] Joint count mismatch. robot active dof=%d, "
        "configured joints=%zu.",
        active_dof, joints_.size());
    return controller_interface::CallbackReturn::ERROR;
  }

  q_ = Eigen::VectorXd::Zero(active_dof);
  qdot_ = Eigen::VectorXd::Zero(active_dof);

  RCLCPP_INFO(get_node()->get_logger(),
              "[OptimoWbcController] Configured with %zu joints (update_rate=%uHz, fallback_dt=%.6f).",
              joints_.size(), update_rate_hz, fallback_dt_);
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn
OptimoWbcController::on_activate(const rclcpp_lifecycle::State& /*previous_state*/) {
  if (command_interfaces_.size() != joints_.size() * 3U) {
    RCLCPP_ERROR(
        get_node()->get_logger(),
        "[OptimoWbcController] Command interfaces mismatch. expected=%zu got=%zu",
        joints_.size() * 3U, command_interfaces_.size());
    return controller_interface::CallbackReturn::ERROR;
  }

  if (state_interfaces_.size() != joints_.size() * 3U) {
    RCLCPP_ERROR(
        get_node()->get_logger(),
        "[OptimoWbcController] State interfaces mismatch. expected=%zu got=%zu",
        joints_.size() * 3U, state_interfaces_.size());
    return controller_interface::CallbackReturn::ERROR;
  }

  for (auto& cmd_if : command_interfaces_) {
    if (!cmd_if.set_value(0.0)) {
      RCLCPP_ERROR(get_node()->get_logger(),
                   "[OptimoWbcController] Failed to initialize command interface.");
      return controller_interface::CallbackReturn::ERROR;
    }
  }

  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn
OptimoWbcController::on_deactivate(const rclcpp_lifecycle::State& /*previous_state*/) {
  for (auto& cmd_if : command_interfaces_) {
    if (!cmd_if.set_value(0.0)) {
      RCLCPP_WARN(get_node()->get_logger(),
                  "[OptimoWbcController] Failed to zero command interface on deactivate.");
    }
  }
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::return_type
OptimoWbcController::update(const rclcpp::Time& time,
                            const rclcpp::Duration& period) {
  if (wbc_arch_ == nullptr) {
    RCLCPP_ERROR(get_node()->get_logger(),
                 "[OptimoWbcController] update called before configure.");
    return controller_interface::return_type::ERROR;
  }

  if (!ReadJointState(q_, qdot_)) {
    return controller_interface::return_type::ERROR;
  }

  const double dt = period.seconds() > std::numeric_limits<double>::epsilon()
                        ? period.seconds()
                        : fallback_dt_;

  try {
    wbc_arch_->Update(q_, qdot_, time.seconds(), dt);
    return WriteCommand(wbc_arch_->GetCommand())
               ? controller_interface::return_type::OK
               : controller_interface::return_type::ERROR;
  } catch (const std::exception& e) {
    RCLCPP_ERROR(get_node()->get_logger(),
                 "[OptimoWbcController] update failed: %s", e.what());
    return controller_interface::return_type::ERROR;
  }
}

std::string OptimoWbcController::ResolvePackageUri(const std::string& uri) {
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

  const std::filesystem::path share_dir =
      ament_index_cpp::get_package_share_directory(package_name);
  if (relative.empty()) {
    return share_dir.string();
  }
  return (share_dir / relative).string();
}

std::string OptimoWbcController::ResolvePackageRootForUri(const std::string& uri) {
  constexpr const char kPackagePrefix[] = "package://";
  if (uri.rfind(kPackagePrefix, 0) == 0U) {
    const std::string suffix = uri.substr(sizeof(kPackagePrefix) - 1U);
    const std::size_t slash = suffix.find('/');
    const std::string package_name =
        (slash == std::string::npos) ? suffix : suffix.substr(0, slash);
    if (package_name.empty()) {
      throw std::runtime_error("Invalid package URI: " + uri);
    }
    const std::filesystem::path share_dir =
        ament_index_cpp::get_package_share_directory(package_name);
    return share_dir.parent_path().string();
  }

  const std::filesystem::path optimo_share =
      ament_index_cpp::get_package_share_directory("optimo_description");
  return optimo_share.parent_path().string();
}

bool OptimoWbcController::ReadJointState(Eigen::VectorXd& q,
                                         Eigen::VectorXd& qdot) const {
  const std::size_t n = joints_.size();
  if (state_interfaces_.size() != n * 3U || q.size() != static_cast<int>(n) ||
      qdot.size() != static_cast<int>(n)) {
    RCLCPP_ERROR(get_node()->get_logger(),
                 "[OptimoWbcController] Invalid state buffer dimensions.");
    return false;
  }

  for (std::size_t i = 0; i < n; ++i) {
    const double qi = state_interfaces_[i].get_value();
    const double qdi = state_interfaces_[n + i].get_value();
    const double taui = state_interfaces_[2U * n + i].get_value();
    if (!IsFinite(qi) || !IsFinite(qdi) || !IsFinite(taui)) {
      RCLCPP_ERROR(get_node()->get_logger(),
                   "[OptimoWbcController] Non-finite state at index %zu.", i);
      return false;
    }
    q(static_cast<Eigen::Index>(i)) = qi;
    qdot(static_cast<Eigen::Index>(i)) = qdi;
  }
  return true;
}

bool OptimoWbcController::WriteCommand(const wbc::RobotCommand& cmd) {
  const std::size_t n = joints_.size();
  if (cmd.q.size() != static_cast<int>(n) ||
      cmd.qdot.size() != static_cast<int>(n) ||
      cmd.tau.size() != static_cast<int>(n) ||
      command_interfaces_.size() != n * 3U) {
    RCLCPP_ERROR(get_node()->get_logger(),
                 "[OptimoWbcController] Command size mismatch.");
    return false;
  }

  for (std::size_t i = 0; i < n; ++i) {
    const auto idx = static_cast<Eigen::Index>(i);
    if (!command_interfaces_[i].set_value(cmd.q(idx))) {
      RCLCPP_ERROR(get_node()->get_logger(),
                   "[OptimoWbcController] Failed to write position command for joint %s.",
                   joints_[i].c_str());
      return false;
    }
    if (!command_interfaces_[n + i].set_value(cmd.qdot(idx))) {
      RCLCPP_ERROR(get_node()->get_logger(),
                   "[OptimoWbcController] Failed to write velocity command for joint %s.",
                   joints_[i].c_str());
      return false;
    }
    if (!command_interfaces_[2U * n + i].set_value(cmd.tau(idx))) {
      RCLCPP_ERROR(get_node()->get_logger(),
                   "[OptimoWbcController] Failed to write effort command for joint %s.",
                   joints_[i].c_str());
      return false;
    }
  }
  return true;
}

} // namespace optimo_controller

PLUGINLIB_EXPORT_CLASS(optimo_controller::OptimoWbcController,
                       controller_interface::ControllerInterface)
