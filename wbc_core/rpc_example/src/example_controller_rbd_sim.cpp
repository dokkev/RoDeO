#include "rpc_example/ExampleController.hpp"
#include "wbc_robot_system/pinocchio_robot_system.hpp"

#include <Eigen/Cholesky>
#include <Eigen/Dense>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

struct SimOptions {
  double dt{0.001};
  double duration{20.0};
  double max_accel{80.0};
  double max_velocity{6.0};
  double dynamics_regularization{1.0e-3};
  int print_every{250};
  bool real_time{true};
  bool publish_joint_states{true};
  bool print_robot_info{false};
  bool zero_torque{false};

  std::string urdf_path;
  std::string yaml_path;
  std::string package_dir;
};

constexpr std::array<double, 7> kDefaultInitQ = {
    0.0, 3.3, 0.0, -2.35, 0.0, -1.13, 0.0};

std::string ToLower(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return value;
}

bool ParseBool(const std::string& raw) {
  const std::string value = ToLower(raw);
  if (value == "1" || value == "true" || value == "yes" || value == "on") {
    return true;
  }
  if (value == "0" || value == "false" || value == "no" || value == "off") {
    return false;
  }
  throw std::runtime_error("Invalid boolean value: " + raw);
}

std::filesystem::path TryShareDir(const std::string& package_name) {
  try {
    return std::filesystem::path(
        ament_index_cpp::get_package_share_directory(package_name));
  } catch (const std::exception&) {
    return {};
  }
}

std::string DefaultUrdfPath() {
  const std::filesystem::path rpc_example_share = TryShareDir("rpc_example");
  if (!rpc_example_share.empty()) {
    const std::filesystem::path urdf =
        rpc_example_share / "description/urdf/optimo.urdf";
    if (std::filesystem::exists(urdf)) {
      return urdf.string();
    }
  }
  return {};
}

std::string DefaultYamlPath() {
  const std::filesystem::path rpc_example_share = TryShareDir("rpc_example");
  if (!rpc_example_share.empty()) {
    const std::filesystem::path yaml =
        rpc_example_share / "config/example_wbic.yaml";
    if (std::filesystem::exists(yaml)) {
      return yaml.string();
    }
  }
  return {};
}

std::string DefaultPackageDir() {
  const std::filesystem::path optimo_description_share =
      TryShareDir("optimo_description");
  if (!optimo_description_share.empty()) {
    // Pinocchio expects a package root that contains <pkg_name>/...
    return optimo_description_share.parent_path().string();
  }
  return {};
}

void PrintUsage(const char* argv0) {
  std::cout
      << "Usage: " << argv0 << " [options]\n"
      << "  --dt <seconds>                     (default: 0.001)\n"
      << "  --duration <seconds>               (default: 20.0)\n"
      << "  --max-accel <rad/s^2>              (default: 80.0)\n"
      << "  --max-velocity <rad/s>             (default: 6.0)\n"
      << "  --dynamics-regularization <value>  (default: 1e-3)\n"
      << "  --print-every <steps>              (default: 250)\n"
      << "  --real-time <true|false>           (default: true)\n"
      << "  --publish-joint-states <true|false>(default: true)\n"
      << "  --print-robot-info <true|false>    (default: false)\n"
      << "  --zero-torque <true|false>         (default: false)\n"
      << "  --urdf <path>\n"
      << "  --yaml <path>\n"
      << "  --package-dir <path>\n"
      << "  --help\n";
}

SimOptions ParseArgs(const std::vector<std::string>& args, const char* argv0) {
  SimOptions options;
  options.urdf_path = DefaultUrdfPath();
  options.yaml_path = DefaultYamlPath();
  options.package_dir = DefaultPackageDir();

  for (std::size_t i = 1; i < args.size(); ++i) {
    const std::string& arg = args[i];
    const auto require_value = [&](const std::string& name) -> const std::string& {
      if (i + 1 >= args.size()) {
        throw std::runtime_error("Missing value for " + name);
      }
      return args[++i];
    };

    if (arg == "--help" || arg == "-h") {
      PrintUsage(argv0);
      std::exit(0);
    } else if (arg == "--dt") {
      options.dt = std::stod(require_value(arg));
    } else if (arg == "--duration") {
      options.duration = std::stod(require_value(arg));
    } else if (arg == "--max-accel") {
      options.max_accel = std::stod(require_value(arg));
    } else if (arg == "--max-velocity") {
      options.max_velocity = std::stod(require_value(arg));
    } else if (arg == "--dynamics-regularization") {
      options.dynamics_regularization = std::stod(require_value(arg));
    } else if (arg == "--print-every") {
      options.print_every = std::stoi(require_value(arg));
    } else if (arg == "--real-time") {
      options.real_time = ParseBool(require_value(arg));
    } else if (arg == "--publish-joint-states") {
      options.publish_joint_states = ParseBool(require_value(arg));
    } else if (arg == "--print-robot-info") {
      options.print_robot_info = ParseBool(require_value(arg));
    } else if (arg == "--zero-torque") {
      options.zero_torque = ParseBool(require_value(arg));
    } else if (arg == "--urdf") {
      options.urdf_path = require_value(arg);
    } else if (arg == "--yaml") {
      options.yaml_path = require_value(arg);
    } else if (arg == "--package-dir") {
      options.package_dir = require_value(arg);
    } else {
      throw std::runtime_error("Unknown argument: " + arg);
    }
  }

  if (options.dt <= 0.0) {
    throw std::runtime_error("--dt must be positive.");
  }
  if (options.duration <= 0.0) {
    throw std::runtime_error("--duration must be positive.");
  }
  if (options.max_accel <= 0.0) {
    throw std::runtime_error("--max-accel must be positive.");
  }
  if (options.max_velocity <= 0.0) {
    throw std::runtime_error("--max-velocity must be positive.");
  }
  if (options.dynamics_regularization < 0.0) {
    throw std::runtime_error("--dynamics-regularization cannot be negative.");
  }
  if (options.urdf_path.empty()) {
    throw std::runtime_error(
        "URDF path is empty. Pass --urdf explicitly or build/source the workspace.");
  }
  if (options.yaml_path.empty()) {
    throw std::runtime_error(
        "YAML path is empty. Pass --yaml explicitly or build/source the workspace.");
  }
  if (options.package_dir.empty()) {
    throw std::runtime_error(
        "Package dir is empty. Pass --package-dir explicitly.");
  }

  return options;
}

std::vector<std::string> BuildOrderedJointNames(
    const std::unordered_map<std::string, int>& joint_name_to_idx,
    int num_joints) {
  std::vector<std::string> names(static_cast<std::size_t>(num_joints));
  for (const auto& kv : joint_name_to_idx) {
    if (kv.second >= 0 && kv.second < num_joints) {
      names[static_cast<std::size_t>(kv.second)] = kv.first;
    }
  }
  return names;
}

void ClampToJointLimits(Eigen::VectorXd& q, const Eigen::MatrixXd& pos_limits) {
  if (q.size() != pos_limits.rows() || pos_limits.cols() < 2) {
    return;
  }

  for (int i = 0; i < q.size(); ++i) {
    const double lo = pos_limits(i, 0);
    const double hi = pos_limits(i, 1);
    if (!std::isfinite(lo) || !std::isfinite(hi) || lo >= hi) {
      continue;
    }
    q(i) = std::clamp(q(i), lo, hi);
  }
}

void ClampToVelocityLimits(Eigen::VectorXd& qdot, const Eigen::MatrixXd& vel_limits,
                           double max_velocity) {
  if (qdot.size() != vel_limits.rows() || vel_limits.cols() < 2) {
    qdot = qdot.cwiseMax(-max_velocity).cwiseMin(max_velocity);
    return;
  }

  for (int i = 0; i < qdot.size(); ++i) {
    double lo = vel_limits(i, 0);
    double hi = vel_limits(i, 1);
    if (!std::isfinite(lo) || !std::isfinite(hi) || lo >= hi) {
      lo = -max_velocity;
      hi = max_velocity;
    } else {
      lo = std::max(lo, -max_velocity);
      hi = std::min(hi, max_velocity);
    }
    qdot(i) = std::clamp(qdot(i), lo, hi);
  }
}

void ClampTorque(Eigen::VectorXd& tau, const Eigen::MatrixXd& tau_limits) {
  if (tau.size() != tau_limits.rows() || tau_limits.cols() < 2) {
    return;
  }

  for (int i = 0; i < tau.size(); ++i) {
    const double lo = tau_limits(i, 0);
    const double hi = tau_limits(i, 1);
    if (!std::isfinite(lo) || !std::isfinite(hi) || lo >= hi) {
      continue;
    }
    tau(i) = std::clamp(tau(i), lo, hi);
  }
}

} // namespace

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);

  int exit_code = 0;
  try {
    const std::vector<std::string> args = rclcpp::remove_ros_arguments(argc, argv);
    const SimOptions options = ParseArgs(args, argv[0]);

    const auto node = rclcpp::Node::make_shared("example_controller_rbd_sim");

    std::cout << "=== ExampleController RBD simulation ===" << std::endl;
    std::cout << "URDF: " << options.urdf_path << std::endl;
    std::cout << "YAML: " << options.yaml_path << std::endl;
    std::cout << "package_dir: " << options.package_dir << std::endl;
    std::cout << "dt: " << options.dt << " s, duration: " << options.duration
              << " s" << std::endl;
    std::cout << "zero_torque: " << (options.zero_torque ? "true" : "false")
              << std::endl;

    auto robot = std::make_unique<wbc::PinocchioRobotSystem>(
        options.urdf_path, options.package_dir, true,
        options.print_robot_info);
    wbc::ExampleController controller(robot.get(), options.yaml_path, options.dt);

    const int n = robot->NumActiveDof();
    if (n <= 0) {
      throw std::runtime_error("Robot has no active DoF.");
    }

    const Eigen::MatrixXd pos_limits = robot->JointPosLimits();
    const Eigen::MatrixXd vel_limits = robot->JointVelLimits();
    const Eigen::MatrixXd tau_limits = robot->JointTrqLimits();

    Eigen::VectorXd q = Eigen::VectorXd::Zero(n);
    Eigen::VectorXd qdot = Eigen::VectorXd::Zero(n);
    if (n == static_cast<int>(kDefaultInitQ.size())) {
      for (int i = 0; i < n; ++i) {
        q(i) = kDefaultInitQ[static_cast<std::size_t>(i)];
      }
    }
    ClampToJointLimits(q, pos_limits);

    std::cout << "initial_q: [";
    for (int i = 0; i < n; ++i) {
      std::cout << q(i);
      if (i + 1 < n) {
        std::cout << ", ";
      }
    }
    std::cout << "]" << std::endl;

    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr joint_pub;
    std::vector<std::string> joint_names;
    if (options.publish_joint_states) {
      joint_pub = node->create_publisher<sensor_msgs::msg::JointState>(
          "joint_states", rclcpp::SystemDefaultsQoS());
      joint_names = BuildOrderedJointNames(robot->GetJointNameAndIndexMap(), n);
    }

    const int steps = std::max(1, static_cast<int>(options.duration / options.dt));
    auto next_tick = std::chrono::steady_clock::now();
    for (int step = 0; rclcpp::ok() && step < steps; ++step) {
      const double t = static_cast<double>(step) * options.dt;

      controller.Update(q, qdot, t, options.dt);
      const wbc::RobotCommand command = controller.GetCommand();

      Eigen::VectorXd tau = Eigen::VectorXd::Zero(n);
      if (command.tau.size() == n) {
        tau = command.tau;
      }
      if (options.zero_torque) {
        tau.setZero();
      }
      ClampTorque(tau, tau_limits);

      Eigen::VectorXd qddot = Eigen::VectorXd::Zero(n);
      const Eigen::MatrixXd& mass = robot->GetMassMatrixRef();
      const Eigen::VectorXd bias = robot->GetCoriolisRef() + robot->GetGravityRef();

      if (mass.rows() == n && mass.cols() == n && bias.size() == n) {
        Eigen::MatrixXd mass_reg = mass;
        if (options.dynamics_regularization > 0.0) {
          mass_reg.diagonal().array() += options.dynamics_regularization;
        }
        Eigen::LDLT<Eigen::MatrixXd> ldlt(mass_reg);
        if (ldlt.info() == Eigen::Success) {
          qddot = ldlt.solve(tau - bias);
        }
      } else {
        throw std::runtime_error(
            "Unexpected dynamics matrix/vector dimensions for fixed-base "
            "simulation.");
      }
      if (!qddot.allFinite()) {
        qddot.setZero();
      }
      qddot = qddot.cwiseMax(-options.max_accel).cwiseMin(options.max_accel);

      qdot += qddot * options.dt;
      ClampToVelocityLimits(qdot, vel_limits, options.max_velocity);
      q += qdot * options.dt;
      ClampToJointLimits(q, pos_limits);

      if (joint_pub != nullptr) {
        sensor_msgs::msg::JointState js;
        js.header.stamp = node->now();
        js.name = joint_names;
        js.position.resize(static_cast<std::size_t>(n));
        js.velocity.resize(static_cast<std::size_t>(n));
        js.effort.resize(static_cast<std::size_t>(n));
        for (int i = 0; i < n; ++i) {
          js.position[static_cast<std::size_t>(i)] = q(i);
          js.velocity[static_cast<std::size_t>(i)] = qdot(i);
          js.effort[static_cast<std::size_t>(i)] = tau(i);
        }
        joint_pub->publish(js);
      }

      if (options.print_every > 0 && (step % options.print_every) == 0) {
        std::cout << "[step " << step << "] t=" << t
                  << " state=" << controller.CurrentStateId();
        std::ostringstream q_state_stream;
        q_state_stream << std::fixed << std::setprecision(3) << q.transpose();
        std::cout << " q=[" << q_state_stream.str() << "]";
        if (command.q.size() == n) {
          std::ostringstream q_stream;
          q_stream << std::fixed << std::setprecision(3) << command.q.transpose();
          std::cout << " cmd_q=[" << q_stream.str() << "]";
        } else {
          std::cout << " cmd_q=[n/a]";
        }
        std::cout << std::endl;
      }

      rclcpp::spin_some(node);
      if (options.real_time) {
        next_tick += std::chrono::duration_cast<std::chrono::steady_clock::duration>(
            std::chrono::duration<double>(options.dt));
        std::this_thread::sleep_until(next_tick);
      }
    }

    std::cout << "Simulation finished." << std::endl;
  } catch (const std::exception& e) {
    std::cerr << "Simulation failed: " << e.what() << std::endl;
    exit_code = 1;
  }

  rclcpp::shutdown();
  return exit_code;
}
