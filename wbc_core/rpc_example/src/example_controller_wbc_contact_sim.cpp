#include "rpc_example/ExampleController.hpp"
#include "wbc_robot_system/pinocchio_robot_system.hpp"

#include <Eigen/Cholesky>
#include <Eigen/Dense>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <pinocchio/algorithm/center-of-mass.hpp>
#include <pinocchio/algorithm/constrained-dynamics.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/parsers/urdf.hpp>
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
  double duration{10.0};
  double base_height{0.95};
  double max_accel{120.0};
  double max_velocity{12.0};
  double contact_kp{120.0};
  double contact_kd{20.0};
  double contact_force_threshold{20.0};
  double prox_abs{1.0e-8};
  double prox_rel{1.0e-8};
  double prox_mu{1.0e-8};
  int prox_iters{30};
  int print_every{250};
  bool real_time{true};
  bool publish_joint_states{true};
  bool print_robot_info{false};
  bool zero_torque{false};

  std::string urdf_path;
  std::string yaml_path;
  std::string package_dir;
  std::string contact_frames_csv{"l_foot_contact,r_foot_contact"};
};

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
  const std::filesystem::path draco_description_share =
      TryShareDir("draco_description");
  if (!draco_description_share.empty()) {
    const std::filesystem::path urdf =
        draco_description_share / "urdf/draco_modified_rviz.urdf";
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
        rpc_example_share / "config/example_wbic_draco_floating.yaml";
    if (std::filesystem::exists(yaml)) {
      return yaml.string();
    }
  }
  return {};
}

std::string DefaultPackageDir() {
  const std::filesystem::path draco_description_share =
      TryShareDir("draco_description");
  if (!draco_description_share.empty()) {
    return draco_description_share.parent_path().string();
  }
  return {};
}

std::vector<std::string> SplitCsv(const std::string& csv) {
  std::vector<std::string> out;
  std::string token;
  std::stringstream ss(csv);
  while (std::getline(ss, token, ',')) {
    token.erase(std::remove_if(token.begin(), token.end(),
                               [](unsigned char c) { return std::isspace(c) != 0; }),
                token.end());
    if (!token.empty()) {
      out.push_back(token);
    }
  }
  return out;
}

void PrintUsage(const char* argv0) {
  std::cout
      << "Usage: " << argv0 << " [options]\n"
      << "  --dt <seconds>                       (default: 0.001)\n"
      << "  --duration <seconds>                 (default: 10.0)\n"
      << "  --base-height <meters>               (default: 0.95)\n"
      << "  --max-accel <rad/s^2>                (default: 120.0)\n"
      << "  --max-velocity <rad/s>               (default: 12.0)\n"
      << "  --contact-kp <value>                 (default: 120.0)\n"
      << "  --contact-kd <value>                 (default: 20.0)\n"
      << "  --contact-force-threshold <N>        (default: 20.0)\n"
      << "  --prox-abs <value>                   (default: 1e-8)\n"
      << "  --prox-rel <value>                   (default: 1e-8)\n"
      << "  --prox-mu <value>                    (default: 1e-8)\n"
      << "  --prox-iters <int>                   (default: 30)\n"
      << "  --contact-frames <csv>               (default: l_foot_contact,r_foot_contact)\n"
      << "  --print-every <steps>                (default: 250)\n"
      << "  --real-time <true|false>             (default: true)\n"
      << "  --publish-joint-states <true|false>  (default: true)\n"
      << "  --print-robot-info <true|false>      (default: false)\n"
      << "  --zero-torque <true|false>           (default: false)\n"
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
    } else if (arg == "--base-height") {
      options.base_height = std::stod(require_value(arg));
    } else if (arg == "--max-accel") {
      options.max_accel = std::stod(require_value(arg));
    } else if (arg == "--max-velocity") {
      options.max_velocity = std::stod(require_value(arg));
    } else if (arg == "--contact-kp") {
      options.contact_kp = std::stod(require_value(arg));
    } else if (arg == "--contact-kd") {
      options.contact_kd = std::stod(require_value(arg));
    } else if (arg == "--contact-force-threshold") {
      options.contact_force_threshold = std::stod(require_value(arg));
    } else if (arg == "--prox-abs") {
      options.prox_abs = std::stod(require_value(arg));
    } else if (arg == "--prox-rel") {
      options.prox_rel = std::stod(require_value(arg));
    } else if (arg == "--prox-mu") {
      options.prox_mu = std::stod(require_value(arg));
    } else if (arg == "--prox-iters") {
      options.prox_iters = std::stoi(require_value(arg));
    } else if (arg == "--contact-frames") {
      options.contact_frames_csv = require_value(arg);
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
  if (options.prox_iters < 1) {
    throw std::runtime_error("--prox-iters must be >= 1.");
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

void ClampToJointLimits(Eigen::VectorXd& q_joint,
                        const Eigen::MatrixXd& pos_limits) {
  if (q_joint.size() != pos_limits.rows() || pos_limits.cols() < 2) {
    return;
  }

  for (int i = 0; i < q_joint.size(); ++i) {
    const double lo = pos_limits(i, 0);
    const double hi = pos_limits(i, 1);
    if (!std::isfinite(lo) || !std::isfinite(hi) || lo >= hi) {
      continue;
    }
    q_joint(i) = std::clamp(q_joint(i), lo, hi);
  }
}

void ClampToVelocityLimits(Eigen::VectorXd& qdot_joint,
                           const Eigen::MatrixXd& vel_limits,
                           double max_velocity) {
  if (qdot_joint.size() != vel_limits.rows() || vel_limits.cols() < 2) {
    qdot_joint = qdot_joint.cwiseMax(-max_velocity).cwiseMin(max_velocity);
    return;
  }

  for (int i = 0; i < qdot_joint.size(); ++i) {
    double lo = vel_limits(i, 0);
    double hi = vel_limits(i, 1);
    if (!std::isfinite(lo) || !std::isfinite(hi) || lo >= hi) {
      lo = -max_velocity;
      hi = max_velocity;
    } else {
      lo = std::max(lo, -max_velocity);
      hi = std::min(hi, max_velocity);
    }
    qdot_joint(i) = std::clamp(qdot_joint(i), lo, hi);
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

Eigen::Quaterniond CoeffsToQuat(const Eigen::Vector4d& coeffs_xyzw) {
  Eigen::Quaterniond q;
  q.coeffs() = coeffs_xyzw;
  q.normalize();
  return q;
}

} // namespace

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);

  int exit_code = 0;
  try {
    const std::vector<std::string> args = rclcpp::remove_ros_arguments(argc, argv);
    const SimOptions options = ParseArgs(args, argv[0]);

    const auto node = rclcpp::Node::make_shared("example_controller_wbc_contact_sim");

    std::cout << "=== ExampleController WBC contact simulation ===" << std::endl;
    std::cout << "URDF: " << options.urdf_path << std::endl;
    std::cout << "YAML: " << options.yaml_path << std::endl;
    std::cout << "package_dir: " << options.package_dir << std::endl;
    std::cout << "dt: " << options.dt << " s, duration: " << options.duration
              << " s" << std::endl;
    std::cout << "zero_torque: " << (options.zero_torque ? "true" : "false")
              << std::endl;

    auto robot = std::make_unique<wbc::PinocchioRobotSystem>(
        options.urdf_path, options.package_dir, false, options.print_robot_info);
    wbc::ExampleController controller(robot.get(), options.yaml_path, options.dt);

    const int n_q = robot->GetNumQ();
    const int n_qdot = robot->NumQdot();
    const int n_act = robot->NumActiveDof();
    const int n_float = robot->NumFloatDof();

    if (n_float != 6 || n_q < 7 || n_qdot < 6 || n_act <= 0) {
      throw std::runtime_error(
          "This simulator expects floating-base model with valid Draco-like dimensions.");
    }

    const Eigen::MatrixXd pos_limits = robot->JointPosLimits();
    const Eigen::MatrixXd vel_limits = robot->JointVelLimits();
    const Eigen::MatrixXd tau_limits = robot->JointTrqLimits();

    pinocchio::Model dyn_model;
    pinocchio::urdf::buildModel(options.urdf_path, pinocchio::JointModelFreeFlyer(),
                                dyn_model);
    pinocchio::Data dyn_data(dyn_model);

    if (dyn_model.nq != n_q || dyn_model.nv != n_qdot) {
      throw std::runtime_error("Dimension mismatch between robot system and dynamics model.");
    }

    Eigen::VectorXd q_full = pinocchio::neutral(dyn_model);
    q_full(2) = options.base_height;
    Eigen::VectorXd v_full = Eigen::VectorXd::Zero(n_qdot);

    const std::vector<std::string> contact_frames =
        SplitCsv(options.contact_frames_csv);
    if (contact_frames.empty()) {
      throw std::runtime_error("No contact frames provided.");
    }

    pinocchio::forwardKinematics(dyn_model, dyn_data, q_full, v_full);
    pinocchio::updateFramePlacements(dyn_model, dyn_data);

    std::vector<pinocchio::RigidConstraintModel,
                Eigen::aligned_allocator<pinocchio::RigidConstraintModel>>
        contact_models;
    contact_models.reserve(contact_frames.size());

    for (const std::string& frame_name : contact_frames) {
      if (!dyn_model.existFrame(frame_name)) {
        throw std::runtime_error("Contact frame not found in URDF: " + frame_name);
      }
      const pinocchio::FrameIndex frame_id = dyn_model.getFrameId(frame_name);
      const pinocchio::JointIndex joint_id = dyn_model.frames[frame_id].parentJoint;
      const pinocchio::SE3 joint1_placement = dyn_model.frames[frame_id].placement;
      const pinocchio::SE3 world_contact = dyn_data.oMf[frame_id];

      pinocchio::RigidConstraintModel contact_model(
          pinocchio::CONTACT_6D, dyn_model, joint_id, joint1_placement, 0,
          world_contact, pinocchio::LOCAL);
      contact_model.corrector.Kp.setConstant(options.contact_kp);
      contact_model.corrector.Kd.setConstant(options.contact_kd);
      contact_models.push_back(contact_model);
    }

    std::vector<pinocchio::RigidConstraintData,
                Eigen::aligned_allocator<pinocchio::RigidConstraintData>>
        contact_datas;
    contact_datas.reserve(contact_models.size());
    for (const auto& cm : contact_models) {
      contact_datas.push_back(cm.createData());
    }

    pinocchio::initConstraintDynamics(dyn_model, dyn_data, contact_models);
    pinocchio::ProximalSettings prox_settings(
        options.prox_abs, options.prox_rel, options.prox_mu, options.prox_iters);

    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr joint_pub;
    std::vector<std::string> joint_names;
    if (options.publish_joint_states) {
      joint_pub = node->create_publisher<sensor_msgs::msg::JointState>(
          "joint_states", rclcpp::SystemDefaultsQoS());
      joint_names = BuildOrderedJointNames(robot->GetJointNameAndIndexMap(), n_act);
    }

    const int steps = std::max(1, static_cast<int>(options.duration / options.dt));
    auto next_tick = std::chrono::steady_clock::now();

    for (int step = 0; rclcpp::ok() && step < steps; ++step) {
      const double t = static_cast<double>(step) * options.dt;

      const Eigen::Vector3d base_pos = q_full.head<3>();
      const Eigen::Quaterniond base_quat =
          CoeffsToQuat(q_full.segment<4>(3));
      const Eigen::Matrix3d rot_w_base = base_quat.toRotationMatrix();
      const Eigen::Vector3d base_lin_vel_world = rot_w_base * v_full.segment<3>(0);
      const Eigen::Vector3d base_ang_vel_world = rot_w_base * v_full.segment<3>(3);

      Eigen::VectorXd q_joint = q_full.tail(n_act);
      Eigen::VectorXd qdot_joint = v_full.tail(n_act);

      robot->UpdateRobotModel(base_pos, base_quat, base_lin_vel_world,
                              base_ang_vel_world, q_joint, qdot_joint, false);

      controller.Update(q_joint, qdot_joint, t, options.dt);
      const wbc::RobotCommand command = controller.GetCommand();

      Eigen::VectorXd tau_joint = Eigen::VectorXd::Zero(n_act);
      if (command.tau.size() == n_act) {
        tau_joint = command.tau;
      }
      if (options.zero_torque) {
        tau_joint.setZero();
      }
      ClampTorque(tau_joint, tau_limits);

      Eigen::VectorXd tau_full = Eigen::VectorXd::Zero(n_qdot);
      tau_full.tail(n_act) = tau_joint;

      Eigen::VectorXd qddot = pinocchio::constraintDynamics(
          dyn_model, dyn_data, q_full, v_full, tau_full, contact_models,
          contact_datas, prox_settings);

      if (!qddot.allFinite()) {
        qddot.setZero();
      }
      qddot = qddot.cwiseMax(-options.max_accel).cwiseMin(options.max_accel);

      v_full += qddot * options.dt;
      v_full = v_full.cwiseMax(-options.max_velocity).cwiseMin(options.max_velocity);

      q_full = pinocchio::integrate(dyn_model, q_full, v_full * options.dt);

      Eigen::Quaterniond q_base_next = CoeffsToQuat(q_full.segment<4>(3));
      q_full.segment<4>(3) = q_base_next.coeffs();

      q_joint = q_full.tail(n_act);
      qdot_joint = v_full.tail(n_act);
      ClampToJointLimits(q_joint, pos_limits);
      ClampToVelocityLimits(qdot_joint, vel_limits, options.max_velocity);
      q_full.tail(n_act) = q_joint;
      v_full.tail(n_act) = qdot_joint;

      if (joint_pub != nullptr) {
        sensor_msgs::msg::JointState js;
        js.header.stamp = node->now();
        js.name = joint_names;
        js.position.resize(static_cast<std::size_t>(n_act));
        js.velocity.resize(static_cast<std::size_t>(n_act));
        js.effort.resize(static_cast<std::size_t>(n_act));
        for (int i = 0; i < n_act; ++i) {
          js.position[static_cast<std::size_t>(i)] = q_joint(i);
          js.velocity[static_cast<std::size_t>(i)] = qdot_joint(i);
          js.effort[static_cast<std::size_t>(i)] = tau_joint(i);
        }
        joint_pub->publish(js);
      }

      if (options.print_every > 0 && (step % options.print_every) == 0) {
        const Eigen::Vector3d com = pinocchio::centerOfMass(dyn_model, dyn_data, q_full, v_full);

        double max_drift = 0.0;
        for (const auto& cd : contact_datas) {
          max_drift = std::max(max_drift, cd.c1Mc2.translation().norm());
        }

        std::ostringstream contact_stream;
        int lambda_offset = 0;
        for (std::size_t ci = 0; ci < contact_models.size(); ++ci) {
          const int dim = contact_models[ci].size();
          const Eigen::VectorXd lambda_seg =
              dyn_data.lambda_c.segment(lambda_offset, dim);
          const pinocchio::Force f_local(lambda_seg);
          const double fz = f_local.linear()(2);
          const bool active = fz > options.contact_force_threshold;
          lambda_offset += dim;

          contact_stream << contact_frames[ci] << ":fz=" << std::fixed
                         << std::setprecision(2) << fz
                         << " active=" << (active ? "true" : "false");
          if (ci + 1 < contact_models.size()) {
            contact_stream << ", ";
          }
        }

        std::ostringstream q_stream;
        q_stream << std::fixed << std::setprecision(3) << q_joint.transpose();

        std::ostringstream com_stream;
        com_stream << std::fixed << std::setprecision(3) << com.transpose();

        std::ostringstream line_stream;
        line_stream << "[step " << step << "] t=" << std::fixed
                    << std::setprecision(3) << t
                    << " state=" << controller.CurrentStateId()
                    << " com=[" << com_stream.str() << "]"
                    << " drift=" << std::scientific << std::setprecision(3)
                    << max_drift
                    << " q=[" << q_stream.str() << "]"
                    << " contacts={" << contact_stream.str() << "}";
        std::cout << line_stream.str() << std::endl;
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
