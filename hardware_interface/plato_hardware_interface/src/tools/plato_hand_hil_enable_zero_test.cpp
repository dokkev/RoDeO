#include "can_hardware_common/can_bus.hpp"
#include "can_hardware_common/command_scheduler.hpp"
#include "plato_hardware_interface/actuator.hpp"
#include "plato_hardware_interface/utils/plato_hand_config_loader.hpp"

#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>

namespace
{
enum class CaseStatus
{
  kPass,
  kFail
};

struct Summary
{
  int passed = 0;
  int failed = 0;
};

struct Options
{
  int timeout_ms = 40;
  int retries = 2;
  int inter_frame_gap_us = 100;
  float servo_position_rad = 0.0f;
  bool disable_on_exit = false;
  bool enable_servo_only = false;
  bool strict_enable = false;
};

CaseStatus report_result(
  const std::string & case_name,
  CaseStatus status,
  const std::string & detail = "")
{
  const char * label = (status == CaseStatus::kPass) ? "PASS" : "FAIL";
  std::cout << "[" << label << "] " << case_name;
  if (!detail.empty()) {
    std::cout << ": " << detail;
  }
  std::cout << "\n";
  return status;
}

void update_summary(Summary & summary, CaseStatus status)
{
  if (status == CaseStatus::kPass) {
    ++summary.passed;
  } else {
    ++summary.failed;
  }
}

bool parse_options(int argc, char ** argv, Options & options)
{
  std::vector<std::string> args(argv + 1, argv + argc);
  for (size_t i = 0; i < args.size(); ++i) {
    if (args[i] == "--timeout-ms" && i + 1 < args.size()) {
      options.timeout_ms = std::stoi(args[++i]);
      continue;
    }
    if (args[i] == "--retries" && i + 1 < args.size()) {
      options.retries = std::stoi(args[++i]);
      continue;
    }
    if (args[i] == "--gap-us" && i + 1 < args.size()) {
      options.inter_frame_gap_us = std::stoi(args[++i]);
      continue;
    }
    if (args[i] == "--servo-pos-rad" && i + 1 < args.size()) {
      options.servo_position_rad = std::stof(args[++i]);
      continue;
    }
    if (args[i] == "--disable-on-exit") {
      options.disable_on_exit = true;
      continue;
    }
    if (args[i] == "--enable-servo-only") {
      options.enable_servo_only = true;
      continue;
    }
    if (args[i] == "--strict-enable") {
      options.strict_enable = true;
      continue;
    }
    if (args[i] == "--help" || args[i] == "-h") {
      std::cout
        << "Usage: plato_hand_hil_enable_zero_test [options]\n"
        << "  --timeout-ms <int>       transaction timeout in ms (default: 40)\n"
        << "  --retries <int>          transaction retries (default: 2)\n"
        << "  --gap-us <int>           inter-frame TX pacing gap in us (default: 100)\n"
        << "  --servo-pos-rad <float>  servo idle target position in rad (default: 0.0)\n"
        << "  --enable-servo-only      enable only servo channels (0x21,0x22)\n"
        << "  --strict-enable          fail if START_MOTOR returns rejected/0x01\n"
        << "  --disable-on-exit        send disable command at the end (optional)\n";
      return false;
    }

    std::cerr << "Unknown argument: " << args[i] << "\n";
    return false;
  }
  return true;
}

std::string hex_u32(uint32_t value)
{
  std::ostringstream oss;
  oss << "0x" << std::uppercase << std::hex << value;
  return oss.str();
}

std::string result_detail(
  const can_hardware_common::CanCommandScheduler::TransactionResult & result)
{
  std::string detail =
    std::string(can_hardware_common::CanCommandScheduler::TransactionResult::status_label(
      result.status));
  if (result.status == can_hardware_common::CanCommandScheduler::TransactionResult::Status::kTransportError) {
    detail += " transport=" + can_hardware_common::CanBus::bus_status_string(
      result.transport_status);
  }
  if (result.result_byte != 0xFF) {
    detail += " result_byte=" + hex_u32(result.result_byte);
  }
  return detail;
}

CaseStatus run_enable_case(
  can_hardware_common::CanCommandScheduler & scheduler,
  std::vector<plato_actuator::Actuator> & actuators,
  std::chrono::microseconds timeout,
  std::size_t retries,
  bool servo_only,
  bool strict_enable)
{
  std::vector<size_t> indices;
  if (servo_only) {
    indices = {0, 1};
  } else {
    indices.resize(actuators.size());
    for (size_t i = 0; i < actuators.size(); ++i) {
      indices[i] = i;
    }
  }

  std::size_t success_count = 0;
  std::size_t tolerated_rejected = 0;
  for (const size_t i : indices) {
    const auto req = actuators[i].make_enable_request(static_cast<uint32_t>(i));
    const auto res = scheduler.execute_blocking(req, timeout, retries);
    const bool confirmed =
      res.status == can_hardware_common::CanCommandScheduler::TransactionResult::Status::kConfirmed;
    const bool tolerated_already_enabled =
      !strict_enable &&
      res.status == can_hardware_common::CanCommandScheduler::TransactionResult::Status::kRejected &&
      res.result_byte == 0x01;

    if (!(confirmed || tolerated_already_enabled)) {
      return report_result(
        "enable",
        CaseStatus::kFail,
        "actuator rx=" + hex_u32(actuators[i].get_rx_id()) + " " + result_detail(res));
    }
    if (tolerated_already_enabled) {
      ++tolerated_rejected;
    }
    ++success_count;
  }

  return report_result(
    "enable",
    CaseStatus::kPass,
    "confirmed_or_tolerated " + std::to_string(success_count) + "/" + std::to_string(indices.size()) +
    " (tolerated_rejected=" + std::to_string(tolerated_rejected) + ")");
}

CaseStatus run_servo_zero_current_case(
  can_hardware_common::CanCommandScheduler & scheduler,
  std::vector<plato_actuator::Actuator> & actuators,
  std::chrono::microseconds timeout,
  std::size_t retries,
  float servo_position_rad)
{
  const std::vector<size_t> servo_indices = {0, 1};
  for (const size_t i : servo_indices) {
    auto cmd = actuators[i].set_servo_idle(servo_position_rad);

    can_hardware_common::CommandRequest req;
    req.key = static_cast<uint32_t>(100 + i);
    req.frame = cmd.frame;
    req.reply.expected_rx_id = actuators[i].get_rx_id();
    req.reply.expected_opcode = cmd.expected_response_opcode;
    req.reply.success_byte = 0x00;

    const auto res = scheduler.execute_blocking(req, timeout, retries);
    if (res.status != can_hardware_common::CanCommandScheduler::TransactionResult::Status::kConfirmed) {
      return report_result(
        "servo_zero_current",
        CaseStatus::kFail,
        "actuator rx=" + hex_u32(actuators[i].get_rx_id()) + " " + result_detail(res));
    }

    if (!actuators[i].is_initialized()) {
      return report_result(
        "servo_zero_current",
        CaseStatus::kFail,
        "actuator rx=" + hex_u32(actuators[i].get_rx_id()) + " reply parsed but no state");
    }
  }

  return report_result("servo_zero_current", CaseStatus::kPass, "both servo replies confirmed");
}

CaseStatus run_disable_case(
  can_hardware_common::CanCommandScheduler & scheduler,
  std::vector<plato_actuator::Actuator> & actuators,
  std::chrono::microseconds timeout,
  std::size_t retries,
  bool servo_only)
{
  std::vector<size_t> indices;
  if (servo_only) {
    indices = {0, 1};
  } else {
    indices.resize(actuators.size());
    for (size_t i = 0; i < actuators.size(); ++i) {
      indices[i] = i;
    }
  }

  for (const size_t i : indices) {
    const auto req = actuators[i].make_disable_request(static_cast<uint32_t>(200 + i));
    const auto res = scheduler.execute_blocking(req, timeout, retries);
    if (res.status != can_hardware_common::CanCommandScheduler::TransactionResult::Status::kConfirmed) {
      return report_result(
        "disable",
        CaseStatus::kFail,
        "actuator rx=" + hex_u32(actuators[i].get_rx_id()) + " " + result_detail(res));
    }
  }

  return report_result("disable", CaseStatus::kPass);
}

}  // namespace

int main(int argc, char ** argv)
{
  Options options;
  if (!parse_options(argc, argv, options)) {
    return 2;
  }

  rclcpp::init(argc, argv);

  Summary summary;
  try {
    can_hardware_common::CanBus transport;
    transport.set_tx_gap(std::chrono::microseconds(options.inter_frame_gap_us));
    can_hardware_common::CanCommandScheduler scheduler(transport);

    auto config = plato_hand::load_default_plato_hand_config();
    std::vector<plato_actuator::Actuator> actuators;
    actuators.reserve(config.actuator_configs.size());
    for (const auto & cfg : config.actuator_configs) {
      actuators.emplace_back(cfg);
    }

    transport.add_rx_observer([&actuators](const TPCANMsg & frame) {
      for (auto & actuator : actuators) {
        if (actuator.get_rx_id() == frame.ID) {
          actuator.process_rx_frame(frame);
          return;
        }
      }
    });

    const auto timeout = std::chrono::milliseconds(options.timeout_ms);
    const auto retries = static_cast<std::size_t>(options.retries);

    update_summary(summary, run_enable_case(
        scheduler, actuators, timeout, retries, options.enable_servo_only, options.strict_enable));
    update_summary(summary, run_servo_zero_current_case(
        scheduler, actuators, timeout, retries, options.servo_position_rad));
    if (options.disable_on_exit) {
      update_summary(summary, run_disable_case(
          scheduler, actuators, timeout, retries, options.enable_servo_only));
    }
  } catch (const std::exception & e) {
    std::cerr << "[FAIL] init: " << e.what() << "\n";
    rclcpp::shutdown();
    return 1;
  }

  std::cout << "[SUMMARY] passed=" << summary.passed
            << " failed=" << summary.failed << "\n";
  rclcpp::shutdown();
  return summary.failed == 0 ? 0 : 1;
}
