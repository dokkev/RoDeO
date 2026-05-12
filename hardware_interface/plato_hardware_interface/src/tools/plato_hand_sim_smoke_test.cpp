#include "plato_hardware_interface/plato_hand.hpp"

#include <chrono>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <string>
#include <thread>
#include <vector>

#include <rclcpp/rclcpp.hpp>

#include "plato_hardware_interface/utils/plato_hand_config_loader.hpp"

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
  int cycles = 30;
  int loop_period_us = 2000;
  int direct_gap_us = 1;
  bool bypass_hardware = true;
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
    if (args[i] == "--cycles" && i + 1 < args.size()) {
      options.cycles = std::stoi(args[++i]);
      continue;
    }
    if (args[i] == "--loop-period-us" && i + 1 < args.size()) {
      options.loop_period_us = std::stoi(args[++i]);
      continue;
    }
    if (args[i] == "--direct-gap-us" && i + 1 < args.size()) {
      options.direct_gap_us = std::stoi(args[++i]);
      continue;
    }
    if (args[i] == "--no-bypass-hardware") {
      options.bypass_hardware = false;
      continue;
    }
    if (args[i] == "--help" || args[i] == "-h") {
      std::cout
        << "Usage: plato_hand_sim_smoke_test [options]\n"
        << "  --cycles <int>              write/read cycles (default: 30)\n"
        << "  --loop-period-us <int>      loop period for cycle test (default: 2000)\n"
        << "  --direct-gap-us <int>       direct TX inter-frame gap (default: 1)\n"
        << "  --no-bypass-hardware        keep real CAN write/read enabled in transport\n";
      return false;
    }

    std::cerr << "Unknown argument: " << args[i] << "\n";
    return false;
  }
  return true;
}

void set_nominal_joint_commands(plato_hand::Hand & hand, double phase)
{
  auto & cmd = hand.joint_commands();
  for (size_t i = 0; i < plato_hand::Hand::kNumJoints; ++i) {
    const double q = 0.05 * std::sin(phase + static_cast<double>(i) * 0.25);
    const double tau = 0.35 * std::cos(phase + static_cast<double>(i) * 0.2);
    cmd.position_at(i) = q;
    cmd.velocity_at(i) = 0.0;
    cmd.effort_at(i) = tau;
    cmd.stiffness_at(i) = 1.0;
    cmd.damping_at(i) = 0.1;
  }
}

CaseStatus run_enable_case(plato_hand::Hand & hand)
{
  if (!hand.enable(false)) {
    return report_result("enable", CaseStatus::kFail, "failed to enable one or more actuators");
  }
  return report_result("enable", CaseStatus::kPass);
}

CaseStatus run_cycle_case(plato_hand::Hand & hand, const Options & options)
{
  for (int cycle = 0; cycle < options.cycles; ++cycle) {
    set_nominal_joint_commands(hand, 0.1 * static_cast<double>(cycle));

    if (!hand.write_joint_commands()) {
      return report_result(
        "write_read_cycles",
        CaseStatus::kFail,
        "write_joint_commands failed at cycle " + std::to_string(cycle));
    }

    if (!hand.read()) {
      return report_result(
        "write_read_cycles",
        CaseStatus::kFail,
        "read failed at cycle " + std::to_string(cycle));
    }

    std::this_thread::sleep_for(std::chrono::microseconds(options.loop_period_us));
  }

  const auto & joint_state = hand.joint_states();
  const auto & actuator_state = hand.actuator_states();
  if (!joint_state.all_finite() || !actuator_state.all_finite()) {
    return report_result("write_read_cycles", CaseStatus::kFail, "non-finite joint/actuator state");
  }

  if (!std::isfinite(actuator_state.position(2)) || !std::isfinite(actuator_state.position(3))) {
    return report_result("write_read_cycles", CaseStatus::kFail, "actuator 0x24/0x23 state missing");
  }

  return report_result("write_read_cycles", CaseStatus::kPass);
}

CaseStatus run_non_finite_guard_case(plato_hand::Hand & hand)
{
  set_nominal_joint_commands(hand, 0.0);
  hand.joint_commands().effort_at(0) = std::numeric_limits<double>::quiet_NaN();

  if (!hand.write_joint_commands()) {
    return report_result("non_finite_guard", CaseStatus::kFail, "write returned false");
  }

  if (!hand.read()) {
    return report_result("non_finite_guard", CaseStatus::kFail, "read failed after non-finite write");
  }

  set_nominal_joint_commands(hand, 0.0);
  return report_result("non_finite_guard", CaseStatus::kPass);
}

CaseStatus run_history_case(const plato_hand::Hand & hand)
{
  if (!hand.has_previous_joint_command() || !hand.has_previous_actuator_command()) {
    return report_result("command_history", CaseStatus::kFail, "previous command history missing");
  }

  if (!hand.previous_joint_command().all_finite()) {
    return report_result("command_history", CaseStatus::kFail, "previous joint command is non-finite");
  }
  if (!hand.previous_actuator_command().all_finite()) {
    return report_result(
      "command_history",
      CaseStatus::kFail,
      "previous actuator command is non-finite");
  }

  return report_result("command_history", CaseStatus::kPass);
}

CaseStatus run_23_24_update_case(plato_hand::Hand & hand, const Options & options)
{
  const auto & before = hand.actuator_states();
  const double before_24 = before.position(2);
  const double before_23 = before.position(3);

  auto & cmd = hand.joint_commands();
  for (size_t i = 0; i < plato_hand::Hand::kNumJoints; ++i) {
    cmd.position_at(i) = 0.0;
    cmd.velocity_at(i) = 0.0;
    cmd.effort_at(i) = 0.0;
    cmd.stiffness_at(i) = 1.0;
    cmd.damping_at(i) = 0.1;
  }
  cmd.effort_at(2) = 1.1;
  cmd.effort_at(3) = -1.1;

  for (int cycle = 0; cycle < options.cycles; ++cycle) {
    if (!hand.write_joint_commands()) {
      return report_result(
        "motor_23_24_update",
        CaseStatus::kFail,
        "write failed at cycle " + std::to_string(cycle));
    }
    if (!hand.read()) {
      return report_result(
        "motor_23_24_update",
        CaseStatus::kFail,
        "read failed at cycle " + std::to_string(cycle));
    }
    std::this_thread::sleep_for(std::chrono::microseconds(options.loop_period_us));
  }

  const auto & after = hand.actuator_states();
  const double after_24 = after.position(2);
  const double after_23 = after.position(3);

  if (!std::isfinite(before_24) || !std::isfinite(before_23) ||
    !std::isfinite(after_24) || !std::isfinite(after_23))
  {
    return report_result("motor_23_24_update", CaseStatus::kFail, "non-finite 0x24/0x23 state");
  }

  const double delta_24 = std::abs(after_24 - before_24);
  const double delta_23 = std::abs(after_23 - before_23);
  if (delta_24 < 1e-4 || delta_23 < 1e-4) {
    return report_result(
      "motor_23_24_update",
      CaseStatus::kFail,
      "state delta too small (delta_24=" + std::to_string(delta_24) +
      ", delta_23=" + std::to_string(delta_23) + ")");
  }

  return report_result(
    "motor_23_24_update",
    CaseStatus::kPass,
    "delta_24=" + std::to_string(delta_24) + ", delta_23=" + std::to_string(delta_23));
}

CaseStatus run_second_enable_case(plato_hand::Hand & hand)
{
  if (!hand.enable(false)) {
    return report_result("second_enable", CaseStatus::kFail, "second enable failed");
  }
  return report_result("second_enable", CaseStatus::kPass);
}

CaseStatus run_disable_case(plato_hand::Hand & hand)
{
  if (!hand.disable()) {
    return report_result("disable", CaseStatus::kFail, "failed to disable one or more actuators");
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
    auto config = plato_hand::load_default_plato_hand_config();
    config.direct_tx_inter_frame_gap = std::chrono::microseconds(options.direct_gap_us);

    plato_hand::Hand hand(std::move(config));

    update_summary(summary, run_enable_case(hand));
    update_summary(summary, run_cycle_case(hand, options));
    update_summary(summary, run_non_finite_guard_case(hand));
    update_summary(summary, run_history_case(hand));
    update_summary(summary, run_23_24_update_case(hand, options));
    update_summary(summary, run_second_enable_case(hand));
    update_summary(summary, run_disable_case(hand));
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
