#include "can_hardware_common/can_bus.hpp"
#include "can_hardware_common/command_scheduler.hpp"
#include "plato_hardware_interface/actuator.hpp"
#include "plato_hardware_interface/utils/plato_hand_config_loader.hpp"

#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <rclcpp/rclcpp.hpp>

namespace
{
using Scheduler = can_hardware_common::CanCommandScheduler;
using ResultStatus = Scheduler::TransactionResult::Status;

struct Options
{
  std::vector<int> gaps_us{1000, 500, 300, 200, 100, 50, 0};
  int timeout_ms = 40;
  int retries = 2;
  int cycles = 200;
  int loop_period_us = 10000;
  int target_cycle_us = 0;
  float servo_position_rad = 0.0f;
  bool disable_on_exit = true;
};

struct CaseStats
{
  size_t rx_frames = 0;
  size_t start_deadline_miss_count = 0;
  size_t finish_deadline_miss_count = 0;
  std::chrono::microseconds max_start_lateness{0};
  std::chrono::microseconds max_finish_overrun{0};
  std::chrono::microseconds max_cycle_work_time{0};
  std::chrono::microseconds total_elapsed{0};
};

bool parse_gap_list(const std::string & raw, std::vector<int> & gaps_us)
{
  std::vector<int> parsed;
  std::stringstream ss(raw);
  std::string item;
  while (std::getline(ss, item, ',')) {
    if (item.empty()) {
      continue;
    }
    parsed.push_back(std::stoi(item));
  }
  if (parsed.empty()) {
    return false;
  }
  gaps_us = std::move(parsed);
  return true;
}

bool parse_options(int argc, char ** argv, Options & options)
{
  std::vector<std::string> args(argv + 1, argv + argc);
  for (size_t i = 0; i < args.size(); ++i) {
    if (args[i] == "--gaps-us" && i + 1 < args.size()) {
      if (!parse_gap_list(args[++i], options.gaps_us)) {
        return false;
      }
      continue;
    }
    if (args[i] == "--timeout-ms" && i + 1 < args.size()) {
      options.timeout_ms = std::stoi(args[++i]);
      continue;
    }
    if (args[i] == "--retries" && i + 1 < args.size()) {
      options.retries = std::stoi(args[++i]);
      continue;
    }
    if (args[i] == "--cycles" && i + 1 < args.size()) {
      options.cycles = std::stoi(args[++i]);
      continue;
    }
    if (args[i] == "--loop-period-us" && i + 1 < args.size()) {
      options.loop_period_us = std::stoi(args[++i]);
      continue;
    }
    if (args[i] == "--target-cycle-us" && i + 1 < args.size()) {
      options.target_cycle_us = std::stoi(args[++i]);
      continue;
    }
    if (args[i] == "--target-cycle-hz" && i + 1 < args.size()) {
      const double hz = std::stod(args[++i]);
      if (!(hz > 0.0)) {
        return false;
      }
      options.target_cycle_us = static_cast<int>(std::llround(1.0e6 / hz));
      continue;
    }
    if (args[i] == "--servo-pos-rad" && i + 1 < args.size()) {
      options.servo_position_rad = std::stof(args[++i]);
      continue;
    }
    if (args[i] == "--keep-enabled") {
      options.disable_on_exit = false;
      continue;
    }
    if (args[i] == "--help" || args[i] == "-h") {
      std::cout
        << "Usage: plato_hand_hil_gap_sweep [options]\n"
        << "  --gaps-us <csv>          comma-separated gap sweep (default: 1000,500,300,200,100,50,0)\n"
        << "  --timeout-ms <int>       blocking transaction timeout in ms (default: 40)\n"
        << "  --retries <int>          blocking transaction retries (default: 2)\n"
        << "  --cycles <int>           streaming write/read cycles per gap (default: 200)\n"
        << "  --loop-period-us <int>   extra delay after each cycle when no target cycle is set"
        << " (default: 10000)\n"
        << "  --target-cycle-us <int>  fixed cycle period target; reports deadline misses\n"
        << "  --target-cycle-hz <num>  fixed cycle frequency target (e.g. 300)\n"
        << "  --servo-pos-rad <float>  servo hold target during sweep (default: 0.0)\n"
        << "  --keep-enabled           skip disable commands at the end of each gap case\n";
      return false;
    }
    std::cerr << "Unknown argument: " << args[i] << "\n";
    return false;
  }
  return true;
}

std::string join_indices(const std::vector<size_t> & indices)
{
  std::ostringstream oss;
  for (size_t i = 0; i < indices.size(); ++i) {
    if (i != 0) {
      oss << ",";
    }
    oss << (indices[i] + 1);
  }
  return oss.str();
}

bool enable_all(
  Scheduler & scheduler,
  std::vector<plato_actuator::Actuator> & actuators,
  std::chrono::microseconds timeout,
  size_t retries)
{
  size_t tolerated_already_enabled = 0;
  for (size_t i = 0; i < actuators.size(); ++i) {
    const auto req = actuators[i].make_enable_request(static_cast<uint32_t>(i));
    const auto res = scheduler.execute_blocking(req, timeout, retries);
    const bool confirmed = res.status == ResultStatus::kConfirmed;
    const bool tolerated_rejected =
      res.status == ResultStatus::kRejected && res.result_byte == 0x01;
    if (!(confirmed || tolerated_rejected)) {
      std::cerr << "[gap_sweep] enable failed for actuator " << (i + 1) << "\n";
      return false;
    }
    if (tolerated_rejected) {
      ++tolerated_already_enabled;
    }
  }
  if (tolerated_already_enabled > 0) {
    std::cout
      << "  tolerated_already_enabled=" << tolerated_already_enabled
      << "\n";
  }
  return true;
}

void disable_all(
  Scheduler & scheduler,
  std::vector<plato_actuator::Actuator> & actuators,
  std::chrono::microseconds timeout,
  size_t retries)
{
  for (size_t i = 0; i < actuators.size(); ++i) {
    const auto req = actuators[i].make_disable_request(static_cast<uint32_t>(100 + i));
    (void)scheduler.execute_blocking(req, timeout, retries);
  }
}

std::vector<size_t> initialized_indices(const std::vector<plato_actuator::Actuator> & actuators)
{
  std::vector<size_t> indices;
  for (size_t i = 0; i < actuators.size(); ++i) {
    if (actuators[i].is_initialized()) {
      indices.push_back(i);
    }
  }
  return indices;
}

void send_streaming_frame(
  can_hardware_common::CanBus & transport,
  const TPCANMsg & frame)
{
  const auto next_send = transport.next_tx_time();
  const auto now = std::chrono::steady_clock::now();
  if (next_send > now) {
    std::this_thread::sleep_until(next_send);
  }
  (void)transport.send_tx_frame(frame);
}

std::chrono::microseconds to_microseconds(std::chrono::steady_clock::duration duration)
{
  return std::chrono::duration_cast<std::chrono::microseconds>(duration);
}

void run_gap_case(const Options & options, int gap_us)
{
  auto config = plato_hand::load_default_plato_hand_config();
  std::vector<plato_actuator::Actuator> actuators;
  actuators.reserve(config.actuator_configs.size());
  for (const auto & cfg : config.actuator_configs) {
    actuators.emplace_back(cfg);
  }

  can_hardware_common::CanBus transport;
  transport.set_tx_gap(std::chrono::microseconds(gap_us));
  transport.add_rx_observer([&actuators](const TPCANMsg & frame) {
    for (auto & actuator : actuators) {
      if (actuator.get_rx_id() == frame.ID) {
        actuator.process_rx_frame(frame);
        return;
      }
    }
  });

  Scheduler scheduler(transport);
  const auto timeout = std::chrono::milliseconds(options.timeout_ms);
  const auto retries = static_cast<size_t>(options.retries);

  std::cout << "[CASE] gap_us=" << gap_us << "\n";
  if (!enable_all(scheduler, actuators, timeout, retries)) {
    std::cout << "  enable=FAIL\n";
    return;
  }

  size_t rx_frames = 0;
  CaseStats stats;
  const auto target_cycle_period = std::chrono::microseconds(options.target_cycle_us);
  auto scheduled_cycle_start = std::chrono::steady_clock::now();
  const auto case_start = scheduled_cycle_start;
  for (int cycle = 0; cycle < options.cycles; ++cycle) {
    if (options.target_cycle_us > 0) {
      const auto now = std::chrono::steady_clock::now();
      if (now < scheduled_cycle_start) {
        std::this_thread::sleep_until(scheduled_cycle_start);
      }
    }

    const auto cycle_start = std::chrono::steady_clock::now();
    if (options.target_cycle_us > 0 && cycle_start > scheduled_cycle_start) {
      ++stats.start_deadline_miss_count;
      stats.max_start_lateness = std::max(
        stats.max_start_lateness,
        to_microseconds(cycle_start - scheduled_cycle_start));
    }

    for (size_t i = 0; i < actuators.size(); ++i) {
      TPCANMsg frame{};
      if (i < 2) {
        frame = actuators[i].set_servo_hold(options.servo_position_rad).frame;
      } else {
        frame = actuators[i].set_joint_torque(0.0f).frame;
      }
      send_streaming_frame(transport, frame);
    }

    const auto rx = transport.poll_rx();
    rx_frames += rx.processed_frames;
    const auto cycle_end = std::chrono::steady_clock::now();
    stats.max_cycle_work_time = std::max(
      stats.max_cycle_work_time,
      to_microseconds(cycle_end - cycle_start));

    if (options.target_cycle_us > 0) {
      const auto cycle_deadline = scheduled_cycle_start + target_cycle_period;
      if (cycle_end > cycle_deadline) {
        ++stats.finish_deadline_miss_count;
        stats.max_finish_overrun = std::max(
          stats.max_finish_overrun,
          to_microseconds(cycle_end - cycle_deadline));
      }
      scheduled_cycle_start += target_cycle_period;
    } else {
      std::this_thread::sleep_for(std::chrono::microseconds(options.loop_period_us));
    }
  }
  stats.rx_frames = rx_frames;
  stats.total_elapsed = to_microseconds(std::chrono::steady_clock::now() - case_start);

  const auto initialized = initialized_indices(actuators);
  std::cout
    << "  initialized=[" << join_indices(initialized) << "]"
    << " rx_frames=" << rx_frames
    << " total_init=" << initialized.size() << "/" << actuators.size() << "\n";
  if (options.target_cycle_us > 0) {
    const double achieved_hz = stats.total_elapsed.count() > 0
      ? (static_cast<double>(options.cycles) * 1.0e6) / static_cast<double>(stats.total_elapsed.count())
      : std::numeric_limits<double>::quiet_NaN();
    std::cout
      << std::fixed << std::setprecision(1)
      << "  target_cycle_us=" << options.target_cycle_us
      << " achieved_hz=" << achieved_hz
      << " start_miss=" << stats.start_deadline_miss_count
      << " finish_miss=" << stats.finish_deadline_miss_count
      << " max_start_late_us=" << stats.max_start_lateness.count()
      << " max_finish_overrun_us=" << stats.max_finish_overrun.count()
      << " max_cycle_work_us=" << stats.max_cycle_work_time.count()
      << "\n";
  }

  if (options.disable_on_exit) {
    disable_all(scheduler, actuators, timeout, retries);
  }
}

}  // namespace

int main(int argc, char ** argv)
{
  Options options;
  if (!parse_options(argc, argv, options)) {
    return 2;
  }

  rclcpp::init(argc, argv);
  try {
    for (const int gap_us : options.gaps_us) {
      run_gap_case(options, gap_us);
    }
  } catch (const std::exception & e) {
    std::cerr << "[gap_sweep] exception: " << e.what() << "\n";
    rclcpp::shutdown();
    return 1;
  }

  rclcpp::shutdown();
  return 0;
}
