#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <rclcpp/rclcpp.hpp>

#include "can_hardware_common/can_bus_manager.hpp"
#include "plato_hardware_interface/actuator.hpp"
#include "plato_hardware_interface/plato_layout.hpp"
#include "plato_hardware_interface/utils/actuator_config_loader.hpp"
#include "plato_utils/watchdog.hpp"

namespace
{
using namespace std::chrono_literals;

constexpr auto kWatchdogTimeout = 50ms;
constexpr auto kWarmupDuration = 2s;
constexpr auto kTestDuration = 7s;

struct TestCase
{
  std::chrono::microseconds min_inter_frame_gap;
};

struct Stats
{
  uint64_t enqueue_count = 0;
  uint64_t sent_count = 0;
  uint64_t dropped_count = 0;
  uint64_t tx_error_count = 0;
  uint64_t rx_count = 0;
  uint64_t watchdog_expired_events = 0;
  uint64_t drain_bus_error_count = 0;
};
}  // namespace

class CanRateProfilerNode : public rclcpp::Node
{
public:
  CanRateProfilerNode()
  : Node("can_rate_profiler")
  {
    setup_test_cases_();
    setup_actuators_();

    can_bus_manager_.set_rx_handler({this, &CanRateProfilerNode::dispatch_rx_static_});

    // Enable all actuators before profiling.
    enable_actuators_();

    configure_current_test_case_();

    tx_timer_ = create_wall_timer(1ms, [this]() { tx_timer_cb_(); });
    rx_timer_ = create_wall_timer(1ms, [this]() { rx_timer_cb_(); });
    report_timer_ = create_wall_timer(1s, [this]() { report_timer_cb_(); });
    case_timer_ = create_wall_timer(kTestDuration, [this]() { advance_test_case_(); });

    RCLCPP_INFO(get_logger(), "CAN rate profiler started (%zu test cases).", test_cases_.size());
  }

  ~CanRateProfilerNode() override
  {
    disable_actuators_();
  }

private:
  // ── Setup ──

  void setup_test_cases_()
  {
    test_cases_ = {
      {1000us},
      {500us},
      {300us},
      {200us},
      {100us},
      {50us},
      {0us},
    };
  }

  void setup_actuators_()
  {
    const auto configs = plato_actuator::load_plato_actuator_configs();
    actuators_.reserve(configs.size());
    for (const auto & cfg : configs) {
      actuators_.emplace_back(cfg);
    }

    for (size_t i = 0; i < actuators_.size(); ++i) {
      rx_id_to_index_.push_back({actuators_[i].get_rx_id(), i});
    }
    std::sort(
      rx_id_to_index_.begin(), rx_id_to_index_.end(),
      [](const auto & a, const auto & b) { return a.first < b.first; });

    for (auto & wd : rx_watchdogs_) {
      wd = plato_utils::Watchdog(kWatchdogTimeout);
    }
  }

  void enable_actuators_()
  {
    for (size_t i = 0; i < actuators_.size(); ++i) {
      const auto cmd = actuators_[i].enable_motor();
      can_bus_manager_.send_and_wait(
        cmd.frame, cmd.expected_response_opcode, actuators_[i].get_rx_id(), 1000us);
    }
    RCLCPP_INFO(get_logger(), "Actuators enabled.");
  }

  void disable_actuators_()
  {
    for (size_t i = 0; i < actuators_.size(); ++i) {
      const auto cmd = actuators_[i].disable_motor();
      can_bus_manager_.send_and_wait(
        cmd.frame, cmd.expected_response_opcode, actuators_[i].get_rx_id(), 1000us);
    }
    RCLCPP_INFO(get_logger(), "Actuators disabled.");
  }

  void configure_current_test_case_()
  {
    const auto & tc = test_cases_.at(test_case_index_);

    can_bus_manager_.set_tx_gap(tc.min_inter_frame_gap);

    stats_ = {};
    warmup_done_ = false;
    case_start_ = std::chrono::steady_clock::now();

    RCLCPP_INFO(
      get_logger(),
      "── Test case %zu/%zu: gap_us=%ld ──",
      test_case_index_ + 1, test_cases_.size(),
      static_cast<long>(tc.min_inter_frame_gap.count()));
  }

  // ── Timers ──

  void tx_timer_cb_()
  {
    // Send harmless commands, skip on pacing busy.
    for (size_t i = 0; i < actuators_.size(); ++i) {
      TPCANMsg frame{};
      if (plato_hand::layout::is_thumb_servo(i)) {
        frame = actuators_[i].set_servo_position(0.0f, 0U).frame;
      } else {
        frame = actuators_[i].set_joint_torque(0.0f).frame;
      }

      if (!warmup_done_) {
        can_bus_manager_.send_tx_frame(frame);
        continue;
      }

      ++stats_.enqueue_count;
      const TPCANStatus status = can_bus_manager_.send_tx_frame(frame);
      if (status == PCAN_ERROR_OK) {
        ++stats_.sent_count;
      } else if (status == PCAN_ERROR_QXMTFULL) {
        ++stats_.dropped_count;
      } else {
        ++stats_.tx_error_count;
      }
    }
  }

  void rx_timer_cb_()
  {
    const auto rx = can_bus_manager_.poll_rx();

    if (warmup_done_ && rx.is_bus_error()) {
      ++stats_.drain_bus_error_count;
    }

    if (warmup_done_) {
      for (const auto & wd : rx_watchdogs_) {
        if (wd.is_expired()) {
          ++stats_.watchdog_expired_events;
        }
      }
    }

    // Check warmup.
    if (!warmup_done_ &&
      std::chrono::steady_clock::now() - case_start_ > kWarmupDuration)
    {
      warmup_done_ = true;
      stats_ = {};
    }
  }

  void report_timer_cb_()
  {
    if (!warmup_done_) {
      RCLCPP_INFO(get_logger(), "  (warmup...)");
      return;
    }

    const auto & tc = test_cases_.at(test_case_index_);
    RCLCPP_INFO(
      get_logger(),
      "  [gap=%ldus] try=%lu sent=%lu rx=%lu skip=%lu txerr=%lu rxerr=%lu wd=%lu",
      static_cast<long>(tc.min_inter_frame_gap.count()),
      stats_.enqueue_count, stats_.sent_count, stats_.rx_count,
      stats_.dropped_count, stats_.tx_error_count,
      stats_.drain_bus_error_count, stats_.watchdog_expired_events);
  }

  void advance_test_case_()
  {
    ++test_case_index_;
    if (test_case_index_ >= test_cases_.size()) {
      RCLCPP_INFO(get_logger(), "All test cases completed.");
      disable_actuators_();
      rclcpp::shutdown();
      return;
    }

    for (auto & wd : rx_watchdogs_) {
      wd = plato_utils::Watchdog(kWatchdogTimeout);
    }

    configure_current_test_case_();
  }

  // ── RX dispatch ──

  static void dispatch_rx_static_(void * context, const TPCANMsg & frame)
  {
    static_cast<CanRateProfilerNode *>(context)->dispatch_rx_(frame);
  }

  void dispatch_rx_(const TPCANMsg & frame)
  {
    if (frame.MSGTYPE != PCAN_MESSAGE_STANDARD) {
      return;
    }

    const auto it = std::lower_bound(
      rx_id_to_index_.begin(), rx_id_to_index_.end(), frame.ID,
      [](const auto & entry, uint32_t id) { return entry.first < id; });

    if (it == rx_id_to_index_.end() || it->first != frame.ID) {
      return;
    }

    rx_watchdogs_[it->second].kick();
    if (warmup_done_) {
      ++stats_.rx_count;
    }
  }

  // ── Members ──

  can_hardware_common::CanBusManager can_bus_manager_;
  std::vector<plato_actuator::Actuator> actuators_;
  std::vector<std::pair<uint32_t, size_t>> rx_id_to_index_;
  std::array<plato_utils::Watchdog, plato_hand::layout::kNumActuators> rx_watchdogs_{
    plato_utils::Watchdog(kWatchdogTimeout), plato_utils::Watchdog(kWatchdogTimeout),
    plato_utils::Watchdog(kWatchdogTimeout), plato_utils::Watchdog(kWatchdogTimeout),
    plato_utils::Watchdog(kWatchdogTimeout), plato_utils::Watchdog(kWatchdogTimeout),
    plato_utils::Watchdog(kWatchdogTimeout), plato_utils::Watchdog(kWatchdogTimeout)};

  std::vector<TestCase> test_cases_;
  size_t test_case_index_ = 0;
  Stats stats_;
  bool warmup_done_ = false;
  std::chrono::steady_clock::time_point case_start_;

  rclcpp::TimerBase::SharedPtr tx_timer_;
  rclcpp::TimerBase::SharedPtr rx_timer_;
  rclcpp::TimerBase::SharedPtr report_timer_;
  rclcpp::TimerBase::SharedPtr case_timer_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<CanRateProfilerNode>());
  rclcpp::shutdown();
  return 0;
}
