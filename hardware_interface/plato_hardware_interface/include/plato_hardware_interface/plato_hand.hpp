#ifndef PLATO_HARDWARE_INTERFACE__PLATO_HAND_HPP_
#define PLATO_HARDWARE_INTERFACE__PLATO_HAND_HPP_

#include <chrono>
#include <mutex>
#include <vector>

#include "can_hardware_common/can_bus.hpp"
#include "can_hardware_common/core/actuator_frame_router.hpp"
#include "can_hardware_common/core/can_frame_executor.hpp"
#include "can_hardware_common/core/can_hand_base.hpp"
#include "plato_hardware_interface/actuator.hpp"
#include "plato_hardware_interface/five_bar_linkage.hpp"
#include "plato_hardware_interface/plato_hand_config.hpp"
#include "plato_hardware_interface/plato_layout.hpp"
#include "plato_hardware_interface/plato_state_helper.hpp"

namespace plato_hand
{

class Hand : public can_hardware_common::core::CanHandBase
{
public:
  static constexpr size_t kNumJoints = layout::kNumJoints;
  static constexpr size_t kNumActuators = layout::kNumActuators;

  explicit Hand(PlatoHandConfig config);
  Hand(const Hand &) = delete;
  Hand & operator=(const Hand &) = delete;
  Hand(Hand &&) = delete;
  Hand & operator=(Hand &&) = delete;
  ~Hand();

  bool write_joint_commands() { return write(); }

  void print_motor_positions();

private:
  using LifecyclePlan = can_hardware_common::core::LifecyclePlan;
  using WritePlan = can_hardware_common::core::WritePlan;

  static constexpr std::chrono::milliseconds kRxStaleTimeout{20};
  static constexpr std::chrono::microseconds kResponseTimeout{25000};
  static constexpr size_t kZeroingProbeRounds = 3;

  bool update_measurements_() override;
  void refresh_state_snapshot_() override;
  void build_ready_write_plan_(WritePlan & plan) override;
  bool execute_write_plan_(const WritePlan & plan) override;
  LifecyclePlan build_lifecycle_plan_(can_hardware_common::core::LifecycleOperation operation) override;
  bool execute_lifecycle_plan_(const LifecyclePlan & plan) override;
  bool execute_standard_lifecycle_(const LifecyclePlan & plan);
  bool execute_zero_lifecycle_();
  bool zero_actuators_();
  bool run_zeroing_probe_rounds_();
  bool capture_zero_offsets_(std::vector<float> & offsets);
  bool persist_zero_offsets_(const std::vector<float> & offsets, bool zeroing_success) const;

  can_hardware_common::CanBus transport_;
  can_hardware_common::core::CanFrameExecutor frame_executor_;
  FiveBarLinkage::Transmission transmission_;
  PlatoStateHelper state_helper_;
  mutable std::mutex state_mutex_;
  std::vector<plato_actuator::Actuator> actuators_;

  std::string actuator_offset_yaml_path_;
  const std::vector<plato_actuator::StaticConfig> actuator_static_configs_;
  std::vector<float> actuator_position_offsets_;
  std::chrono::microseconds direct_tx_frame_timeout_{std::chrono::microseconds(2000)};
  double servo_stiffness_scale_ = 0.0;
  bool disable_on_destruction_ = true;
  bool enable_requested_ = false;
  bool disable_requested_ = false;
};

}  // namespace plato_hand

#endif  // PLATO_HARDWARE_INTERFACE__PLATO_HAND_HPP_
