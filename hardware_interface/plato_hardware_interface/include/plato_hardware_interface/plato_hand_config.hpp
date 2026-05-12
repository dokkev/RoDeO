#ifndef PLATO_HARDWARE_INTERFACE__PLATO_HAND_CONFIG_HPP_
#define PLATO_HARDWARE_INTERFACE__PLATO_HAND_CONFIG_HPP_

#include <chrono>
#include <string>
#include <vector>

#include "plato_hardware_interface/actuator.hpp"
#include "plato_hardware_interface/five_bar_linkage.hpp"

namespace plato_hand
{

struct PlatoHandConfig
{
  std::vector<plato_actuator::Config> actuator_configs;
  FiveBarLinkage::FiveBarLinkageConfig linkage_config;
  std::string actuator_offset_yaml_path;
  std::chrono::microseconds direct_tx_inter_frame_gap{std::chrono::microseconds(100)};
  double servo_stiffness_scale{0.0};
  bool disable_on_destruction{true};
};

}  // namespace plato_hand

#endif  // PLATO_HARDWARE_INTERFACE__PLATO_HAND_CONFIG_HPP_
