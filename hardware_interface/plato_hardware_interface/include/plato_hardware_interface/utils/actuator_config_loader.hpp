#ifndef PLATO_HARDWARE_INTERFACE__UTILS__ACTUATOR_CONFIG_LOADER_HPP_
#define PLATO_HARDWARE_INTERFACE__UTILS__ACTUATOR_CONFIG_LOADER_HPP_

#include <array>
#include <string>
#include <vector>

#include "plato_hardware_interface/actuator.hpp"
#include "plato_hardware_interface/plato_layout.hpp"

namespace plato_actuator
{

const std::array<const char *, plato_hand::layout::kNumActuators> &
expected_plato_actuator_names();

std::vector<Config> load_plato_actuator_configs();
std::vector<Config> load_plato_actuator_configs(const std::string & yaml_path);

}  // namespace plato_actuator

#endif  // PLATO_HARDWARE_INTERFACE__UTILS__ACTUATOR_CONFIG_LOADER_HPP_
