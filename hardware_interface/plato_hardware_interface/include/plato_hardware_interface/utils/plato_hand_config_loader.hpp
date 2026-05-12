#ifndef PLATO_HARDWARE_INTERFACE__UTILS__PLATO_HAND_CONFIG_LOADER_HPP_
#define PLATO_HARDWARE_INTERFACE__UTILS__PLATO_HAND_CONFIG_LOADER_HPP_

#include <string>

#include "plato_hardware_interface/plato_hand_config.hpp"

namespace plato_hand
{

PlatoHandConfig load_default_plato_hand_config();
PlatoHandConfig load_plato_hand_config(const std::string & config_dir);

}  // namespace plato_hand

#endif  // PLATO_HARDWARE_INTERFACE__UTILS__PLATO_HAND_CONFIG_LOADER_HPP_
