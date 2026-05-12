#ifndef PLATO_HARDWARE_INTERFACE__UTILS__LINKAGE_CONFIG_LOADER_HPP_
#define PLATO_HARDWARE_INTERFACE__UTILS__LINKAGE_CONFIG_LOADER_HPP_

#include <string>

#include "plato_hardware_interface/five_bar_linkage.hpp"

namespace FiveBarLinkage
{

FiveBarLinkageConfig load_plato_linkage_config();
FiveBarLinkageConfig load_plato_linkage_config(const std::string & yaml_path);

}  // namespace FiveBarLinkage

#endif  // PLATO_HARDWARE_INTERFACE__UTILS__LINKAGE_CONFIG_LOADER_HPP_
