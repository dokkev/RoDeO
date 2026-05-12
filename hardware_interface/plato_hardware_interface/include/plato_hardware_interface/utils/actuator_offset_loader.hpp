#ifndef PLATO_HARDWARE_INTERFACE__UTILS__ACTUATOR_OFFSET_LOADER_HPP_
#define PLATO_HARDWARE_INTERFACE__UTILS__ACTUATOR_OFFSET_LOADER_HPP_

#include <string>
#include <vector>

namespace plato_actuator
{

using PositionOffsets = std::vector<float>;

PositionOffsets load_plato_actuator_position_offsets();
PositionOffsets load_plato_actuator_position_offsets(const std::string & yaml_path);

void save_plato_actuator_position_offsets(const PositionOffsets & offsets);
void save_plato_actuator_position_offsets(
  const PositionOffsets & offsets,
  const std::string & yaml_path);

}  // namespace plato_actuator

#endif  // PLATO_HARDWARE_INTERFACE__UTILS__ACTUATOR_OFFSET_LOADER_HPP_
