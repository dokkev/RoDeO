// Copyright 2026
//
// Licensed under the Apache License, Version 2.0.

#pragma once

#include <cstddef>
#include <string>

#include <yaml-cpp/yaml.h>

#include "mppi_core/core/mppi_config.hpp"

namespace mppi_core {

MPPIConfig ParseMPPIConfig(const YAML::Node& params, std::size_t action_dim,
                           MPPIConfig defaults = {});

MPPIConfig LoadMPPIConfigFromYamlFile(const std::string& yaml_path,
                                      std::size_t action_dim,
                                      MPPIConfig defaults = {});

}  // namespace mppi_core
