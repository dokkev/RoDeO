// Copyright 2026
//
// Licensed under the Apache License, Version 2.0.

#pragma once

#include <cstddef>
#include <string>

#include <yaml-cpp/yaml.h>

#include "mppi_core/costs/grasp_stability_cost.hpp"

namespace mppi_core {

GraspStabilityCostConfig ParseGraspConfig(
    const YAML::Node& params, std::size_t action_dim,
    GraspStabilityCostConfig defaults = {});

GraspStabilityCostConfig LoadGraspConfigFromYamlFile(
    const std::string& yaml_path, std::size_t action_dim,
    GraspStabilityCostConfig defaults = {});

}  // namespace mppi_core
