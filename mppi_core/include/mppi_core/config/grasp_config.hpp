// Copyright 2026
//
// Licensed under the Apache License, Version 2.0.

#pragma once

#include <cstddef>
#include <string>

#include <yaml-cpp/yaml.h>

#include "mppi_core/costs/grasp_stability_cost.hpp"
#include "mppi_core/grasp/contact_force_rollout.hpp"
#include "mppi_core/model/delta_q_reference_rollout_model.hpp"

namespace mppi_core {

GraspStabilityCostConfig ParseGraspConfig(
    const YAML::Node& params, std::size_t action_dim,
    GraspStabilityCostConfig defaults = {});

GraspStabilityCostConfig LoadGraspConfigFromYamlFile(
    const std::string& yaml_path, std::size_t action_dim,
    GraspStabilityCostConfig defaults = {});

DeltaQReferenceRolloutConfig ParseDeltaQReferenceRolloutConfig(
    const YAML::Node& params, DeltaQReferenceRolloutConfig defaults = {});

DeltaQReferenceRolloutConfig LoadDeltaQReferenceRolloutConfigFromYamlFile(
    const std::string& yaml_path,
    DeltaQReferenceRolloutConfig defaults = {});

ContactForceRolloutConfig ParseContactForceRolloutConfig(
    const YAML::Node& params, ContactForceRolloutConfig defaults = {});

ContactForceRolloutConfig LoadContactForceRolloutConfigFromYamlFile(
    const std::string& yaml_path, ContactForceRolloutConfig defaults = {});

}  // namespace mppi_core
