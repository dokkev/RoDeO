//
// Copyright (c) 2026
//
// RuntimeConfig: assembled runtime object graph.
//

#ifndef WBC_CORE_RUNTIME_RUNTIME_CONFIG_HPP_
#define WBC_CORE_RUNTIME_RUNTIME_CONFIG_HPP_

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <Eigen/Dense>
#include <yaml-cpp/yaml.h>

#include "wbc_core/architecture/state_machine.hpp"
#include "wbc_core/contacts/contact-base.hpp"
#include "wbc_core/controller/wbmc-step-input.hpp"
#include "wbc_core/runtime/compiled_config.hpp"
#include "wbc_core/tasks/task-motion.hpp"
#include "wbc_core/tasks/task-base.hpp"

namespace wbc {

struct RuntimeTaskEntry {
  std::shared_ptr<tsid::tasks::TaskMotion> task;
  unsigned int level{1};
  double weight{1.0};
};

struct RuntimeContactEntry {
  std::shared_ptr<tsid::contacts::ContactBase> contact;
};

/// Per-state runtime selection (task/contact subsets + params).
struct StateConfig {
  StateId id{-1};
  std::string type;
  std::string name;
  std::vector<std::string> task_names;
  std::vector<double> task_weights;
  std::vector<std::string> contact_names;
  YAML::Node params;
};

/// Configuration for one global constraint (e.g. torque limits).
struct ConstraintConfig {
  GlobalConstraintTypeSpec type{
      GlobalConstraintTypeSpec::kJointTrqLimitConstraint};
  bool enabled{false};
  double scale{1.0};
  bool is_soft{false};
  double soft_weight{1e5};
};

/// Runtime configuration: owns created tasks/contacts and state maps.
struct RuntimeConfig {
  std::unordered_map<std::string, RuntimeTaskEntry> task_pool;

  std::unordered_map<std::string, RuntimeContactEntry> contact_pool;

  std::vector<ConstraintConfig> global_constraints;

  std::unordered_map<StateId, StateConfig> states;

  StateId start_state_id{0};

  tsid::WBMCRegularizationParams regularization;
  double contact_accel_weight{100.0};

  double kp_acc{120.0};
  double kd_acc{22.0};
  double dt{0.001};

  bool torque_limits_enabled{false};
  Eigen::VectorXd tau_lb;
  Eigen::VectorXd tau_ub;

  std::vector<std::shared_ptr<tsid::tasks::TaskBase>> constraint_tasks;
};

}  // namespace wbc

#endif  // WBC_CORE_RUNTIME_RUNTIME_CONFIG_HPP_
