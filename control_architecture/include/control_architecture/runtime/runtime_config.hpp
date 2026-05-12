//
// Copyright (c) 2026
//
// RuntimeConfig: assembled runtime object graph.
//

#ifndef CONTROL_ARCHITECTURE_RUNTIME_RUNTIME_CONFIG_HPP_
#define CONTROL_ARCHITECTURE_RUNTIME_RUNTIME_CONFIG_HPP_

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <Eigen/Dense>
#include <yaml-cpp/yaml.h>

#include "control_architecture/state_machine/state_machine.hpp"
#include "wbc_core/contacts/contact-base.hpp"
#include "wbc_core/controller/id-problem.hpp"
#include "control_architecture/runtime/compiled_config.hpp"
#include "wbc_core/tasks/task-motion.hpp"
#include "wbc_core/tasks/task-base.hpp"

namespace wbc {

struct RuntimeTaskEntry {
  std::shared_ptr<wbc::tasks::TaskMotion> task;
  unsigned int level{1};
  double weight{1.0};
};

struct RuntimeContactEntry {
  std::shared_ptr<wbc::contacts::ContactBase> contact;
};

/// Per-state runtime selection (task/contact subsets + params).
struct StateConfig {
  StateId id{-1};
  std::string name;
  StateLifecycle lifecycle;
  std::vector<std::string> task_names;
  std::vector<double> task_weights;
  std::vector<int> task_levels;
  std::vector<std::string> contact_names;
  YAML::Node params;
};

/// Configuration for one IDProblem constraint primitive (e.g. torque limits).
struct ConstraintConfig {
  ConstraintTypeSpec type{ConstraintTypeSpec::kJointTorque};
  bool enabled{false};
  double scale{1.0};
};

/// Runtime configuration: owns created tasks/contacts and state maps.
struct RuntimeConfig {
  std::unordered_map<std::string, RuntimeTaskEntry> task_pool;

  std::unordered_map<std::string, RuntimeContactEntry> contact_pool;

  std::vector<ConstraintConfig> constraints;

  std::unordered_map<StateId, StateConfig> states;

  StateId start_state_id{0};

  wbc::IDRegularizationParams regularization;
  double contact_accel_weight{100.0};

  double kp_acc{120.0};
  double kd_acc{22.0};
  double dt{0.001};
  bool qddot_ref_enabled{true};
  bool debug_enabled{false};
  double debug_print_interval{5.0};
  solvers::SolverHQP solver_type{DefaultSolverTypeSpec()};
  solvers::SolverQPParams solver_qp_params;

  bool torque_limits_enabled{false};
  Eigen::VectorXd tau_lb;
  Eigen::VectorXd tau_ub;

  std::vector<std::shared_ptr<wbc::tasks::TaskBase>> constraint_tasks;
};

}  // namespace wbc

#endif  // CONTROL_ARCHITECTURE_RUNTIME_RUNTIME_CONFIG_HPP_
