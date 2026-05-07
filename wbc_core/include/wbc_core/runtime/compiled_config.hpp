//
// Copyright (c) 2026
//
// CompiledConfig: parsed YAML configuration (no TSID object ownership).
//

#ifndef WBC_CORE_RUNTIME_COMPILED_CONFIG_HPP_
#define WBC_CORE_RUNTIME_COMPILED_CONFIG_HPP_

#include <string>
#include <vector>

#include "wbc_core/architecture/state_machine.hpp"

namespace wbc {

enum class TaskTypeSpec {
  kJointTask,
  kLinkPosTask,
  kLinkOriTask,
  kComTask,
  kForceTask,
};

enum class ContactTypeSpec {
  kSurfaceContact,
  kPointContact,
};

enum class GlobalConstraintTypeSpec {
  kJointTrqLimitConstraint,
};

inline const char* ToString(TaskTypeSpec type) {
  switch (type) {
    case TaskTypeSpec::kJointTask:
      return "JointTask";
    case TaskTypeSpec::kLinkPosTask:
      return "LinkPosTask";
    case TaskTypeSpec::kLinkOriTask:
      return "LinkOriTask";
    case TaskTypeSpec::kComTask:
      return "ComTask";
    case TaskTypeSpec::kForceTask:
      return "ForceTask";
  }
  return "UnknownTaskType";
}

inline const char* ToString(ContactTypeSpec type) {
  switch (type) {
    case ContactTypeSpec::kSurfaceContact:
      return "SurfaceContact";
    case ContactTypeSpec::kPointContact:
      return "PointContact";
  }
  return "UnknownContactType";
}

inline const char* ToString(GlobalConstraintTypeSpec type) {
  switch (type) {
    case GlobalConstraintTypeSpec::kJointTrqLimitConstraint:
      return "JointTrqLimitConstraint";
  }
  return "UnknownConstraintType";
}

/// Scalar-or-vector representation used by typed config entries.
struct ScalarOrVectorSpec {
  bool has_value{false};
  bool is_vector{false};
  double scalar{0.0};
  std::vector<double> values;
};

/// Task pool entry (pure typed config representation).
struct TaskSpec {
  std::string name;
  TaskTypeSpec type{TaskTypeSpec::kJointTask};
  double weight{1.0};
  double kp_ik{1.0};
  unsigned int level{1};
  std::string target_frame;
  ScalarOrVectorSpec kp;
  ScalarOrVectorSpec kd;
};

/// Contact pool entry (pure typed config representation).
struct ContactSpec {
  std::string name;
  ContactTypeSpec type{ContactTypeSpec::kPointContact};
  std::string target_frame;
  double mu{0.5};
  double force_reg_weight{1e-5};
  double fMin{5.0};
  double fMax{1000.0};
  double kp_contact{100.0};
  double sole_thickness{0.0};
  bool has_foot_half_length{false};
  bool has_foot_half_width{false};
  double foot_half_length{0.0};
  double foot_half_width{0.0};
};

/// Task activation entry inside one state.
struct StateTaskSelection {
  std::string name;
  double weight{-1.0};  // -1.0 => use pool default
};

/// Contact activation entry inside one state.
struct StateContactSelection {
  std::string name;
};

/// One state entry from config, still solver-agnostic.
struct StateSpec {
  StateId id{-1};
  std::string type;
  std::string name;
  std::string params_yaml;
  std::vector<StateTaskSelection> tasks;
  std::vector<StateContactSelection> contacts;
};

/// One global constraint config entry.
struct GlobalConstraintSpec {
  GlobalConstraintTypeSpec type{
      GlobalConstraintTypeSpec::kJointTrqLimitConstraint};
  bool enabled{false};
  double scale{1.0};
  bool is_soft{false};
  double soft_weight{1e5};
};

/// WBMC regularization section from YAML.
struct RegularizationSpec {
  double w_delta_qddot{1e-4};
  double w_xc_ddot{100.0};
};

/// Controller section from YAML.
struct ControllerSpec {
  double kp_acc{120.0};
  double kd_acc{22.0};
  double dt{0.001};
};

/// Pure parsed compiled config for one runtime.
struct CompiledConfig {
  std::vector<TaskSpec> task_pool;
  std::vector<ContactSpec> contact_pool;
  std::vector<StateSpec> states;
  std::vector<GlobalConstraintSpec> global_constraints;

  RegularizationSpec regularization;
  ControllerSpec controller;
};

}  // namespace wbc

#endif  // WBC_CORE_RUNTIME_COMPILED_CONFIG_HPP_
