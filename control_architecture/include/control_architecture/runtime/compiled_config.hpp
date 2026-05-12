//
// Copyright (c) 2026
//
// CompiledConfig: parsed YAML configuration (no TSID object ownership).
//

#ifndef CONTROL_ARCHITECTURE_RUNTIME_COMPILED_CONFIG_HPP_
#define CONTROL_ARCHITECTURE_RUNTIME_COMPILED_CONFIG_HPP_

#include <string>
#include <vector>

#include "control_architecture/state_machine/state_machine.hpp"
#include "wbc_core/solvers/fwd.hpp"
#include "wbc_core/solvers/solver-qp-params.hpp"

namespace wbc {

enum class TaskTypeSpec {
  kJointTask,
  kSE3Task,
  kComTask,
  kForceTask,
};

enum class ContactTypeSpec {
  kSurfaceContact,
  kPointContact,
};

enum class ConstraintTypeSpec {
  kJointTorque,
};

inline const char* ToString(TaskTypeSpec type) {
  switch (type) {
    case TaskTypeSpec::kJointTask:
      return "JointTask";
    case TaskTypeSpec::kSE3Task:
      return "SE3Task";
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

inline const char* ToString(ConstraintTypeSpec type) {
  switch (type) {
    case ConstraintTypeSpec::kJointTorque:
      return "JointTorque";
  }
  return "UnknownConstraintType";
}

inline const char* ToString(solvers::SolverHQP type) {
  switch (type) {
    case solvers::SOLVER_HQP_EIQUADPROG:
      return "SOLVER_HQP_EIQUADPROG";
    case solvers::SOLVER_HQP_EIQUADPROG_FAST:
      return "SOLVER_HQP_EIQUADPROG_FAST";
    case solvers::SOLVER_HQP_EIQUADPROG_RT:
      return "SOLVER_HQP_EIQUADPROG_RT";
#ifdef TSID_QPMAD_FOUND
    case solvers::SOLVER_HQP_QPMAD:
      return "SOLVER_HQP_QPMAD";
#endif
#ifdef TSID_WITH_PROXSUITE
    case solvers::SOLVER_HQP_PROXQP:
      return "SOLVER_HQP_PROXQP";
#endif
#ifdef TSID_WITH_OSQP
    case solvers::SOLVER_HQP_OSQP:
      return "SOLVER_HQP_OSQP";
#endif
#ifdef QPOASES_FOUND
    case solvers::SOLVER_HQP_OASES:
      return "SOLVER_HQP_OASES";
#endif
  }
  return "UnknownSolverHQP";
}

inline solvers::SolverHQP DefaultSolverTypeSpec() {
#ifdef TSID_WITH_PROXSUITE
  return solvers::SOLVER_HQP_PROXQP;
#else
  return solvers::SOLVER_HQP_EIQUADPROG;
#endif
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
  unsigned int level{1};
  std::string target_frame;
  bool use_position{true};
  bool use_orientation{true};
  ScalarOrVectorSpec kp;
  ScalarOrVectorSpec kd;
};

/// Contact pool entry (pure typed config representation).
struct ContactSpec {
  std::string name;
  ContactTypeSpec type{ContactTypeSpec::kPointContact};
  std::string target_frame;
  double mu{0.5};
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
  int level{-1};        // -1 => use pool default
};

/// Contact activation entry inside one state.
struct StateContactSelection {
  std::string name;
};

/// One state entry from config, still solver-agnostic.
struct StateSpec {
  StateId id{-1};
  std::string name;  // StateFactory key and runtime state name.
  StateLifecycle lifecycle;
  YAML::Node params;
  std::vector<StateTaskSelection> tasks;
  std::vector<StateContactSelection> contacts;
};

/// One IDProblem constraint primitive config entry.
struct ConstraintSpec {
  ConstraintTypeSpec type{ConstraintTypeSpec::kJointTorque};
  bool enabled{false};
  double scale{1.0};
};

/// IDHQP regularization section from YAML.
struct RegularizationSpec {
  double w_delta_qddot{1e-4};
  double w_lambda{1e-5};
  double w_xc_ddot{100.0};
};

/// Controller section from YAML.
struct ControllerSpec {
  double kp_acc{120.0};
  double kd_acc{22.0};
  double dt{0.001};
  bool qddot_ref{true};
};

/// Optional runtime diagnostics section from YAML.
struct DebugSpec {
  bool enabled{false};
  double print_interval{5.0};
};

/// Solver section from YAML.
struct SolverSpec {
  solvers::SolverHQP type{DefaultSolverTypeSpec()};
  solvers::SolverQPParams qp_params;
};

/// Pure parsed compiled config for one runtime.
struct CompiledConfig {
  std::vector<TaskSpec> task_pool;
  std::vector<ContactSpec> contact_pool;
  std::vector<StateSpec> states;
  std::vector<ConstraintSpec> constraints;

  RegularizationSpec regularization;
  ControllerSpec controller;
  DebugSpec debug;
  SolverSpec solver;
};

}  // namespace wbc

#endif  // CONTROL_ARCHITECTURE_RUNTIME_COMPILED_CONFIG_HPP_
