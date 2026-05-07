//
// Copyright (c) 2026
//

#include "wbc_core/runtime/runtime_assembler.hpp"

#include <functional>
#include <stdexcept>
#include <unordered_map>

#include <yaml-cpp/yaml.h>

#include "wbc_core/architecture/states/cartesian_teleop_state.hpp"
#include "wbc_core/architecture/states/home_state.hpp"
#include "wbc_core/architecture/states/initialize_state.hpp"
#include "wbc_core/architecture/states/joint_teleop_state.hpp"
#include "wbc_core/contacts/contact-6d.hpp"
#include "wbc_core/contacts/contact-point.hpp"
#include "wbc_core/tasks/task-com-equality.hpp"
#include "wbc_core/tasks/task-joint-posture.hpp"
#include "wbc_core/tasks/task-se3-equality.hpp"

namespace wbc {

using namespace tsid;

namespace {

Eigen::VectorXd parseScalarOrVectorSpec(const ScalarOrVectorSpec& spec,
                                        int dim, double default_scalar) {
  if (!spec.has_value) {
    return Eigen::VectorXd::Constant(dim, default_scalar);
  }
  if (!spec.is_vector) {
    return Eigen::VectorXd::Constant(dim, spec.scalar);
  }
  if (static_cast<int>(spec.values.size()) != dim) {
    throw std::invalid_argument(
        "RuntimeAssembler::Assemble: gain vector dimension mismatch "
        "(expected " +
        std::to_string(dim) + ", got " + std::to_string(spec.values.size()) +
        ")");
  }
  Eigen::VectorXd out(dim);
  for (int i = 0; i < dim; ++i) {
    out(i) = spec.values[static_cast<std::size_t>(i)];
  }
  return out;
}

RuntimeTaskEntry buildTaskEntry(const TaskSpec& spec,
                                robots::RobotWrapper& robot) {
  RuntimeTaskEntry entry;
  entry.level = spec.level;
  entry.weight = spec.weight;

  if (spec.type == TaskTypeSpec::kJointTask) {
    const int na = robot.na();
    auto task = std::make_shared<tasks::TaskJointPosture>(spec.name, robot);
    task->Kp(parseScalarOrVectorSpec(spec.kp, na, 100.0));
    task->Kd(parseScalarOrVectorSpec(spec.kd, na, 10.0));
    entry.task = task;
    return entry;
  }

  if (spec.type == TaskTypeSpec::kLinkPosTask ||
      spec.type == TaskTypeSpec::kLinkOriTask) {
    if (spec.target_frame.empty()) {
      throw std::invalid_argument(
          "RuntimeAssembler::Assemble: task '" + spec.name +
          "' requires target_frame");
    }
    auto task = std::make_shared<tasks::TaskSE3Equality>(spec.name, robot,
                                                         spec.target_frame);
    task->Kp(parseScalarOrVectorSpec(spec.kp, 6, 100.0));
    task->Kd(parseScalarOrVectorSpec(spec.kd, 6, 10.0));

    // TSID Motion convention: [angular(3); linear(3)]
    tsid::math::Vector mask = tsid::math::Vector::Zero(6);
    if (spec.type == TaskTypeSpec::kLinkPosTask) {
      mask.tail(3).setOnes();
    } else {
      mask.head(3).setOnes();
    }
    task->setMask(mask);
    entry.task = task;
    return entry;
  }

  if (spec.type == TaskTypeSpec::kComTask) {
    auto task = std::make_shared<tasks::TaskComEquality>(spec.name, robot);
    task->Kp(parseScalarOrVectorSpec(spec.kp, 3, 100.0));
    task->Kd(parseScalarOrVectorSpec(spec.kd, 3, 10.0));
    entry.task = task;
    return entry;
  }

  throw std::invalid_argument(
      "RuntimeAssembler::Assemble: task '" + spec.name +
      "' uses unsupported type '" + std::string(ToString(spec.type)) +
      "' in WBMC v1");
}

RuntimeContactEntry buildContactEntry(const ContactSpec& spec,
                                      robots::RobotWrapper& robot) {
  RuntimeContactEntry entry;
  if (spec.target_frame.empty()) {
    throw std::invalid_argument(
        "RuntimeAssembler::Assemble: contact '" + spec.name +
        "' requires target_frame");
  }

  const tsid::math::Vector3 contact_normal = tsid::math::Vector3::UnitZ();
  if (spec.type == ContactTypeSpec::kSurfaceContact) {
    if (!spec.has_foot_half_length || !spec.has_foot_half_width) {
      throw std::invalid_argument(
          "RuntimeAssembler::Assemble: SurfaceContact '" + spec.name +
          "' requires foot_half_length and foot_half_width");
    }

    tsid::math::Matrix3x contact_points(3, 4);
    contact_points << -spec.foot_half_length, -spec.foot_half_length,
        +spec.foot_half_length, +spec.foot_half_length, -spec.foot_half_width,
        +spec.foot_half_width, -spec.foot_half_width, +spec.foot_half_width,
        spec.sole_thickness, spec.sole_thickness, spec.sole_thickness,
        spec.sole_thickness;

    auto contact = std::make_shared<contacts::Contact6d>(
        spec.name, robot, spec.target_frame, contact_points, contact_normal,
        spec.mu, spec.fMin, spec.fMax);
    contact->Kp(spec.kp_contact * tsid::math::Vector::Ones(6));
    contact->Kd(2.0 * contact->Kp().cwiseSqrt());
    entry.contact = contact;
    return entry;
  }

  if (spec.type == ContactTypeSpec::kPointContact) {
    auto contact = std::make_shared<contacts::ContactPoint>(
        spec.name, robot, spec.target_frame, contact_normal, spec.mu,
        spec.fMin, spec.fMax);
    contact->Kp(spec.kp_contact * tsid::math::Vector::Ones(3));
    contact->Kd(2.0 * contact->Kp().cwiseSqrt());
    entry.contact = contact;
    return entry;
  }

  throw std::invalid_argument(
      "RuntimeAssembler::Assemble: contact '" + spec.name +
      "' uses unsupported type '" + std::string(ToString(spec.type)) + "'");
}

void wireRegistryBindings(RuntimeConfig& config,
                          WBMCRegistry& registry,
                          pinocchio::Data& data) {
  // Apply regularization to WBMC runtime layer.
  registry.regularization() = config.regularization;
  // Contact consistency is a hard physics layer in WBMC v1; keep legacy
  // w_xc_ddot parsed but intentionally unused here.

  // Add all contacts to registry; active subsets are selected per state.
  for (auto& [name, contact_info] : config.contact_pool) {
    (void)name;
    if (auto* c6d = dynamic_cast<contacts::Contact6d*>(contact_info.contact.get())) {
      auto frame_id = c6d->getMotionTask().frame_id();
      c6d->setReference(data.oMf[frame_id]);
    }
    registry.addContact(*contact_info.contact);
  }

  // Add all motion tasks to registry with their explicit HQP levels.
  for (auto& [name, task_info] : config.task_pool) {
    (void)name;
    registry.addTask(*task_info.task, task_info.level, task_info.weight);
  }
}

void wireConstraintBindings(RuntimeConfig& config,
                            WBMCRegistry& registry,
                            robots::RobotWrapper& robot) {
  // Add global constraints.
  for (const auto& constraint : config.global_constraints) {
    if (!constraint.enabled) {
      continue;
    }

    switch (constraint.type) {
      case GlobalConstraintTypeSpec::kJointTrqLimitConstraint: {
      const auto& model = robot.model();
      const int na = robot.na();
      config.torque_limits_enabled = true;
      config.tau_lb = -constraint.scale * model.effortLimit.tail(na);
      config.tau_ub = constraint.scale * model.effortLimit.tail(na);
      registry.setTorqueBounds(&config.tau_lb, &config.tau_ub);
      break;
      }
    }
  }
}

using StateCreator = std::function<std::unique_ptr<StateMachine>(
    StateId, const std::string&, const StateMachineContext&)>;

std::unordered_map<std::string, StateCreator> buildStateFactory() {
  std::unordered_map<std::string, StateCreator> factory;
  factory["initialize"] = [](StateId id, const std::string& name,
                             const StateMachineContext& ctx) {
    return std::make_unique<InitializeState>(id, name, ctx);
  };
  factory["home"] = [](StateId id, const std::string& name,
                       const StateMachineContext& ctx) {
    return std::make_unique<HomeState>(id, name, ctx);
  };
  factory["cartesian_teleop"] = [](StateId id, const std::string& name,
                                   const StateMachineContext& ctx) {
    return std::make_unique<CartesianTeleopState>(id, name, ctx);
  };
  factory["joint_teleop"] = [](StateId id, const std::string& name,
                               const StateMachineContext& ctx) {
    return std::make_unique<JointTeleopState>(id, name, ctx);
  };
  return factory;
}

void assembleFsmObjects(RuntimeConfig& config,
                        FSMHandler& fsm_handler,
                        StateProvider& state_provider,
                        robots::RobotWrapper& robot,
                        pinocchio::Data& data) {
  // Build state machine context.
  StateMachineContext context;
  context.robot = &robot;
  context.data = &data;
  context.state_provider = &state_provider;

  const auto factory = buildStateFactory();

  for (auto& [state_id, state_cfg] : config.states) {
    (void)state_id;
    std::unique_ptr<StateMachine> state;

    auto it = factory.find(state_cfg.type);
    if (it != factory.end()) {
      state = it->second(state_cfg.id, state_cfg.name, context);
    } else {
      throw std::invalid_argument(
          "RuntimeAssembler::InitializeFsm: state '" + state_cfg.name +
          "' uses unknown type '" + state_cfg.type + "'");
    }

    for (const auto& task_name : state_cfg.task_names) {
      auto task_it = config.task_pool.find(task_name);
      if (task_it != config.task_pool.end()) {
        state->assignTask(task_name, task_it->second.task);
      }
    }

    state->SetParameters(state_cfg.params);
    fsm_handler.RegisterState(state_cfg.id, std::move(state));
  }

  fsm_handler.SetStartState(config.start_state_id);
}

}  // namespace

RuntimeConfig RuntimeAssembler::Assemble(
    const CompiledConfig& compiled_config,
    robots::RobotWrapper& robot) {
  RuntimeConfig config;

  for (const auto& contact_spec : compiled_config.contact_pool) {
    config.contact_pool[contact_spec.name] =
        buildContactEntry(contact_spec, robot);
  }

  for (const auto& task_spec : compiled_config.task_pool) {
    if (task_spec.type == TaskTypeSpec::kForceTask) {
      throw std::invalid_argument(
          "RuntimeAssembler::Assemble: task '" + task_spec.name +
          "' uses unsupported type '" + std::string(ToString(task_spec.type)) +
          "' in WBMC v1");
    }
    config.task_pool[task_spec.name] = buildTaskEntry(task_spec, robot);
  }

  bool first_state = true;
  for (const auto& state_spec : compiled_config.states) {
    StateConfig state;
    state.id = state_spec.id;
    state.name = state_spec.name;
    state.type = state_spec.type;
    if (!state_spec.params_yaml.empty()) {
      state.params = YAML::Load(state_spec.params_yaml);
    }

    state.task_names.reserve(state_spec.tasks.size());
    state.task_weights.reserve(state_spec.tasks.size());
    for (const auto& task : state_spec.tasks) {
      if (config.task_pool.find(task.name) == config.task_pool.end()) {
        throw std::invalid_argument(
            "RuntimeAssembler::Assemble: state '" + state_spec.name +
            "' references unknown task '" + task.name + "'");
      }
      state.task_names.push_back(task.name);
      state.task_weights.push_back(task.weight);
    }

    state.contact_names.reserve(state_spec.contacts.size());
    for (const auto& contact : state_spec.contacts) {
      if (config.contact_pool.find(contact.name) == config.contact_pool.end()) {
        throw std::invalid_argument(
            "RuntimeAssembler::Assemble: state '" + state_spec.name +
            "' references unknown contact '" + contact.name + "'");
      }
      state.contact_names.push_back(contact.name);
    }

    if (first_state) {
      config.start_state_id = state.id;
      first_state = false;
    }

    config.states[state.id] = std::move(state);
  }

  config.regularization.w_delta_qddot =
      compiled_config.regularization.w_delta_qddot;
  config.contact_accel_weight = compiled_config.regularization.w_xc_ddot;
  config.kp_acc = compiled_config.controller.kp_acc;
  config.kd_acc = compiled_config.controller.kd_acc;
  config.dt = compiled_config.controller.dt;

  config.global_constraints.reserve(compiled_config.global_constraints.size());
  for (const auto& constraint : compiled_config.global_constraints) {
    ConstraintConfig runtime_constraint;
    runtime_constraint.type = constraint.type;
    runtime_constraint.enabled = constraint.enabled;
    runtime_constraint.scale = constraint.scale;
    runtime_constraint.is_soft = constraint.is_soft;
    runtime_constraint.soft_weight = constraint.soft_weight;
    config.global_constraints.push_back(std::move(runtime_constraint));
  }

  return config;
}

void RuntimeAssembler::InitializeFsm(
    RuntimeConfig& config,
    WBMCRegistry& registry,
    FSMHandler& fsm_handler,
    StateProvider& state_provider,
    robots::RobotWrapper& robot,
    pinocchio::Data& data) {
  wireRegistryBindings(config, registry, data);
  wireConstraintBindings(config, registry, robot);
  assembleFsmObjects(config, fsm_handler, state_provider, robot, data);
}

}  // namespace wbc
