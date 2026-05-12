//
// Copyright (c) 2026
//

#include "control_architecture/runtime/runtime_assembler.hpp"

#include <stdexcept>
#include <utility>

#include <yaml-cpp/yaml.h>

#include "wbc_core/contacts/contact-6d.hpp"
#include "wbc_core/contacts/contact-point.hpp"
#include "wbc_core/tasks/task-com-equality.hpp"
#include "wbc_core/tasks/task-joint-posture.hpp"
#include "wbc_core/tasks/task-se3-equality.hpp"

namespace wbc {

namespace {

Eigen::VectorXd parseScalarOrVectorSpec(const ScalarOrVectorSpec& spec, int dim,
                                        double default_scalar) {
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
                                robots::RobotSystem& robot) {
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

  if (spec.type == TaskTypeSpec::kSE3Task) {
    if (spec.target_frame.empty()) {
      throw std::invalid_argument("RuntimeAssembler::Assemble: task '" +
                                  spec.name + "' requires target_frame");
    }
    auto task = std::make_shared<tasks::TaskSE3Equality>(spec.name, robot,
                                                         spec.target_frame);
    task->Kp(parseScalarOrVectorSpec(spec.kp, 6, 100.0));
    task->Kd(parseScalarOrVectorSpec(spec.kd, 6, 10.0));

    // TSID Motion convention: [angular(3); linear(3)]
    wbc::math::Vector mask = wbc::math::Vector::Zero(6);
    if (spec.use_orientation) {
      mask.head(3).setOnes();
    }
    if (spec.use_position) {
      mask.tail(3).setOnes();
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

  throw std::invalid_argument("RuntimeAssembler::Assemble: task '" + spec.name +
                              "' uses unsupported type '" +
                              std::string(ToString(spec.type)) + "' in IDHQP");
}

RuntimeContactEntry buildContactEntry(const ContactSpec& spec,
                                      robots::RobotSystem& robot) {
  RuntimeContactEntry entry;
  if (spec.target_frame.empty()) {
    throw std::invalid_argument("RuntimeAssembler::Assemble: contact '" +
                                spec.name + "' requires target_frame");
  }

  const wbc::math::Vector3 contact_normal = wbc::math::Vector3::UnitZ();
  if (spec.type == ContactTypeSpec::kSurfaceContact) {
    if (!spec.has_foot_half_length || !spec.has_foot_half_width) {
      throw std::invalid_argument(
          "RuntimeAssembler::Assemble: SurfaceContact '" + spec.name +
          "' requires foot_half_length and foot_half_width");
    }

    wbc::math::Matrix3x contact_points(3, 4);
    contact_points << -spec.foot_half_length, -spec.foot_half_length,
        +spec.foot_half_length, +spec.foot_half_length, -spec.foot_half_width,
        +spec.foot_half_width, -spec.foot_half_width, +spec.foot_half_width,
        spec.sole_thickness, spec.sole_thickness, spec.sole_thickness,
        spec.sole_thickness;

    auto contact = std::make_shared<contacts::Contact6d>(
        spec.name, robot, spec.target_frame, contact_points, contact_normal,
        spec.mu, spec.fMin, spec.fMax);
    contact->Kp(spec.kp_contact * wbc::math::Vector::Ones(6));
    contact->Kd(2.0 * contact->Kp().cwiseSqrt());
    entry.contact = contact;
    return entry;
  }

  if (spec.type == ContactTypeSpec::kPointContact) {
    auto contact = std::make_shared<contacts::ContactPoint>(
        spec.name, robot, spec.target_frame, contact_normal, spec.mu, spec.fMin,
        spec.fMax);
    contact->Kp(spec.kp_contact * wbc::math::Vector::Ones(3));
    contact->Kd(2.0 * contact->Kp().cwiseSqrt());
    entry.contact = contact;
    return entry;
  }

  throw std::invalid_argument("RuntimeAssembler::Assemble: contact '" +
                              spec.name + "' uses unsupported type '" +
                              std::string(ToString(spec.type)) + "'");
}

}  // namespace

RuntimeConfig RuntimeAssembler::Assemble(const CompiledConfig& compiled_config,
                                         robots::RobotSystem& robot) {
  RuntimeConfig config;

  for (const auto& contact_spec : compiled_config.contact_pool) {
    config.contact_pool[contact_spec.name] =
        buildContactEntry(contact_spec, robot);
  }

  for (const auto& task_spec : compiled_config.task_pool) {
    if (task_spec.type == TaskTypeSpec::kForceTask) {
      throw std::invalid_argument("RuntimeAssembler::Assemble: task '" +
                                  task_spec.name + "' uses unsupported type '" +
                                  std::string(ToString(task_spec.type)) +
                                  "' in IDHQP");
    }
    config.task_pool[task_spec.name] = buildTaskEntry(task_spec, robot);
  }

  bool first_state = true;
  for (const auto& state_spec : compiled_config.states) {
    StateConfig state;
    state.id = state_spec.id;
    state.name = state_spec.name;
    state.lifecycle = state_spec.lifecycle;
    state.params = state_spec.params;

    state.task_names.reserve(state_spec.tasks.size());
    state.task_weights.reserve(state_spec.tasks.size());
    state.task_levels.reserve(state_spec.tasks.size());
    for (const auto& task : state_spec.tasks) {
      if (config.task_pool.find(task.name) == config.task_pool.end()) {
        throw std::invalid_argument("RuntimeAssembler::Assemble: state '" +
                                    state.name + "' references unknown task '" +
                                    task.name + "'");
      }
      state.task_names.push_back(task.name);
      state.task_weights.push_back(task.weight);
      state.task_levels.push_back(task.level);
    }

    state.contact_names.reserve(state_spec.contacts.size());
    for (const auto& contact : state_spec.contacts) {
      if (config.contact_pool.find(contact.name) == config.contact_pool.end()) {
        throw std::invalid_argument(
            "RuntimeAssembler::Assemble: state '" + state.name +
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
  config.regularization.w_lambda = compiled_config.regularization.w_lambda;
  config.contact_accel_weight = compiled_config.regularization.w_xc_ddot;
  config.kp_acc = compiled_config.controller.kp_acc;
  config.kd_acc = compiled_config.controller.kd_acc;
  config.dt = compiled_config.controller.dt;
  config.qddot_ref_enabled = compiled_config.controller.qddot_ref;
  config.debug_enabled = compiled_config.debug.enabled;
  config.debug_print_interval = compiled_config.debug.print_interval;
  config.solver_type = compiled_config.solver.type;
  config.solver_qp_params = compiled_config.solver.qp_params;

  config.constraints.reserve(compiled_config.constraints.size());
  for (const auto& constraint : compiled_config.constraints) {
    ConstraintConfig runtime_constraint;
    runtime_constraint.type = constraint.type;
    runtime_constraint.enabled = constraint.enabled;
    runtime_constraint.scale = constraint.scale;
    config.constraints.push_back(std::move(runtime_constraint));
  }

  return config;
}

void BindRegistry(RuntimeConfig& config, IDProblemRegistry& registry,
                  robots::RobotSystem& robot, pinocchio::Data& data) {
  registry.regularization() = config.regularization;
  registry.setReferenceAccelerationEnabled(config.qddot_ref_enabled);
  // Contact consistency is a hard physics layer in IDHQP; keep legacy
  // w_xc_ddot parsed but intentionally unused here.

  for (auto& [name, contact_info] : config.contact_pool) {
    (void)name;
    if (auto* c6d =
            dynamic_cast<contacts::Contact6d*>(contact_info.contact.get())) {
      auto frame_id = c6d->getMotionTask().frame_id();
      c6d->setReference(data.oMf[frame_id]);
    }
    registry.addContact(*contact_info.contact);
  }

  for (auto& [name, task_info] : config.task_pool) {
    (void)name;
    registry.addTask(*task_info.task, task_info.level, task_info.weight);
  }

  for (const auto& constraint : config.constraints) {
    if (!constraint.enabled) {
      continue;
    }

    switch (constraint.type) {
      case ConstraintTypeSpec::kJointTorque: {
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

}  // namespace wbc
