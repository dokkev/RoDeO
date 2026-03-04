/**
 * @file wbc_core/wbc_architecture/src/runtime_config.cpp
 * @brief Runtime configuration store methods — no YAML parsing here.
 */
#include "wbc_architecture/runtime_config.hpp"

#include <stdexcept>

namespace wbc {

const StateConfig& RuntimeConfig::State(StateId state_id) const {
  const auto it = states_.find(state_id);
  if (it == states_.end()) {
    throw std::runtime_error("[RuntimeConfig] Unknown state id: " +
                             std::to_string(state_id));
  }
  return it->second;
}

const StateConfig* RuntimeConfig::FindState(StateId state_id) const {
  const auto it = states_.find(state_id);
  if (it == states_.end()) {
    return nullptr;
  }
  return &it->second;
}

void RuntimeConfig::BuildFormulation(StateId state_id,
                                      WbcFormulation& out) const {
  const StateConfig& state = State(state_id);
  BuildFormulation(state, out);
}

void RuntimeConfig::BuildFormulation(const StateConfig& state,
                                      WbcFormulation& out) const {
  // Use clear+insert instead of copy-assignment to guarantee capacity reuse
  // (copy-assignment may reallocate even when capacity is sufficient).
  out.Clear();

  out.motion_tasks.insert(out.motion_tasks.end(),
                          state.motion.begin(), state.motion.end());
  out.contact_constraints.insert(out.contact_constraints.end(),
                                 state.contacts.begin(), state.contacts.end());
  out.force_tasks.insert(out.force_tasks.end(),
                         state.forces.begin(), state.forces.end());

  // Global constraints first, then per-state kinematic constraints.
  // Capacity was reserved via Reserve() at startup.
  out.kinematic_constraints.insert(out.kinematic_constraints.end(),
                                   global_constraints_.begin(), global_constraints_.end());
  out.kinematic_constraints.insert(out.kinematic_constraints.end(),
                                   state.kin.begin(), state.kin.end());
}

std::vector<bool> RuntimeConfig::BuildActuationMask() const {
  const int num_qdot = num_qdot_;
  std::vector<bool> mask(static_cast<std::size_t>(num_qdot), true);

  if (!robot_model_hints_.unactuated_qdot_indices.empty()) {
    for (int idx : robot_model_hints_.unactuated_qdot_indices) {
      if (idx < 0 || idx >= num_qdot) {
        throw std::runtime_error(
            "[RuntimeConfig] robot_model.unactuated_qdot_indices contains "
            "out-of-range index " +
            std::to_string(idx) + " for NumQdot()=" +
            std::to_string(num_qdot) + ".");
      }
      mask[static_cast<std::size_t>(idx)] = false;
    }
    return mask;
  }

  const bool floating_base =
      robot_model_hints_.floating_base_override.has_value()
          ? *robot_model_hints_.floating_base_override
          : (num_float_dof_ > 0);

  if (floating_base) {
    const int num_virtual = std::min(6, num_qdot);
    for (int i = 0; i < num_virtual; ++i) {
      mask[static_cast<std::size_t>(i)] = false;
    }
  }
  return mask;
}

int RuntimeConfig::MaxContactDim() const {
  if (max_contact_dim_ > 0) {
    return max_contact_dim_;
  }

  int estimated_dim = 0;
  const auto& constraints = constraint_registry_->GetConstraints();
  for (const auto& [name, constraint] : constraints) {
    (void)name;
    const Contact* contact = dynamic_cast<Contact*>(constraint.get());
    if (contact != nullptr) {
      estimated_dim += contact->Dim();
    }
  }
  return estimated_dim > 0 ? estimated_dim : 24;
}

StateId RuntimeConfig::StartStateId() const {
  if (configured_start_state_id_.has_value()) {
    return *configured_start_state_id_;
  }
  return first_state_id_;
}

const std::string& RuntimeConfig::BaseFrameName() const {
  return robot_model_hints_.base_frame_name;
}

const std::string& RuntimeConfig::EndEffectorFrameName() const {
  return robot_model_hints_.end_effector_frame_name;
}

void RuntimeConfig::ValidateRobotDimensions() const {
  if (robot_model_hints_.expected_num_qdot.has_value() &&
      num_qdot_ != *robot_model_hints_.expected_num_qdot) {
    throw std::runtime_error(
        "[RuntimeConfig] expected_num_qdot mismatch. expected=" +
        std::to_string(*robot_model_hints_.expected_num_qdot) +
        ", actual=" + std::to_string(num_qdot_));
  }
  if (robot_model_hints_.expected_num_active_dof.has_value() &&
      num_active_dof_ != *robot_model_hints_.expected_num_active_dof) {
    throw std::runtime_error(
        "[RuntimeConfig] expected_num_active_dof mismatch. expected=" +
        std::to_string(*robot_model_hints_.expected_num_active_dof) +
        ", actual=" + std::to_string(num_active_dof_));
  }
  if (robot_model_hints_.expected_num_float_dof.has_value() &&
      num_float_dof_ != *robot_model_hints_.expected_num_float_dof) {
    throw std::runtime_error(
        "[RuntimeConfig] expected_num_float_dof mismatch. expected=" +
        std::to_string(*robot_model_hints_.expected_num_float_dof) +
        ", actual=" + std::to_string(num_float_dof_));
  }
}

void RuntimeConfig::ApplyStateOverrides(const StateConfig& state,
                                         WbcType wbc_type) const {
  for (std::size_t i = 0; i < state.motion.size(); ++i) {
    Task* task = state.motion[i];
    if (task == nullptr) continue;

    const TaskConfig* cfg = nullptr;
    if (i < state.motion_cfg.size() && state.motion_cfg[i] != nullptr) {
      cfg = state.motion_cfg[i];
    } else {
      const auto it = default_motion_task_cfg_.find(task);
      if (it != default_motion_task_cfg_.end()) {
        cfg = &it->second;
      }
    }
    if (cfg != nullptr) {
      task->SetParameters(*cfg, wbc_type);
    }
  }

  for (std::size_t i = 0; i < state.forces.size(); ++i) {
    ForceTask* force_task = state.forces[i];
    if (force_task == nullptr) continue;

    const ForceTaskConfig* cfg = nullptr;
    if (i < state.force_cfg.size() && state.force_cfg[i] != nullptr) {
      cfg = state.force_cfg[i];
    } else {
      const auto it = default_force_task_cfg_.find(force_task);
      if (it != default_force_task_cfg_.end()) {
        cfg = &it->second;
      }
    }
    if (cfg != nullptr) {
      force_task->SetParameters(*cfg);
    }
  }
}

} // namespace wbc
