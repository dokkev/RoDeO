//
// Copyright (c) 2026
//
// Registry layer for assembling IDProblem from runtime task/contact primitives.
//

#ifndef WBC_CORE_CONTROLLER_BASE_ID_PROBLEM_REGISTRY_HPP_
#define WBC_CORE_CONTROLLER_BASE_ID_PROBLEM_REGISTRY_HPP_

#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "wbc_core/bias/joint-accel-bias.hpp"
#include "wbc_core/contacts/contact-level.hpp"
#include "wbc_core/formulations/id-problem.hpp"
#include "wbc_core/math/constraint-inequality.hpp"
#include "wbc_core/robots/robot-system.hpp"
#include "wbc_core/tasks/task-motion.hpp"

namespace wbc {

class IDProblemRegistry {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  // IDProblemRegistry is a runtime task/contact registry and per-cycle
  // IDProblem assembler. It does not own solver semantics, bias generation, or
  // state transitions.

  struct TaskRegistration {
    tasks::TaskMotion* task{nullptr};
    unsigned int level{1};
    double weight{1.0};
  };

  explicit IDProblemRegistry(robots::RobotSystem& robot)
      : m_robot(robot), m_data(robot.model()) {
    m_qddotRef.setZero(robot.nv());
  }

  void addTask(tasks::TaskMotion& task, unsigned int level, double weight) {
    if (m_tasks.find(task.name()) == m_tasks.end()) {
      m_taskOrder.push_back(task.name());
    }
    m_tasks[task.name()] = {&task, level, weight};
  }

  void addJointAccelerationObjective(const std::string& name,
                                     const math::Vector* qddot_target,
                                     unsigned int level, double weight) {
    m_jointAccelerationObjectives.emplace_back(name, qddot_target, level,
                                               weight);
  }

  void addJointAccelBias(const bias::JointAccelBias& bias) {
    addJointAccelerationObjective(bias.name, bias.qddot_bias, bias.level,
                                  bias.weight);
  }

  void addJointAccelBias(const bias::JointAccelBias& bias, unsigned int level) {
    addJointAccelerationObjective(bias.name, bias.qddot_bias, level,
                                  bias.weight);
  }

  void addContact(contacts::ContactBase& contact) {
    if (m_contacts.find(contact.name()) == m_contacts.end()) {
      m_contactOrder.push_back(contact.name());
    }
    m_contacts[contact.name()] = std::make_shared<ContactLevel>(contact);
  }

  bool updateTaskWeight(const std::string& name, double weight) {
    auto it = m_tasks.find(name);
    if (it == m_tasks.end()) {
      return false;
    }
    it->second.weight = weight;
    return true;
  }

  void setReferenceAcceleration(math::ConstRefVector qddot_ref) {
    m_qddotRef = qddot_ref;
    m_hasExternalReference = true;
  }

  void clearReferenceAcceleration() { m_hasExternalReference = false; }

  void setReferenceAccelerationEnabled(bool enabled) {
    m_referenceAccelerationEnabled = enabled;
  }

  bool referenceAccelerationEnabled() const {
    return m_referenceAccelerationEnabled;
  }

  void setJointTorqueBounds(const math::Vector* tau_lb,
                            const math::Vector* tau_ub) {
    m_jointTorqueLimits.lower = tau_lb;
    m_jointTorqueLimits.upper = tau_ub;
  }

  void setExternalGeneralizedWrench(const math::Vector* h_ext) {
    m_hExt = h_ext;
  }

  IDProblem buildProblem(double time, math::ConstRefVector q_joints,
                         math::ConstRefVector qdot_joints) {
    static const std::vector<std::string> kEmptyNames;
    static const std::vector<double> kEmptyWeights;
    return buildProblem(time, q_joints, qdot_joints, kEmptyNames,
                        kEmptyWeights, kEmptyNames);
  }

  IDProblem buildProblem(double time, math::ConstRefVector q_joints,
                         math::ConstRefVector qdot_joints,
                         const robots::BaseState& base) {
    static const std::vector<std::string> kEmptyNames;
    static const std::vector<double> kEmptyWeights;
    return buildProblem(time, q_joints, qdot_joints, base,
                        kEmptyNames, kEmptyWeights, kEmptyNames);
  }

  IDProblem buildProblem(double time, math::ConstRefVector q_joints,
                         math::ConstRefVector qdot_joints,
                         const std::vector<std::string>& active_task_names,
                         const std::vector<double>& task_weights,
                         const std::vector<std::string>& active_contact_names) {
    static const std::vector<int> kEmptyLevels;
    return buildProblem(time, q_joints, qdot_joints, active_task_names,
                        task_weights, kEmptyLevels, active_contact_names);
  }

  IDProblem buildProblem(double time, math::ConstRefVector q_joints,
                         math::ConstRefVector qdot_joints,
                         const robots::BaseState& base,
                         const std::vector<std::string>& active_task_names,
                         const std::vector<double>& task_weights,
                         const std::vector<std::string>& active_contact_names) {
    static const std::vector<int> kEmptyLevels;
    return buildProblem(time, q_joints, qdot_joints, base,
                        active_task_names, task_weights, kEmptyLevels,
                        active_contact_names);
  }

  IDProblem buildProblem(double time, math::ConstRefVector q_joints,
                         math::ConstRefVector qdot_joints,
                         const std::vector<std::string>& active_task_names,
                         const std::vector<double>& task_weights,
                         const std::vector<int>& task_levels,
                         const std::vector<std::string>& active_contact_names) {
    m_robot.setTime(time);
    m_robot.updateState(makeJointState(q_joints, qdot_joints));
    m_qBuffer = m_robot.generalized_q();
    m_vBuffer = m_robot.generalized_v();
    m_robot.computeAllTerms(m_data, m_qBuffer, m_vBuffer);
    return buildProblemFromData(time, m_qBuffer, m_vBuffer, m_data,
                                active_task_names, task_weights, task_levels,
                                active_contact_names);
  }

  IDProblem buildProblem(double time, math::ConstRefVector q_joints,
                         math::ConstRefVector qdot_joints,
                         const robots::BaseState& base,
                         const std::vector<std::string>& active_task_names,
                         const std::vector<double>& task_weights,
                         const std::vector<int>& task_levels,
                         const std::vector<std::string>& active_contact_names) {
    m_robot.setTime(time);
    m_robot.updateState(makeJointState(q_joints, qdot_joints), base);
    m_qBuffer = m_robot.generalized_q();
    m_vBuffer = m_robot.generalized_v();
    m_robot.computeAllTerms(m_data, m_qBuffer, m_vBuffer);
    return buildProblemFromData(time, m_qBuffer, m_vBuffer, m_data,
                                active_task_names, task_weights, task_levels,
                                active_contact_names);
  }

  IDProblem buildProblem(double time, math::ConstRefVector q_joints,
                         math::ConstRefVector qdot_joints,
                         pinocchio::Data& data,
                         const std::vector<std::string>& active_task_names,
                         const std::vector<double>& task_weights,
                         const std::vector<int>& task_levels,
                         const std::vector<std::string>& active_contact_names) {
    m_robot.setTime(time);
    m_robot.updateState(makeJointState(q_joints, qdot_joints));
    m_qBuffer = m_robot.generalized_q();
    m_vBuffer = m_robot.generalized_v();
    return buildProblemFromData(time, m_qBuffer, m_vBuffer, data,
                                active_task_names, task_weights, task_levels,
                                active_contact_names);
  }

  IDRegularizationParams& regularization() { return m_regularization; }
  const IDRegularizationParams& regularization() const {
    return m_regularization;
  }

 private:
  robots::JointState makeJointState(math::ConstRefVector q_joints,
                                    math::ConstRefVector qdot_joints) const {
    robots::JointState joint;
    joint.q = q_joints;
    joint.qdot = qdot_joints;
    joint.tau = m_robot.tau_actuated();
    return joint;
  }

  IDProblem buildProblemFromData(
      double time, math::ConstRefVector q, math::ConstRefVector v,
      pinocchio::Data& data, const std::vector<std::string>& active_task_names,
      const std::vector<double>& task_weights,
      const std::vector<int>& task_levels,
      const std::vector<std::string>& active_contact_names) {
    IDProblem problem;
    problem.joint_torque_limits = m_jointTorqueLimits;
    problem.h_ext = m_hExt;
    problem.regularization = m_regularization;

    resolveReferenceAcceleration();
    problem.qddot_ref = &m_qddotRef;

    const bool use_task_filter = !active_task_names.empty();
    problem.motion_objectives.reserve(
        (use_task_filter ? active_task_names.size() : m_tasks.size()));
    problem.joint_acceleration_objectives.reserve(
        m_jointAccelerationObjectives.size());

    auto weightOverride =
        [&active_task_names,
         &task_weights](const std::string& name) -> std::optional<double> {
      for (std::size_t i = 0; i < active_task_names.size(); ++i) {
        if (active_task_names[i] == name) {
          if (i < task_weights.size() && task_weights[i] >= 0.0) {
            return task_weights[i];
          }
          return std::nullopt;
        }
      }
      return std::nullopt;
    };

    auto levelOverride =
        [&active_task_names,
         &task_levels](const std::string& name) -> std::optional<unsigned int> {
      for (std::size_t i = 0; i < active_task_names.size(); ++i) {
        if (active_task_names[i] == name) {
          if (i < task_levels.size() && task_levels[i] > 0) {
            return static_cast<unsigned int>(task_levels[i]);
          }
          return std::nullopt;
        }
      }
      return std::nullopt;
    };

    auto appendTask = [&](const std::string& name, TaskRegistration& entry) {
      const double weight = weightOverride(name).value_or(entry.weight);
      const unsigned int level = levelOverride(name).value_or(entry.level);
      problem.motion_objectives.push_back(
          computeMotionObjective(*entry.task, time, q, v, data, level,
                                 weight));
    };

    if (use_task_filter) {
      for (const auto& name : active_task_names) {
        auto it = m_tasks.find(name);
        if (it == m_tasks.end()) {
          throw std::invalid_argument(
              "IDProblemRegistry::buildProblem: unknown active task '" + name +
              "'");
        }
        appendTask(name, it->second);
      }
    } else {
      for (const auto& name : m_taskOrder) {
        auto it = m_tasks.find(name);
        if (it != m_tasks.end()) {
          appendTask(name, it->second);
        }
      }
    }

    problem.joint_acceleration_objectives = m_jointAccelerationObjectives;

    if (!active_contact_names.empty()) {
      for (const auto& name : active_contact_names) {
        auto it = m_contacts.find(name);
        if (it == m_contacts.end()) {
          throw std::invalid_argument(
              "IDProblemRegistry::buildProblem: unknown active contact '" +
              name + "'");
        }
        problem.contacts.push_back(
            snapshotContact(it->second->contact, time, q, v, data));
      }
    } else {
      for (const auto& name : m_contactOrder) {
        auto it = m_contacts.find(name);
        if (it == m_contacts.end()) {
          continue;
        }
        problem.contacts.push_back(
            snapshotContact(it->second->contact, time, q, v, data));
      }
    }

    return problem;
  }

  static MotionObjective computeMotionObjective(tasks::TaskMotion& task,
                                                double time,
                                                math::ConstRefVector q,
                                                math::ConstRefVector v,
                                                pinocchio::Data& data,
                                                unsigned int level,
                                                double weight) {
    const auto& constraint = task.compute(time, q, v, data);
    return MotionObjective{task.name(), &constraint, level, weight};
  }

  static ContactConstraintData snapshotContact(contacts::ContactBase& contact,
                                               double time,
                                               math::ConstRefVector q,
                                               math::ConstRefVector v,
                                               pinocchio::Data& data) {
    contact.computeMotionConstraint(time, q, v, data);
    contact.computeForceTask(time, q, v, data);
    contact.computeForceRegularizationTask(time, q, v, data);

    ContactConstraintData out;
    out.name = contact.name();

    const auto& motion_cst = contact.getMotionConstraint();
    out.Jc = motion_cst.matrix();
    out.motion_rhs = motion_cst.vector();

    out.T = contact.getForceGeneratorMatrix();

    const auto& force_cst = contact.getForceConstraint();
    out.Uf = force_cst.matrix();
    out.uf_lb = force_cst.lowerBound();
    out.uf_ub = force_cst.upperBound();
    return out;
  }

  void resolveReferenceAcceleration() {
    if (!m_referenceAccelerationEnabled) {
      m_qddotRef.setZero(m_robot.nv());
      return;
    }

    if (m_hasExternalReference) {
      return;
    }

    m_qddotRef.setZero(m_robot.nv());
  }

  robots::RobotSystem& m_robot;
  pinocchio::Data m_data;

  std::unordered_map<std::string, TaskRegistration> m_tasks;
  std::vector<std::string> m_taskOrder;
  std::vector<JointAccelerationObjective> m_jointAccelerationObjectives;
  std::unordered_map<std::string, std::shared_ptr<ContactLevel>> m_contacts;
  std::vector<std::string> m_contactOrder;

  IDRegularizationParams m_regularization;

  math::Vector m_qBuffer;
  math::Vector m_vBuffer;
  math::Vector m_qddotRef;
  bool m_hasExternalReference{false};
  bool m_referenceAccelerationEnabled{true};
  constraints::JointTorqueLimits m_jointTorqueLimits;
  const math::Vector* m_hExt{nullptr};
};

}  // namespace wbc

#endif  // WBC_CORE_CONTROLLER_BASE_ID_PROBLEM_REGISTRY_HPP_
