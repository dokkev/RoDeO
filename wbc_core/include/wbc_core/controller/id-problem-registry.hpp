//
// Copyright (c) 2026
//
// Registry layer for assembling IDProblem from TSID primitives.
//

#ifndef __wbc_controller_id_problem_registry_hpp__
#define __wbc_controller_id_problem_registry_hpp__

#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "wbc_core/bias/joint-accel-bias.hpp"
#include "wbc_core/controller/id-problem.hpp"
#include "wbc_core/formulations/contact-level.hpp"
#include "wbc_core/math/constraint-inequality.hpp"
#include "wbc_core/nominal/nominal-acceleration-provider.hpp"
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

  struct JointAccelerationTargetRegistration {
    JointAccelerationTarget target;
    unsigned int level{2};
    double weight{1.0};
  };

  explicit IDProblemRegistry(robots::RobotSystem& robot)
      : m_robot(robot), m_data(robot.model()) {
    m_qddotRef.setZero(robot.nv());
  }

  void addTask(tasks::TaskMotion& task, unsigned int level, double weight) {
    m_tasks[task.name()] = {&task, level, weight};
  }

  // Legacy shim: semantic operational tasks map to level 1.
  void addOperationalTask(tasks::TaskMotion& task, double weight) {
    addTask(task, 1u, weight);
  }

  // Legacy shim: semantic bias tasks map to level 2.
  void addTaskSpaceBias(tasks::TaskMotion& task, double weight) {
    addTask(task, 2u, weight);
  }

  void addJointAccelerationTarget(const JointAccelerationTarget& target,
                                  unsigned int level, double weight) {
    m_jointAccelerationTargets.push_back({target, level, weight});
  }

  void addJointAccelBias(const bias::JointAccelBias& bias) {
    addJointAccelerationTarget(
        JointAccelerationTarget{bias.name, bias.qddot_bias}, bias.level,
        bias.weight);
  }

  void addJointAccelBias(const bias::JointAccelBias& bias, unsigned int level) {
    addJointAccelerationTarget(
        JointAccelerationTarget{bias.name, bias.qddot_bias}, level,
        bias.weight);
  }

  void addContact(contacts::ContactBase& contact) {
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

  void clearNominalProvider() { m_nominalProvider.reset(); }

  void setNominalProvider(
      std::shared_ptr<nominal::NominalAccelerationProvider> provider) {
    m_nominalProvider = std::move(provider);
  }

  void setReferenceAcceleration(math::ConstRefVector qddot_ref) {
    m_qddotRef = qddot_ref;
    m_hasExternalReference = true;
  }

  void clearReferenceAcceleration() { m_hasExternalReference = false; }

  void setNominalAcceleration(math::ConstRefVector qddot_ref) {
    setReferenceAcceleration(qddot_ref);
  }

  void clearNominalAcceleration() { clearReferenceAcceleration(); }

  void setReferenceAccelerationEnabled(bool enabled) {
    m_referenceAccelerationEnabled = enabled;
  }

  bool referenceAccelerationEnabled() const {
    return m_referenceAccelerationEnabled;
  }

  void setTorqueBounds(const math::Vector* tau_lb, const math::Vector* tau_ub) {
    m_torqueLimits.lower = tau_lb;
    m_torqueLimits.upper = tau_ub;
  }

  void setExternalGeneralizedWrench(const math::Vector* h_ext) {
    m_hExt = h_ext;
  }

  IDProblem buildProblem(double time, math::ConstRefVector q,
                         math::ConstRefVector qdot) {
    static const std::vector<std::string> kEmptyNames;
    static const std::vector<double> kEmptyWeights;
    return buildProblem(time, q, qdot, kEmptyNames, kEmptyWeights, kEmptyNames);
  }

  IDProblem buildProblem(double time, math::ConstRefVector q,
                         math::ConstRefVector qdot,
                         const std::vector<std::string>& active_task_names,
                         const std::vector<double>& task_weights,
                         const std::vector<std::string>& active_contact_names) {
    static const std::vector<int> kEmptyLevels;
    return buildProblem(time, q, qdot, active_task_names, task_weights,
                        kEmptyLevels, active_contact_names);
  }

  IDProblem buildProblem(double time, math::ConstRefVector q,
                         math::ConstRefVector qdot,
                         const std::vector<std::string>& active_task_names,
                         const std::vector<double>& task_weights,
                         const std::vector<int>& task_levels,
                         const std::vector<std::string>& active_contact_names) {
    m_robot.updateState(q, qdot);
    m_qBuffer = q;
    m_qdotBuffer = qdot;
    m_robot.computeAllTerms(m_data, q, qdot);
    return buildProblemFromData(time, q, qdot, m_data, active_task_names,
                                task_weights, task_levels,
                                active_contact_names);
  }

  IDProblem buildProblem(double time, math::ConstRefVector q,
                         math::ConstRefVector qdot, pinocchio::Data& data,
                         const std::vector<std::string>& active_task_names,
                         const std::vector<double>& task_weights,
                         const std::vector<int>& task_levels,
                         const std::vector<std::string>& active_contact_names) {
    m_robot.updateState(q, qdot);
    m_qBuffer = q;
    m_qdotBuffer = qdot;
    return buildProblemFromData(time, q, qdot, data, active_task_names,
                                task_weights, task_levels,
                                active_contact_names);
  }

  IDHierarchyPolicy& hierarchy() { return m_hierarchy; }
  const IDHierarchyPolicy& hierarchy() const { return m_hierarchy; }
  IDRegularizationParams& regularization() { return m_regularization; }
  const IDRegularizationParams& regularization() const {
    return m_regularization;
  }

 private:
  IDProblem buildProblemFromData(
      double time, math::ConstRefVector q, math::ConstRefVector qdot,
      pinocchio::Data& data, const std::vector<std::string>& active_task_names,
      const std::vector<double>& task_weights,
      const std::vector<int>& task_levels,
      const std::vector<std::string>& active_contact_names) {
    IDProblem problem;
    problem.torque_limits = m_torqueLimits;
    problem.h_ext = m_hExt;
    problem.hierarchy = m_hierarchy;
    problem.regularization = m_regularization;

    resolveReferenceAcceleration(time, q, qdot);
    problem.qddot_ref = &m_qddotRef;

    const bool use_task_filter = !active_task_names.empty();
    problem.objectives.reserve(
        m_jointAccelerationTargets.size() +
        (use_task_filter ? active_task_names.size() : m_tasks.size()));

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
      auto constraint =
          computeMotionConstraint(*entry.task, time, q, qdot, data);
      problem.objectives.push_back(
          ObjectiveTerm::MakeMotionConstraint(constraint, level, weight));
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
      for (auto& [name, entry] : m_tasks) {
        appendTask(name, entry);
      }
    }

    for (const auto& target : m_jointAccelerationTargets) {
      problem.objectives.push_back(ObjectiveTerm::MakeJointAccelerationTarget(
          target.target, target.level, target.weight));
    }

    std::vector<std::shared_ptr<ContactLevel>> activeContacts;
    if (!active_contact_names.empty()) {
      activeContacts.reserve(active_contact_names.size());
      for (const auto& name : active_contact_names) {
        auto it = m_contacts.find(name);
        if (it == m_contacts.end()) {
          throw std::invalid_argument(
              "IDProblemRegistry::buildProblem: unknown active contact '" +
              name + "'");
        }
        activeContacts.push_back(it->second);
        problem.contacts.push_back(
            snapshotContact(it->second->contact, time, q, qdot, data));
      }
    } else {
      activeContacts.reserve(m_contacts.size());
      for (auto& [name, level] : m_contacts) {
        activeContacts.push_back(level);
        problem.contacts.push_back(
            snapshotContact(level->contact, time, q, qdot, data));
      }
    }

    return problem;
  }

  static MotionConstraintRef computeMotionConstraint(tasks::TaskMotion& task,
                                                     double time,
                                                     math::ConstRefVector q,
                                                     math::ConstRefVector qdot,
                                                     pinocchio::Data& data) {
    const auto& constraint = task.compute(time, q, qdot, data);
    return MotionConstraintRef{task.name(), &constraint};
  }

  static ContactConstraintData snapshotContact(contacts::ContactBase& contact,
                                               double time,
                                               math::ConstRefVector q,
                                               math::ConstRefVector qdot,
                                               pinocchio::Data& data) {
    contact.computeMotionConstraint(time, q, qdot, data);
    contact.computeForceTask(time, q, qdot, data);
    contact.computeForceRegularizationTask(time, q, qdot, data);

    ContactConstraintData out;
    out.name = contact.name();

    const auto& motion_cst = contact.getMotionConstraint();
    const auto& motion_task = contact.getMotionTask();
    out.Jc = motion_cst.matrix();
    out.Jcdot_qdot = motion_task.getDesiredAcceleration() - motion_cst.vector();

    out.T = contact.getForceGeneratorMatrix();

    const auto& force_cst = contact.getForceConstraint();
    out.Uf = force_cst.matrix();
    out.uf_lb = force_cst.lowerBound();
    out.uf_ub = force_cst.upperBound();
    return out;
  }

  void resolveReferenceAcceleration(double time, math::ConstRefVector q,
                                    math::ConstRefVector qdot) {
    (void)q;
    (void)qdot;
    if (!m_referenceAccelerationEnabled) {
      m_qddotRef.setZero(m_robot.nv());
      return;
    }

    if (m_hasExternalReference) {
      return;
    }

    if (m_nominalProvider) {
      nominal::NominalAccelerationContext ctx;
      ctx.time = time;
      ctx.q = &m_qBuffer;
      ctx.v = &m_qdotBuffer;
      ctx.nv = m_robot.nv();
      ctx.lambdaDim = 0;
      if (m_nominalProvider->compute(ctx, m_qddotRef)) {
        return;
      }
    }

    m_qddotRef.setZero(m_robot.nv());
  }

  robots::RobotSystem& m_robot;
  pinocchio::Data m_data;

  std::unordered_map<std::string, TaskRegistration> m_tasks;
  std::vector<JointAccelerationTargetRegistration> m_jointAccelerationTargets;
  std::unordered_map<std::string, std::shared_ptr<ContactLevel>> m_contacts;

  IDHierarchyPolicy m_hierarchy;
  IDRegularizationParams m_regularization;
  std::shared_ptr<nominal::NominalAccelerationProvider> m_nominalProvider;

  math::Vector m_qBuffer;
  math::Vector m_qdotBuffer;
  math::Vector m_qddotRef;
  bool m_hasExternalReference{false};
  bool m_referenceAccelerationEnabled{true};
  constraints::ActuatorTorqueLimits m_torqueLimits;
  const math::Vector* m_hExt{nullptr};
};

}  // namespace wbc

#endif  // ifndef __wbc_controller_id_problem_registry_hpp__
