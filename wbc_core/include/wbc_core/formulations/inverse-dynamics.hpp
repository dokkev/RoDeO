//
// Copyright (c) 2017 CNRS, 2026
//

#ifndef __invdyn_inverse_dynamics_hpp__
#define __invdyn_inverse_dynamics_hpp__

#include "wbc_core/deprecated.hh"
#include "wbc_core/math/fwd.hpp"
#include "wbc_core/robots/robot-system.hpp"
#include "wbc_core/tasks/task-actuation.hpp"
#include "wbc_core/tasks/task-motion.hpp"
#include "wbc_core/tasks/task-contact-force.hpp"
#include "wbc_core/contacts/contact-base.hpp"
#include "wbc_core/contacts/measured-force-base.hpp"
#include "wbc_core/solvers/solver-HQP-base.hpp"

#include <optional>
#include <string>

namespace wbc {

struct TaskLevel {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  tasks::TaskBase& task;
  std::shared_ptr<math::ConstraintBase> constraint;
  unsigned int priority;

  TaskLevel(tasks::TaskBase& task, unsigned int priority);
};

struct TaskLevelForce {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  tasks::TaskContactForce& task;
  std::shared_ptr<math::ConstraintBase> constraint;
  unsigned int priority;

  TaskLevelForce(tasks::TaskContactForce& task, unsigned int priority);
};

struct MeasuredForceLevel {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  contacts::MeasuredForceBase& measuredForce;

  MeasuredForceLevel(contacts::MeasuredForceBase& measuredForce);
};

class InverseDynamicsBase {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef pinocchio::Data Data;
  typedef math::Vector Vector;
  typedef math::RefVector RefVector;
  typedef math::ConstRefVector ConstRefVector;
  typedef tasks::TaskMotion TaskMotion;
  typedef tasks::TaskContactForce TaskContactForce;
  typedef tasks::TaskActuation TaskActuation;
  typedef tasks::TaskBase TaskBase;
  typedef contacts::MeasuredForceBase MeasuredForceBase;
  typedef contacts::ContactBase ContactBase;
  typedef solvers::HQPData HQPData;
  typedef solvers::HQPOutput HQPOutput;
  typedef robots::RobotSystem RobotSystem;

  InverseDynamicsBase(const std::string& name, RobotSystem& robot,
                      bool verbose = false);

  virtual ~InverseDynamicsBase() = default;

  virtual Data& data() = 0;

  virtual unsigned int nVar() const = 0;
  virtual unsigned int nEq() const = 0;
  virtual unsigned int nIn() const = 0;

  virtual bool addMotionTask(TaskMotion& task, double weight,
                             unsigned int priorityLevel,
                             double transition_duration = 0.0) = 0;

  virtual bool addForceTask(TaskContactForce& task, double weight,
                            unsigned int priorityLevel,
                            double transition_duration = 0.0) = 0;

  virtual bool addActuationTask(TaskActuation& task, double weight,
                                unsigned int priorityLevel,
                                double transition_duration = 0.0) = 0;

  virtual bool updateTaskWeight(const std::string& task_name,
                                double weight) = 0;

  virtual bool addRigidContact(ContactBase& contact,
                               double force_regularization_weight,
                               double motion_weight = 1.0,
                               unsigned int motion_priority_level = 0) = 0;

  TSID_DEPRECATED virtual bool addRigidContact(ContactBase& contact);

  virtual bool updateRigidContactWeights(const std::string& contact_name,
                                         double force_regularization_weight,
                                         double motion_weight = -1.0) = 0;

  virtual bool addMeasuredForce(MeasuredForceBase& measuredForce) = 0;

  virtual bool removeTask(const std::string& taskName,
                          double transition_duration = 0.0) = 0;

  virtual bool removeRigidContact(const std::string& contactName,
                                  double transition_duration = 0.0) = 0;

  virtual bool removeMeasuredForce(const std::string& measuredForceName) = 0;

  virtual const HQPData& computeProblemData(double time, ConstRefVector q,
                                            ConstRefVector v) = 0;

  /// @brief Decode solver output (e.g., extract qddot/forces/tau).
  ///        Default no-op; IDHQP overrides to split the stacked decision vector.
  virtual bool decodeSolution(const HQPOutput& /*sol*/) { return true; }

  virtual const Vector& getActuatorForces(const HQPOutput& sol) = 0;
  virtual const Vector& getAccelerations(const HQPOutput& sol) = 0;
  virtual const Vector& getContactForces(const HQPOutput& sol) = 0;
  virtual bool getContactForces(const std::string& name, const HQPOutput& sol,
                                RefVector f) = 0;

  virtual std::optional<unsigned int> getTaskPriority(
      const std::string& name) = 0;

 protected:
  std::string m_name;
  RobotSystem m_robot;
  bool m_verbose;
};

}  // namespace wbc

#endif  // ifndef __invdyn_inverse_dynamics_hpp__
