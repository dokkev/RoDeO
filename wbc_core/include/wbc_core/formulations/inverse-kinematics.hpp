//
// Copyright (c) 2026
//
// Contact/task-consistent redundancy resolver + PD acceleration reference.
//
// Stage 1: Cascaded HQP → jposRef, jvelRef
// Stage 2: qddot_ref = kp*(jposRef-q) + kd*(jvelRef-qdot)
//

#ifndef __invdyn_inverse_kinematics_hpp__
#define __invdyn_inverse_kinematics_hpp__

#include <memory>
#include <vector>

#include <Eigen/Dense>

#include "wbc_core/formulations/ik-kinematic-reference.hpp"
#include "wbc_core/formulations/ik-task-level.hpp"
#include "wbc_core/math/fwd.hpp"
#include "wbc_core/math/constraint-equality.hpp"
#include "wbc_core/tasks/task-motion.hpp"
#include "wbc_core/contacts/contact-base.hpp"
#include "wbc_core/solvers/solver-HQP-base.hpp"

namespace wbc {

struct ContactLevel;

/// Configuration for the IK stage.
struct IKConfig {
  double velocityClampDt{1e-3};
  double velocityAbsMax{5.0};
};

/// Contact/task-consistent redundancy resolver.
///
/// Finds delta_q that preserves contacts and operational tasks,
/// then uses remaining freedom for preference selection.
///
/// Stage 1: 4-level cascaded HQP (requires cascade-capable solver):
///   Level 0: contact hard equality    (Jc * dq = 0)
///   Level 1: operational preservation (Jop * dq = kp * e_op  or  0)
///   Level 2: preference selection     (Jpref * dq = kp * e_pref)
///   Level 3: regularization           (I * dq = 0, weight 1e-4)
///
/// Stage 2: qddot_ref = kp*(jposRef - q) + kd*(jvelRef - qdot)
class InverseKinematics {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef math::Vector Vector;
  typedef math::Matrix Matrix;
  typedef math::ConstRefVector ConstRefVector;

  /// @param nv  Total velocity DOFs (model.nv)
  /// @param na  Actuated DOFs (nv for fixed-base, nv-6 for floating)
  InverseKinematics(int nv, int na);

  /// Stage 1: Solve cascaded HQP for jposRef, jvelRef.
  /// @param solver           Cascade-capable HQP solver (caller owns).
  /// @param operationalTasks Tasks to preserve/track at level 1.
  /// @param selectionTasks   Null-space selection tasks at level 2.
  /// @param contacts         Contact constraints at level 0.
  bool solveKinematicReference(
      solvers::SolverHQPBase& solver,
      const std::vector<std::shared_ptr<IKTaskLevel>>& operationalTasks,
      const std::vector<std::shared_ptr<IKTaskLevel>>& selectionTasks,
      const std::vector<std::shared_ptr<ContactLevel>>& contacts,
      ConstRefVector q,
      const pinocchio::Model& model);

  /// Stage 2: Build PD acceleration reference from IK output.
  bool buildPostureAccelReference(ConstRefVector q, ConstRefVector v);

  // -- Configuration ---------------------------------------------------------

  void setAccelReferenceGains(double kpAcc, double kdAcc);
  void setAccelReferenceGains(ConstRefVector kpAcc, ConstRefVector kdAcc);
  void setConfig(const IKConfig& config) { m_config = config; }
  IKConfig& config() { return m_config; }
  const IKConfig& config() const { return m_config; }

  // -- Output accessors ------------------------------------------------------

  const IKKinematicReference& kinematicReference() const { return m_kinRef; }
  const Vector& qddotPostureRef() const { return m_qddotPostureRef; }

 private:
  // HQP level indices (fixed semantics)
  static constexpr unsigned int kLevelContact = 0;
  static constexpr unsigned int kLevelOperational = 1;
  static constexpr unsigned int kLevelPreference = 2;
  static constexpr unsigned int kLevelReg = 3;
  static constexpr unsigned int kNumLevels = 4;

  /// 4-level cascaded HQP solve.
  bool solveHQP(
      solvers::SolverHQPBase& solver,
      const std::vector<std::shared_ptr<IKTaskLevel>>& operationalTasks,
      const std::vector<std::shared_ptr<IKTaskLevel>>& selectionTasks,
      const std::vector<std::shared_ptr<ContactLevel>>& contacts);

  /// Post-processing: extract actuated refs, clamp velocity/position.
  void postProcess(ConstRefVector q, const pinocchio::Model& model);

  int m_nv;
  int m_na;
  int m_nvFloat;

  // IK configuration
  IKConfig m_config;

  // Stage 1 buffers
  Vector m_deltaQRef;    ///< Full (nv) position delta from IK
  Vector m_qdotRef;      ///< Full (nv) velocity reference from IK

  // Stage 2 PD gains
  Vector m_kpAcc;
  Vector m_kdAcc;

  // Stage 2 buffer
  Vector m_qddotPostureRef;

  // Output
  IKKinematicReference m_kinRef;

  // HQP constraint data (rebuilt each tick, solver provided by caller)
  solvers::HQPData m_hqpData;

  // HQP constraint storage (reused across ticks)
  std::shared_ptr<math::ConstraintEquality> m_contactHardConstraint;
  std::vector<std::shared_ptr<math::ConstraintEquality>> m_hqpTaskConstraints;
};

}  // namespace wbc

#endif  // ifndef __invdyn_inverse_kinematics_hpp__
