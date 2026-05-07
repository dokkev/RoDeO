//
// Copyright (c) 2026
//
// Contact/task-consistent redundancy resolver + PD acceleration reference.
//

#include "wbc_core/formulations/inverse-kinematics.hpp"

#include <cassert>
#include <cmath>

#include "wbc_core/formulations/contact-level.hpp"

namespace tsid {

using namespace math;

// =============================================================================
// Construction
// =============================================================================

InverseKinematics::InverseKinematics(int nv, int na)
    : m_nv(nv), m_na(na), m_nvFloat(nv - na) {
  m_deltaQRef = Vector::Zero(m_nv);
  m_qdotRef = Vector::Zero(m_nv);

  // Critically damped defaults
  m_kpAcc = Vector::Constant(m_na, 120.0);
  m_kdAcc = Vector::Constant(m_na, 22.0);

  m_qddotPostureRef = Vector::Zero(m_nv);

  m_kinRef.jposRef = Vector::Zero(m_na);
  m_kinRef.jvelRef = Vector::Zero(m_na);
}

// =============================================================================
// Configuration
// =============================================================================

void InverseKinematics::setAccelReferenceGains(double kpAcc, double kdAcc) {
  m_kpAcc = Vector::Constant(m_na, kpAcc);
  m_kdAcc = Vector::Constant(m_na, kdAcc);
}

void InverseKinematics::setAccelReferenceGains(
    ConstRefVector kpAcc, ConstRefVector kdAcc) {
  assert(kpAcc.size() == m_na && kdAcc.size() == m_na);
  m_kpAcc = kpAcc;
  m_kdAcc = kdAcc;
}

// =============================================================================
// Stage 1: Solve redundancy-consistent IK
// =============================================================================

bool InverseKinematics::solveKinematicReference(
    solvers::SolverHQPBase& solver,
    const std::vector<std::shared_ptr<IKTaskLevel>>& operationalTasks,
    const std::vector<std::shared_ptr<IKTaskLevel>>& selectionTasks,
    const std::vector<std::shared_ptr<ContactLevel>>& contacts,
    ConstRefVector q,
    const pinocchio::Model& model) {
  if (operationalTasks.empty() && selectionTasks.empty()) {
    m_kinRef.jposRef = q.tail(m_na);
    m_kinRef.jvelRef.setZero();
    return true;
  }

  bool ok = solveHQP(solver, operationalTasks, selectionTasks, contacts);

  if (ok) {
    postProcess(q, model);
  }
  return ok;
}

// =============================================================================
// Stage 1: 4-level cascaded HQP
//
// Decision variable: delta_q (nv)
//
// Level 0 (contact):     Jc * dq = 0          — preserve contacts
// Level 1 (operational): Jop * dq = kp * eop   — preserve/track task-space
// Level 2 (preference):  Jpref * dq = kp * ep  — preference selection in null-space
// Level 3 (reg):         I * dq = 0            — minimum-norm regularization
//
// Strict priority hierarchy via cascade solver.
// Velocity reference: dq / dt (no independent velocity solve).
// =============================================================================

bool InverseKinematics::solveHQP(
    solvers::SolverHQPBase& solver,
    const std::vector<std::shared_ptr<IKTaskLevel>>& operationalTasks,
    const std::vector<std::shared_ptr<IKTaskLevel>>& selectionTasks,
    const std::vector<std::shared_ptr<ContactLevel>>& contacts) {
  // Fixed 4-level structure
  m_hqpData.resize(kNumLevels);
  for (auto& level : m_hqpData) {
    level.clear();
  }

  // -- Level 0: Contact hard equality (Jc * dq = 0) --------------------------

  int totalContactMotion = 0;
  for (const auto& cl : contacts) {
    totalContactMotion += static_cast<int>(cl->contact.n_motion());
  }

  if (totalContactMotion > 0) {
    const int contactRows = static_cast<int>(m_contactHardConstraint ?
        m_contactHardConstraint->rows() : 0);
    const int contactCols = static_cast<int>(m_contactHardConstraint ?
        m_contactHardConstraint->cols() : 0);
    if (!m_contactHardConstraint ||
        contactRows != totalContactMotion ||
        contactCols != m_nv) {
      m_contactHardConstraint = std::make_shared<ConstraintEquality>(
          "ik-contact", totalContactMotion, m_nv);
    }
    m_contactHardConstraint->matrix().setZero();
    m_contactHardConstraint->vector().setZero();

    int row = 0;
    for (const auto& cl : contacts) {
      const auto& motionCst = cl->contact.getMotionConstraint();
      const int nMotion = static_cast<int>(motionCst.rows());
      m_contactHardConstraint->matrix().block(row, 0, nMotion, m_nv) =
          motionCst.matrix();
      row += nMotion;
    }

    m_hqpData[kLevelContact].push_back(
        solvers::make_pair<double, std::shared_ptr<ConstraintBase>>(
            1.0, m_contactHardConstraint));
  }

  // Reuse cached constraints from m_hqpTaskConstraints to avoid heap allocs.
  size_t cstIdx = 0;

  // -- Level 1: Operational task preservation/tracking ------------------------

  for (const auto& tl : operationalTasks) {
    const auto& task = static_cast<const tasks::TaskMotion&>(tl->task);
    const auto& constraint = task.getConstraint();
    const Matrix& J = constraint.matrix();
    const int rows = static_cast<int>(J.rows());
    const int cstRows = static_cast<int>(cstIdx < m_hqpTaskConstraints.size() ?
        m_hqpTaskConstraints[cstIdx]->rows() : 0);
    const int cstCols = static_cast<int>(cstIdx < m_hqpTaskConstraints.size() ?
        m_hqpTaskConstraints[cstIdx]->cols() : 0);

    if (cstIdx >= m_hqpTaskConstraints.size()) {
      m_hqpTaskConstraints.push_back(std::make_shared<ConstraintEquality>(
          task.name() + "_ik_op", rows, m_nv));
    } else if (cstRows != rows || cstCols != m_nv) {
      m_hqpTaskConstraints[cstIdx]->resize(rows, m_nv);
    }

    auto& cst = m_hqpTaskConstraints[cstIdx];
    cst->matrix() = J;
    if (tl->ik_mode == IKMode::kTrack) {
      cst->vector() = tl->kp_ik * tl->ik_error_sign * task.position_error();
    } else {
      cst->vector().setZero();
    }
    ++cstIdx;

    m_hqpData[kLevelOperational].push_back(
        solvers::make_pair<double, std::shared_ptr<ConstraintBase>>(1.0, cst));
  }

  // -- Level 2: Preference selection in remaining null-space -----------------

  for (const auto& tl : selectionTasks) {
    const auto& task = static_cast<const tasks::TaskMotion&>(tl->task);
    const auto& constraint = task.getConstraint();
    const Matrix& J = constraint.matrix();
    const Vector& posErr = task.position_error();
    const double kp_ik = tl->kp_ik;
    const int rows = static_cast<int>(J.rows());
    const int cstRows = static_cast<int>(cstIdx < m_hqpTaskConstraints.size() ?
        m_hqpTaskConstraints[cstIdx]->rows() : 0);
    const int cstCols = static_cast<int>(cstIdx < m_hqpTaskConstraints.size() ?
        m_hqpTaskConstraints[cstIdx]->cols() : 0);

    if (cstIdx >= m_hqpTaskConstraints.size()) {
      m_hqpTaskConstraints.push_back(std::make_shared<ConstraintEquality>(
          task.name() + "_ik_post", rows, m_nv));
    } else if (cstRows != rows || cstCols != m_nv) {
      m_hqpTaskConstraints[cstIdx]->resize(rows, m_nv);
    }

    auto& cst = m_hqpTaskConstraints[cstIdx];
    cst->matrix() = J;
    cst->vector() = kp_ik * tl->ik_error_sign * posErr;
    ++cstIdx;

    m_hqpData[kLevelPreference].push_back(
        solvers::make_pair<double, std::shared_ptr<ConstraintBase>>(1.0, cst));
  }

  // -- Level 3: Minimum-norm regularization ----------------------------------

  if (cstIdx >= m_hqpTaskConstraints.size()) {
    m_hqpTaskConstraints.push_back(std::make_shared<ConstraintEquality>(
        "ik-reg", m_nv, m_nv));
  } else if (static_cast<int>(m_hqpTaskConstraints[cstIdx]->rows()) != m_nv ||
             static_cast<int>(m_hqpTaskConstraints[cstIdx]->cols()) != m_nv) {
    m_hqpTaskConstraints[cstIdx]->resize(m_nv, m_nv);
  }
  auto& regCst = m_hqpTaskConstraints[cstIdx];
  regCst->matrix().setIdentity();
  regCst->vector().setZero();
  ++cstIdx;
  m_hqpTaskConstraints.resize(cstIdx);

  m_hqpData[kLevelReg].push_back(
      solvers::make_pair<double, std::shared_ptr<ConstraintBase>>(
          1e-4, regCst));

  // -- Solve -----------------------------------------------------------------

  unsigned int nEq = static_cast<unsigned int>(totalContactMotion);
  solver.resize(static_cast<unsigned int>(m_nv), nEq, 0);

  const auto& sol = solver.solve(m_hqpData);
  if (sol.status != solvers::HQP_STATUS_OPTIMAL) {
    return false;
  }
  m_deltaQRef = sol.x;

  // Velocity from position displacement (consistent null-space branch)
  m_qdotRef.setZero();
  m_qdotRef.tail(m_na) = m_deltaQRef.tail(m_na) / m_config.velocityClampDt;

  return true;
}

// =============================================================================
// Post-processing: extract actuated refs, clamp
// =============================================================================

void InverseKinematics::postProcess(
    ConstRefVector q, const pinocchio::Model& model) {
  const auto qAct = q.tail(m_na);

  // Extract actuated-joint references
  m_kinRef.jposRef = qAct + m_deltaQRef.tail(m_na);
  m_kinRef.jvelRef = m_qdotRef.tail(m_na);

  // Velocity clamping
  for (int i = 0; i < m_na; ++i) {
    m_kinRef.jvelRef(i) = std::clamp(
        m_kinRef.jvelRef(i),
        -m_config.velocityAbsMax, m_config.velocityAbsMax);
  }

  // Joint position clamping (from Pinocchio model limits)
  // Use nq-based offset (not nv) because position limits are in config space.
  // For floating base: nq=nv+1 (quaternion), so q_offset = nq - na.
  const int qOffset = model.nq - m_na;
  for (int i = 0; i < m_na; ++i) {
    const int qi = qOffset + i;
    if (qi < model.lowerPositionLimit.size()) {
      m_kinRef.jposRef(i) = std::clamp(
          m_kinRef.jposRef(i),
          model.lowerPositionLimit(qi),
          model.upperPositionLimit(qi));
    }
  }
}

// =============================================================================
// Stage 2: PD acceleration reference
// =============================================================================

bool InverseKinematics::buildPostureAccelReference(
    ConstRefVector q, ConstRefVector v) {
  const auto qAct = q.tail(m_na);
  const auto vAct = v.tail(m_na);

  m_qddotPostureRef.setZero();
  m_qddotPostureRef.tail(m_na).array() =
      m_kpAcc.array() * (m_kinRef.jposRef - qAct).array() +
      m_kdAcc.array() * (m_kinRef.jvelRef - vAct).array();

  return m_qddotPostureRef.allFinite();
}

}  // namespace tsid
