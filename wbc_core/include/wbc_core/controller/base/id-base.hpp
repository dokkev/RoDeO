//
// Copyright (c) 2026
//
// Solver-policy-agnostic base interface for inverse-dynamics controllers.
//

#ifndef WBC_CORE_CONTROLLER_BASE_ID_BASE_HPP_
#define WBC_CORE_CONTROLLER_BASE_ID_BASE_HPP_

#include <cassert>
#include <cmath>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <Eigen/Core>
#include <pinocchio/multibody/data.hpp>

#include "wbc_core/contacts/contact-constraint-data.hpp"
#include "wbc_core/formulations/fwd.hpp"
#include "wbc_core/formulations/id-problem.hpp"
#include "wbc_core/formulations/id-solution.hpp"
#include "wbc_core/math/fwd.hpp"
#include "wbc_core/robots/robot-system.hpp"

namespace wbc {

class InverseDynamicsBase {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  using Data = pinocchio::Data;

  explicit InverseDynamicsBase(robots::RobotSystem& robot)
      : m_robot(robot),
        m_data(robot.model()),
        m_solution(std::make_unique<IDSolution>()) {
    m_nv = robot.nv();
    m_na = robot.na();
    m_nvFloat = m_nv - m_na;

    m_zeroQddotRef = math::Vector::Zero(m_nv);
    m_zero_h_ext = math::Vector::Zero(m_nv);
    m_tauFull = math::Vector::Zero(m_nv);
    resetSolution(m_zeroQddotRef, 0);
  }

  virtual ~InverseDynamicsBase() = default;

  virtual const IDSolution& solve(const IDProblem& problem, double dt) = 0;

  virtual const IDSolution& solution() const { return *m_solution; }

  virtual Data& data() { return m_data; }
  virtual const Data& data() const { return m_data; }

  virtual void setTimingEnabled(bool enabled) { m_timingEnabled = enabled; }
  virtual bool timingEnabled() const { return m_timingEnabled; }

 protected:
  struct ContactLambdaLayout {
    int lambdaOffset{0};
    int lambdaDim{0};
  };

  struct StackedContactData {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    int lambdaDim{0};
    math::Matrix Jc;
    math::Vector contact_motion_rhs;
    math::Matrix Uf;
    math::Vector uf_lb;
    math::Vector uf_ub;
    std::unordered_map<std::string, ContactLambdaLayout> contactLayout;

    bool hasContactForces() const { return lambdaDim > 0; }
    bool hasContactKinematics() const { return Jc.rows() > 0; }
    bool hasFrictionConstraints() const {
      return hasContactForces() && Uf.rows() > 0;
    }
  };

  robots::RobotSystem& robot() { return m_robot; }
  const robots::RobotSystem& robot() const { return m_robot; }

  int nv() const { return m_nv; }
  int na() const { return m_na; }
  int nvFloat() const { return m_nvFloat; }

  const math::Vector& zeroExternalWrench() const { return m_zero_h_ext; }

  void beginSolveCycle(const math::Vector& qddot_ref) {
    resetSolution(qddot_ref, 0);
  }

  void stackContactData(StackedContactData& out,
                        const std::vector<ContactConstraintData>& contacts)
      const {
    out = StackedContactData{};

    int totalLambdaDim = 0;
    int totalUfRows = 0;
    int totalMotionDim = 0;
    for (const auto& contact : contacts) {
      totalLambdaDim += contact.lambdaDim();
      totalUfRows += static_cast<int>(contact.Uf.rows());
      totalMotionDim += contact.motionDim();
    }

    out.Jc.setZero(totalMotionDim, m_nv);
    out.contact_motion_rhs.setZero(totalMotionDim);
    out.Uf.setZero(totalUfRows, totalLambdaDim);
    out.uf_lb.setZero(totalUfRows);
    out.uf_ub.setZero(totalUfRows);

    int motionRow = 0;
    int lambdaOffset = 0;
    int ufRow = 0;

    for (const auto& contact : contacts) {
      const int motionDim = contact.motionDim();
      const int lambdaDim = contact.lambdaDim();

      assert(contact.Jc.cols() == m_nv);
      assert(contact.motion_rhs.size() == motionDim);
      assert(contact.T.cols() == lambdaDim);
      if (motionDim > 0) {
        out.Jc.block(motionRow, 0, motionDim, m_nv) = contact.Jc;
        out.contact_motion_rhs.segment(motionRow, motionDim) =
            contact.motion_rhs;
      }

      if (lambdaDim > 0) {
        assert(contact.Uf.cols() == lambdaDim);
        assert(contact.uf_lb.size() == contact.Uf.rows());
        assert(contact.uf_ub.size() == contact.Uf.rows());
        out.Uf.block(ufRow, lambdaOffset, contact.Uf.rows(), lambdaDim) =
            contact.Uf;
        out.uf_lb.segment(ufRow, contact.Uf.rows()) = contact.uf_lb;
        out.uf_ub.segment(ufRow, contact.Uf.rows()) = contact.uf_ub;
      }

      out.contactLayout[contact.name] = {lambdaOffset, lambdaDim};
      motionRow += motionDim;
      lambdaOffset += lambdaDim;
      ufRow += static_cast<int>(contact.Uf.rows());
    }

    out.lambdaDim = totalLambdaDim;
  }

  const math::Vector& problemQddotRef(const IDProblem& problem) const {
    if (problem.qddot_ref) {
      assert(problem.qddot_ref->size() == m_nv);
      return *problem.qddot_ref;
    }
    return m_zeroQddotRef;
  }

  bool validateBaseInput(double dt) const {
    return m_robot.hasState() && std::isfinite(dt) && dt >= 0.0;
  }

  void updateRobotModel() {
    assert(m_robot.generalized_q().size() == m_robot.nq());
    assert(m_robot.generalized_v().size() == m_nv);
    m_robot.computeAllTerms(m_data, m_robot.generalized_q(),
                            m_robot.generalized_v());
  }

  void resetSolution(math::ConstRefVector qddot_ref, int lambda_dim) {
    assert(qddot_ref.size() == m_nv);
    m_solution->qddot_ref = qddot_ref;
    m_solution->delta_qddot_sol.setZero(m_nv);
    m_solution->qddot_sol = qddot_ref;
    if (lambda_dim > 0) {
      m_solution->lambda_sol.setZero(lambda_dim);
    } else {
      m_solution->lambda_sol.resize(0);
    }
    m_solution->tau_sol.setZero(m_na);
    m_solution->success = false;
  }

  void setAccelerationSolution(math::ConstRefVector qddot_ref,
                               math::ConstRefVector delta_qddot_sol) {
    assert(qddot_ref.size() == m_nv);
    assert(delta_qddot_sol.size() == m_nv);
    m_solution->qddot_ref = qddot_ref;
    m_solution->delta_qddot_sol = delta_qddot_sol;
    m_solution->qddot_sol = qddot_ref + m_solution->delta_qddot_sol;
  }

  bool solutionVectorsFinite() const {
    return m_solution->qddot_ref.allFinite() &&
           m_solution->delta_qddot_sol.allFinite() &&
           m_solution->qddot_sol.allFinite() &&
           m_solution->lambda_sol.allFinite() &&
           m_solution->tau_sol.allFinite();
  }

  robots::RobotSystem& m_robot;
  Data m_data;
  std::unique_ptr<IDSolution> m_solution;
  bool m_timingEnabled{false};

  int m_nv{0};
  int m_na{0};
  int m_nvFloat{0};

  math::Vector m_zeroQddotRef;
  math::Vector m_zero_h_ext;
  math::Vector m_tauFull;
};

}  // namespace wbc

#endif  // WBC_CORE_CONTROLLER_BASE_ID_BASE_HPP_
