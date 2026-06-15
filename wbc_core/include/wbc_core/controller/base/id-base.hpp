//
// Copyright (c) 2026
//
// Base interface for inverse-dynamics controllers.
//

#ifndef WBC_CORE_CONTROLLER_BASE_ID_BASE_HPP_
#define WBC_CORE_CONTROLLER_BASE_ID_BASE_HPP_

#include <memory>

#include <Eigen/Core>
#include <pinocchio/multibody/data.hpp>

#include "wbc_core/formulations/id-problem.hpp"
#include "wbc_core/formulations/id-solution.hpp"
#include "wbc_core/math/constraint-base.hpp"
#include "wbc_core/solvers/solver-HQP-base.hpp"
#include "wbc_core/solvers/solver-qp-params.hpp"

namespace wbc {

class IDBase {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  using Data = pinocchio::Data;
  using ConstraintLevel = solvers::ConstraintLevel;
  using ConstraintPtr = std::shared_ptr<math::ConstraintBase>;
  using HQPData = solvers::HQPData;

  virtual ~IDBase() = default;

  virtual const IDSolution& solve(const IDProblem& problem, double dt) = 0;
  virtual const IDSolution& solution() const = 0;

  virtual Data& data() = 0;
  virtual const Data& data() const = 0;

  virtual void setTimingEnabled(bool enabled) = 0;
  virtual bool timingEnabled() const = 0;

  virtual solvers::SolverHQP solverType() const = 0;
  virtual void setSolverType(solvers::SolverHQP solver_type) = 0;

  virtual const solvers::SolverQPParams& qpParams() const = 0;
  virtual void setQPParams(const solvers::SolverQPParams& qp_params) = 0;

 protected:
  // ID controllers assemble every solve request as HQPData. Tasks are weighted
  // soft terms; constraints are hard feasibility terms at a hierarchy level.
  static void resizeHQPData(HQPData& hqp_data, unsigned int num_levels) {
    hqp_data.clear();
    hqp_data.resize(num_levels);
  }

  static void addConstraint(HQPData& hqp_data, unsigned int level,
                            double weight, const ConstraintPtr& constraint) {
    ensureHQPLevel(hqp_data, level);
    addConstraint(hqp_data[level], weight, constraint);
  }

  static void addConstraint(ConstraintLevel& level, double weight,
                            const ConstraintPtr& constraint) {
    level.emplace_back(weight, constraint);
  }

  static void addTask(HQPData& hqp_data, unsigned int level, double weight,
                      const ConstraintPtr& constraint) {
    ensureHQPLevel(hqp_data, level);
    addTask(hqp_data[level], weight, constraint);
  }

  static void addTask(ConstraintLevel& level, double weight,
                      const ConstraintPtr& constraint) {
    level.emplace_back(weight, constraint);
  }

 private:
  static void ensureHQPLevel(HQPData& hqp_data, unsigned int level) {
    if (hqp_data.size() <= level) {
      hqp_data.resize(level + 1);
    }
  }
};

}  // namespace wbc

#endif  // WBC_CORE_CONTROLLER_BASE_ID_BASE_HPP_
