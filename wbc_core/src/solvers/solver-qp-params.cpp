//
// Copyright (c) 2026
//

#include "wbc_core/solvers/solver-qp-params.hpp"

#ifdef TSID_WITH_OSQP
#include "wbc_core/solvers/solver-osqp.hpp"
#endif

#ifdef TSID_WITH_PROXSUITE
#include "wbc_core/solvers/solver-proxqp.hpp"
#endif

namespace wbc {
namespace solvers {

void ApplySolverQPParams(SolverHQPBase& solver, const SolverQPParams& params) {
  if (params.max_iter) {
    solver.setMaximumIterations(*params.max_iter);
  }
  if (params.max_time) {
    solver.setMaximumTime(*params.max_time);
  }
  if (params.warm_start) {
    solver.setUseWarmStart(*params.warm_start);
  }

#ifdef TSID_WITH_PROXSUITE
  if (auto* proxqp = dynamic_cast<SolverProxQP*>(&solver)) {
    if (params.verbose) {
      proxqp->setVerbose(*params.verbose);
    }
    if (params.rho) {
      proxqp->setRho(*params.rho);
    }
    if (params.mu_eq) {
      proxqp->setMuEquality(*params.mu_eq);
    }
    if (params.mu_ineq) {
      proxqp->setMuInequality(*params.mu_ineq);
    }
    if (params.eps_abs) {
      proxqp->setEpsilonAbsolute(*params.eps_abs);
    }
    if (params.eps_rel) {
      proxqp->setEpsilonRelative(*params.eps_rel);
    }
  }
#endif

#ifdef TSID_WITH_OSQP
  if (auto* osqp = dynamic_cast<SolverOSQP*>(&solver)) {
    if (params.verbose) {
      osqp->setVerbose(*params.verbose);
    }
    if (params.rho) {
      osqp->setRho(*params.rho);
    }
    if (params.eps_abs) {
      osqp->setEpsilonAbsolute(*params.eps_abs);
    }
    if (params.eps_rel) {
      osqp->setEpsilonRelative(*params.eps_rel);
    }
  }
#endif
}

}  // namespace solvers
}  // namespace wbc
