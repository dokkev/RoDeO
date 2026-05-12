//
// Copyright (c) 2017 CNRS
//

#include <wbc_core/solvers/solver-HQP-factory.hpp>
#include <wbc_core/solvers/solver-HQP-eiquadprog.hpp>
#include <wbc_core/solvers/solver-HQP-eiquadprog-fast.hpp>

#ifdef TSID_QPMAD_FOUND
#include <wbc_core/solvers/solver-HQP-qpmad.hpp>
#endif

#ifdef TSID_WITH_PROXSUITE
#include <wbc_core/solvers/solver-proxqp.hpp>
#endif

#ifdef TSID_WITH_OSQP
#include <wbc_core/solvers/solver-osqp.hpp>
#endif

#ifdef QPOASES_FOUND
#include <wbc_core/solvers/solver-HQP-qpoases.hh>
#endif

namespace wbc {
namespace solvers {

SolverHQPBase* SolverHQPFactory::createNewSolver(const SolverHQP solverType,
                                                 const std::string& name) {
  if (solverType == SOLVER_HQP_EIQUADPROG) return new SolverHQuadProg(name);

  if (solverType == SOLVER_HQP_EIQUADPROG_FAST)
    return new SolverHQuadProgFast(name);

#ifdef TSID_QPMAD_FOUND
  if (solverType == SOLVER_HQP_QPMAD) return new SolverHQpmad(name);
#endif

#ifdef TSID_WITH_PROXSUITE
  if (solverType == SOLVER_HQP_PROXQP) return new SolverProxQP(name);
#endif

#ifdef TSID_WITH_OSQP
  if (solverType == SOLVER_HQP_OSQP) return new SolverOSQP(name);
#endif

#ifdef QPOASES_FOUND
  if (solverType == SOLVER_HQP_QPOASES) return new Solver_HQP_qpoases(name);
#endif

  PINOCCHIO_CHECK_INPUT_ARGUMENT(false, "Specified solver type not recognized");
  return NULL;
}

}  // namespace solvers
}  // namespace wbc
