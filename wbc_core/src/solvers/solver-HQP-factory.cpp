//
// Copyright (c) 2017 CNRS
//

#include <wbc_core/solvers/solver-HQP-factory.hpp>
#include <wbc_core/solvers/solver-HQP-eiquadprog.hpp>
#include <wbc_core/solvers/solver-HQP-eiquadprog-fast.hpp>
#include <wbc_core/solvers/solver-HQP-cascade.hpp>

#include <wbc_core/solvers/solver-proxqp.hpp>

namespace tsid {
namespace solvers {

std::unique_ptr<SolverHQPBase> SolverHQPFactory::createNewSolver(
    const SolverHQP solverType, const std::string& name) {
  if (solverType == SOLVER_HQP_EIQUADPROG) {
    return std::make_unique<SolverHQuadProg>(name);
  }
  if (solverType == SOLVER_HQP_EIQUADPROG_FAST) {
    return std::make_unique<SolverHQuadProgFast>(name);
  }
  if (solverType == SOLVER_HQP_CASCADE) {
    return std::make_unique<SolverHQPCascade>(name);
  }
  if (solverType == SOLVER_HQP_PROXQP) {
    return std::make_unique<SolverProxQP>(name);
  }

  PINOCCHIO_CHECK_INPUT_ARGUMENT(false, "Specified solver type not recognized");
  return nullptr;
}

}  // namespace solvers
}  // namespace tsid
