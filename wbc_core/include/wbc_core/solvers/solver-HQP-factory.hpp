//
// Copyright (c) 2017 CNRS
//

#ifndef __invdyn_solvers_hqp_factory_hpp__
#define __invdyn_solvers_hqp_factory_hpp__

#include <wbc_core/solvers/solver-HQP-base.hpp>

#include <memory>
#include <pinocchio/macros.hpp>  // for input argument checking and exceptions

namespace tsid {
namespace solvers {

struct SolverHQPFactory {
  /**
   * @brief Create a new HQP solver of the specified type.
   *
   * @param solverType Type of HQP solver.
   * @param name Name of the solver.
   *
   * @return A unique_ptr to the new solver.
   */
  static std::unique_ptr<SolverHQPBase> createNewSolver(
      const SolverHQP solverType, const std::string& name);

  /**
   * @brief Create a new HQP solver of the specified type (compile-time sized).
   *
   * @param solverType Type of HQP solver.
   * @param name Name of the solver.
   *
   * @return A unique_ptr to the new solver.
   */
  template <int nVars, int nEqCon, int nIneqCon>
  static std::unique_ptr<SolverHQPBase> createNewSolver(
      const SolverHQP solverType, const std::string& name);
};

}  // namespace solvers
}  // namespace tsid

#endif  // ifndef __invdyn_solvers_hqp_factory_hpp__
