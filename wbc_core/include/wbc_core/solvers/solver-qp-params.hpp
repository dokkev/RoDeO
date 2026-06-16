//
// Copyright (c) 2026
//
// Optional tuning parameters for HQP/QP solver backends.
//

#ifndef __wbc_solvers_solver_qp_params_hpp__
#define __wbc_solvers_solver_qp_params_hpp__

#include <optional>

namespace wbc {
namespace solvers {

class SolverHQPBase;

struct SolverQPParams {
  std::optional<unsigned int> max_iter;
  std::optional<double> max_time;
  std::optional<bool> warm_start;

  std::optional<bool> verbose;
  std::optional<double> rho;
  std::optional<double> mu_eq;
  std::optional<double> mu_ineq;
  std::optional<double> eps_abs;
  std::optional<double> eps_rel;
};

void ApplySolverQPParams(SolverHQPBase& solver, const SolverQPParams& params);

}  // namespace solvers
}  // namespace wbc

#endif  // ifndef __wbc_solvers_solver_qp_params_hpp__
