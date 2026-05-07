//
// Copyright (c) 2026
//
// Cascaded HQP solver: solves multi-level hierarchical QPs by iterating
// from the highest-priority soft level to the lowest. Each level's optimal
// cost is preserved as an equality constraint for subsequent levels.
//
// Algorithm:
//   1. Extract level-0 hard constraints (equalities + inequalities).
//   2. For each soft level k = 1, 2, ..., K:
//      a. Build QP_k: minimize level-k weighted cost, subject to:
//         - Level-0 hard constraints
//         - For each prior soft level j < k: A_j * x = A_j * x_{j}*
//           (preserves level-j optimal cost as equality constraints)
//      b. Solve QP_k.
//   3. Return x_K (the last level's solution).
//

#ifndef __invdyn_solvers_hqp_cascade_hpp__
#define __invdyn_solvers_hqp_cascade_hpp__

#include "wbc_core/solvers/solver-HQP-base.hpp"

#include <Eigen/Dense>
#include <memory>

namespace tsid {
namespace solvers {

class TSID_DLLAPI SolverHQPCascade : public SolverHQPBase {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef math::Matrix Matrix;
  typedef math::Vector Vector;

  SolverHQPCascade(const std::string& name);

  void resize(unsigned int n, unsigned int neq, unsigned int nin) override;

  const HQPOutput& solve(const HQPData& problemData) override;

  void retrieveQPData(const HQPData& problemData,
                      const bool hessianRegularization = true) override;

  double getObjectiveValue() override;

  bool setMaximumIterations(unsigned int maxIter) override;
  void setVerbose(bool isVerbose) override;

 private:
  unsigned int m_n{0};
  unsigned int m_neq{0};
  unsigned int m_nin{0};

  // Per-level QP buffers
  Matrix m_H;
  Vector m_g;
  Matrix m_CE;
  Vector m_ce0;
  Matrix m_CI;
  Vector m_ci_lb;
  Vector m_ci_ub;

  // Accumulated equality constraints from prior levels
  Matrix m_cascadeA;
  Vector m_cascadeB;
  int m_cascadeRows{0};

  double m_hessianReg{DEFAULT_HESSIAN_REGULARIZATION};

  // Inner QP solver (ProxQP or eiquadprog-fast)
  std::unique_ptr<SolverHQPBase> m_innerSolver;
};

}  // namespace solvers
}  // namespace tsid

#endif  // ifndef __invdyn_solvers_hqp_cascade_hpp__
