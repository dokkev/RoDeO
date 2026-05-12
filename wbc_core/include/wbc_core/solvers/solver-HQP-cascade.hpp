//
// Copyright (c) 2026
//
// Cascaded HQP solver: solves multi-level hierarchical QPs by iterating
// from the highest-priority soft level to the lowest. Each level's optimal
// cost is preserved as an equality constraint for subsequent levels.
//

#ifndef __invdyn_solvers_hqp_cascade_hpp__
#define __invdyn_solvers_hqp_cascade_hpp__

#include "wbc_core/solvers/solver-HQP-base.hpp"
#include "wbc_core/solvers/solver-qp-params.hpp"

#include <Eigen/Dense>
#include <memory>

namespace wbc {
namespace solvers {

class TSID_DLLAPI SolverHQPCascade : public SolverHQPBase {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef math::Matrix Matrix;
  typedef math::Vector Vector;

  explicit SolverHQPCascade(const std::string& name);
  SolverHQPCascade(const std::string& name, SolverHQP inner_solver_type);

  void resize(unsigned int n, unsigned int neq, unsigned int nin) override;

  const HQPOutput& solve(const HQPData& problemData) override;

  void retrieveQPData(const HQPData& problemData,
                      const bool hessianRegularization = true) override;

  double getObjectiveValue() override;

  bool setMaximumIterations(unsigned int maxIter) override;

  SolverHQP innerSolverType() const { return m_innerSolverType; }
  void setInnerSolverType(SolverHQP inner_solver_type);
  const SolverQPParams& qpParams() const { return m_qpParams; }
  void setQPParams(const SolverQPParams& params);

 private:
  std::unique_ptr<SolverHQPBase> makeInnerSolver() const;

  unsigned int m_n{0};
  unsigned int m_neq{0};
  unsigned int m_nin{0};

  Matrix m_cascadeA;
  Vector m_cascadeB;
  int m_cascadeRows{0};

  SolverHQP m_innerSolverType;
  SolverQPParams m_qpParams;
  std::unique_ptr<SolverHQPBase> m_innerSolver;
};

}  // namespace solvers
}  // namespace wbc

#endif  // ifndef __invdyn_solvers_hqp_cascade_hpp__
