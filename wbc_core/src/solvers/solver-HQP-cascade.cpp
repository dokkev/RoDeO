//
// Copyright (c) 2026
//
// Cascaded HQP solver implementation.
//

#include "wbc_core/solvers/solver-HQP-cascade.hpp"

#include <algorithm>

#include <Eigen/QR>

#include "wbc_core/math/constraint-equality.hpp"
#include "wbc_core/solvers/solver-HQP-factory.hpp"

namespace wbc {
namespace solvers {

using namespace math;

namespace {

SolverHQP defaultInnerSolverType() {
#ifdef TSID_WITH_PROXSUITE
  return SOLVER_HQP_PROXQP;
#else
  return SOLVER_HQP_EIQUADPROG;
#endif
}

}  // namespace

SolverHQPCascade::SolverHQPCascade(const std::string& name)
    : SolverHQPCascade(name, defaultInnerSolverType()) {}

SolverHQPCascade::SolverHQPCascade(const std::string& name,
                                   SolverHQP inner_solver_type)
    : SolverHQPBase(name), m_innerSolverType(inner_solver_type) {
  m_innerSolver = makeInnerSolver();
}

std::unique_ptr<SolverHQPBase> SolverHQPCascade::makeInnerSolver() const {
  auto solver = std::unique_ptr<SolverHQPBase>(
      SolverHQPFactory::createNewSolver(m_innerSolverType, m_name + "_inner"));
  ApplySolverQPParams(*solver, m_qpParams);
  return solver;
}

void SolverHQPCascade::setInnerSolverType(SolverHQP inner_solver_type) {
  if (m_innerSolverType == inner_solver_type) {
    return;
  }
  m_innerSolverType = inner_solver_type;
  m_innerSolver = makeInnerSolver();
}

void SolverHQPCascade::resize(unsigned int n, unsigned int neq,
                              unsigned int nin) {
  m_n = n;
  m_neq = neq;
  m_nin = nin;
}

void SolverHQPCascade::retrieveQPData(const HQPData& /*problemData*/,
                                      const bool /*hessianRegularization*/) {
  // Not used directly; solve() handles the per-level subproblems.
}

double SolverHQPCascade::getObjectiveValue() {
  return m_innerSolver ? m_innerSolver->getObjectiveValue() : 0.0;
}

bool SolverHQPCascade::setMaximumIterations(unsigned int maxIter) {
  SolverHQPBase::setMaximumIterations(maxIter);
  if (m_innerSolver) {
    m_innerSolver->setMaximumIterations(maxIter);
  }
  return true;
}

void SolverHQPCascade::setQPParams(const SolverQPParams& params) {
  m_qpParams = params;
  ApplySolverQPParams(*this, m_qpParams);
  if (m_innerSolver) {
    ApplySolverQPParams(*m_innerSolver, m_qpParams);
  }
}

const HQPOutput& SolverHQPCascade::solve(const HQPData& problemData) {
  const int nLevels = static_cast<int>(problemData.size());

  if (nLevels == 0) {
    m_output.status = HQP_STATUS_ERROR;
    return m_output;
  }

  if (nLevels <= 2) {
    HQPData twoLevel;
    twoLevel.resize(2);
    twoLevel[0] = problemData[0];
    if (nLevels > 1) {
      twoLevel[1] = problemData[1];
    }

    unsigned int n = 0;
    unsigned int neq = 0;
    unsigned int nin = 0;
    for (const auto& pair : twoLevel[0]) {
      if (n == 0) {
        n = pair.second->cols();
      }
      if (pair.second->isEquality()) {
        neq += pair.second->rows();
      } else {
        nin += pair.second->rows();
      }
    }
    if (n == 0 && twoLevel.size() > 1 && !twoLevel[1].empty()) {
      n = twoLevel[1][0].second->cols();
    }

    m_innerSolver->resize(n, neq, nin);
    m_output = m_innerSolver->solve(twoLevel);
    return m_output;
  }

  const ConstraintLevel& cl0 = problemData[0];
  unsigned int n = 0;
  unsigned int baseNeq = 0;

  for (const auto& pair : cl0) {
    if (n == 0) {
      n = pair.second->cols();
    }
    if (pair.second->isEquality()) {
      baseNeq += pair.second->rows();
    }
  }

  if (n == 0) {
    for (int k = 1; k < nLevels; ++k) {
      if (!problemData[k].empty()) {
        n = problemData[k][0].second->cols();
        break;
      }
    }
  }

  if (n == 0) {
    m_output.status = HQP_STATUS_ERROR;
    return m_output;
  }

  m_cascadeRows = 0;
  int maxCascadeRows = 0;
  for (int k = 1; k < nLevels; ++k) {
    for (const auto& pair : problemData[k]) {
      maxCascadeRows += pair.second->rows();
    }
  }
  m_cascadeA.setZero(maxCascadeRows, n);
  m_cascadeB.setZero(maxCascadeRows);

  Matrix compressedA;
  Vector compressedB;
  int compressedRows = 0;

  for (int k = 1; k < nLevels; ++k) {
    const ConstraintLevel& clk = problemData[k];
    if (clk.empty()) {
      continue;
    }

    HQPData subProblem;
    subProblem.resize(2);
    subProblem[0] = cl0;

    if (compressedRows > 0) {
      auto cascadeCst = std::make_shared<ConstraintEquality>("cascade_prior",
                                                             compressedRows, n);
      cascadeCst->matrix() = compressedA.topRows(compressedRows);
      cascadeCst->vector() = compressedB.head(compressedRows);
      subProblem[0].push_back(
          wbc::solvers::make_pair<double, std::shared_ptr<ConstraintBase>>(
              1.0, cascadeCst));
    }

    subProblem[1] = clk;

    unsigned int subNeq = 0;
    unsigned int subNin = 0;
    for (const auto& pair : subProblem[0]) {
      if (pair.second->isEquality()) {
        subNeq += pair.second->rows();
      } else {
        subNin += pair.second->rows();
      }
    }

    m_innerSolver->resize(n, subNeq, subNin);
    const HQPOutput& subOutput = m_innerSolver->solve(subProblem);

    if (subOutput.status != HQP_STATUS_OPTIMAL) {
      m_output.status = subOutput.status;
      m_output.x = subOutput.x;
      m_output.iterations = subOutput.iterations;
      return m_output;
    }

    const Vector& xk = subOutput.x;
    for (const auto& pair : clk) {
      const auto& cst = pair.second;
      if (!cst->isEquality()) {
        continue;
      }
      const int rows = cst->rows();
      const Matrix& A = cst->matrix();

      m_cascadeA.middleRows(m_cascadeRows, rows) = A;
      m_cascadeB.segment(m_cascadeRows, rows) = A * xk;
      m_cascadeRows += rows;
    }

    if (m_cascadeRows > 0) {
      const int totalRows = static_cast<int>(baseNeq) + m_cascadeRows;
      Matrix stacked(totalRows, n);
      Vector stackedB(totalRows);

      int row = 0;
      for (const auto& pair : cl0) {
        if (!pair.second->isEquality()) {
          continue;
        }
        const int r = pair.second->rows();
        stacked.middleRows(row, r) = pair.second->matrix();
        stackedB.segment(row, r) = pair.second->vector();
        row += r;
      }
      stacked.bottomRows(m_cascadeRows) = m_cascadeA.topRows(m_cascadeRows);
      stackedB.tail(m_cascadeRows) = m_cascadeB.head(m_cascadeRows);

      constexpr double kRankThreshold = 1e-8;
      Eigen::ColPivHouseholderQR<Matrix> qr(stacked);
      qr.setThreshold(kRankThreshold);
      const int combinedRank = static_cast<int>(qr.rank());
      const int cascadeRank =
          std::max(0, combinedRank - static_cast<int>(baseNeq));

      if (cascadeRank > 0 && cascadeRank < m_cascadeRows) {
        Eigen::ColPivHouseholderQR<Matrix> cascadeQr(
            m_cascadeA.topRows(m_cascadeRows));
        cascadeQr.setThreshold(kRankThreshold);
        const int selfRank =
            std::min(static_cast<int>(cascadeQr.rank()), cascadeRank);

        const Matrix Q = cascadeQr.householderQ() *
                         Matrix::Identity(m_cascadeRows, selfRank);
        compressedA = Q.transpose() * m_cascadeA.topRows(m_cascadeRows);
        compressedB = Q.transpose() * m_cascadeB.head(m_cascadeRows);
        compressedRows = selfRank;
      } else if (cascadeRank >= m_cascadeRows) {
        compressedA = m_cascadeA.topRows(m_cascadeRows);
        compressedB = m_cascadeB.head(m_cascadeRows);
        compressedRows = m_cascadeRows;
      } else {
        compressedRows = 0;
      }
    }

    m_output.x = xk;
    m_output.status = HQP_STATUS_OPTIMAL;
    m_output.iterations = subOutput.iterations;
    m_output.lambda = subOutput.lambda;

    if (compressedRows + static_cast<int>(baseNeq) >= static_cast<int>(n)) {
      break;
    }
  }

  return m_output;
}

}  // namespace solvers
}  // namespace wbc
