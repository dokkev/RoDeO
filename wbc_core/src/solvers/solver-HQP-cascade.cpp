//
// Copyright (c) 2026
//
// Cascaded HQP solver implementation.
//

#include "wbc_core/solvers/solver-HQP-cascade.hpp"
#include "wbc_core/solvers/solver-HQP-eiquadprog-fast.hpp"
#include "wbc_core/math/constraint-equality.hpp"
#include "wbc_core/math/utils.hpp"

#include <Eigen/QR>

namespace tsid {
namespace solvers {

using namespace math;

SolverHQPCascade::SolverHQPCascade(const std::string& name)
    : SolverHQPBase(name) {
  m_innerSolver = std::make_unique<SolverHQuadProgFast>(name + "_inner");
}

void SolverHQPCascade::resize(unsigned int n, unsigned int neq,
                               unsigned int nin) {
  m_n = n;
  m_neq = neq;
  m_nin = nin;
}

void SolverHQPCascade::retrieveQPData(const HQPData& /*problemData*/,
                                       const bool /*hessianRegularization*/) {
  // Not used directly; solve() handles everything.
}

double SolverHQPCascade::getObjectiveValue() {
  return m_innerSolver ? m_innerSolver->getObjectiveValue() : 0.0;
}

bool SolverHQPCascade::setMaximumIterations(unsigned int maxIter) {
  SolverHQPBase::setMaximumIterations(maxIter);
  if (m_innerSolver) m_innerSolver->setMaximumIterations(maxIter);
  return true;
}

void SolverHQPCascade::setVerbose(bool isVerbose) {
  SolverHQPBase::setVerbose(isVerbose);
  if (m_innerSolver) {
    m_innerSolver->setVerbose(isVerbose);
  }
}

const HQPOutput& SolverHQPCascade::solve(const HQPData& problemData) {
  const int nLevels = static_cast<int>(problemData.size());

  // Need at least level 0 (hard constraints)
  if (nLevels == 0) {
    m_output.status = HQP_STATUS_ERROR;
    return m_output;
  }

  // If only 2 levels, delegate directly to inner solver (fast path)
  if (nLevels <= 2) {
    // Build a 2-level HQPData and solve directly
    HQPData twoLevel;
    twoLevel.resize(2);
    twoLevel[0] = problemData[0];
    if (nLevels > 1) {
      twoLevel[1] = problemData[1];
    }

    // Count constraints for resize
    unsigned int n = 0, neq = 0, nin = 0;
    for (const auto& pair : twoLevel[0]) {
      if (n == 0) n = pair.second->cols();
      if (pair.second->isEquality())
        neq += pair.second->rows();
      else
        nin += pair.second->rows();
    }
    if (n == 0 && twoLevel.size() > 1 && !twoLevel[1].empty()) {
      n = twoLevel[1][0].second->cols();
    }

    m_innerSolver->resize(n, neq, nin);
    m_output = m_innerSolver->solve(twoLevel);
    return m_output;
  }

  // ── Multi-level cascaded solve ──────────────────────────────────────────

  // 1. Extract level-0 hard constraints
  const ConstraintLevel& cl0 = problemData[0];
  unsigned int n = 0;
  unsigned int baseNeq = 0, baseNin = 0;

  for (const auto& pair : cl0) {
    if (n == 0) n = pair.second->cols();
    if (pair.second->isEquality())
      baseNeq += pair.second->rows();
    else
      baseNin += pair.second->rows();
  }

  // If n not determined from level 0, get from level 1
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

  // Reset cascade constraints accumulator (raw, before compression)
  m_cascadeRows = 0;
  // Pre-allocate max possible cascade rows (sum of all soft level dims)
  int maxCascadeRows = 0;
  for (int k = 1; k < nLevels; ++k) {
    for (const auto& pair : problemData[k]) {
      maxCascadeRows += pair.second->rows();
    }
  }
  m_cascadeA.setZero(maxCascadeRows, n);
  m_cascadeB.setZero(maxCascadeRows);

  // Compressed cascade constraint (rank-reduced via QR)
  Matrix compressedA;
  Vector compressedB;
  int compressedRows = 0;

  // 2. For each soft level k = 1, ..., K
  for (int k = 1; k < nLevels; ++k) {
    const ConstraintLevel& clk = problemData[k];
    if (clk.empty()) continue;

    // Build 2-level HQPData for this sub-QP:
    //   Level 0: base hard constraints + cascade equalities from prior levels
    //   Level 1: this level's costs
    HQPData subProblem;
    subProblem.resize(2);

    // Copy base hard constraints
    subProblem[0] = cl0;

    // Add compressed cascade equality constraints from prior levels
    if (compressedRows > 0) {
      auto cascadeCst = std::make_shared<ConstraintEquality>(
          "cascade_prior", compressedRows, n);
      cascadeCst->matrix() = compressedA.topRows(compressedRows);
      cascadeCst->vector() = compressedB.head(compressedRows);
      subProblem[0].push_back(
          solvers::make_pair<double, std::shared_ptr<math::ConstraintBase>>(
              1.0, cascadeCst));
    }

    // Level 1 costs: this level's tasks
    subProblem[1] = clk;

    // Count constraints for resize
    unsigned int subNeq = 0, subNin = 0;
    for (const auto& pair : subProblem[0]) {
      if (pair.second->isEquality())
        subNeq += pair.second->rows();
      else
        subNin += pair.second->rows();
    }

    m_innerSolver->resize(n, subNeq, subNin);
    const HQPOutput& subOutput = m_innerSolver->solve(subProblem);

    if (subOutput.status != HQP_STATUS_OPTIMAL) {
      m_output.status = subOutput.status;
      m_output.x = subOutput.x;
      m_output.iterations = subOutput.iterations;
      return m_output;
    }

    // 3. Freeze this level's optimal residual as equality constraints.
    //    For each cost term A*x = b in this level, add: A*x = A*x_k*
    //    This preserves the optimal cost of level k for subsequent levels.
    const Vector& xk = subOutput.x;
    for (const auto& pair : clk) {
      const auto& cst = pair.second;
      if (!cst->isEquality()) continue;
      const int rows = cst->rows();
      const Matrix& A = cst->matrix();

      m_cascadeA.middleRows(m_cascadeRows, rows) = A;
      m_cascadeB.segment(m_cascadeRows, rows) = A * xk;
      m_cascadeRows += rows;
    }

    // 4. Compress accumulated cascade constraints via rank-revealing QR.
    //    The cascade constraints share the decision variable space with the
    //    hard equality constraints. To avoid over-determining the system
    //    (total equality rows > n), we:
    //    a) Stack hard equality matrices with cascade matrices
    //    b) QR decompose the combined system
    //    c) Keep only cascade directions that are independent of hard equalities
    if (m_cascadeRows > 0) {
      // Stack hard equalities and cascade rows for joint rank analysis
      const int totalRows = static_cast<int>(baseNeq) + m_cascadeRows;
      Matrix stacked(totalRows, n);
      Vector stackedB(totalRows);

      // Fill hard equality rows
      int row = 0;
      for (const auto& pair : cl0) {
        if (!pair.second->isEquality()) continue;
        const int r = pair.second->rows();
        stacked.middleRows(row, r) = pair.second->matrix();
        stackedB.segment(row, r) = pair.second->vector();
        row += r;
      }
      // Fill cascade rows
      stacked.bottomRows(m_cascadeRows) = m_cascadeA.topRows(m_cascadeRows);
      stackedB.tail(m_cascadeRows) = m_cascadeB.head(m_cascadeRows);

      constexpr double kRankThreshold = 1e-8;
      Eigen::ColPivHouseholderQR<Matrix> qr(stacked);
      qr.setThreshold(kRankThreshold);
      const int combinedRank = static_cast<int>(qr.rank());

      // Cascade can add at most (combinedRank - baseNeq) independent rows
      const int cascadeRank =
          std::max(0, combinedRank - static_cast<int>(baseNeq));

      if (cascadeRank > 0 && cascadeRank < m_cascadeRows) {
        // Use Q^T to rotate cascade constraints, keep first cascadeRank rows
        // that are independent of hard constraints
        Eigen::ColPivHouseholderQR<Matrix> cascadeQr(
            m_cascadeA.topRows(m_cascadeRows));
        cascadeQr.setThreshold(kRankThreshold);
        const int selfRank = std::min(
            static_cast<int>(cascadeQr.rank()), cascadeRank);

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
        // cascadeRank == 0: cascade adds nothing beyond hard constraints
        compressedRows = 0;
      }
    }

    // Store the current best solution
    m_output.x = xk;
    m_output.status = HQP_STATUS_OPTIMAL;
    m_output.iterations = subOutput.iterations;
    m_output.lambda = subOutput.lambda;

    // If cascade constraints + hard equalities fully determine x,
    // lower-priority levels have no remaining freedom. Stop early.
    if (compressedRows + static_cast<int>(baseNeq) >=
        static_cast<int>(n)) {
      break;
    }
  }

  return m_output;
}

}  // namespace solvers
}  // namespace tsid
