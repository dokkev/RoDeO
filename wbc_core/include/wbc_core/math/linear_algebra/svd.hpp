#pragma once

#include "wbc_core/math/fwd.hpp"

#include <Eigen/SVD>

namespace wbc::math {

void solveDamped(Eigen::JacobiSVD<Eigen::MatrixXd>& svd, ConstRefVector b,
                 RefVector solution, double damping = 0.0);

void solveDamped(ConstRefMatrix A, ConstRefVector b, RefVector solution,
                 double damping = 0.0);

void pseudoInverse(ConstRefMatrix A, RefMatrix Apinv, double tolerance,
                   unsigned int computationOptions = Eigen::ComputeThinU |
                                                     Eigen::ComputeThinV);

void pseudoInverse(ConstRefMatrix A,
                   Eigen::JacobiSVD<Eigen::MatrixXd>& svdDecomposition,
                   RefMatrix Apinv, double tolerance,
                   unsigned int computationOptions);

void pseudoInverse(ConstRefMatrix A,
                   Eigen::JacobiSVD<Eigen::MatrixXd>& svdDecomposition,
                   RefMatrix Apinv, double tolerance, double* nullSpaceBasisOfA,
                   int& nullSpaceRows, int& nullSpaceCols,
                   unsigned int computationOptions);

void dampedPseudoInverse(ConstRefMatrix A,
                         Eigen::JacobiSVD<Eigen::MatrixXd>& svdDecomposition,
                         RefMatrix Apinv, double tolerance,
                         double dampingFactor,
                         unsigned int computationOptions = Eigen::ComputeThinU |
                                                           Eigen::ComputeThinV,
                         double* nullSpaceBasisOfA = 0, int* nullSpaceRows = 0,
                         int* nullSpaceCols = 0);

void nullspaceBasis(
    const Eigen::JacobiSVD<Eigen::MatrixXd>& svdDecomposition, double tolerance,
    double* nullSpaceBasisMatrix, int& rows, int& cols);

void nullspaceBasis(
    const Eigen::JacobiSVD<Eigen::MatrixXd>& svdDecomposition, int rank,
    double* nullSpaceBasisMatrix, int& rows, int& cols);

template <typename Derived>
inline bool isFinite(const Eigen::MatrixBase<Derived>& x) {
  return ((x - x).array() == (x - x).array()).all();
}

template <typename Derived>
inline bool hasNaN(const Eigen::MatrixBase<Derived>& x) {
  return !((x.array() == x.array())).all();
}

}  // namespace wbc::math
