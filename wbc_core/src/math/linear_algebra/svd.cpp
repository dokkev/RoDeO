#include "wbc_core/math/linear_algebra/svd.hpp"

#include <cassert>

namespace wbc::math {

void solveDamped(Eigen::JacobiSVD<Eigen::MatrixXd>& svd, ConstRefVector b,
                 RefVector solution, double damping) {
  assert(svd.rows() == b.size());
  const double d2 = damping * damping;
  const long int nzsv = svd.nonzeroSingularValues();
  Eigen::VectorXd tmp(svd.cols());
  tmp.noalias() = svd.matrixU().leftCols(nzsv).adjoint() * b;
  double sv;
  for (long int i = 0; i < nzsv; i++) {
    sv = svd.singularValues()(i);
    tmp(i) *= sv / (sv * sv + d2);
  }
  solution = svd.matrixV().leftCols(nzsv) * tmp;
}

void solveDamped(ConstRefMatrix A, ConstRefVector b, RefVector solution,
                 double damping) {
  assert(A.rows() == b.size());
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(A.rows(), A.cols());
  svd.compute(A, Eigen::ComputeThinU | Eigen::ComputeThinV);

  solveDamped(svd, b, solution, damping);
}

void pseudoInverse(ConstRefMatrix A, RefMatrix Apinv, double tolerance,
                   unsigned int computationOptions) {
  Eigen::JacobiSVD<Eigen::MatrixXd> svdDecomposition(A.rows(), A.cols());
  pseudoInverse(A, svdDecomposition, Apinv, tolerance, computationOptions);
}

void pseudoInverse(ConstRefMatrix A,
                   Eigen::JacobiSVD<Eigen::MatrixXd>& svdDecomposition,
                   RefMatrix Apinv, double tolerance,
                   unsigned int computationOptions) {
  int nullSpaceRows = -1;
  int nullSpaceCols = -1;
  pseudoInverse(A, svdDecomposition, Apinv, tolerance, (double*)0,
                nullSpaceRows, nullSpaceCols, computationOptions);
}

void pseudoInverse(ConstRefMatrix A,
                   Eigen::JacobiSVD<Eigen::MatrixXd>& svdDecomposition,
                   RefMatrix Apinv, double tolerance, double* nullSpaceBasisOfA,
                   int& nullSpaceRows, int& nullSpaceCols,
                   unsigned int computationOptions) {
  using namespace Eigen;

  if (computationOptions == 0) {
    return;
  }
  svdDecomposition.compute(A, computationOptions);

  JacobiSVD<MatrixXd>::SingularValuesType singularValues =
      svdDecomposition.singularValues();
  long int singularValuesSize = singularValues.size();
  int rank = 0;
  for (long int idx = 0; idx < singularValuesSize; idx++) {
    if (tolerance > 0 && singularValues(idx) > tolerance) {
      singularValues(idx) = 1.0 / singularValues(idx);
      rank++;
    } else {
      singularValues(idx) = 0.0;
    }
  }

  Apinv = svdDecomposition.matrixV().leftCols(singularValuesSize) *
          singularValues.asDiagonal() *
          svdDecomposition.matrixU().leftCols(singularValuesSize).adjoint();

  if (nullSpaceBasisOfA && (computationOptions & ComputeFullV)) {
    nullspaceBasis(svdDecomposition, rank, nullSpaceBasisOfA, nullSpaceRows,
                   nullSpaceCols);
  }
}

void dampedPseudoInverse(ConstRefMatrix A,
                         Eigen::JacobiSVD<Eigen::MatrixXd>& svdDecomposition,
                         RefMatrix Apinv, double tolerance,
                         double dampingFactor, unsigned int computationOptions,
                         double* nullSpaceBasisOfA, int* nullSpaceRows,
                         int* nullSpaceCols) {
  using namespace Eigen;

  if (computationOptions == 0) {
    return;
  }
  svdDecomposition.compute(A, computationOptions);

  JacobiSVD<MatrixXd>::SingularValuesType singularValues =
      svdDecomposition.singularValues();

  const long int singularValuesSize = singularValues.size();
  const double d2 = dampingFactor * dampingFactor;
  int rank = 0;
  for (int idx = 0; idx < singularValuesSize; idx++) {
    if (singularValues(idx) > tolerance) rank++;
    singularValues(idx) =
        singularValues(idx) / ((singularValues(idx) * singularValues(idx)) + d2);
  }

  Apinv = svdDecomposition.matrixV().leftCols(singularValuesSize) *
          singularValues.asDiagonal() *
          svdDecomposition.matrixU().leftCols(singularValuesSize).adjoint();

  if (nullSpaceBasisOfA && nullSpaceRows && nullSpaceCols &&
      (computationOptions & ComputeFullV)) {
    nullspaceBasis(svdDecomposition, rank, nullSpaceBasisOfA, *nullSpaceRows,
                   *nullSpaceCols);
  }
}

void nullspaceBasis(
    const Eigen::JacobiSVD<Eigen::MatrixXd>& svdDecomposition, double tolerance,
    double* nullSpaceBasisMatrix, int& rows, int& cols) {
  using namespace Eigen;
  JacobiSVD<MatrixXd>::SingularValuesType singularValues =
      svdDecomposition.singularValues();
  int rank = 0;
  for (int idx = 0; idx < singularValues.size(); idx++) {
    if (tolerance > 0 && singularValues(idx) > tolerance) {
      rank++;
    }
  }
  nullspaceBasis(svdDecomposition, rank, nullSpaceBasisMatrix, rows, cols);
}

void nullspaceBasis(
    const Eigen::JacobiSVD<Eigen::MatrixXd>& svdDecomposition, int rank,
    double* nullSpaceBasisMatrix, int& rows, int& cols) {
  using namespace Eigen;
  const MatrixXd& vMatrix = svdDecomposition.matrixV();
  rows = (int)vMatrix.cols();
  cols = (int)vMatrix.cols() - rank;
  Map<MatrixXd> map(nullSpaceBasisMatrix, rows, cols);
  map = vMatrix.rightCols(vMatrix.cols() - rank);
}

}  // namespace wbc::math
