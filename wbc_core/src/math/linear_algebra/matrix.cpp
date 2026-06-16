#include "wbc_core/math/linear_algebra/matrix.hpp"

#include <cassert>

namespace wbc::math {

double clamp(const double value, const double lower, const double upper) {
  if (value < lower) {
    return lower;
  } else if (value > upper) {
    return upper;
  } else {
    return value;
  }
}

Eigen::VectorXd clamp(const Eigen::VectorXd& values,
                      const Eigen::VectorXd& lower,
                      const Eigen::VectorXd& upper) {
  assert(lower.size() == upper.size());
  assert(values.size() == lower.size());

  Eigen::VectorXd clamped = Eigen::VectorXd::Zero(values.size());
  for (int i = 0; i < clamped.size(); i++) {
    clamped[i] = clamp(values[i], lower[i], upper[i]);
  }
  return clamped;
}

Eigen::Vector2d clamp(const Eigen::Vector2d& values,
                      const Eigen::Vector2d& lower,
                      const Eigen::Vector2d& upper) {
  Eigen::Vector2d clamped = values;
  for (int i = 0; i < values.size(); i++) {
    clamped[i] = clamp(values[i], lower[i], upper[i]);
  }
  return clamped;
}

void pseudoInverse(Eigen::MatrixXd const& matrix, double threshold,
                   Eigen::MatrixXd& inverse,
                   Eigen::VectorXd* singular_values) {
  if ((1 == matrix.rows()) && (1 == matrix.cols())) {
    inverse.resize(1, 1);
    if (matrix.coeff(0, 0) > threshold) {
      inverse.coeffRef(0, 0) = 1.0 / matrix.coeff(0, 0);
    } else {
      inverse.coeffRef(0, 0) = 0.0;
    }
    if (singular_values) {
      singular_values->resize(1);
      singular_values->coeffRef(0) = matrix.coeff(0, 0);
    }
    return;
  }

  Eigen::JacobiSVD<Eigen::MatrixXd> svd(matrix, Eigen::ComputeThinU |
                                                    Eigen::ComputeThinV);
  int const nrows(svd.singularValues().rows());
  Eigen::MatrixXd invS;
  invS = Eigen::MatrixXd::Zero(nrows, nrows);
  for (int ii(0); ii < nrows; ++ii) {
    if (svd.singularValues().coeff(ii) > threshold) {
      invS.coeffRef(ii, ii) = 1.0 / svd.singularValues().coeff(ii);
    }
  }
  inverse = svd.matrixV() * invS * svd.matrixU().transpose();
  if (singular_values) {
    *singular_values = svd.singularValues();
  }
}

Eigen::MatrixXd pseudoInverse(const Eigen::MatrixXd& matrix,
                              const double threshold) {
  Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd> cod(matrix.rows(),
                                                              matrix.cols());
  cod.setThreshold(threshold);
  cod.compute(matrix);
  return cod.pseudoInverse();
}

Eigen::MatrixXd nullspaceProjector(const Eigen::MatrixXd& jacobian,
                                   const double threshold,
                                   const Eigen::MatrixXd* weight) {
  Eigen::MatrixXd projector(jacobian.cols(), jacobian.cols());
  Eigen::MatrixXd jacobian_pinv;
  weight ? weightedPseudoInverse(jacobian, *weight, threshold, jacobian_pinv)
         : pseudoInverse(jacobian, threshold, jacobian_pinv);
  projector = Eigen::MatrixXd::Identity(jacobian.cols(), jacobian.cols()) -
              jacobian_pinv * jacobian;
  return projector;
}

void weightedPseudoInverse(const Eigen::MatrixXd& jacobian,
                           const Eigen::MatrixXd& weight,
                           const double threshold,
                           Eigen::MatrixXd& inverse) {
  Eigen::MatrixXd lambda(jacobian * weight * jacobian.transpose());
  Eigen::MatrixXd lambda_inv;
  pseudoInverse(lambda, threshold, lambda_inv);
  inverse = weight * jacobian.transpose() * lambda_inv;
}

Eigen::MatrixXd weightedPseudoInverse(const Eigen::MatrixXd& jacobian,
                                      const Eigen::MatrixXd& weight,
                                      const double threshold) {
  Eigen::MatrixXd inverse;
  Eigen::MatrixXd lambda(jacobian * weight * jacobian.transpose());
  Eigen::MatrixXd lambda_inv;
  pseudoInverse(lambda, threshold, lambda_inv);
  inverse = weight * jacobian.transpose() * lambda_inv;
  return inverse;
}

Eigen::MatrixXd hstack(const Eigen::MatrixXd& a, const Eigen::MatrixXd& b) {
  assert(a.rows() == b.rows());
  Eigen::MatrixXd ab = Eigen::MatrixXd::Zero(a.rows(), a.cols() + b.cols());
  ab << a, b;
  return ab;
}

Eigen::MatrixXd vstack(const Eigen::MatrixXd& a, const Eigen::MatrixXd& b) {
  assert(a.cols() == b.cols());
  Eigen::MatrixXd ab = Eigen::MatrixXd::Zero(a.rows() + b.rows(), a.cols());
  ab << a, b;
  return ab;
}

Eigen::MatrixXd blockDiag(const Eigen::MatrixXd& a,
                          const Eigen::MatrixXd& b) {
  Eigen::MatrixXd ret =
      Eigen::MatrixXd::Zero(a.rows() + b.rows(), a.cols() + b.cols());
  ret.block(0, 0, a.rows(), a.cols()) = a;
  ret.block(a.rows(), a.cols(), b.rows(), b.cols()) = b;
  return ret;
}

}  // namespace wbc::math
