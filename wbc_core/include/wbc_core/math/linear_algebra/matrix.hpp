#pragma once

#include <Eigen/Dense>
#include <Eigen/QR>

namespace wbc::math {

double clamp(double value, double lower = 0.0, double upper = 1.0);
Eigen::VectorXd clamp(const Eigen::VectorXd& values,
                      const Eigen::VectorXd& lower,
                      const Eigen::VectorXd& upper);
Eigen::Vector2d clamp(const Eigen::Vector2d& values,
                      const Eigen::Vector2d& lower,
                      const Eigen::Vector2d& upper);

void pseudoInverse(Eigen::MatrixXd const& matrix, double threshold,
                   Eigen::MatrixXd& inverse,
                   Eigen::VectorXd* singular_values = nullptr);
Eigen::MatrixXd pseudoInverse(const Eigen::MatrixXd& matrix, double threshold);

Eigen::MatrixXd nullspaceProjector(const Eigen::MatrixXd& jacobian,
                                   double threshold = 0.00001,
                                   const Eigen::MatrixXd* weight = nullptr);

Eigen::MatrixXd weightedPseudoInverse(const Eigen::MatrixXd& jacobian,
                                      const Eigen::MatrixXd& weight,
                                      double threshold = 0.0001);
void weightedPseudoInverse(const Eigen::MatrixXd& jacobian,
                           const Eigen::MatrixXd& weight,
                           double threshold, Eigen::MatrixXd& inverse);

Eigen::MatrixXd hstack(const Eigen::MatrixXd& a, const Eigen::MatrixXd& b);
Eigen::MatrixXd vstack(const Eigen::MatrixXd& a, const Eigen::MatrixXd& b);
Eigen::MatrixXd blockDiag(const Eigen::MatrixXd& a, const Eigen::MatrixXd& b);

}  // namespace wbc::math
