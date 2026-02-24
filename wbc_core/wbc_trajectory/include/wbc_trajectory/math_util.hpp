#pragma once

#include <array>

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/QR>

namespace util {

enum class CoordinateAxis { X, Y, Z };

Eigen::Matrix3d SkewSymmetric(const Eigen::Vector3d& omg);
Eigen::MatrixXd Adjoint(const Eigen::MatrixXd& R, const Eigen::Vector3d& p);

Eigen::Vector3d QuatToExp(const Eigen::Quaternion<double>& quat);
Eigen::Quaternion<double> ExpToQuat(const Eigen::Vector3d& exp);
bool NormalizeQuaternionXYZW(std::array<double, 4>* quat_xyzw);
Eigen::Quaterniond XYZWToQuaternion(const std::array<double, 4>& quat_xyzw);
std::array<double, 4> QuaternionToXYZW(const Eigen::Quaterniond& quat);

Eigen::Quaterniond EulerZYXtoQuat(double roll, double pitch, double yaw);
Eigen::Quaterniond EulerZYXtoQuat(const Eigen::Vector3d& rpy);
Eigen::Vector3d QuatToEulerZYX(const Eigen::Quaterniond& quat_in);
Eigen::Vector3d QuatToEulerXYZ(const Eigen::Quaterniond& quat_in);

Eigen::Vector3d EulerZYXRatestoAngVel(double roll, double pitch, double yaw,
                                      double roll_rate, double pitch_rate,
                                      double yaw_rate);

Eigen::Vector3d RPYFromSO3(const Eigen::Matrix3d& R);
Eigen::Matrix3d SO3FromRPY(double r, double p, double y);

Eigen::Matrix3d CoordinateRotation(CoordinateAxis axis, double theta);
Eigen::Matrix3d RotMatrixToYawMatrix(const Eigen::Matrix3d& rot);
Eigen::Matrix3d QuaternionToYawMatrix(const Eigen::Quaterniond& quat);
double QuaternionToYaw(const Eigen::Quaterniond& quat);
void WrapYawToPi(Eigen::Quaterniond& quat);
void WrapYawToPi(Eigen::Vector3d& rpy);

void AvoidQuatJump(const Eigen::Quaternion<double>& des_ori,
                   Eigen::Quaternion<double>& act_ori);

double Clamp(double s_in, double lo = 0.0, double hi = 1.0);
Eigen::VectorXd ClampVector(const Eigen::VectorXd& vec_in,
                            const Eigen::VectorXd& vec_min,
                            const Eigen::VectorXd& vec_max);
Eigen::Vector2d Clamp2DVector(const Eigen::Vector2d& vec_in,
                              const Eigen::Vector2d& vec_min,
                              const Eigen::Vector2d& vec_max);

void PseudoInverse(Eigen::MatrixXd const& matrix, double sigma_threshold,
                   Eigen::MatrixXd& inv_matrix,
                   Eigen::VectorXd* opt_sigma_out = nullptr);
Eigen::MatrixXd PseudoInverse(const Eigen::MatrixXd& matrix,
                              const double& threshold);

Eigen::MatrixXd GetNullSpace(const Eigen::MatrixXd& J,
                             double threshold = 0.00001,
                             const Eigen::MatrixXd* W = nullptr);

Eigen::MatrixXd WeightedPseudoInverse(const Eigen::MatrixXd& J,
                                      const Eigen::MatrixXd& W,
                                      double sigma_threshold = 0.0001);
void WeightedPseudoInverse(const Eigen::MatrixXd& J, const Eigen::MatrixXd& W,
                           double sigma_threshold, Eigen::MatrixXd& Jinv);

Eigen::MatrixXd HStack(const Eigen::MatrixXd& a, const Eigen::MatrixXd& b);
Eigen::MatrixXd VStack(const Eigen::MatrixXd& a, const Eigen::MatrixXd& b);
Eigen::MatrixXd BlockDiagonalMatrix(const Eigen::MatrixXd& a,
                                    const Eigen::MatrixXd& b);

} // namespace util
