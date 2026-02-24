#include "wbc_formulation/basic_contact.hpp"

#include <cassert>

namespace wbc {
////////////////////////////////////////////////////////////////////////////////
Contact::Contact(PinocchioRobotSystem* robot, int dim, int target_link_idx,
                 double mu)
    : Constraint(robot, dim, target_link_idx),
      mu_(mu),
      rf_z_max_(kContactMinNormalForceZ),
      kp_(Eigen::VectorXd::Zero(dim)),
      kd_(Eigen::VectorXd::Zero(dim)),
      op_cmd_(Eigen::VectorXd::Zero(dim)) {
  des_pos_.setZero();
  des_quat_.setIdentity();
}

////////////////////////////////////////////////////////////////////////////////
PointContact::PointContact(PinocchioRobotSystem* robot, int target_link_idx,
                           double mu)
    : Contact(robot, 3, target_link_idx, mu) {
  constraint_matrix_ = Eigen::MatrixXd::Zero(6, dim_);
  constraint_vector_ = Eigen::VectorXd::Zero(6);
  rot_w_l_ = Eigen::MatrixXd::Zero(dim_, dim_);
}

////////////////////////////////////////////////////////////////////////////////
void PointContact::UpdateJacobian() {
  jacobian_ = robot_->GetLinkBodyJacobian(target_link_idx_)
                  .block(dim_, 0, dim_, robot_->NumQdot());
}

////////////////////////////////////////////////////////////////////////////////
void PointContact::UpdateJacobianDotQdot() {
  jacobian_dot_q_dot_ =
      robot_->GetLinkBodyJacobianDotQdot(target_link_idx_).tail(dim_);
}

////////////////////////////////////////////////////////////////////////////////
void PointContact::UpdateConeConstraint() {
  constraint_matrix_.setZero();
  constraint_vector_.setZero();

  constraint_matrix_(0, 2) = 1.0;
  constraint_matrix_(1, 0) = 1.0;
  constraint_matrix_(1, 2) = mu_;
  constraint_matrix_(2, 0) = -1.0;
  constraint_matrix_(2, 2) = mu_;
  constraint_matrix_(3, 1) = 1.0;
  constraint_matrix_(3, 2) = mu_;
  constraint_matrix_(4, 1) = -1.0;
  constraint_matrix_(4, 2) = mu_;
  constraint_matrix_(5, 2) = -1.0;

  constraint_vector_[5] = -rf_z_max_;
}

////////////////////////////////////////////////////////////////////////////////
void PointContact::UpdateOpCommand() {
  const Eigen::Isometry3d link_iso = robot_->GetLinkIsometry(target_link_idx_);
  const Eigen::Matrix3d rot_w_b = link_iso.linear();

  const Eigen::Vector3d pos_err_w = des_pos_ - link_iso.translation();
  const Eigen::Vector3d vel_err_w =
      -robot_->GetLinkSpatialVel(target_link_idx_).tail<3>();

  const Eigen::Vector3d pos_err_b = rot_w_b.transpose() * pos_err_w;
  const Eigen::Vector3d vel_err_b = rot_w_b.transpose() * vel_err_w;

  // Body Jacobian is used for contact constraints, so command must be in body frame.
  op_cmd_ = kp_.cwiseProduct(pos_err_b) + kd_.cwiseProduct(vel_err_b);
}

////////////////////////////////////////////////////////////////////////////////
void PointContact::SetParameters(const ContactConfig& config) {
  assert(config.kp.size() == 3 && config.kd.size() == 3 &&
         "PointContact SetParameters requires size-3 gains");
  mu_ = config.mu;
  kp_ = config.kp;
  kd_ = config.kd;
  rf_z_max_ = config.max_fz;
}

////////////////////////////////////////////////////////////////////////////////
SurfaceContact::SurfaceContact(PinocchioRobotSystem* robot, int target_link_idx,
                               double mu, double foot_half_length,
                               double foot_half_width)
    : Contact(robot, 6, target_link_idx, mu),
      foot_half_length_(foot_half_length),
      foot_half_width_(foot_half_width) {
  constraint_matrix_ = Eigen::MatrixXd::Zero(18, dim_);
  constraint_vector_ = Eigen::VectorXd::Zero(18);
  rot_w_l_ = Eigen::MatrixXd::Zero(dim_, dim_);
}

////////////////////////////////////////////////////////////////////////////////
void SurfaceContact::UpdateJacobian() {
  jacobian_ = robot_->GetLinkBodyJacobian(target_link_idx_);
}

////////////////////////////////////////////////////////////////////////////////
void SurfaceContact::UpdateJacobianDotQdot() {
  jacobian_dot_q_dot_ = robot_->GetLinkBodyJacobianDotQdot(target_link_idx_);
}

////////////////////////////////////////////////////////////////////////////////
void SurfaceContact::UpdateConeConstraint() {
  constraint_matrix_.setZero();
  constraint_vector_.setZero();

  constraint_matrix_(0, 5) = 1.0;

  constraint_matrix_(1, 3) = 1.0;
  constraint_matrix_(1, 5) = mu_;
  constraint_matrix_(2, 3) = -1.0;
  constraint_matrix_(2, 5) = mu_;
  constraint_matrix_(3, 4) = 1.0;
  constraint_matrix_(3, 5) = mu_;
  constraint_matrix_(4, 4) = -1.0;
  constraint_matrix_(4, 5) = mu_;

  constraint_matrix_(5, 0) = 1.0;
  constraint_matrix_(5, 5) = foot_half_width_;
  constraint_matrix_(6, 0) = -1.0;
  constraint_matrix_(6, 5) = foot_half_width_;
  constraint_matrix_(7, 1) = 1.0;
  constraint_matrix_(7, 5) = foot_half_length_;
  constraint_matrix_(8, 1) = -1.0;
  constraint_matrix_(8, 5) = foot_half_length_;

  constraint_matrix_(9, 0) = -mu_;
  constraint_matrix_(9, 1) = -mu_;
  constraint_matrix_(9, 2) = 1.0;
  constraint_matrix_(9, 3) = foot_half_width_;
  constraint_matrix_(9, 4) = foot_half_length_;
  constraint_matrix_(9, 5) =
      (foot_half_length_ + foot_half_width_) * mu_;

  constraint_matrix_(10, 0) = -mu_;
  constraint_matrix_(10, 1) = mu_;
  constraint_matrix_(10, 2) = 1.0;
  constraint_matrix_(10, 3) = foot_half_width_;
  constraint_matrix_(10, 4) = -foot_half_length_;
  constraint_matrix_(10, 5) =
      (foot_half_length_ + foot_half_width_) * mu_;

  constraint_matrix_(11, 0) = mu_;
  constraint_matrix_(11, 1) = -mu_;
  constraint_matrix_(11, 2) = 1.0;
  constraint_matrix_(11, 3) = -foot_half_width_;
  constraint_matrix_(11, 4) = foot_half_length_;
  constraint_matrix_(11, 5) =
      (foot_half_length_ + foot_half_width_) * mu_;

  constraint_matrix_(12, 0) = mu_;
  constraint_matrix_(12, 1) = mu_;
  constraint_matrix_(12, 2) = 1.0;
  constraint_matrix_(12, 3) = -foot_half_width_;
  constraint_matrix_(12, 4) = -foot_half_length_;
  constraint_matrix_(12, 5) =
      (foot_half_length_ + foot_half_width_) * mu_;

  constraint_matrix_(13, 0) = -mu_;
  constraint_matrix_(13, 1) = -mu_;
  constraint_matrix_(13, 2) = -1.0;
  constraint_matrix_(13, 3) = -foot_half_width_;
  constraint_matrix_(13, 4) = -foot_half_length_;
  constraint_matrix_(13, 5) =
      (foot_half_length_ + foot_half_width_) * mu_;

  constraint_matrix_(14, 0) = -mu_;
  constraint_matrix_(14, 1) = mu_;
  constraint_matrix_(14, 2) = -1.0;
  constraint_matrix_(14, 3) = -foot_half_width_;
  constraint_matrix_(14, 4) = foot_half_length_;
  constraint_matrix_(14, 5) =
      (foot_half_length_ + foot_half_width_) * mu_;

  constraint_matrix_(15, 0) = mu_;
  constraint_matrix_(15, 1) = -mu_;
  constraint_matrix_(15, 2) = -1.0;
  constraint_matrix_(15, 3) = foot_half_width_;
  constraint_matrix_(15, 4) = -foot_half_length_;
  constraint_matrix_(15, 5) =
      (foot_half_length_ + foot_half_width_) * mu_;

  constraint_matrix_(16, 0) = mu_;
  constraint_matrix_(16, 1) = mu_;
  constraint_matrix_(16, 2) = -1.0;
  constraint_matrix_(16, 3) = foot_half_width_;
  constraint_matrix_(16, 4) = foot_half_length_;
  constraint_matrix_(16, 5) =
      (foot_half_length_ + foot_half_width_) * mu_;

  constraint_matrix_(17, 5) = -1.0;
  constraint_vector_[17] = -rf_z_max_;
}

////////////////////////////////////////////////////////////////////////////////
void SurfaceContact::UpdateOpCommand() {
  const Eigen::Isometry3d link_iso = robot_->GetLinkIsometry(target_link_idx_);
  const Eigen::Matrix3d rot_w_b = link_iso.linear();

  const Eigen::Vector3d pos_err_w = des_pos_ - link_iso.translation();
  const Eigen::Vector3d vel_err_w =
      -robot_->GetLinkSpatialVel(target_link_idx_).tail<3>();
  const Eigen::Vector3d pos_err_b = rot_w_b.transpose() * pos_err_w;
  const Eigen::Vector3d vel_err_b = rot_w_b.transpose() * vel_err_w;
  op_cmd_.tail<3>() = kp_.tail<3>().cwiseProduct(pos_err_b) +
                      kd_.tail<3>().cwiseProduct(vel_err_b);

  Eigen::Quaterniond curr_quat(rot_w_b);
  curr_quat.normalize();
  Eigen::Quaterniond des_quat = des_quat_.normalized();
  if (des_quat.coeffs().dot(curr_quat.coeffs()) < 0.0) {
    curr_quat.coeffs() *= -1.0;
  }
  const Eigen::Quaterniond quat_err_b = curr_quat.inverse() * des_quat;
  const Eigen::AngleAxisd aa(quat_err_b);
  const Eigen::Vector3d ori_err_b = aa.axis() * aa.angle();

  const Eigen::Vector3d ang_vel_b =
      robot_->GetLinkBodySpatialVel(target_link_idx_).head<3>();
  op_cmd_.head<3>() = kp_.head<3>().cwiseProduct(ori_err_b) +
                      kd_.head<3>().cwiseProduct(-ang_vel_b);
}

////////////////////////////////////////////////////////////////////////////////
void SurfaceContact::SetParameters(const ContactConfig& config) {
  assert(config.kp.size() == 6 && config.kd.size() == 6 &&
         "SurfaceContact SetParameters requires size-6 gains");
  mu_ = config.mu;
  kp_ = config.kp;
  kd_ = config.kd;
  rf_z_max_ = config.max_fz;
}

////////////////////////////////////////////////////////////////////////////////
void SurfaceContact::SetParameters(const SurfaceContactConfig& config) {
  SetParameters(static_cast<const ContactConfig&>(config));
  foot_half_length_ = config.foot_half_length;
  foot_half_width_ = config.foot_half_width;
}

} // namespace wbc
