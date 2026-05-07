/**
 * @file wbc_core/wbc_formulation/src/contact_constraint.cpp
 * @brief Doxygen documentation for contact_constraint module.
 */
#include "wbc_formulation/basic_contact.hpp"

#include <iostream>

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
    : Contact(robot, 3, target_link_idx, mu),
      full_jac_scratch_(Eigen::MatrixXd::Zero(6, robot->NumQdot())) {
  constraint_matrix_ = Eigen::MatrixXd::Zero(6, dim_);
  constraint_vector_ = Eigen::VectorXd::Zero(6);
}

////////////////////////////////////////////////////////////////////////////////
void PointContact::UpdateJacobian() {
  // FillLinkBodyJacobian writes into the pre-allocated scratch in-place,
  // avoiding the temporary 6xN heap allocation from GetLinkBodyJacobian().
  robot_->FillLinkBodyJacobian(target_link_idx_, full_jac_scratch_);
  jacobian_.noalias() = full_jac_scratch_.bottomRows(dim_);  // rows 3-5: linear part
}

////////////////////////////////////////////////////////////////////////////////
void PointContact::UpdateJacobianDotQdot() {
  jacobian_dot_q_dot_ =
      robot_->GetLinkLocalJacobianDotQdot(target_link_idx_).tail(dim_);
}

////////////////////////////////////////////////////////////////////////////////
void PointContact::UpdateConeConstraint() {
  if (!cone_dirty_) return;
  cone_dirty_ = false;

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
  if (static_cast<int>(config.kp.size()) != dim_ ||
      static_cast<int>(config.kd.size()) != dim_) {
    std::cerr << "[PointContact] SetParameters: invalid gain sizes (expected "
              << dim_ << "), skipping.\n";
    return;
  }
  mu_ = config.mu;
  kp_ = config.kp;
  kd_ = config.kd;
  rf_z_max_ = config.max_fz;
  cone_dirty_ = true;
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
}

////////////////////////////////////////////////////////////////////////////////
void SurfaceContact::UpdateJacobian() {
  robot_->FillLinkBodyJacobian(target_link_idx_, jacobian_);
}

////////////////////////////////////////////////////////////////////////////////
void SurfaceContact::UpdateJacobianDotQdot() {
  jacobian_dot_q_dot_ = robot_->GetLinkLocalJacobianDotQdot(target_link_idx_);
}

////////////////////////////////////////////////////////////////////////////////
void SurfaceContact::UpdateConeConstraint() {
  if (!cone_dirty_) return;
  cone_dirty_ = false;

  // Constraint format: constraint_matrix_ * rf >= constraint_vector_
  // rf = [tau_x, tau_y, tau_z, f_x, f_y, f_z] in body frame.
  // 18 rows encode:
  //   Row   0   : normal force lower bound        (f_z >= 0)
  //   Rows 1-4  : linearized 4-sided friction cone (|f_x|, |f_y| <= mu*f_z)
  //   Rows 5-8  : COP within footprint             (|tau_x| <= w*f_z, |tau_y| <= l*f_z)
  //   Rows 9-16 : 8-sided tipping + torsion pyramid
  //   Row  17   : normal force upper bound         (f_z <= rf_z_max_)

  constraint_matrix_.setZero();
  constraint_vector_.setZero();

  // Row 0: f_z >= 0
  constraint_matrix_(0, 5) = 1.0;

  // Rows 1-4: linearized friction cone  (|f_x| <= mu*f_z, |f_y| <= mu*f_z)
  constraint_matrix_(1, 3) = 1.0;   constraint_matrix_(1, 5) = mu_;   // f_x + mu*f_z >= 0
  constraint_matrix_(2, 3) = -1.0;  constraint_matrix_(2, 5) = mu_;   // -f_x + mu*f_z >= 0
  constraint_matrix_(3, 4) = 1.0;   constraint_matrix_(3, 5) = mu_;   // f_y + mu*f_z >= 0
  constraint_matrix_(4, 4) = -1.0;  constraint_matrix_(4, 5) = mu_;   // -f_y + mu*f_z >= 0

  // Rows 5-8: COP within foot support polygon (ZMP constraint)
  // tau_x = f_z * y_cop  =>  |y_cop| <= foot_half_width_
  constraint_matrix_(5, 0) = 1.0;   constraint_matrix_(5, 5) = foot_half_width_;
  constraint_matrix_(6, 0) = -1.0;  constraint_matrix_(6, 5) = foot_half_width_;
  // tau_y = -f_z * x_cop  =>  |x_cop| <= foot_half_length_
  constraint_matrix_(7, 1) = 1.0;   constraint_matrix_(7, 5) = foot_half_length_;
  constraint_matrix_(8, 1) = -1.0;  constraint_matrix_(8, 5) = foot_half_length_;

  // Rows 9-16: 8-sided combined tipping and torsional friction pyramid.
  // Each row encodes (+/-mu)*tau_x + (+/-mu)*tau_y + (+/-)tau_z
  //   + foot_half_width*f_x + foot_half_length*f_y + (l+w)*mu*f_z >= 0
  const double lw_mu = (foot_half_length_ + foot_half_width_) * mu_;

  constraint_matrix_(9,  0) = -mu_; constraint_matrix_(9,  1) = -mu_; constraint_matrix_(9,  2) =  1.0;
  constraint_matrix_(9,  3) =  foot_half_width_; constraint_matrix_(9,  4) =  foot_half_length_; constraint_matrix_(9,  5) = lw_mu;

  constraint_matrix_(10, 0) = -mu_; constraint_matrix_(10, 1) =  mu_; constraint_matrix_(10, 2) =  1.0;
  constraint_matrix_(10, 3) =  foot_half_width_; constraint_matrix_(10, 4) = -foot_half_length_; constraint_matrix_(10, 5) = lw_mu;

  constraint_matrix_(11, 0) =  mu_; constraint_matrix_(11, 1) = -mu_; constraint_matrix_(11, 2) =  1.0;
  constraint_matrix_(11, 3) = -foot_half_width_; constraint_matrix_(11, 4) =  foot_half_length_; constraint_matrix_(11, 5) = lw_mu;

  constraint_matrix_(12, 0) =  mu_; constraint_matrix_(12, 1) =  mu_; constraint_matrix_(12, 2) =  1.0;
  constraint_matrix_(12, 3) = -foot_half_width_; constraint_matrix_(12, 4) = -foot_half_length_; constraint_matrix_(12, 5) = lw_mu;

  constraint_matrix_(13, 0) = -mu_; constraint_matrix_(13, 1) = -mu_; constraint_matrix_(13, 2) = -1.0;
  constraint_matrix_(13, 3) = -foot_half_width_; constraint_matrix_(13, 4) = -foot_half_length_; constraint_matrix_(13, 5) = lw_mu;

  constraint_matrix_(14, 0) = -mu_; constraint_matrix_(14, 1) =  mu_; constraint_matrix_(14, 2) = -1.0;
  constraint_matrix_(14, 3) = -foot_half_width_; constraint_matrix_(14, 4) =  foot_half_length_; constraint_matrix_(14, 5) = lw_mu;

  constraint_matrix_(15, 0) =  mu_; constraint_matrix_(15, 1) = -mu_; constraint_matrix_(15, 2) = -1.0;
  constraint_matrix_(15, 3) =  foot_half_width_; constraint_matrix_(15, 4) = -foot_half_length_; constraint_matrix_(15, 5) = lw_mu;

  constraint_matrix_(16, 0) =  mu_; constraint_matrix_(16, 1) =  mu_; constraint_matrix_(16, 2) = -1.0;
  constraint_matrix_(16, 3) =  foot_half_width_; constraint_matrix_(16, 4) =  foot_half_length_; constraint_matrix_(16, 5) = lw_mu;

  // Row 17: f_z <= rf_z_max_  (upper bound on normal force)
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
      robot_->GetLinkLocalSpatialVelocity(target_link_idx_).head<3>();
  op_cmd_.head<3>() = kp_.head<3>().cwiseProduct(ori_err_b) +
                      kd_.head<3>().cwiseProduct(-ang_vel_b);
}

////////////////////////////////////////////////////////////////////////////////
void SurfaceContact::SetParameters(const ContactConfig& config) {
  if (static_cast<int>(config.kp.size()) != dim_ ||
      static_cast<int>(config.kd.size()) != dim_) {
    std::cerr << "[SurfaceContact] SetParameters: invalid gain sizes (expected "
              << dim_ << "), skipping.\n";
    return;
  }
  mu_ = config.mu;
  kp_ = config.kp;
  kd_ = config.kd;
  rf_z_max_ = config.max_fz;
  cone_dirty_ = true;
}

////////////////////////////////////////////////////////////////////////////////
void SurfaceContact::SetParameters(const SurfaceContactConfig& config) {
  SetParameters(static_cast<const ContactConfig&>(config));
  foot_half_length_ = config.foot_half_length;
  foot_half_width_ = config.foot_half_width;
}

} // namespace wbc
