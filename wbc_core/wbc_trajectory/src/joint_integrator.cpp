/**
 * @file wbc_core/wbc_trajectory/src/joint_integrator.cpp
 * @brief Doxygen documentation for joint_integrator module.
 */
#include "wbc_trajectory/joint_integrator.hpp"

#include <iostream>

namespace util {

JointIntegrator::JointIntegrator(int num_joints, double dt,
                                 const Eigen::VectorXd& pos_min,
                                 const Eigen::VectorXd& pos_max,
                                 const Eigen::VectorXd& vel_min,
                                 const Eigen::VectorXd& vel_max)
    : num_joints_(num_joints),
      dt_(dt),
      pos_min_(pos_min),
      pos_max_(pos_max),
      vel_min_(vel_min),
      vel_max_(vel_max),
      alpha_pos_(0.0),
      alpha_vel_(0.0),
      pos_max_error_vec_(Eigen::VectorXd::Zero(num_joints)),
      jpos_(Eigen::VectorXd::Zero(num_joints)),
      jvel_(Eigen::VectorXd::Zero(num_joints)),
      is_initialized_(false) {}

void JointIntegrator::SetCutoffFrequency(double pos_cutoff_freq,
                                         double vel_cutoff_freq) {
  alpha_pos_ = GetAlphaFromFrequency(pos_cutoff_freq, dt_);
  alpha_vel_ = GetAlphaFromFrequency(vel_cutoff_freq, dt_);
}

double JointIntegrator::GetAlphaFromFrequency(double hz, double dt) {
  double omega = 2.0 * M_PI * hz;
  double alpha = (omega * dt) / (1.0 + (omega * dt));
  return std::clamp(alpha, 0.0, 1.0);
}

void JointIntegrator::SetMaxPositionError(double pos_max_error) {
  pos_max_error_vec_ = pos_max_error * Eigen::VectorXd::Ones(num_joints_);
}

void JointIntegrator::Initialize(const Eigen::VectorXd& init_jpos,
                                 const Eigen::VectorXd& init_jvel) {
  jpos_ = init_jpos;
  jvel_ = init_jvel;
  is_initialized_ = true;
}

Eigen::VectorXd JointIntegrator::ClampVector(const Eigen::VectorXd& v,
                                             const Eigen::VectorXd& lo,
                                             const Eigen::VectorXd& hi) {
  return v.cwiseMax(lo).cwiseMin(hi);
}

void JointIntegrator::Integrate(const Eigen::VectorXd& cmd_jacc,
                                const Eigen::VectorXd& curr_jpos,
                                const Eigen::VectorXd& /*curr_jvel*/,
                                Eigen::VectorXd& cmd_jpos,
                                Eigen::VectorXd& cmd_jvel) {
  if (!is_initialized_) {
    std::cerr << "[util::JointIntegrator] Not initialized. Call Initialize() first."
              << std::endl;
    return;
  }

  // Velocity integration: decay desired velocity toward zero, then add acceleration
  jvel_ = (1.0 - alpha_vel_) * jvel_;
  jvel_ += cmd_jacc * dt_;
  cmd_jvel = ClampVector(jvel_, vel_min_, vel_max_);
  jvel_ = cmd_jvel;

  // Position integration: decay desired position toward measured, then add velocity
  jpos_ = (1.0 - alpha_pos_) * jpos_ + alpha_pos_ * curr_jpos;
  jpos_ += jvel_ * dt_;
  // Clamp to maximum position error relative to current position
  jpos_ = ClampVector(jpos_, curr_jpos - pos_max_error_vec_,
                      curr_jpos + pos_max_error_vec_);
  cmd_jpos = ClampVector(jpos_, pos_min_, pos_max_);
  jpos_ = cmd_jpos;
}

} // namespace util
