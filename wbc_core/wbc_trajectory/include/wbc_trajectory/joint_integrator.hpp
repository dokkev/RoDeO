#pragma once

#include <Eigen/Dense>
#include <cmath>

namespace util {

/**
 * @brief Leaky joint integrator for converting accelerations to position/velocity
 * commands.
 *
 * Uses IHMC-style integration with exponential decay toward measured state.
 * Ported from rpc_source JointIntegrator.
 *
 * Usage: construct → SetCutoffFrequency → SetMaxPositionError → Initialize →
 * Integrate (each cycle).
 */
class JointIntegrator {
public:
  JointIntegrator(int num_joints, double dt, const Eigen::VectorXd& pos_min,
                  const Eigen::VectorXd& pos_max, const Eigen::VectorXd& vel_min,
                  const Eigen::VectorXd& vel_max);
  ~JointIntegrator() = default;

  void SetCutoffFrequency(double pos_cutoff_freq, double vel_cutoff_freq);
  void SetMaxPositionError(double pos_max_error);

  void Initialize(const Eigen::VectorXd& init_jpos, const Eigen::VectorXd& init_jvel);

  void Integrate(const Eigen::VectorXd& cmd_jacc, const Eigen::VectorXd& curr_jpos,
                 const Eigen::VectorXd& curr_jvel, Eigen::VectorXd& cmd_jpos,
                 Eigen::VectorXd& cmd_jvel);

  bool IsInitialized() const { return b_initialized_; }

private:
  static double GetAlphaFromFrequency(double hz, double dt);
  static Eigen::VectorXd ClampVector(const Eigen::VectorXd& v,
                                     const Eigen::VectorXd& lo,
                                     const Eigen::VectorXd& hi);

  int num_joints_;
  double dt_;
  Eigen::VectorXd pos_min_, pos_max_;
  Eigen::VectorXd vel_min_, vel_max_;

  double alpha_pos_;
  double alpha_vel_;
  Eigen::VectorXd pos_max_error_vec_;

  Eigen::VectorXd jpos_;
  Eigen::VectorXd jvel_;
  bool b_initialized_;
};

} // namespace util
