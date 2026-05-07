/**
 * @file wbc_core/wbc_trajectory/src/interpolation.cpp
 * @brief Doxygen documentation for interpolation module.
 */
#include "wbc_trajectory/interpolation.hpp"

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace util
{
double Smooth(double ini, double fin, double rat)
{
  if (rat < 0) {
    return ini;
  } else if (rat > 1) {
    return fin;
  } else {
    return ini + (fin - ini) * rat;
  }
}

double SmoothPos(
  double ini, double end, double moving_duration,
  double curr_time)
{
  if (moving_duration <= 0.0) {
    return end;
  }
  double ret;
  ret = ini + (end - ini) * 0.5 * (1 - cos(curr_time / moving_duration * M_PI));
  if (curr_time > moving_duration) {
    ret = end;
  }
  return ret;
}

double SmoothVel(
  double ini, double end, double moving_duration,
  double curr_time)
{
  if (moving_duration <= 0.0) {
    return 0.0;
  }
  double ret;
  ret = (end - ini) * 0.5 * (M_PI / moving_duration) *
    sin(curr_time / moving_duration * M_PI);
  if (curr_time > moving_duration) {
    ret = 0.0;
  }
  return ret;
}
double SmoothAcc(
  double ini, double end, double moving_duration,
  double curr_time)
{
  if (moving_duration <= 0.0) {
    return 0.0;
  }
  double ret;
  ret = (end - ini) * 0.5 * (M_PI / moving_duration) *
    (M_PI / moving_duration) * cos(curr_time / moving_duration * M_PI);
  if (curr_time > moving_duration) {
    ret = 0.0;
  }
  return ret;
}
void SinusoidTrajectory(
  const Eigen::VectorXd & mid_point,
  const Eigen::VectorXd & amp, const Eigen::VectorXd & freq,
  double eval_time, Eigen::VectorXd & p,
  Eigen::VectorXd & v, Eigen::VectorXd & a,
  double smoothing_dur)
{
  assert(amp.size() == freq.size());
  assert(mid_point.size() == amp.size());

  const int n = amp.size();
  p = Eigen::VectorXd::Zero(n);
  v = Eigen::VectorXd::Zero(n);
  a = Eigen::VectorXd::Zero(n);
  for (int i = 0; i < n; ++i) {
    p[i] = amp[i] * sin(2 * M_PI * freq[i] * (eval_time)) + mid_point[i];
    v[i] = amp[i] * 2 * M_PI * freq[i] * cos(2 * M_PI * freq[i] * (eval_time));
    a[i] = -amp[i] * 2 * M_PI * freq[i] * 2 * M_PI * freq[i] *
      sin(2 * M_PI * freq[i] * (eval_time));
  }
  if (eval_time < smoothing_dur) {
    double s = SmoothPos(0., 1., smoothing_dur, eval_time);
    for (int i = 0; i < n; ++i) {
      p[i] = (1 - s) * mid_point[i] + s * p[i];
      v[i] *= s;
      a[i] *= s;
    }
  }
}

void SinusoidTrajectory(
  const Eigen::VectorXd & amp, const Eigen::VectorXd & freq,
  double eval_time, Eigen::VectorXd & p,
  Eigen::VectorXd & v, Eigen::VectorXd & a,
  double smoothing_dur)
{
  assert(amp.size() == freq.size());

  const int n = amp.size();
  p = Eigen::VectorXd::Zero(n);
  v = Eigen::VectorXd::Zero(n);
  a = Eigen::VectorXd::Zero(n);
  for (int i = 0; i < n; ++i) {
    p[i] = amp[i] * sin(2 * M_PI * freq[i] * (eval_time));
    v[i] = amp[i] * 2 * M_PI * freq[i] * cos(2 * M_PI * freq[i] * (eval_time));
    a[i] = -amp[i] * 2 * M_PI * freq[i] * 2 * M_PI * freq[i] *
      sin(2 * M_PI * freq[i] * (eval_time));
  }
  if (eval_time < smoothing_dur) {
    double s = SmoothPos(0., 1., smoothing_dur, eval_time);
    for (int i = 0; i < n; ++i) {
      p[i] *= s;
      v[i] *= s;
      a[i] *= s;
    }
  }
}
// Constructor
HermiteCurve::HermiteCurve()
{
  p1 = 0;
  v1 = 0;
  p2 = 0;
  v2 = 0;
  t_dur = 0.5;
  s_ = 0;
}

HermiteCurve::HermiteCurve(
  const double & start_pos, const double & start_vel,
  const double & end_pos, const double & end_vel,
  const double & duration)
: p1(start_pos), v1(start_vel), p2(end_pos), v2(end_vel), t_dur(duration)
{
  s_ = 0;
  if (t_dur < 1e-3) {
    t_dur = 1e-3;
  }
}

// Destructor
HermiteCurve::~HermiteCurve() {}

// Cubic Hermite Spline:
// From https://en.wikipedia.org/wiki/Cubic_Hermite_spline#Unit_interval_(0,_1)
// p(s) = (2s^3 - 3s^2 + 1)*p1 + (-2*s^3 + 3*s^2)*p2 + (s^3 - 2s^2 + s)*v1 +
// (s^3 - s^2)*v2 where 0 <= s <= 1.
double HermiteCurve::Evaluate(const double & t_in)
{
  s_ = this->Clamp(t_in / t_dur);
  return p1 * (2 * std::pow(s_, 3) - 3 * std::pow(s_, 2) + 1) +
         p2 * (-2 * std::pow(s_, 3) + 3 * std::pow(s_, 2)) +
         v1 * t_dur * (std::pow(s_, 3) - 2 * std::pow(s_, 2) + s_) +
         v2 * t_dur * (std::pow(s_, 3) - std::pow(s_, 2));
}

double HermiteCurve::EvaluateFirstDerivative(const double & t_in)
{
  s_ = this->Clamp(t_in / t_dur);
  return (p1 * (6 * std::pow(s_, 2) - 6 * s_) +
         p2 * (-6 * std::pow(s_, 2) + 6 * s_) +
         v1 * t_dur * (3 * std::pow(s_, 2) - 4 * s_ + 1) +
         v2 * t_dur * (3 * std::pow(s_, 2) - 2 * s_)) /
         t_dur;
}

double HermiteCurve::EvaluateSecondDerivative(const double & t_in)
{
  s_ = this->Clamp(t_in / t_dur);
  return (p1 * (12 * s_ - 6) + p2 * (-12 * s_ + 6) + v1 * t_dur * (6 * s_ - 4) +
         v2 * t_dur * (6 * s_ - 2)) /
         t_dur / t_dur;
}

double HermiteCurve::Clamp(const double & s_in, double lo, double hi)
{
  if (s_in < lo) {
    return lo;
  } else if (s_in > hi) {
    return hi;
  } else {
    return s_in;
  }
}
// Constructor
HermiteCurveVec::HermiteCurveVec() {}
// Destructor
HermiteCurveVec::~HermiteCurveVec() {}

HermiteCurveVec::HermiteCurveVec(
  const Eigen::VectorXd & start_pos,
  const Eigen::VectorXd & start_vel,
  const Eigen::VectorXd & end_pos,
  const Eigen::VectorXd & end_vel,
  const double & duration)
{
  Initialize(start_pos, start_vel, end_pos, end_vel, duration);
}

void HermiteCurveVec::Initialize(
  const Eigen::VectorXd & start_pos,
  const Eigen::VectorXd & start_vel,
  const Eigen::VectorXd & end_pos,
  const Eigen::VectorXd & end_vel,
  const double & duration)
{
  p1 = start_pos;
  v1 = start_vel;
  p2 = end_pos;
  v2 = end_vel;
  t_dur = duration;

  const int n = start_pos.size();
  if (curves.size() != static_cast<size_t>(n)) {
    curves.resize(n);
    output = Eigen::VectorXd::Zero(n);
  }
  for (int i = 0; i < n; i++) {
    curves[i] = HermiteCurve(start_pos[i], start_vel[i], end_pos[i],
                              end_vel[i], t_dur);
  }
}

// Evaluation functions
Eigen::VectorXd HermiteCurveVec::Evaluate(const double & t_in)
{
  for (int i = 0; i < p1.size(); i++) {
    output[i] = curves[i].Evaluate(t_in);
  }
  return output;
}

Eigen::VectorXd HermiteCurveVec::EvaluateFirstDerivative(const double & t_in)
{
  for (int i = 0; i < p1.size(); i++) {
    output[i] = curves[i].EvaluateFirstDerivative(t_in);
  }
  return output;
}

Eigen::VectorXd HermiteCurveVec::EvaluateSecondDerivative(const double & t_in)
{
  for (int i = 0; i < p1.size(); i++) {
    output[i] = curves[i].EvaluateSecondDerivative(t_in);
  }
  return output;
}

HermiteQuaternionCurve::HermiteQuaternionCurve() {}

HermiteQuaternionCurve::HermiteQuaternionCurve(
  const Eigen::Quaterniond & quat_start,
  const Eigen::Vector3d & angular_velocity_start,
  const Eigen::Quaterniond & quat_end,
  const Eigen::Vector3d & angular_velocity_end, double duration)
{
  Initialize(quat_start, angular_velocity_start, quat_end, angular_velocity_end,
             duration);
}

void HermiteQuaternionCurve::Initialize(
  const Eigen::Quaterniond & quat_start,
  const Eigen::Vector3d & angular_velocity_start,
  const Eigen::Quaterniond & quat_end,
  const Eigen::Vector3d & angular_velocity_end, double duration)
{
  qa = quat_start;
  omega_a = angular_velocity_start;

  qb = quat_end;
  omega_b = angular_velocity_end;

  t_dur = duration;

  Initialize_data_structures();
}

HermiteQuaternionCurve::~HermiteQuaternionCurve() {}

void HermiteQuaternionCurve::Initialize_data_structures()
{
  //
  // q(t) = exp( theta(t) ) * qa : global frame
  // q(t) = qa * exp( theta(t) ) : local frame
  // where theta(t) is hermite cubic spline with
  // theta(0) = 0, theta(t_dur) = log(delq_ab)
  // dot_theta(0) = omega_a, dot_theta(1) = omega_b

  Eigen::AngleAxisd delq_ab = Eigen::AngleAxisd(qb * qa.inverse());
  // Eigen::AngleAxisd del_qab = qa.inverse()*qb;

  Eigen::VectorXd start_pos = Eigen::VectorXd::Zero(3);
  Eigen::VectorXd start_vel = omega_a;
  Eigen::VectorXd end_pos = delq_ab.axis() * delq_ab.angle();
  Eigen::VectorXd end_vel = omega_b;

  theta_ab.Initialize(start_pos, start_vel, end_pos, end_vel, t_dur);
}

void HermiteQuaternionCurve::Evaluate(
  const double & t_in,
  Eigen::Quaterniond & quat_out)
{
  Eigen::VectorXd delq_vec = theta_ab.Evaluate(t_in);

  if (delq_vec.norm() < 1e-6) {
    delq = Eigen::Quaterniond(1, 0, 0, 0);
  } else {
    delq = Eigen::AngleAxisd(delq_vec.norm(), delq_vec / delq_vec.norm());
  }
  // quat_out = q0 * delq; // local frame
  quat_out = delq * qa; // global frame
}

void HermiteQuaternionCurve::GetAngularVelocity(
  const double & t_in,
  Eigen::Vector3d & ang_vel_out)
{
  ang_vel_out = theta_ab.EvaluateFirstDerivative(t_in);
}

// For world frame
void HermiteQuaternionCurve::GetAngularAcceleration(
  const double & t_in, Eigen::Vector3d & ang_acc_out)
{
  ang_acc_out = theta_ab.EvaluateSecondDerivative(t_in);
}

void HermiteQuaternionCurve::PrintQuat(const Eigen::Quaterniond & quat)
{
  std::cout << quat.x() << " " << quat.y() << " " << quat.z() << " " << quat.w()
            << " " << std::endl;
}

//****************************************************************************
// NormalizedHermiteQuaternionCurve
//****************************************************************************
NormalizedHermiteQuaternionCurve::NormalizedHermiteQuaternionCurve() {}

NormalizedHermiteQuaternionCurve::NormalizedHermiteQuaternionCurve(
  const Eigen::Quaterniond & quat_start,
  const Eigen::Vector3d & angular_velocity_start,
  const Eigen::Quaterniond & quat_end,
  const Eigen::Vector3d & angular_velocity_end)
{
  Initialize(quat_start, angular_velocity_start, quat_end,
             angular_velocity_end);
}

void NormalizedHermiteQuaternionCurve::Initialize(
  const Eigen::Quaterniond & quat_start,
  const Eigen::Vector3d & angular_velocity_start,
  const Eigen::Quaterniond & quat_end,
  const Eigen::Vector3d & angular_velocity_end)
{
  qa = quat_start;
  omega_a = angular_velocity_start;

  qb = quat_end;
  omega_b = angular_velocity_end;

  s_ = 0.0;
  Initialize_data_structures();
}

void NormalizedHermiteQuaternionCurve::SetDesired(
  const Eigen::Quaterniond & quat_end,
  const Eigen::Vector3d & angular_velocity_end)
{
  qb = quat_end;
  omega_b = angular_velocity_end;
}

void NormalizedHermiteQuaternionCurve::SetInitial(
  const Eigen::Quaterniond & quat_start,
  const Eigen::Vector3d & angular_velocity_start)
{
  qa = quat_start;
  omega_a = angular_velocity_start;

  s_ = 0.0;
  Initialize_data_structures();
}

NormalizedHermiteQuaternionCurve::~NormalizedHermiteQuaternionCurve() {}

void NormalizedHermiteQuaternionCurve::Initialize_data_structures()
{
  q0 = qa;

  if (omega_a.norm() < 1e-6) {
    q1 = qa * Eigen::Quaterniond(1, 0, 0, 0);
  } else {
    q1 = qa * Eigen::Quaterniond(Eigen::AngleAxisd(
                  omega_a.norm() / 3.0,
                  omega_a / omega_a.norm())); // q1 = qa*exp(wa/3.0)
  }

  if (omega_b.norm() < 1e-6) {
    q2 = qb * Eigen::Quaterniond(1, 0, 0, 0);
  } else {
    q2 = qb * Eigen::Quaterniond(Eigen::AngleAxisd(
                  omega_b.norm() / 3.0,
                  -omega_b / omega_b.norm())); // q2 = qb*exp(wb/3.0)^-1
  }

  q3 = qb;

  // for world frame angular velocities, do: q_1*q_0.inverse(). for local frame
  // do: q_0.inverse()*q_1

  // Global Frame
  omega_1aa = q1 * q0.inverse();
  omega_2aa = q2 * q1.inverse();
  omega_3aa = q3 * q2.inverse();

  // Local Frame:
  // omega_1aa = q0.inverse()*q1;
  // omega_2aa = q1.inverse()*q2;
  // omega_3aa = q2.inverse()*q3;

  omega_1 = omega_1aa.axis() * omega_1aa.angle();
  omega_2 = omega_2aa.axis() * omega_2aa.angle();
  omega_3 = omega_3aa.axis() * omega_3aa.angle();
}

void NormalizedHermiteQuaternionCurve::ComputeBasis(const double & s_in)
{
  s_ = this->Clamp(s_in);
  b1 = 1 - std::pow((1 - s_), 3);
  b2 = 3 * std::pow(s_, 2) - 2 * std::pow((s_), 3);
  b3 = std::pow(s_, 3);

  bdot1 = 3 * std::pow((1 - s_), 2);
  bdot2 = 6 * s_ - 6 * std::pow((s_), 2);
  bdot3 = 3 * std::pow((s_), 2);

  bddot1 = -6 * (1 - s_);
  bddot2 = 6 - 12 * s_;
  bddot3 = 6 * s_;
}

Eigen::Quaterniond NormalizedHermiteQuaternionCurve::GetOrientation(const double & s_in)
{
  s_ = this->Clamp(s_in);
  ComputeBasis(s_);

  qtmp1 = Eigen::AngleAxisd(omega_1aa.angle() * b1, omega_1aa.axis());
  qtmp2 = Eigen::AngleAxisd(omega_2aa.angle() * b2, omega_2aa.axis());
  qtmp3 = Eigen::AngleAxisd(omega_3aa.angle() * b3, omega_3aa.axis());

  // quat_out = q0*qtmp1*qtmp2*qtmp3; // local frame
  Eigen::Quaterniond quat_out = qtmp3 * qtmp2 * qtmp1 * q0; // global frame
  return quat_out;
}

Eigen::Vector3d
NormalizedHermiteQuaternionCurve::GetAngularVelocity(const double & s_in)
{
  s_ = this->Clamp(s_in);
  ComputeBasis(s_);
  Eigen::Vector3d ang_vel_out =
    omega_1 * bdot1 + omega_2 * bdot2 + omega_3 * bdot3;
  return ang_vel_out;
}

// For world frame
Eigen::Vector3d
NormalizedHermiteQuaternionCurve::GetAngularAcceleration(const double & s_in)
{
  s_ = this->Clamp(s_in);
  ComputeBasis(s_);
  Eigen::Vector3d ang_acc_out =
    omega_1 * bddot1 + omega_2 * bddot2 + omega_3 * bddot3;
  return ang_acc_out;
}

double NormalizedHermiteQuaternionCurve::Clamp(
  const double & s_in, double lo,
  double hi)
{
  if (s_in < lo) {
    return lo;
  } else if (s_in > hi) {
    return hi;
  } else {
    return s_in;
  }
}

MinJerkCurve::MinJerkCurve() {Initialization();}

MinJerkCurve::MinJerkCurve(
  const Eigen::Vector3d & init,
  const Eigen::Vector3d & end, const double time_start,
  const double time_end)
{
  Initialization();
  SetParams(init, end, time_start, time_end);
}

// Destructor
MinJerkCurve::~MinJerkCurve() {}

void MinJerkCurve::Initialization()
{
  for (int i = 0; i < 6; ++i) b_[i] = 0.0;
  inv_T_ = 1.0;
  init_cond.setZero();
  end_cond.setZero();
  to = 0.0;
  tf = 1.0;
}

void MinJerkCurve::SetParams(
  const Eigen::Vector3d & init,
  const Eigen::Vector3d & end,
  const double time_start, const double time_end)
{
  init_cond = init;
  end_cond  = end;
  to = time_start;
  tf = time_end;

  const double T  = tf - to;
  inv_T_ = (T > 1e-9) ? 1.0 / T : 0.0;
  const double T2 = T * T;

  // Tau-domain coefficients: p(tau) = b0 + b1*tau + ... + b5*tau^5
  // where tau = (t - to) / T and dp/dt = dp/dtau * inv_T_
  b_[0] = init_cond[0];
  b_[1] = init_cond[1] * T;
  b_[2] = init_cond[2] * T2 * 0.5;

  // Remainders to satisfy end boundary conditions at tau = 1
  const double D0 = end_cond[0] - b_[0] - b_[1] - b_[2];
  const double D1 = end_cond[1] * T  - b_[1] - 2.0 * b_[2];
  const double D2 = end_cond[2] * T2 - 2.0 * b_[2];

  // Closed-form inverse of [[1,1,1],[3,4,5],[6,12,20]] (det = 2):
  b_[3] =  10.0 * D0 - 4.0 * D1 + 0.5 * D2;
  b_[4] = -15.0 * D0 + 7.0 * D1 -       D2;
  b_[5] =   6.0 * D0 - 3.0 * D1 + 0.5 * D2;
}

void MinJerkCurve::GetPos(const double time, double & pos) const
{
  const double tau = std::min(std::max((time - to) * inv_T_, 0.0), 1.0);
  pos = b_[0] + tau * (b_[1] + tau * (b_[2] + tau * (b_[3] + tau * (b_[4] + tau * b_[5]))));
}
void MinJerkCurve::GetVel(const double time, double & vel) const
{
  if (time >= tf) { vel = end_cond[1];  return; }
  if (time <= to) { vel = init_cond[1]; return; }
  const double tau = (time - to) * inv_T_;
  // p'(tau) via Horner; multiply by inv_T_ to convert dtau/dt → dt/dt
  vel = (b_[1] + tau * (2.0 * b_[2] + tau * (3.0 * b_[3] +
         tau * (4.0 * b_[4] + tau * 5.0 * b_[5])))) * inv_T_;
}
void MinJerkCurve::GetAcc(const double time, double & acc) const
{
  if (time >= tf) { acc = end_cond[2];  return; }
  if (time <= to) { acc = init_cond[2]; return; }
  const double tau = (time - to) * inv_T_;
  // p''(tau) via Horner; multiply by inv_T_^2
  acc = (2.0 * b_[2] + tau * (6.0 * b_[3] +
         tau * (12.0 * b_[4] + tau * 20.0 * b_[5]))) * inv_T_ * inv_T_;
}

MinJerkCurveVec::MinJerkCurveVec() {}

MinJerkCurveVec::MinJerkCurveVec(
  const Eigen::VectorXd & start_pos,
  const Eigen::VectorXd & start_vel,
  const Eigen::VectorXd & start_acc,
  const Eigen::VectorXd & end_pos,
  const Eigen::VectorXd & end_vel,
  const Eigen::VectorXd & end_acc,
  const double duration)
: Ts_(duration), p1_(start_pos), v1_(start_vel), a1_(start_acc),
  p2_(end_pos), v2_(end_vel), a2_(end_acc)
{
  // Create N minjerk curves_ with the specified boundary conditions
  for (int i = 0; i < start_pos.size(); i++) {
    curves_.push_back(MinJerkCurve(Eigen::Vector3d(p1_[i], v1_[i], a1_[i]),
                                   Eigen::Vector3d(p2_[i], v2_[i], a2_[i]), 0.0,
                                   Ts_));
  }
  pos_out_.setZero(start_pos.size());
  vel_out_.setZero(start_pos.size());
  acc_out_.setZero(start_pos.size());
}

// Destructor
MinJerkCurveVec::~MinJerkCurveVec() {}

void MinJerkCurveVec::Initialize(
  const Eigen::VectorXd & start_pos,
  const Eigen::VectorXd & start_vel,
  const Eigen::VectorXd & start_acc,
  const Eigen::VectorXd & end_pos,
  const Eigen::VectorXd & end_vel,
  const Eigen::VectorXd & end_acc,
  const double duration)
{
  Ts_ = duration;
  p1_ = start_pos;
  v1_ = start_vel;
  a1_ = start_acc;
  p2_ = end_pos;
  v2_ = end_vel;
  a2_ = end_acc;

  const int n = start_pos.size();
  if (static_cast<int>(curves_.size()) != n) curves_.resize(n);
  for (int i = 0; i < n; i++) {
    curves_[i].SetParams(Eigen::Vector3d(p1_[i], v1_[i], a1_[i]),
                         Eigen::Vector3d(p2_[i], v2_[i], a2_[i]), 0.0, Ts_);
  }
  pos_out_.setZero(n);
  vel_out_.setZero(n);
  acc_out_.setZero(n);
}

// Evaluation functions — return const& to pre-allocated buffers; no heap alloc.
const Eigen::VectorXd& MinJerkCurveVec::Evaluate(const double t_in)
{
  for (int i = 0; i < static_cast<int>(curves_.size()); i++) {
    curves_[i].GetPos(t_in, pos_out_[i]);
  }
  return pos_out_;
}

const Eigen::VectorXd& MinJerkCurveVec::EvaluateFirstDerivative(const double t_in)
{
  for (int i = 0; i < static_cast<int>(curves_.size()); i++) {
    curves_[i].GetVel(t_in, vel_out_[i]);
  }
  return vel_out_;
}

const Eigen::VectorXd& MinJerkCurveVec::EvaluateSecondDerivative(const double t_in)
{
  for (int i = 0; i < static_cast<int>(curves_.size()); i++) {
    curves_[i].GetAcc(t_in, acc_out_[i]);
  }
  return acc_out_;
}

} // namespace util
