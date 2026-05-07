/**
 * @file wbc_core/wbc_trajectory/include/wbc_trajectory/filter.hpp
 * @brief Digital filter utilities ported from rpc_source/controller/filter.
 *
 * @details
 * Header-only, Eigen-only, namespace wbc.  All filters follow the same
 * call pattern:
 *
 *   filter.Update(raw_value);          // feed one sample
 *   const auto& smooth = filter.Value(); // read filtered output
 *   filter.Reset(...);                 // re-initialize state
 *
 * Scalar filters (`LowPassFilter`, `DerivativeLowPassFilter`, `MovingAverage`)
 * work on `double`.  Vector filters (`FirstOrderLpf`, `VelocityLpf`,
 * `ExponentialMovingAverage`) work on `Eigen::VectorXd` and return `const&`
 * to avoid copies.  `QuaternionLpf` works on `Eigen::Quaterniond` via SLERP.
 *
 * ## Mapping from rpc_source
 * | rpc_source class               | wbc class                 |
 * |--------------------------------|---------------------------|
 * | LowPassFilter                  | LowPassFilter             |
 * | DerivativeLowPassFilter        | DerivativeLowPassFilter   |
 * | SimpleMovingAverage            | MovingAverage             |
 * | FirstOrderLowPassFilter        | FirstOrderLpf             |
 * | LowPassVelocityFilter          | VelocityLpf               |
 * | ExponentialMovingAverageFilter | ExponentialMovingAverage  |
 * | (new)                          | QuaternionLpf             |
 *
 * Note: ButterWorthFilter (raw heap buffer, FIR convolution) is not ported —
 * `LowPassFilter` (2nd-order IIR) is preferred for all scalar LPF needs.
 */
#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Geometry>

namespace wbc {

// ---------------------------------------------------------------------------
// LowPassFilter — 2nd-order Butterworth scalar IIR
// ---------------------------------------------------------------------------
/**
 * @brief Second-order Butterworth low-pass filter for scalar signals.
 *
 * @details Bilinear-transform IIR. Ported from rpc_source `LowPassFilter`.
 * The `float den` precision bug in the original is fixed here (all `double`).
 *
 * @param w_c  Cutoff angular frequency (rad/s) = 2π × f_c
 * @param t_s  Sampling period (seconds)
 */
class LowPassFilter {
public:
  LowPassFilter() = default;
  LowPassFilter(double w_c, double t_s) { Initialize(w_c, t_s); }

  void Initialize(double w_c, double t_s) {
    const double den = 2500.0 * t_s * t_s * w_c * w_c
                     + 7071.0 * t_s * w_c + 10000.0;
    in1_  = 2500.0 * t_s * t_s * w_c * w_c / den;
    in2_  = 5000.0 * t_s * t_s * w_c * w_c / den;
    in3_  = in1_;
    out1_ = -(5000.0 * t_s * t_s * w_c * w_c - 20000.0) / den;
    out2_ = -(2500.0 * t_s * t_s * w_c * w_c - 7071.0 * t_s * w_c + 10000.0) / den;
    Reset();
  }

  void Reset() {
    in_prev_[0] = in_prev_[1] = 0.0;
    out_prev_[0] = out_prev_[1] = 0.0;
    value_ = 0.0;
  }

  double Update(double x) {
    value_ = in1_ * x          + in2_  * in_prev_[0]  + in3_  * in_prev_[1]
           + out1_ * out_prev_[0] + out2_ * out_prev_[1];
    in_prev_[1]  = in_prev_[0];  in_prev_[0]  = x;
    out_prev_[1] = out_prev_[0]; out_prev_[0] = value_;
    return value_;
  }

  double Value() const { return value_; }

private:
  double in1_{1.0}, in2_{0.0}, in3_{0.0};
  double out1_{0.0}, out2_{0.0};
  double in_prev_[2]{0.0, 0.0};
  double out_prev_[2]{0.0, 0.0};
  double value_{0.0};
};

// ---------------------------------------------------------------------------
// DerivativeLowPassFilter — 2nd-order scalar IIR with differentiation
// ---------------------------------------------------------------------------
/**
 * @brief Scalar filter that differentiates the input and applies 2nd-order LPF.
 *
 * @details Useful for estimating a filtered derivative (e.g. velocity from
 * position) from a noisy scalar signal. Ported from rpc_source
 * `DerivativeLowPassFilter`.
 *
 * @param w_c  Cutoff angular frequency (rad/s)
 * @param t_s  Sampling period (seconds)
 */
class DerivativeLowPassFilter {
public:
  DerivativeLowPassFilter() = default;
  DerivativeLowPassFilter(double w_c, double t_s) { Initialize(w_c, t_s); }

  void Initialize(double w_c, double t_s) {
    constexpr double a   = 1.4142;
    const double     den = 4.0 + 2.0 * a * w_c * t_s + t_s * t_s * w_c * w_c;
    in1_  =  2.0 * t_s * w_c * w_c / den;
    in2_  =  0.0;
    in3_  = -2.0 * t_s * w_c * w_c / den;
    out1_ = -(-8.0 + 2.0 * t_s * t_s * w_c * w_c) / den;
    out2_ = -(4.0 - 2.0 * a * w_c * t_s + t_s * t_s * w_c * w_c) / den;
    Reset();
  }

  void Reset() {
    in_prev_[0] = in_prev_[1] = 0.0;
    out_prev_[0] = out_prev_[1] = 0.0;
    value_ = 0.0;
  }

  double Update(double x) {
    value_ = in1_ * x          + in2_  * in_prev_[0]  + in3_  * in_prev_[1]
           + out1_ * out_prev_[0] + out2_ * out_prev_[1];
    in_prev_[1]  = in_prev_[0];  in_prev_[0]  = x;
    out_prev_[1] = out_prev_[0]; out_prev_[0] = value_;
    return value_;
  }

  double Value() const { return value_; }

private:
  double in1_{0.0}, in2_{0.0}, in3_{0.0};
  double out1_{0.0}, out2_{0.0};
  double in_prev_[2]{0.0, 0.0};
  double out_prev_[2]{0.0, 0.0};
  double value_{0.0};
};

// ---------------------------------------------------------------------------
// MovingAverage — scalar circular-buffer moving average
// ---------------------------------------------------------------------------
/**
 * @brief Scalar simple moving average over a fixed window.
 *
 * @details O(1) per sample via running sum. Ported from rpc_source
 * `SimpleMovingAverage`.
 */
class MovingAverage {
public:
  MovingAverage() = default;
  explicit MovingAverage(int window) { Initialize(window); }

  void Initialize(int window) {
    num_data_ = std::max(1, window);
    buffer_.assign(num_data_, 0.0);
    idx_ = 0;
    sum_ = 0.0;
  }

  void Reset() {
    std::fill(buffer_.begin(), buffer_.end(), 0.0);
    idx_ = 0;
    sum_ = 0.0;
  }

  double Update(double x) {
    sum_ -= buffer_[idx_];
    sum_ += x;
    buffer_[idx_] = x;
    idx_ = (idx_ + 1) % num_data_;
    value_ = sum_ / num_data_;
    return value_;
  }

  double Value() const { return value_; }

private:
  std::vector<double> buffer_;
  int    num_data_{1};
  int    idx_{0};
  double sum_{0.0};
  double value_{0.0};
};

// ---------------------------------------------------------------------------
// FirstOrderLpf — 1st-order IIR for Eigen::VectorXd
// ---------------------------------------------------------------------------
/**
 * @brief First-order IIR low-pass filter for vector signals.
 *
 * @details
 *   alpha = dt / max(cutoff_period, 2*dt)   (Nyquist-Shannon safeguard)
 *   y[n]  = alpha * x[n] + (1 - alpha) * y[n-1]
 *
 * Ported from rpc_source `FirstOrderLowPassFilter`.
 * Returns `const&` to internal buffer — avoids copies on the hot path.
 *
 * @param dt             Sampling period (seconds)
 * @param cutoff_period  Filter cutoff period (seconds) = 1 / f_c
 * @param dim            Signal dimension
 */
class FirstOrderLpf {
public:
  FirstOrderLpf() = default;
  FirstOrderLpf(double dt, double cutoff_period, int dim) {
    Initialize(dt, cutoff_period, dim);
  }

  void Initialize(double dt, double cutoff_period, int dim) {
    dt_    = dt;
    alpha_ = dt / std::max(cutoff_period, 2.0 * dt);
    value_.setZero(dim);
  }

  /** @brief Reset filter state to val. */
  void Reset(const Eigen::VectorXd& val) { value_ = val; }
  /** @brief Reset filter state to zero. */
  void Reset() { value_.setZero(); }

  const Eigen::VectorXd& Update(const Eigen::VectorXd& x) {
    value_ = alpha_ * x + (1.0 - alpha_) * value_;
    return value_;
  }

  const Eigen::VectorXd& Value() const { return value_; }
  int    Dim()   const { return static_cast<int>(value_.size()); }
  double Alpha() const { return alpha_; }

private:
  double dt_{0.001};
  double alpha_{1.0};
  Eigen::VectorXd value_;
};

// ---------------------------------------------------------------------------
// VelocityLpf — 1st-order IIR velocity estimator from position
// ---------------------------------------------------------------------------
/**
 * @brief Estimates velocity from successive position samples with IIR LPF.
 *
 * @details
 *   disc_vel = (new_pos - prev_pos) / dt
 *   vel[n]   = alpha * disc_vel + (1 - alpha) * vel[n-1]
 *
 * Ported from rpc_source `LowPassVelocityFilter`.
 */
class VelocityLpf {
public:
  VelocityLpf() = default;
  VelocityLpf(double dt, double cutoff_period, int dim) {
    Initialize(dt, cutoff_period, dim);
  }

  void Initialize(double dt, double cutoff_period, int dim) {
    dt_    = dt;
    alpha_ = dt / std::max(cutoff_period, 2.0 * dt);
    pos_.setZero(dim);
    vel_.setZero(dim);
  }

  /** @brief Reset position to pos, zero velocity. */
  void Reset(const Eigen::VectorXd& pos) {
    pos_ = pos;
    vel_.setZero();
  }

  /** @brief Feed new position sample; returns filtered velocity. */
  const Eigen::VectorXd& Update(const Eigen::VectorXd& new_pos) {
    const Eigen::VectorXd disc_vel = (new_pos - pos_) / dt_;
    vel_ = alpha_ * disc_vel + (1.0 - alpha_) * vel_;
    pos_ = new_pos;
    return vel_;
  }

  const Eigen::VectorXd& Velocity() const { return vel_; }
  const Eigen::VectorXd& Position() const { return pos_; }

private:
  double dt_{0.001};
  double alpha_{1.0};
  Eigen::VectorXd pos_;
  Eigen::VectorXd vel_;
};

// ---------------------------------------------------------------------------
// ExponentialMovingAverage — vector EMA with exact alpha = 1 - exp(-dt/T)
// ---------------------------------------------------------------------------
/**
 * @brief Exponential moving average for vector signals.
 *
 * @details Uses `alpha = 1 - exp(-dt / T)` (more accurate than linear
 * `dt/T` for large time steps). Ported from rpc_source
 * `ExponentialMovingAverageFilter` (clipping removed — clamp externally if needed).
 *
 *   average[n] = average[n-1] + alpha * (x[n] - average[n-1])
 *
 * @param dt            Sampling period (seconds)
 * @param time_constant Filter time constant T (seconds); clamped to 2*dt
 */
class ExponentialMovingAverage {
public:
  ExponentialMovingAverage() = default;
  ExponentialMovingAverage(double dt, double time_constant, int dim,
                           const Eigen::VectorXd& init_val = Eigen::VectorXd()) {
    Initialize(dt, time_constant, dim, init_val);
  }

  void Initialize(double dt, double time_constant, int dim,
                  const Eigen::VectorXd& init_val = Eigen::VectorXd()) {
    const double T = std::max(time_constant, 2.0 * dt);
    alpha_         = 1.0 - std::exp(-dt / T);
    if (init_val.size() == dim) {
      value_ = init_val;
    } else {
      value_.setZero(dim);
    }
  }

  void Reset(const Eigen::VectorXd& val) { value_ = val; }
  void Reset() { value_.setZero(); }

  const Eigen::VectorXd& Update(const Eigen::VectorXd& x) {
    value_ += alpha_ * (x - value_);
    return value_;
  }

  const Eigen::VectorXd& Value() const { return value_; }
  double Alpha() const { return alpha_; }

private:
  double alpha_{1.0};
  Eigen::VectorXd value_;
};

// ---------------------------------------------------------------------------
// QuaternionLpf — SLERP-based 1st-order LPF for quaternions
// ---------------------------------------------------------------------------
/**
 * @brief SLERP-based first-order low-pass filter for Eigen::Quaterniond.
 *
 * @details
 * Each Update() step spherically interpolates from the current filtered
 * quaternion toward the new input:
 *
 *   q[n] = q[n-1].slerp(alpha, q_new)
 *   alpha = dt / max(cutoff_period, 2*dt)
 *
 * Eigen's SLERP preserves unit-norm without explicit renormalization.
 * The input is normalized on arrival.
 *
 * @param dt             Sampling period (seconds)
 * @param cutoff_period  Filter cutoff period (seconds) = 1 / f_c
 */
class QuaternionLpf {
public:
  QuaternionLpf() = default;
  QuaternionLpf(double dt, double cutoff_period) { Initialize(dt, cutoff_period); }

  void Initialize(double dt, double cutoff_period) {
    alpha_  = dt / std::max(cutoff_period, 2.0 * dt);
    value_  = Eigen::Quaterniond::Identity();
  }

  void Reset(const Eigen::Quaterniond& q = Eigen::Quaterniond::Identity()) {
    value_ = q.normalized();
  }

  const Eigen::Quaterniond& Update(const Eigen::Quaterniond& q) {
    value_ = value_.slerp(alpha_, q.normalized());
    return value_;
  }

  const Eigen::Quaterniond& Value() const { return value_; }
  double Alpha() const { return alpha_; }

private:
  double alpha_{1.0};
  Eigen::Quaterniond value_{Eigen::Quaterniond::Identity()};
};

} // namespace wbc
