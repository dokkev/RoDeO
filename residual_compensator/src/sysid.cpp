/**
 * @file residual_compensator/src/sysid.cpp
 * @brief Offline SysID excitation helper for joint-space residual model fitting.
 */
#include "residual_compensator/sysid.hpp"

#include <algorithm>
#include <cmath>

namespace {
constexpr double kPi = 3.14159265358979323846;
constexpr double kTwoPi = 2.0 * kPi;
}  // namespace

namespace wbc {

void SysID::Setup(int num_active) {
  n_active_ = std::max(0, num_active);
  hold_q_.setZero(n_active_);
  cfg_.joint_idx = 0;

  active_ = false;
  finished_ = false;
  aborted_ = false;
  phase_ = SysIDPhase::IDLE;
  segment_idx_ = 0;
  reverse_ = false;

  start_time_sec_ = 0.0;
  phase_start_time_sec_ = 0.0;
  chirp_phase_ = 0.0;
  sweep_x_rel_ = 0.0;

  segment_q_start_ = 0.0;
  segment_q_goal_ = 0.0;

  last_target_q_ = 0.0;
  last_target_qdot_ = 0.0;
  last_target_qddot_ = 0.0;
  last_reason_.clear();
}

void SysID::Configure(const SysIDConfig& cfg) {
  cfg_ = cfg;

  cfg_.start_delay = std::max(0.0, cfg_.start_delay);
  cfg_.ramp_time = std::max(0.0, cfg_.ramp_time);
  cfg_.dwell_time = std::max(0.0, cfg_.dwell_time);
  cfg_.duration = std::max(0.0, cfg_.duration);
  cfg_.amplitude = std::abs(cfg_.amplitude);
  cfg_.cruise_vel = std::abs(cfg_.cruise_vel);
  cfg_.frequency_hz = std::max(0.0, cfg_.frequency_hz);
  cfg_.chirp_f0_hz = std::max(0.0, cfg_.chirp_f0_hz);
  cfg_.chirp_f1_hz = std::max(cfg_.chirp_f0_hz, cfg_.chirp_f1_hz);
  cfg_.chirp_duration = std::max(0.0, cfg_.chirp_duration);

  cfg_.max_tracking_err = std::max(0.0, cfg_.max_tracking_err);
  cfg_.max_meas_vel = std::max(0.0, cfg_.max_meas_vel);
  cfg_.max_tau_ratio = std::max(0.0, cfg_.max_tau_ratio);
}

void SysID::Reset(const Eigen::Ref<const Eigen::VectorXd>& hold_q) {
  if (n_active_ <= 0 || hold_q.size() != n_active_) {
    return;
  }
  hold_q_ = hold_q;

  active_ = false;
  finished_ = false;
  aborted_ = false;
  phase_ = SysIDPhase::IDLE;
  segment_idx_ = 0;
  reverse_ = false;

  chirp_phase_ = cfg_.phase0;
  sweep_x_rel_ = 0.0;

  const int j = ClampedJointIdx();
  segment_q_start_ = hold_q_[j] + cfg_.offset;
  segment_q_goal_ = segment_q_start_;

  last_target_q_ = segment_q_start_;
  last_target_qdot_ = 0.0;
  last_target_qddot_ = 0.0;
  last_reason_.clear();
}

void SysID::Start(double start_time_sec) {
  if (!IsSetup()) {
    Abort("not setup");
    return;
  }

  finished_ = false;
  aborted_ = false;
  last_reason_.clear();

  active_ = cfg_.enabled && (cfg_.mode != SysIDMode::OFF);
  start_time_sec_ = start_time_sec;
  phase_start_time_sec_ = start_time_sec;
  segment_idx_ = 0;
  reverse_ = false;
  chirp_phase_ = cfg_.phase0;
  sweep_x_rel_ = 0.0;

  const int j = ClampedJointIdx();
  segment_q_start_ = hold_q_[j] + cfg_.offset;
  segment_q_goal_ = segment_q_start_;
  last_target_q_ = segment_q_start_;
  last_target_qdot_ = 0.0;
  last_target_qddot_ = 0.0;

  if (!active_) {
    phase_ = SysIDPhase::IDLE;
    return;
  }

  if (cfg_.mode == SysIDMode::GRAVITY_GRID && cfg_.gravity_offsets_rad.empty()) {
    Abort("gravity_offsets_rad is empty");
    return;
  }

  EnterPhase(SysIDPhase::START_DELAY, start_time_sec);
}

void SysID::Stop() {
  active_ = false;
  finished_ = false;
  aborted_ = false;
  phase_ = SysIDPhase::IDLE;
  last_reason_ = "stopped";
}

void SysID::Abort(const std::string& reason) {
  active_ = false;
  finished_ = false;
  aborted_ = true;
  phase_ = SysIDPhase::ABORT;
  last_reason_ = reason;
}

void SysID::Update(double time_sec, double dt_sec,
                   Eigen::Ref<Eigen::VectorXd> q_ref,
                   Eigen::Ref<Eigen::VectorXd> qdot_ref,
                   Eigen::Ref<Eigen::VectorXd> qddot_ref) {
  if (!IsSetup() || q_ref.size() != n_active_ || qdot_ref.size() != n_active_ ||
      qddot_ref.size() != n_active_) {
    return;
  }

  q_ref = hold_q_;
  qdot_ref.setZero();
  qddot_ref.setZero();

  const int j = ClampedJointIdx();
  const double q_hold = hold_q_[j] + cfg_.offset;
  last_target_q_ = q_hold;
  last_target_qdot_ = 0.0;
  last_target_qddot_ = 0.0;

  if (!active_ || aborted_ || finished_ || cfg_.mode == SysIDMode::OFF) {
    return;
  }

  const double elapsed = Elapsed(time_sec);
  if (phase_ == SysIDPhase::START_DELAY && elapsed >= cfg_.start_delay) {
    switch (cfg_.mode) {
      case SysIDMode::GRAVITY_GRID: {
        segment_idx_ = 0;
        segment_q_start_ = q_hold;
        segment_q_goal_ = q_hold + cfg_.gravity_offsets_rad[segment_idx_];
        EnterPhase(SysIDPhase::RAMP, time_sec);
      } break;
      case SysIDMode::FRICTION_SWEEP:
      case SysIDMode::SINE:
      case SysIDMode::CHIRP:
        EnterPhase(SysIDPhase::CRUISE, time_sec);
        break;
      case SysIDMode::OFF:
      default:
        MarkDone("mode off");
        return;
    }
  }

  switch (cfg_.mode) {
    case SysIDMode::GRAVITY_GRID:
      UpdateGravityGrid(time_sec, q_ref, qdot_ref, qddot_ref);
      break;
    case SysIDMode::FRICTION_SWEEP:
      UpdateFrictionSweep(time_sec, dt_sec, q_ref, qdot_ref, qddot_ref);
      break;
    case SysIDMode::SINE:
      UpdateSine(time_sec, q_ref, qdot_ref, qddot_ref);
      break;
    case SysIDMode::CHIRP:
      UpdateChirp(time_sec, dt_sec, q_ref, qdot_ref, qddot_ref);
      break;
    case SysIDMode::OFF:
    default:
      MarkDone("mode off");
      return;
  }

  last_target_q_ = q_ref[j];
  last_target_qdot_ = qdot_ref[j];
  last_target_qddot_ = qddot_ref[j];
}

bool SysID::CheckSafety(const Eigen::Ref<const Eigen::VectorXd>& q_meas,
                        const Eigen::Ref<const Eigen::VectorXd>& qdot_meas,
                        const Eigen::Ref<const Eigen::VectorXd>& tau_cmd,
                        const Eigen::Ref<const Eigen::MatrixXd>& tau_limits,
                        std::string* reason) const {
  if (!active_) {
    return true;
  }

  const int j = ClampedJointIdx();
  if (q_meas.size() != n_active_ || qdot_meas.size() != n_active_ ||
      tau_cmd.size() != n_active_ || tau_limits.rows() < n_active_ ||
      tau_limits.cols() < 2) {
    if (reason != nullptr) {
      *reason = "invalid signal dimensions";
    }
    return false;
  }

  if (!std::isfinite(q_meas[j]) || !std::isfinite(qdot_meas[j]) ||
      !std::isfinite(tau_cmd[j])) {
    if (reason != nullptr) {
      *reason = "non-finite signal";
    }
    return false;
  }

  const double q_err = std::abs(last_target_q_ - q_meas[j]);
  if (q_err > cfg_.max_tracking_err) {
    if (reason != nullptr) {
      *reason = "tracking error limit exceeded";
    }
    return false;
  }

  const double qdot_abs = std::abs(qdot_meas[j]);
  if (qdot_abs > cfg_.max_meas_vel) {
    if (reason != nullptr) {
      *reason = "measured velocity limit exceeded";
    }
    return false;
  }

  const double tau_limit =
      std::max(std::abs(tau_limits(j, 0)), std::abs(tau_limits(j, 1)));
  const double tau_allow = cfg_.max_tau_ratio * tau_limit;
  if (std::abs(tau_cmd[j]) > tau_allow) {
    if (reason != nullptr) {
      *reason = "torque ratio limit exceeded";
    }
    return false;
  }

  return true;
}

bool SysID::CheckSafetyAndAbort(const Eigen::Ref<const Eigen::VectorXd>& q_meas,
                                const Eigen::Ref<const Eigen::VectorXd>& qdot_meas,
                                const Eigen::Ref<const Eigen::VectorXd>& tau_cmd,
                                const Eigen::Ref<const Eigen::MatrixXd>& tau_limits,
                                std::string* reason) {
  std::string local_reason;
  if (CheckSafety(q_meas, qdot_meas, tau_cmd, tau_limits, &local_reason)) {
    if (reason != nullptr) {
      reason->clear();
    }
    return true;
  }

  Abort(local_reason);
  if (reason != nullptr) {
    *reason = local_reason;
  }
  return false;
}

SysIDRuntimeState SysID::GetRuntimeState(double time_sec) const {
  SysIDRuntimeState rt;
  rt.active = active_;
  rt.finished = finished_;
  rt.aborted = aborted_;
  rt.mode = cfg_.mode;
  rt.phase = phase_;
  rt.joint_idx = ClampedJointIdx();
  rt.segment_idx = segment_idx_;
  rt.reverse = reverse_;
  rt.elapsed = Elapsed(time_sec);
  rt.target_q = last_target_q_;
  rt.target_qdot = last_target_qdot_;
  rt.target_qddot = last_target_qddot_;
  return rt;
}

double SysID::Clamp01(double x) {
  return std::max(0.0, std::min(1.0, x));
}

double SysID::SmoothStep(double s) {
  const double c = Clamp01(s);
  return c * c * (3.0 - 2.0 * c);
}

double SysID::SmoothStepDot(double s) {
  const double c = Clamp01(s);
  return 6.0 * c * (1.0 - c);
}

int SysID::ClampedJointIdx() const {
  if (n_active_ <= 0) {
    return 0;
  }
  return std::max(0, std::min(cfg_.joint_idx, n_active_ - 1));
}

double SysID::Elapsed(double time_sec) const {
  return std::max(0.0, time_sec - start_time_sec_);
}

double SysID::EffectiveChirpDuration() const {
  if (cfg_.chirp_duration > 0.0) {
    return cfg_.chirp_duration;
  }
  return cfg_.duration;
}

void SysID::EnterPhase(SysIDPhase phase, double time_sec) {
  phase_ = phase;
  phase_start_time_sec_ = time_sec;
}

void SysID::MarkDone(const std::string& reason) {
  active_ = false;
  finished_ = true;
  aborted_ = false;
  phase_ = SysIDPhase::DONE;
  last_reason_ = reason;
}

void SysID::UpdateGravityGrid(double time_sec,
                              Eigen::Ref<Eigen::VectorXd> q_ref,
                              Eigen::Ref<Eigen::VectorXd> qdot_ref,
                              Eigen::Ref<Eigen::VectorXd> qddot_ref) {
  if (phase_ == SysIDPhase::START_DELAY) {
    return;
  }

  const int j = ClampedJointIdx();
  const double q_hold = hold_q_[j] + cfg_.offset;
  // Consume phase transitions with exact boundary times to avoid
  // sample-time-dependent drift (important for offline repeatability).
  int transition_guard = 0;
  while (transition_guard++ < 8) {
    if (segment_idx_ >= static_cast<int>(cfg_.gravity_offsets_rad.size())) {
      MarkDone("gravity grid completed");
      q_ref[j] = q_hold;
      qdot_ref[j] = 0.0;
      qddot_ref[j] = 0.0;
      return;
    }

    if (phase_ == SysIDPhase::RAMP) {
      const double ramp = cfg_.ramp_time;
      if (ramp <= 1e-9) {
        q_ref[j] = segment_q_goal_;
        qdot_ref[j] = 0.0;
        qddot_ref[j] = 0.0;
        if (cfg_.dwell_time > 1e-9) {
          EnterPhase(SysIDPhase::DWELL, phase_start_time_sec_);
        } else {
          ++segment_idx_;
          if (segment_idx_ >= static_cast<int>(cfg_.gravity_offsets_rad.size())) {
            MarkDone("gravity grid completed");
            return;
          }
          segment_q_start_ = segment_q_goal_;
          segment_q_goal_ = q_hold + cfg_.gravity_offsets_rad[segment_idx_];
          EnterPhase(SysIDPhase::RAMP, phase_start_time_sec_);
        }
        continue;
      }

      const double local_t = std::max(0.0, time_sec - phase_start_time_sec_);
      if (local_t < ramp) {
        const double s = local_t / ramp;
        const double a = SmoothStep(s);
        const double adot = SmoothStepDot(s) / ramp;
        const double dq = segment_q_goal_ - segment_q_start_;
        q_ref[j] = segment_q_start_ + a * dq;
        qdot_ref[j] = adot * dq;
        qddot_ref[j] = 0.0;
        return;
      }

      q_ref[j] = segment_q_goal_;
      qdot_ref[j] = 0.0;
      qddot_ref[j] = 0.0;
      const double ramp_end = phase_start_time_sec_ + ramp;
      if (cfg_.dwell_time > 1e-9) {
        EnterPhase(SysIDPhase::DWELL, ramp_end);
      } else {
        ++segment_idx_;
        if (segment_idx_ >= static_cast<int>(cfg_.gravity_offsets_rad.size())) {
          MarkDone("gravity grid completed");
          return;
        }
        segment_q_start_ = segment_q_goal_;
        segment_q_goal_ = q_hold + cfg_.gravity_offsets_rad[segment_idx_];
        EnterPhase(SysIDPhase::RAMP, ramp_end);
      }
      continue;
    }

    if (phase_ == SysIDPhase::DWELL) {
      q_ref[j] = segment_q_goal_;
      qdot_ref[j] = 0.0;
      qddot_ref[j] = 0.0;

      const double dwell = cfg_.dwell_time;
      if (dwell <= 1e-9) {
        ++segment_idx_;
        if (segment_idx_ >= static_cast<int>(cfg_.gravity_offsets_rad.size())) {
          MarkDone("gravity grid completed");
          return;
        }
        segment_q_start_ = segment_q_goal_;
        segment_q_goal_ = q_hold + cfg_.gravity_offsets_rad[segment_idx_];
        EnterPhase(SysIDPhase::RAMP, phase_start_time_sec_);
        continue;
      }

      const double local_t = std::max(0.0, time_sec - phase_start_time_sec_);
      if (local_t < dwell) {
        return;
      }

      ++segment_idx_;
      if (segment_idx_ >= static_cast<int>(cfg_.gravity_offsets_rad.size())) {
        MarkDone("gravity grid completed");
        return;
      }

      segment_q_start_ = segment_q_goal_;
      segment_q_goal_ = q_hold + cfg_.gravity_offsets_rad[segment_idx_];
      EnterPhase(SysIDPhase::RAMP, phase_start_time_sec_ + dwell);
      continue;
    }

    return;
  }
}

void SysID::UpdateFrictionSweep(double time_sec, double dt_sec,
                                Eigen::Ref<Eigen::VectorXd> q_ref,
                                Eigen::Ref<Eigen::VectorXd> qdot_ref,
                                Eigen::Ref<Eigen::VectorXd> qddot_ref) {
  const int j = ClampedJointIdx();
  const double t_mode = std::max(0.0, Elapsed(time_sec) - cfg_.start_delay);

  if (phase_ == SysIDPhase::START_DELAY) {
    return;
  }

  if (cfg_.duration > 1e-9 && t_mode >= cfg_.duration) {
    MarkDone("friction sweep duration reached");
    return;
  }

  EnterPhase(SysIDPhase::CRUISE, time_sec);

  const double v = std::abs(cfg_.cruise_vel);
  const double a = std::abs(cfg_.amplitude);
  if (a <= 1e-9 || v <= 1e-9) {
    q_ref[j] = hold_q_[j] + cfg_.offset;
    return;
  }

  if (dt_sec > 0.0) {
    const double dir = reverse_ ? -1.0 : 1.0;
    sweep_x_rel_ += dir * v * dt_sec;

    if (sweep_x_rel_ >= a) {
      sweep_x_rel_ = a;
      reverse_ = true;
    } else if (sweep_x_rel_ <= -a) {
      sweep_x_rel_ = -a;
      reverse_ = false;
    }
  }

  q_ref[j] = hold_q_[j] + cfg_.offset + sweep_x_rel_;
  qdot_ref[j] = reverse_ ? -v : v;
  qddot_ref[j] = 0.0;
}

void SysID::UpdateSine(double time_sec,
                       Eigen::Ref<Eigen::VectorXd> q_ref,
                       Eigen::Ref<Eigen::VectorXd> qdot_ref,
                       Eigen::Ref<Eigen::VectorXd> qddot_ref) {
  const int j = ClampedJointIdx();
  const double t_mode = std::max(0.0, Elapsed(time_sec) - cfg_.start_delay);

  if (phase_ == SysIDPhase::START_DELAY) {
    return;
  }

  if (cfg_.duration > 1e-9 && t_mode >= cfg_.duration) {
    MarkDone("sine duration reached");
    return;
  }

  EnterPhase(SysIDPhase::CRUISE, time_sec);

  const double omega = kTwoPi * std::max(0.0, cfg_.frequency_hz);
  const double theta = cfg_.phase0 + omega * t_mode;
  const double s = std::sin(theta);
  const double c = std::cos(theta);

  q_ref[j] = hold_q_[j] + cfg_.offset + cfg_.amplitude * s;
  qdot_ref[j] = cfg_.amplitude * omega * c;
  qddot_ref[j] = -cfg_.amplitude * omega * omega * s;
}

void SysID::UpdateChirp(double time_sec, double dt_sec,
                        Eigen::Ref<Eigen::VectorXd> q_ref,
                        Eigen::Ref<Eigen::VectorXd> qdot_ref,
                        Eigen::Ref<Eigen::VectorXd> qddot_ref) {
  const int j = ClampedJointIdx();
  const double t_mode = std::max(0.0, Elapsed(time_sec) - cfg_.start_delay);
  const double duration = std::max(EffectiveChirpDuration(), 1e-9);

  if (phase_ == SysIDPhase::START_DELAY) {
    return;
  }

  if (t_mode >= duration) {
    MarkDone("chirp duration reached");
    return;
  }

  EnterPhase(SysIDPhase::CRUISE, time_sec);

  const double f0 = std::max(cfg_.chirp_f0_hz, 1e-6);
  const double f1 = std::max(cfg_.chirp_f1_hz, f0);
  const double beta = Clamp01(t_mode / duration);
  const double f = f0 * std::exp(std::log(f1 / f0) * beta);
  const double omega = kTwoPi * f;

  if (dt_sec > 0.0) {
    chirp_phase_ += omega * dt_sec;
  }

  const double theta = chirp_phase_;
  const double s = std::sin(theta);
  const double c = std::cos(theta);

  q_ref[j] = hold_q_[j] + cfg_.offset + cfg_.amplitude * s;
  qdot_ref[j] = cfg_.amplitude * omega * c;
  qddot_ref[j] = -cfg_.amplitude * omega * omega * s;
}

}  // namespace wbc
