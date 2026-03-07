/**
 * @file residual_compensator/include/residual_compensator/sysid.hpp
 * @brief Offline SysID excitation helper for joint-space residual model fitting.
 */
#pragma once

#include <string>
#include <vector>

#include <Eigen/Dense>

namespace wbc {

enum class SysIDMode {
  OFF = 0,
  GRAVITY_GRID,
  FRICTION_SWEEP,
  SINE,
  CHIRP,
};

enum class SysIDPhase {
  IDLE = 0,
  START_DELAY,
  RAMP,
  DWELL,
  CRUISE,
  DONE,
  ABORT,
};

struct SysIDConfig {
  bool enabled{false};
  SysIDMode mode{SysIDMode::OFF};

  // Active joint index in [0, num_active).
  int joint_idx{0};

  // Timing (seconds).
  double start_delay{1.0};
  double ramp_time{1.0};     // used by gravity_grid
  double dwell_time{1.0};    // used by gravity_grid
  double duration{20.0};     // used by friction_sweep/sine/chirp (if chirp_duration <= 0)

  // Excitation parameters.
  double amplitude{0.05};
  double offset{0.0};
  double cruise_vel{0.2};      // friction_sweep
  double frequency_hz{0.5};    // sine
  double chirp_f0_hz{0.2};
  double chirp_f1_hz{3.0};
  double chirp_duration{20.0};
  double phase0{0.0};

  // Gravity map points relative to hold posture.
  std::vector<double> gravity_offsets_rad;

  // Safety gates.
  double max_tracking_err{0.15};
  double max_meas_vel{2.0};
  double max_tau_ratio{0.9};
};

struct SysIDRuntimeState {
  bool active{false};
  bool finished{false};
  bool aborted{false};

  SysIDMode mode{SysIDMode::OFF};
  SysIDPhase phase{SysIDPhase::IDLE};

  int joint_idx{0};
  int segment_idx{0};
  bool reverse{false};

  double elapsed{0.0};
  double target_q{0.0};
  double target_qdot{0.0};
  double target_qddot{0.0};
};

class SysID {
public:
  SysID() = default;

  void Setup(int num_active);
  bool IsSetup() const { return n_active_ > 0; }

  void Configure(const SysIDConfig& cfg);
  const SysIDConfig& Config() const { return cfg_; }

  // Captures hold posture used as baseline for excitation references.
  void Reset(const Eigen::Ref<const Eigen::VectorXd>& hold_q);

  void Start(double start_time_sec);
  void Stop();
  void Abort(const std::string& reason);

  bool IsActive() const { return active_; }
  bool IsFinished() const { return finished_; }
  bool IsAborted() const { return aborted_; }
  SysIDPhase Phase() const { return phase_; }
  const std::string& LastReason() const { return last_reason_; }

  // Writes full references (all joints). Non-target joints stay at hold posture.
  void Update(double time_sec, double dt_sec,
              Eigen::Ref<Eigen::VectorXd> q_ref,
              Eigen::Ref<Eigen::VectorXd> qdot_ref,
              Eigen::Ref<Eigen::VectorXd> qddot_ref);

  // Returns true when safety checks pass or SysID is inactive.
  bool CheckSafety(const Eigen::Ref<const Eigen::VectorXd>& q_meas,
                   const Eigen::Ref<const Eigen::VectorXd>& qdot_meas,
                   const Eigen::Ref<const Eigen::VectorXd>& tau_cmd,
                   const Eigen::Ref<const Eigen::MatrixXd>& tau_limits,
                   std::string* reason = nullptr) const;

  // Check safety and switch to ABORT state on failure.
  bool CheckSafetyAndAbort(const Eigen::Ref<const Eigen::VectorXd>& q_meas,
                           const Eigen::Ref<const Eigen::VectorXd>& qdot_meas,
                           const Eigen::Ref<const Eigen::VectorXd>& tau_cmd,
                           const Eigen::Ref<const Eigen::MatrixXd>& tau_limits,
                           std::string* reason = nullptr);

  SysIDRuntimeState GetRuntimeState(double time_sec) const;

private:
  static double Clamp01(double x);
  static double SmoothStep(double s);
  static double SmoothStepDot(double s);

  int ClampedJointIdx() const;
  double Elapsed(double time_sec) const;
  double EffectiveChirpDuration() const;

  void EnterPhase(SysIDPhase phase, double time_sec);
  void MarkDone(const std::string& reason);

  void UpdateGravityGrid(double time_sec,
                         Eigen::Ref<Eigen::VectorXd> q_ref,
                         Eigen::Ref<Eigen::VectorXd> qdot_ref,
                         Eigen::Ref<Eigen::VectorXd> qddot_ref);
  void UpdateFrictionSweep(double time_sec, double dt_sec,
                           Eigen::Ref<Eigen::VectorXd> q_ref,
                           Eigen::Ref<Eigen::VectorXd> qdot_ref,
                           Eigen::Ref<Eigen::VectorXd> qddot_ref);
  void UpdateSine(double time_sec,
                  Eigen::Ref<Eigen::VectorXd> q_ref,
                  Eigen::Ref<Eigen::VectorXd> qdot_ref,
                  Eigen::Ref<Eigen::VectorXd> qddot_ref);
  void UpdateChirp(double time_sec, double dt_sec,
                   Eigen::Ref<Eigen::VectorXd> q_ref,
                   Eigen::Ref<Eigen::VectorXd> qdot_ref,
                   Eigen::Ref<Eigen::VectorXd> qddot_ref);

  int n_active_{0};
  SysIDConfig cfg_;

  bool active_{false};
  bool finished_{false};
  bool aborted_{false};

  SysIDPhase phase_{SysIDPhase::IDLE};
  int segment_idx_{0};
  bool reverse_{false};

  double start_time_sec_{0.0};
  double phase_start_time_sec_{0.0};
  double chirp_phase_{0.0};
  double sweep_x_rel_{0.0};

  double segment_q_start_{0.0};
  double segment_q_goal_{0.0};

  // Last generated target sample for safety checks.
  double last_target_q_{0.0};
  double last_target_qdot_{0.0};
  double last_target_qddot_{0.0};

  std::string last_reason_;
  Eigen::VectorXd hold_q_;
};

}  // namespace wbc
