#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <future>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <geometry_msgs/msg/twist_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>

#include "wbc_msgs/msg/task_state.hpp"
#include "wbc_msgs/msg/wbc_state.hpp"
#include "wbc_msgs/srv/residual_dynamics_service.hpp"
#include "wbc_msgs/srv/task_gain_service.hpp"
#include "wbc_msgs/srv/task_weight_service.hpp"
#include "wbc_msgs/srv/transition_state.hpp"

namespace {

constexpr double kRadToDeg = 57.29577951308232;

struct RunningStat {
  double sum_sq{0.0};
  std::size_t count{0};

  void Add(double v) {
    sum_sq += v * v;
    ++count;
  }

  double Rms() const {
    if (count == 0) {
      return 0.0;
    }
    return std::sqrt(sum_sq / static_cast<double>(count));
  }
};

struct JointSegment {
  std::vector<double> qdot;
  double duration_sec{0.0};
  bool is_hold{false};
};

struct JointProfile {
  std::string name;
  std::vector<JointSegment> segments;
};

struct CartesianSegment {
  std::array<double, 3> linear{{0.0, 0.0, 0.0}};
  std::array<double, 3> angular{{0.0, 0.0, 0.0}};
  double duration_sec{0.0};
  bool is_hold{false};
};

struct CartesianProfile {
  std::string name;
  std::vector<CartesianSegment> segments;
};

struct Candidate {
  double jpos_kp{0.0};
  double jpos_kd{0.0};
  double ee_kp{0.0};
  double ee_kd{0.0};
  double cart_jpos_weight{0.0};
  double cart_ee_weight{0.0};

  bool friction_enabled{false};
  double gamma_c{0.0};
  double gamma_v{0.0};
  double max_f_c{0.0};
  double max_f_v{0.0};

  bool observer_enabled{false};
  double observer_k{0.0};
  double max_tau_dist{0.0};

  std::string label;
};

struct CandidateResult {
  Candidate candidate;

  bool success{false};
  std::string fail_reason;

  double cart_pos_rms_mm{0.0};
  double cart_pos_hold_mm{0.0};
  double cart_ori_rms_deg{0.0};
  double joint_rms_mrad{0.0};
  double joint_hold_mrad{0.0};
  double max_tau_ratio{0.0};
  double score{std::numeric_limits<double>::infinity()};
};

struct SampleData {
  bool has_jpos_err{false};
  bool has_ee_pos_err{false};
  bool has_ee_ori_err{false};

  double jpos_err_norm{0.0};
  double ee_pos_err_norm{0.0};
  double ee_ori_err_norm{0.0};

  double max_tau_ratio{0.0};
};

std::string JoinNs(const std::string& ns, const std::string& name) {
  if (ns.empty() || ns == "/") {
    return "/" + name;
  }
  if (ns.back() == '/') {
    return ns + name;
  }
  return ns + "/" + name;
}

double VectorNorm(const std::vector<double>& v) {
  if (v.empty()) {
    return 0.0;
  }
  double sum_sq = 0.0;
  for (double x : v) {
    sum_sq += x * x;
  }
  return std::sqrt(sum_sq);
}

bool FindTaskErrorNorm(const wbc_msgs::msg::WbcState& msg,
                       const std::string& task_name,
                       double& out_err_norm) {
  for (const auto& task : msg.tasks) {
    if (task.name == task_name) {
      out_err_norm = VectorNorm(task.x_err);
      return true;
    }
  }
  return false;
}

template <typename T>
std::vector<T> ExpandOrDefault(const std::vector<T>& src,
                               std::size_t n,
                               T fallback) {
  if (n == 0) {
    return {};
  }
  if (src.empty()) {
    return std::vector<T>(n, fallback);
  }
  if (src.size() == 1) {
    return std::vector<T>(n, src.front());
  }
  if (src.size() == n) {
    return src;
  }
  std::vector<T> out(n, fallback);
  const std::size_t copy_n = std::min(n, src.size());
  std::copy(src.begin(), src.begin() + static_cast<std::ptrdiff_t>(copy_n),
            out.begin());
  return out;
}

}  // namespace

namespace optimo_controller {

class GainSweepNode : public rclcpp::Node {
public:
  GainSweepNode()
  : rclcpp::Node("gain_sweep_node") {
    DeclareParameters();
    LoadParameters();

    joint_vel_pub_ = create_publisher<std_msgs::msg::Float64MultiArray>(
        JoinNs(controller_ns_, "joint_vel_cmd"), rclcpp::SensorDataQoS());
    ee_vel_pub_ = create_publisher<geometry_msgs::msg::TwistStamped>(
        JoinNs(controller_ns_, "ee_vel_cmd"), rclcpp::SensorDataQoS());

    wbc_state_sub_ = create_subscription<wbc_msgs::msg::WbcState>(
        JoinNs(controller_ns_, "wbc_state"),
        rclcpp::SensorDataQoS(),
        [this](const wbc_msgs::msg::WbcState::SharedPtr msg) {
          std::lock_guard<std::mutex> lock(state_mutex_);
          latest_state_ = *msg;
          latest_state_recv_time_ = this->now();
          if (joint_count_ == 0 && !msg->q_curr.empty()) {
            joint_count_ = msg->q_curr.size();
          }
          has_state_ = true;
        });

    transition_client_ = create_client<wbc_msgs::srv::TransitionState>(
        JoinNs(controller_ns_, "set_state"));
    task_gain_client_ = create_client<wbc_msgs::srv::TaskGainService>(
        JoinNs(controller_ns_, "set_task_gains"));
    task_weight_client_ = create_client<wbc_msgs::srv::TaskWeightService>(
        JoinNs(controller_ns_, "set_task_weights"));
    residual_client_ = create_client<wbc_msgs::srv::ResidualDynamicsService>(
        JoinNs(controller_ns_, "set_residual_dynamics"));
  }

  int Run() {
    if (!WaitForServices()) {
      RCLCPP_ERROR(get_logger(), "Service wait failed. Aborting.");
      return 1;
    }
    if (!WaitForFirstWbcState()) {
      RCLCPP_ERROR(get_logger(), "No ~/wbc_state received. Aborting.");
      return 1;
    }

    if (warn_torque_constraint_) {
      RCLCPP_WARN(
          get_logger(),
          "This sweep node assumes JointTrqLimitConstraint is enabled in your WBC YAML during search.");
    }

    BuildTrajectoryProfiles();
    BuildCandidates();

    if (candidates_.empty()) {
      RCLCPP_ERROR(get_logger(), "No candidates generated. Check search grid params.");
      return 1;
    }

    RCLCPP_INFO(get_logger(),
                "Starting sweep: %zu candidates, %zu joint profiles, %zu cartesian profiles",
                candidates_.size(), joint_profiles_.size(), cartesian_profiles_.size());

    std::vector<CandidateResult> results;
    results.reserve(candidates_.size());

    for (std::size_t i = 0; i < candidates_.size(); ++i) {
      const auto& candidate = candidates_[i];
      RCLCPP_INFO(get_logger(), "[%zu/%zu] Evaluating %s",
                  i + 1, candidates_.size(), candidate.label.c_str());

      auto result = EvaluateCandidate(candidate);
      results.push_back(result);

      if (result.success) {
        RCLCPP_INFO(get_logger(),
                    "  score=%.3f, cart_rms=%.3fmm, cart_hold=%.3fmm, cart_ori=%.3fdeg, "
                    "joint_rms=%.3fmrad, joint_hold=%.3fmrad, tau_ratio=%.3f",
                    result.score,
                    result.cart_pos_rms_mm,
                    result.cart_pos_hold_mm,
                    result.cart_ori_rms_deg,
                    result.joint_rms_mrad,
                    result.joint_hold_mrad,
                    result.max_tau_ratio);
      } else {
        RCLCPP_WARN(get_logger(), "  failed: %s", result.fail_reason.c_str());
      }
    }

    if (!WriteCsv(results)) {
      RCLCPP_WARN(get_logger(), "Failed to write CSV output: %s", csv_output_path_.c_str());
    }

    std::vector<const CandidateResult*> successful;
    successful.reserve(results.size());
    for (const auto& r : results) {
      if (r.success) {
        successful.push_back(&r);
      }
    }

    if (successful.empty()) {
      RCLCPP_ERROR(get_logger(), "Sweep finished but no successful candidates.");
      return 2;
    }

    std::sort(successful.begin(), successful.end(),
              [](const CandidateResult* a, const CandidateResult* b) {
                return a->score < b->score;
              });

    const std::size_t report_n = std::min(top_k_report_, successful.size());
    RCLCPP_INFO(get_logger(), "Top %zu candidates:", report_n);
    for (std::size_t i = 0; i < report_n; ++i) {
      const auto* r = successful[i];
      RCLCPP_INFO(get_logger(),
                  "  #%zu: %s | score=%.3f | cart_rms=%.3fmm hold=%.3fmm ori=%.3fdeg | "
                  "joint_rms=%.3fmrad hold=%.3fmrad | tau_ratio=%.3f",
                  i + 1,
                  r->candidate.label.c_str(),
                  r->score,
                  r->cart_pos_rms_mm,
                  r->cart_pos_hold_mm,
                  r->cart_ori_rms_deg,
                  r->joint_rms_mrad,
                  r->joint_hold_mrad,
                  r->max_tau_ratio);
    }

    const auto* best = successful.front();
    RCLCPP_INFO(get_logger(), "Best candidate: %s", best->candidate.label.c_str());

    if (apply_best_at_end_) {
      if (ApplyCandidateServices(best->candidate)) {
        SetCartesianPhaseWeights(best->candidate);
        RequestState(home_state_name_);
        RCLCPP_INFO(get_logger(), "Applied best candidate to controller.");
      } else {
        RCLCPP_WARN(get_logger(), "Could not apply best candidate at end.");
      }
    }

    RCLCPP_INFO(get_logger(), "Sweep complete. CSV: %s", csv_output_path_.c_str());
    return 0;
  }

private:
  struct JointEval {
    RunningStat transit_err_rad;
    RunningStat hold_err_rad;
    double max_tau_ratio{0.0};
  };

  struct CartesianEval {
    RunningStat transit_pos_err_m;
    RunningStat hold_pos_err_m;
    RunningStat transit_ori_err_rad;
    double max_tau_ratio{0.0};
  };

  void DeclareParameters() {
    declare_parameter<std::string>("controller_ns", "/optimo/wbc_controller");
    declare_parameter<std::string>("joint_task_name", "jpos_task");
    declare_parameter<std::string>("ee_pos_task_name", "ee_pos_task");
    declare_parameter<std::string>("ee_ori_task_name", "ee_ori_task");

    declare_parameter<std::string>("home_state_name", "home");
    declare_parameter<std::string>("joint_state_name", "joint_teleop");
    declare_parameter<std::string>("cartesian_state_name", "cartesian_teleop");

    declare_parameter<double>("publish_hz", 100.0);
    declare_parameter<double>("service_timeout_sec", 5.0);
    declare_parameter<double>("state_data_timeout_sec", 0.25);
    declare_parameter<double>("transition_settle_sec", 0.35);
    declare_parameter<double>("home_settle_sec", 2.25);

    declare_parameter<double>("joint_phase_jpos_weight", 100.0);
    declare_parameter<double>("joint_phase_ee_weight", 1e-6);

    declare_parameter<bool>("apply_best_at_end", true);
    declare_parameter<bool>("warn_torque_constraint", true);

    declare_parameter<bool>("fail_on_torque_violation", true);
    declare_parameter<double>("torque_ratio_fail_threshold", 1.05);
    declare_parameter<std::vector<double>>(
        "torque_limits", std::vector<double>{79.0, 95.0, 32.0, 40.0, 15.0, 15.0, 15.0});

    declare_parameter<double>("score_w_cart_rms", 1.0);
    declare_parameter<double>("score_w_cart_hold", 1.0);
    declare_parameter<double>("score_w_cart_ori", 0.5);
    declare_parameter<double>("score_w_joint_rms", 1.0);
    declare_parameter<double>("score_w_joint_hold", 1.0);
    declare_parameter<double>("score_w_tau_ratio", 25.0);

    declare_parameter<int>("top_k_report", 5);
    declare_parameter<int>("seed", 12345);

    declare_parameter<int>("joint_random_profiles", 4);
    declare_parameter<int>("cartesian_random_profiles", 4);
    declare_parameter<int>("joint_segments_per_profile", 6);
    declare_parameter<int>("cartesian_segments_per_profile", 8);
    declare_parameter<double>("segment_duration_min_sec", 0.5);
    declare_parameter<double>("segment_duration_max_sec", 1.2);
    declare_parameter<double>("profile_hold_sec", 0.8);

    declare_parameter<std::vector<double>>(
        "joint_vel_limits", std::vector<double>{0.4, 0.4, 0.35, 0.35, 0.25, 0.25, 0.25});
    declare_parameter<double>("cartesian_linear_vel_max", 0.08);
    declare_parameter<double>("cartesian_angular_vel_max", 0.35);

    declare_parameter<std::vector<double>>("search.jpos_kp", std::vector<double>{200.0});
    declare_parameter<std::vector<double>>("search.jpos_kd", std::vector<double>{28.0});
    declare_parameter<std::vector<double>>("search.ee_kp", std::vector<double>{3200.0, 6400.0});
    declare_parameter<std::vector<double>>("search.ee_kd", std::vector<double>{113.0, 160.0});
    declare_parameter<std::vector<double>>("search.cart_jpos_weight", std::vector<double>{1.0});
    declare_parameter<std::vector<double>>("search.cart_ee_weight", std::vector<double>{100.0});

    declare_parameter<bool>("search.include_baseline", true);
    declare_parameter<bool>("search.include_observer_only", true);
    declare_parameter<bool>("search.include_friction_observer", true);
    declare_parameter<bool>("search.include_friction_only", false);

    declare_parameter<std::vector<double>>("search.observer_k", std::vector<double>{50.0, 100.0});
    declare_parameter<std::vector<double>>("search.observer_max_tau_dist", std::vector<double>{20.0});
    declare_parameter<std::vector<double>>("search.gamma_c", std::vector<double>{5.0, 10.0});
    declare_parameter<std::vector<double>>("search.gamma_v", std::vector<double>{2.0, 4.0});
    declare_parameter<std::vector<double>>("search.max_f_c", std::vector<double>{8.0});
    declare_parameter<std::vector<double>>("search.max_f_v", std::vector<double>{5.0});

    declare_parameter<int>("search.max_candidates", 0);

    declare_parameter<std::string>("csv_output", "/tmp/optimo_gain_sweep.csv");
  }

  void LoadParameters() {
    controller_ns_ = get_parameter("controller_ns").as_string();

    joint_task_name_ = get_parameter("joint_task_name").as_string();
    ee_pos_task_name_ = get_parameter("ee_pos_task_name").as_string();
    ee_ori_task_name_ = get_parameter("ee_ori_task_name").as_string();

    home_state_name_ = get_parameter("home_state_name").as_string();
    joint_state_name_ = get_parameter("joint_state_name").as_string();
    cartesian_state_name_ = get_parameter("cartesian_state_name").as_string();

    publish_hz_ = std::max(10.0, get_parameter("publish_hz").as_double());
    service_timeout_sec_ = std::max(0.5, get_parameter("service_timeout_sec").as_double());
    state_data_timeout_sec_ = std::max(0.05, get_parameter("state_data_timeout_sec").as_double());
    transition_settle_sec_ = std::max(0.0, get_parameter("transition_settle_sec").as_double());
    home_settle_sec_ = std::max(0.0, get_parameter("home_settle_sec").as_double());

    joint_phase_jpos_weight_ = get_parameter("joint_phase_jpos_weight").as_double();
    joint_phase_ee_weight_ = get_parameter("joint_phase_ee_weight").as_double();

    apply_best_at_end_ = get_parameter("apply_best_at_end").as_bool();
    warn_torque_constraint_ = get_parameter("warn_torque_constraint").as_bool();

    fail_on_torque_violation_ = get_parameter("fail_on_torque_violation").as_bool();
    torque_ratio_fail_threshold_ =
        std::max(0.1, get_parameter("torque_ratio_fail_threshold").as_double());
    torque_limits_ = get_parameter("torque_limits").as_double_array();

    score_w_cart_rms_ = get_parameter("score_w_cart_rms").as_double();
    score_w_cart_hold_ = get_parameter("score_w_cart_hold").as_double();
    score_w_cart_ori_ = get_parameter("score_w_cart_ori").as_double();
    score_w_joint_rms_ = get_parameter("score_w_joint_rms").as_double();
    score_w_joint_hold_ = get_parameter("score_w_joint_hold").as_double();
    score_w_tau_ratio_ = get_parameter("score_w_tau_ratio").as_double();

    top_k_report_ = static_cast<std::size_t>(
        std::max<int64_t>(1, get_parameter("top_k_report").as_int()));
    seed_ = static_cast<unsigned int>(get_parameter("seed").as_int());

    joint_random_profiles_ =
        static_cast<int>(std::max<int64_t>(0, get_parameter("joint_random_profiles").as_int()));
    cart_random_profiles_ =
        static_cast<int>(std::max<int64_t>(0, get_parameter("cartesian_random_profiles").as_int()));
    joint_segments_per_profile_ =
        static_cast<int>(std::max<int64_t>(1, get_parameter("joint_segments_per_profile").as_int()));
    cart_segments_per_profile_ =
        static_cast<int>(std::max<int64_t>(1, get_parameter("cartesian_segments_per_profile").as_int()));
    segment_duration_min_sec_ =
        std::max(0.05, get_parameter("segment_duration_min_sec").as_double());
    segment_duration_max_sec_ =
        std::max(segment_duration_min_sec_, get_parameter("segment_duration_max_sec").as_double());
    profile_hold_sec_ = std::max(0.0, get_parameter("profile_hold_sec").as_double());

    joint_vel_limits_ = get_parameter("joint_vel_limits").as_double_array();
    cart_linear_vel_max_ = std::max(0.01, get_parameter("cartesian_linear_vel_max").as_double());
    cart_angular_vel_max_ = std::max(0.01, get_parameter("cartesian_angular_vel_max").as_double());

    search_jpos_kp_ = get_parameter("search.jpos_kp").as_double_array();
    search_jpos_kd_ = get_parameter("search.jpos_kd").as_double_array();
    search_ee_kp_ = get_parameter("search.ee_kp").as_double_array();
    search_ee_kd_ = get_parameter("search.ee_kd").as_double_array();
    search_cart_jpos_weight_ = get_parameter("search.cart_jpos_weight").as_double_array();
    search_cart_ee_weight_ = get_parameter("search.cart_ee_weight").as_double_array();

    search_include_baseline_ = get_parameter("search.include_baseline").as_bool();
    search_include_observer_only_ = get_parameter("search.include_observer_only").as_bool();
    search_include_friction_observer_ =
        get_parameter("search.include_friction_observer").as_bool();
    search_include_friction_only_ = get_parameter("search.include_friction_only").as_bool();

    search_observer_k_ = get_parameter("search.observer_k").as_double_array();
    search_observer_max_tau_ = get_parameter("search.observer_max_tau_dist").as_double_array();
    search_gamma_c_ = get_parameter("search.gamma_c").as_double_array();
    search_gamma_v_ = get_parameter("search.gamma_v").as_double_array();
    search_max_f_c_ = get_parameter("search.max_f_c").as_double_array();
    search_max_f_v_ = get_parameter("search.max_f_v").as_double_array();

    search_max_candidates_ = get_parameter("search.max_candidates").as_int();

    csv_output_path_ = get_parameter("csv_output").as_string();

    rng_.seed(seed_);
  }

  bool WaitForServices() {
    const auto timeout = std::chrono::duration<double>(service_timeout_sec_);
    const auto wait_client = [this, timeout](auto& client, const char* name) {
      if (!client->wait_for_service(timeout)) {
        RCLCPP_ERROR(get_logger(), "Service not available: %s", name);
        return false;
      }
      return true;
    };

    return wait_client(transition_client_, "set_state") &&
           wait_client(task_gain_client_, "set_task_gains") &&
           wait_client(task_weight_client_, "set_task_weights") &&
           wait_client(residual_client_, "set_residual_dynamics");
  }

  bool WaitForFirstWbcState() {
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::duration<double>(service_timeout_sec_);
    rclcpp::WallRate rate(200.0);

    while (rclcpp::ok()) {
      {
        std::lock_guard<std::mutex> lock(state_mutex_);
        if (has_state_) {
          if (joint_count_ == 0) {
            joint_count_ = latest_state_.q_curr.size();
          }
          if (joint_count_ > 0) {
            return true;
          }
        }
      }

      if (std::chrono::steady_clock::now() > deadline) {
        return false;
      }

      SpinSome();
      rate.sleep();
    }

    return false;
  }

  void BuildTrajectoryProfiles() {
    joint_profiles_.clear();
    cartesian_profiles_.clear();

    const auto vel_limits = ExpandOrDefault<double>(joint_vel_limits_, joint_count_, 0.25);

    // Deterministic joint profile 1: piecewise single-joint motion.
    {
      JointProfile p;
      p.name = "joint_piecewise_primary";
      p.segments.push_back(MakeJointSegment(vel_limits, {{0, +0.5}}, 1.0, false));
      p.segments.push_back(MakeJointSegment(vel_limits, {{0, -0.5}}, 1.0, false));
      p.segments.push_back(MakeJointSegment(vel_limits, {{2, +0.5}}, 1.0, false));
      p.segments.push_back(MakeJointSegment(vel_limits, {{2, -0.5}}, 1.0, false));
      p.segments.push_back(ZeroJointSegment(profile_hold_sec_, true));
      joint_profiles_.push_back(std::move(p));
    }

    // Deterministic joint profile 2: coupled motion.
    {
      JointProfile p;
      p.name = "joint_piecewise_coupled";
      p.segments.push_back(MakeJointSegment(vel_limits, {{0, +0.45}, {1, -0.35}}, 0.9, false));
      p.segments.push_back(MakeJointSegment(vel_limits, {{3, +0.35}, {5, +0.30}}, 0.9, false));
      p.segments.push_back(MakeJointSegment(vel_limits, {{1, +0.30}, {4, -0.30}}, 0.9, false));
      p.segments.push_back(MakeJointSegment(vel_limits, {{0, -0.45}, {3, -0.35}}, 0.9, false));
      p.segments.push_back(ZeroJointSegment(profile_hold_sec_, true));
      joint_profiles_.push_back(std::move(p));
    }

    // Randomized joint profiles.
    std::uniform_int_distribution<int> active_dist(1, std::min<int>(3, joint_count_));
    std::uniform_real_distribution<double> dur_dist(segment_duration_min_sec_, segment_duration_max_sec_);
    std::uniform_real_distribution<double> sign_dist(-1.0, 1.0);
    std::uniform_int_distribution<int> joint_dist(0, static_cast<int>(joint_count_ - 1));
    std::uniform_real_distribution<double> mag_scale(0.35, 0.95);

    for (int p_idx = 0; p_idx < joint_random_profiles_; ++p_idx) {
      JointProfile p;
      p.name = "joint_random_" + std::to_string(p_idx);
      for (int seg = 0; seg < joint_segments_per_profile_; ++seg) {
        std::vector<double> qdot(joint_count_, 0.0);
        const int active = active_dist(rng_);
        std::vector<int> used;
        used.reserve(static_cast<std::size_t>(active));

        for (int k = 0; k < active; ++k) {
          int j = joint_dist(rng_);
          int guard = 0;
          while (std::find(used.begin(), used.end(), j) != used.end() && guard < 20) {
            j = joint_dist(rng_);
            ++guard;
          }
          used.push_back(j);
          const double sgn = (sign_dist(rng_) >= 0.0) ? 1.0 : -1.0;
          qdot[static_cast<std::size_t>(j)] = sgn * vel_limits[static_cast<std::size_t>(j)] * mag_scale(rng_);
        }

        p.segments.push_back(JointSegment{qdot, dur_dist(rng_), false});
      }
      p.segments.push_back(ZeroJointSegment(profile_hold_sec_, true));
      joint_profiles_.push_back(std::move(p));
    }

    // Deterministic cartesian profile 1: rectangle in x-z.
    {
      CartesianProfile p;
      p.name = "cart_rectangle_xz";
      p.segments.push_back(CartesianSegment{{+0.05, 0.00, 0.00}, {0.0, 0.0, 0.0}, 1.0, false});
      p.segments.push_back(CartesianSegment{{0.00, 0.00, +0.05}, {0.0, 0.0, 0.0}, 1.0, false});
      p.segments.push_back(CartesianSegment{{-0.05, 0.00, 0.00}, {0.0, 0.0, 0.0}, 1.0, false});
      p.segments.push_back(CartesianSegment{{0.00, 0.00, -0.05}, {0.0, 0.0, 0.0}, 1.0, false});
      p.segments.push_back(CartesianSegment{{0.00, 0.00, 0.00}, {0.0, 0.0, 0.0}, profile_hold_sec_, true});
      cartesian_profiles_.push_back(std::move(p));
    }

    // Deterministic cartesian profile 2: depth probing with orientation rate.
    {
      CartesianProfile p;
      p.name = "cart_probe_yaw";
      p.segments.push_back(CartesianSegment{{0.00, +0.04, 0.00}, {0.0, 0.0, +0.15}, 1.0, false});
      p.segments.push_back(CartesianSegment{{0.00, -0.06, 0.02}, {0.0, 0.0, -0.20}, 1.0, false});
      p.segments.push_back(CartesianSegment{{0.02, +0.02, -0.03}, {0.0, +0.15, 0.0}, 1.0, false});
      p.segments.push_back(CartesianSegment{{-0.02, 0.00, +0.01}, {0.0, -0.15, 0.0}, 1.0, false});
      p.segments.push_back(CartesianSegment{{0.00, 0.00, 0.00}, {0.0, 0.0, 0.0}, profile_hold_sec_, true});
      cartesian_profiles_.push_back(std::move(p));
    }

    // Randomized cartesian profiles.
    std::uniform_real_distribution<double> cart_dur_dist(segment_duration_min_sec_, segment_duration_max_sec_);
    std::uniform_real_distribution<double> dir_dist(-1.0, 1.0);
    std::uniform_real_distribution<double> scale_dist(0.35, 0.95);

    for (int p_idx = 0; p_idx < cart_random_profiles_; ++p_idx) {
      CartesianProfile p;
      p.name = "cart_random_" + std::to_string(p_idx);

      for (int seg = 0; seg < cart_segments_per_profile_; ++seg) {
        std::array<double, 3> lin{{0.0, 0.0, 0.0}};
        std::array<double, 3> ang{{0.0, 0.0, 0.0}};

        for (int i = 0; i < 3; ++i) {
          lin[static_cast<std::size_t>(i)] =
              cart_linear_vel_max_ * scale_dist(rng_) * dir_dist(rng_);
          ang[static_cast<std::size_t>(i)] =
              cart_angular_vel_max_ * scale_dist(rng_) * dir_dist(rng_);
        }

        p.segments.push_back(CartesianSegment{lin, ang, cart_dur_dist(rng_), false});
      }

      p.segments.push_back(CartesianSegment{{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, profile_hold_sec_, true});
      cartesian_profiles_.push_back(std::move(p));
    }
  }

  JointSegment MakeJointSegment(const std::vector<double>& vel_limits,
                                const std::vector<std::pair<int, double>>& weighted_axes,
                                double duration_sec,
                                bool is_hold) const {
    std::vector<double> qdot(joint_count_, 0.0);
    for (const auto& [idx, scale] : weighted_axes) {
      if (idx < 0 || static_cast<std::size_t>(idx) >= qdot.size()) {
        continue;
      }
      qdot[static_cast<std::size_t>(idx)] =
          vel_limits[static_cast<std::size_t>(idx)] * scale;
    }
    return JointSegment{qdot, duration_sec, is_hold};
  }

  JointSegment ZeroJointSegment(double duration_sec, bool is_hold) const {
    return JointSegment{std::vector<double>(joint_count_, 0.0), duration_sec, is_hold};
  }

  void BuildCandidates() {
    candidates_.clear();

    const auto push_candidate = [this](Candidate c) {
      if (search_max_candidates_ > 0 &&
          static_cast<int>(candidates_.size()) >= search_max_candidates_) {
        return;
      }
      candidates_.push_back(std::move(c));
    };

    for (double jkp : search_jpos_kp_) {
      for (double jkd : search_jpos_kd_) {
        for (double ekp : search_ee_kp_) {
          for (double ekd : search_ee_kd_) {
            for (double jw : search_cart_jpos_weight_) {
              for (double ew : search_cart_ee_weight_) {
                const auto make_base = [&](const std::string& suffix) {
                  Candidate c;
                  c.jpos_kp = jkp;
                  c.jpos_kd = jkd;
                  c.ee_kp = ekp;
                  c.ee_kd = ekd;
                  c.cart_jpos_weight = jw;
                  c.cart_ee_weight = ew;

                  std::ostringstream os;
                  os << "j(" << jkp << "," << jkd << ") "
                     << "ee(" << ekp << "," << ekd << ") "
                     << "w(j=" << jw << ",ee=" << ew << ") "
                     << suffix;
                  c.label = os.str();
                  return c;
                };

                if (search_include_baseline_) {
                  push_candidate(make_base("baseline"));
                }

                if (search_include_observer_only_) {
                  for (double ko : search_observer_k_) {
                    for (double mtd : search_observer_max_tau_) {
                      Candidate c = make_base("obs");
                      c.observer_enabled = true;
                      c.observer_k = ko;
                      c.max_tau_dist = mtd;

                      std::ostringstream os;
                      os << "obs(Ko=" << ko << ",max=" << mtd << ")";
                      c.label += os.str();
                      push_candidate(std::move(c));
                    }
                  }
                }

                if (search_include_friction_only_) {
                  for (double gc : search_gamma_c_) {
                    for (double gv : search_gamma_v_) {
                      for (double mfc : search_max_f_c_) {
                        for (double mfv : search_max_f_v_) {
                          Candidate c = make_base("fric");
                          c.friction_enabled = true;
                          c.gamma_c = gc;
                          c.gamma_v = gv;
                          c.max_f_c = mfc;
                          c.max_f_v = mfv;

                          std::ostringstream os;
                          os << "fric(gc=" << gc << ",gv=" << gv
                             << ",mfc=" << mfc << ",mfv=" << mfv << ")";
                          c.label += os.str();
                          push_candidate(std::move(c));
                        }
                      }
                    }
                  }
                }

                if (search_include_friction_observer_) {
                  for (double ko : search_observer_k_) {
                    for (double mtd : search_observer_max_tau_) {
                      for (double gc : search_gamma_c_) {
                        for (double gv : search_gamma_v_) {
                          for (double mfc : search_max_f_c_) {
                            for (double mfv : search_max_f_v_) {
                              Candidate c = make_base("fric+obs");
                              c.friction_enabled = true;
                              c.gamma_c = gc;
                              c.gamma_v = gv;
                              c.max_f_c = mfc;
                              c.max_f_v = mfv;
                              c.observer_enabled = true;
                              c.observer_k = ko;
                              c.max_tau_dist = mtd;

                              std::ostringstream os;
                              os << "fric(gc=" << gc << ",gv=" << gv
                                 << ",mfc=" << mfc << ",mfv=" << mfv << ")"
                                 << " obs(Ko=" << ko << ",max=" << mtd << ")";
                              c.label += os.str();
                              push_candidate(std::move(c));
                            }
                          }
                        }
                      }
                    }
                  }
                }

                if (search_max_candidates_ > 0 &&
                    static_cast<int>(candidates_.size()) >= search_max_candidates_) {
                  return;
                }
              }
            }
          }
        }
      }
    }
  }

  CandidateResult EvaluateCandidate(const Candidate& candidate) {
    CandidateResult result;
    result.candidate = candidate;

    if (!ApplyCandidateServices(candidate)) {
      result.fail_reason = "service apply failed";
      return result;
    }

    if (!SetJointPhaseWeights()) {
      result.fail_reason = "joint-phase weight set failed";
      return result;
    }

    if (!RequestState(home_state_name_)) {
      result.fail_reason = "failed to enter home";
      return result;
    }
    if (!RunIdle(home_settle_sec_)) {
      result.fail_reason = "home settle failed";
      return result;
    }

    RunningStat joint_rms_stat;
    RunningStat joint_hold_stat;
    double max_tau_ratio = 0.0;

    for (const auto& profile : joint_profiles_) {
      if (!RequestState(joint_state_name_)) {
        result.fail_reason = "failed to enter joint_teleop";
        return result;
      }
      if (!RunIdle(transition_settle_sec_)) {
        result.fail_reason = "joint transition settle failed";
        return result;
      }

      JointEval eval;
      if (!RunJointProfile(profile, eval)) {
        result.fail_reason = "joint profile failed: " + profile.name;
        return result;
      }

      joint_rms_stat.Add(eval.transit_err_rad.Rms());
      joint_hold_stat.Add(eval.hold_err_rad.Rms());
      max_tau_ratio = std::max(max_tau_ratio, eval.max_tau_ratio);
    }

    if (!SetCartesianPhaseWeights(candidate)) {
      result.fail_reason = "cart-phase weight set failed";
      return result;
    }

    if (!RequestState(home_state_name_)) {
      result.fail_reason = "failed to re-enter home";
      return result;
    }
    if (!RunIdle(home_settle_sec_)) {
      result.fail_reason = "home settle before cart failed";
      return result;
    }

    RunningStat cart_pos_rms_stat;
    RunningStat cart_hold_stat;
    RunningStat cart_ori_rms_stat;

    for (const auto& profile : cartesian_profiles_) {
      if (!RequestState(cartesian_state_name_)) {
        result.fail_reason = "failed to enter cartesian_teleop";
        return result;
      }
      if (!RunIdle(transition_settle_sec_)) {
        result.fail_reason = "cart transition settle failed";
        return result;
      }

      CartesianEval eval;
      if (!RunCartesianProfile(profile, eval)) {
        result.fail_reason = "cart profile failed: " + profile.name;
        return result;
      }

      cart_pos_rms_stat.Add(eval.transit_pos_err_m.Rms());
      cart_hold_stat.Add(eval.hold_pos_err_m.Rms());
      cart_ori_rms_stat.Add(eval.transit_ori_err_rad.Rms());
      max_tau_ratio = std::max(max_tau_ratio, eval.max_tau_ratio);
    }

    result.joint_rms_mrad = joint_rms_stat.Rms() * 1000.0;
    result.joint_hold_mrad = joint_hold_stat.Rms() * 1000.0;
    result.cart_pos_rms_mm = cart_pos_rms_stat.Rms() * 1000.0;
    result.cart_pos_hold_mm = cart_hold_stat.Rms() * 1000.0;
    result.cart_ori_rms_deg = cart_ori_rms_stat.Rms() * kRadToDeg;
    result.max_tau_ratio = max_tau_ratio;

    const double tau_penalty = std::max(0.0, result.max_tau_ratio - 1.0);
    result.score =
        score_w_cart_rms_ * result.cart_pos_rms_mm +
        score_w_cart_hold_ * result.cart_pos_hold_mm +
        score_w_cart_ori_ * result.cart_ori_rms_deg +
        score_w_joint_rms_ * result.joint_rms_mrad +
        score_w_joint_hold_ * result.joint_hold_mrad +
        score_w_tau_ratio_ * tau_penalty;

    result.success = true;
    return result;
  }

  bool ApplyCandidateServices(const Candidate& c) {
    if (!SetTaskGains(c)) {
      return false;
    }
    if (!SetResidual(c)) {
      return false;
    }
    return true;
  }

  bool SetTaskGains(const Candidate& c) {
    auto req = std::make_shared<wbc_msgs::srv::TaskGainService::Request>();
    req->task_names = {joint_task_name_, ee_pos_task_name_, ee_ori_task_name_};
    req->kp = {c.jpos_kp, c.ee_kp, c.ee_kp};
    req->kd = {c.jpos_kd, c.ee_kd, c.ee_kd};

    wbc_msgs::srv::TaskGainService::Response::SharedPtr resp;
    if (!CallService<wbc_msgs::srv::TaskGainService>(task_gain_client_, req, resp)) {
      return false;
    }
    if (!resp->success) {
      RCLCPP_WARN(get_logger(), "TaskGainService rejected: %s", resp->message.c_str());
      return false;
    }
    return true;
  }

  bool SetResidual(const Candidate& c) {
    auto req = std::make_shared<wbc_msgs::srv::ResidualDynamicsService::Request>();
    req->friction_enabled = c.friction_enabled;
    req->observer_enabled = c.observer_enabled;

    if (c.friction_enabled) {
      req->gamma_c = {c.gamma_c};
      req->gamma_v = {c.gamma_v};
      req->max_f_c = {c.max_f_c};
      req->max_f_v = {c.max_f_v};
    }

    if (c.observer_enabled) {
      req->k_o = {c.observer_k};
      req->max_tau_dist = {c.max_tau_dist};
    }

    wbc_msgs::srv::ResidualDynamicsService::Response::SharedPtr resp;
    if (!CallService<wbc_msgs::srv::ResidualDynamicsService>(residual_client_, req, resp)) {
      return false;
    }
    if (!resp->success) {
      RCLCPP_WARN(get_logger(), "ResidualDynamicsService rejected: %s", resp->message.c_str());
      return false;
    }
    return true;
  }

  bool SetJointPhaseWeights() {
    auto req = std::make_shared<wbc_msgs::srv::TaskWeightService::Request>();
    req->task_names = {joint_task_name_, ee_pos_task_name_, ee_ori_task_name_};
    req->weight = {joint_phase_jpos_weight_, joint_phase_ee_weight_, joint_phase_ee_weight_};

    wbc_msgs::srv::TaskWeightService::Response::SharedPtr resp;
    if (!CallService<wbc_msgs::srv::TaskWeightService>(task_weight_client_, req, resp)) {
      return false;
    }
    if (!resp->success) {
      RCLCPP_WARN(get_logger(), "TaskWeightService(joint-phase) rejected: %s",
                  resp->message.c_str());
      return false;
    }
    return true;
  }

  bool SetCartesianPhaseWeights(const Candidate& c) {
    auto req = std::make_shared<wbc_msgs::srv::TaskWeightService::Request>();
    req->task_names = {ee_pos_task_name_, ee_ori_task_name_, joint_task_name_};
    req->weight = {c.cart_ee_weight, c.cart_ee_weight, c.cart_jpos_weight};

    wbc_msgs::srv::TaskWeightService::Response::SharedPtr resp;
    if (!CallService<wbc_msgs::srv::TaskWeightService>(task_weight_client_, req, resp)) {
      return false;
    }
    if (!resp->success) {
      RCLCPP_WARN(get_logger(), "TaskWeightService(cart-phase) rejected: %s",
                  resp->message.c_str());
      return false;
    }
    return true;
  }

  bool RequestState(const std::string& state_name) {
    auto req = std::make_shared<wbc_msgs::srv::TransitionState::Request>();
    req->state_name = state_name;

    wbc_msgs::srv::TransitionState::Response::SharedPtr resp;
    if (!CallService<wbc_msgs::srv::TransitionState>(transition_client_, req, resp)) {
      return false;
    }
    if (!resp->success) {
      RCLCPP_WARN(get_logger(), "TransitionState(%s) rejected: %s",
                  state_name.c_str(), resp->message.c_str());
      return false;
    }
    return true;
  }

  bool RunIdle(double duration_sec) {
    if (duration_sec <= 0.0) {
      return true;
    }

    const std::size_t steps =
        std::max<std::size_t>(1, static_cast<std::size_t>(std::llround(duration_sec * publish_hz_)));

    std_msgs::msg::Float64MultiArray joint_msg;
    joint_msg.data.assign(joint_count_, 0.0);

    geometry_msgs::msg::TwistStamped ee_msg;
    ee_msg.twist.linear.x = 0.0;
    ee_msg.twist.linear.y = 0.0;
    ee_msg.twist.linear.z = 0.0;
    ee_msg.twist.angular.x = 0.0;
    ee_msg.twist.angular.y = 0.0;
    ee_msg.twist.angular.z = 0.0;

    rclcpp::WallRate rate(publish_hz_);
    for (std::size_t i = 0; i < steps && rclcpp::ok(); ++i) {
      ee_msg.header.stamp = now();
      joint_vel_pub_->publish(joint_msg);
      ee_vel_pub_->publish(ee_msg);
      SpinSome();
      rate.sleep();
    }
    return true;
  }

  bool RunJointProfile(const JointProfile& profile, JointEval& eval) {
    rclcpp::WallRate rate(publish_hz_);

    std_msgs::msg::Float64MultiArray joint_msg;

    for (const auto& seg : profile.segments) {
      const std::size_t steps =
          std::max<std::size_t>(1, static_cast<std::size_t>(std::llround(seg.duration_sec * publish_hz_)));
      joint_msg.data = seg.qdot;

      for (std::size_t i = 0; i < steps && rclcpp::ok(); ++i) {
        joint_vel_pub_->publish(joint_msg);
        SpinSome();

        SampleData sample;
        if (GetLatestSample(sample)) {
          if (sample.has_jpos_err) {
            if (seg.is_hold) {
              eval.hold_err_rad.Add(sample.jpos_err_norm);
            } else {
              eval.transit_err_rad.Add(sample.jpos_err_norm);
            }
          }
          eval.max_tau_ratio = std::max(eval.max_tau_ratio, sample.max_tau_ratio);

          if (fail_on_torque_violation_ &&
              sample.max_tau_ratio > torque_ratio_fail_threshold_) {
            RCLCPP_WARN(get_logger(),
                        "Torque ratio violation during joint profile %s: %.3f",
                        profile.name.c_str(), sample.max_tau_ratio);
            return false;
          }
        }

        rate.sleep();
      }
    }

    return eval.transit_err_rad.count > 0;
  }

  bool RunCartesianProfile(const CartesianProfile& profile, CartesianEval& eval) {
    rclcpp::WallRate rate(publish_hz_);

    geometry_msgs::msg::TwistStamped msg;

    for (const auto& seg : profile.segments) {
      const std::size_t steps =
          std::max<std::size_t>(1, static_cast<std::size_t>(std::llround(seg.duration_sec * publish_hz_)));

      msg.twist.linear.x = seg.linear[0];
      msg.twist.linear.y = seg.linear[1];
      msg.twist.linear.z = seg.linear[2];
      msg.twist.angular.x = seg.angular[0];
      msg.twist.angular.y = seg.angular[1];
      msg.twist.angular.z = seg.angular[2];

      for (std::size_t i = 0; i < steps && rclcpp::ok(); ++i) {
        msg.header.stamp = now();
        ee_vel_pub_->publish(msg);
        SpinSome();

        SampleData sample;
        if (GetLatestSample(sample)) {
          if (sample.has_ee_pos_err) {
            if (seg.is_hold) {
              eval.hold_pos_err_m.Add(sample.ee_pos_err_norm);
            } else {
              eval.transit_pos_err_m.Add(sample.ee_pos_err_norm);
            }
          }
          if (sample.has_ee_ori_err && !seg.is_hold) {
            eval.transit_ori_err_rad.Add(sample.ee_ori_err_norm);
          }

          eval.max_tau_ratio = std::max(eval.max_tau_ratio, sample.max_tau_ratio);
          if (fail_on_torque_violation_ &&
              sample.max_tau_ratio > torque_ratio_fail_threshold_) {
            RCLCPP_WARN(get_logger(),
                        "Torque ratio violation during cart profile %s: %.3f",
                        profile.name.c_str(), sample.max_tau_ratio);
            return false;
          }
        }

        rate.sleep();
      }
    }

    return eval.transit_pos_err_m.count > 0;
  }

  bool GetLatestSample(SampleData& out) {
    wbc_msgs::msg::WbcState msg;
    rclcpp::Time recv_time(0, 0, RCL_ROS_TIME);
    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      if (!has_state_) {
        return false;
      }
      msg = latest_state_;
      recv_time = latest_state_recv_time_;
    }

    const double age_sec = (now() - recv_time).seconds();
    if (age_sec > state_data_timeout_sec_) {
      return false;
    }

    double err = 0.0;
    if (FindTaskErrorNorm(msg, joint_task_name_, err)) {
      out.has_jpos_err = true;
      out.jpos_err_norm = err;
    }
    if (FindTaskErrorNorm(msg, ee_pos_task_name_, err)) {
      out.has_ee_pos_err = true;
      out.ee_pos_err_norm = err;
    }
    if (FindTaskErrorNorm(msg, ee_ori_task_name_, err)) {
      out.has_ee_ori_err = true;
      out.ee_ori_err_norm = err;
    }

    if (!msg.tau.empty()) {
      const auto limits = ExpandOrDefault<double>(torque_limits_, msg.tau.size(), 1.0);
      double ratio = 0.0;
      for (std::size_t i = 0; i < msg.tau.size(); ++i) {
        const double lim = std::max(1e-6, std::abs(limits[i]));
        ratio = std::max(ratio, std::abs(msg.tau[i]) / lim);
      }
      out.max_tau_ratio = ratio;
    }

    return true;
  }

  bool WriteCsv(const std::vector<CandidateResult>& results) const {
    if (csv_output_path_.empty()) {
      return false;
    }

    std::error_code ec;
    const std::filesystem::path out_path(csv_output_path_);
    if (out_path.has_parent_path()) {
      std::filesystem::create_directories(out_path.parent_path(), ec);
    }

    std::ofstream ofs(csv_output_path_);
    if (!ofs.is_open()) {
      return false;
    }

    ofs << "success,score,label,"
        << "jpos_kp,jpos_kd,ee_kp,ee_kd,cart_jpos_weight,cart_ee_weight,"
        << "friction_enabled,gamma_c,gamma_v,max_f_c,max_f_v,"
        << "observer_enabled,observer_k,max_tau_dist,"
        << "cart_pos_rms_mm,cart_pos_hold_mm,cart_ori_rms_deg,"
        << "joint_rms_mrad,joint_hold_mrad,max_tau_ratio,fail_reason\n";

    for (const auto& r : results) {
      ofs << (r.success ? 1 : 0) << ','
          << r.score << ','
          << '"' << EscapeCsv(r.candidate.label) << '"' << ','
          << r.candidate.jpos_kp << ','
          << r.candidate.jpos_kd << ','
          << r.candidate.ee_kp << ','
          << r.candidate.ee_kd << ','
          << r.candidate.cart_jpos_weight << ','
          << r.candidate.cart_ee_weight << ','
          << (r.candidate.friction_enabled ? 1 : 0) << ','
          << r.candidate.gamma_c << ','
          << r.candidate.gamma_v << ','
          << r.candidate.max_f_c << ','
          << r.candidate.max_f_v << ','
          << (r.candidate.observer_enabled ? 1 : 0) << ','
          << r.candidate.observer_k << ','
          << r.candidate.max_tau_dist << ','
          << r.cart_pos_rms_mm << ','
          << r.cart_pos_hold_mm << ','
          << r.cart_ori_rms_deg << ','
          << r.joint_rms_mrad << ','
          << r.joint_hold_mrad << ','
          << r.max_tau_ratio << ','
          << '"' << EscapeCsv(r.fail_reason) << '"'
          << '\n';
    }

    return true;
  }

  template <typename ServiceT>
  bool CallService(const typename rclcpp::Client<ServiceT>::SharedPtr& client,
                   const typename ServiceT::Request::SharedPtr& req,
                   typename ServiceT::Response::SharedPtr& out_resp) {
    auto future = client->async_send_request(req);
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::duration<double>(service_timeout_sec_);

    while (rclcpp::ok()) {
      if (future.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready) {
        out_resp = future.get();
        return static_cast<bool>(out_resp);
      }

      if (std::chrono::steady_clock::now() > deadline) {
        return false;
      }

      SpinSome();
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

    return false;
  }

  void SpinSome() {
    rclcpp::spin_some(get_node_base_interface());
  }

  static std::string EscapeCsv(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
      if (c == '"') {
        out.push_back('"');
      }
      out.push_back(c);
    }
    return out;
  }

private:
  // Parameters
  std::string controller_ns_;
  std::string joint_task_name_;
  std::string ee_pos_task_name_;
  std::string ee_ori_task_name_;

  std::string home_state_name_;
  std::string joint_state_name_;
  std::string cartesian_state_name_;

  double publish_hz_{100.0};
  double service_timeout_sec_{5.0};
  double state_data_timeout_sec_{0.25};
  double transition_settle_sec_{0.35};
  double home_settle_sec_{2.25};

  double joint_phase_jpos_weight_{100.0};
  double joint_phase_ee_weight_{1e-6};

  bool apply_best_at_end_{true};
  bool warn_torque_constraint_{true};

  bool fail_on_torque_violation_{true};
  double torque_ratio_fail_threshold_{1.05};
  std::vector<double> torque_limits_;

  double score_w_cart_rms_{1.0};
  double score_w_cart_hold_{1.0};
  double score_w_cart_ori_{0.5};
  double score_w_joint_rms_{1.0};
  double score_w_joint_hold_{1.0};
  double score_w_tau_ratio_{25.0};

  std::size_t top_k_report_{5};
  unsigned int seed_{12345};

  int joint_random_profiles_{4};
  int cart_random_profiles_{4};
  int joint_segments_per_profile_{6};
  int cart_segments_per_profile_{8};
  double segment_duration_min_sec_{0.5};
  double segment_duration_max_sec_{1.2};
  double profile_hold_sec_{0.8};

  std::vector<double> joint_vel_limits_;
  double cart_linear_vel_max_{0.08};
  double cart_angular_vel_max_{0.35};

  std::vector<double> search_jpos_kp_;
  std::vector<double> search_jpos_kd_;
  std::vector<double> search_ee_kp_;
  std::vector<double> search_ee_kd_;
  std::vector<double> search_cart_jpos_weight_;
  std::vector<double> search_cart_ee_weight_;

  bool search_include_baseline_{true};
  bool search_include_observer_only_{true};
  bool search_include_friction_observer_{true};
  bool search_include_friction_only_{false};

  std::vector<double> search_observer_k_;
  std::vector<double> search_observer_max_tau_;
  std::vector<double> search_gamma_c_;
  std::vector<double> search_gamma_v_;
  std::vector<double> search_max_f_c_;
  std::vector<double> search_max_f_v_;

  int search_max_candidates_{0};

  std::string csv_output_path_;

  // Runtime
  std::size_t joint_count_{0};
  std::mt19937 rng_;

  std::vector<JointProfile> joint_profiles_;
  std::vector<CartesianProfile> cartesian_profiles_;
  std::vector<Candidate> candidates_;

  // ROS interfaces
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr joint_vel_pub_;
  rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr ee_vel_pub_;

  rclcpp::Subscription<wbc_msgs::msg::WbcState>::SharedPtr wbc_state_sub_;

  rclcpp::Client<wbc_msgs::srv::TransitionState>::SharedPtr transition_client_;
  rclcpp::Client<wbc_msgs::srv::TaskGainService>::SharedPtr task_gain_client_;
  rclcpp::Client<wbc_msgs::srv::TaskWeightService>::SharedPtr task_weight_client_;
  rclcpp::Client<wbc_msgs::srv::ResidualDynamicsService>::SharedPtr residual_client_;

  mutable std::mutex state_mutex_;
  wbc_msgs::msg::WbcState latest_state_;
  rclcpp::Time latest_state_recv_time_{0, 0, RCL_ROS_TIME};
  bool has_state_{false};
};

}  // namespace optimo_controller

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<optimo_controller::GainSweepNode>();
  const int rc = node->Run();
  rclcpp::shutdown();
  return rc;
}
