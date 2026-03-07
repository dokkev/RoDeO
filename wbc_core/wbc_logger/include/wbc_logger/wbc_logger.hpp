/**
 * @file wbc_core/wbc_logger/include/wbc_logger/wbc_logger.hpp
 * @brief WBC state logger: console print + ROS-free data snapshot.
 *
 * Two output paths:
 *   1. Console print at configurable rate (for quick debugging)
 *   2. WbcStateData snapshot (for RT-safe ROS publishing via controller)
 */
#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include <Eigen/Dense>

namespace wbc {

class Task;
class ForceTask;
struct WbcFormulation;

/// Per-task snapshot (ROS-free, fixed-size after init).
struct TaskStateData {
  std::string name;
  int dim{0};
  int priority{0};
  std::vector<double> x_des;
  std::vector<double> xdot_des;
  std::vector<double> x_curr;
  std::vector<double> x_err;
  std::vector<double> op_cmd;
  std::vector<double> kp;
  std::vector<double> kd;
  std::vector<double> weight;
  double x_err_norm{0.0};
};

/// QP solve diagnostics snapshot.
struct QpStateData {
  bool solved{false};
  int status{-1};
  int iter{0};
  double pri_res{0.0};
  double dua_res{0.0};
  double obj{0.0};
  double setup_time_us{0.0};
  double solve_time_us{0.0};
};

/// Full WBC pipeline snapshot (ROS-free, pre-allocated).
struct WbcStateData {
  int state_id{0};

  // Task references (_des)
  std::vector<double> q_des;
  std::vector<double> qdot_des;

  // Measured state
  std::vector<double> q_curr;
  std::vector<double> qdot_curr;

  // IK output (_cmd) from FindConfiguration
  std::vector<double> q_cmd;
  std::vector<double> qdot_cmd;
  std::vector<double> qddot_cmd;

  // Dynamics output from MakeTorque
  std::vector<double> tau_ff;
  std::vector<double> tau_fb;
  std::vector<double> tau;
  std::vector<double> gravity;

  // QP diagnostics (WBIC correction QP)
  bool qp_solved{false};
  int qp_status{-1};
  int qp_iter{0};
  double qp_pri_res{0.0};
  double qp_dua_res{0.0};
  double qp_obj{0.0};
  double qp_setup_time_us{0.0};
  double qp_solve_time_us{0.0};

  // Tracking performance (instantaneous)
  double joint_pos_err_norm{0.0};
  double joint_vel_err_norm{0.0};
  double joint_pos_err_max{0.0};
  double joint_vel_err_max{0.0};
  double tau_fb_norm{0.0};

  // Per-task details
  std::vector<TaskStateData> tasks;

  /// Pre-allocate all vectors. Call once at init.
  void Resize(int n_active, int n_qdot, int max_tasks);
};

class WbcLogger {
public:
  WbcLogger() = default;
  ~WbcLogger() = default;

  /// Call once after robot DOFs are known.
  void Initialize(int n_active, int n_qdot, int max_tasks = 8);

  /// Register task pointer -> name mapping (call at init, before logging).
  void RegisterTaskName(const Task* task, const std::string& name);
  void RegisterForceTaskName(const ForceTask* task, const std::string& name);

  /// Log one control tick. Updates internal snapshot and optionally prints.
  void LogTick(double time, int state_id,
               const Eigen::VectorXd& q_cmd,
               const Eigen::VectorXd& qdot_cmd,
               const Eigen::VectorXd& qddot_cmd,
               const Eigen::VectorXd& tau_ff,
               const Eigen::VectorXd& tau_fb,
               const Eigen::VectorXd& tau,
               const Eigen::VectorXd& q_curr,
               const Eigen::VectorXd& qdot_curr,
               const Eigen::VectorXd& gravity,
               const WbcFormulation& formulation,
               const QpStateData* qp_state = nullptr);

  /// Access the latest snapshot (for RT publisher).
  const WbcStateData& GetStateData() const { return state_data_; }

  bool enabled{false};
  double print_rate_hz{0.0};  // 0 = disabled (use ROS topic instead)
  double publish_rate_hz{100.0};

  /// Returns true if a new snapshot is ready for publishing.
  bool HasNewData() const { return has_new_data_; }
  void ClearNewData() { has_new_data_ = false; }

private:
  int n_active_{0};
  int n_qdot_{0};
  bool initialized_{false};
  double last_print_time_{-1e9};
  double last_publish_time_{-1e9};
  bool has_new_data_{false};

  WbcStateData state_data_;

  std::unordered_map<const Task*, std::string> task_names_;
  std::unordered_map<const ForceTask*, std::string> force_task_names_;
  static const std::string kUnknownTask;

  const std::string& LookupTaskName(const Task* task) const;

  void UpdateStateData(int state_id,
                       const Eigen::VectorXd& q_cmd,
                       const Eigen::VectorXd& qdot_cmd,
                       const Eigen::VectorXd& qddot_cmd,
                       const Eigen::VectorXd& tau_ff,
                       const Eigen::VectorXd& tau_fb,
                       const Eigen::VectorXd& tau,
                       const Eigen::VectorXd& q_curr,
                       const Eigen::VectorXd& qdot_curr,
                       const Eigen::VectorXd& gravity,
                       const WbcFormulation& formulation,
                       const QpStateData* qp_state);

  void PrintToConsole(double time, int state_id,
                      const Eigen::VectorXd& q_curr,
                      const Eigen::VectorXd& qdot_curr,
                      const Eigen::VectorXd& q_cmd,
                      const Eigen::VectorXd& qdot_cmd,
                      const Eigen::VectorXd& qddot_cmd,
                      const Eigen::VectorXd& tau_ff,
                      const Eigen::VectorXd& tau_fb,
                      const Eigen::VectorXd& tau,
                      const Eigen::VectorXd& gravity,
                      const WbcFormulation& formulation);
};

} // namespace wbc
