/**
 * @file wbc_core/wbc_logger/src/wbc_logger.cpp
 */
#include "wbc_logger/wbc_logger.hpp"

#include <algorithm>
#include <iomanip>
#include <iostream>

#include "wbc_formulation/interface/task.hpp"
#include "wbc_formulation/wbc_formulation.hpp"

namespace wbc {

// --- WbcStateData ---

void WbcStateData::Resize(int n_active, int n_qdot, int max_tasks) {
  (void)n_qdot;
  q_des.resize(n_active, 0.0);
  qdot_des.resize(n_active, 0.0);
  q_curr.resize(n_active, 0.0);
  qdot_curr.resize(n_active, 0.0);
  q_cmd.resize(n_active, 0.0);
  qdot_cmd.resize(n_active, 0.0);
  qddot_cmd.resize(n_active, 0.0);
  tau_ff.resize(n_active, 0.0);
  tau_fb.resize(n_active, 0.0);
  tau.resize(n_active, 0.0);
  gravity.resize(n_active, 0.0);
  tasks.reserve(max_tasks);
}

// --- WbcLogger ---

const std::string WbcLogger::kUnknownTask = "unknown";

void WbcLogger::Initialize(int n_active, int n_qdot, int max_tasks) {
  n_active_ = n_active;
  n_qdot_ = n_qdot;
  state_data_.Resize(n_active, n_qdot, max_tasks);
  initialized_ = true;
}

void WbcLogger::RegisterTaskName(const Task* task, const std::string& name) {
  task_names_[task] = name;
}

void WbcLogger::RegisterForceTaskName(const ForceTask* task,
                                       const std::string& name) {
  force_task_names_[task] = name;
}

const std::string& WbcLogger::LookupTaskName(const Task* task) const {
  auto it = task_names_.find(task);
  return (it != task_names_.end()) ? it->second : kUnknownTask;
}

// Helper: copy Eigen vector to std::vector without realloc (sizes must match).
static void EigenToStd(const Eigen::VectorXd& src, std::vector<double>& dst) {
  const int n = static_cast<int>(dst.size());
  const int copy_n = std::min(n, static_cast<int>(src.size()));
  for (int i = 0; i < copy_n; ++i) dst[i] = src[i];
}

template <typename Fn>
void ForEachMotionTask(const WbcFormulation& formulation, Fn&& fn) {
  for (const Task* task : formulation.operational_tasks) {
    fn(task);
  }
  for (const Task* task : formulation.posture_tasks) {
    fn(task);
  }
}

void WbcLogger::UpdateStateData(
    int state_id,
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
    const QpStateData* qp_state) {

  state_data_.state_id = state_id;
  EigenToStd(q_cmd,     state_data_.q_cmd);
  EigenToStd(qdot_cmd,  state_data_.qdot_cmd);
  EigenToStd(qddot_cmd, state_data_.qddot_cmd);
  EigenToStd(tau_ff,    state_data_.tau_ff);
  EigenToStd(tau_fb,    state_data_.tau_fb);
  EigenToStd(tau,       state_data_.tau);
  EigenToStd(q_curr,    state_data_.q_curr);
  EigenToStd(qdot_curr, state_data_.qdot_curr);
  EigenToStd(gravity,   state_data_.gravity);

  if (qp_state != nullptr) {
    state_data_.qp_solved = qp_state->solved;
    state_data_.qp_status = qp_state->status;
    state_data_.qp_iter = qp_state->iter;
    state_data_.qp_pri_res = qp_state->pri_res;
    state_data_.qp_dua_res = qp_state->dua_res;
    state_data_.qp_obj = qp_state->obj;
    state_data_.qp_setup_time_us = qp_state->setup_time_us;
    state_data_.qp_solve_time_us = qp_state->solve_time_us;
  } else {
    state_data_.qp_solved = false;
    state_data_.qp_status = -1;
    state_data_.qp_iter = 0;
    state_data_.qp_pri_res = 0.0;
    state_data_.qp_dua_res = 0.0;
    state_data_.qp_obj = 0.0;
    state_data_.qp_setup_time_us = 0.0;
    state_data_.qp_solve_time_us = 0.0;
  }

  const Eigen::VectorXd q_err = q_cmd - q_curr;
  const Eigen::VectorXd qdot_err = qdot_cmd - qdot_curr;
  state_data_.joint_pos_err_norm = q_err.norm();
  state_data_.joint_vel_err_norm = qdot_err.norm();
  state_data_.joint_pos_err_max =
      (q_err.size() > 0) ? q_err.cwiseAbs().maxCoeff() : 0.0;
  state_data_.joint_vel_err_max =
      (qdot_err.size() > 0) ? qdot_err.cwiseAbs().maxCoeff() : 0.0;
  state_data_.tau_fb_norm = tau_fb.norm();

  // Extract q_des/qdot_des from JointTask (dim == n_active).
  bool found_joint_task = false;
  ForEachMotionTask(formulation, [&](const Task* task) {
    if (found_joint_task || task == nullptr) {
      return;
    }
    if (task->Dim() == n_active_) {
      EigenToStd(task->DesiredPos(), state_data_.q_des);
      EigenToStd(task->DesiredVel(), state_data_.qdot_des);
      found_joint_task = true;
    }
  });

  // Per-task snapshots — reuse allocated TaskStateData entries.
  const std::size_t num_tasks =
      formulation.operational_tasks.size() + formulation.posture_tasks.size();
  state_data_.tasks.resize(num_tasks);
  std::size_t i = 0;
  ForEachMotionTask(formulation, [&](const Task* task) {
    if (task == nullptr) return;
    auto& td = state_data_.tasks[i];
    td.name = LookupTaskName(task);
    td.dim = task->Dim();

    // Resize only if dimension changed (first tick or state change).
    if (static_cast<int>(td.x_des.size()) != td.dim) {
      td.x_des.resize(td.dim);
      td.xdot_des.resize(td.dim);
      td.x_curr.resize(td.dim);
      td.x_err.resize(td.dim);
      td.op_cmd.resize(td.dim);
      td.kp.resize(td.dim);
      td.kd.resize(td.dim);
      td.weight.resize(td.dim);
    }

    EigenToStd(task->DesiredPos(), td.x_des);
    EigenToStd(task->DesiredVel(), td.xdot_des);
    EigenToStd(task->CurrentPos(), td.x_curr);
    EigenToStd(task->PosError(),   td.x_err);
    EigenToStd(task->OpCommand(),  td.op_cmd);
    EigenToStd(task->Kp(),         td.kp);
    EigenToStd(task->Kd(),         td.kd);
    EigenToStd(task->Weight(),     td.weight);
    td.x_err_norm = task->PosError().norm();
    ++i;
  });
  state_data_.tasks.resize(i);
}

// Eigen vector -> compact string: [v0, v1, v2, ...]
static std::string VecStr(const Eigen::VectorXd& v, int precision = 4) {
  std::ostringstream os;
  os << std::fixed << std::setprecision(precision) << "[";
  for (int i = 0; i < v.size(); ++i) {
    if (i > 0) os << ", ";
    os << v(i);
  }
  os << "]";
  return os.str();
}

void WbcLogger::PrintToConsole(
    double time, int state_id,
    const Eigen::VectorXd& q_curr,
    const Eigen::VectorXd& qdot_curr,
    const Eigen::VectorXd& q_cmd,
    const Eigen::VectorXd& qdot_cmd,
    const Eigen::VectorXd& qddot_cmd,
    const Eigen::VectorXd& tau_ff,
    const Eigen::VectorXd& tau_fb,
    const Eigen::VectorXd& tau,
    const Eigen::VectorXd& gravity,
    const WbcFormulation& formulation) {

  std::cout << "\n========== WBC Logger [t=" << std::fixed
            << std::setprecision(3) << time
            << " state=" << state_id << "] ==========\n";

  // Task references from JointTask
  bool found_joint_task = false;
  ForEachMotionTask(formulation, [&](const Task* task) {
    if (found_joint_task || task == nullptr) {
      return;
    }
    if (task->Dim() == n_active_) {
      std::cout << "  q_des:      " << VecStr(task->DesiredPos()) << "\n";
      std::cout << "  qdot_des:   " << VecStr(task->DesiredVel()) << "\n";
      found_joint_task = true;
    }
  });

  std::cout << "  q_curr:     " << VecStr(q_curr) << "\n";
  std::cout << "  qdot_curr:  " << VecStr(qdot_curr) << "\n";
  std::cout << "  q_cmd:      " << VecStr(q_cmd) << "\n";
  std::cout << "  qdot_cmd:   " << VecStr(qdot_cmd) << "\n";
  std::cout << "  qddot_cmd:  " << VecStr(qddot_cmd) << "\n";
  std::cout << "  tau_ff:     " << VecStr(tau_ff) << "\n";
  std::cout << "  tau_fb:     " << VecStr(tau_fb) << "\n";
  std::cout << "  tau:        " << VecStr(tau) << "\n";
  std::cout << "  gravity:    " << VecStr(gravity) << "\n";

  ForEachMotionTask(formulation, [&](const Task* task) {
    if (task == nullptr) return;
    const std::string& name = LookupTaskName(task);
    std::cout << "  --- task: " << name << " (dim=" << task->Dim() << ") ---\n";
    std::cout << "    x_des:    " << VecStr(task->DesiredPos()) << "\n";
    std::cout << "    x_curr:   " << VecStr(task->CurrentPos()) << "\n";
    std::cout << "    x_err:    " << VecStr(task->PosError()) << "\n";
    std::cout << "    xdot_des: " << VecStr(task->DesiredVel()) << "\n";
    std::cout << "    op_cmd:   " << VecStr(task->OpCommand()) << "\n";
  });

  std::cout << std::flush;
}

void WbcLogger::LogTick(double time, int state_id,
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
                         const QpStateData* qp_state) {
  if (!enabled || !initialized_) return;

  // Always update state data for publisher (decimated by publish_rate_hz).
  const double pub_interval = (publish_rate_hz > 0.0) ? (1.0 / publish_rate_hz) : 0.01;
  if (time - last_publish_time_ >= pub_interval) {
    UpdateStateData(state_id, q_cmd, qdot_cmd, qddot_cmd,
                    tau_ff, tau_fb, tau, q_curr, qdot_curr,
                    gravity, formulation, qp_state);
    last_publish_time_ = time;
    has_new_data_ = true;
  }

  // Console print at lower rate (disabled when print_rate_hz <= 0).
  if (print_rate_hz > 0.0) {
    const double print_interval = 1.0 / print_rate_hz;
    if (time - last_print_time_ >= print_interval) {
      PrintToConsole(time, state_id, q_curr, qdot_curr,
                     q_cmd, qdot_cmd, qddot_cmd,
                     tau_ff, tau_fb, tau, gravity, formulation);
      last_print_time_ = time;
    }
  }
}

} // namespace wbc
