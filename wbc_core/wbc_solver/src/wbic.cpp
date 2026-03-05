/**
 * @file wbc_core/wbc_solver/src/wbic.cpp
 * @brief Doxygen documentation for wbic module.
 */
#include "wbc_solver/wbic.hpp"

#include <cassert>
#include <chrono>
#include <cmath>
#include <iterator>
#include <limits>

#include "wbc_formulation/kinematic_constraint.hpp"

namespace wbc {
namespace {
// Damped Least Squares (DLS) pseudo-inverse via LLT Cholesky.
// Reference: Siciliano et al., "Robotics: Modelling, Planning and Control".
//
//   J# = J^T (J J^T + λ²I)^{-1}
//
// Fixed damping λ² ensures (J J^T + λ²I) is always SPD, so LLT always
// succeeds. λ = 0.05 is small enough to not affect tracking far from
// singularity, but prevents inverse blowup near singularities.
constexpr double kDlsLambdaMax = 0.05;
constexpr double kDlsLambdaSq = kDlsLambdaMax * kDlsLambdaMax;
} // namespace

WBIC::WBIC(const std::vector<bool>& act_qdot_list, QPParams* qp_params)
    : WBC(act_qdot_list),

      Ni_dyn_(Eigen::MatrixXd::Identity(num_qdot_, num_qdot_)),
      N_pre_(Eigen::MatrixXd::Identity(num_qdot_, num_qdot_)),
      N_pre_dyn_(Eigen::MatrixXd::Identity(num_qdot_, num_qdot_)),
      N_nx_(Eigen::MatrixXd::Identity(num_qdot_, num_qdot_)),
      N_nx_dyn_(Eigen::MatrixXd::Identity(num_qdot_, num_qdot_)),
      Jc_bar_(Eigen::MatrixXd::Zero(num_qdot_, num_qdot_)),
      qddot_cmd_(Eigen::VectorXd::Zero(num_qdot_)),
      delta_q_cmd_(Eigen::VectorXd::Zero(num_qdot_)),
      qdot_cmd_(Eigen::VectorXd::Zero(num_qdot_)),
      prev_qddot_cmd_(Eigen::VectorXd::Zero(num_qdot_)),
      prev_delta_q_cmd_(Eigen::VectorXd::Zero(num_qdot_)),
      prev_qdot_cmd_(Eigen::VectorXd::Zero(num_qdot_)),
      trq_(Eigen::VectorXd::Zero(num_qdot_)),
      UNi_(Eigen::MatrixXd::Zero(num_active_, num_qdot_)),
      UNi_bar_(Eigen::MatrixXd::Zero(num_qdot_, num_active_)),
      wbic_data_(std::make_unique<WBICData>(num_qdot_, qp_params)),
      tau_0_(Eigen::VectorXd::Zero(num_qdot_)),
      l_bnd_(Eigen::VectorXd::Zero(num_qdot_)),
      u_bnd_(Eigen::VectorXd::Zero(num_qdot_)),
      sa_tau0_scratch_(Eigen::VectorXd::Zero(num_active_)) {
  wM_scratch_.resize(num_qdot_, num_qdot_);
}

void WBIC::ReserveCapacity(int max_contact_dim, int max_uf_rows) {
  // Contact stacking buffers (FindConfiguration)
  stacked_contact_jacobian_.resize(max_contact_dim, num_qdot_);
  stacked_contact_jdot_qdot_.resize(max_contact_dim);
  stacked_contact_op_cmd_.resize(max_contact_dim);

  // Contact matrices (MakeTorque)
  Jc_.resize(max_contact_dim, num_qdot_);
  JcDotQdot_.resize(max_contact_dim);
  Uf_mat_.resize(max_uf_rows, max_contact_dim);
  Uf_vec_.resize(max_uf_rows);
  des_rf_.resize(max_contact_dim);

  // QP cost scratch buffers
  wJc_scratch_.resize(max_contact_dim, num_qdot_);
  xc_res_scratch_.resize(max_contact_dim);
}

bool WBIC::FindConfiguration(
    const WbcFormulation& formulation, const Eigen::VectorXd& curr_jpos,
    Eigen::VectorXd& jpos_cmd, Eigen::VectorXd& jvel_cmd,
    Eigen::VectorXd& wbc_qddot_cmd) {
  if (!settings_updated_) {
    return false;
  }
  const auto t0_fc = std::chrono::high_resolution_clock::now();

  const std::vector<Task*>& task_vector = formulation.motion_tasks;
  const std::vector<Contact*>& contact_vector = formulation.contact_constraints;

  if (task_vector.empty()) {
    return false;
  }
  has_contact_ = !contact_vector.empty();

  Ni_dyn_.setIdentity();
  N_pre_.setIdentity();
  N_pre_dyn_.setIdentity();
  delta_q_cmd_.setZero();
  qdot_cmd_.setZero();
  prev_delta_q_cmd_.setZero();
  prev_qdot_cmd_.setZero();

  // Contact null-space: project all contacts into N_pre_, N_pre_dyn_
  // and compute initial qddot_cmd_ from contact constraints.
  InitContactProjection(contact_vector);
  prev_qddot_cmd_ = qddot_cmd_;

  // Hierarchical null-space task projection
  for (size_t i = 0; i < task_vector.size(); ++i) {
    Task* task = task_vector[i];
    const Eigen::MatrixXd& Jt = task->Jacobian();
    const Eigen::VectorXd& JtDotQdot = task->JacobianDotQdot();

    JtPre_.noalias() = Jt * N_pre_;
    PseudoInverse(JtPre_, JtPre_pinv_);
    JtPre_dyn_.noalias() = Jt * N_pre_dyn_;
    WeightedPseudoInverse(JtPre_dyn_, Minv_, JtPre_bar_);

    delta_q_cmd_ =
        prev_delta_q_cmd_ +
        JtPre_pinv_ * (task->KpIK().cwiseProduct(task->LocalPosError()) -
                        Jt * prev_delta_q_cmd_);
    qdot_cmd_ = prev_qdot_cmd_ +
                JtPre_pinv_ * (task->DesiredVel() - Jt * prev_qdot_cmd_);
    qddot_cmd_ =
        prev_qddot_cmd_ +
        JtPre_bar_ * (task->OpCommand() - JtDotQdot - Jt * prev_qddot_cmd_);

    prev_delta_q_cmd_ = delta_q_cmd_;
    prev_qdot_cmd_ = qdot_cmd_;
    prev_qddot_cmd_ = qddot_cmd_;

    if (i + 1 < task_vector.size()) {
      BuildProjectionMatrix(JtPre_, N_nx_);
      N_pre_ *= N_nx_;
      BuildProjectionMatrix(JtPre_dyn_, N_nx_dyn_, &Minv_);
      N_pre_dyn_ *= N_nx_dyn_;
    }
  }

  // Sanity check — propagated NaN from a degenerate Jacobian would
  // silently corrupt all downstream torque commands.
  if (!delta_q_cmd_.allFinite() || !qdot_cmd_.allFinite() ||
      !qddot_cmd_.allFinite()) {
    return false;
  }
  jpos_cmd = curr_jpos + delta_q_cmd_.tail(num_qdot_ - num_floating_);
  jvel_cmd = qdot_cmd_.tail(num_qdot_ - num_floating_);
  wbc_qddot_cmd = qddot_cmd_;

  if (enable_timing_) {
    const auto t1_fc = std::chrono::high_resolution_clock::now();
    timing_stats_.find_config_us =
        std::chrono::duration<double, std::micro>(t1_fc - t0_fc).count();
  }
  return true;
}

bool WBIC::MakeTorque(const WbcFormulation& formulation,
                      const Eigen::VectorXd& wbc_qddot_cmd,
                      Eigen::VectorXd& jtrq_cmd) {
  if (!settings_updated_) {
    return false;
  }
  if (!wbc_qddot_cmd.allFinite()) {
    return false;
  }

  const std::vector<ForceTask*>& force_task_vector = formulation.force_tasks;
  const std::vector<Contact*>& contact_constraints = formulation.contact_constraints;
  has_contact_ = !contact_constraints.empty();

  if (has_contact_) {
    BuildContactMtxVect(contact_constraints);
    GetDesiredReactionForce(force_task_vector);
    if (des_rf_.size() != dim_contact_) {
      return false;
    }
  } else {
    dim_contact_ = 0;
    des_rf_.resize(0);
  }

  // Cache typed constraint pointers (one dynamic_cast per type per MakeTorque call).
  // Reused by SetQPInEqualityConstraint without additional dynamic_cast.
  cached_pos_c_ = nullptr;
  cached_vel_c_ = nullptr;
  cached_trq_c_ = nullptr;
  for (const Constraint* c : formulation.kinematic_constraints) {
    if (!cached_pos_c_) {
      if (auto* p = dynamic_cast<const JointPosLimitConstraint*>(c)) { cached_pos_c_ = p; continue; }
    }
    if (!cached_vel_c_) {
      if (auto* v = dynamic_cast<const JointVelLimitConstraint*>(c)) { cached_vel_c_ = v; continue; }
    }
    if (!cached_trq_c_) {
      if (auto* t = dynamic_cast<const JointTrqLimitConstraint*>(c)) { cached_trq_c_ = t; continue; }
    }
  }
  dim_slack_pos_ = (cached_pos_c_ && soft_params_.pos) ? num_active_ : 0;
  dim_slack_vel_ = (cached_vel_c_ && soft_params_.vel) ? num_active_ : 0;
  dim_slack_trq_ = (cached_trq_c_ && soft_params_.trq) ? num_active_ : 0;
  dim_slack_total_ = dim_slack_pos_ + dim_slack_vel_ + dim_slack_trq_;

  // Always run QP — even without contacts, delta_qddot regularizes torque
  const auto t0_mt = std::chrono::high_resolution_clock::now();
  SetQPCost(wbc_qddot_cmd);
  SetQPEqualityConstraint(wbc_qddot_cmd);
  SetQPInEqualityConstraint(formulation, wbc_qddot_cmd);
  const auto t1_mt = std::chrono::high_resolution_clock::now();
  if (!SolveQP(wbc_qddot_cmd)) {
    return false;
  }
  const auto t2_mt = std::chrono::high_resolution_clock::now();

  GetSolution(wbc_qddot_cmd, jtrq_cmd);

  if (enable_timing_) {
    const auto t3_mt = std::chrono::high_resolution_clock::now();
    timing_stats_.qp_setup_us =
        std::chrono::duration<double, std::micro>(t1_mt - t0_mt).count();
    timing_stats_.qp_solve_us =
        std::chrono::duration<double, std::micro>(t2_mt - t1_mt).count();
    timing_stats_.torque_recovery_us =
        std::chrono::duration<double, std::micro>(t3_mt - t2_mt).count();
  }
  return true;
}

void WBIC::PseudoInverse(const Eigen::MatrixXd& jac, Eigen::MatrixXd& out) {
  if (jac.size() == 0) {
    out.resize(jac.cols(), jac.rows());
    out.setZero();
    return;
  }
  const int m = jac.rows();
  assert(m <= kMaxPInvDim && "PseudoInverse: jac.rows() exceeds kMaxPInvDim");
  // J J^T + λ²I  (m×m, always SPD with λ² > 0)
  // PInvSquare uses MaxRows/MaxCols → inline storage, no heap alloc on resize.
  JWJt_scratch_.resize(m, m);
  JWJt_scratch_.noalias() = jac * jac.transpose();
  JWJt_scratch_.diagonal().array() += kDlsLambdaSq;
  llt_scratch_.compute(JWJt_scratch_);
  // J# = J^T (J J^T + λ²I)^{-1}
  JWJt_pinv_scratch_.setIdentity(m, m);
  llt_scratch_.solveInPlace(JWJt_pinv_scratch_);
  out.noalias() = jac.transpose() * JWJt_pinv_scratch_;
}

void WBIC::WeightedPseudoInverse(const Eigen::MatrixXd& jac,
                                 const Eigen::MatrixXd& W,
                                 Eigen::MatrixXd& out) {
  if (jac.size() == 0) {
    out.resize(jac.cols(), jac.rows());
    out.setZero();
    return;
  }
  const int m = jac.rows();
  assert(m <= kMaxPInvDim && "WeightedPseudoInverse: jac.rows() exceeds kMaxPInvDim");
  // J W J^T + λ²I  (m×m, always SPD)
  // PInvSquare uses MaxRows/MaxCols → inline storage, no heap alloc on resize.
  JWJt_scratch_.resize(m, m);
  JWJt_scratch_.noalias() = jac * W * jac.transpose();
  JWJt_scratch_.diagonal().array() += kDlsLambdaSq;
  llt_scratch_.compute(JWJt_scratch_);
  // J_bar = W J^T (J W J^T + λ²I)^{-1}
  JWJt_pinv_scratch_.setIdentity(m, m);
  llt_scratch_.solveInPlace(JWJt_pinv_scratch_);
  out.noalias() = W * jac.transpose() * JWJt_pinv_scratch_;
}

void WBIC::BuildProjectionMatrix(const Eigen::MatrixXd& jac, Eigen::MatrixXd& N,
                                 const Eigen::MatrixXd* W) {
  if (jac.cols() == 0) {
    N.resize(0, 0);
    return;
  }
  if (W == nullptr) {
    PseudoInverse(jac, Jbar_scratch_);
  } else {
    WeightedPseudoInverse(jac, *W, Jbar_scratch_);
  }
  N.setIdentity(jac.cols(), jac.cols());
  N.noalias() -= Jbar_scratch_ * jac;
}

void WBIC::InitContactProjection(const std::vector<Contact*>& contacts) {
  if (contacts.empty()) {
    qddot_cmd_.setZero();
    return;
  }

  int total_dim = 0;
  for (const auto* c : contacts) {
    total_dim += c->Dim();
  }

  stacked_contact_jacobian_.resize(total_dim, num_qdot_);
  stacked_contact_jdot_qdot_.resize(total_dim);
  stacked_contact_op_cmd_.resize(total_dim);

  int row = 0;
  for (const auto* c : contacts) {
    const int dim = c->Dim();
    stacked_contact_jacobian_.block(row, 0, dim, num_qdot_) = c->Jacobian();
    stacked_contact_jdot_qdot_.segment(row, dim) = c->JacobianDotQdot();
    stacked_contact_op_cmd_.segment(row, dim) = c->OpCommand();
    row += dim;
  }

  BuildProjectionMatrix(stacked_contact_jacobian_, N_pre_);
  BuildProjectionMatrix(stacked_contact_jacobian_, N_pre_dyn_, &Minv_);
  WeightedPseudoInverse(stacked_contact_jacobian_, Minv_, Jc_bar_);
  qddot_cmd_ = Jc_bar_ * (stacked_contact_op_cmd_ - stacked_contact_jdot_qdot_);
}

void WBIC::BuildContactMtxVect(const std::vector<Contact*>& contacts) {
  int total_contact_dim = 0;
  int total_uf_rows = 0;
  for (const Contact* c : contacts) {
    total_contact_dim += c->Dim();
    total_uf_rows += c->UfMatrix().rows();
  }

  dim_contact_ = total_contact_dim;
  Jc_.resize(total_contact_dim, num_qdot_);
  JcDotQdot_.resize(total_contact_dim);
  Uf_mat_.resize(total_uf_rows, total_contact_dim);
  Uf_mat_.setZero();
  Uf_vec_.resize(total_uf_rows);
  wJc_scratch_.resize(total_contact_dim, num_qdot_);
  xc_res_scratch_.resize(total_contact_dim);

  int contact_row_offset = 0;
  int uf_row_offset = 0;
  for (const Contact* c : contacts) {
    const int dim = c->Dim();
    const int uf_rows = c->UfMatrix().rows();

    Jc_.block(contact_row_offset, 0, dim, num_qdot_) = c->Jacobian();
    JcDotQdot_.segment(contact_row_offset, dim) = c->JacobianDotQdot();
    Uf_mat_.block(uf_row_offset, contact_row_offset, uf_rows, dim) =
        c->UfMatrix();
    Uf_vec_.segment(uf_row_offset, uf_rows) = c->UfVector();

    contact_row_offset += dim;
    uf_row_offset += uf_rows;
  }
}

void WBIC::GetDesiredReactionForce(
    const std::vector<ForceTask*>& force_task_vector) {
  if (force_task_vector.empty()) {
    des_rf_.resize(dim_contact_);
    des_rf_.setZero();
    return;
  }

  int total_force_dim = 0;
  for (const ForceTask* force_task : force_task_vector) {
    total_force_dim += force_task->Dim();
  }

  des_rf_.resize(total_force_dim);
  int row_offset = 0;
  for (const ForceTask* force_task : force_task_vector) {
    const int dim = force_task->Dim();
    des_rf_.segment(row_offset, dim) = force_task->DesiredRf();
    row_offset += dim;
  }
}

void WBIC::ExtractBoxBounds(const Constraint* c,
                            const Eigen::VectorXd& wbc_qddot_cmd) {
  constexpr double kInf = std::numeric_limits<double>::infinity();
  l_bnd_.setConstant(-kInf);
  u_bnd_.setConstant(kInf);

  const Eigen::MatrixXd& A_kin = c->ConstraintMatrix();
  const Eigen::VectorXd& b_kin = c->ConstraintVector();

  for (int r = 0; r < A_kin.rows(); ++r) {
    int idx = -1;
    double sign = 0.0;
    for (int j = 0; j < A_kin.cols(); ++j) {
      if (std::abs(A_kin(r, j)) > 0.5) {
        idx = j;
        sign = A_kin(r, j);
        break;
      }
    }
    if (idx < 0) continue;

    if (sign > 0.0) {
      u_bnd_(idx) = std::min(u_bnd_(idx), b_kin(r) - wbc_qddot_cmd(idx));
    } else {
      l_bnd_(idx) = std::max(l_bnd_(idx), -b_kin(r) - wbc_qddot_cmd(idx));
    }
  }
}

int WBIC::BuildKinematicLimitConstraint(const Constraint* c, bool is_soft,
                                        bool use_box_solver,
                                        const Eigen::VectorXd& wbc_qddot_cmd,
                                        int row, int& slack_col) {
  ExtractBoxBounds(c, wbc_qddot_cmd);

  if (is_soft) {
    for (int i = 0; i < num_active_; ++i) {
      const int idx = num_floating_ + i;
      C_(row + i, idx) = 1.0;
      C_(row + i, slack_col + i) = -1.0;
      l_(row + i) = l_bnd_(idx);
      u_(row + i) = u_bnd_(idx);
    }
    slack_col += num_active_;
    return num_active_;
  } else if (!use_box_solver) {
    for (int i = 0; i < num_active_; ++i) {
      const int idx = num_floating_ + i;
      C_(row + i, idx) = 1.0;
      l_(row + i) = l_bnd_(idx);
      u_(row + i) = u_bnd_(idx);
    }
    return num_active_;
  } else {
    for (int i = 0; i < num_qdot_; ++i) {
      l_box_(i) = std::max(l_box_(i), l_bnd_(i));
      u_box_(i) = std::min(u_box_(i), u_bnd_(i));
    }
    return 0;
  }
}

int WBIC::BuildTorqueLimitConstraint(const JointTrqLimitConstraint* c, bool is_soft,
                                      bool use_box_solver,
                                      int row, int& slack_col) {
  const Eigen::MatrixXd& limits = c->EffectiveLimits();
  sa_tau0_scratch_.noalias() = sa_ * tau_0_;

  if (is_soft) {
    C_.block(row, 0, num_active_, num_qdot_) = sa_ * M_;
    if (dim_contact_ > 0) {
      C_.block(row, num_qdot_, num_active_, dim_contact_) =
          -sa_ * (Jc_ * Ni_dyn_).transpose();
    }
    for (int i = 0; i < num_active_; ++i) {
      C_(row + i, slack_col + i) = -1.0;
      l_(row + i) = limits(i, 0) - sa_tau0_scratch_(i);
      u_(row + i) = limits(i, 1) - sa_tau0_scratch_(i);
    }
    slack_col += num_active_;
    return num_active_;
  } else if (!use_box_solver) {
    // Full dynamics torque constraint in C matrix.
    C_.block(row, 0, num_active_, num_qdot_) = sa_ * M_;
    if (dim_contact_ > 0) {
      C_.block(row, num_qdot_, num_active_, dim_contact_) =
          -sa_ * (Jc_ * Ni_dyn_).transpose();
    }
    for (int i = 0; i < num_active_; ++i) {
      l_(row + i) = limits(i, 0) - sa_tau0_scratch_(i);
      u_(row + i) = limits(i, 1) - sa_tau0_scratch_(i);
    }
    return num_active_;
  } else {
    // Diagonal mass approximation as box bounds.
    for (int i = 0; i < num_active_; ++i) {
      const int idx = num_floating_ + i;
      const double M_diag = M_(idx, idx);
      if (M_diag < 1e-10) continue;
      l_box_(idx) = std::max(l_box_(idx), (limits(i, 0) - sa_tau0_scratch_(i)) / M_diag);
      u_box_(idx) = std::min(u_box_(idx), (limits(i, 1) - sa_tau0_scratch_(i)) / M_diag);
    }
    return 0;
  }
}

int WBIC::BuildFrictionConeConstraint(int row) {
  const int n_fric = Uf_mat_.rows();
  C_.block(row, num_qdot_, n_fric, dim_contact_) = Uf_mat_;
  l_.segment(row, n_fric) = Uf_vec_ - Uf_mat_ * des_rf_;
  return n_fric;
}

void WBIC::EnforceBoxFeasibilityGuard(int qp_dim) {
  for (int i = 0; i < qp_dim; ++i) {
    if (l_box_(i) > u_box_(i)) {
      l_box_(i) = u_box_(i);
    }
  }
}

void WBIC::SetQPCost(const Eigen::VectorXd& wbc_qddot_cmd) {
  const int qp_dim = num_qdot_ + dim_contact_ + dim_slack_total_;
  H_.setZero(qp_dim, qp_dim);
  g_.setZero(qp_dim);

  AddQddotTrackingCost();
  AddTorqueMinimizationCost(wbc_qddot_cmd);

  if (has_contact_ && dim_contact_ > 0) {
    AddContactAccelerationCost(wbc_qddot_cmd);
    AddReactionForceCost();
  }

  if (dim_slack_total_ > 0) {
    AddSlackVariablePenalties();
  }
}

void WBIC::AddQddotTrackingCost() {
  const auto& w = wbic_data_->qp_params_->W_delta_qddot_;
  if (w.size() == num_qdot_) {
    H_.diagonal().head(num_qdot_) += w;
  } else {
    H_.diagonal().head(num_qdot_).array() += 1.0;
  }
}

void WBIC::AddTorqueMinimizationCost(const Eigen::VectorXd& wbc_qddot_cmd) {
  // Baseline torque: tau_0 = M * qddot_ik + Ni_dyn^T * (cori + grav) - (Jc * Ni_dyn)^T * des_rf
  tau_0_.noalias() = M_ * wbc_qddot_cmd;
  tau_0_.noalias() += Ni_dyn_.transpose() * (cori_ + grav_);
  if (has_contact_ && des_rf_.size() > 0) {
    tau_0_ -= (Jc_ * Ni_dyn_).transpose() * des_rf_;
  }

  const auto& w_tau = wbic_data_->qp_params_->W_tau_;
  const auto& w_tau_dot = wbic_data_->qp_params_->W_tau_dot_;

  if (w_tau.squaredNorm() > 0.0) {
    wM_scratch_.noalias() = w_tau.asDiagonal() * M_;
    H_.topLeftCorner(num_qdot_, num_qdot_).noalias() += M_.transpose() * wM_scratch_;
    trq_ = w_tau.cwiseProduct(tau_0_);
    g_.head(num_qdot_).noalias() += M_.transpose() * trq_;
  }
  if (w_tau_dot.squaredNorm() > 0.0) {
    wM_scratch_.noalias() = w_tau_dot.asDiagonal() * M_;
    H_.topLeftCorner(num_qdot_, num_qdot_).noalias() += M_.transpose() * wM_scratch_;
    trq_ = tau_0_ - wbic_data_->tau_prev_;
    trq_.array() *= w_tau_dot.array();
    g_.head(num_qdot_).noalias() += M_.transpose() * trq_;
  }
}

void WBIC::AddContactAccelerationCost(const Eigen::VectorXd& wbc_qddot_cmd) {
  const auto& w_xc = wbic_data_->qp_params_->W_xc_ddot_;
  if (w_xc.size() == dim_contact_) {
    wJc_scratch_.noalias() = w_xc.asDiagonal() * Jc_;
  } else {
    wJc_scratch_ = Jc_;
  }
  H_.topLeftCorner(num_qdot_, num_qdot_).noalias() += Jc_.transpose() * wJc_scratch_;

  xc_res_scratch_.noalias() = Jc_ * wbc_qddot_cmd;
  xc_res_scratch_ += JcDotQdot_;
  if (w_xc.size() == dim_contact_) {
    xc_res_scratch_.array() *= w_xc.array();
  }
  g_.head(num_qdot_).noalias() += Jc_.transpose() * xc_res_scratch_;
}

void WBIC::AddReactionForceCost() {
  const auto& w_rf = wbic_data_->qp_params_->W_delta_rf_;
  const auto& w_fd = wbic_data_->qp_params_->W_f_dot_;

  auto rf_diag = H_.diagonal().segment(num_qdot_, dim_contact_);
  if (w_rf.size() == dim_contact_) rf_diag += w_rf;
  else                             rf_diag.array() += 1.0;
  if (w_fd.size() == dim_contact_) rf_diag += w_fd;
  else                             rf_diag.array() += 1.0;

  if (w_fd.size() == dim_contact_) {
    g_.segment(num_qdot_, dim_contact_) += w_fd.cwiseProduct(des_rf_ - wbic_data_->rf_prev_cmd_);
  } else {
    g_.segment(num_qdot_, dim_contact_) += des_rf_ - wbic_data_->rf_prev_cmd_;
  }
}

void WBIC::AddSlackVariablePenalties() {
  int slack_idx = num_qdot_ + dim_contact_;
  if (dim_slack_pos_ > 0) {
    H_.diagonal().segment(slack_idx, dim_slack_pos_).setConstant(soft_params_.w_pos);
    slack_idx += dim_slack_pos_;
  }
  if (dim_slack_vel_ > 0) {
    H_.diagonal().segment(slack_idx, dim_slack_vel_).setConstant(soft_params_.w_vel);
    slack_idx += dim_slack_vel_;
  }
  if (dim_slack_trq_ > 0) {
    H_.diagonal().segment(slack_idx, dim_slack_trq_).setConstant(soft_params_.w_trq);
  }
}

void WBIC::SetQPEqualityConstraint(const Eigen::VectorXd& wbc_qddot_cmd) {
  const int qp_dim = num_qdot_ + dim_contact_ + dim_slack_total_;

  A_.setZero(num_floating_, qp_dim);
  if (num_floating_ > 0) {
    // sf_ * M picks the floating-base rows of M (all num_qdot_ columns)
    A_.leftCols(num_qdot_) = sf_ * M_;
    if (dim_contact_ > 0) {
      A_.block(0, num_qdot_, num_floating_, dim_contact_) = -sf_ * Jc_.transpose();
    }
    b_ = sf_ * (Jc_.transpose() * des_rf_ - M_ * wbc_qddot_cmd - cori_ - grav_);
  } else {
    // Fixed-base: no equality constraints (0 rows)
    b_.resize(0);
  }
}

void WBIC::SetQPInEqualityConstraint(const WbcFormulation& /*formulation*/,
                                     const Eigen::VectorXd& wbc_qddot_cmd) {
  const int qp_dim = num_qdot_ + dim_contact_ + dim_slack_total_;
  const int n_friction = has_contact_ ? Uf_mat_.rows() : 0;
  constexpr double kInf = std::numeric_limits<double>::infinity();

  // Use constraint pointers cached in MakeTorque (no dynamic_cast here).
  const JointPosLimitConstraint* pos_c = cached_pos_c_;
  const JointVelLimitConstraint* vel_c = cached_vel_c_;
  const JointTrqLimitConstraint* trq_c = cached_trq_c_;

  const bool has_any_hard =
      (pos_c && !soft_params_.pos) ||
      (vel_c && !soft_params_.vel) ||
      (trq_c && !soft_params_.trq);

  // When slack (soft) variables exist, disable box solver to avoid
  // ProxQP ADMM scaling collapse between huge box bounds and unbounded slacks.
  const bool force_dense = (dim_slack_total_ > 0);
  const bool use_box_solver = has_any_hard && !force_dense;

  // --- Count general inequality rows ---
  int n_ineq = n_friction;
  if (pos_c && (soft_params_.pos || !use_box_solver)) n_ineq += num_active_;
  if (vel_c && (soft_params_.vel || !use_box_solver)) n_ineq += num_active_;
  if (trq_c && (soft_params_.trq || !use_box_solver)) n_ineq += num_active_;

  // --- Box constraints ---
  if (use_box_solver) {
    l_box_.resize(qp_dim);
    u_box_.resize(qp_dim);
    l_box_.setConstant(-kInf);
    u_box_.setConstant(kInf);
  } else {
    l_box_.resize(0);
    u_box_.resize(0);
  }

  // --- General inequality matrix ---
  C_.setZero(n_ineq, qp_dim);
  l_.setZero(n_ineq);
  u_.resize(n_ineq);
  u_.setConstant(kInf);

  int row = 0;
  int slack_col = num_qdot_ + dim_contact_;

  if (n_friction > 0)
    row += BuildFrictionConeConstraint(row);
  if (pos_c)
    row += BuildKinematicLimitConstraint(pos_c, soft_params_.pos, use_box_solver, wbc_qddot_cmd, row, slack_col);
  if (vel_c)
    row += BuildKinematicLimitConstraint(vel_c, soft_params_.vel, use_box_solver, wbc_qddot_cmd, row, slack_col);
  if (trq_c)
    row += BuildTorqueLimitConstraint(trq_c, soft_params_.trq, use_box_solver, row, slack_col);

  if (use_box_solver)
    EnforceBoxFeasibilityGuard(qp_dim);
}

bool WBIC::SolveQP(const Eigen::VectorXd& wbc_qddot_cmd) {
  const int dim = num_qdot_ + dim_contact_ + dim_slack_total_;
  const int n_eq = A_.rows();
  const int n_ineq = C_.rows();

  const bool use_box = (l_box_.size() == dim);

  // Lazy init / re-init on dimension change or box mode change
  if (!qp_solver_ ||
      qp_solver_->model.dim != dim ||
      qp_solver_->model.n_eq != n_eq ||
      qp_solver_->model.n_in != n_ineq ||
      qp_solver_->is_box_constrained() != use_box) {
    qp_solver_ = std::make_unique<proxsuite::proxqp::dense::QP<double>>(
        dim, n_eq, n_ineq, use_box);
    qp_solver_->settings.eps_abs = 1e-3;
    qp_solver_->settings.verbose = false;
    qp_solver_->settings.max_iter = 1000;
    qp_solver_->settings.initial_guess =
        proxsuite::proxqp::InitialGuessStatus::WARM_START_WITH_PREVIOUS_RESULT;
    if (use_box) {
      qp_solver_->init(H_, g_, A_, b_, C_, l_, u_, l_box_, u_box_);
    } else {
      qp_solver_->init(H_, g_, A_, b_, C_, l_, std::nullopt);
    }
  } else {
    if (use_box) {
      qp_solver_->update(H_, g_, A_, b_, C_, l_, u_, l_box_, u_box_);
    } else {
      qp_solver_->update(H_, g_, A_, b_, C_, l_, std::nullopt);
    }
  }

  qp_solver_->solve();

  if (qp_solver_->results.info.status !=
      proxsuite::proxqp::QPSolverOutput::PROXQP_SOLVED) {
    return false;
  }

  const Eigen::VectorXd& qp_sol = qp_solver_->results.x;
  if (!qp_sol.allFinite()) {
    return false;
  }

  // Extract solution
  wbic_data_->delta_qddot_ = qp_sol.head(num_qdot_);
  wbic_data_->corrected_wbc_qddot_cmd_ = wbc_qddot_cmd + wbic_data_->delta_qddot_;

  if (dim_contact_ > 0) {
    wbic_data_->delta_rf_ = qp_sol.segment(num_qdot_, dim_contact_);
    wbic_data_->rf_cmd_.noalias() = des_rf_ + wbic_data_->delta_rf_;
    wbic_data_->rf_prev_cmd_ = wbic_data_->rf_cmd_;
    wbic_data_->Xc_ddot_.noalias() = Jc_ * wbic_data_->corrected_wbc_qddot_cmd_;
    wbic_data_->Xc_ddot_ += JcDotQdot_;
  } else {
    wbic_data_->delta_rf_.resize(0);
    wbic_data_->rf_cmd_.resize(0);
    wbic_data_->Xc_ddot_.resize(0);
  }

  // Diagnostic costs — only computed in Debug builds (skipped in Release).
  // Uses x^T * diag(w) * x = w.cwiseProduct(x).dot(x), avoiding NxN matrix allocation.
#ifndef NDEBUG
  {
    const auto& w = wbic_data_->qp_params_->W_delta_qddot_;
    const auto& dx = wbic_data_->delta_qddot_;
    wbic_data_->delta_qddot_cost_ =
        (w.size() == num_qdot_) ? w.cwiseProduct(dx).dot(dx) : dx.squaredNorm();
  }
  if (dim_contact_ > 0) {
    const auto& w_rf = wbic_data_->qp_params_->W_delta_rf_;
    const auto& drf = wbic_data_->delta_rf_;
    wbic_data_->delta_rf_cost_ =
        (w_rf.size() == dim_contact_) ? w_rf.cwiseProduct(drf).dot(drf) : drf.squaredNorm();
    const auto& w_xc = wbic_data_->qp_params_->W_xc_ddot_;
    const auto& xc = wbic_data_->Xc_ddot_;
    wbic_data_->Xc_ddot_cost_ =
        (w_xc.size() == dim_contact_) ? w_xc.cwiseProduct(xc).dot(xc) : xc.squaredNorm();
  } else {
    wbic_data_->delta_rf_cost_ = 0.0;
    wbic_data_->Xc_ddot_cost_ = 0.0;
  }
#endif

  return true;
}

void WBIC::GetSolution(const Eigen::VectorXd& /*wbc_qddot_cmd*/,
                       Eigen::VectorXd& jtrq_cmd) {
  // Always use QP-corrected qddot (delta_qddot applied to all DOFs)
  trq_ = M_ * wbic_data_->corrected_wbc_qddot_cmd_ +
         Ni_dyn_.transpose() * (cori_ + grav_);
  if (has_contact_ && wbic_data_->rf_cmd_.size() > 0) {
    trq_ -= (Jc_ * Ni_dyn_).transpose() * wbic_data_->rf_cmd_;
  }

  // Store full-DOF torque for next-tick rate-of-change cost.
  wbic_data_->tau_prev_ = trq_;

  UNi_.noalias() = sa_ * Ni_dyn_;
  WeightedPseudoInverse(UNi_, Minv_, UNi_bar_);
  jtrq_cmd = UNi_bar_.transpose() * trq_;
  jtrq_cmd = snf_ * sa_.transpose() * jtrq_cmd;
}

} // namespace wbc
