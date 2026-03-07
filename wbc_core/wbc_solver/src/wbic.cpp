/**
 * @file wbc_core/wbc_solver/src/wbic.cpp
 * @brief Doxygen documentation for wbic module.
 */
#include "wbc_solver/wbic.hpp"

#include <algorithm>
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
      qddot_cmd_(Eigen::VectorXd::Zero(num_qdot_)),
      delta_q_cmd_(Eigen::VectorXd::Zero(num_qdot_)),
      qdot_cmd_(Eigen::VectorXd::Zero(num_qdot_)),
      trq_(Eigen::VectorXd::Zero(num_qdot_)),
      UNi_(Eigen::MatrixXd::Zero(num_active_, num_qdot_)),
      UNi_bar_(Eigen::MatrixXd::Zero(num_qdot_, num_active_)),
      wbic_data_(std::make_unique<WBICData>(num_qdot_, qp_params)),
      tau_0_(Eigen::VectorXd::Zero(num_qdot_)),
      l_bnd_(Eigen::VectorXd::Zero(num_qdot_)),
      u_bnd_(Eigen::VectorXd::Zero(num_qdot_)),
      sa_tau0_scratch_(Eigen::VectorXd::Zero(num_active_)),
      zero_qddot_scratch_(Eigen::VectorXd::Zero(num_qdot_)),
      H_ik_(Eigen::MatrixXd::Zero(num_qdot_, num_qdot_)),
      g_ik_pos_(Eigen::VectorXd::Zero(num_qdot_)),
      g_ik_vel_(Eigen::VectorXd::Zero(num_qdot_)),
      g_ik_acc_(Eigen::VectorXd::Zero(num_qdot_)),
      l_box_ik_(Eigen::VectorXd::Zero(num_qdot_)),
      u_box_ik_(Eigen::VectorXd::Zero(num_qdot_)) {
  wM_scratch_.resize(num_qdot_, num_qdot_);
}

void WBIC::ReserveCapacity(int max_contact_dim, int max_uf_rows) {
  // Contact matrices (MakeTorque)
  Jc_.resize(max_contact_dim, num_qdot_);
  JcDotQdot_.resize(max_contact_dim);
  Uf_mat_.resize(max_uf_rows, max_contact_dim);
  Uf_vec_.resize(max_uf_rows);
  des_rf_.resize(max_contact_dim);

  // QP cost scratch buffers
  wJc_scratch_.resize(max_contact_dim, num_qdot_);
  xc_res_scratch_.resize(max_contact_dim);

  // Pre-allocate QP matrices to worst-case dimensions to avoid
  // per-tick heap allocation when contact/slack configurations change.
  const int max_slack = 3 * num_active_;  // pos + vel + trq soft limits
  max_qp_dim_ = num_qdot_ + max_contact_dim + max_slack;
  max_n_eq_ = num_floating_;
  max_n_ineq_ = max_uf_rows + 3 * num_active_;

  H_.resize(max_qp_dim_, max_qp_dim_);
  g_.resize(max_qp_dim_);
  A_.resize(max_n_eq_, max_qp_dim_);
  b_.resize(max_n_eq_);
  C_.resize(max_n_ineq_, max_qp_dim_);
  l_.resize(max_n_ineq_);
  u_.resize(max_n_ineq_);
  l_box_.resize(max_qp_dim_);
  u_box_.resize(max_qp_dim_);

  // Pre-allocate LU factorization for fully-actuated torque recovery.
  lu_scratch_.compute(Eigen::MatrixXd::Identity(num_active_, num_active_));

  // Pre-allocate ProxQP solvers so the first tick doesn't heap-allocate.
  // MakeTorque solver: pre-created at max dims. Re-created only on rare
  // dimension changes (state transitions). Init with dummy data so first
  // hot-path call can use update() if dims match.
  qp_solver_ = std::make_unique<proxsuite::proxqp::dense::QP<double>>(
      max_qp_dim_, max_n_eq_, max_n_ineq_, true);
  qp_solver_->settings.eps_abs = 1e-3;
  qp_solver_->settings.verbose = false;
  qp_solver_->settings.max_iter = 1000;
  qp_solver_->settings.initial_guess =
      proxsuite::proxqp::InitialGuessStatus::WARM_START_WITH_PREVIOUS_RESULT;
  {
    H_.setZero(max_qp_dim_, max_qp_dim_);
    for (int i = 0; i < max_qp_dim_; ++i) H_(i, i) = 1.0;
    g_.setZero(max_qp_dim_);
    A_.setZero(max_n_eq_, max_qp_dim_);
    b_.setZero(max_n_eq_);
    C_.setZero(max_n_ineq_, max_qp_dim_);
    l_.setZero(max_n_ineq_);
    u_.setZero(max_n_ineq_);
    l_box_.setConstant(max_qp_dim_, -1e30);
    u_box_.setConstant(max_qp_dim_,  1e30);
    qp_solver_->init(H_, g_, A_, b_, C_, l_, u_, l_box_, u_box_);
  }

  // Optional pre-allocation for IK solver; FindConfiguration can also lazy-init.
  EnsureIkQPSolverInitialized();
}

void WBIC::EnsureIkQPSolverInitialized() {
  if (qp_ik_solver_) {
    return;
  }
  qp_ik_solver_ = std::make_unique<proxsuite::proxqp::dense::QP<double>>(
      num_qdot_, 0, 0, true);
  qp_ik_solver_->settings.eps_abs = 1e-3;
  qp_ik_solver_->settings.verbose = false;
  qp_ik_solver_->settings.max_iter = 1000;
  qp_ik_solver_->settings.initial_guess =
      proxsuite::proxqp::InitialGuessStatus::WARM_START_WITH_PREVIOUS_RESULT;

  Eigen::MatrixXd H_init = Eigen::MatrixXd::Identity(num_qdot_, num_qdot_);
  Eigen::VectorXd g_init = Eigen::VectorXd::Zero(num_qdot_);
  Eigen::VectorXd lb = Eigen::VectorXd::Constant(num_qdot_, -1e30);
  Eigen::VectorXd ub = Eigen::VectorXd::Constant(num_qdot_, 1e30);
  qp_ik_solver_->init(H_init, g_init, std::nullopt, std::nullopt,
                      std::nullopt, std::nullopt, std::nullopt, lb, ub);
}

bool WBIC::FindConfiguration(
    const WbcFormulation& formulation, const Eigen::VectorXd& curr_jpos,
    Eigen::VectorXd& jpos_cmd, Eigen::VectorXd& jvel_cmd,
    Eigen::VectorXd& wbc_qddot_cmd) {
  // Weighted-QP is the only supported IK path.
  (void)ik_method_;  // kept for source compatibility with legacy configs/tests.
  return FindConfigurationWeightedQP(formulation, curr_jpos, jpos_cmd,
                                     jvel_cmd, wbc_qddot_cmd);
}

////////////////////////////////////////////////////////////////////////////////
// Weighted QP IK: replaces hierarchical null-space with a single weighted
// least-squares solve. Task weights determine effective priority.
// q_cmd and qdot_cmd use LLT (unconstrained); qddot_cmd uses ProxQP with
// joint-limit box constraints for safety.
////////////////////////////////////////////////////////////////////////////////
bool WBIC::FindConfigurationWeightedQP(
    const WbcFormulation& formulation, const Eigen::VectorXd& curr_jpos,
    Eigen::VectorXd& jpos_cmd, Eigen::VectorXd& jvel_cmd,
    Eigen::VectorXd& wbc_qddot_cmd) {
  if (!settings_updated_) return false;
  const auto t0_fc = std::chrono::steady_clock::now();

  const auto& task_vector = formulation.motion_tasks;
  const auto& contact_vector = formulation.contact_constraints;
  if (task_vector.empty()) return false;
  has_contact_ = !contact_vector.empty();

  // 1. Zero IK matrices
  H_ik_.setZero();
  g_ik_pos_.setZero();
  g_ik_vel_.setZero();
  g_ik_acc_.setZero();

  // 2. Contact constraints (near-rigid: weight 1e6)
  constexpr double kWContact = 1e6;
  for (const auto* c : contact_vector) {
    const Eigen::MatrixXd& Jc = c->Jacobian();
    H_ik_.noalias() += kWContact * Jc.transpose() * Jc;
    g_ik_acc_.noalias() -=
        kWContact * Jc.transpose() * (c->OpCommand() - c->JacobianDotQdot());
  }

  // 3. Motion tasks (weight from YAML determines priority)
  for (const auto* task : task_vector) {
    const Eigen::MatrixXd& Jt = task->Jacobian();
    const Eigen::VectorXd& w = task->Weight();
    const int rows = static_cast<int>(Jt.rows());

    for (int i = 0; i < rows; ++i) {
      const double wi = (w.size() == rows) ? w(i) : 1.0;
      if (wi <= 0.0) continue;

      const auto Jt_row_t = Jt.row(i).transpose();
      H_ik_.noalias() += wi * Jt_row_t * Jt_row_t.transpose();

      g_ik_pos_.noalias() -=
          wi * Jt_row_t * (task->KpIK()(i) * task->LocalPosError()(i));
      g_ik_vel_.noalias() -= wi * Jt_row_t * task->DesiredVel()(i);
      g_ik_acc_.noalias() -=
          wi * Jt_row_t * (task->OpCommand()(i) - task->JacobianDotQdot()(i));
    }
  }

  // 4. Regularization (prevents singularity blow-up)
  H_ik_.diagonal().array() += 1e-4;

  // 5. Position and velocity: unconstrained LLT (fastest path)
  llt_scratch_.compute(H_ik_);
  delta_q_cmd_ = llt_scratch_.solve(-g_ik_pos_);
  qdot_cmd_ = llt_scratch_.solve(-g_ik_vel_);

  // 6. Acceleration: QP with joint-limit box constraints
  // Cache constraint type pointers (reused by MakeTorque later this tick).
  CacheConstraintPointers(formulation.kinematic_constraints);

  constexpr double kInf = std::numeric_limits<double>::infinity();
  l_box_ik_.setConstant(-kInf);
  u_box_ik_.setConstant(kInf);

  const Constraint* ik_limits[] = {cached_pos_c_, cached_vel_c_};
  for (const Constraint* c : ik_limits) {
    if (!c) continue;
    ExtractAxisAlignedBoxBounds(c, zero_qddot_scratch_);
    for (int i = 0; i < num_qdot_; ++i) {
      l_box_ik_(i) = std::max(l_box_ik_(i), l_bnd_(i));
      u_box_ik_(i) = std::min(u_box_ik_(i), u_bnd_(i));
    }
  }
  // Feasibility guard
  for (int i = 0; i < num_qdot_; ++i) {
    if (l_box_ik_(i) > u_box_ik_(i)) l_box_ik_(i) = u_box_ik_(i);
  }

  // Check if any finite bounds exist
  bool has_bounds = false;
  for (int i = 0; i < num_qdot_; ++i) {
    if (l_box_ik_(i) > -1e30 || u_box_ik_(i) < 1e30) {
      has_bounds = true;
      break;
    }
  }

  if (has_bounds) {
    EnsureIkQPSolverInitialized();
    // QP solve with box constraints
    qp_ik_solver_->update(H_ik_, g_ik_acc_, std::nullopt, std::nullopt,
                           std::nullopt, std::nullopt, std::nullopt,
                           l_box_ik_, u_box_ik_);
    qp_ik_solver_->solve();

    if (qp_ik_solver_->results.info.status ==
        proxsuite::proxqp::QPSolverOutput::PROXQP_SOLVED) {
      qddot_cmd_ = qp_ik_solver_->results.x;
    } else {
      return false;
    }
  } else {
    // No kinematic constraints: pure LLT
    qddot_cmd_ = llt_scratch_.solve(-g_ik_acc_);
  }

  if (!delta_q_cmd_.allFinite() || !qdot_cmd_.allFinite() ||
      !qddot_cmd_.allFinite()) {
    return false;
  }

  jpos_cmd = curr_jpos + delta_q_cmd_.tail(num_qdot_ - num_floating_);
  jvel_cmd = qdot_cmd_.tail(num_qdot_ - num_floating_);
  wbc_qddot_cmd = qddot_cmd_;

  if (enable_timing_) {
    const auto t1_fc = std::chrono::steady_clock::now();
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
    Jc_.resize(0, num_qdot_);
    JcDotQdot_.resize(0);
    Uf_mat_.resize(0, 0);
    Uf_vec_.resize(0);
    des_rf_.resize(0);
    wbic_data_->delta_rf_.resize(0);
    wbic_data_->rf_cmd_.resize(0);
    wbic_data_->rf_prev_cmd_.resize(0);
    wbic_data_->Xc_ddot_.resize(0);
  }

  // Cache typed constraint pointers (reused by SetQPInEqualityConstraint).
  // FindConfigurationWeightedQP already caches these, so this is usually a
  // no-cost re-cache with the same formulation.
  CacheConstraintPointers(formulation.kinematic_constraints);
  dim_slack_pos_ = (cached_pos_c_ && soft_params_.pos) ? num_active_ : 0;
  dim_slack_vel_ = (cached_vel_c_ && soft_params_.vel) ? num_active_ : 0;
  dim_slack_trq_ = (cached_trq_c_ && soft_params_.trq) ? num_active_ : 0;
  dim_slack_total_ = dim_slack_pos_ + dim_slack_vel_ + dim_slack_trq_;

  // Always run QP — even without contacts, delta_qddot regularizes torque
  const auto t0_mt = std::chrono::steady_clock::now();
  SetQPCost(wbc_qddot_cmd);
  SetQPEqualityConstraint(wbc_qddot_cmd);
  SetQPInEqualityConstraint(formulation, wbc_qddot_cmd);
  const auto t1_mt = std::chrono::steady_clock::now();
  if (!SolveQP(wbc_qddot_cmd)) {
    return false;
  }
  const auto t2_mt = std::chrono::steady_clock::now();

  GetSolution(wbc_qddot_cmd, jtrq_cmd);

  if (enable_timing_) {
    const auto t3_mt = std::chrono::steady_clock::now();
    timing_stats_.qp_setup_us =
        std::chrono::duration<double, std::micro>(t1_mt - t0_mt).count();
    timing_stats_.qp_solve_us =
        std::chrono::duration<double, std::micro>(t2_mt - t1_mt).count();
    timing_stats_.torque_recovery_us =
        std::chrono::duration<double, std::micro>(t3_mt - t2_mt).count();
  }
  return true;
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
  if (m > kMaxPInvDim) {
    // Fallback for larger systems: preserve correctness with dynamic temporaries.
    Eigen::MatrixXd JWJt = jac * W * jac.transpose();
    JWJt.diagonal().array() += kDlsLambdaSq;
    Eigen::LLT<Eigen::MatrixXd> llt_dyn(JWJt);
    Eigen::MatrixXd JWJt_inv = Eigen::MatrixXd::Identity(m, m);
    llt_dyn.solveInPlace(JWJt_inv);
    out.noalias() = W * jac.transpose() * JWJt_inv;
    return;
  }
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

void WBIC::CacheConstraintPointers(const std::vector<Constraint*>& constraints) {
  cached_pos_c_ = nullptr;
  cached_vel_c_ = nullptr;
  cached_trq_c_ = nullptr;
  for (const Constraint* c : constraints) {
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

void WBIC::ExtractAxisAlignedBoxBounds(
    const Constraint* c, const Eigen::VectorXd& wbc_qddot_cmd) {
  constexpr double kInf = std::numeric_limits<double>::infinity();
  constexpr double kCoeffTol = 1e-12;
  l_bnd_.setConstant(-kInf);
  u_bnd_.setConstant(kInf);

  const Eigen::MatrixXd& A_kin = c->ConstraintMatrix();
  const Eigen::VectorXd& b_kin = c->ConstraintVector();

  // This extractor is intentionally narrow: each row must represent a single-axis
  // bound `a_i * delta_qddot_j <= rhs` (one non-zero coefficient).
  // Generic coupled constraints should stay in dense inequality form.
  for (int r = 0; r < A_kin.rows(); ++r) {
    int idx = -1;
    double coeff = 0.0;
    int nonzero_count = 0;
    for (int j = 0; j < A_kin.cols(); ++j) {
      const double a = A_kin(r, j);
      if (std::abs(a) > kCoeffTol) {
        ++nonzero_count;
        if (nonzero_count == 1) {
          idx = j;
          coeff = a;
        }
      }
    }
    if (nonzero_count != 1 || idx < 0) {
      assert(nonzero_count == 1 &&
             "ExtractAxisAlignedBoxBounds: row is not one-hot axis-aligned.");
      continue;
    }

    const double rhs = (b_kin(r) - coeff * wbc_qddot_cmd(idx)) / coeff;
    if (coeff > 0.0) {
      u_bnd_(idx) = std::min(u_bnd_(idx), rhs);
    } else {
      l_bnd_(idx) = std::max(l_bnd_(idx), rhs);
    }
  }
}

int WBIC::BuildKinematicLimitConstraint(const Constraint* c, bool is_soft,
                                        bool use_box_solver,
                                        const Eigen::VectorXd& wbc_qddot_cmd,
                                        int row, int& slack_col) {
  ExtractAxisAlignedBoxBounds(c, wbc_qddot_cmd);

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
          -sa_ * Jc_.transpose();
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
          -sa_ * Jc_.transpose();
    }
    for (int i = 0; i < num_active_; ++i) {
      l_(row + i) = limits(i, 0) - sa_tau0_scratch_(i);
      u_(row + i) = limits(i, 1) - sa_tau0_scratch_(i);
    }
    return num_active_;
  } else {
    // Fast diagonal-M approximation (not exact full coupled torque dynamics).
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
      ++box_conflict_count_;
      last_box_conflict_index_ = i;
      l_box_(i) = u_box_(i);
    }
  }
}

void WBIC::SetQPCost(const Eigen::VectorXd& wbc_qddot_cmd) {
  const int qp_dim = num_qdot_ + dim_contact_ + dim_slack_total_;
  // resize() only reallocates when rows*cols changes (stable in steady state).
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
  // tau_0 = M * qddot_ik + (cori + grav) - Jc^T * des_rf
  // NOTE: Ni_dyn (passive-joint nullspace) is identity for all supported robots.
  tau_0_.noalias() = M_ * wbc_qddot_cmd;
  tau_0_ += cori_ + grav_;
  if (has_contact_ && des_rf_.size() > 0) {
    tau_0_.noalias() -= Jc_.transpose() * des_rf_;
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
  if (wbic_data_->rf_prev_cmd_.size() != dim_contact_) {
    wbic_data_->rf_prev_cmd_.setZero(dim_contact_);
  }

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

  A_.resize(num_floating_, qp_dim);
  A_.setZero();
  if (num_floating_ > 0) {
    // sf_ * M picks the floating-base rows of M (all num_qdot_ columns)
    A_.leftCols(num_qdot_).noalias() = sf_ * M_;
    if (dim_contact_ > 0) {
      A_.block(0, num_qdot_, num_floating_, dim_contact_).noalias() = -sf_ * Jc_.transpose();
    }
    b_.resize(num_floating_);
    b_.noalias() = -sf_ * (M_ * wbc_qddot_cmd + cori_ + grav_);
    if (dim_contact_ > 0) {
      b_.noalias() += sf_ * Jc_.transpose() * des_rf_;
    }
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

  const bool force_exact_trq_dense =
      (trq_c && !soft_params_.trq &&
       hard_torque_limit_mode_ == HardTorqueLimitMode::EXACT_DENSE);
  // When slack (soft) variables exist, disable box solver to avoid
  // ProxQP ADMM scaling collapse between huge box bounds and unbounded slacks.
  // Exact hard torque mode also forces dense inequalities for coupled dynamics.
  const bool force_dense = (dim_slack_total_ > 0) || force_exact_trq_dense;
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
  C_.resize(n_ineq, qp_dim);
  C_.setZero();
  l_.resize(n_ineq);
  l_.setZero();
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
      qp_solver_->init(H_, g_, A_, b_, C_, l_, u_);
    }
  } else {
    if (use_box) {
      qp_solver_->update(H_, g_, A_, b_, C_, l_, u_, l_box_, u_box_);
    } else {
      qp_solver_->update(H_, g_, A_, b_, C_, l_, u_);
    }
  }

  qp_solver_->solve();

  const auto& info = qp_solver_->results.info;
  wbic_data_->qp_status_ = static_cast<int>(info.status);
  wbic_data_->qp_iter_ = info.iter;
  wbic_data_->qp_pri_res_ = info.pri_res;
  wbic_data_->qp_dua_res_ = info.dua_res;
  wbic_data_->qp_obj_ = info.objValue;
  wbic_data_->qp_setup_time_us_ = info.setup_time;
  wbic_data_->qp_solve_time_us_ = info.solve_time;
  wbic_data_->qp_solved_ =
      (info.status == proxsuite::proxqp::QPSolverOutput::PROXQP_SOLVED);

  if (!wbic_data_->qp_solved_) {
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
    wbic_data_->rf_prev_cmd_.resize(0);
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
  // Torque equation: tau = M * qddot + (cori + grav) - Jc^T * rf
  // NOTE: Ni_dyn (passive-joint nullspace) is identity for all supported robots.
  trq_.noalias() = M_ * wbic_data_->corrected_wbc_qddot_cmd_;
  trq_ += cori_ + grav_;
  if (has_contact_ && wbic_data_->rf_cmd_.size() > 0) {
    trq_.noalias() -= Jc_.transpose() * wbic_data_->rf_cmd_;
  }

  // Store full-DOF torque for next-tick rate-of-change cost.
  wbic_data_->tau_prev_ = trq_;

  // UNi = Sa (selection matrix for actuated DOFs; Ni_dyn is identity)
  UNi_.noalias() = sa_;
  if (UNi_.rows() == UNi_.cols()) {
    // Fully actuated: exact inverse via pre-allocated LU (no DLS damping distortion).
    lu_scratch_.compute(UNi_.transpose());
    jtrq_cmd = lu_scratch_.solve(trq_);
  } else {
    WeightedPseudoInverse(UNi_, Minv_, UNi_bar_);
    jtrq_cmd = UNi_bar_.transpose() * trq_;
  }
  jtrq_cmd = snf_ * sa_.transpose() * jtrq_cmd;
}

} // namespace wbc
