/**
 * @file wbc_core/wbc_solver/src/wbic.cpp
 * @brief Doxygen documentation for wbic module.
 */
#include "wbc_solver/wbic.hpp"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <limits>
#include <utility>

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
constexpr double kDlsLambda = 0.05;
constexpr double kDlsLambdaSq = kDlsLambda * kDlsLambda;

bool IsValidOptionalWeight(const Eigen::VectorXd& values, int expected_size) {
  if (values.size() != 0 && values.size() != expected_size) {
    return false;
  }
  return values.allFinite() &&
         (values.size() == 0 || (values.array() >= 0.0).all());
}

bool IsNonnegativeFiniteWeight(const Eigen::VectorXd& values) {
  return IsValidOptionalWeight(values, values.size());
}
} // namespace

WBIC::WBIC(const std::vector<bool>& act_qdot_list, QPParams* qp_params)
    : WBC(act_qdot_list),
      delta_q_ref_(Eigen::VectorXd::Zero(num_qdot_)),
      qdot_ref_(Eigen::VectorXd::Zero(num_qdot_)),
      tau_gen_sol_(Eigen::VectorXd::Zero(num_qdot_)),
      tau_cost_scratch_(Eigen::VectorXd::Zero(num_qdot_)),
      kp_acc_(Eigen::VectorXd::Constant(num_active_, 120.0)),
      kd_acc_(Eigen::VectorXd::Constant(num_active_, 20.0)),
      sa_pinv_scratch_(Eigen::MatrixXd::Zero(num_qdot_, num_active_)),
      wbic_data_(std::make_unique<WBICData>(num_qdot_, qp_params)),
      tau_0_(Eigen::VectorXd::Zero(num_qdot_)),
      l_bnd_(Eigen::VectorXd::Zero(num_qdot_)),
      u_bnd_(Eigen::VectorXd::Zero(num_qdot_)),
      pos_bounded_active_indices_(),
      vel_bounded_active_indices_(),
      sa_tau0_scratch_(Eigen::VectorXd::Zero(num_active_)),
      H_ik_(Eigen::MatrixXd::Zero(num_qdot_, num_qdot_)),
      g_ik_pos_(Eigen::VectorXd::Zero(num_qdot_)),
      g_ik_vel_(Eigen::VectorXd::Zero(num_qdot_)) {
  // WBIC assumes "active joints = non-floating DOFs" throughout the solver.
  assert(num_active_ == (num_qdot_ - num_floating_));
  wM_scratch_.resize(num_qdot_, num_qdot_);
  pos_bounded_active_indices_.reserve(num_active_);
  vel_bounded_active_indices_.reserve(num_active_);
}

void WBIC::ReserveCapacity(int max_contact_dim, int max_uf_rows) {
  // Contact matrices (SolveInverseDynamics)
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
  // Worst case (dense truth path):
  // - pos soft: 2 * num_active slacks
  // - vel soft: 2 * num_active slacks
  // - trq soft: 1 * num_active slacks
  const int max_slack = 5 * num_active_;
  const int max_qp_dim = num_qdot_ + max_contact_dim + max_slack;
  const int max_n_eq = num_floating_;
  const int max_n_ineq = max_uf_rows + 5 * num_active_;

  H_.resize(max_qp_dim, max_qp_dim);
  g_.resize(max_qp_dim);
  A_.resize(max_n_eq, max_qp_dim);
  b_.resize(max_n_eq);
  C_.resize(max_n_ineq, max_qp_dim);
  l_.resize(max_n_ineq);
  u_.resize(max_n_ineq);
  l_box_.resize(max_qp_dim);
  u_box_.resize(max_qp_dim);

  // Pre-allocate ProxQP solvers so the first tick doesn't heap-allocate.
  // SolveInverseDynamics solver: pre-created at max dims. Re-created only on rare
  // dimension changes (state transitions). Init with dummy data so first
  // hot-path call can use update() if dims match.
  qp_solver_ = std::make_unique<proxsuite::proxqp::dense::QP<double>>(
      max_qp_dim, max_n_eq, max_n_ineq, true);
  qp_solver_->settings.eps_abs = 1e-3;
  qp_solver_->settings.verbose = false;
  qp_solver_->settings.max_iter = 1000;
  qp_solver_->settings.initial_guess =
      proxsuite::proxqp::InitialGuessStatus::WARM_START_WITH_PREVIOUS_RESULT;
  {
    H_.setZero(max_qp_dim, max_qp_dim);
    for (int i = 0; i < max_qp_dim; ++i) H_(i, i) = 1.0;
    g_.setZero(max_qp_dim);
    A_.setZero(max_n_eq, max_qp_dim);
    b_.setZero(max_n_eq);
    C_.setZero(max_n_ineq, max_qp_dim);
    // ProxQP rejects C=0 when n_ineq>0.  Set a dummy identity-like structure
    // so the pre-allocation init passes validation; the first real SolveQP
    // call will overwrite C with actual constraint data.
    for (int i = 0; i < max_n_ineq && i < max_qp_dim; ++i) C_(i, i) = 1.0;
    l_.setZero(max_n_ineq);
    u_.setZero(max_n_ineq);
    l_box_.setConstant(max_qp_dim, -1e30);
    u_box_.setConstant(max_qp_dim,  1e30);
    qp_solver_->init(H_, g_, A_, b_, C_, l_, u_, l_box_, u_box_);
  }

}

bool WBIC::ComputeKinematicReference(
    const WbcFormulation& formulation, const Eigen::VectorXd& curr_jpos,
    Eigen::VectorXd& jpos_ref, Eigen::VectorXd& jvel_ref) {
  if (!settings_updated_) {
    return false;
  }
  if (!ValidateIKRuntimeConfig()) {
    return false;
  }
  if (curr_jpos.size() != (num_qdot_ - num_floating_) || !curr_jpos.allFinite()) {
    return false;
  }
  if (!ValidatePostureTasks(formulation.posture_tasks) ||
      !ValidateIKContacts(formulation.contact_constraints)) {
    return false;
  }
  if (!CacheConstraintPointers(formulation.kinematic_constraints)) {
    return false;
  }
  return SolveKinematicReferenceQP(formulation, curr_jpos, jpos_ref, jvel_ref);
}

bool WBIC::SetJointAccelReferenceGains(double kp_acc, double kd_acc) {
  const Eigen::VectorXd kp_vec = Eigen::VectorXd::Constant(num_active_, kp_acc);
  const Eigen::VectorXd kd_vec = Eigen::VectorXd::Constant(num_active_, kd_acc);
  return SetJointAccelReferenceGains(kp_vec, kd_vec);
}

bool WBIC::SetJointAccelReferenceGains(const Eigen::VectorXd& kp_acc,
                                       const Eigen::VectorXd& kd_acc) {
  if (kp_acc.size() != num_active_ || kd_acc.size() != num_active_) {
    return false;
  }
  if (!kp_acc.allFinite() || !kd_acc.allFinite()) {
    return false;
  }
  if ((kp_acc.array() < 0.0).any() || (kd_acc.array() < 0.0).any()) {
    return false;
  }
  kp_acc_ = kp_acc;
  kd_acc_ = kd_acc;
  return true;
}

bool WBIC::ComputeKinematicReference(const WbcFormulation& formulation,
                                     const Eigen::VectorXd& curr_jpos,
                                     KinematicReference& out_ref) {
  Eigen::VectorXd jpos_ref;
  Eigen::VectorXd jvel_ref;
  if (!ComputeKinematicReference(formulation, curr_jpos, jpos_ref, jvel_ref)) {
    return false;
  }
  out_ref.jpos_ref = std::move(jpos_ref);
  out_ref.jvel_ref = std::move(jvel_ref);
  return true;
}

bool WBIC::BuildPostureAccelReference(const KinematicReference& kin_ref,
                                      const Eigen::VectorXd& jpos_meas,
                                      const Eigen::VectorXd& jvel_meas,
                                      Eigen::VectorXd& qddot_posture_ref) const {
  const int n_joint = num_qdot_ - num_floating_;
  if (kin_ref.jpos_ref.size() != n_joint || kin_ref.jvel_ref.size() != n_joint ||
      jpos_meas.size() != n_joint || jvel_meas.size() != n_joint) {
    return false;
  }
  if (kp_acc_.size() != n_joint || kd_acc_.size() != n_joint) {
    return false;
  }
  if (!kin_ref.jpos_ref.allFinite() || !kin_ref.jvel_ref.allFinite() ||
      !jpos_meas.allFinite() || !jvel_meas.allFinite()) {
    return false;
  }

  qddot_posture_ref.setZero(num_qdot_);
  qddot_posture_ref.tail(n_joint).array() =
      kp_acc_.array() * (kin_ref.jpos_ref - jpos_meas).array() +
      kd_acc_.array() * (kin_ref.jvel_ref - jvel_meas).array();
  return qddot_posture_ref.allFinite();
}

////////////////////////////////////////////////////////////////////////////////
// Weighted QP IK: replaces hierarchical null-space with a single weighted
// least-squares solve. Task weights determine effective priority.
// q_ref and qdot_ref use LLT (unconstrained reference solve).
// qddot_posture_ref is built in BuildPostureAccelReference from measured state feedback.
////////////////////////////////////////////////////////////////////////////////
bool WBIC::SolveKinematicReferenceQP(
    const WbcFormulation& formulation, const Eigen::VectorXd& curr_jpos,
    Eigen::VectorXd& jpos_ref, Eigen::VectorXd& jvel_ref) {
  // Public entry (ComputeKinematicReference) validates runtime preconditions.
  assert(settings_updated_);
  assert(ValidateIKRuntimeConfig());
  assert(curr_jpos.size() == (num_qdot_ - num_floating_));
  assert(curr_jpos.allFinite());
  assert(ValidatePostureTasks(formulation.posture_tasks));
  assert(ValidateIKContacts(formulation.contact_constraints));
  const auto t0_fc = std::chrono::steady_clock::now();

  const auto& task_vector = formulation.posture_tasks;
  const auto& contact_vector = formulation.contact_constraints;
  assert(cached_pos_c_ == nullptr || HasSupportedJointLimitShape(cached_pos_c_));
  assert(cached_vel_c_ == nullptr || HasSupportedJointLimitShape(cached_vel_c_));

  // If no posture task is configured, keep a neutral posture reference.
  if (task_vector.empty()) {
    jpos_ref = curr_jpos;
    jvel_ref = Eigen::VectorXd::Zero(num_qdot_ - num_floating_);
    ProjectKinematicReferenceToJointBounds(curr_jpos, jpos_ref, jvel_ref);
    return true;
  }

  // 1. Zero IK matrices
  H_ik_.setZero();
  g_ik_pos_.setZero();
  g_ik_vel_.setZero();

  // 2. Contact constraints
  // This is NOT hard contact preservation. It only biases posture IK away from
  // contact-blind motion. Final contact consistency is handled in ID-QP.
  for (const auto* c : contact_vector) {
    const Eigen::MatrixXd& Jc = c->Jacobian();
    H_ik_.noalias() += ik_contact_penalty_weight_ * Jc.transpose() * Jc;
  }

  // 3. Motion tasks (weight from YAML determines priority)
  for (const auto* task : task_vector) {
    const Eigen::MatrixXd& Jt = task->Jacobian();
    const Eigen::VectorXd& w = task->Weight();
    const int rows = static_cast<int>(Jt.rows());
    const Eigen::VectorXd& kp_ik = task->KpIK();
    const Eigen::VectorXd& local_pos_err = task->LocalPosError();
    const Eigen::VectorXd& desired_vel = task->DesiredVel();

    for (int i = 0; i < rows; ++i) {
      const double wi = (w.size() == rows) ? w(i) : 1.0;
      if (wi <= 0.0) continue;

      const auto Jt_row_t = Jt.row(i).transpose();
      H_ik_.noalias() += wi * Jt_row_t * Jt_row_t.transpose();

      g_ik_pos_.noalias() -=
          wi * Jt_row_t * (kp_ik(i) * local_pos_err(i));
      g_ik_vel_.noalias() -= wi * Jt_row_t * desired_vel(i);
    }
  }

  // 4. Regularization (prevents singularity blow-up)
  H_ik_.diagonal().array() += 1e-4;

  // 5. Position/velocity reference
  llt_scratch_.compute(H_ik_);
  if (llt_scratch_.info() != Eigen::Success) {
    return false;
  }
  delta_q_ref_ = llt_scratch_.solve(-g_ik_pos_);
  if (independent_velocity_ref_) {
    qdot_ref_ = llt_scratch_.solve(-g_ik_vel_);
  } else {
    qdot_ref_.setZero();
    const int n_joint = num_qdot_ - num_floating_;
    qdot_ref_.tail(n_joint) = delta_q_ref_.tail(n_joint) / ik_velocity_clamp_dt_;
  }

  if (!delta_q_ref_.allFinite() || !qdot_ref_.allFinite()) {
    return false;
  }

  jpos_ref = curr_jpos + delta_q_ref_.tail(num_qdot_ - num_floating_);
  jvel_ref = qdot_ref_.tail(num_qdot_ - num_floating_);
  ProjectKinematicReferenceToJointBounds(curr_jpos, jpos_ref, jvel_ref);
  if (!jpos_ref.allFinite() || !jvel_ref.allFinite()) {
    return false;
  }

  if (enable_timing_) {
    const auto t1_fc = std::chrono::steady_clock::now();
    timing_stats_.find_config_us =
        std::chrono::duration<double, std::micro>(t1_fc - t0_fc).count();
  }
  return true;
}

bool WBIC::MakeTorque(const WbcFormulation& formulation,
                      const Eigen::VectorXd& qddot_posture_ref,
                      Eigen::VectorXd& jtrq_cmd) {
  return SolveInverseDynamics(formulation, qddot_posture_ref, jtrq_cmd);
}

bool WBIC::SolveInverseDynamics(const WbcFormulation& formulation,
                                const Eigen::VectorXd& qddot_posture_ref,
                                Eigen::VectorXd& jtrq_cmd) {
  if (!settings_updated_) {
    return false;
  }
  if (qddot_posture_ref.size() != num_qdot_ || !qddot_posture_ref.allFinite()) {
    return false;
  }
  if (!ValidateOperationalTasks(formulation.operational_tasks)) {
    return false;
  }
  if (!PrepareContactContext(formulation)) {
    return false;
  }
  if (!ValidateIDRuntimeConfig()) {
    return false;
  }

  // Cache typed constraint pointers for inequality assembly.
  if (!CacheConstraintPointers(formulation.kinematic_constraints)) {
    return false;
  }
  ComputeSlackDimensions(qddot_posture_ref);

  // Always run QP — even without contacts, delta_qddot regularizes torque
  const auto t0_mt = std::chrono::steady_clock::now();
  SetQPCost(formulation, qddot_posture_ref);
  SetQPEqualityConstraint(qddot_posture_ref);
  SetQPInEqualityConstraint(qddot_posture_ref);
  const auto t1_mt = std::chrono::steady_clock::now();
  if (!SolveQP(qddot_posture_ref)) {
    return false;
  }
  const auto t2_mt = std::chrono::steady_clock::now();

  if (!GetSolution(jtrq_cmd)) {
    return false;
  }

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

bool WBIC::SolveInverseDynamics(const WbcFormulation& formulation,
                                const Eigen::VectorXd& qddot_posture_ref,
                                InverseDynamicsSolution& out_sol) {
  Eigen::VectorXd jtrq_cmd;
  if (!SolveInverseDynamics(formulation, qddot_posture_ref, jtrq_cmd)) {
    return false;
  }
  out_sol.qddot_sol = wbic_data_->qddot_sol_;
  out_sol.rf_sol = wbic_data_->rf_sol_;
  out_sol.tau_gen_sol = tau_gen_sol_;
  out_sol.tau_cmd = std::move(jtrq_cmd);
  return true;
}

bool WBIC::PrepareContactContext(const WbcFormulation& formulation) {
  const std::vector<ForceTask*>& force_task_vector = formulation.force_tasks;
  const std::vector<Contact*>& contact_constraints =
      formulation.contact_constraints;

  if (!contact_constraints.empty()) {
    if (!BuildContactMtxVect(contact_constraints) ||
        !GetDesiredReactionForce(force_task_vector) ||
        des_rf_.size() != dim_contact_) {
      return false;
    }
  } else {
    if (!force_task_vector.empty()) {
      return false;
    }
    ClearContactContext();
  }

  return true;
}

void WBIC::ClearContactContext() {
  if (dim_contact_ == 0 && contact_rf_blocks_.empty() &&
      Jc_.rows() == 0 && JcDotQdot_.size() == 0 &&
      Uf_mat_.rows() == 0 && Uf_mat_.cols() == 0 &&
      Uf_vec_.size() == 0 && des_rf_.size() == 0 &&
      wbic_data_->delta_rf_.size() == 0 &&
      wbic_data_->rf_sol_.size() == 0 &&
      wbic_data_->rf_prev_sol_.size() == 0 &&
      wbic_data_->Xc_ddot_.size() == 0) {
    return;
  }

  dim_contact_ = 0;
  contact_rf_blocks_.clear();
  if (Jc_.rows() != 0) {
    Jc_.resize(0, num_qdot_);
  }
  if (JcDotQdot_.size() != 0) {
    JcDotQdot_.resize(0);
  }
  if (Uf_mat_.rows() != 0 || Uf_mat_.cols() != 0) {
    Uf_mat_.resize(0, 0);
  }
  if (Uf_vec_.size() != 0) {
    Uf_vec_.resize(0);
  }
  if (des_rf_.size() != 0) {
    des_rf_.resize(0);
  }
  if (wbic_data_->delta_rf_.size() != 0) {
    wbic_data_->delta_rf_.resize(0);
  }
  if (wbic_data_->rf_sol_.size() != 0) {
    wbic_data_->rf_sol_.resize(0);
  }
  if (wbic_data_->rf_prev_sol_.size() != 0) {
    wbic_data_->rf_prev_sol_.resize(0);
  }
  if (wbic_data_->Xc_ddot_.size() != 0) {
    wbic_data_->Xc_ddot_.resize(0);
  }
}

void WBIC::ComputeSlackDimensions(const Eigen::VectorXd& qddot_ref) {
  assert(qddot_ref.size() == num_qdot_);
  assert(qddot_ref.allFinite());
  pos_bounded_active_indices_.clear();
  vel_bounded_active_indices_.clear();

  const auto collect_axis_bounded_indices =
      [this, &qddot_ref](const Constraint* c, std::vector<int>& out_indices) {
        out_indices.clear();
        if (c == nullptr || !ExtractAxisAlignedBoxBounds(c, qddot_ref)) {
          return false;
        }
        for (int i = 0; i < num_active_; ++i) {
          const int idx = num_floating_ + i;
          if (std::isfinite(l_bnd_(idx)) || std::isfinite(u_bnd_(idx))) {
            out_indices.push_back(i);
          }
        }
        return true;
      };

  const auto axis_aligned_bound_dim =
      [&](const Constraint* c, std::vector<int>& bounded_indices) {
        if (c == nullptr) {
          return 0;
        }
        if (collect_axis_bounded_indices(c, bounded_indices)) {
          return static_cast<int>(bounded_indices.size());
        }
        bounded_indices.clear();
        return static_cast<int>(c->ConstraintMatrix().rows());
      };

  const int pos_bound_dim =
      axis_aligned_bound_dim(cached_pos_c_, pos_bounded_active_indices_);
  const int vel_bound_dim =
      axis_aligned_bound_dim(cached_vel_c_, vel_bounded_active_indices_);

  dim_slack_pos_ = (cached_pos_c_ && soft_params_.pos) ? pos_bound_dim : 0;
  dim_slack_vel_ = (cached_vel_c_ && soft_params_.vel) ? vel_bound_dim : 0;
  dim_slack_trq_ = (cached_trq_c_ && soft_params_.trq) ? num_active_ : 0;
  dim_slack_total_ = dim_slack_pos_ + dim_slack_vel_ + dim_slack_trq_;
}

bool WBIC::WeightedPseudoInverse(const Eigen::MatrixXd& jac,
                                 const Eigen::MatrixXd& W,
                                 Eigen::MatrixXd& out) {
  if (jac.size() == 0) {
    out.resize(jac.cols(), jac.rows());
    out.setZero();
    return true;
  }
  if (W.rows() != jac.cols() || W.cols() != jac.cols() || !jac.allFinite() ||
      !W.allFinite()) {
    return false;
  }
  const int m = jac.rows();
  if (m > kMaxPInvDim) {
    // Fallback for larger systems: preserve correctness with dynamic temporaries.
    Eigen::MatrixXd JWJt = jac * W * jac.transpose();
    JWJt.diagonal().array() += kDlsLambdaSq;
    Eigen::LLT<Eigen::MatrixXd> llt_dyn(JWJt);
    if (llt_dyn.info() != Eigen::Success) {
      return false;
    }
    Eigen::MatrixXd JWJt_inv = Eigen::MatrixXd::Identity(m, m);
    llt_dyn.solveInPlace(JWJt_inv);
    out.noalias() = W * jac.transpose() * JWJt_inv;
    return out.allFinite();
  }
  assert(m <= kMaxPInvDim && "WeightedPseudoInverse: jac.rows() exceeds kMaxPInvDim");
  // J W J^T + λ²I  (m×m, always SPD)
  // PInvSquare uses MaxRows/MaxCols → inline storage, no heap alloc on resize.
  JWJt_scratch_.resize(m, m);
  JWJt_scratch_.noalias() = jac * W * jac.transpose();
  JWJt_scratch_.diagonal().array() += kDlsLambdaSq;
  llt_scratch_.compute(JWJt_scratch_);
  if (llt_scratch_.info() != Eigen::Success) {
    return false;
  }
  // J_bar = W J^T (J W J^T + λ²I)^{-1}
  JWJt_pinv_scratch_.setIdentity(m, m);
  llt_scratch_.solveInPlace(JWJt_pinv_scratch_);
  out.noalias() = W * jac.transpose() * JWJt_pinv_scratch_;
  return out.allFinite();
}

bool WBIC::CacheConstraintPointers(const std::vector<Constraint*>& constraints) {
  cached_pos_c_ = nullptr;
  cached_vel_c_ = nullptr;
  cached_trq_c_ = nullptr;
  for (const Constraint* c : constraints) {
    if (c == nullptr) {
      return false;
    }
    if (auto* p = dynamic_cast<const JointPosLimitConstraint*>(c)) {
      if (cached_pos_c_ != nullptr) {
        return false;
      }
      if (!HasSupportedJointLimitShape(p)) {
        return false;
      }
      cached_pos_c_ = p;
      continue;
    }
    if (auto* v = dynamic_cast<const JointVelLimitConstraint*>(c)) {
      if (cached_vel_c_ != nullptr) {
        return false;
      }
      if (!HasSupportedJointLimitShape(v)) {
        return false;
      }
      cached_vel_c_ = v;
      continue;
    }
    if (auto* t = dynamic_cast<const JointTrqLimitConstraint*>(c)) {
      if (cached_trq_c_ != nullptr) {
        return false;
      }
      cached_trq_c_ = t;
      continue;
    }
    // Current WBIC implementation supports exactly these three kinematic limits.
    // Any unknown constraint type is a configuration error.
    return false;
  }
  return true;
}

bool WBIC::HasSupportedJointLimitShape(
    const JointPosLimitConstraint* c) const {
  if (c == nullptr) {
    return false;
  }
  const Eigen::MatrixXd& A_kin = c->ConstraintMatrix();
  const Eigen::VectorXd& b_kin = c->ConstraintVector();
  const Eigen::MatrixXd& limits = c->EffectiveLimits();
  // Constraint matrix has num_qdot_ cols (covers floating base + active joints).
  return A_kin.cols() == num_qdot_ &&
         limits.rows() == num_active_ && limits.cols() >= 2 &&
         b_kin.size() == A_kin.rows() && A_kin.allFinite() &&
         b_kin.allFinite() && limits.allFinite();
}

bool WBIC::HasSupportedJointLimitShape(
    const JointVelLimitConstraint* c) const {
  if (c == nullptr) {
    return false;
  }
  const Eigen::MatrixXd& A_kin = c->ConstraintMatrix();
  const Eigen::VectorXd& b_kin = c->ConstraintVector();
  const Eigen::MatrixXd& limits = c->EffectiveLimits();
  // Constraint matrix has num_qdot_ cols (covers floating base + active joints).
  return A_kin.cols() == num_qdot_ &&
         limits.rows() == num_active_ && limits.cols() >= 2 &&
         b_kin.size() == A_kin.rows() && A_kin.allFinite() &&
         b_kin.allFinite() && limits.allFinite();
}

bool WBIC::ValidateOperationalTasks(
    const std::vector<Task*>& operational_tasks) const {
  for (const Task* task : operational_tasks) {
    if (task == nullptr) {
      return false;
    }
    const Eigen::MatrixXd& J = task->Jacobian();
    const Eigen::VectorXd& JdotQdot = task->JacobianDotQdot();
    const Eigen::VectorXd& a_des = task->OpCommand();
    const int rows = static_cast<int>(J.rows());
    if (rows <= 0 || J.cols() != num_qdot_ ||
        JdotQdot.size() != rows || a_des.size() != rows ||
        !J.allFinite() || !JdotQdot.allFinite() || !a_des.allFinite()) {
      return false;
    }
    const Eigen::VectorXd& w = task->Weight();
    if (!(w.size() == 0 || w.size() == rows) || !w.allFinite() ||
        (w.size() == rows && (w.array() < 0.0).any())) {
      return false;
    }
  }
  return true;
}

bool WBIC::ValidatePostureTasks(const std::vector<Task*>& posture_tasks) const {
  for (const Task* task : posture_tasks) {
    if (task == nullptr) {
      return false;
    }
    const Eigen::MatrixXd& J = task->Jacobian();
    const Eigen::VectorXd& kp_ik = task->KpIK();
    const Eigen::VectorXd& local_pos_err = task->LocalPosError();
    const Eigen::VectorXd& desired_vel = task->DesiredVel();
    const int rows = static_cast<int>(J.rows());
    if (rows <= 0 || J.cols() != num_qdot_ || !J.allFinite() ||
        kp_ik.size() != rows || local_pos_err.size() != rows ||
        desired_vel.size() != rows || !kp_ik.allFinite() ||
        !local_pos_err.allFinite() || !desired_vel.allFinite()) {
      return false;
    }
    const Eigen::VectorXd& w = task->Weight();
    if (!(w.size() == 0 || w.size() == rows) || !w.allFinite() ||
        (w.size() == rows && (w.array() < 0.0).any())) {
      return false;
    }
  }
  return true;
}

bool WBIC::ValidateIKContacts(
    const std::vector<Contact*>& contact_constraints) const {
  for (const Contact* c : contact_constraints) {
    if (c == nullptr || c->Jacobian().cols() != num_qdot_ ||
        !c->Jacobian().allFinite()) {
      return false;
    }
  }
  return true;
}

bool WBIC::ValidateIKRuntimeConfig() const {
  return std::isfinite(ik_contact_penalty_weight_) &&
         ik_contact_penalty_weight_ >= 0.0 &&
         std::isfinite(ik_velocity_clamp_dt_) && ik_velocity_clamp_dt_ > 0.0 &&
         std::isfinite(ik_velocity_ref_abs_max_) &&
         ik_velocity_ref_abs_max_ > 0.0;
}

bool WBIC::ValidateIDRuntimeConfig() const {
  const QPParams* qp_params = wbic_data_->qp_params_;
  if (qp_params == nullptr) {
    return false;
  }
  if (!IsValidOptionalWeight(qp_params->W_delta_qddot_, num_qdot_) ||
      !IsValidOptionalWeight(qp_params->W_tau_, num_qdot_) ||
      !IsValidOptionalWeight(qp_params->W_tau_dot_, num_qdot_)) {
    return false;
  }
  if (dim_contact_ > 0) {
    if (!IsValidOptionalWeight(qp_params->W_delta_rf_, dim_contact_) ||
        !IsValidOptionalWeight(qp_params->W_xc_ddot_, dim_contact_) ||
        !IsValidOptionalWeight(qp_params->W_f_dot_, dim_contact_)) {
      return false;
    }
  } else {
    if (!IsNonnegativeFiniteWeight(qp_params->W_delta_rf_) ||
        !IsNonnegativeFiniteWeight(qp_params->W_xc_ddot_) ||
        !IsNonnegativeFiniteWeight(qp_params->W_f_dot_)) {
      return false;
    }
  }
  if (!std::isfinite(soft_params_.w_pos) || soft_params_.w_pos < 0.0 ||
      !std::isfinite(soft_params_.w_vel) || soft_params_.w_vel < 0.0 ||
      !std::isfinite(soft_params_.w_trq) || soft_params_.w_trq < 0.0) {
    return false;
  }
  if (!std::isfinite(posture_bias_contact_scale_) ||
      posture_bias_contact_scale_ < 0.0) {
    return false;
  }
  return true;
}

void WBIC::ProjectKinematicReferenceToJointBounds(
    const Eigen::VectorXd& curr_jpos,
    Eigen::VectorXd& jpos_ref,
    Eigen::VectorXd& jvel_ref) const {
  const int n_joint = num_qdot_ - num_floating_;
  assert(curr_jpos.size() == n_joint);
  assert(jpos_ref.size() == n_joint);
  assert(jvel_ref.size() == n_joint);
  assert(std::isfinite(ik_velocity_ref_abs_max_) && ik_velocity_ref_abs_max_ > 0.0);

  const Eigen::MatrixXd* pos_limits = nullptr;
  if (cached_pos_c_ != nullptr) {
    const Eigen::MatrixXd& limits = cached_pos_c_->EffectiveLimits();
    assert(limits.rows() == n_joint && limits.cols() >= 2);
    pos_limits = &limits;
    for (int i = 0; i < n_joint; ++i) {
      jpos_ref(i) = std::clamp(jpos_ref(i), limits(i, 0), limits(i, 1));
    }
  }

  const Eigen::MatrixXd* vel_limits = nullptr;
  if (cached_vel_c_ != nullptr) {
    const Eigen::MatrixXd& limits = cached_vel_c_->EffectiveLimits();
    assert(limits.rows() == n_joint && limits.cols() >= 2);
    vel_limits = &limits;
  }
  for (int i = 0; i < n_joint; ++i) {
    // Always apply an absolute cap to avoid aggressive velocity references.
    double lower = -ik_velocity_ref_abs_max_;
    double upper = ik_velocity_ref_abs_max_;
    if (vel_limits != nullptr) {
      lower = (*vel_limits)(i, 0);
      upper = (*vel_limits)(i, 1);
      lower = std::max(lower, -ik_velocity_ref_abs_max_);
      upper = std::min(upper, ik_velocity_ref_abs_max_);
    }
    if (pos_limits != nullptr) {
      // Position-aware one-step velocity window based on current position.
      const double lower_pos =
          ((*pos_limits)(i, 0) - curr_jpos(i)) / ik_velocity_clamp_dt_;
      const double upper_pos =
          ((*pos_limits)(i, 1) - curr_jpos(i)) / ik_velocity_clamp_dt_;
      lower = std::max(lower, lower_pos);
      upper = std::min(upper, upper_pos);
    }
    if (lower > upper) {
      const double collapsed = 0.5 * (lower + upper);
      lower = collapsed;
      upper = collapsed;
    }
    jvel_ref(i) = std::clamp(jvel_ref(i), lower, upper);
  }
}

bool WBIC::BuildContactMtxVect(const std::vector<Contact*>& contacts) {
  int total_contact_dim = 0;
  int total_uf_rows = 0;
  contact_rf_blocks_.clear();
  for (size_t i = 0; i < contacts.size(); ++i) {
    const Contact* c = contacts[i];
    for (size_t j = 0; j < i; ++j) {
      if (contacts[j] == c) {
        return false;
      }
    }
    if (c == nullptr || c->Dim() <= 0 || c->Jacobian().cols() != num_qdot_ ||
        c->Jacobian().rows() != c->Dim() ||
        c->JacobianDotQdot().size() != c->Dim() ||
        c->UfMatrix().cols() != c->Dim() ||
        c->UfVector().size() != c->UfMatrix().rows() ||
        !c->Jacobian().allFinite() || !c->JacobianDotQdot().allFinite() ||
        !c->UfMatrix().allFinite() || !c->UfVector().allFinite()) {
      return false;
    }
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
    contact_rf_blocks_.emplace(c, ContactStackBlock{contact_row_offset, dim});

    contact_row_offset += dim;
    uf_row_offset += uf_rows;
  }
  return true;
}

bool WBIC::GetDesiredReactionForce(
    const std::vector<ForceTask*>& force_task_vector) {
  des_rf_.setZero(dim_contact_);
  if (force_task_vector.empty()) {
    return true;
  }

  for (size_t i = 0; i < force_task_vector.size(); ++i) {
    const ForceTask* force_task = force_task_vector[i];
    if (force_task == nullptr || force_task->GetContact() == nullptr) {
      return false;
    }
    const Contact* c = force_task->GetContact();
    for (size_t j = 0; j < i; ++j) {
      const ForceTask* prev_task = force_task_vector[j];
      if (prev_task == nullptr || prev_task->GetContact() == nullptr) {
        return false;
      }
      if (prev_task->GetContact() == c) {
        return false;
      }
    }
    const auto it = contact_rf_blocks_.find(c);
    if (it == contact_rf_blocks_.end()) {
      return false;
    }

    const int dim = force_task->Dim();
    const Eigen::VectorXd& rf_des = force_task->DesiredRf();
    if (dim != it->second.dim || rf_des.size() != dim || !rf_des.allFinite()) {
      return false;
    }
    des_rf_.segment(it->second.rf_offset, dim) = rf_des;
  }
  return true;
}

bool WBIC::IsAxisAlignedConstraint(const Constraint* c) const {
  if (c == nullptr) {
    return false;
  }

  constexpr double kCoeffTol = 1e-12;
  const Eigen::MatrixXd& A_kin = c->ConstraintMatrix();
  if (A_kin.rows() <= 0 || A_kin.cols() != num_active_) {
    return false;
  }

  for (int r = 0; r < A_kin.rows(); ++r) {
    int nonzero_count = 0;
    for (int j = 0; j < A_kin.cols(); ++j) {
      if (std::abs(A_kin(r, j)) > kCoeffTol) {
        ++nonzero_count;
        if (nonzero_count > 1) {
          return false;
        }
      }
    }
    if (nonzero_count != 1) {
      return false;
    }
  }
  return true;
}

bool WBIC::ExtractAxisAlignedBoxBounds(
    const Constraint* c, const Eigen::VectorXd& qddot_ref) {
  constexpr double kInf = std::numeric_limits<double>::infinity();
  constexpr double kCoeffTol = 1e-12;
  l_bnd_.setConstant(-kInf);
  u_bnd_.setConstant(kInf);

  if (c == nullptr) {
    return false;
  }

  const Eigen::MatrixXd& A_kin = c->ConstraintMatrix();
  const Eigen::VectorXd& b_kin = c->ConstraintVector();
  if (A_kin.cols() != num_active_) {
    return false;
  }
  if (qddot_ref.size() != num_qdot_ || b_kin.size() != A_kin.rows()) {
    return false;
  }

  // Fast path: convert one-hot rows into per-axis bounds.
  // If any row is coupled, caller should assemble the dense inequality path.
  for (int r = 0; r < A_kin.rows(); ++r) {
    int idx = -1;
    double coeff = 0.0;
    int nonzero_count = 0;

    for (int j = 0; j < num_active_; ++j) {
      const double a = A_kin(r, j);
      if (std::abs(a) > kCoeffTol) {
        ++nonzero_count;
        if (nonzero_count == 1) {
          idx = num_floating_ + j;
          coeff = a;
        } else {
          return false;
        }
      }
    }

    if (nonzero_count != 1 || idx < 0) {
      return false;
    }

    const double rhs = (b_kin(r) - coeff * qddot_ref(idx)) / coeff;
    if (coeff > 0.0) {
      u_bnd_(idx) = std::min(u_bnd_(idx), rhs);
    } else {
      l_bnd_(idx) = std::max(l_bnd_(idx), rhs);
    }
  }
  return true;
}

int WBIC::BuildKinematicLimitConstraint(const Constraint* c, bool is_soft,
                                        bool use_box_solver,
                                        const Eigen::VectorXd& qddot_ref,
                                        int row, int& slack_col) {
  assert(c != nullptr);
  assert(qddot_ref.size() == num_qdot_);
  const bool axis_aligned = ExtractAxisAlignedBoxBounds(c, qddot_ref);

  if (axis_aligned) {
    return BuildAxisAlignedKinematicLimitConstraint(c, is_soft, use_box_solver,
                                                    row, slack_col);
  }
  return BuildDenseKinematicLimitConstraint(c, is_soft, qddot_ref, row,
                                            slack_col);
}

int WBIC::BuildAxisAlignedKinematicLimitConstraint(const Constraint* c, bool is_soft,
                                                   bool use_box_solver, int row,
                                                   int& slack_col) {
  assert(c != nullptr);
  const std::vector<int>* bounded_indices = nullptr;
  if (c == cached_pos_c_) {
    bounded_indices = &pos_bounded_active_indices_;
  } else if (c == cached_vel_c_) {
    bounded_indices = &vel_bounded_active_indices_;
  } else {
    assert(false && "Unexpected constraint pointer for axis-aligned joint limits");
    return 0;
  }
  const int bounded_dim = static_cast<int>(bounded_indices->size());

  if (is_soft) {
    for (int r = 0; r < bounded_dim; ++r) {
      const int i = (*bounded_indices)[r];
      const int idx = num_floating_ + i;
      C_(row + r, idx) = 1.0;
      C_(row + r, slack_col + r) = -1.0;
      l_(row + r) = l_bnd_(idx);
      u_(row + r) = u_bnd_(idx);
    }
    slack_col += bounded_dim;
    return bounded_dim;
  }

  if (!use_box_solver) {
    for (int r = 0; r < bounded_dim; ++r) {
      const int i = (*bounded_indices)[r];
      const int idx = num_floating_ + i;
      C_(row + r, idx) = 1.0;
      l_(row + r) = l_bnd_(idx);
      u_(row + r) = u_bnd_(idx);
    }
    return bounded_dim;
  }

  for (int r = 0; r < bounded_dim; ++r) {
    const int i = (*bounded_indices)[r];
    const int idx = num_floating_ + i;
    l_box_(idx) = std::max(l_box_(idx), l_bnd_(idx));
    u_box_(idx) = std::min(u_box_(idx), u_bnd_(idx));
  }
  return 0;
}

int WBIC::BuildDenseKinematicLimitConstraint(const Constraint* c, bool is_soft,
                                             const Eigen::VectorXd& qddot_ref,
                                             int row, int& slack_col) {
  constexpr double kInf = std::numeric_limits<double>::infinity();
  assert(c != nullptr);
  assert(qddot_ref.size() == num_qdot_);

  // Dense truth path: A * qddot <= b  ->  A * delta_qddot <= b - A * qddot_ref
  const Eigen::MatrixXd& A_kin = c->ConstraintMatrix();
  const Eigen::VectorXd& b_kin = c->ConstraintVector();
  if (A_kin.rows() == 0 && b_kin.size() == 0) {
    return 0;
  }
  assert(b_kin.size() == A_kin.rows());
  assert(A_kin.rows() > 0);
  const int rows = static_cast<int>(A_kin.rows());

  // Joint position/velocity limits are expressed in actuated-joint coordinates.
  assert(A_kin.cols() == num_active_);
  auto c_block = C_.block(row, 0, rows, num_qdot_);
  c_block.setZero();
  c_block.block(0, num_floating_, rows, num_active_) = A_kin;

  l_.segment(row, rows).setConstant(-kInf);
  u_.segment(row, rows) = b_kin;
  u_.segment(row, rows).noalias() -= c_block * qddot_ref;

  if (is_soft) {
    for (int r = 0; r < rows; ++r) {
      C_(row + r, slack_col + r) = -1.0;
    }
    slack_col += rows;
  }
  return rows;
}

int WBIC::BuildTorqueLimitConstraint(const JointTrqLimitConstraint* c, bool is_soft,
                                      bool use_box_solver,
                                      int row, int& slack_col) {
  assert(c != nullptr);
  const Eigen::MatrixXd& limits = c->EffectiveLimits();
  assert(limits.rows() == num_active_);
  assert(limits.cols() >= 2);
  assert(limits.allFinite());
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
  int conflicts_this_tick = 0;
  int last_conflict_idx = -1;
  for (int i = 0; i < qp_dim; ++i) {
    if (l_box_(i) > u_box_(i)) {
      ++conflicts_this_tick;
      last_conflict_idx = i;
      l_box_(i) = u_box_(i);
    }
  }
  wbic_data_->box_conflict_count_ = conflicts_this_tick;
  wbic_data_->box_last_conflict_index_ = last_conflict_idx;
  wbic_data_->box_conflict_active_ = (conflicts_this_tick > 0);
}

void WBIC::SetQPCost(const WbcFormulation& formulation,
                     const Eigen::VectorXd& qddot_posture_ref) {
  assert(qddot_posture_ref.size() == num_qdot_);
  assert(qddot_posture_ref.allFinite());
  const int qp_dim = num_qdot_ + dim_contact_ + dim_slack_total_;
  // resize() only reallocates when rows*cols changes (stable in steady state).
  H_.setZero(qp_dim, qp_dim);
  g_.setZero(qp_dim);

  const bool force_contact_mode = (dim_contact_ > 0) && !formulation.force_tasks.empty();

  AddOperationalTaskCosts(formulation.operational_tasks, qddot_posture_ref);
  AddNominalAccelTrackingCost(force_contact_mode);
  AddExactTorqueRegularization(qddot_posture_ref);

  if (dim_contact_ > 0) {
    AddReactionForceCost();
    if (contact_mode_ == ContactMode::kSoftTracking) {
      AddContactAccelerationCost(qddot_posture_ref);
    }
  }

  if (dim_slack_total_ > 0) {
    AddSlackVariablePenalties();
  }
}

void WBIC::AddOperationalTaskCosts(const std::vector<Task*>& operational_tasks,
                                   const Eigen::VectorXd& qddot_nominal) {
  assert(qddot_nominal.size() == num_qdot_);
  assert(qddot_nominal.allFinite());
  assert(ValidateOperationalTasks(operational_tasks));
  for (const Task* task : operational_tasks) {
    const Eigen::MatrixXd& J = task->Jacobian();
    const Eigen::VectorXd& JdotQdot = task->JacobianDotQdot();
    const Eigen::VectorXd& a_des = task->OpCommand();
    const int rows = static_cast<int>(J.rows());

    xc_res_scratch_.resize(rows);
    xc_res_scratch_.noalias() = J * qddot_nominal;
    xc_res_scratch_ += JdotQdot;
    xc_res_scratch_ -= a_des;

    const Eigen::VectorXd& w = task->Weight();
    if (w.size() == rows) {
      wJc_scratch_.resize(rows, num_qdot_);
      wJc_scratch_.noalias() = w.asDiagonal() * J;
      H_.topLeftCorner(num_qdot_, num_qdot_).noalias() +=
          J.transpose() * wJc_scratch_;

      tau_cost_scratch_.resize(rows);
      tau_cost_scratch_ = xc_res_scratch_;
      tau_cost_scratch_.array() *= w.array();
      g_.head(num_qdot_).noalias() += J.transpose() * tau_cost_scratch_;
    } else {
      H_.topLeftCorner(num_qdot_, num_qdot_).noalias() +=
          J.transpose() * J;
      g_.head(num_qdot_).noalias() += J.transpose() * xc_res_scratch_;
    }
  }
}

void WBIC::AddNominalAccelTrackingCost(bool force_contact_mode) {
  const auto& w = wbic_data_->qp_params_->W_delta_qddot_;
  tau_cost_scratch_.setOnes(num_qdot_);
  assert((w.size() == 0 || w.size() == num_qdot_));
  if (w.size() == num_qdot_) {
    assert(w.allFinite());
    assert((w.array() >= 0.0).all());
    tau_cost_scratch_ = w;
  }
  if (force_contact_mode) {
    tau_cost_scratch_ *= posture_bias_contact_scale_;
  }
  if (num_floating_ > 0) {
    tau_cost_scratch_.head(num_floating_).setZero();
  }
  H_.diagonal().head(num_qdot_) += tau_cost_scratch_;
}

void WBIC::AddExactTorqueRegularization(const Eigen::VectorXd& qddot_nominal) {
  assert(qddot_nominal.size() == num_qdot_);
  assert(qddot_nominal.allFinite());
  assert(M_.rows() == num_qdot_ && M_.cols() == num_qdot_);
  assert(cori_.size() == num_qdot_ && grav_.size() == num_qdot_);
  assert(M_.allFinite() && cori_.allFinite() && grav_.allFinite());
  // Baseline torque:
  //   tau_0 = M * qddot_nominal + Ni_dyn^T * (cori + grav) - (Jc * Ni_dyn)^T * des_rf
  //   tau_0 = M * qddot_nominal + (cori + grav) - Jc^T * des_rf
  // NOTE: Ni_dyn (passive-joint nullspace) is identity for all supported robots.
  tau_0_.noalias() = M_ * qddot_nominal;
  tau_0_ += cori_ + grav_;
  if (dim_contact_ > 0) {
    assert(des_rf_.size() == dim_contact_);
    tau_0_.noalias() -= Jc_.transpose() * des_rf_;
  }

  const auto& w_tau = wbic_data_->qp_params_->W_tau_;
  const auto& w_tau_dot = wbic_data_->qp_params_->W_tau_dot_;

  auto add_weighted_torque_term =
      [&](const Eigen::VectorXd& w_diag, const Eigen::VectorXd& c_vec) {
        if (w_diag.size() == 0 || w_diag.squaredNorm() <= 0.0) {
          return;
        }
        assert(w_diag.size() == num_qdot_);
        assert(w_diag.allFinite());
        assert((w_diag.array() >= 0.0).all());
        assert(c_vec.allFinite());

        // qddot block: M^T W M
        wM_scratch_.noalias() = w_diag.asDiagonal() * M_;
        H_.topLeftCorner(num_qdot_, num_qdot_).noalias() +=
            M_.transpose() * wM_scratch_;

        // gradient qddot block: M^T W c
        tau_cost_scratch_ = w_diag.cwiseProduct(c_vec);
        g_.head(num_qdot_).noalias() += M_.transpose() * tau_cost_scratch_;

        if (dim_contact_ <= 0) {
          return;
        }

        // rf coupling blocks with B = -Jc^T:
        // H_q_rf += M^T W B, H_rf_rf += B^T W B, g_rf += B^T W c
        wJc_scratch_.resize(num_qdot_, dim_contact_);
        wJc_scratch_.noalias() = -(w_diag.asDiagonal() * Jc_.transpose());

        H_.block(0, num_qdot_, num_qdot_, dim_contact_).noalias() +=
            M_.transpose() * wJc_scratch_;
        H_.block(num_qdot_, 0, dim_contact_, num_qdot_).noalias() +=
            wJc_scratch_.transpose() * M_;
        H_.block(num_qdot_, num_qdot_, dim_contact_, dim_contact_).noalias() +=
            (-Jc_) * wJc_scratch_;

        g_.segment(num_qdot_, dim_contact_).noalias() +=
            (-Jc_) * tau_cost_scratch_;
      };

  add_weighted_torque_term(w_tau, tau_0_);
  add_weighted_torque_term(w_tau_dot, tau_0_ - wbic_data_->tau_prev_);
}

void WBIC::AddContactAccelerationCost(const Eigen::VectorXd& qddot_nominal) {
  if (dim_contact_ <= 0) {
    return;
  }
  assert(qddot_nominal.size() == num_qdot_);
  assert(qddot_nominal.allFinite());
  assert(Jc_.rows() == dim_contact_ && Jc_.cols() == num_qdot_);
  assert(JcDotQdot_.size() == dim_contact_);
  assert(Jc_.allFinite() && JcDotQdot_.allFinite());
  const auto& w_xc = wbic_data_->qp_params_->W_xc_ddot_;
  assert((w_xc.size() == 0 || w_xc.size() == dim_contact_));
  if (w_xc.size() == dim_contact_) {
    assert(w_xc.allFinite());
    assert((w_xc.array() >= 0.0).all());
    wJc_scratch_.noalias() = w_xc.asDiagonal() * Jc_;
  } else {
    wJc_scratch_ = Jc_;
  }
  H_.topLeftCorner(num_qdot_, num_qdot_).noalias() += Jc_.transpose() * wJc_scratch_;

  xc_res_scratch_.noalias() = Jc_ * qddot_nominal;
  xc_res_scratch_ += JcDotQdot_;
  if (w_xc.size() == dim_contact_) {
    xc_res_scratch_.array() *= w_xc.array();
  }
  g_.head(num_qdot_).noalias() += Jc_.transpose() * xc_res_scratch_;
}

void WBIC::AddReactionForceCost() {
  const auto& w_rf = wbic_data_->qp_params_->W_delta_rf_;
  const auto& w_fd = wbic_data_->qp_params_->W_f_dot_;
  assert(des_rf_.size() == dim_contact_);
  assert(des_rf_.allFinite());
  assert((w_rf.size() == 0 || w_rf.size() == dim_contact_));
  assert((w_fd.size() == 0 || w_fd.size() == dim_contact_));
  assert(w_rf.allFinite());
  assert(w_fd.allFinite());
  assert((w_rf.array() >= 0.0).all());
  assert((w_fd.array() >= 0.0).all());
  if (wbic_data_->rf_prev_sol_.size() != dim_contact_) {
    wbic_data_->rf_prev_sol_.setZero(dim_contact_);
  }

  auto rf_diag = H_.diagonal().segment(num_qdot_, dim_contact_);
  if (w_rf.size() == dim_contact_) rf_diag += w_rf;
  else                             rf_diag.array() += 1.0;
  if (w_fd.size() == dim_contact_) rf_diag += w_fd;
  else                             rf_diag.array() += 1.0;

  if (w_fd.size() == dim_contact_) {
    g_.segment(num_qdot_, dim_contact_) += w_fd.cwiseProduct(des_rf_ - wbic_data_->rf_prev_sol_);
  } else {
    g_.segment(num_qdot_, dim_contact_) += des_rf_ - wbic_data_->rf_prev_sol_;
  }
}

void WBIC::AddSlackVariablePenalties() {
  assert(std::isfinite(soft_params_.w_pos) && soft_params_.w_pos >= 0.0);
  assert(std::isfinite(soft_params_.w_vel) && soft_params_.w_vel >= 0.0);
  assert(std::isfinite(soft_params_.w_trq) && soft_params_.w_trq >= 0.0);
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

void WBIC::SetQPEqualityConstraint(const Eigen::VectorXd& qddot_ref) {
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
    b_.noalias() = -sf_ * (M_ * qddot_ref + cori_ + grav_);
    if (dim_contact_ > 0) {
      b_.noalias() += sf_ * Jc_.transpose() * des_rf_;
    }
  } else {
    // Fixed-base: no equality constraints (0 rows)
    b_.resize(0);
  }
}

WBIC::InequalityMode WBIC::BuildInequalityMode() const {
  InequalityMode mode;
  const JointPosLimitConstraint* pos_c = cached_pos_c_;
  const JointVelLimitConstraint* vel_c = cached_vel_c_;
  const JointTrqLimitConstraint* trq_c = cached_trq_c_;

  mode.pos_axis_aligned = pos_c && IsAxisAlignedConstraint(pos_c);
  mode.vel_axis_aligned = vel_c && IsAxisAlignedConstraint(vel_c);

  const bool has_any_hard =
      (pos_c && !soft_params_.pos) ||
      (vel_c && !soft_params_.vel) ||
      (trq_c && !soft_params_.trq);
  const bool force_exact_trq_dense =
      (trq_c && !soft_params_.trq &&
       (hard_torque_limit_mode_ == HardTorqueLimitMode::EXACT_DENSE ||
        dim_contact_ > 0));
  const bool force_dense_kinematic =
      (pos_c && !soft_params_.pos && !mode.pos_axis_aligned) ||
      (vel_c && !soft_params_.vel && !mode.vel_axis_aligned);

  // When slack (soft) variables exist, disable box solver to avoid
  // ProxQP ADMM scaling collapse between huge box bounds and unbounded slacks.
  // Exact hard torque mode and dense kinematic rows also force dense inequalities.
  mode.force_dense =
      (dim_slack_total_ > 0) || force_exact_trq_dense || force_dense_kinematic;
  mode.use_box_solver = has_any_hard && !mode.force_dense;

  const auto kinematic_row_count =
      [&](const Constraint* c, bool axis_aligned, bool is_soft) {
        if (c == nullptr) {
          return 0;
        }
        if (is_soft || !mode.use_box_solver) {
          if (axis_aligned) {
            if (c == cached_pos_c_) {
              return static_cast<int>(pos_bounded_active_indices_.size());
            }
            if (c == cached_vel_c_) {
              return static_cast<int>(vel_bounded_active_indices_.size());
            }
          }
          return static_cast<int>(c->ConstraintMatrix().rows());
        }
        return 0;
      };

  mode.n_friction = (dim_contact_ > 0) ? Uf_mat_.rows() : 0;
  mode.n_ineq = mode.n_friction;
  mode.n_ineq +=
      kinematic_row_count(pos_c, mode.pos_axis_aligned, soft_params_.pos);
  mode.n_ineq +=
      kinematic_row_count(vel_c, mode.vel_axis_aligned, soft_params_.vel);
  if (trq_c && (soft_params_.trq || !mode.use_box_solver)) {
    mode.n_ineq += num_active_;
  }
  return mode;
}

void WBIC::ResizeInequalityStorage(const InequalityMode& mode, int qp_dim) {
  constexpr double kInf = std::numeric_limits<double>::infinity();
  if (mode.use_box_solver) {
    l_box_.resize(qp_dim);
    u_box_.resize(qp_dim);
    l_box_.setConstant(-kInf);
    u_box_.setConstant(kInf);
  } else {
    l_box_.resize(0);
    u_box_.resize(0);
  }

  C_.resize(mode.n_ineq, qp_dim);
  C_.setZero();
  l_.resize(mode.n_ineq);
  l_.setZero();
  u_.resize(mode.n_ineq);
  u_.setConstant(kInf);
}

void WBIC::AssembleInequalityRows(const InequalityMode& mode,
                                  const Eigen::VectorXd& qddot_ref) {
  assert(qddot_ref.size() == num_qdot_);
  assert(qddot_ref.allFinite());
  const int qp_dim = num_qdot_ + dim_contact_ + dim_slack_total_;
  const JointPosLimitConstraint* pos_c = cached_pos_c_;
  const JointVelLimitConstraint* vel_c = cached_vel_c_;
  const JointTrqLimitConstraint* trq_c = cached_trq_c_;

  int row = 0;
  int slack_col = num_qdot_ + dim_contact_;
  if (mode.n_friction > 0) {
    row += BuildFrictionConeConstraint(row);
  }
  if (pos_c) {
    const int added = BuildKinematicLimitConstraint(
        pos_c, soft_params_.pos, mode.use_box_solver, qddot_ref, row, slack_col);
    assert(added >= 0);
    row += added;
  }
  if (vel_c) {
    const int added = BuildKinematicLimitConstraint(
        vel_c, soft_params_.vel, mode.use_box_solver, qddot_ref, row, slack_col);
    assert(added >= 0);
    row += added;
  }
  if (trq_c) {
    const int added = BuildTorqueLimitConstraint(
        trq_c, soft_params_.trq, mode.use_box_solver, row, slack_col);
    assert(added >= 0);
    row += added;
  }
  assert(row == mode.n_ineq);
  if (mode.use_box_solver) {
    EnforceBoxFeasibilityGuard(qp_dim);
  }
}

void WBIC::SetQPInEqualityConstraint(const Eigen::VectorXd& qddot_ref) {
  assert(qddot_ref.size() == num_qdot_);
  assert(qddot_ref.allFinite());
  const int qp_dim = num_qdot_ + dim_contact_ + dim_slack_total_;
  wbic_data_->box_conflict_count_ = 0;
  wbic_data_->box_last_conflict_index_ = -1;
  wbic_data_->box_conflict_active_ = false;

  const InequalityMode mode = BuildInequalityMode();
  ResizeInequalityStorage(mode, qp_dim);
  AssembleInequalityRows(mode, qddot_ref);
}

bool WBIC::SolveQP(const Eigen::VectorXd& qddot_ref) {
  if (qddot_ref.size() != num_qdot_ || !qddot_ref.allFinite()) {
    return false;
  }
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
  wbic_data_->qddot_sol_ = qddot_ref + wbic_data_->delta_qddot_;

  if (dim_contact_ > 0) {
    wbic_data_->delta_rf_ = qp_sol.segment(num_qdot_, dim_contact_);
    wbic_data_->rf_sol_.noalias() = des_rf_ + wbic_data_->delta_rf_;
    wbic_data_->rf_prev_sol_ = wbic_data_->rf_sol_;
    wbic_data_->Xc_ddot_.noalias() = Jc_ * wbic_data_->qddot_sol_;
    wbic_data_->Xc_ddot_ += JcDotQdot_;
  } else {
    wbic_data_->delta_rf_.resize(0);
    wbic_data_->rf_sol_.resize(0);
    wbic_data_->rf_prev_sol_.resize(0);
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

bool WBIC::GetSolution(Eigen::VectorXd& jtrq_cmd) {
  // Torque equation: tau = M * qddot + (cori + grav) - Jc^T * rf
  // NOTE: Ni_dyn (passive-joint nullspace) is identity for all supported robots.
  tau_gen_sol_.noalias() = M_ * wbic_data_->qddot_sol_;
  tau_gen_sol_ += cori_ + grav_;
  if (dim_contact_ > 0 && wbic_data_->rf_sol_.size() > 0) {
    tau_gen_sol_.noalias() -= Jc_.transpose() * wbic_data_->rf_sol_;
  }
  if (!tau_gen_sol_.allFinite()) {
    return false;
  }

  // Store full-DOF torque for next-tick rate-of-change cost.
  wbic_data_->tau_prev_ = tau_gen_sol_;

  if (num_floating_ == 0 && num_active_ == num_qdot_) {
    // Fixed-base fully-actuated shortcut: direct projection to command space.
    jtrq_cmd = snf_ * tau_gen_sol_;
    return jtrq_cmd.allFinite();
  }

  const bool pinv_ok = WeightedPseudoInverse(sa_, Minv_, sa_pinv_scratch_);
  assert(pinv_ok);
  if (!pinv_ok) {
    return false;
  }
  jtrq_cmd = sa_pinv_scratch_.transpose() * tau_gen_sol_;
  jtrq_cmd = snf_ * sa_.transpose() * jtrq_cmd;
  return jtrq_cmd.allFinite();
}

} // namespace wbc
