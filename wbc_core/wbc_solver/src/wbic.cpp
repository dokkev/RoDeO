/**
 * @file wbc_core/wbc_solver/src/wbic.cpp
 * @brief Doxygen documentation for wbic module.
 */
#include "wbc_solver/wbic.hpp"

#include <cassert>
#include <cmath>
#include <iterator>
#include <limits>

#include "wbc_formulation/kinematic_constraint.hpp"
#include "wbc_util/constants.hpp"

namespace wbc {
namespace {
// Variable Damped Least Squares (DLS) pseudo-inverse.
// Reference: Chiaverini 1997, Nakamura & Hanafusa 1986.
//
// When the minimum singular value drops below `threshold`, a damping factor
// lambda^2 ramps up smoothly from 0 to lambda_max^2, preventing the inverse
// from blowing up near singularities.
//
//   s_inv_i = s_i / (s_i^2 + lambda^2)
//
// Far from singularity (min_s >> threshold): lambda=0 → exact pseudo-inverse.
// At singularity (min_s → 0): lambda → lambda_max → bounded output.
constexpr double kDlsLambdaMax = 0.05;

Eigen::MatrixXd PseudoInverseSvd(const Eigen::MatrixXd& m, double threshold) {
  if (m.size() == 0) {
    return Eigen::MatrixXd::Zero(m.cols(), m.rows());
  }

  const Eigen::JacobiSVD<Eigen::MatrixXd> svd(
      m, Eigen::ComputeThinU | Eigen::ComputeThinV);
  const Eigen::VectorXd& s = svd.singularValues();
  Eigen::VectorXd s_inv = Eigen::VectorXd::Zero(s.size());

  // Compute adaptive damping from the minimum singular value.
  double lambda_sq = 0.0;
  if (s.size() > 0 && threshold > 0.0) {
    const double min_s = s(s.size() - 1);  // SVD returns descending order
    if (min_s < threshold) {
      const double ratio = min_s / threshold;
      lambda_sq = (1.0 - ratio * ratio) * (kDlsLambdaMax * kDlsLambdaMax);
    }
  }

  for (int i = 0; i < s.size(); ++i) {
    const double denom = s[i] * s[i] + lambda_sq;
    s_inv[i] = (denom > 1e-18) ? s[i] / denom : 0.0;
  }

  return svd.matrixV() * s_inv.asDiagonal() * svd.matrixU().transpose();
}

Eigen::MatrixXd WeightedPinv(const Eigen::MatrixXd& J, const Eigen::MatrixXd& W,
                             double threshold) {
  if (J.size() == 0) {
    return Eigen::MatrixXd::Zero(J.cols(), J.rows());
  }
  const Eigen::MatrixXd JWJt = J * W * J.transpose();
  const Eigen::MatrixXd JWJt_pinv = PseudoInverseSvd(JWJt, threshold);
  return W * J.transpose() * JWJt_pinv;
}

Eigen::MatrixXd NullSpaceProjector(const Eigen::MatrixXd& J, double threshold,
                                   const Eigen::MatrixXd* W) {
  if (J.cols() == 0) {
    return Eigen::MatrixXd();
  }
  Eigen::MatrixXd Jbar;
  if (W == nullptr) {
    Jbar = PseudoInverseSvd(J, threshold);
  } else {
    Jbar = WeightedPinv(J, *W, threshold);
  }
  return Eigen::MatrixXd::Identity(J.cols(), J.cols()) - Jbar * J;
}

Eigen::MatrixXd DiagOrIdentity(const Eigen::VectorXd& w, int dim) {
  if (w.size() == dim) {
    return w.asDiagonal();
  }
  return Eigen::MatrixXd::Identity(dim, dim);
}

} // namespace

WBIC::WBIC(const std::vector<bool>& act_qdot_list, QPParams* qp_params)
    : WBC(act_qdot_list),
      threshold_(kDefaultSvdThreshold),
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
      u_bnd_(Eigen::VectorXd::Zero(num_qdot_)) {}

bool WBIC::FindConfiguration(
    const WbcFormulation& formulation, const Eigen::VectorXd& curr_jpos,
    Eigen::VectorXd& jpos_cmd, Eigen::VectorXd& jvel_cmd,
    Eigen::VectorXd& wbc_qddot_cmd) {
  if (!settings_updated_) {
    return false;
  }

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
  prev_qddot_cmd_.setZero();

  if (has_contact_) {
    int total_contact_dim = 0;
    for (const auto* c : contact_vector) {
      total_contact_dim += c->Dim();
    }

    stacked_contact_jacobian_.resize(total_contact_dim, num_qdot_);
    stacked_contact_jdot_qdot_.resize(total_contact_dim);
    stacked_contact_op_cmd_.resize(total_contact_dim);

    int row_offset = 0;
    for (const auto* c : contact_vector) {
      const int dim = c->Dim();
      stacked_contact_jacobian_.block(row_offset, 0, dim, num_qdot_) = c->Jacobian();
      stacked_contact_jdot_qdot_.segment(row_offset, dim) = c->JacobianDotQdot();
      stacked_contact_op_cmd_.segment(row_offset, dim) = c->OpCommand();
      row_offset += dim;
    }

    BuildProjectionMatrix(stacked_contact_jacobian_, N_pre_);
    BuildProjectionMatrix(stacked_contact_jacobian_, N_pre_dyn_, &Minv_);
    WeightedPseudoInverse(stacked_contact_jacobian_, Minv_, Jc_bar_);
    qddot_cmd_ = Jc_bar_ * (stacked_contact_op_cmd_ - stacked_contact_jdot_qdot_);
  } else {
    qddot_cmd_.setZero();
  }

  for (auto it = task_vector.begin(); it != task_vector.end(); ++it) {
    Task* task = *it;
    const Eigen::MatrixXd& Jt = task->Jacobian();
    const Eigen::VectorXd& JtDotQdot = task->JacobianDotQdot();
    // Use pre-allocated member buffers to avoid per-tick heap allocation.
    JtPre_.noalias() = Jt * N_pre_;
    PseudoInverse(JtPre_, JtPre_pinv_);
    JtPre_dyn_.noalias() = Jt * N_pre_dyn_;
    WeightedPseudoInverse(JtPre_dyn_, Minv_, JtPre_bar_);

    if (it == task_vector.begin()) {
      delta_q_cmd_ =
          JtPre_pinv_ * task->KpIK().cwiseProduct(task->LocalPosError());
      qdot_cmd_ = JtPre_pinv_ * task->DesiredVel();
      qddot_cmd_ = qddot_cmd_ +
                   JtPre_bar_ *
                       (task->OpCommand() - JtDotQdot - Jt * qddot_cmd_);
    } else {
      delta_q_cmd_ =
          prev_delta_q_cmd_ +
          JtPre_pinv_ * (task->KpIK().cwiseProduct(task->LocalPosError()) -
                         Jt * prev_delta_q_cmd_);
      qdot_cmd_ = prev_qdot_cmd_ +
                  JtPre_pinv_ * (task->DesiredVel() - Jt * prev_qdot_cmd_);
      qddot_cmd_ =
          prev_qddot_cmd_ +
          JtPre_bar_ * (task->OpCommand() - JtDotQdot - Jt * prev_qddot_cmd_);
    }

    if (std::next(it) != task_vector.end()) {
      prev_delta_q_cmd_ = delta_q_cmd_;
      prev_qdot_cmd_ = qdot_cmd_;
      prev_qddot_cmd_ = qddot_cmd_;
      BuildProjectionMatrix(JtPre_, N_nx_);
      N_pre_ *= N_nx_;
      BuildProjectionMatrix(JtPre_dyn_, N_nx_dyn_, &Minv_);
      N_pre_dyn_ *= N_nx_dyn_;
    } else {
      // Sanity check before writing outputs — propagated NaN from a degenerate
      // Jacobian would silently corrupt all downstream torque commands.
      if (!delta_q_cmd_.allFinite() || !qdot_cmd_.allFinite() ||
          !qddot_cmd_.allFinite()) {
        return false;
      }
      jpos_cmd = curr_jpos + delta_q_cmd_.tail(num_qdot_ - num_floating_);
      jvel_cmd = qdot_cmd_.tail(num_qdot_ - num_floating_);
      wbc_qddot_cmd = qddot_cmd_;
    }
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

  // Compute slack dimensions based on constraint types and soft flags.
  // Slack variables are added only for constraints that exist AND are set to soft.
  dim_slack_pos_ = 0;
  dim_slack_vel_ = 0;
  dim_slack_trq_ = 0;
  for (const Constraint* c : formulation.kinematic_constraints) {
    if (dynamic_cast<const JointPosLimitConstraint*>(c) && soft_params_.pos) {
      dim_slack_pos_ = num_active_;
    } else if (dynamic_cast<const JointVelLimitConstraint*>(c) && soft_params_.vel) {
      dim_slack_vel_ = num_active_;
    } else if (dynamic_cast<const JointTrqLimitConstraint*>(c) && soft_params_.trq) {
      dim_slack_trq_ = num_active_;
    }
  }
  dim_slack_total_ = dim_slack_pos_ + dim_slack_vel_ + dim_slack_trq_;

  // Always run QP — even without contacts, delta_qddot regularizes torque
  SetQPCost(wbc_qddot_cmd);
  SetQPEqualityConstraint(wbc_qddot_cmd);
  SetQPInEqualityConstraint(formulation, wbc_qddot_cmd);
  if (!SolveQP(wbc_qddot_cmd)) {
    return false;
  }

  GetSolution(wbc_qddot_cmd, jtrq_cmd);
  return true;
}

void WBIC::PseudoInverse(const Eigen::MatrixXd& jac, Eigen::MatrixXd& jac_inv) {
  jac_inv = PseudoInverseSvd(jac, threshold_);
}

void WBIC::WeightedPseudoInverse(const Eigen::MatrixXd& jac,
                                 const Eigen::MatrixXd& W,
                                 Eigen::MatrixXd& jac_bar) {
  jac_bar = WeightedPinv(jac, W, threshold_);
}

void WBIC::BuildProjectionMatrix(const Eigen::MatrixXd& jac, Eigen::MatrixXd& N,
                                 const Eigen::MatrixXd* W) {
  N = NullSpaceProjector(jac, threshold_, W);
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
  Uf_mat_ = Eigen::MatrixXd::Zero(total_uf_rows, total_contact_dim);
  Uf_vec_.resize(total_uf_rows);

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
    des_rf_ = Eigen::VectorXd::Zero(dim_contact_);
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

void WBIC::SetQPCost(const Eigen::VectorXd& wbc_qddot_cmd) {
  const int qp_dim = num_qdot_ + dim_contact_ + dim_slack_total_;

  H_.setZero(qp_dim, qp_dim);
  g_.setZero(qp_dim);

  // Term 1: delta_qddot tracking (all DOFs)
  const Eigen::MatrixXd w_dq =
      DiagOrIdentity(wbic_data_->qp_params_->W_delta_qddot_, num_qdot_);
  H_.topLeftCorner(num_qdot_, num_qdot_) += w_dq;

  // Baseline torque before QP correction (used by torque cost and torque constraint).
  // tau_0 = M * qddot_ik + Ni_dyn^T * (cori + grav) - (Jc * Ni_dyn)^T * des_rf
  tau_0_ = M_ * wbc_qddot_cmd + Ni_dyn_.transpose() * (cori_ + grav_);
  if (has_contact_ && des_rf_.size() > 0) {
    tau_0_ -= (Jc_ * Ni_dyn_).transpose() * des_rf_;
  }

  // Term 2: torque minimization
  // d(tau)/d(delta_qddot) = M → cost = ||tau_0 + M*dx||^2_W_tau
  const Eigen::VectorXd& w_tau = wbic_data_->qp_params_->W_tau_;
  const Eigen::VectorXd& w_tau_dot = wbic_data_->qp_params_->W_tau_dot_;
  const bool use_tau_cost =
      (w_tau.squaredNorm() > 0.0) || (w_tau_dot.squaredNorm() > 0.0);
  if (use_tau_cost) {
    const Eigen::MatrixXd MtWM = [&]() {
      Eigen::MatrixXd acc =
          Eigen::MatrixXd::Zero(num_qdot_, num_qdot_);
      if (w_tau.squaredNorm() > 0.0) {
        acc.noalias() += M_.transpose() * w_tau.asDiagonal() * M_;
      }
      if (w_tau_dot.squaredNorm() > 0.0) {
        acc.noalias() += M_.transpose() * w_tau_dot.asDiagonal() * M_;
      }
      return acc;
    }();
    H_.topLeftCorner(num_qdot_, num_qdot_) += MtWM;
    // g: M^T * W_tau * tau_0  +  M^T * W_tau_dot * (tau_0 - tau_prev)
    if (w_tau.squaredNorm() > 0.0) {
      g_.head(num_qdot_) += M_.transpose() * w_tau.asDiagonal() * tau_0_;
    }
    if (w_tau_dot.squaredNorm() > 0.0) {
      g_.head(num_qdot_) += M_.transpose() * w_tau_dot.asDiagonal() *
                            (tau_0_ - wbic_data_->tau_prev_);
    }
  }

  // Term 3: contact acceleration minimization (if contacts)
  if (has_contact_ && dim_contact_ > 0) {
    const Eigen::MatrixXd xc_w =
        DiagOrIdentity(wbic_data_->qp_params_->W_xc_ddot_, Jc_.rows());
    // Jc is (dim_contact x num_qdot), so Jc^T * W * Jc is (num_qdot x num_qdot)
    H_.topLeftCorner(num_qdot_, num_qdot_) += Jc_.transpose() * xc_w * Jc_;
    g_.head(num_qdot_) +=
        Jc_.transpose() * xc_w * (Jc_ * wbc_qddot_cmd + JcDotQdot_);

    // Term 4: reaction force (delta_rf + force rate of change)
    H_.block(num_qdot_, num_qdot_, dim_contact_, dim_contact_) =
        (DiagOrIdentity(wbic_data_->qp_params_->W_delta_rf_, dim_contact_).diagonal() +
         DiagOrIdentity(wbic_data_->qp_params_->W_f_dot_, dim_contact_).diagonal())
            .asDiagonal();
    g_.segment(num_qdot_, dim_contact_) +=
        DiagOrIdentity(wbic_data_->qp_params_->W_f_dot_, dim_contact_) *
        (des_rf_ - wbic_data_->rf_prev_cmd_);
  }

  // Term 5: Slack variable quadratic penalties
  if (dim_slack_total_ > 0) {
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

void WBIC::SetQPInEqualityConstraint(const WbcFormulation& formulation,
                                     const Eigen::VectorXd& wbc_qddot_cmd) {
  const int qp_dim = num_qdot_ + dim_contact_ + dim_slack_total_;
  const int n_friction = has_contact_ ? Uf_mat_.rows() : 0;
  constexpr double kInf = std::numeric_limits<double>::infinity();

  // --- Collect constraint pointers by type (deterministic ordering) ---
  const JointPosLimitConstraint* pos_c = nullptr;
  const JointVelLimitConstraint* vel_c = nullptr;
  const JointTrqLimitConstraint* trq_c = nullptr;
  for (const Constraint* c : formulation.kinematic_constraints) {
    if (auto* p = dynamic_cast<const JointPosLimitConstraint*>(c)) pos_c = p;
    else if (auto* v = dynamic_cast<const JointVelLimitConstraint*>(c)) vel_c = v;
    else if (auto* t = dynamic_cast<const JointTrqLimitConstraint*>(c)) trq_c = t;
  }

  // --- Count general inequality rows (friction + soft constraint rows) ---
  int n_soft_rows = 0;
  if (pos_c && soft_params_.pos) n_soft_rows += num_active_;
  if (vel_c && soft_params_.vel) n_soft_rows += num_active_;
  if (trq_c && soft_params_.trq) n_soft_rows += num_active_;
  const int n_ineq = n_friction + n_soft_rows;

  // --- Determine if any hard constraint needs box mode ---
  const bool has_any_hard =
      (pos_c && !soft_params_.pos) ||
      (vel_c && !soft_params_.vel) ||
      (trq_c && !soft_params_.trq);

  // --- Box constraints ---
  if (has_any_hard) {
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

  // --- Friction cone constraints ---
  if (n_friction > 0) {
    C_.block(0, num_qdot_, n_friction, dim_contact_) = Uf_mat_;
    l_.head(n_friction) = Uf_vec_ - Uf_mat_ * des_rf_;
    row += n_friction;
  }

  // --- Joint Position Limit ---
  if (pos_c) {
    ExtractBoxBounds(pos_c, wbc_qddot_cmd);
    if (soft_params_.pos) {
      // Soft: l_bnd <= Δqddot[idx] - ε_pos[i] <= u_bnd (general inequality + slack)
      for (int i = 0; i < num_active_; ++i) {
        const int idx = num_floating_ + i;
        C_(row + i, idx) = 1.0;
        C_(row + i, slack_col + i) = -1.0;
        l_(row + i) = l_bnd_(idx);
        u_(row + i) = u_bnd_(idx);
      }
      row += num_active_;
      slack_col += num_active_;
    } else {
      // Hard: intersect into box bounds
      for (int i = 0; i < num_qdot_; ++i) {
        l_box_(i) = std::max(l_box_(i), l_bnd_(i));
        u_box_(i) = std::min(u_box_(i), u_bnd_(i));
      }
    }
  }

  // --- Joint Velocity Limit ---
  if (vel_c) {
    ExtractBoxBounds(vel_c, wbc_qddot_cmd);
    if (soft_params_.vel) {
      // Soft: l_bnd <= Δqddot[idx] - ε_vel[i] <= u_bnd
      for (int i = 0; i < num_active_; ++i) {
        const int idx = num_floating_ + i;
        C_(row + i, idx) = 1.0;
        C_(row + i, slack_col + i) = -1.0;
        l_(row + i) = l_bnd_(idx);
        u_(row + i) = u_bnd_(idx);
      }
      row += num_active_;
      slack_col += num_active_;
    } else {
      // Hard: intersect into box bounds
      for (int i = 0; i < num_qdot_; ++i) {
        l_box_(i) = std::max(l_box_(i), l_bnd_(i));
        u_box_(i) = std::min(u_box_(i), u_bnd_(i));
      }
    }
  }

  // --- Joint Torque Limit ---
  if (trq_c) {
    const Eigen::MatrixXd& limits = trq_c->EffectiveLimits();
    const Eigen::VectorXd sa_tau0 = sa_ * tau_0_;
    if (soft_params_.trq) {
      // Soft: full coupled dynamics in C matrix
      // τ = τ_0 + M*Δqddot - (Jc*Ni_dyn)^T*Δrf
      // → τ_min - Sa*τ_0 <= Sa*M*Δqddot - Sa*(Jc*Ni_dyn)^T*Δrf - ε_trq <= τ_max - Sa*τ_0
      C_.block(row, 0, num_active_, num_qdot_) = sa_ * M_;
      if (dim_contact_ > 0) {
        C_.block(row, num_qdot_, num_active_, dim_contact_) =
            -sa_ * (Jc_ * Ni_dyn_).transpose();
      }
      for (int i = 0; i < num_active_; ++i) {
        C_(row + i, slack_col + i) = -1.0;
        l_(row + i) = limits(i, 0) - sa_tau0(i);
        u_(row + i) = limits(i, 1) - sa_tau0(i);
      }
      row += num_active_;
      slack_col += num_active_;
    } else {
      // Hard: diagonal mass approximation as box bounds
      // τ_i ≈ τ_0_i + M_ii * Δqddot_i
      // → (τ_min_i - τ_0_i) / M_ii <= Δqddot_i <= (τ_max_i - τ_0_i) / M_ii
      for (int i = 0; i < num_active_; ++i) {
        const int idx = num_floating_ + i;
        const double M_diag = M_(idx, idx);
        if (M_diag < 1e-10) continue;
        l_box_(idx) = std::max(l_box_(idx), (limits(i, 0) - sa_tau0(i)) / M_diag);
        u_box_(idx) = std::min(u_box_(idx), (limits(i, 1) - sa_tau0(i)) / M_diag);
      }
    }
  }

  // --- Feasibility guard for hard bounds ---
  if (has_any_hard) {
    for (int i = 0; i < qp_dim; ++i) {
      if (l_box_(i) > u_box_(i)) {
        l_box_(i) = u_box_(i);
      }
    }
  }
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
    qp_solver_->settings.eps_abs = 1e-5;
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
    wbic_data_->rf_cmd_ = des_rf_ + wbic_data_->delta_rf_;
    wbic_data_->rf_prev_cmd_ = wbic_data_->rf_cmd_;
    wbic_data_->Xc_ddot_ =
        Jc_ * wbic_data_->corrected_wbc_qddot_cmd_ + JcDotQdot_;
  } else {
    wbic_data_->delta_rf_.resize(0);
    wbic_data_->rf_cmd_.resize(0);
    wbic_data_->Xc_ddot_.resize(0);
  }

  // Compute diagnostic costs
  const Eigen::MatrixXd w_qddot =
      DiagOrIdentity(wbic_data_->qp_params_->W_delta_qddot_, num_qdot_);
  wbic_data_->delta_qddot_cost_ =
      wbic_data_->delta_qddot_.transpose() * w_qddot * wbic_data_->delta_qddot_;

  if (dim_contact_ > 0) {
    const Eigen::MatrixXd w_rf =
        DiagOrIdentity(wbic_data_->qp_params_->W_delta_rf_, dim_contact_);
    const Eigen::MatrixXd w_xc =
        DiagOrIdentity(wbic_data_->qp_params_->W_xc_ddot_, Jc_.rows());
    wbic_data_->delta_rf_cost_ =
        wbic_data_->delta_rf_.transpose() * w_rf * wbic_data_->delta_rf_;
    wbic_data_->Xc_ddot_cost_ =
        wbic_data_->Xc_ddot_.transpose() * w_xc * wbic_data_->Xc_ddot_;
  } else {
    wbic_data_->delta_rf_cost_ = 0.0;
    wbic_data_->Xc_ddot_cost_ = 0.0;
  }

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
