#include "wbc_solver/wbic.hpp"

#include <cassert>
#include <cmath>
#include <iterator>
#include <limits>

#include "wbc_util/constants.hpp"

namespace wbc {
namespace {
Eigen::MatrixXd PseudoInverseSvd(const Eigen::MatrixXd& m, double threshold) {
  if (m.size() == 0) {
    return Eigen::MatrixXd::Zero(m.cols(), m.rows());
  }
  const Eigen::JacobiSVD<Eigen::MatrixXd> svd(
      m, Eigen::ComputeThinU | Eigen::ComputeThinV);
  const Eigen::VectorXd s = svd.singularValues();
  Eigen::VectorXd s_inv = Eigen::VectorXd::Zero(s.size());
  for (int i = 0; i < s.size(); ++i) {
    if (s[i] > threshold) {
      s_inv[i] = 1.0 / s[i];
    }
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
      wbic_data_(std::make_unique<WBICData>(num_floating_, num_qdot_, qp_params)),
      qp_dim_cache_(-1),
      qp_eq_cache_(-1),
      qp_ineq_cache_(-1) {}

bool WBIC::FindConfiguration(
    const WbcFormulation& formulation, const Eigen::VectorXd& curr_jpos,
    Eigen::VectorXd& jpos_cmd, Eigen::VectorXd& jvel_cmd,
    Eigen::VectorXd& wbc_qddot_cmd) {
  if (!b_update_setting_) {
    return false;
  }

  const std::vector<Task*>& task_vector = formulation.motion_tasks;
  const std::vector<Contact*>& contact_vector = formulation.contact_constraints;

  if (task_vector.empty()) {
    return false;
  }
  b_contact_ = !contact_vector.empty();

  Ni_dyn_ = Eigen::MatrixXd::Identity(num_qdot_, num_qdot_);

  Eigen::MatrixXd N_pre = Eigen::MatrixXd::Identity(num_qdot_, num_qdot_);
  Eigen::MatrixXd N_pre_dyn = Ni_dyn_;
  Eigen::MatrixXd Jc_bar;
  if (b_contact_) {
    int total_contact_dim = 0;
    for (const auto* c : contact_vector) {
      total_contact_dim += c->Dim();
    }

    Jc_cfg_.resize(total_contact_dim, num_qdot_);
    JcDotQdot_cfg_.resize(total_contact_dim);
    xc_ddot_des_cfg_.resize(total_contact_dim);

    int row_offset = 0;
    for (const auto* c : contact_vector) {
      const int dim = c->Dim();
      Jc_cfg_.block(row_offset, 0, dim, num_qdot_) = c->Jacobian();
      JcDotQdot_cfg_.segment(row_offset, dim) = c->JacobianDotQdot();
      xc_ddot_des_cfg_.segment(row_offset, dim) = c->OpCommand();
      row_offset += dim;
    }

    BuildProjectionMatrix(Jc_cfg_, N_pre);
    BuildProjectionMatrix(Jc_cfg_, N_pre_dyn, &Minv_);
    Ni_Nci_dyn_ = N_pre_dyn;
    WeightedPseudoInverse(Jc_cfg_, Minv_, Jc_bar);
  }

  Eigen::VectorXd delta_q_cmd, qdot_cmd, qddot_cmd, JtDotQdot, prev_delta_q_cmd,
      prev_qdot_cmd, prev_qddot_cmd;
  Eigen::MatrixXd Jt, JtPre, JtPre_dyn, JtPre_pinv, JtPre_bar, N_nx, N_nx_dyn;

  if (b_contact_) {
    qddot_cmd = Jc_bar * (xc_ddot_des_cfg_ - JcDotQdot_cfg_);
  } else {
    qddot_cmd = Eigen::VectorXd::Zero(num_qdot_);
  }

  for (auto it = task_vector.begin(); it != task_vector.end(); ++it) {
    Task* task = *it;
    if (it == task_vector.begin()) {
      Jt = task->Jacobian();
      JtDotQdot = task->JacobianDotQdot();
      JtPre = Jt * N_pre;
      PseudoInverse(JtPre, JtPre_pinv);
      JtPre_dyn = Jt * N_pre_dyn;
      WeightedPseudoInverse(JtPre_dyn, Minv_, JtPre_bar);

      delta_q_cmd = JtPre_pinv * task->KpIK().cwiseProduct(task->LocalPosError());
      qdot_cmd = JtPre_pinv * task->DesiredVel();
      qddot_cmd = qddot_cmd + JtPre_bar * (task->OpCommand() - JtDotQdot - Jt * qddot_cmd);
    } else {
      Jt = task->Jacobian();
      JtDotQdot = task->JacobianDotQdot();
      JtPre = Jt * N_pre;
      PseudoInverse(JtPre, JtPre_pinv);
      JtPre_dyn = Jt * N_pre_dyn;
      WeightedPseudoInverse(JtPre_dyn, Minv_, JtPre_bar);

      delta_q_cmd = prev_delta_q_cmd +
                    JtPre_pinv * (task->KpIK().cwiseProduct(task->PosError()) -
                                  Jt * prev_delta_q_cmd);
      qdot_cmd = prev_qdot_cmd +
                 JtPre_pinv * (task->DesiredVel() - Jt * prev_qdot_cmd);
      qddot_cmd = prev_qddot_cmd +
                  JtPre_bar *
                      (task->OpCommand() - JtDotQdot - Jt * prev_qddot_cmd);
    }

    if (std::next(it) != task_vector.end()) {
      prev_delta_q_cmd = delta_q_cmd;
      prev_qdot_cmd = qdot_cmd;
      prev_qddot_cmd = qddot_cmd;
      BuildProjectionMatrix(JtPre, N_nx);
      N_pre *= N_nx;
      BuildProjectionMatrix(JtPre_dyn, N_nx_dyn, &Minv_);
      N_pre_dyn *= N_nx_dyn;
    } else {
      jpos_cmd = curr_jpos + delta_q_cmd.tail(num_qdot_ - num_floating_);
      jvel_cmd = qdot_cmd.tail(num_qdot_ - num_floating_);
      wbc_qddot_cmd = qddot_cmd;
    }
  }

  return true;
}

bool WBIC::MakeTorque(const WbcFormulation& formulation,
                      const Eigen::VectorXd& wbc_qddot_cmd,
                      Eigen::VectorXd& jtrq_cmd) {
  if (!b_update_setting_) {
    return false;
  }
  if (!wbc_qddot_cmd.allFinite()) {
    return false;
  }

  const std::vector<ForceTask*>& force_task_vector = formulation.force_tasks;
  const std::vector<Contact*>& contact_constraints = formulation.contact_constraints;
  b_contact_ = !contact_constraints.empty();

  if (b_contact_) {
    BuildContactMtxVect(contact_constraints);
    GetDesiredReactionForce(force_task_vector);
    if (des_rf_.size() != dim_contact_) {
      return false;
    }
    SetQPCost(wbc_qddot_cmd);
    SetQPEqualityConstraint(wbc_qddot_cmd);
    SetQPInEqualityConstraint();
    if (!SolveQP(wbc_qddot_cmd)) {
      return false;
    }
  } else {
    wbic_data_->corrected_wbc_qddot_cmd_ = wbc_qddot_cmd;
    wbic_data_->rf_cmd_ = Eigen::VectorXd::Zero(0);
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

void WBIC::SetQPCost(const Eigen::VectorXd& wbc_qddot_cmd) {
  H_ = Eigen::MatrixXd::Zero(num_floating_ + dim_contact_,
                             num_floating_ + dim_contact_);
  const Eigen::MatrixXd delta_qddot_cost =
      DiagOrIdentity(wbic_data_->qp_params_->W_delta_qddot_, num_floating_);
  const Eigen::MatrixXd xc_w =
      DiagOrIdentity(wbic_data_->qp_params_->W_xc_ddot_, Jc_.rows());
  const Eigen::MatrixXd xc_ddot_cost =
      (Jc_.transpose() * xc_w * Jc_)
          .topLeftCorner(num_floating_, num_floating_);
  H_.topLeftCorner(num_floating_, num_floating_) = delta_qddot_cost + xc_ddot_cost;
  H_.bottomRightCorner(dim_contact_, dim_contact_) =
      (DiagOrIdentity(wbic_data_->qp_params_->W_delta_rf_, dim_contact_).diagonal() +
       DiagOrIdentity(wbic_data_->qp_params_->W_force_rate_of_change_, dim_contact_).diagonal())
          .asDiagonal();

  g_ = Eigen::VectorXd::Zero(num_floating_ + dim_contact_);
  g_.head(num_floating_) =
      (wbc_qddot_cmd.transpose() * Jc_.transpose() * xc_w * Jc_ +
       JcDotQdot_.transpose() * xc_w * Jc_)
          .head(num_floating_);
  g_.tail(dim_contact_) =
      wbic_data_->rf_prev_cmd_.transpose() *
      DiagOrIdentity(wbic_data_->qp_params_->W_force_rate_of_change_,
                     dim_contact_);
}

void WBIC::SetQPEqualityConstraint(const Eigen::VectorXd& wbc_qddot_cmd) {
  A_ = Eigen::MatrixXd::Zero(num_floating_, num_floating_ + dim_contact_);
  A_.leftCols(num_floating_) = sf_ * M_.leftCols(num_floating_);
  A_.rightCols(dim_contact_) = -sf_ * Jc_.transpose();
  b_ = sf_ * (Jc_.transpose() * des_rf_ - M_ * wbc_qddot_cmd - cori_ - grav_);
}

void WBIC::SetQPInEqualityConstraint() {
  C_ = Eigen::MatrixXd::Zero(Uf_mat_.rows(), num_floating_ + dim_contact_);
  C_.rightCols(dim_contact_) = Uf_mat_;
  l_ = Uf_vec_ - Uf_mat_ * des_rf_;
}

bool WBIC::SolveQP(const Eigen::VectorXd& wbc_qddot_cmd) {
  const int dim = num_floating_ + dim_contact_;
  const int n_eq = num_floating_;
  const int n_ineq = C_.rows();

  if (dim != qp_dim_cache_ || n_eq != qp_eq_cache_ || n_ineq != qp_ineq_cache_) {
    x_.resize(dim);
    G_.resize(dim, dim);
    g0_.resize(dim);
    CE_.resize(dim, n_eq);
    ce0_.resize(n_eq);
    CI_.resize(dim, n_ineq);
    ci0_.resize(n_ineq);
    qp_dim_cache_ = dim;
    qp_eq_cache_ = n_eq;
    qp_ineq_cache_ = n_ineq;
  }

  for (int i = 0; i < dim; ++i) {
    for (int j = 0; j < dim; ++j) {
      G_[j][i] = H_(i, j);
    }
    g0_[i] = g_[i];
  }
  for (int i = 0; i < n_eq; ++i) {
    for (int j = 0; j < dim; ++j) {
      CE_[j][i] = A_(i, j);
    }
    ce0_[i] = -b_[i];
  }
  for (int i = 0; i < n_ineq; ++i) {
    for (int j = 0; j < dim; ++j) {
      CI_[j][i] = C_(i, j);
    }
    ci0_[i] = -l_[i];
  }

  double objective = std::numeric_limits<double>::infinity();
  try {
    objective = solve_quadprog(G_, g0_, CE_, ce0_, CI_, ci0_, x_);
  } catch (const std::exception&) {
    return false;
  }
  if (!std::isfinite(objective)) {
    return false;
  }

  Eigen::VectorXd qp_sol = Eigen::VectorXd::Zero(dim);
  for (int i = 0; i < dim; ++i) {
    qp_sol[i] = x_[i];
    if (!std::isfinite(qp_sol[i])) {
      return false;
    }
  }

  wbic_data_->delta_qddot_ = qp_sol.head(num_floating_);
  wbic_data_->delta_rf_ = qp_sol.tail(dim_contact_);
  wbic_data_->corrected_wbc_qddot_cmd_ = wbc_qddot_cmd;
  wbic_data_->corrected_wbc_qddot_cmd_.head(num_floating_) +=
      wbic_data_->delta_qddot_;
  wbic_data_->rf_cmd_ = des_rf_ + wbic_data_->delta_rf_;
  wbic_data_->rf_prev_cmd_ = wbic_data_->rf_cmd_;
  wbic_data_->Xc_ddot_ =
      Jc_ * wbic_data_->corrected_wbc_qddot_cmd_ + JcDotQdot_;

  const Eigen::MatrixXd w_qddot =
      DiagOrIdentity(wbic_data_->qp_params_->W_delta_qddot_, num_floating_);
  const Eigen::MatrixXd w_rf =
      DiagOrIdentity(wbic_data_->qp_params_->W_delta_rf_, dim_contact_);
  const Eigen::MatrixXd w_xc =
      DiagOrIdentity(wbic_data_->qp_params_->W_xc_ddot_, Jc_.rows());
  wbic_data_->delta_qddot_cost_ =
      wbic_data_->delta_qddot_.transpose() * w_qddot * wbic_data_->delta_qddot_;
  wbic_data_->delta_rf_cost_ =
      wbic_data_->delta_rf_.transpose() * w_rf * wbic_data_->delta_rf_;
  wbic_data_->Xc_ddot_cost_ =
      wbic_data_->Xc_ddot_.transpose() * w_xc * wbic_data_->Xc_ddot_;

  return true;
}

void WBIC::GetSolution(const Eigen::VectorXd& wbc_qddot_cmd,
                       Eigen::VectorXd& jtrq_cmd) {
  Eigen::VectorXd trq;
  if (b_contact_ && wbic_data_->rf_cmd_.size() > 0) {
    trq = M_ * wbic_data_->corrected_wbc_qddot_cmd_ +
          Ni_dyn_.transpose() * (cori_ + grav_) -
          (Jc_ * Ni_dyn_).transpose() * wbic_data_->rf_cmd_;
  } else {
    trq = M_ * wbc_qddot_cmd + Ni_dyn_.transpose() * (cori_ + grav_);
  }
  const Eigen::MatrixXd UNi = sa_ * Ni_dyn_;
  Eigen::MatrixXd UNi_bar;
  WeightedPseudoInverse(UNi, Minv_, UNi_bar);
  jtrq_cmd = UNi_bar.transpose() * trq;
  jtrq_cmd = snf_ * sa_.transpose() * jtrq_cmd;
}

} // namespace wbc
