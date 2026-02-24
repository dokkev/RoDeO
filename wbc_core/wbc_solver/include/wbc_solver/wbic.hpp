#pragma once

#include <Eigen/Dense>
#include <memory>
#include <vector>

#include "wbc_solver/quadprog/QuadProg++.hh"
#include "wbc_solver/interface/wbc.hpp"
#include "wbc_formulation/wbc_formulation.hpp"

namespace wbc {
struct QPParams {
  QPParams(int num_float, int dim_contact)
      : W_delta_qddot_(Eigen::VectorXd::Zero(num_float)),
        W_delta_rf_(Eigen::VectorXd::Zero(dim_contact)),
        W_xc_ddot_(Eigen::VectorXd::Zero(dim_contact)),
        W_force_rate_of_change_(Eigen::VectorXd::Zero(dim_contact)) {}

  Eigen::VectorXd W_delta_qddot_;
  Eigen::VectorXd W_delta_rf_;
  Eigen::VectorXd W_xc_ddot_;
  Eigen::VectorXd W_force_rate_of_change_;
};

struct WBICData {
  WBICData(int num_float, int num_qdot, QPParams* qp_params)
      : qp_params_(qp_params),
        delta_qddot_(Eigen::VectorXd::Zero(num_float)),
        delta_rf_(Eigen::VectorXd::Zero(qp_params->W_delta_rf_.size())),
        delta_qddot_cost_(0.0),
        delta_rf_cost_(0.0),
        Xc_ddot_cost_(0.0),
        corrected_wbc_qddot_cmd_(Eigen::VectorXd::Zero(num_qdot)),
        rf_cmd_(Eigen::VectorXd::Zero(qp_params->W_delta_rf_.size())),
        rf_prev_cmd_(Eigen::VectorXd::Zero(qp_params->W_delta_rf_.size())),
        Xc_ddot_(Eigen::VectorXd::Zero(qp_params->W_delta_rf_.size())) {}

  QPParams* qp_params_;
  Eigen::VectorXd delta_qddot_;
  Eigen::VectorXd delta_rf_;
  double delta_qddot_cost_;
  double delta_rf_cost_;
  double Xc_ddot_cost_;
  Eigen::VectorXd corrected_wbc_qddot_cmd_;
  Eigen::VectorXd rf_cmd_;
  Eigen::VectorXd rf_prev_cmd_;
  Eigen::VectorXd Xc_ddot_;
};

class WBIC : public WBC {
public:
  WBIC(const std::vector<bool>& act_qdot_list, QPParams* qp_params);
  ~WBIC() override = default;

  bool FindConfiguration(const WbcFormulation& formulation,
                         const Eigen::VectorXd& curr_jpos,
                         Eigen::VectorXd& jpos_cmd, Eigen::VectorXd& jvel_cmd,
                         Eigen::VectorXd& wbc_qddot_cmd);

  bool MakeTorque(const WbcFormulation& formulation,
                  const Eigen::VectorXd& wbc_qddot_cmd,
                  Eigen::VectorXd& jtrq_cmd) override;

  void SetParameters() override {}
  WBICData* GetWBICData() { return wbic_data_.get(); }

private:
  void PseudoInverse(const Eigen::MatrixXd& jac, Eigen::MatrixXd& jac_inv);
  void WeightedPseudoInverse(const Eigen::MatrixXd& jac, const Eigen::MatrixXd& W,
                             Eigen::MatrixXd& jac_bar);
  void BuildProjectionMatrix(const Eigen::MatrixXd& jac, Eigen::MatrixXd& N,
                             const Eigen::MatrixXd* W = nullptr);
  void BuildContactMtxVect(const std::vector<Contact*>& contacts);
  void GetDesiredReactionForce(const std::vector<ForceTask*>& force_task_vector);
  void SetQPCost(const Eigen::VectorXd& wbc_qddot_cmd);
  void SetQPEqualityConstraint(const Eigen::VectorXd& wbc_qddot_cmd);
  void SetQPInEqualityConstraint();
  bool SolveQP(const Eigen::VectorXd& wbc_qddot_cmd);
  void GetSolution(const Eigen::VectorXd& wbc_qddot_cmd,
                   Eigen::VectorXd& jtrq_cmd);

  double threshold_;
  Eigen::MatrixXd Ni_dyn_;
  Eigen::MatrixXd Ni_Nci_dyn_;

  // Reused buffers for contact/task stacking in control loop.
  Eigen::MatrixXd Jc_cfg_;
  Eigen::VectorXd JcDotQdot_cfg_;
  Eigen::VectorXd xc_ddot_des_cfg_;

  Eigen::MatrixXd Jc_;
  Eigen::VectorXd JcDotQdot_;
  Eigen::MatrixXd Uf_mat_;
  Eigen::VectorXd Uf_vec_;
  Eigen::VectorXd des_rf_;
  std::unique_ptr<WBICData> wbic_data_;

  Eigen::MatrixXd H_;
  Eigen::VectorXd g_;
  Eigen::MatrixXd A_;
  Eigen::VectorXd b_;
  Eigen::MatrixXd C_;
  Eigen::VectorXd l_;

  GolDIdnani::GVect<double> x_;
  GolDIdnani::GMatr<double> G_;
  GolDIdnani::GVect<double> g0_;
  GolDIdnani::GMatr<double> CE_;
  GolDIdnani::GVect<double> ce0_;
  GolDIdnani::GMatr<double> CI_;
  GolDIdnani::GVect<double> ci0_;
  int qp_dim_cache_;
  int qp_eq_cache_;
  int qp_ineq_cache_;
};

} // namespace wbc
