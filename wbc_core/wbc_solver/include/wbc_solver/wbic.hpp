/**
 * @file wbc_core/wbc_solver/include/wbc_solver/wbic.hpp
 * @brief Doxygen documentation for wbic module.
 */
#pragma once

#include <Eigen/Cholesky>
#include <Eigen/Dense>
#include <chrono>
#include <memory>
#include <vector>

#include <proxsuite/proxqp/dense/dense.hpp>

#include "wbc_solver/interface/wbc.hpp"
#include "wbc_formulation/wbc_formulation.hpp"

namespace wbc {
/**
 * @brief Runtime QP weight set for WBIC correction stage.
 */
struct QPParams {
  QPParams(int num_qdot, int dim_contact)
      : W_delta_qddot_(Eigen::VectorXd::Zero(num_qdot)),
        W_delta_rf_(Eigen::VectorXd::Zero(dim_contact)),
        W_xc_ddot_(Eigen::VectorXd::Zero(dim_contact)),
        W_f_dot_(Eigen::VectorXd::Zero(dim_contact)),
        W_tau_(Eigen::VectorXd::Zero(num_qdot)),
        W_tau_dot_(Eigen::VectorXd::Zero(num_qdot)) {}

  Eigen::VectorXd W_delta_qddot_;
  Eigen::VectorXd W_delta_rf_;
  Eigen::VectorXd W_xc_ddot_;
  Eigen::VectorXd W_f_dot_;
  Eigen::VectorXd W_tau_;
  Eigen::VectorXd W_tau_dot_;
};

/**
 * @brief WBIC internal buffers and solve results.
 */
struct WBICData {
  WBICData(int num_qdot, QPParams* qp_params)
      : qp_params_(qp_params),
        delta_qddot_(Eigen::VectorXd::Zero(num_qdot)),
        delta_rf_(Eigen::VectorXd::Zero(qp_params->W_delta_rf_.size())),
        delta_qddot_cost_(0.0),
        delta_rf_cost_(0.0),
        Xc_ddot_cost_(0.0),
        tau_cost_(0.0),
        tau_dot_cost_(0.0),
        corrected_wbc_qddot_cmd_(Eigen::VectorXd::Zero(num_qdot)),
        rf_cmd_(Eigen::VectorXd::Zero(qp_params->W_delta_rf_.size())),
        rf_prev_cmd_(Eigen::VectorXd::Zero(qp_params->W_delta_rf_.size())),
        Xc_ddot_(Eigen::VectorXd::Zero(qp_params->W_delta_rf_.size())),
        tau_prev_(Eigen::VectorXd::Zero(num_qdot)) {}

  QPParams* qp_params_;
  Eigen::VectorXd delta_qddot_;
  Eigen::VectorXd delta_rf_;
  double delta_qddot_cost_;
  double delta_rf_cost_;
  double Xc_ddot_cost_;
  double tau_cost_;
  double tau_dot_cost_;
  Eigen::VectorXd corrected_wbc_qddot_cmd_;
  Eigen::VectorXd rf_cmd_;
  Eigen::VectorXd rf_prev_cmd_;
  Eigen::VectorXd Xc_ddot_;
  Eigen::VectorXd tau_prev_;
};

/**
 * @brief Whole-Body Impulse Control (WBIC) solver implementation.
 */
class WBIC : public WBC {
public:
  WBIC(const std::vector<bool>& act_qdot_list, QPParams* qp_params);
  ~WBIC() override = default;

  /**
   * @brief Compute joint-space configuration command (q, qdot, qddot).
   */
  bool FindConfiguration(const WbcFormulation& formulation,
                         const Eigen::VectorXd& curr_jpos,
                         Eigen::VectorXd& jpos_cmd, Eigen::VectorXd& jvel_cmd,
                         Eigen::VectorXd& wbc_qddot_cmd);

  bool MakeTorque(const WbcFormulation& formulation,
                  const Eigen::VectorXd& wbc_qddot_cmd,
                  Eigen::VectorXd& jtrq_cmd) override;

  void SetParameters() override {}
  WBICData* GetWbicData() { return wbic_data_.get(); }

  /**
   * @brief Pre-allocate solver buffers for the worst-case dimensions.
   *
   * Call once after construction (before the first MakeTorque) to avoid
   * first-tick heap allocations in the RT loop.
   */
  void ReserveCapacity(int max_contact_dim, int max_uf_rows);

  /// Per-phase timing stats (populated only when enable_timing_ is true).
  struct WbicTimingStats {
    double find_config_us{0};
    double qp_setup_us{0};
    double qp_solve_us{0};
    double torque_recovery_us{0};
  };
  bool enable_timing_{false};
  WbicTimingStats timing_stats_;

  /// Per-constraint soft/hard toggle and penalty weights.
  /// When soft, the constraint becomes a slack variable with quadratic penalty
  /// in the QP cost, allowing controlled violation. When hard, it becomes a
  /// box constraint (or diagonal-M box approximation for torque).
  struct SoftLimitParams {
    bool pos{false};   ///< Joint position limit: soft (slack) or hard (box)
    bool vel{false};   ///< Joint velocity limit: soft (slack) or hard (box)
    bool trq{false};   ///< Joint torque limit: soft (slack+full dynamics) or hard (box+diag M)
    double w_pos{1e5}; ///< Position slack penalty weight
    double w_vel{1e5}; ///< Velocity slack penalty weight
    double w_trq{1e5}; ///< Torque slack penalty weight
  } soft_params_;

private:
  void PseudoInverse(const Eigen::MatrixXd& jac, Eigen::MatrixXd& jac_inv);
  void WeightedPseudoInverse(const Eigen::MatrixXd& jac, const Eigen::MatrixXd& W,
                             Eigen::MatrixXd& jac_bar);
  void BuildProjectionMatrix(const Eigen::MatrixXd& jac, Eigen::MatrixXd& N,
                             const Eigen::MatrixXd* W = nullptr);
  void InitContactProjection(const std::vector<Contact*>& contacts);
  void BuildContactMtxVect(const std::vector<Contact*>& contacts);
  void GetDesiredReactionForce(const std::vector<ForceTask*>& force_task_vector);
  // QP cost assembly
  void SetQPCost(const Eigen::VectorXd& wbc_qddot_cmd);
  void AddQddotTrackingCost();
  void AddTorqueMinimizationCost(const Eigen::VectorXd& wbc_qddot_cmd);
  void AddContactAccelerationCost(const Eigen::VectorXd& wbc_qddot_cmd);
  void AddReactionForceCost();
  void AddSlackVariablePenalties();

  // QP equality/inequality assembly
  void SetQPEqualityConstraint(const Eigen::VectorXd& wbc_qddot_cmd);
  void SetQPInEqualityConstraint(const WbcFormulation& formulation,
                                 const Eigen::VectorXd& wbc_qddot_cmd);
  void ExtractBoxBounds(const Constraint* c, const Eigen::VectorXd& wbc_qddot_cmd);
  int BuildFrictionConeConstraint(int row);
  int BuildKinematicLimitConstraint(const Constraint* c, bool is_soft,
                                    bool use_box_solver,
                                    const Eigen::VectorXd& wbc_qddot_cmd,
                                    int row, int& slack_col);
  int BuildTorqueLimitConstraint(const JointTrqLimitConstraint* c, bool is_soft,
                                  bool use_box_solver,
                                  int row, int& slack_col);
  void EnforceBoxFeasibilityGuard(int qp_dim);

  bool SolveQP(const Eigen::VectorXd& wbc_qddot_cmd);
  void GetSolution(const Eigen::VectorXd& wbc_qddot_cmd,
                   Eigen::VectorXd& jtrq_cmd);


  Eigen::MatrixXd Ni_dyn_;
  Eigen::MatrixXd N_pre_;
  Eigen::MatrixXd N_pre_dyn_;
  Eigen::MatrixXd N_nx_;
  Eigen::MatrixXd N_nx_dyn_;
  Eigen::MatrixXd Jc_bar_;
  Eigen::VectorXd qddot_cmd_;
  Eigen::VectorXd delta_q_cmd_;
  Eigen::VectorXd qdot_cmd_;
  Eigen::VectorXd prev_qddot_cmd_;
  Eigen::VectorXd prev_delta_q_cmd_;
  Eigen::VectorXd prev_qdot_cmd_;
  Eigen::VectorXd trq_;
  Eigen::MatrixXd UNi_;
  Eigen::MatrixXd UNi_bar_;

  // Reused buffers for contact/task stacking in control loop.
  Eigen::MatrixXd stacked_contact_jacobian_;
  Eigen::VectorXd stacked_contact_jdot_qdot_;
  Eigen::VectorXd stacked_contact_op_cmd_;

  Eigen::MatrixXd Jc_;
  Eigen::VectorXd JcDotQdot_;
  Eigen::MatrixXd Uf_mat_;
  Eigen::VectorXd Uf_vec_;
  Eigen::VectorXd des_rf_;
  std::unique_ptr<WBICData> wbic_data_;

  // QP matrices (Eigen — passed directly to ProxQP)
  Eigen::MatrixXd H_;
  Eigen::VectorXd g_;
  Eigen::MatrixXd A_;
  Eigen::VectorXd b_;
  Eigen::MatrixXd C_;
  Eigen::VectorXd l_;
  Eigen::VectorXd u_;
  Eigen::VectorXd l_box_;
  Eigen::VectorXd u_box_;

  // Torque regularization buffer
  Eigen::VectorXd tau_0_;

  // Slack variable dimensions (0 when corresponding limit is hard or absent)
  int dim_slack_pos_{0};
  int dim_slack_vel_{0};
  int dim_slack_trq_{0};
  int dim_slack_total_{0};

  // Scratch buffers for per-DOF bound extraction
  Eigen::VectorXd l_bnd_;
  Eigen::VectorXd u_bnd_;

  // Reused buffers for per-task nullspace projection in FindConfiguration.
  // Declared as members to avoid per-tick heap allocation inside the task loop.
  Eigen::MatrixXd JtPre_;      // task jacobian projected into null space (dim x N)
  Eigen::MatrixXd JtPre_pinv_; // pseudo-inverse of JtPre_
  Eigen::MatrixXd JtPre_dyn_;  // dynamically-weighted projected jacobian
  Eigen::MatrixXd JtPre_bar_;  // weighted pseudo-inverse of JtPre_dyn_

  // Scratch buffers for LLT-based damped pseudo-inverse.
  // MaxRows/MaxCols = 36 ensures inline storage (no heap alloc) for all
  // practical pseudo-inverse calls: tasks (≤6), contacts (≤24), UNi (≤num_active).
  static constexpr int kMaxPInvDim = 36;
  using PInvSquare = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                                   Eigen::ColMajor, kMaxPInvDim, kMaxPInvDim>;
  Eigen::LLT<PInvSquare> llt_scratch_;
  PInvSquare JWJt_scratch_;
  PInvSquare JWJt_pinv_scratch_;
  Eigen::MatrixXd Jbar_scratch_;

  // Cached kinematic constraint pointers (set once per MakeTorque call,
  // avoids repeated dynamic_cast in SetQPInEqualityConstraint).
  const JointPosLimitConstraint* cached_pos_c_{nullptr};
  const JointVelLimitConstraint* cached_vel_c_{nullptr};
  const JointTrqLimitConstraint* cached_trq_c_{nullptr};

  // Scratch buffer for Sa * tau_0 (avoids per-tick heap allocation in torque limits).
  Eigen::VectorXd sa_tau0_scratch_;

  // Scratch buffers for SetQPCost (avoid per-tick heap allocation from DiagOrIdentity).
  Eigen::MatrixXd wM_scratch_;       // num_qdot × num_qdot: diag(w) * M
  Eigen::MatrixXd wJc_scratch_;      // dim_contact × num_qdot: diag(w_xc) * Jc
  Eigen::VectorXd xc_res_scratch_;   // dim_contact: Jc * qddot + JcDotQdot

  // ProxQP dense solver (lazy-initialized on first SolveQP call)
  std::unique_ptr<proxsuite::proxqp::dense::QP<double>> qp_solver_;
};

} // namespace wbc
