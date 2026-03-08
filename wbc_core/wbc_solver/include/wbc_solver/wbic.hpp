/**
 * @file wbc_core/wbc_solver/include/wbc_solver/wbic.hpp
 * @brief Doxygen documentation for wbic module.
 */
#pragma once

#include <Eigen/Cholesky>
#include <Eigen/Dense>
#include <chrono>
#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

#include <proxsuite/proxqp/dense/dense.hpp>

#include "wbc_solver/interface/wbc.hpp"
#include "wbc_formulation/wbc_formulation.hpp"

namespace wbc {

/**
 * @brief Hard torque-limit handling mode in WBIC correction QP.
 */
enum class HardTorqueLimitMode {
  DIAGONAL_M_BOX,  ///< Fast approximate box using only diag(M); not an exact hard torque limit
  EXACT_DENSE,     ///< Exact coupled hard torque limits in dense inequality rows
};

/**
 * @brief Contact handling mode for ID-QP motion consistency terms.
 */
enum class ContactMode {
  kNone,           ///< no contact acceleration term
  kSoftTracking,   ///< soft contact-acceleration tracking in the QP cost
  kRigidEquality,  ///< reserved for future hard contact-equality mode
};

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
        qddot_sol_(Eigen::VectorXd::Zero(num_qdot)),
        rf_sol_(Eigen::VectorXd::Zero(qp_params->W_delta_rf_.size())),
        rf_prev_sol_(Eigen::VectorXd::Zero(qp_params->W_delta_rf_.size())),
        Xc_ddot_(Eigen::VectorXd::Zero(qp_params->W_delta_rf_.size())),
        tau_prev_(Eigen::VectorXd::Zero(num_qdot)),
        qp_solved_(false),
        qp_status_(-1),
        qp_iter_(0),
        qp_pri_res_(0.0),
        qp_dua_res_(0.0),
        qp_obj_(0.0),
        qp_setup_time_us_(0.0),
        qp_solve_time_us_(0.0),
        box_conflict_count_(0),
        box_last_conflict_index_(-1),
        box_conflict_active_(false) {}

  QPParams* qp_params_;
  Eigen::VectorXd delta_qddot_;
  Eigen::VectorXd delta_rf_;
  double delta_qddot_cost_;
  double delta_rf_cost_;
  double Xc_ddot_cost_;
  Eigen::VectorXd qddot_sol_;
  Eigen::VectorXd rf_sol_;
  Eigen::VectorXd rf_prev_sol_;
  Eigen::VectorXd Xc_ddot_;
  Eigen::VectorXd tau_prev_;

  // QP diagnostics from the latest SolveQP call.
  bool qp_solved_;
  int qp_status_;
  int qp_iter_;
  double qp_pri_res_;
  double qp_dua_res_;
  double qp_obj_;
  double qp_setup_time_us_;
  double qp_solve_time_us_;
  int box_conflict_count_;
  int box_last_conflict_index_;
  bool box_conflict_active_;
};

/**
 * @brief Kinematic-stage reference bundle.
 *
 * This is the structured output of IK/reference generation. It intentionally
 * carries references only (not feasible dynamics solutions).
 */
struct KinematicReference {
  Eigen::VectorXd jpos_ref;
  Eigen::VectorXd jvel_ref;
};

/**
 * @brief Inverse-dynamics QP solution bundle.
 *
 * This groups the feasible solution solved by ID-QP.
 */
struct InverseDynamicsSolution {
  Eigen::VectorXd qddot_sol;
  Eigen::VectorXd rf_sol;
  Eigen::VectorXd tau_gen_sol;
  Eigen::VectorXd tau_cmd;
};

/**
 * @brief Optional final command bundle for downstream command builders.
 *
 * This is introduced for interface clarity and is not yet the primary command
 * path in ControlArchitecture.
 */
struct WbcCommand {
  std::optional<Eigen::VectorXd> q_cmd;
  std::optional<Eigen::VectorXd> qdot_cmd;
  std::optional<Eigen::VectorXd> tau_cmd;
};

/**
 * @brief Whole-Body Impulse Control (WBIC) solver implementation.
 *
 * @note Joint position/velocity limit constraints are interpreted in
 * actuated-joint space only (`num_active_` columns, `EffectiveLimits` sized
 * to actuated joints).
 */
class WBIC : public WBC {
public:
  WBIC(const std::vector<bool>& act_qdot_list, QPParams* qp_params);
  ~WBIC() override = default;

  /**
   * @brief IK/reference stage.
   */
  bool ComputeKinematicReference(const WbcFormulation& formulation,
                                 const Eigen::VectorXd& curr_jpos,
                                 Eigen::VectorXd& jpos_ref,
                                 Eigen::VectorXd& jvel_ref);

  /**
   * @brief Structured-output overload of ComputeKinematicReference.
   */
  bool ComputeKinematicReference(const WbcFormulation& formulation,
                                 const Eigen::VectorXd& curr_jpos,
                                 KinematicReference& out_ref);

  /**
   * @brief Build joint acceleration reference from kinematic references and measurements.
   *
   * @details
   * This stage is the owner of `qddot_posture_ref` authority. IK computes only
   * `jpos_ref/jvel_ref`; posture acceleration reference is built here using
   * joint-space feedback on measured state.
   */
  bool BuildPostureAccelReference(const KinematicReference& kin_ref,
                                  const Eigen::VectorXd& jpos_meas,
                                  const Eigen::VectorXd& jvel_meas,
                                  Eigen::VectorXd& qddot_posture_ref) const;

  /**
   * @brief Set joint-space acceleration-reference gains.
   *
   * @param kp_acc Position error gain (must be finite and non-negative).
   * @param kd_acc Velocity error gain (must be finite and non-negative).
   * @return true on successful update.
   */
  bool SetJointAccelReferenceGains(double kp_acc, double kd_acc);
  bool SetJointAccelReferenceGains(const Eigen::VectorXd& kp_acc,
                                   const Eigen::VectorXd& kd_acc);
  void SetIndependentVelocityReference(bool enabled) {
    independent_velocity_ref_ = enabled;
  }
  bool GetIndependentVelocityReference() const {
    return independent_velocity_ref_;
  }

  // WBC base-interface entry point. For direct WBIC usage, prefer
  // SolveInverseDynamics for clearer stage semantics.
  bool MakeTorque(const WbcFormulation& formulation,
                  const Eigen::VectorXd& qddot_ref,
                  Eigen::VectorXd& jtrq_cmd) override;

  /**
   * @brief Inverse-dynamics stage.
   */
  bool SolveInverseDynamics(const WbcFormulation& formulation,
                            const Eigen::VectorXd& qddot_ref,
                            Eigen::VectorXd& jtrq_cmd);

  /**
   * @brief Structured-output overload of SolveInverseDynamics.
   */
  bool SolveInverseDynamics(const WbcFormulation& formulation,
                            const Eigen::VectorXd& qddot_ref,
                            InverseDynamicsSolution& out_sol);

  void SetParameters() override {}
  WBICData* GetWbicData() { return wbic_data_.get(); }

  void SetHardTorqueLimitMode(HardTorqueLimitMode m) {
    hard_torque_limit_mode_ = m;
  }
  HardTorqueLimitMode GetHardTorqueLimitMode() const {
    return hard_torque_limit_mode_;
  }

  void SetContactMode(ContactMode mode) { contact_mode_ = mode; }
  ContactMode GetContactMode() const { return contact_mode_; }

  /**
   * @brief Pre-allocate solver buffers for the worst-case dimensions.
   *
   * Call once after construction (before the first SolveInverseDynamics) to avoid
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
  /// box constraint (torque hard-limit handling is selected by
  /// `hard_torque_limit_mode_`).
  struct SoftLimitParams {
    bool pos{false};   ///< Joint position limit: soft (slack) or hard (box)
    bool vel{false};   ///< Joint velocity limit: soft (slack) or hard (box)
    bool trq{false};   ///< Joint torque limit: soft (slack+full dynamics) or hard (box+diag M)
    double w_pos{1e5}; ///< Position slack penalty weight
    double w_vel{1e5}; ///< Velocity slack penalty weight
    double w_trq{1e5}; ///< Torque slack penalty weight
  } soft_params_;

private:
  bool WeightedPseudoInverse(const Eigen::MatrixXd& jac, const Eigen::MatrixXd& W,
                             Eigen::MatrixXd& jac_bar);
  bool SolveKinematicReferenceQP(const WbcFormulation& formulation,
                                 const Eigen::VectorXd& curr_jpos,
                                 Eigen::VectorXd& jpos_ref,
                                 Eigen::VectorXd& jvel_ref);
  bool PrepareContactContext(const WbcFormulation& formulation);
  bool BuildContactMtxVect(const std::vector<Contact*>& contacts);
  bool GetDesiredReactionForce(
      const std::vector<ForceTask*>& force_task_vector);
  void ClearContactContext();
  bool CacheConstraintPointers(const std::vector<Constraint*>& constraints);
  // Joint pos/vel limit contract in WBIC:
  // - ConstraintMatrix columns must be actuated-joint space (num_active_)
  // - EffectiveLimits must be [num_active_ x >=2] and finite
  bool HasSupportedJointLimitShape(const JointPosLimitConstraint* c) const;
  bool HasSupportedJointLimitShape(const JointVelLimitConstraint* c) const;
  void ComputeSlackDimensions(const Eigen::VectorXd& qddot_ref);
  bool ValidateOperationalTasks(
      const std::vector<Task*>& operational_tasks) const;
  bool ValidatePostureTasks(
      const std::vector<Task*>& posture_tasks) const;
  bool ValidateIKContacts(
      const std::vector<Contact*>& contact_constraints) const;
  bool ValidateIKRuntimeConfig() const;
  bool ValidateIDRuntimeConfig() const;
  void ProjectKinematicReferenceToJointBounds(const Eigen::VectorXd& curr_jpos,
                                              Eigen::VectorXd& jpos_ref,
                                              Eigen::VectorXd& jvel_ref) const;
  struct InequalityMode {
    bool use_box_solver{false};
    bool force_dense{false};
    bool pos_axis_aligned{false};
    bool vel_axis_aligned{false};
    int n_friction{0};
    int n_ineq{0};
  };

  // QP cost assembly
  void SetQPCost(const WbcFormulation& formulation,
                 const Eigen::VectorXd& qddot_posture_ref);
  void AddOperationalTaskCosts(const std::vector<Task*>& operational_tasks,
                               const Eigen::VectorXd& qddot_nominal);
  void AddNominalAccelTrackingCost(bool force_contact_mode);
  void AddExactTorqueRegularization(const Eigen::VectorXd& qddot_nominal);
  void AddContactAccelerationCost(const Eigen::VectorXd& qddot_nominal);
  void AddReactionForceCost();
  void AddSlackVariablePenalties();

  // QP equality/inequality assembly
  void SetQPEqualityConstraint(const Eigen::VectorXd& qddot_ref);
  void SetQPInEqualityConstraint(const Eigen::VectorXd& qddot_ref);
  InequalityMode BuildInequalityMode() const;
  void ResizeInequalityStorage(const InequalityMode& mode, int qp_dim);
  void AssembleInequalityRows(const InequalityMode& mode,
                              const Eigen::VectorXd& qddot_ref);
  bool ExtractAxisAlignedBoxBounds(const Constraint* c,
                                   const Eigen::VectorXd& qddot_ref);
  bool IsAxisAlignedConstraint(const Constraint* c) const;
  int BuildFrictionConeConstraint(int row);
  int BuildKinematicLimitConstraint(const Constraint* c, bool is_soft,
                                    bool use_box_solver,
                                    const Eigen::VectorXd& qddot_ref,
                                    int row, int& slack_col);
  int BuildAxisAlignedKinematicLimitConstraint(const Constraint* c, bool is_soft,
                                               bool use_box_solver, int row,
                                               int& slack_col);
  int BuildDenseKinematicLimitConstraint(const Constraint* c, bool is_soft,
                                         const Eigen::VectorXd& qddot_ref,
                                         int row, int& slack_col);
  int BuildTorqueLimitConstraint(const JointTrqLimitConstraint* c, bool is_soft,
                                  bool use_box_solver,
                                  int row, int& slack_col);
  void EnforceBoxFeasibilityGuard(int qp_dim);

  bool SolveQP(const Eigen::VectorXd& qddot_ref);
  bool GetSolution(Eigen::VectorXd& jtrq_cmd);


  // NOTE: Ni_dyn_ (internal/passive-joint nullspace) is not implemented.
  // All supported robots are fully actuated with no passive joints.
  // When passive joint support is added, restore Ni_dyn_ and multiply through
  // in tau_0_ and GetSolution. Until then, all Ni_dyn_ terms are identity.

  Eigen::VectorXd delta_q_ref_;
  Eigen::VectorXd qdot_ref_;
  Eigen::VectorXd tau_gen_sol_;
  Eigen::VectorXd tau_cost_scratch_;
  // Joint-space posture-acceleration gains (actuated-joint sized).
  Eigen::VectorXd kp_acc_;
  Eigen::VectorXd kd_acc_;
  Eigen::MatrixXd sa_pinv_scratch_;

  Eigen::MatrixXd Jc_;
  Eigen::VectorXd JcDotQdot_;
  Eigen::MatrixXd Uf_mat_;
  Eigen::VectorXd Uf_vec_;
  Eigen::VectorXd des_rf_;
  struct ContactStackBlock {
    int rf_offset{0};
    int dim{0};
  };
  std::unordered_map<const Contact*, ContactStackBlock> contact_rf_blocks_;
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
  // Actuated-joint indices that have at least one finite axis-aligned bound.
  std::vector<int> pos_bounded_active_indices_;
  std::vector<int> vel_bounded_active_indices_;

  // Scratch buffers for LLT-based damped pseudo-inverse.
  // MaxRows/MaxCols = 36 keeps common paths inline. Larger rows fall back to
  // dynamic temporaries to preserve correctness when systems scale up.
  static constexpr int kMaxPInvDim = 36;
  using PInvSquare = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                                   Eigen::ColMajor, kMaxPInvDim, kMaxPInvDim>;
  Eigen::LLT<PInvSquare> llt_scratch_;
  PInvSquare JWJt_scratch_;
  PInvSquare JWJt_pinv_scratch_;

  // Cached kinematic constraint pointers (set once per SolveInverseDynamics call,
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

  // Weighted-QP configuration
  HardTorqueLimitMode hard_torque_limit_mode_{
      HardTorqueLimitMode::EXACT_DENSE};

  // Weighted QP IK buffers (pre-allocated in constructor, reused per tick)
  Eigen::MatrixXd H_ik_;
  Eigen::VectorXd g_ik_pos_;
  Eigen::VectorXd g_ik_vel_;

  // Contact handling mode for motion-consistency terms.
  ContactMode contact_mode_{ContactMode::kSoftTracking};
  // IK-stage contact penalty (soft near-rigid shaping, not a hard equality).
  // This only prevents posture IK from becoming contact-blind.
  // Final contact consistency/feasibility is solved by ID-QP.
  double ik_contact_penalty_weight_{1e6};
  // One-step horizon for position-aware velocity-reference clamping.
  double ik_velocity_clamp_dt_{1e-3};
  // Global safety cap on IK velocity reference (rad/s).
  double ik_velocity_ref_abs_max_{5.0};
  // true: solve qdot_ref from IK LS; false: derive qdot_ref from delta_q_ref/dt.
  bool independent_velocity_ref_{true};
  double posture_bias_contact_scale_{0.2};

};

} // namespace wbc
