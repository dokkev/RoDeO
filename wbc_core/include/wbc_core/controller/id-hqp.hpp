//
// Copyright (c) 2026
//
/// \file id-hqp.hpp
/// \brief Delta-form inverse-dynamics HQP controller.
//

#ifndef __wbc_controller_id_hqp_hpp__
#define __wbc_controller_id_hqp_hpp__

#include <memory>
#include <vector>

#include <Eigen/Dense>

#include "wbc_core/controller/base/id-base.hpp"
#include "wbc_core/formulations/id-problem.hpp"
#include "wbc_core/formulations/id-solution.hpp"
#include "wbc_core/formulations/hqp/blocks/contact-acceleration-block.hpp"
#include "wbc_core/formulations/hqp/blocks/floating-base-dynamics-block.hpp"
#include "wbc_core/formulations/hqp/blocks/friction-cone-block.hpp"
#include "wbc_core/formulations/hqp/blocks/joint-accel-bias-block.hpp"
#include "wbc_core/formulations/hqp/blocks/joint-torque-limit-block.hpp"
#include "wbc_core/formulations/hqp/blocks/motion-constraint-block.hpp"
#include "wbc_core/formulations/hqp/blocks/qddot-regularization-block.hpp"
#include "wbc_core/formulations/hqp/blocks/rf-regularization-block.hpp"
#include "wbc_core/formulations/hqp/hqp-build-context.hpp"
#include "wbc_core/robots/robot-system.hpp"
#include "wbc_core/solvers/solver-HQP-base.hpp"
#include "wbc_core/solvers/solver-qp-params.hpp"

namespace wbc {

/// \brief Solves a ready inverse-dynamics problem through an HQP cascade.
///
/// `IDHQP` consumes an `IDProblem` assembled by higher-level runtime code and
/// solves for the decision vector `x = [delta_qddot, lambda]`. The final
/// acceleration is `qddot_sol = qddot_ref + delta_qddot_sol`; `tau_sol` is then
/// computed from model dynamics and contact forces.
///
/// This class owns HQP block assembly, cascade execution, and the numerical
/// solver backend. It does not own task/contact objects, state-machine policy,
/// command integration, or final `RobotCommand` construction.
///
/// \note `problem.qddot_ref == nullptr` is treated as a zero reference
///       acceleration. Runtime assembly should normally provide an explicit
///       vector, including the zero vector when no reference is desired.
/// \invariant All per-tick task/contact data passed through `IDProblem` is
///            consumed immediately by `solve()` and is not retained.
class IDHQP : public InverseDynamicsBase {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef math::Vector Vector;
  typedef math::Matrix Matrix;
  typedef pinocchio::Data Data;

  /// \brief Timing data for the most recent solve call.
  struct TimingStats {
    /// HQP block assembly and solver resize time, in microseconds.
    double qp_setup_us{0.0};
    /// Numerical HQP cascade solve time, in microseconds.
    double qp_solve_us{0.0};
  };

  /// \brief Creates an IDHQP controller with the default available backend.
  ///
  /// \param robot Robot model/state wrapper. The object must outlive this
  ///        controller.
  explicit IDHQP(robots::RobotSystem& robot);

  /// \brief Creates an IDHQP controller with an explicit HQP backend.
  ///
  /// \param robot Robot model/state wrapper. The object must outlive this
  ///        controller.
  /// \param solver_type Inner QP backend used by the HQP cascade.
  IDHQP(robots::RobotSystem& robot, solvers::SolverHQP solver_type);

  /// \brief Creates an IDHQP controller with backend-specific QP parameters.
  ///
  /// \param robot Robot model/state wrapper. The object must outlive this
  ///        controller.
  /// \param solver_type Inner QP backend used by the HQP cascade.
  /// \param qp_params Numerical parameters applied to the selected backend.
  IDHQP(robots::RobotSystem& robot, solvers::SolverHQP solver_type,
        const solvers::SolverQPParams& qp_params);

  /// \brief Solves one inverse-dynamics HQP problem.
  ///
  /// \param problem Per-tick solve input. Task/contact vectors and matrices are
  ///        read during this call only.
  /// \param dt Control period in seconds. Must be finite and non-negative.
  /// \return Last solution owned by this controller. The reference remains
  ///         valid until the next `solve()` call or controller destruction.
  /// \pre The associated `RobotSystem` has an accepted current state.
  const IDSolution& solve(const IDProblem& problem, double dt) override;

  /// \brief Reference acceleration used by the most recent solution, size nv().
  const Vector& referenceAcceleration() const { return m_solution->qddot_ref; }

  /// \brief Solved acceleration correction, size nv().
  const Vector& deltaAcceleration() const {
    return m_solution->delta_qddot_sol;
  }

  /// \brief Final solved acceleration, `qddot_ref + delta_qddot_sol`.
  const Vector& solvedAcceleration() const { return m_solution->qddot_sol; }

  /// \brief Returns setup/solve timings for the most recent solve call.
  const TimingStats& timingStats() const { return m_timingStats; }

  /// \brief Returns the selected inner QP backend.
  solvers::SolverHQP solverType() const { return m_solverType; }

  /// \brief Recreates the HQP cascade with a different inner backend.
  ///
  /// Cached solver dimensions are cleared, so the next `solve()` will resize
  /// the backend from assembled `HQPData`.
  void setSolverType(solvers::SolverHQP solver_type);

  /// \brief Returns the current numerical QP parameters.
  const solvers::SolverQPParams& qpParams() const {
    return m_qpParams;
  }

  /// \brief Applies numerical QP parameters to the current backend.
  void setQPParams(const solvers::SolverQPParams& qp_params);

 private:
  /// Cached block for a motion objective from `IDProblem::motion_objectives`.
  struct MotionObjectiveSlot {
    std::unique_ptr<MotionConstraintBlock> block;
    std::shared_ptr<math::ConstraintBase> constraint;
  };

  /// Cached block for a joint-acceleration objective.
  struct JointAccelerationObjectiveSlot {
    std::unique_ptr<JointAccelerationBias> block;
    std::shared_ptr<math::ConstraintBase> constraint;
  };

  /// Scratch data valid only during one `solve()` call.
  struct SolveWorkspace {
    bool hasContactForces{false};
    bool hasContactKinematics{false};
    bool hasFrictionConstraints{false};
    bool hasJointTorqueLimits{false};
    bool regularizeLambda{false};
    StackedContactData contacts;
  };

  /// Resets per-solve scratch state and base solution buffers.
  void beginCycle(const IDProblem& problem);

  /// Returns the current reset/failed solution without changing state.
  const IDSolution& fail() const { return *m_solution; }

  /// Checks cheap base input and HQP level conventions.
  bool validateInput(const IDProblem& problem, double dt) const;

  /// Checks HQP level conventions for active objectives.
  bool validateHierarchy(const IDProblem& problem) const;

  /// Builds contact stacks, active flags, and per-solve solution buffers.
  void prepareSolveWorkspace(const IDProblem& problem);

  /// Populates the HQP block build context from current model/problem data.
  void buildHqpContext(const IDProblem& problem);

  /// Builds hard feasibility constraints for HQP level 0.
  void buildHardConstraints(const IDProblem& problem);

  /// Builds acceleration/contact-force regularization blocks.
  void buildRegularizationBlocks(const IDProblem& problem);

  /// Builds weighted objective blocks from `IDProblem`.
  void buildObjectiveBlocks(const IDProblem& problem);

  /// Assembles the IDHQP hierarchy for the current `IDProblem`.
  void assembleHierarchy(const IDProblem& problem);

  /// Decodes `[delta_qddot, lambda]` and computes model torque.
  const IDSolution& decodeSolution(const IDProblem& problem,
                                   const solvers::HQPOutput& hqpSol);

  /// Computes `tau_sol` from `qddot_sol`, contact forces, and model terms.
  void computeModelTorque(const IDProblem& problem);

  /// Resizes or recreates the numerical backend from assembled HQP dimensions.
  void resizeSolverFromHQPData();

  solvers::HQPData m_hqpData;
  std::unique_ptr<solvers::SolverHQPBase> m_solver;
  solvers::SolverHQP m_solverType;
  solvers::SolverQPParams m_qpParams;
  unsigned int m_solverVarDim{0};
  unsigned int m_solverEqDim{0};
  unsigned int m_solverInDim{0};
  HQPBuildContext m_ctx;
  FloatingBaseDynamicsConstraint m_dynamicsConstraint;
  ContactConsistencyConstraint m_contactConsistencyConstraint;
  FrictionConeConstraint m_frictionConeConstraint;
  JointTorqueLimitConstraint m_jointTorqueLimitConstraint;
  AccelerationRegularization m_accelerationRegularization;
  ContactForceRegularization m_lambdaRegularization;

  std::vector<MotionObjectiveSlot> m_motionObjectiveSlots;
  std::vector<JointAccelerationObjectiveSlot> m_jointAccelerationObjectiveSlots;

  SolveWorkspace m_workspace;

  constraints::JointTorqueLimits m_jointTorqueLimits;
  const Eigen::VectorXd* m_h_ext{nullptr};

  TimingStats m_timingStats;
};

}  // namespace wbc

#endif  // ifndef __wbc_controller_id_hqp_hpp__
