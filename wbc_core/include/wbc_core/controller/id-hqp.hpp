//
// Copyright (c) 2026
//
// Final-form IDHQP core header.
//
// IDHQP is the delta-form generic level-based inverse-dynamics HQP controller.
// Shared step-problem / solution schemas live in dedicated headers and are
// consumed directly by this core.
//
// IDHQP does NOT:
// - model objects as optimization states or dynamic bodies
// - optimize hand-object interaction forces for dexterous manipulation
// - perform manipulation strategy reasoning
// - generate nominal or bias references internally
//
// IDHQP ONLY:
// - enforces dynamic feasibility
// - solves user-supplied soft objectives at explicit levels
// - computes minimal correction around qddot_ref
//

#ifndef __wbc_controller_id_hqp_hpp__
#define __wbc_controller_id_hqp_hpp__

#include <memory>
#include <unordered_map>
#include <vector>

#include <Eigen/Dense>

#include "wbc_core/controller/id-hierarchy-policy.hpp"
#include "wbc_core/controller/id-problem.hpp"
#include "wbc_core/controller/id-solution.hpp"
#include "wbc_core/formulations/hqp/blocks/contact-acceleration-block.hpp"
#include "wbc_core/formulations/hqp/blocks/floating-base-dynamics-block.hpp"
#include "wbc_core/formulations/hqp/blocks/friction-cone-block.hpp"
#include "wbc_core/formulations/hqp/blocks/joint-accel-bias-block.hpp"
#include "wbc_core/formulations/hqp/blocks/motion-constraint-block.hpp"
#include "wbc_core/formulations/hqp/blocks/qddot-regularization-block.hpp"
#include "wbc_core/formulations/hqp/blocks/rf-regularization-block.hpp"
#include "wbc_core/formulations/hqp/blocks/torque-limit-block.hpp"
#include "wbc_core/formulations/hqp/hqp-build-context.hpp"
#include "wbc_core/robots/robot-system.hpp"
#include "wbc_core/solvers/solver-HQP-base.hpp"
#include "wbc_core/solvers/solver-qp-params.hpp"

namespace wbc {

enum class HardTorqueLimitMode {
  DIAGONAL_M_BOX,
  EXACT_DENSE,
};

class IDHQP {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef math::Vector Vector;
  typedef math::Matrix Matrix;
  typedef pinocchio::Data Data;

  struct TimingStats {
    double qp_setup_us{0.0};
    double qp_solve_us{0.0};
  };

  explicit IDHQP(robots::RobotSystem& robot);
  IDHQP(robots::RobotSystem& robot, solvers::SolverHQP solver_type);
  IDHQP(robots::RobotSystem& robot, solvers::SolverHQP solver_type,
        const solvers::SolverQPParams& qp_params);

  const IDSolution& solve(const IDProblem& problem, double dt);

  const IDSolution& solution() const { return *m_solution; }
  const Vector& referenceAcceleration() const { return m_solution->qddot_ref; }
  const Vector& deltaAcceleration() const { return m_solution->delta_qddot; }
  const Vector& solvedAcceleration() const { return m_solution->qddot_sol; }
  Data& data() { return m_data; }
  const Data& data() const { return m_data; }
  const HQPBuildContext& context() const { return m_ctx; }
  void setTimingEnabled(bool enabled) { m_timingEnabled = enabled; }
  bool timingEnabled() const { return m_timingEnabled; }
  const TimingStats& timingStats() const { return m_timingStats; }
  solvers::SolverHQP solverType() const { return m_solverType; }
  void setSolverType(solvers::SolverHQP solver_type);
  const solvers::SolverQPParams& qpParams() const { return m_qpParams; }
  void setQPParams(const solvers::SolverQPParams& qp_params);

  // Compatibility shim only. Generic IDHQP always uses the current torque-limit
  // block path and does not branch on this mode.
  [[deprecated("HardTorqueLimitMode is not wired in generic IDHQP")]]
  void setHardTorqueLimitMode(HardTorqueLimitMode mode) {
    (void)mode;
  }

  [[deprecated("HardTorqueLimitMode is not wired in generic IDHQP")]]
  HardTorqueLimitMode hardTorqueLimitMode() const {
    return HardTorqueLimitMode::EXACT_DENSE;
  }

  [[deprecated("HardTorqueLimitMode is not wired in generic IDHQP")]]
  void SetHardTorqueLimitMode(HardTorqueLimitMode mode) {
    setHardTorqueLimitMode(mode);
  }

 private:
  struct ObjectiveSlot {
    std::unique_ptr<MotionConstraintBlock> motion_constraint;
    std::unique_ptr<JointAccelerationBias> joint_bias;
    std::shared_ptr<math::ConstraintBase> constraint;

    bool holdsMotionConstraint() const {
      return static_cast<bool>(motion_constraint);
    }
    bool holdsJointBias() const { return static_cast<bool>(joint_bias); }
  };

  struct ContactLayout {
    int lambdaOffset{0};
    int lambdaDim{0};
  };

  struct CycleWorkspace {
    int lambdaDim{0};
    bool hasContactForces{false};
    bool hasContactKinematics{false};
    bool hasFrictionConstraints{false};
    bool hasTorqueLimits{false};
    bool regularizeLambda{false};

    Matrix Jc;
    Vector Jcdot_qdot;
    Matrix Uf;
    Vector uf_lb;
    Vector uf_ub;
    std::unordered_map<std::string, ContactLayout> contactLayout;
  };

  static void ensureObjectiveCapacity(std::vector<ObjectiveSlot>& pool,
                                      std::size_t nObjectives);

  void beginCycle(const IDProblem& problem);
  const IDSolution& fail() const { return *m_solution; }
  bool validateInput(const IDProblem& problem, double dt) const;
  void updateRobotModel();
  void prepareCycleWorkspace(const IDProblem& problem);
  void stackContactData(const IDProblem& problem);
  void buildContext(const IDProblem& problem);
  void buildHardConstraints(const IDProblem& problem);
  void buildRegularizationBlocks(const IDProblem& problem);
  void buildObjectiveBlocks(const IDProblem& problem);
  void assembleHierarchy(const IDProblem& problem);
  const IDSolution& decodeSolution(const IDProblem& problem,
                                   const solvers::HQPOutput& hqpSol, double dt);
  void resetFailedSolution();
  void integrateSolutionState(double dt);
  void recoverTorque(const IDProblem& problem);
  void resizeSolverFromHQPData();

  robots::RobotSystem& m_robot;
  Data m_data;
  int m_nv;
  int m_na;
  int m_nvFloat;

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
  TorqueLimitConstraint m_torqueLimitConstraint;
  AccelerationRegularization m_accelerationRegularization;
  ContactForceRegularization m_lambdaRegularization;

  std::vector<ObjectiveSlot> m_objectiveSlots;

  CycleWorkspace m_cycle;

  constraints::ActuatorTorqueLimits m_torqueLimits;
  const Eigen::VectorXd* m_h_ext{nullptr};

  Vector m_zeroQddotRef;
  Vector m_qddotRefCurrent;
  Vector m_tauFull;
  Vector m_zero_h_ext;
  Vector m_integrateDelta;
  std::unique_ptr<IDSolution> m_solution;
  bool m_timingEnabled{false};
  TimingStats m_timingStats;
};

}  // namespace wbc

#endif  // ifndef __wbc_controller_id_hqp_hpp__
