//
// Copyright (c) 2026
//
// Final-form WBMC core header.
//
// WBMC is the delta-form generic level-based inverse-dynamics HQP controller.
// Shared step-input / solution schemas live in dedicated headers and are
// consumed directly by this core.
//
// WBMC does NOT:
// - model objects as optimization states or dynamic bodies
// - optimize hand-object interaction forces for dexterous manipulation
// - perform manipulation strategy reasoning
// - generate nominal or bias references internally
//
// WBMC ONLY:
// - enforces dynamic feasibility
// - solves user-supplied soft objectives at explicit levels
// - computes minimal correction around qddot_ref
//

#ifndef __wbc_controller_wbmc_hpp__
#define __wbc_controller_wbmc_hpp__

#include <memory>
#include <unordered_map>
#include <vector>

#include <Eigen/Dense>

#include "wbc_core/controller/wbmc-hierarchy-policy.hpp"
#include "wbc_core/controller/wbmc-step-input.hpp"
#include "wbc_core/controller/wbmc-solution.hpp"
#include "wbc_core/formulations/hqp/blocks/contact-acceleration-block.hpp"
#include "wbc_core/formulations/hqp/blocks/floating-base-dynamics-block.hpp"
#include "wbc_core/formulations/hqp/blocks/friction-cone-block.hpp"
#include "wbc_core/formulations/hqp/blocks/joint-accel-bias-block.hpp"
#include "wbc_core/formulations/hqp/blocks/motion-task-block.hpp"
#include "wbc_core/formulations/hqp/blocks/qddot-regularization-block.hpp"
#include "wbc_core/formulations/hqp/blocks/rf-regularization-block.hpp"
#include "wbc_core/formulations/hqp/blocks/torque-limit-block.hpp"
#include "wbc_core/formulations/hqp/hqp-build-context.hpp"
#include "wbc_core/robots/robot-system.hpp"
#include "wbc_core/solvers/solver-HQP-base.hpp"

namespace tsid {

enum class HardTorqueLimitMode {
  DIAGONAL_M_BOX,
  EXACT_DENSE,
};

class WBMC {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef math::Vector Vector;
  typedef math::Matrix Matrix;
  typedef pinocchio::Data Data;

  struct TimingStats {
    double qp_setup_us{0.0};
    double qp_solve_us{0.0};
  };

  explicit WBMC(robots::RobotSystem& robot);

  const WBMCSolution& solve(const WBMCStepInput& input);

  const WBMCSolution& solution() const { return *m_solution; }
  const Vector& referenceAcceleration() const { return m_solution->qddot_ref; }
  const Vector& deltaAcceleration() const { return m_solution->delta_qddot; }
  const Vector& solvedAcceleration() const { return m_solution->qddot_sol; }
  Data& data() { return m_data; }
  const Data& data() const { return m_data; }
  const HQPBuildContext& context() const { return m_ctx; }
  void setTimingEnabled(bool enabled) { m_timingEnabled = enabled; }
  bool timingEnabled() const { return m_timingEnabled; }
  const TimingStats& timingStats() const { return m_timingStats; }

  // Compatibility shim only. Generic WBMC always uses the current torque-limit
  // block path and does not branch on this mode.
  [[deprecated("HardTorqueLimitMode is not wired in generic WBMC")]]
  void setHardTorqueLimitMode(HardTorqueLimitMode mode) {
    (void)mode;
  }

  [[deprecated("HardTorqueLimitMode is not wired in generic WBMC")]]
  HardTorqueLimitMode hardTorqueLimitMode() const {
    return HardTorqueLimitMode::EXACT_DENSE;
  }

  [[deprecated("HardTorqueLimitMode is not wired in generic WBMC")]]
  void SetHardTorqueLimitMode(HardTorqueLimitMode mode) {
    setHardTorqueLimitMode(mode);
  }

 private:
  struct ObjectiveSlot {
    std::unique_ptr<MotionTask> motion;
    std::unique_ptr<JointAccelerationBias> joint_bias;
    std::shared_ptr<math::ConstraintBase> constraint;

    bool holdsMotion() const { return static_cast<bool>(motion); }
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

  void beginCycle(const WBMCStepInput& input);
  const WBMCSolution& fail() const { return *m_solution; }
  bool validateInput(const WBMCStepInput& input) const;
  void updateRobotModel(const WBMCStepInput& input);
  void prepareCycleWorkspace(const WBMCStepInput& input);
  void stackContactData(const WBMCStepInput& input);
  void buildContext(const WBMCStepInput& input);
  void buildHardConstraints(const WBMCStepInput& input);
  void buildRegularizationBlocks(const WBMCStepInput& input);
  void buildObjectiveBlocks(const WBMCStepInput& input);
  void assembleHierarchy(const WBMCStepInput& input);
  const WBMCSolution& decodeSolution(const WBMCStepInput& input,
                                     const solvers::HQPOutput& hqpSol);
  void resetFailedSolution();
  void recoverTorque(const WBMCStepInput& input);
  void resizeSolverFromHQPData();

  robots::RobotSystem& m_robot;
  Data m_data;
  int m_nv;
  int m_na;
  int m_nvFloat;

  solvers::HQPData m_hqpData;
  std::unique_ptr<solvers::SolverHQPBase> m_solver;
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

  const Eigen::VectorXd* m_tau_lb{nullptr};
  const Eigen::VectorXd* m_tau_ub{nullptr};
  const Eigen::VectorXd* m_h_ext{nullptr};

  Vector m_zeroQddotRef;
  Vector m_qddotRefCurrent;
  Vector m_tauFull;
  Vector m_zero_h_ext;
  std::unique_ptr<WBMCSolution> m_solution;
  bool m_timingEnabled{false};
  TimingStats m_timingStats;
};

}  // namespace tsid

#endif  // ifndef __wbc_controller_wbmc_hpp__
