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

class IDHQP : public IDBase {
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

  const IDSolution& solve(const IDProblem& problem, double dt) override;

  const IDSolution& solution() const override { return *m_solution; }
  const Vector& referenceAcceleration() const { return m_solution->qddot_ref; }
  const Vector& deltaAcceleration() const {
    return m_solution->delta_qddot_sol;
  }
  const Vector& solvedAcceleration() const { return m_solution->qddot_sol; }
  Data& data() override { return m_data; }
  const Data& data() const override { return m_data; }
  void setTimingEnabled(bool enabled) override { m_timingEnabled = enabled; }
  bool timingEnabled() const override { return m_timingEnabled; }
  const TimingStats& timingStats() const { return m_timingStats; }
  solvers::SolverHQP solverType() const override { return m_solverType; }
  void setSolverType(solvers::SolverHQP solver_type) override;
  const solvers::SolverQPParams& qpParams() const override {
    return m_qpParams;
  }
  void setQPParams(const solvers::SolverQPParams& qp_params) override;

 private:
  struct MotionObjectiveSlot {
    std::unique_ptr<MotionConstraintBlock> block;
    std::shared_ptr<math::ConstraintBase> constraint;
  };

  struct JointAccelerationObjectiveSlot {
    std::unique_ptr<JointAccelerationBias> block;
    std::shared_ptr<math::ConstraintBase> constraint;
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
    bool hasJointTorqueLimits{false};
    bool regularizeLambda{false};

    Matrix Jc;
    Vector contact_motion_rhs;
    Matrix Uf;
    Vector uf_lb;
    Vector uf_ub;
    std::unordered_map<std::string, ContactLayout> contactLayout;
  };

  void beginCycle(const IDProblem& problem);
  const IDSolution& fail() const { return *m_solution; }
  bool validateInput(const IDProblem& problem, double dt) const;
  bool validateHierarchy(const IDProblem& problem) const;
  void updateRobotModel();
  void prepareCycleWorkspace(const IDProblem& problem);
  void stackContactData(const IDProblem& problem);
  void buildContext(const IDProblem& problem);
  void buildHardConstraints(const IDProblem& problem);
  void buildRegularizationBlocks(const IDProblem& problem);
  void buildObjectiveBlocks(const IDProblem& problem);
  void assembleHierarchy(const IDProblem& problem);
  const IDSolution& decodeSolution(const IDProblem& problem,
                                   const solvers::HQPOutput& hqpSol);
  void resetFailedSolution();
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
  JointTorqueLimitConstraint m_jointTorqueLimitConstraint;
  AccelerationRegularization m_accelerationRegularization;
  ContactForceRegularization m_lambdaRegularization;

  std::vector<MotionObjectiveSlot> m_motionObjectiveSlots;
  std::vector<JointAccelerationObjectiveSlot> m_jointAccelerationObjectiveSlots;

  CycleWorkspace m_cycle;

  constraints::JointTorqueLimits m_jointTorqueLimits;
  const Eigen::VectorXd* m_h_ext{nullptr};

  Vector m_zeroQddotRef;
  Vector m_qddotRefCurrent;
  Vector m_tauFull;
  Vector m_zero_h_ext;
  std::unique_ptr<IDSolution> m_solution;
  bool m_timingEnabled{false};
  TimingStats m_timingStats;
};

}  // namespace wbc

#endif  // ifndef __wbc_controller_id_hqp_hpp__
