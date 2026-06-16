//
// Copyright (c) 2026
//
// Final-form IDHQP inverse-dynamics HQP core.
//

#include "wbc_core/controller/id-hqp.hpp"

#include <cassert>
#include <chrono>
#include <cmath>
#include <memory>

#include "wbc_core/solvers/hqp-data-utils.hpp"
#include "wbc_core/solvers/solver-HQP-cascade.hpp"
#include "wbc_core/solvers/solver-HQP-factory.hpp"
#include "wbc_core/solvers/solver-qp-params.hpp"

namespace wbc {

using namespace math;

namespace {

solvers::SolverHQP defaultSolverType() {
#ifdef TSID_WITH_PROXSUITE
  return solvers::SOLVER_HQP_PROXQP;
#else
  return solvers::SOLVER_HQP_EIQUADPROG;
#endif
}

std::unique_ptr<solvers::SolverHQPBase> makeSolver(
    solvers::SolverHQP solver_type, const solvers::SolverQPParams& qp_params) {
  auto solver =
      std::make_unique<solvers::SolverHQPCascade>("id-hqp", solver_type);
  solver->setQPParams(qp_params);
  return solver;
}

bool isObjectiveLevelValid(unsigned int level) {
  return level > solvers::hqp::kLevel0;
}

unsigned int regularizationLevel(unsigned int max_objective_level) {
  return max_objective_level + 1u;
}

unsigned int numHierarchyLevels(unsigned int max_objective_level) {
  return regularizationLevel(max_objective_level) + 1u;
}

}  // namespace

IDHQP::IDHQP(robots::RobotSystem& robot) : IDHQP(robot, defaultSolverType()) {}

IDHQP::IDHQP(robots::RobotSystem& robot, solvers::SolverHQP solver_type)
    : IDHQP(robot, solver_type, solvers::SolverQPParams{}) {}

IDHQP::IDHQP(robots::RobotSystem& robot, solvers::SolverHQP solver_type,
             const solvers::SolverQPParams& qp_params)
    : InverseDynamicsBase(robot),
      m_solverType(solver_type),
      m_qpParams(qp_params),
      m_dynamicsConstraint(0),
      m_contactConsistencyConstraint(0),
      m_frictionConeConstraint(0),
      m_jointTorqueLimitConstraint(0),
      m_accelerationRegularization(0, 1e-4),
      m_lambdaRegularization(0, 1e-5) {
  m_jointTorqueLimitConstraint.preallocate(m_na);

  m_solver = makeSolver(m_solverType, m_qpParams);
}

void IDHQP::setSolverType(solvers::SolverHQP solver_type) {
  if (m_solverType == solver_type) {
    return;
  }
  m_solverType = solver_type;
  m_solver = makeSolver(m_solverType, m_qpParams);
  m_solverVarDim = 0;
  m_solverEqDim = 0;
  m_solverInDim = 0;
}

void IDHQP::setQPParams(const solvers::SolverQPParams& qp_params) {
  m_qpParams = qp_params;
  if (auto* cascade =
          dynamic_cast<solvers::SolverHQPCascade*>(m_solver.get())) {
    cascade->setQPParams(m_qpParams);
  } else if (m_solver) {
    solvers::ApplySolverQPParams(*m_solver, m_qpParams);
  }
}

const IDSolution& IDHQP::solve(const IDProblem& problem, double dt) {
  using Clock = std::chrono::high_resolution_clock;
  const auto t0 = Clock::now();
  beginCycle(problem);
  if (!validateInput(problem, dt)) {
    return fail();
  }

  updateRobotModel();
  prepareSolveWorkspace(problem);
  buildHardConstraints(problem);
  buildRegularizationBlocks(problem);
  buildObjectiveBlocks(problem);
  assembleHierarchy(problem);
  resizeSolverFromHQPData();
  const auto t1 = Clock::now();

  const auto& hqpSol = m_solver->solve(m_hqpData);
  const auto t2 = Clock::now();
  if (m_timingEnabled) {
    auto us = [](auto a, auto b) {
      return std::chrono::duration<double, std::micro>(b - a).count();
    };
    m_timingStats.qp_setup_us = us(t0, t1);
    m_timingStats.qp_solve_us = us(t1, t2);
  }
  return decodeSolution(problem, hqpSol);
}

void IDHQP::beginCycle(const IDProblem& problem) {
  m_workspace = SolveWorkspace{};
  beginSolveCycle(problem);
}

bool IDHQP::validateInput(const IDProblem& problem, double dt) const {
  return validateBaseInput(dt) && validateHierarchy(problem);
}

bool IDHQP::validateHierarchy(const IDProblem& problem) const {
  for (const auto& objective : problem.motion_objectives) {
    if (objective.isEquality() && !isObjectiveLevelValid(objective.level)) {
      return false;
    }
  }
  for (const auto& objective : problem.joint_acceleration_objectives) {
    if (!isObjectiveLevelValid(objective.level)) {
      return false;
    }
  }
  return solvers::hqp::kLevel0 <
         regularizationLevel(problem.maxObjectiveLevel());
}

void IDHQP::prepareSolveWorkspace(const IDProblem& problem) {
  stackContactData(m_workspace.contacts, problem.contacts);

  m_jointTorqueLimits = problem.joint_torque_limits;
  m_h_ext = problem.h_ext;

  m_workspace.hasContactForces = m_workspace.contacts.hasContactForces();
  m_workspace.hasContactKinematics =
      m_workspace.contacts.hasContactKinematics();
  m_workspace.hasFrictionConstraints =
      m_workspace.contacts.hasFrictionConstraints();
  m_workspace.hasJointTorqueLimits = m_jointTorqueLimits.enabled();
  m_workspace.regularizeLambda =
      m_workspace.hasContactForces && (problem.regularization.w_lambda > 0.0);

  resetSolution(problemQddotRef(problem), m_workspace.contacts.lambdaDim);
  buildHqpContext(problem);
}

void IDHQP::buildHardConstraints(const IDProblem& problem) {
  (void)problem;
  m_dynamicsConstraint.build(m_ctx);
  if (m_workspace.hasContactKinematics) {
    m_contactConsistencyConstraint.build(m_ctx);
  }
  if (m_workspace.hasFrictionConstraints) {
    m_frictionConeConstraint.build(m_ctx);
  }
  if (m_workspace.hasJointTorqueLimits) {
    m_jointTorqueLimitConstraint.build(m_ctx);
  }
}

void IDHQP::buildRegularizationBlocks(const IDProblem& problem) {
  m_accelerationRegularization.setWeight(problem.regularization.w_delta_qddot);
  m_accelerationRegularization.build(m_ctx);
  if (m_workspace.regularizeLambda) {
    m_lambdaRegularization.setWeight(problem.regularization.w_lambda);
    m_lambdaRegularization.build(m_ctx);
  }
}

void IDHQP::buildObjectiveBlocks(const IDProblem& problem) {
  m_motionObjectiveSlots.resize(problem.motion_objectives.size());
  m_jointAccelerationObjectiveSlots.resize(
      problem.joint_acceleration_objectives.size());

  for (std::size_t i = 0; i < problem.motion_objectives.size(); ++i) {
    const auto& objective = problem.motion_objectives[i];
    auto& slot = m_motionObjectiveSlots[i];
    const bool needsRebuild =
        !slot.block || slot.block->level() != objective.level ||
        slot.block->name() != objective.name;
    if (needsRebuild) {
      auto block = std::make_unique<MotionConstraintBlock>(
          objective.name, objective.level, objective.weight);
      slot.constraint = block->constraint();
      slot.block = std::move(block);
    }

    slot.block->setWeight(objective.weight);
    slot.block->build(objective, m_ctx);
    slot.constraint = slot.block->constraint();
  }

  for (std::size_t i = 0; i < problem.joint_acceleration_objectives.size();
       ++i) {
    const auto& objective = problem.joint_acceleration_objectives[i];
    auto& slot = m_jointAccelerationObjectiveSlots[i];
    const bool needsRebuild = !slot.block ||
                              slot.block->level() != objective.level ||
                              slot.block->name() != objective.name;
    if (needsRebuild) {
      auto biasUnit = std::make_unique<JointAccelerationBias>(
          objective.name, objective.level, objective.weight);
      slot.constraint = biasUnit->constraint();
      slot.block = std::move(biasUnit);
    }
    slot.block->setWeight(objective.weight);
    slot.block->setQddotBias(objective.qddot_target);
    slot.block->build(m_ctx);
    slot.constraint = slot.block->constraint();
  }
}

void IDHQP::assembleHierarchy(const IDProblem& problem) {
  solvers::hqp::resizeData(
      m_hqpData, numHierarchyLevels(problem.maxObjectiveLevel()));

  if (m_nvFloat > 0) {
    solvers::hqp::addTerm(m_hqpData, solvers::hqp::kLevel0, 1.0,
                          m_dynamicsConstraint.constraint());
  }
  if (m_workspace.hasContactKinematics) {
    solvers::hqp::addTerm(m_hqpData, solvers::hqp::kLevel0, 1.0,
                          m_contactConsistencyConstraint.constraint());
  }
  if (m_workspace.hasFrictionConstraints) {
    solvers::hqp::addTerm(m_hqpData, solvers::hqp::kLevel0, 1.0,
                          m_frictionConeConstraint.constraint());
  }
  if (m_workspace.hasJointTorqueLimits) {
    solvers::hqp::addTerm(m_hqpData, solvers::hqp::kLevel0, 1.0,
                          m_jointTorqueLimitConstraint.constraint());
  }

  for (std::size_t i = 0; i < problem.motion_objectives.size(); ++i) {
    const auto& objective = problem.motion_objectives[i];
    if (objective.isEquality()) {
      solvers::hqp::addTerm(m_hqpData, objective.level, objective.weight,
                            m_motionObjectiveSlots[i].constraint);
    } else {
      solvers::hqp::addTerm(m_hqpData, solvers::hqp::kLevel0, 1.0,
                            m_motionObjectiveSlots[i].constraint);
    }
  }

  for (std::size_t i = 0; i < problem.joint_acceleration_objectives.size();
       ++i) {
    const auto& objective = problem.joint_acceleration_objectives[i];
    solvers::hqp::addTerm(m_hqpData, objective.level, objective.weight,
                          m_jointAccelerationObjectiveSlots[i].constraint);
  }

  const unsigned int regLevel =
      regularizationLevel(problem.maxObjectiveLevel());
  solvers::hqp::addTerm(m_hqpData, regLevel,
                        problem.regularization.w_delta_qddot,
                        m_accelerationRegularization.constraint());
  if (m_workspace.regularizeLambda) {
    solvers::hqp::addTerm(m_hqpData, regLevel,
                          problem.regularization.w_lambda,
                          m_lambdaRegularization.constraint());
  }
}

void IDHQP::resizeSolverFromHQPData() {
  const solvers::hqp::Dimensions dimensions =
      solvers::hqp::dimensions(m_hqpData);

  if (dimensions.variables > 0) {
    if (!m_solver || m_solverVarDim != dimensions.variables ||
        m_solverEqDim != dimensions.equalities ||
        m_solverInDim != dimensions.inequalities) {
      m_solver = makeSolver(m_solverType, m_qpParams);
      m_solverVarDim = dimensions.variables;
      m_solverEqDim = dimensions.equalities;
      m_solverInDim = dimensions.inequalities;
    }
    m_solver->resize(dimensions.variables, dimensions.equalities,
                     dimensions.inequalities);
  }
}

void IDHQP::buildHqpContext(const IDProblem& problem) {
  m_ctx = HQPBuildContext{};
  m_ctx.M = &m_robot.mass(m_data);
  m_ctx.h = &m_robot.nonLinearEffects(m_data);
  m_ctx.nv = m_nv;
  m_ctx.na = m_na;
  m_ctx.nvFloat = m_nvFloat;
  m_ctx.lambdaDim = m_workspace.contacts.lambdaDim;

  m_ctx.contactInfos.reserve(problem.contacts.size());
  for (const auto& contact : problem.contacts) {
    auto it = m_workspace.contacts.contactLayout.find(contact.name);
    assert(it != m_workspace.contacts.contactLayout.end());
    m_ctx.contactInfos.push_back({&contact.Jc, &contact.T,
                                  it->second.lambdaOffset,
                                  it->second.lambdaDim});
  }

  if (m_workspace.hasContactKinematics) {
    m_ctx.Jc = &m_workspace.contacts.Jc;
    m_ctx.contact_motion_rhs = &m_workspace.contacts.contact_motion_rhs;
  }
  if (m_workspace.hasContactForces) {
    m_ctx.Uf = &m_workspace.contacts.Uf;
    m_ctx.uf_lb = &m_workspace.contacts.uf_lb;
    m_ctx.uf_ub = &m_workspace.contacts.uf_ub;
  }

  m_ctx.h_ext = m_h_ext ? m_h_ext : &zeroExternalWrench();

  if (m_workspace.hasJointTorqueLimits) {
    m_ctx.enableJointTorqueLimits = true;
    m_ctx.tau_lb = m_jointTorqueLimits.lower;
    m_ctx.tau_ub = m_jointTorqueLimits.upper;
  }

  m_ctx.qddot_ref = &problemQddotRef(problem);
}

const IDSolution& IDHQP::decodeSolution(const IDProblem& problem,
                                        const solvers::HQPOutput& hqpSol) {
  if (hqpSol.status != solvers::HQP_STATUS_OPTIMAL ||
      hqpSol.x.size() < m_nv + m_workspace.contacts.lambdaDim) {
    resetSolution(problemQddotRef(problem), m_workspace.contacts.lambdaDim);
    return fail();
  }

  setAccelerationSolution(problemQddotRef(problem), hqpSol.x.head(m_nv));

  if (m_workspace.hasContactForces) {
    m_solution->lambda_sol =
        hqpSol.x.segment(m_nv, m_workspace.contacts.lambdaDim);
  } else {
    m_solution->lambda_sol.resize(0);
  }

  computeModelTorque(problem);
  m_solution->success = solutionVectorsFinite();
  return *m_solution;
}

void IDHQP::computeModelTorque(const IDProblem& problem) {
  const Matrix& M = m_robot.mass(m_data);
  const Vector& h = m_robot.nonLinearEffects(m_data);

  m_tauFull.noalias() = M * m_solution->qddot_sol;
  m_tauFull += h;

  if (m_h_ext) {
    m_tauFull -= *m_h_ext;
  }

  if (m_workspace.hasContactForces) {
    for (const auto& contact : problem.contacts) {
      auto it = m_workspace.contacts.contactLayout.find(contact.name);
      if (it == m_workspace.contacts.contactLayout.end() ||
          it->second.lambdaDim == 0) {
        continue;
      }

      const auto& block = it->second;
      m_tauFull.noalias() -=
          contact.Jc.transpose() * contact.T *
          m_solution->lambda_sol.segment(block.lambdaOffset, block.lambdaDim);
    }
  }

  m_solution->tau_sol = m_tauFull.tail(m_na);
}

}  // namespace wbc
