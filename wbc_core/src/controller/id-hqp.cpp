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
  const Vector& qddot_ref = problemQddotRef(problem);

  beginCycle(qddot_ref);
  if (!validateInput(problem, dt)) {
    return fail();
  }

  updateRobotModel();
  prepareProblemAssembly(problem);
  resetSolution(qddot_ref, m_contacts.lambdaDim);
  const HQPBlockContext ctx = makeHqpBlockContext(problem.contacts, qddot_ref);
  buildHardConstraints(ctx);
  buildRegularizationBlocks(problem.regularization, ctx);
  buildObjectiveBlocks(problem.motion_objectives,
                       problem.joint_acceleration_objectives, ctx);
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
  return decodeSolution(problem, hqpSol, qddot_ref);
}

void IDHQP::beginCycle(const Vector& qddot_ref) {
  m_contacts = StackedContactData{};
  beginSolveCycle(qddot_ref);
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

void IDHQP::prepareProblemAssembly(const IDProblem& problem) {
  stackContactData(m_contacts, problem.contacts);

  m_jointTorqueLimits = problem.joint_torque_limits;
  m_h_ext = problem.h_ext;
}

bool IDHQP::lambdaRegularizationEnabled(
    const IDRegularizationParams& regularization) const {
  return m_contacts.hasContactForces() &&
         (regularization.w_lambda > 0.0);
}

HQPBlockContext IDHQP::makeHqpBlockContext(
    const std::vector<ContactConstraintData>& contacts,
    const Vector& qddot_ref) {
  HQPBlockContext ctx;
  ctx.M = &m_robot.mass(m_data);
  ctx.h = &m_robot.nonLinearEffects(m_data);
  ctx.nv = m_nv;
  ctx.na = m_na;
  ctx.nvFloat = m_nvFloat;
  ctx.lambdaDim = m_contacts.lambdaDim;

  ctx.contactInfos.reserve(contacts.size());
  for (const auto& contact : contacts) {
    auto it = m_contacts.contactLayout.find(contact.name);
    assert(it != m_contacts.contactLayout.end());
    ctx.contactInfos.push_back({&contact.Jc, &contact.T,
                                it->second.lambdaOffset,
                                it->second.lambdaDim});
  }

  ctx.Jc = &m_contacts.Jc;
  ctx.contact_motion_rhs = &m_contacts.contact_motion_rhs;
  ctx.Uf = &m_contacts.Uf;
  ctx.uf_lb = &m_contacts.uf_lb;
  ctx.uf_ub = &m_contacts.uf_ub;

  ctx.h_ext = m_h_ext ? m_h_ext : &zeroExternalWrench();

  ctx.enableJointTorqueLimits = m_jointTorqueLimits.enabled();
  ctx.tau_lb = m_jointTorqueLimits.lower;
  ctx.tau_ub = m_jointTorqueLimits.upper;

  ctx.qddot_ref = &qddot_ref;
  return ctx;
}

void IDHQP::buildHardConstraints(const HQPBlockContext& ctx) {
  m_dynamicsConstraint.build(ctx);
  if (m_contacts.hasContactKinematics()) {
    m_contactConsistencyConstraint.build(ctx);
  }
  if (m_contacts.hasFrictionConstraints()) {
    m_frictionConeConstraint.build(ctx);
  }
  if (m_jointTorqueLimits.enabled()) {
    m_jointTorqueLimitConstraint.build(ctx);
  }
}

void IDHQP::buildRegularizationBlocks(
    const IDRegularizationParams& regularization,
    const HQPBlockContext& ctx) {
  m_accelerationRegularization.setWeight(regularization.w_delta_qddot);
  m_accelerationRegularization.build(ctx);
  if (lambdaRegularizationEnabled(regularization)) {
    m_lambdaRegularization.setWeight(regularization.w_lambda);
    m_lambdaRegularization.build(ctx);
  }
}

void IDHQP::buildObjectiveBlocks(
    const std::vector<MotionObjective>& motion_objectives,
    const std::vector<JointAccelerationObjective>& joint_acceleration_objectives,
    const HQPBlockContext& ctx) {
  m_motionObjectiveSlots.resize(motion_objectives.size());
  m_jointAccelerationObjectiveSlots.resize(
      joint_acceleration_objectives.size());

  for (std::size_t i = 0; i < motion_objectives.size(); ++i) {
    const auto& objective = motion_objectives[i];
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
    slot.block->build(objective, ctx);
    slot.constraint = slot.block->constraint();
  }

  for (std::size_t i = 0; i < joint_acceleration_objectives.size(); ++i) {
    const auto& objective = joint_acceleration_objectives[i];
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
    slot.block->build(ctx);
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
  if (m_contacts.hasContactKinematics()) {
    solvers::hqp::addTerm(m_hqpData, solvers::hqp::kLevel0, 1.0,
                          m_contactConsistencyConstraint.constraint());
  }
  if (m_contacts.hasFrictionConstraints()) {
    solvers::hqp::addTerm(m_hqpData, solvers::hqp::kLevel0, 1.0,
                          m_frictionConeConstraint.constraint());
  }
  if (m_jointTorqueLimits.enabled()) {
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
  if (lambdaRegularizationEnabled(problem.regularization)) {
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

const IDSolution& IDHQP::decodeSolution(const IDProblem& problem,
                                        const solvers::HQPOutput& hqpSol,
                                        const Vector& qddot_ref) {
  if (hqpSol.status != solvers::HQP_STATUS_OPTIMAL ||
      hqpSol.x.size() < m_nv + m_contacts.lambdaDim) {
    resetSolution(qddot_ref, m_contacts.lambdaDim);
    return fail();
  }

  setAccelerationSolution(qddot_ref, hqpSol.x.head(m_nv));

  if (m_contacts.hasContactForces()) {
    m_solution->lambda_sol = hqpSol.x.segment(m_nv, m_contacts.lambdaDim);
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

  if (m_contacts.hasContactForces()) {
    for (const auto& contact : problem.contacts) {
      auto it = m_contacts.contactLayout.find(contact.name);
      if (it == m_contacts.contactLayout.end() || it->second.lambdaDim == 0) {
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
