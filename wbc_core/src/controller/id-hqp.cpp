//
// Copyright (c) 2026
//
// Final-form IDHQP inverse-dynamics HQP core.
//

#include "wbc_core/controller/id-hqp.hpp"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>

#include <pinocchio/algorithm/joint-configuration.hpp>

#include "wbc_core/solvers/solver-HQP-cascade.hpp"
#include "wbc_core/solvers/solver-HQP-factory.hpp"
#include "wbc_core/solvers/solver-qp-params.hpp"

namespace wbc {

using namespace math;

namespace {

void appendTerm(solvers::ConstraintLevel& level, double weight,
                const std::shared_ptr<ConstraintBase>& constraint) {
  level.emplace_back(weight, constraint);
}

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

}  // namespace

IDHQP::IDHQP(robots::RobotSystem& robot) : IDHQP(robot, defaultSolverType()) {}

IDHQP::IDHQP(robots::RobotSystem& robot, solvers::SolverHQP solver_type)
    : IDHQP(robot, solver_type, solvers::SolverQPParams{}) {}

IDHQP::IDHQP(robots::RobotSystem& robot, solvers::SolverHQP solver_type,
             const solvers::SolverQPParams& qp_params)
    : m_robot(robot),
      m_data(robot.model()),
      m_solverType(solver_type),
      m_qpParams(qp_params),
      m_dynamicsConstraint(0),
      m_contactConsistencyConstraint(0),
      m_frictionConeConstraint(0),
      m_torqueLimitConstraint(0),
      m_accelerationRegularization(0, 1e-4),
      m_lambdaRegularization(0, 1e-5) {
  m_nv = robot.nv();
  m_na = robot.na();
  m_nvFloat = m_nv - m_na;

  m_zeroQddotRef = Vector::Zero(m_nv);
  m_qddotRefCurrent = Vector::Zero(m_nv);
  m_zero_h_ext = Vector::Zero(m_nv);
  m_tauFull = Vector::Zero(m_nv);
  m_integrateDelta = Vector::Zero(m_nv);
  m_solution = std::make_unique<IDSolution>();
  m_solution->qddot_ref = Vector::Zero(m_nv);
  m_solution->delta_qddot = Vector::Zero(m_nv);
  m_solution->qddot_sol = Vector::Zero(m_nv);
  m_solution->qdot_cmd = Vector::Zero(m_nv);
  m_solution->q_cmd = pinocchio::neutral(robot.model());
  m_solution->lambda_sol = Vector::Zero(0);
  m_solution->tau_ff_cmd = Vector::Zero(m_na);
  m_solution->tau_fb_cmd = Vector::Zero(m_na);
  m_solution->tau_cmd = Vector::Zero(m_na);

  m_torqueLimitConstraint.preallocate(m_na);

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
  prepareCycleWorkspace(problem);
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
  return decodeSolution(problem, hqpSol, dt);
}

void IDHQP::ensureObjectiveCapacity(std::vector<ObjectiveSlot>& pool,
                                    std::size_t nObjectives) {
  if (pool.size() < nObjectives) {
    pool.resize(nObjectives);
  }
}

void IDHQP::beginCycle(const IDProblem& problem) {
  m_solution->success = false;
  m_cycle = CycleWorkspace{};

  if (problem.qddot_ref) {
    assert(problem.qddot_ref->size() == m_nv);
    m_qddotRefCurrent = *problem.qddot_ref;
  } else {
    m_qddotRefCurrent = m_zeroQddotRef;
  }

  m_solution->qddot_ref = m_qddotRefCurrent;
  resetFailedSolution();
}

bool IDHQP::validateInput(const IDProblem& problem, double dt) const {
  return m_robot.hasState() && std::isfinite(dt) && dt >= 0.0 &&
         problem.hasValidHierarchy();
}

void IDHQP::updateRobotModel() {
  assert(m_robot.generalized_q().size() == m_robot.nq());
  assert(m_robot.generalized_v().size() == m_nv);
  m_robot.computeAllTerms(m_data, m_robot.generalized_q(),
                          m_robot.generalized_v());
}

void IDHQP::prepareCycleWorkspace(const IDProblem& problem) {
  stackContactData(problem);

  m_torqueLimits = problem.torque_limits;
  m_h_ext = problem.h_ext;

  m_cycle.hasContactForces = (m_cycle.lambdaDim > 0);
  m_cycle.hasContactKinematics =
      m_cycle.hasContactForces && (m_cycle.Jc.rows() > 0);
  m_cycle.hasFrictionConstraints =
      m_cycle.hasContactForces && (m_cycle.Uf.rows() > 0);
  m_cycle.hasTorqueLimits = m_torqueLimits.enabled();
  m_cycle.regularizeLambda =
      m_cycle.hasContactForces && (problem.regularization.w_lambda > 0.0);

  resetFailedSolution();
  buildContext(problem);
}

void IDHQP::buildHardConstraints(const IDProblem& problem) {
  (void)problem;
  m_dynamicsConstraint.build(m_ctx);
  if (m_cycle.hasContactKinematics) {
    m_contactConsistencyConstraint.build(m_ctx);
  }
  if (m_cycle.hasFrictionConstraints) {
    m_frictionConeConstraint.build(m_ctx);
  }
  if (m_cycle.hasTorqueLimits) {
    m_torqueLimitConstraint.build(m_ctx);
  }
}

void IDHQP::buildRegularizationBlocks(const IDProblem& problem) {
  m_accelerationRegularization.setWeight(problem.regularization.w_delta_qddot);
  m_accelerationRegularization.build(m_ctx);
  if (m_cycle.regularizeLambda) {
    m_lambdaRegularization.setWeight(problem.regularization.w_lambda);
    m_lambdaRegularization.build(m_ctx);
  }
}

void IDHQP::buildObjectiveBlocks(const IDProblem& problem) {
  ensureObjectiveCapacity(m_objectiveSlots, problem.objectives.size());

  for (std::size_t i = 0; i < problem.objectives.size(); ++i) {
    const auto& objective = problem.objectives[i];
    auto& slot = m_objectiveSlots[i];

    if (objective.isMotionConstraint()) {
      const auto& motion = objective.motionConstraint();
      const bool needsRebuild =
          !slot.motion_constraint ||
          slot.motion_constraint->level() != objective.level ||
          slot.motion_constraint->name() != motion.name;
      if (needsRebuild) {
        slot.joint_bias.reset();
        auto block = std::make_unique<MotionConstraintBlock>(
            motion.name, objective.level, objective.weight);
        slot.constraint = block->constraint();
        slot.motion_constraint = std::move(block);
      }

      slot.motion_constraint->setWeight(objective.weight);
      slot.motion_constraint->build(motion, m_ctx);
      slot.constraint = slot.motion_constraint->constraint();
      continue;
    }

    const auto& jointObjective = objective.jointAccelerationTarget();
    const bool needsRebuild = !slot.joint_bias ||
                              slot.joint_bias->level() != objective.level ||
                              slot.joint_bias->name() != jointObjective.name;
    if (needsRebuild) {
      slot.motion_constraint.reset();
      auto biasUnit = std::make_unique<JointAccelerationBias>(
          jointObjective.name, objective.level, objective.weight);
      slot.constraint = biasUnit->constraint();
      slot.joint_bias = std::move(biasUnit);
    }
    slot.joint_bias->setWeight(objective.weight);
    slot.joint_bias->setQddotBias(jointObjective.qddot_target);
    slot.joint_bias->build(m_ctx);
    slot.constraint = slot.joint_bias->constraint();
  }
}

void IDHQP::assembleHierarchy(const IDProblem& problem) {
  const auto& policy = problem.hierarchy;
  m_hqpData.clear();
  m_hqpData.resize(policy.numLevels(problem.maxObjectiveLevel()));

  auto& physicsLevel = m_hqpData[policy.physicsLevel];
  if (m_nvFloat > 0) {
    appendTerm(physicsLevel, 1.0, m_dynamicsConstraint.constraint());
  }
  if (m_cycle.hasContactKinematics) {
    appendTerm(physicsLevel, 1.0, m_contactConsistencyConstraint.constraint());
  }
  if (m_cycle.hasFrictionConstraints) {
    appendTerm(physicsLevel, 1.0, m_frictionConeConstraint.constraint());
  }
  if (m_cycle.hasTorqueLimits) {
    appendTerm(physicsLevel, 1.0, m_torqueLimitConstraint.constraint());
  }

  for (std::size_t i = 0; i < problem.objectives.size(); ++i) {
    const auto& objective = problem.objectives[i];
    appendTerm(m_hqpData[objective.level], objective.weight,
               m_objectiveSlots[i].constraint);
  }

  auto& regLevel =
      m_hqpData[policy.regularizationLevel(problem.maxObjectiveLevel())];
  appendTerm(regLevel, problem.regularization.w_delta_qddot,
             m_accelerationRegularization.constraint());
  if (m_cycle.regularizeLambda) {
    appendTerm(regLevel, problem.regularization.w_lambda,
               m_lambdaRegularization.constraint());
  }
}

void IDHQP::resizeSolverFromHQPData() {
  unsigned int n = 0;
  unsigned int neq = 0;
  unsigned int nin = 0;

  for (const auto& level : m_hqpData) {
    for (const auto& pair : level) {
      if (n == 0 && pair.second) {
        n = pair.second->cols();
      }
      if (!pair.second) {
        continue;
      }
      if (pair.second->isEquality()) {
        neq += pair.second->rows();
      } else {
        nin += pair.second->rows();
      }
    }
  }

  if (n > 0) {
    if (!m_solver || m_solverVarDim != n || m_solverEqDim != neq ||
        m_solverInDim != nin) {
      m_solver = makeSolver(m_solverType, m_qpParams);
      m_solverVarDim = n;
      m_solverEqDim = neq;
      m_solverInDim = nin;
    }
    m_solver->resize(n, neq, nin);
  }
}

void IDHQP::stackContactData(const IDProblem& problem) {
  m_cycle.contactLayout.clear();
  m_cycle.lambdaDim = 0;

  int totalLambdaDim = 0;
  int totalUfRows = 0;
  int totalMotionDim = 0;
  for (const auto& contact : problem.contacts) {
    totalLambdaDim += contact.lambdaDim();
    totalUfRows += static_cast<int>(contact.Uf.rows());
    totalMotionDim += contact.motionDim();
  }

  m_cycle.Jc.setZero(totalMotionDim, m_nv);
  m_cycle.Jcdot_qdot.setZero(totalMotionDim);
  m_cycle.Uf.setZero(totalUfRows, totalLambdaDim);
  m_cycle.uf_lb.setZero(totalUfRows);
  m_cycle.uf_ub.setZero(totalUfRows);

  int motionRow = 0;
  int lambdaOffset = 0;
  int ufRow = 0;

  for (const auto& contact : problem.contacts) {
    const int motionDim = contact.motionDim();
    const int lambdaDim = contact.lambdaDim();

    assert(contact.Jc.cols() == m_nv);
    assert(contact.Jcdot_qdot.size() == motionDim);
    assert(contact.T.cols() == lambdaDim);
    if (motionDim > 0) {
      m_cycle.Jc.block(motionRow, 0, motionDim, m_nv) = contact.Jc;
      m_cycle.Jcdot_qdot.segment(motionRow, motionDim) = contact.Jcdot_qdot;
    }

    if (lambdaDim > 0) {
      assert(contact.Uf.cols() == lambdaDim);
      assert(contact.uf_lb.size() == contact.Uf.rows());
      assert(contact.uf_ub.size() == contact.Uf.rows());
      m_cycle.Uf.block(ufRow, lambdaOffset, contact.Uf.rows(), lambdaDim) =
          contact.Uf;
      m_cycle.uf_lb.segment(ufRow, contact.Uf.rows()) = contact.uf_lb;
      m_cycle.uf_ub.segment(ufRow, contact.Uf.rows()) = contact.uf_ub;
    }

    m_cycle.contactLayout[contact.name] = {lambdaOffset, lambdaDim};
    motionRow += motionDim;
    lambdaOffset += lambdaDim;
    ufRow += static_cast<int>(contact.Uf.rows());
  }

  m_cycle.lambdaDim = totalLambdaDim;
}

void IDHQP::buildContext(const IDProblem& problem) {
  m_ctx = HQPBuildContext{};
  m_ctx.M = &m_robot.mass(m_data);
  m_ctx.h = &m_robot.nonLinearEffects(m_data);
  m_ctx.nv = m_nv;
  m_ctx.na = m_na;
  m_ctx.nvFloat = m_nvFloat;
  m_ctx.lambdaDim = m_cycle.lambdaDim;

  m_ctx.contactInfos.reserve(problem.contacts.size());
  for (const auto& contact : problem.contacts) {
    auto it = m_cycle.contactLayout.find(contact.name);
    assert(it != m_cycle.contactLayout.end());
    m_ctx.contactInfos.push_back({&contact.Jc, &contact.T,
                                  it->second.lambdaOffset,
                                  it->second.lambdaDim});
  }

  if (m_cycle.hasContactKinematics) {
    m_ctx.Jc = &m_cycle.Jc;
    m_ctx.Jcdot_qdot = &m_cycle.Jcdot_qdot;
  }
  if (m_cycle.hasContactForces) {
    m_ctx.Uf = &m_cycle.Uf;
    m_ctx.uf_lb = &m_cycle.uf_lb;
    m_ctx.uf_ub = &m_cycle.uf_ub;
  }

  m_ctx.h_ext = m_h_ext ? m_h_ext : &m_zero_h_ext;

  if (m_cycle.hasTorqueLimits) {
    m_ctx.enableTorqueLimits = true;
    m_ctx.tau_lb = m_torqueLimits.lower;
    m_ctx.tau_ub = m_torqueLimits.upper;
  }

  m_ctx.qddot_ref = &m_qddotRefCurrent;
}

const IDSolution& IDHQP::decodeSolution(const IDProblem& problem,
                                        const solvers::HQPOutput& hqpSol,
                                        double dt) {
  if (hqpSol.status != solvers::HQP_STATUS_OPTIMAL ||
      hqpSol.x.size() < m_nv + m_cycle.lambdaDim) {
    resetFailedSolution();
    return fail();
  }

  m_solution->delta_qddot = hqpSol.x.head(m_nv);
  m_solution->qddot_sol = m_qddotRefCurrent + m_solution->delta_qddot;
  integrateSolutionState(dt);

  if (m_cycle.hasContactForces) {
    m_solution->lambda_sol = hqpSol.x.segment(m_nv, m_cycle.lambdaDim);
  } else {
    m_solution->lambda_sol.resize(0);
  }

  recoverTorque(problem);
  m_solution->success =
      m_solution->qddot_sol.allFinite() && m_solution->qdot_cmd.allFinite() &&
      m_solution->q_cmd.allFinite() && m_solution->tau_ff_cmd.allFinite() &&
      m_solution->tau_fb_cmd.allFinite() && m_solution->tau_cmd.allFinite() &&
      m_solution->lambda_sol.allFinite();
  return *m_solution;
}

void IDHQP::resetFailedSolution() {
  m_solution->qddot_ref = m_qddotRefCurrent;
  m_solution->delta_qddot.setZero(m_nv);
  m_solution->qddot_sol = m_qddotRefCurrent;
  if (m_robot.hasState()) {
    m_solution->q_cmd = m_robot.generalized_q();
    m_solution->qdot_cmd = m_robot.generalized_v();
  } else {
    m_solution->q_cmd = pinocchio::neutral(m_robot.model());
    m_solution->qdot_cmd.setZero(m_nv);
  }
  if (m_cycle.lambdaDim > 0) {
    m_solution->lambda_sol.setZero(m_cycle.lambdaDim);
  } else {
    m_solution->lambda_sol.resize(0);
  }
  m_solution->tau_ff_cmd.setZero(m_na);
  m_solution->tau_fb_cmd.setZero(m_na);
  m_solution->tau_cmd.setZero(m_na);
  m_solution->success = false;
}

void IDHQP::integrateSolutionState(double dt) {
  assert(m_robot.hasState());
  assert(m_robot.generalized_q().size() == m_robot.nq());
  assert(m_robot.generalized_v().size() == m_nv);

  m_solution->qdot_cmd.noalias() =
      m_robot.generalized_v() + dt * m_solution->qddot_sol;
  m_integrateDelta.noalias() = dt * m_solution->qdot_cmd;
  pinocchio::integrate(m_robot.model(), m_robot.generalized_q(),
                       m_integrateDelta, m_solution->q_cmd);
}

void IDHQP::recoverTorque(const IDProblem& problem) {
  const Matrix& M = m_robot.mass(m_data);
  const Vector& h = m_robot.nonLinearEffects(m_data);

  m_tauFull.noalias() = M * m_solution->qddot_sol;
  m_tauFull += h;

  if (m_h_ext) {
    m_tauFull -= *m_h_ext;
  }

  if (m_cycle.hasContactForces) {
    for (const auto& contact : problem.contacts) {
      auto it = m_cycle.contactLayout.find(contact.name);
      if (it == m_cycle.contactLayout.end() || it->second.lambdaDim == 0) {
        continue;
      }

      const auto& block = it->second;
      m_tauFull.noalias() -=
          contact.Jc.transpose() * contact.T *
          m_solution->lambda_sol.segment(block.lambdaOffset, block.lambdaDim);
    }
  }

  m_solution->tau_ff_cmd = m_tauFull.tail(m_na);
  m_solution->tau_fb_cmd.setZero(m_na);
  m_solution->tau_cmd = m_solution->tau_ff_cmd + m_solution->tau_fb_cmd;
}

}  // namespace wbc
