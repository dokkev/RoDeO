//
// Copyright (c) 2026
//
// Final-form WBMC inverse-dynamics HQP core.
//

#include "wbc_core/controller/wbmc.hpp"

#include <algorithm>
#include <cassert>
#include <chrono>

#include "wbc_core/solvers/solver-HQP-factory.hpp"

namespace tsid {

using namespace math;

namespace {

void appendTerm(solvers::ConstraintLevel& level, double weight,
                const std::shared_ptr<ConstraintBase>& constraint) {
  level.emplace_back(weight, constraint);
}

}  // namespace

WBMC::WBMC(robots::RobotSystem& robot)
    : m_robot(robot),
      m_data(robot.model()),
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
  m_solution = std::make_unique<WBMCSolution>();
  m_solution->qddot_ref = Vector::Zero(m_nv);
  m_solution->delta_qddot = Vector::Zero(m_nv);
  m_solution->qddot_sol = Vector::Zero(m_nv);
  m_solution->lambda = Vector::Zero(0);
  m_solution->tau = Vector::Zero(m_na);

  m_torqueLimitConstraint.preallocate(m_na);

  m_solver = solvers::SolverHQPFactory::createNewSolver(
      solvers::SOLVER_HQP_CASCADE, "wbmc-id");
}

const WBMCSolution& WBMC::solve(const WBMCStepInput& input) {
  using Clock = std::chrono::high_resolution_clock;
  const auto t0 = Clock::now();
  beginCycle(input);
  if (!validateInput(input)) {
    return fail();
  }

  updateRobotModel(input);
  prepareCycleWorkspace(input);
  buildHardConstraints(input);
  buildRegularizationBlocks(input);
  buildObjectiveBlocks(input);
  assembleHierarchy(input);
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
  return decodeSolution(input, hqpSol);
}

void WBMC::ensureObjectiveCapacity(std::vector<ObjectiveSlot>& pool,
                                   std::size_t nObjectives) {
  if (pool.size() < nObjectives) {
    pool.resize(nObjectives);
  }
}

void WBMC::beginCycle(const WBMCStepInput& input) {
  m_solution->success = false;
  m_cycle = CycleWorkspace{};

  if (input.qddot_ref) {
    assert(input.qddot_ref->size() == m_nv);
    m_qddotRefCurrent = *input.qddot_ref;
  } else {
    m_qddotRefCurrent = m_zeroQddotRef;
  }

  m_solution->qddot_ref = m_qddotRefCurrent;
  resetFailedSolution();
}

bool WBMC::validateInput(const WBMCStepInput& input) const {
  return input.hasState() && input.hasValidHierarchy();
}

void WBMC::updateRobotModel(const WBMCStepInput& input) {
  assert(input.q->size() == m_robot.nq());
  assert(input.qdot->size() == m_nv);
  m_robot.update(m_data, *input.q, *input.qdot);
}

void WBMC::prepareCycleWorkspace(const WBMCStepInput& input) {
  stackContactData(input);

  m_tau_lb = input.tau_lb;
  m_tau_ub = input.tau_ub;
  m_h_ext = input.h_ext;

  m_cycle.hasContactForces = (m_cycle.lambdaDim > 0);
  m_cycle.hasContactKinematics =
      m_cycle.hasContactForces && (m_cycle.Jc.rows() > 0);
  m_cycle.hasFrictionConstraints =
      m_cycle.hasContactForces && (m_cycle.Uf.rows() > 0);
  m_cycle.hasTorqueLimits = (m_tau_lb != nullptr && m_tau_ub != nullptr);
  m_cycle.regularizeLambda =
      m_cycle.hasContactForces && (input.regularization.w_lambda > 0.0);

  resetFailedSolution();
  buildContext(input);
}

void WBMC::buildHardConstraints(const WBMCStepInput& input) {
  (void)input;
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

void WBMC::buildRegularizationBlocks(const WBMCStepInput& input) {
  m_accelerationRegularization.setWeight(input.regularization.w_delta_qddot);
  m_accelerationRegularization.build(m_ctx);
  if (m_cycle.regularizeLambda) {
    m_lambdaRegularization.setWeight(input.regularization.w_lambda);
    m_lambdaRegularization.build(m_ctx);
  }
}

void WBMC::buildObjectiveBlocks(const WBMCStepInput& input) {
  ensureObjectiveCapacity(m_objectiveSlots, input.objectives.size());

  for (std::size_t i = 0; i < input.objectives.size(); ++i) {
    const auto& objective = input.objectives[i];
    auto& slot = m_objectiveSlots[i];

    if (objective.isMotion()) {
      const auto& motion = objective.motion();
      const bool needsRebuild =
          !slot.motion || slot.motion->level() != objective.level ||
          slot.motion->name() != motion.name;
      if (needsRebuild) {
        slot.joint_bias.reset();
        auto taskUnit = std::make_unique<MotionTask>(
            motion.name, objective.level, objective.weight);
        slot.constraint = taskUnit->constraint();
        slot.motion = std::move(taskUnit);
      }

      slot.motion->setWeight(objective.weight);
      slot.motion->build(motion, m_ctx);
      slot.constraint = slot.motion->constraint();
      continue;
    }

    const auto& jointObjective = objective.jointAcceleration();
    const bool needsRebuild =
        !slot.joint_bias || slot.joint_bias->level() != objective.level ||
        slot.joint_bias->name() != jointObjective.name;
    if (needsRebuild) {
      slot.motion.reset();
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

void WBMC::assembleHierarchy(const WBMCStepInput& input) {
  const auto& policy = input.hierarchy;
  m_hqpData.clear();
  m_hqpData.resize(policy.numLevels(input.maxObjectiveLevel()));

  auto& physicsLevel = m_hqpData[policy.physicsLevel];
  if (m_nvFloat > 0) {
    appendTerm(physicsLevel, 1.0, m_dynamicsConstraint.constraint());
  }
  if (m_cycle.hasContactKinematics) {
    appendTerm(physicsLevel, 1.0,
               m_contactConsistencyConstraint.constraint());
  }
  if (m_cycle.hasFrictionConstraints) {
    appendTerm(physicsLevel, 1.0, m_frictionConeConstraint.constraint());
  }
  if (m_cycle.hasTorqueLimits) {
    appendTerm(physicsLevel, 1.0, m_torqueLimitConstraint.constraint());
  }

  for (std::size_t i = 0; i < input.objectives.size(); ++i) {
    const auto& objective = input.objectives[i];
    appendTerm(m_hqpData[objective.level], objective.weight,
               m_objectiveSlots[i].constraint);
  }

  auto& regLevel = m_hqpData[policy.regularizationLevel(
      input.maxObjectiveLevel())];
  appendTerm(regLevel, input.regularization.w_delta_qddot,
             m_accelerationRegularization.constraint());
  if (m_cycle.regularizeLambda) {
    appendTerm(regLevel, input.regularization.w_lambda,
               m_lambdaRegularization.constraint());
  }
}

void WBMC::resizeSolverFromHQPData() {
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
      m_solver = solvers::SolverHQPFactory::createNewSolver(
          solvers::SOLVER_HQP_CASCADE, "wbmc-id");
      m_solverVarDim = n;
      m_solverEqDim = neq;
      m_solverInDim = nin;
    }
    m_solver->resize(n, neq, nin);
  }
}

void WBMC::stackContactData(const WBMCStepInput& input) {
  m_cycle.contactLayout.clear();
  m_cycle.lambdaDim = 0;

  int totalLambdaDim = 0;
  int totalUfRows = 0;
  int totalMotionDim = 0;
  for (const auto& contact : input.contacts) {
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

  for (const auto& contact : input.contacts) {
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

void WBMC::buildContext(const WBMCStepInput& input) {
  m_ctx = HQPBuildContext{};
  m_ctx.M = &m_robot.mass(m_data);
  m_ctx.h = &m_robot.nonLinearEffects(m_data);
  m_ctx.nv = m_nv;
  m_ctx.na = m_na;
  m_ctx.nvFloat = m_nvFloat;
  m_ctx.lambdaDim = m_cycle.lambdaDim;

  m_ctx.contactInfos.reserve(input.contacts.size());
  for (const auto& contact : input.contacts) {
    auto it = m_cycle.contactLayout.find(contact.name);
    assert(it != m_cycle.contactLayout.end());
    m_ctx.contactInfos.push_back(
        {&contact.Jc, &contact.T, it->second.lambdaOffset, it->second.lambdaDim});
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
    m_ctx.tau_lb = m_tau_lb;
    m_ctx.tau_ub = m_tau_ub;
  }

  m_ctx.qddot_ref = &m_qddotRefCurrent;
}

const WBMCSolution& WBMC::decodeSolution(const WBMCStepInput& input,
                                         const solvers::HQPOutput& hqpSol) {
  if (hqpSol.status != solvers::HQP_STATUS_OPTIMAL ||
      hqpSol.x.size() < m_nv + m_cycle.lambdaDim) {
    resetFailedSolution();
    return fail();
  }

  m_solution->delta_qddot = hqpSol.x.head(m_nv);
  m_solution->qddot_sol = m_qddotRefCurrent + m_solution->delta_qddot;

  if (m_cycle.hasContactForces) {
    m_solution->lambda = hqpSol.x.segment(m_nv, m_cycle.lambdaDim);
  } else {
    m_solution->lambda.resize(0);
  }

  recoverTorque(input);
  m_solution->success = m_solution->qddot_sol.allFinite() &&
                        m_solution->tau.allFinite() &&
                        m_solution->lambda.allFinite();
  return *m_solution;
}

void WBMC::resetFailedSolution() {
  m_solution->qddot_ref = m_qddotRefCurrent;
  m_solution->delta_qddot.setZero(m_nv);
  m_solution->qddot_sol = m_qddotRefCurrent;
  if (m_cycle.lambdaDim > 0) {
    m_solution->lambda.setZero(m_cycle.lambdaDim);
  } else {
    m_solution->lambda.resize(0);
  }
  m_solution->tau.setZero(m_na);
  m_solution->success = false;
}

void WBMC::recoverTorque(const WBMCStepInput& input) {
  const Matrix& M = m_robot.mass(m_data);
  const Vector& h = m_robot.nonLinearEffects(m_data);

  m_tauFull.noalias() = M * m_solution->qddot_sol;
  m_tauFull += h;

  if (m_h_ext) {
    m_tauFull -= *m_h_ext;
  }

  if (m_cycle.hasContactForces) {
    for (const auto& contact : input.contacts) {
      auto it = m_cycle.contactLayout.find(contact.name);
      if (it == m_cycle.contactLayout.end() || it->second.lambdaDim == 0) {
        continue;
      }

      const auto& block = it->second;
      m_tauFull.noalias() -=
          contact.Jc.transpose() * contact.T *
          m_solution->lambda.segment(block.lambdaOffset, block.lambdaDim);
    }
  }

  m_solution->tau = m_tauFull.tail(m_na);
}

}  // namespace tsid
