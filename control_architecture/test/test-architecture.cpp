//
// Copyright (c) 2026
//
// Tests for wbc_core architecture layer:
//   - ConfigCompiler/ConfigValidator/RuntimeAssembler pipeline
//   - FSMHandler (state lifecycle and transitions)
//   - State base + injectable StateFactory
//   - ControlArchitecture (end-to-end solve)
//

#include <gtest/gtest.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <initializer_list>
#include <memory>
#include <ostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <pinocchio/algorithm/joint-configuration.hpp>

#include <wbc_core/robots/robot-system.hpp>
#include <wbc_core/tasks/task-joint-posture.hpp>
#include <wbc_core/trajectories/trajectory-base.hpp>

#include "control_architecture/control_architecture.hpp"
#include "control_architecture/runtime/config_compiler.hpp"
#include "control_architecture/runtime/config_loader.hpp"
#include "control_architecture/runtime/config_validator.hpp"
#include "control_architecture/runtime/runtime_assembler.hpp"
#include "control_architecture/runtime/state_machine_assembler.hpp"
#include "control_architecture/state_machine/fsm_handler.hpp"
#include "control_architecture/state_machine/state_machine.hpp"
#include "wbc_core/controller/id-hqp.hpp"
#include "wbc_core/controller/id-problem-registry.hpp"

using namespace wbc;
using namespace wbc::robots;
using namespace std;

namespace {

RuntimeConfig compileRuntimeConfig(const YAML::Node& root, RobotSystem& robot) {
  CompiledConfig compiled_config = ConfigCompiler::Compile(root);
  ConfigValidator::Validate(compiled_config);
  return RuntimeAssembler::Assemble(compiled_config, robot);
}

class TestCustomState final : public State {
 public:
  TestCustomState(StateId id, const std::string& name, const StateContext& ctx)
      : State(id, name, ctx) {}

  void OnEnter() override {}
  void OnUpdate() override {}
  void OnExit() override {}
};

class CountingState final : public State {
 public:
  CountingState(StateId id, const std::string& name, const StateContext& ctx,
                int* first_visit_count, int* one_step_count)
      : State(id, name, ctx),
        first_visit_count_(first_visit_count),
        one_step_count_(one_step_count) {}

  void OnEnter() override {
    if (first_visit_count_) ++(*first_visit_count_);
  }

  void OnUpdate() override {
    if (one_step_count_) ++(*one_step_count_);
  }

  void OnExit() override {}

 private:
  int* first_visit_count_{nullptr};
  int* one_step_count_{nullptr};
};

class YamlJointCommandState final : public State {
 public:
  YamlJointCommandState(StateId id, const std::string& name,
                        const StateContext& ctx)
      : State(id, name, ctx) {}

  void Configure(const YAML::Node& node) override {
    State::Configure(node);
    target_q_ = ReadVector(node, "target_jpos", robot_->nq_actuated());
    target_qdot_ = ReadVector(node, "target_jvel", robot_->na());
    target_qddot_ = ReadVector(node, "target_jacc", robot_->na());
  }

  void OnEnter() override { ApplyReference(); }
  void OnUpdate() override { ApplyReference(); }
  void OnExit() override {}

 private:
  static Eigen::VectorXd ReadVector(const YAML::Node& node,
                                    const std::string& key,
                                    Eigen::Index expected_size) {
    if (!node[key]) {
      return Eigen::VectorXd::Zero(expected_size);
    }

    const auto values = node[key].as<std::vector<double>>();
    if (static_cast<Eigen::Index>(values.size()) != expected_size) {
      throw std::invalid_argument("YamlJointCommandState: '" + key +
                                  "' has wrong size");
    }

    Eigen::VectorXd out(expected_size);
    for (Eigen::Index i = 0; i < expected_size; ++i) {
      out(i) = values[static_cast<std::size_t>(i)];
    }
    return out;
  }

  void ApplyReference() {
    auto task = std::dynamic_pointer_cast<wbc::tasks::TaskJointPosture>(
        Task("jpos_task"));
    if (!task) {
      throw std::runtime_error(
          "YamlJointCommandState: jpos_task is not assigned");
    }

    wbc::trajectories::TrajectorySample ref(robot_->nq_actuated(),
                                            robot_->na());
    ref.setValue(target_q_);
    ref.setDerivative(target_qdot_);
    ref.setSecondDerivative(target_qddot_);
    task->setReference(ref);
  }

  Eigen::VectorXd target_q_;
  Eigen::VectorXd target_qdot_;
  Eigen::VectorXd target_qddot_;
};

StateFactory MakeTestStateFactory(std::initializer_list<std::string> keys = {
                                      "initialize", "home",
                                      "cartesian_reference",
                                      "joint_reference"}) {
  StateFactory factory;
  for (const auto& key : keys) {
    factory.Register<TestCustomState>(key);
  }
  return factory;
}

void RegisterTestStates(ControlArchitecture& arch) {
  for (const auto& key :
       {"initialize", "home", "cartesian_reference", "joint_reference"}) {
    arch.RegisterState(
        key, [](StateId id, const std::string& name, const StateContext& ctx) {
          return std::make_unique<TestCustomState>(id, name, ctx);
        });
  }
}

void RegisterYamlJointCommandState(ControlArchitecture& arch) {
  arch.RegisterState(
      "yaml_joint_command",
      [](StateId id, const std::string& name, const StateContext& ctx) {
        return std::make_unique<YamlJointCommandState>(id, name, ctx);
      });
}

std::unique_ptr<ControlArchitecture> MakeTestArchitecture(
    const YAML::Node& root, std::shared_ptr<RobotSystem> robot) {
  auto config = compileRuntimeConfig(root, *robot);
  auto arch = std::make_unique<ControlArchitecture>(std::move(config),
                                                    std::move(robot));
  RegisterTestStates(*arch);
  return arch;
}

RobotState MakeZeroRobotState(const RobotSystem& robot) {
  RobotState state;
  state.q = pinocchio::neutral(robot.model());
  state.qdot = math::Vector::Zero(robot.nv());
  return state;
}

void WriteYamlVector(std::ostream& out, const std::string& key,
                     const Eigen::VectorXd& values, const std::string& indent) {
  out << indent << key << ": [";
  out << std::setprecision(17);
  for (Eigen::Index i = 0; i < values.size(); ++i) {
    if (i > 0) out << ", ";
    out << values(i);
  }
  out << "]\n";
}

template <typename Derived>
double InfNorm(const Eigen::MatrixBase<Derived>& v) {
  return v.template lpNorm<Eigen::Infinity>();
}

int TotalLambdaDim(const std::vector<ContactConstraintData>& contacts) {
  int lambda_dim = 0;
  for (const auto& contact : contacts) {
    lambda_dim += contact.lambdaDim();
  }
  return lambda_dim;
}

}  // namespace

// ─────────────────────────────────────────────────────────────────────────────
// Fixture: Romeo robot (reusable)
// ─────────────────────────────────────────────────────────────────────────────

class ArchitectureTest : public ::testing::Test {
 protected:
  shared_ptr<RobotSystem> robot;
  string model_dir;

  void SetUp() override {
    model_dir = TSID_MODEL_DIR;
    vector<string> package_dirs = {model_dir};
    const string urdfFile = model_dir + "/romeo/urdf/romeo.urdf";
    robot = make_shared<RobotSystem>(urdfFile, package_dirs, false);
  }

  /// Build a minimal YAML config node for testing.
  YAML::Node makeMinimalConfig() {
    YAML::Node root;

    // Task pool
    YAML::Node jposTask;
    jposTask["name"] = "jpos_task";
    jposTask["type"] = "JointTask";
    jposTask["role"] = "bias_task";
    jposTask["kp"] = 100.0;
    jposTask["kd"] = 10.0;
    jposTask["weight"] = 1.0;
    root["task_pool"].push_back(jposTask);

    // State machine
    YAML::Node initState;
    initState["id"] = 0;
    initState["name"] = "initialize";
    initState["duration"] = 1.0;
    initState["wait_time"] = 0.0;
    initState["next_state_id"] = 1;
    YAML::Node initTask;
    initTask["name"] = "jpos_task";
    initTask["weight"] = 1.0;
    initState["tasks"].push_back(initTask);
    root["state_machine"].push_back(initState);

    YAML::Node homeState;
    homeState["id"] = 1;
    homeState["name"] = "home";
    homeState["stay_here"] = true;
    homeState["duration"] = 1.0;
    YAML::Node homeTask;
    homeTask["name"] = "jpos_task";
    homeTask["weight"] = 1.0;
    homeState["tasks"].push_back(homeTask);
    root["state_machine"].push_back(homeState);

    return root;
  }

  /// Build a config with SE3 task for cartesian reference tracking.
  YAML::Node makeCartesianConfig() {
    YAML::Node root = makeMinimalConfig();

    // Add SE3 task.
    YAML::Node se3Task;
    se3Task["name"] = "ee_task";
    se3Task["type"] = "SE3Task";
    se3Task["role"] = "operational_task";
    se3Task["target_frame"] = "RWristPitch";
    se3Task["position"] = true;
    se3Task["orientation"] = true;
    se3Task["kp"] = 50.0;
    se3Task["kd"] = 10.0;
    se3Task["weight"] = 100.0;
    root["task_pool"].push_back(se3Task);

    // Add cartesian reference state.
    YAML::Node cartState;
    cartState["id"] = 2;
    cartState["name"] = "cartesian_reference";
    cartState["stay_here"] = true;
    YAML::Node cartJpos;
    cartJpos["name"] = "jpos_task";
    cartJpos["weight"] = 1.0;
    cartJpos["level"] = 3;
    YAML::Node cartEePos;
    cartEePos["name"] = "ee_task";
    cartEePos["weight"] = 100.0;
    cartState["tasks"].push_back(cartJpos);
    cartState["tasks"].push_back(cartEePos);
    root["state_machine"].push_back(cartState);

    return root;
  }
};

// ═══════════════════════════════════════════════════════════════════════════════
// ConfigCompiler
// ═══════════════════════════════════════════════════════════════════════════════

TEST_F(ArchitectureTest, ConfigCompiler_ParsesMinimalCompiledConfig) {
  auto root = makeMinimalConfig();
  auto compiled_config = ConfigCompiler::Compile(root);

  EXPECT_EQ(compiled_config.task_pool.size(), 1u);
  EXPECT_EQ(compiled_config.contact_pool.size(), 0u);
  EXPECT_EQ(compiled_config.states.size(), 2u);
  ASSERT_EQ(compiled_config.states[0].tasks.size(), 1u);
  EXPECT_EQ(compiled_config.states[0].tasks[0].name, "jpos_task");
  EXPECT_EQ(compiled_config.task_pool[0].name, "jpos_task");
  EXPECT_EQ(compiled_config.task_pool[0].level, 2u);
  EXPECT_EQ(compiled_config.task_pool[0].type, TaskTypeSpec::kJointTask);
}

TEST_F(ArchitectureTest, RuntimeAssembler_AssemblesCompiledConfig) {
  auto root = makeMinimalConfig();
  auto compiled_config = ConfigCompiler::Compile(root);
  auto config = RuntimeAssembler::Assemble(compiled_config, *robot);

  EXPECT_EQ(config.task_pool.size(), 1u);
  EXPECT_TRUE(config.task_pool.count("jpos_task"));
  EXPECT_EQ(config.states.size(), 2u);
  EXPECT_TRUE(config.states.count(0));
  EXPECT_EQ(config.start_state_id, 0);

  const auto task_it = config.task_pool.find("jpos_task");
  ASSERT_NE(task_it, config.task_pool.end());
  EXPECT_NE(task_it->second.task, nullptr);
  EXPECT_EQ(task_it->second.level, 2u);
  EXPECT_DOUBLE_EQ(task_it->second.weight, 1.0);
}

TEST_F(ArchitectureTest,
       RuntimeAssembler_BuildsSurfaceContactFromCompiledConfig) {
  auto root = makeMinimalConfig();
  YAML::Node contact;
  contact["name"] = "rfoot";
  contact["type"] = "SurfaceContact";
  contact["target_frame"] = "RAnkleRoll";
  contact["foot_half_length"] = 0.1;
  contact["foot_half_width"] = 0.05;
  root["contact_pool"].push_back(contact);
  root["state_machine"][0]["contacts"].push_back("rfoot");

  vector<string> pkgs = {model_dir};
  RobotSystem fb_robot(model_dir + "/romeo/urdf/romeo.urdf", pkgs,
                       pinocchio::JointModelFreeFlyer());

  auto compiled_config = ConfigCompiler::Compile(root);
  auto config = RuntimeAssembler::Assemble(compiled_config, fb_robot);

  ASSERT_EQ(config.contact_pool.size(), 1u);
  const auto it = config.contact_pool.find("rfoot");
  ASSERT_NE(it, config.contact_pool.end());
  EXPECT_NE(it->second.contact, nullptr);
}

TEST_F(ArchitectureTest,
       RuntimeAssembler_RejectsSurfaceContactWithoutFootHalfDimensions) {
  auto root = makeMinimalConfig();
  YAML::Node contact;
  contact["name"] = "rfoot";
  contact["type"] = "SurfaceContact";
  contact["target_frame"] = "RAnkleRoll";
  root["contact_pool"].push_back(contact);
  root["state_machine"][0]["contacts"].push_back("rfoot");

  vector<string> pkgs = {model_dir};
  RobotSystem fb_robot(model_dir + "/romeo/urdf/romeo.urdf", pkgs,
                       pinocchio::JointModelFreeFlyer());

  auto compiled_config = ConfigCompiler::Compile(root);
  EXPECT_THROW(RuntimeAssembler::Assemble(compiled_config, fb_robot),
               std::invalid_argument);
}

TEST_F(ArchitectureTest, ConfigValidator_RejectsStateUnknownTaskReference) {
  auto root = makeMinimalConfig();
  root["state_machine"][0]["tasks"][0]["name"] = "missing_task";
  const auto compiled_config = ConfigCompiler::Compile(root);

  EXPECT_THROW(ConfigValidator::Validate(compiled_config),
               std::invalid_argument);
}

TEST_F(ArchitectureTest, ConfigValidator_RejectsStateUnknownContactReference) {
  auto root = makeMinimalConfig();
  root["state_machine"][0]["contacts"].push_back("missing_contact");
  const auto compiled_config = ConfigCompiler::Compile(root);

  EXPECT_THROW(ConfigValidator::Validate(compiled_config),
               std::invalid_argument);
}

TEST_F(ArchitectureTest, ConfigValidator_RejectsDuplicateTaskPoolNames) {
  auto root = makeMinimalConfig();
  YAML::Node dupTask = root["task_pool"][0];
  root["task_pool"].push_back(dupTask);
  const auto compiled_config = ConfigCompiler::Compile(root);

  EXPECT_THROW(ConfigValidator::Validate(compiled_config),
               std::invalid_argument);
}

TEST_F(ArchitectureTest, ConfigValidator_RejectsDuplicateStateIds) {
  auto root = makeMinimalConfig();
  root["state_machine"][1]["id"] = root["state_machine"][0]["id"].as<int>();
  const auto compiled_config = ConfigCompiler::Compile(root);

  EXPECT_THROW(ConfigValidator::Validate(compiled_config),
               std::invalid_argument);
}

TEST_F(ArchitectureTest, ConfigValidator_RejectsDuplicateStateNames) {
  auto root = makeMinimalConfig();
  root["state_machine"][1]["name"] =
      root["state_machine"][0]["name"].as<std::string>();
  const auto compiled_config = ConfigCompiler::Compile(root);

  EXPECT_THROW(ConfigValidator::Validate(compiled_config),
               std::invalid_argument);
}

TEST_F(ArchitectureTest, ConfigValidator_RejectsInvalidStateLifecycle) {
  auto root = makeMinimalConfig();
  root["state_machine"][0]["duration"] = -1.0;
  auto compiled_config = ConfigCompiler::Compile(root);
  EXPECT_THROW(ConfigValidator::Validate(compiled_config),
               std::invalid_argument);

  root = makeMinimalConfig();
  root["state_machine"][0]["next_state_id"] = 42;
  compiled_config = ConfigCompiler::Compile(root);
  EXPECT_THROW(ConfigValidator::Validate(compiled_config),
               std::invalid_argument);
}

TEST_F(ArchitectureTest, ConfigValidator_RejectsInvalidControllerDt) {
  auto root = makeMinimalConfig();
  root["controller"]["dt"] = 0.0;
  const auto compiled_config = ConfigCompiler::Compile(root);

  EXPECT_THROW(ConfigValidator::Validate(compiled_config),
               std::invalid_argument);
}

TEST_F(ArchitectureTest, ConfigValidator_RejectsInvalidDebugPrintInterval) {
  auto root = makeMinimalConfig();
  root["debug"]["enabled"] = true;
  root["debug"]["print_interval"] = 0.0;
  const auto compiled_config = ConfigCompiler::Compile(root);

  EXPECT_THROW(ConfigValidator::Validate(compiled_config),
               std::invalid_argument);
}

TEST_F(ArchitectureTest, RuntimeAssembler_UnsupportedForceTaskThrows) {
  auto root = makeMinimalConfig();
  YAML::Node forceTask;
  forceTask["name"] = "force_task";
  forceTask["type"] = "ForceTask";
  forceTask["role"] = "operational_task";
  forceTask["weight"] = 1.0;
  root["task_pool"].push_back(forceTask);

  auto compiled_config = ConfigCompiler::Compile(root);
  EXPECT_THROW(RuntimeAssembler::Assemble(compiled_config, *robot),
               std::invalid_argument);
}

TEST_F(ArchitectureTest, ConfigPipeline_ParseMinimalRuntimeConfig) {
  auto root = makeMinimalConfig();
  auto config = compileRuntimeConfig(root, *robot);

  // Should have 1 task
  EXPECT_EQ(config.task_pool.size(), 1u);
  EXPECT_TRUE(config.task_pool.count("jpos_task"));

  // Should have 2 states
  EXPECT_EQ(config.states.size(), 2u);
  EXPECT_TRUE(config.states.count(0));
  EXPECT_TRUE(config.states.count(1));

  // Start state = 0
  EXPECT_EQ(config.start_state_id, 0);

  // State 0 should reference jpos_task
  EXPECT_EQ(config.states[0].task_names.size(), 1u);
  EXPECT_EQ(config.states[0].task_names[0], "jpos_task");
}

TEST_F(ArchitectureTest, ConfigCompiler_ParseRegularization) {
  auto root = makeMinimalConfig();
  root["controller"]["qddot_ref"] = false;
  root["solver"]["type"] = "SOLVER_HQP_PROXQP";
  root["solver"]["qp_params"]["max_iter"] = 250;
  root["solver"]["qp_params"]["rho"] = 1.0e-6;
  root["solver"]["qp_params"]["mu_eq"] = 1.0e-3;
  root["solver"]["qp_params"]["mu_ineq"] = 1.0e-1;
  root["solver"]["qp_params"]["eps_abs"] = 1.0e-5;
  root["solver"]["qp_params"]["eps_rel"] = 0.0;
  root["solver"]["qp_params"]["verbose"] = false;
  root["regularization"]["w_qddot"] = 0.05;
  root["regularization"]["w_lambda"] = 0.005;
  root["regularization"]["w_xc_ddot"] = 50.0;
  root["controller"]["kp_acc"] = 200.0;
  root["controller"]["kd_acc"] = 30.0;

  auto config = compileRuntimeConfig(root, *robot);

  EXPECT_DOUBLE_EQ(config.regularization.w_delta_qddot, 0.05);
  EXPECT_DOUBLE_EQ(config.regularization.w_lambda, 0.005);
  EXPECT_DOUBLE_EQ(config.contact_accel_weight, 50.0);
  EXPECT_DOUBLE_EQ(config.kp_acc, 200.0);
  EXPECT_DOUBLE_EQ(config.kd_acc, 30.0);
  EXPECT_FALSE(config.qddot_ref_enabled);
  EXPECT_EQ(config.solver_type, wbc::solvers::SOLVER_HQP_PROXQP);
  ASSERT_TRUE(config.solver_qp_params.max_iter.has_value());
  EXPECT_EQ(*config.solver_qp_params.max_iter, 250u);
  ASSERT_TRUE(config.solver_qp_params.rho.has_value());
  EXPECT_DOUBLE_EQ(*config.solver_qp_params.rho, 1.0e-6);
  ASSERT_TRUE(config.solver_qp_params.mu_eq.has_value());
  EXPECT_DOUBLE_EQ(*config.solver_qp_params.mu_eq, 1.0e-3);
  ASSERT_TRUE(config.solver_qp_params.mu_ineq.has_value());
  EXPECT_DOUBLE_EQ(*config.solver_qp_params.mu_ineq, 1.0e-1);
  ASSERT_TRUE(config.solver_qp_params.eps_abs.has_value());
  EXPECT_DOUBLE_EQ(*config.solver_qp_params.eps_abs, 1.0e-5);
  ASSERT_TRUE(config.solver_qp_params.eps_rel.has_value());
  EXPECT_DOUBLE_EQ(*config.solver_qp_params.eps_rel, 0.0);
  ASSERT_TRUE(config.solver_qp_params.verbose.has_value());
  EXPECT_FALSE(*config.solver_qp_params.verbose);
}

TEST_F(ArchitectureTest, ConfigCompiler_RejectsLegacyTaskHierarchyField) {
  auto root = makeMinimalConfig();

  root["state_machine"][0].remove("tasks");
  root["state_machine"][0]["task_hierarchy"].push_back("jpos_task");

  EXPECT_THROW(ConfigCompiler::Compile(root), std::invalid_argument);
}

TEST_F(ArchitectureTest, ConfigCompiler_RejectsLegacyContactConstraintsField) {
  auto root = makeMinimalConfig();

  root["state_machine"][0]["contact_constraints"].push_back("left_foot");

  EXPECT_THROW(ConfigCompiler::Compile(root), std::invalid_argument);
}

TEST_F(ArchitectureTest, ConfigCompiler_RejectsStateTypeField) {
  auto root = makeMinimalConfig();

  root["state_machine"][0]["type"] = "initialize";

  EXPECT_THROW(ConfigCompiler::Compile(root), std::invalid_argument);
}

TEST_F(ArchitectureTest, ConfigCompiler_StateTaskSelectionPreserved) {
  auto root = makeCartesianConfig();
  auto config = compileRuntimeConfig(root, *robot);

  // State 2 (cartesian_reference) should preserve selected tasks and weights.
  auto& sc = config.states[2];
  ASSERT_EQ(sc.task_names.size(), 2u);
  EXPECT_EQ(sc.task_names[0], "jpos_task");
  EXPECT_EQ(sc.task_names[1], "ee_task");
  ASSERT_EQ(sc.task_weights.size(), 2u);
  EXPECT_DOUBLE_EQ(sc.task_weights[0], 1.0);
  EXPECT_DOUBLE_EQ(sc.task_weights[1], 100.0);
  ASSERT_EQ(sc.task_levels.size(), 2u);
  EXPECT_EQ(sc.task_levels[0], 3);
  EXPECT_EQ(sc.task_levels[1], -1);
}

TEST_F(ArchitectureTest, ConfigCompiler_StateUnknownTaskReferenceThrows) {
  auto root = makeMinimalConfig();
  root["state_machine"][0]["tasks"][0]["name"] = "missing_task";

  EXPECT_THROW(compileRuntimeConfig(root, *robot), std::invalid_argument);
}

TEST_F(ArchitectureTest, ConfigCompiler_RejectsStateTaskSelectionWithoutName) {
  auto root = makeMinimalConfig();
  root["state_machine"][0]["tasks"][0].remove("name");

  EXPECT_THROW(ConfigCompiler::Compile(root), std::invalid_argument);
}

TEST_F(ArchitectureTest, ConfigCompiler_StateUnknownContactReferenceThrows) {
  auto root = makeMinimalConfig();
  root["state_machine"][0]["contacts"].push_back("missing_contact");

  EXPECT_THROW(compileRuntimeConfig(root, *robot), std::invalid_argument);
}

TEST_F(ArchitectureTest, ConfigCompiler_RejectsStateHierarchyOverride) {
  auto root = makeMinimalConfig();
  root["state_machine"][0]["solver_hierarchy"]["physics_level"] = 0;

  EXPECT_THROW(ConfigCompiler::Compile(root), std::invalid_argument);
}

TEST_F(ArchitectureTest, ConfigCompiler_RejectsTaskPrioritiesField) {
  auto root = makeMinimalConfig();
  root["state_machine"][0]["task_priorities"].push_back(0);

  EXPECT_THROW(ConfigCompiler::Compile(root), std::invalid_argument);
}

TEST_F(ArchitectureTest, ConfigCompiler_RejectsRemovedPostureTaskRole) {
  auto root = makeMinimalConfig();
  root["task_pool"][0]["role"] = "posture_task";

  EXPECT_THROW(ConfigCompiler::Compile(root), std::invalid_argument);
}

TEST_F(ArchitectureTest, ConfigCompiler_RejectsUnknownConstraintType) {
  auto root = makeMinimalConfig();
  root["constraints"]["BadConstraint"]["enabled"] = true;

  EXPECT_THROW(ConfigCompiler::Compile(root), std::invalid_argument);
}

TEST_F(ArchitectureTest, ConfigCompiler_ParsesJointTorqueConstraint) {
  auto root = makeMinimalConfig();
  root["constraints"]["JointTorque"]["enabled"] = true;
  root["constraints"]["JointTorque"]["scale"] = 0.8;

  const auto config = compileRuntimeConfig(root, *robot);

  ASSERT_EQ(config.constraints.size(), 1u);
  EXPECT_EQ(config.constraints[0].type, ConstraintTypeSpec::kJointTorque);
  EXPECT_TRUE(config.constraints[0].enabled);
  EXPECT_DOUBLE_EQ(config.constraints[0].scale, 0.8);
}

TEST_F(ArchitectureTest, ConfigCompiler_RejectsLegacyGlobalConstraintsField) {
  auto root = makeMinimalConfig();
  root["global_constraints"]["JointTorque"]["enabled"] = true;

  EXPECT_THROW(ConfigCompiler::Compile(root), std::invalid_argument);
}

TEST_F(ArchitectureTest, StateMachineAssembler_RejectsUnknownStateName) {
  auto root = makeMinimalConfig();
  root["state_machine"][0]["name"] = "not_a_real_state";
  auto config = compileRuntimeConfig(root, *robot);

  wbc::IDHQP solver(*robot);
  auto q0 = pinocchio::neutral(robot->model());
  auto v0 = math::Vector::Zero(robot->nv());
  robot->computeAllTerms(solver.data(), q0, v0);

  FSMHandler fsm;

  EXPECT_THROW(StateMachineAssembler::Assemble(
                   config, fsm, *robot, solver.data(), MakeTestStateFactory()),
               std::invalid_argument);
}

TEST_F(ArchitectureTest, StateMachineAssembler_UsesInjectedStateFactory) {
  auto root = makeMinimalConfig();
  root["state_machine"][0]["name"] = "custom_state";
  auto config = compileRuntimeConfig(root, *robot);

  StateFactory factory = MakeTestStateFactory({"custom_state", "home"});

  wbc::IDHQP solver(*robot);
  auto q0 = pinocchio::neutral(robot->model());
  auto v0 = math::Vector::Zero(robot->nv());
  robot->computeAllTerms(solver.data(), q0, v0);

  FSMHandler fsm;
  StateMachineAssembler::Assemble(config, fsm, *robot, solver.data(), factory);

  const auto custom_id = fsm.FindStateIdByName("custom_state");
  ASSERT_TRUE(custom_id.has_value());
  EXPECT_EQ(*custom_id, 0);
}

TEST_F(ArchitectureTest, ConfigCompiler_RejectsStateImplementationField) {
  auto root = makeMinimalConfig();
  root["state_machine"][0]["implementation"] = "joint_posture_impl";

  EXPECT_THROW(ConfigCompiler::Compile(root), std::invalid_argument);
}

TEST_F(ArchitectureTest, ConfigLoader_LoadFileAndResolve_MergesExternalYaml) {
  namespace fs = std::filesystem;
  const auto stamp = std::to_string(
      std::chrono::steady_clock::now().time_since_epoch().count());
  const fs::path temp_dir =
      fs::temp_directory_path() / ("wbc_core_config_loader_test_" + stamp);
  fs::create_directories(temp_dir);

  const fs::path root_path = temp_dir / "root.yaml";
  const fs::path task_pool_path = temp_dir / "task_pool.yaml";
  const fs::path state_machine_path = temp_dir / "state_machine.yaml";

  {
    std::ofstream out(task_pool_path);
    out << "task_pool:\n"
           "  - name: ext_task\n"
           "    type: JointTask\n"
           "    role: bias_task\n"
           "    kp: 100.0\n"
           "    kd: 10.0\n"
           "contact_pool:\n"
           "  - name: ext_contact\n"
           "    type: PointContact\n"
           "    target_frame: RWristPitch\n";
  }
  {
    std::ofstream out(state_machine_path);
    out << "state_machine:\n"
           "  - id: 7\n"
           "    name: initialize\n"
           "    tasks:\n"
           "      - name: ext_task\n";
  }
  {
    std::ofstream out(root_path);
    out << "task_pool_yaml: task_pool.yaml\n"
           "state_machine_yaml: state_machine.yaml\n"
           "task_pool:\n"
           "  - name: inline_task\n"
           "    type: JointTask\n"
           "    role: bias_task\n"
           "state_machine:\n"
           "  - id: 3\n"
           "    name: home\n";
  }

  const YAML::Node resolved =
      ConfigLoader::LoadFileAndResolve(root_path.string());

  ASSERT_TRUE(resolved["task_pool"]);
  ASSERT_EQ(resolved["task_pool"].size(), 2u);
  EXPECT_EQ(resolved["task_pool"][0]["name"].as<std::string>(), "inline_task");
  EXPECT_EQ(resolved["task_pool"][1]["name"].as<std::string>(), "ext_task");

  ASSERT_TRUE(resolved["contact_pool"]);
  ASSERT_EQ(resolved["contact_pool"].size(), 1u);
  EXPECT_EQ(resolved["contact_pool"][0]["name"].as<std::string>(),
            "ext_contact");

  ASSERT_TRUE(resolved["state_machine"]);
  ASSERT_EQ(resolved["state_machine"].size(), 2u);
  EXPECT_EQ(resolved["state_machine"][0]["id"].as<int>(), 3);
  EXPECT_EQ(resolved["state_machine"][1]["id"].as<int>(), 7);

  fs::remove_all(temp_dir);
}

TEST_F(ArchitectureTest,
       RuntimeAssembler_RegistryBoundary_UsesStateTaskOverrides) {
  auto root = makeMinimalConfig();
  root["task_pool"][0]["role"] = "operational_task";
  root["state_machine"][0]["tasks"][0]["weight"] = 3.5;
  root["state_machine"][0]["tasks"][0]["level"] = 3;

  auto config = compileRuntimeConfig(root, *robot);

  wbc::IDProblemRegistry registry(*robot);
  wbc::IDHQP solver(*robot);
  auto q0 = pinocchio::neutral(robot->model());
  auto v0 = math::Vector::Zero(robot->nv());
  robot->computeAllTerms(solver.data(), q0, v0);

  BindRegistry(config, registry, *robot, solver.data());

  const auto& state_cfg = config.states.at(0);
  const auto problem = registry.buildProblem(
      0.0, q0, v0, state_cfg.task_names, state_cfg.task_weights,
      state_cfg.task_levels, state_cfg.contact_names);

  ASSERT_EQ(problem.objectives.size(), 1u);
  ASSERT_TRUE(problem.objectives[0].isMotionConstraint());
  EXPECT_EQ(problem.objectives[0].motionConstraint().name, "jpos_task");
  EXPECT_DOUBLE_EQ(problem.objectives[0].weight, 3.5);
  EXPECT_EQ(problem.objectives[0].level, 3u);
}

TEST_F(ArchitectureTest, BindRegistry_BindsRuntimePolicyAndTorqueLimits) {
  auto root = makeMinimalConfig();
  root["controller"]["qddot_ref"] = false;
  root["regularization"]["w_delta_qddot"] = 0.07;
  root["regularization"]["w_lambda"] = 0.003;
  root["constraints"]["JointTorque"]["enabled"] = true;
  root["constraints"]["JointTorque"]["scale"] = 0.5;

  auto config = compileRuntimeConfig(root, *robot);

  wbc::IDProblemRegistry registry(*robot);
  wbc::IDHQP solver(*robot);
  auto q0 = pinocchio::neutral(robot->model());
  auto v0 = math::Vector::Zero(robot->nv());
  robot->computeAllTerms(solver.data(), q0, v0);

  BindRegistry(config, registry, *robot, solver.data());

  EXPECT_FALSE(registry.referenceAccelerationEnabled());

  const auto& state_cfg = config.states.at(0);
  const auto problem = registry.buildProblem(
      0.0, q0, v0, state_cfg.task_names, state_cfg.task_weights,
      state_cfg.task_levels, state_cfg.contact_names);

  EXPECT_DOUBLE_EQ(problem.regularization.w_delta_qddot, 0.07);
  EXPECT_DOUBLE_EQ(problem.regularization.w_lambda, 0.003);
  ASSERT_TRUE(problem.torque_limits.enabled());
  ASSERT_NE(problem.torque_limits.lower, nullptr);
  ASSERT_NE(problem.torque_limits.upper, nullptr);

  const auto expected_upper =
      0.5 * robot->model().effortLimit.tail(robot->na());
  EXPECT_TRUE(problem.torque_limits.upper->isApprox(expected_upper));
  EXPECT_TRUE(problem.torque_limits.lower->isApprox(-expected_upper));
}

TEST_F(ArchitectureTest, ConfigCompiler_IgnoresJointVelLimitConstraint) {
  auto root = makeMinimalConfig();
  root["constraints"]["JointVelLimitConstraint"]["enabled"] = true;
  const auto compiled_config = ConfigCompiler::Compile(root);
  EXPECT_TRUE(compiled_config.constraints.empty());
}

TEST_F(ArchitectureTest, ConfigCompiler_IgnoresJointPosLimitConstraint) {
  auto root = makeMinimalConfig();
  root["constraints"]["JointPosLimitConstraint"]["enabled"] = true;
  const auto compiled_config = ConfigCompiler::Compile(root);
  EXPECT_TRUE(compiled_config.constraints.empty());
}

// ═══════════════════════════════════════════════════════════════════════════════
// FSMHandler
// ═══════════════════════════════════════════════════════════════════════════════

TEST_F(ArchitectureTest, FSMHandler_RegisterAndStart) {
  FSMHandler fsm;
  StateContext ctx;
  ctx.robot = robot.get();

  // Create dummy states
  auto s0 = make_unique<TestCustomState>(0, "init", ctx);
  auto s1 = make_unique<TestCustomState>(1, "home", ctx);
  fsm.RegisterState(0, move(s0));
  fsm.RegisterState(1, move(s1));

  EXPECT_TRUE(fsm.SetStartState(0));
  EXPECT_EQ(fsm.GetCurrentStateId(), 0);
}

TEST_F(ArchitectureTest, FSMHandler_RejectsInvalidRegistrations) {
  FSMHandler fsm;
  StateContext ctx;
  ctx.robot = robot.get();

  EXPECT_THROW(fsm.RegisterState(0, nullptr), std::invalid_argument);
  EXPECT_THROW(
      fsm.RegisterState(0, make_unique<TestCustomState>(1, "wrong_id", ctx)),
      std::invalid_argument);

  fsm.RegisterState(0, make_unique<TestCustomState>(0, "init", ctx));
  EXPECT_THROW(
      fsm.RegisterState(0, make_unique<TestCustomState>(0, "duplicate", ctx)),
      std::invalid_argument);
}

TEST_F(ArchitectureTest, FSMHandler_StateTransition) {
  FSMHandler fsm;
  StateContext ctx;
  ctx.robot = robot.get();

  auto s0 = make_unique<TestCustomState>(0, "init", ctx);
  auto s1 = make_unique<TestCustomState>(1, "home", ctx);
  fsm.RegisterState(0, move(s0));
  fsm.RegisterState(1, move(s1));
  fsm.SetStartState(0);

  // First update triggers OnEnter
  fsm.Update(0.0);
  EXPECT_EQ(fsm.GetCurrentStateId(), 0);

  // Request transition
  EXPECT_TRUE(fsm.RequestState(1));
  fsm.Update(0.001);
  EXPECT_EQ(fsm.GetCurrentStateId(), 1);
}

TEST_F(ArchitectureTest, FSMHandler_AutoTransition) {
  FSMHandler fsm;
  StateContext ctx;
  ctx.robot = robot.get();

  int s0_first_visits = 0;
  int s0_steps = 0;
  int s1_first_visits = 0;
  int s1_steps = 0;

  auto s0 =
      make_unique<CountingState>(0, "init", ctx, &s0_first_visits, &s0_steps);
  StateLifecycle lifecycle;
  lifecycle.duration = 0.001;
  lifecycle.wait_time = 0.0;
  lifecycle.next_state_id = 1;
  s0->ConfigureLifecycle(lifecycle);

  auto s1 =
      make_unique<CountingState>(1, "home", ctx, &s1_first_visits, &s1_steps);
  StateLifecycle stay_here;
  stay_here.stay_here = true;
  s1->ConfigureLifecycle(stay_here);

  fsm.RegisterState(0, move(s0));
  fsm.RegisterState(1, move(s1));
  fsm.SetStartState(0);

  // Tick past duration
  fsm.Update(0.0);  // OnEnter at t=0
  EXPECT_EQ(fsm.GetCurrentStateId(), 0);
  EXPECT_EQ(s0_first_visits, 1);
  EXPECT_EQ(s0_steps, 1);
  EXPECT_EQ(s1_first_visits, 0);
  EXPECT_EQ(s1_steps, 0);

  fsm.Update(0.01);  // elapsed > duration → auto-transition
  EXPECT_EQ(fsm.GetCurrentStateId(), 1);
  EXPECT_EQ(s0_steps, 1);
  EXPECT_EQ(s1_first_visits, 1);
  EXPECT_EQ(s1_steps, 1);
}

TEST_F(ArchitectureTest, FSMHandler_FindByName) {
  FSMHandler fsm;
  StateContext ctx;
  ctx.robot = robot.get();

  fsm.RegisterState(0, make_unique<TestCustomState>(0, "init", ctx));
  fsm.RegisterState(1, make_unique<TestCustomState>(1, "home", ctx));

  auto id = fsm.FindStateIdByName("home");
  ASSERT_TRUE(id.has_value());
  EXPECT_EQ(*id, 1);

  auto missing = fsm.FindStateIdByName("nonexistent");
  EXPECT_FALSE(missing.has_value());
}

TEST_F(ArchitectureTest, FSMHandler_RequestInvalidState) {
  FSMHandler fsm;
  EXPECT_FALSE(fsm.RequestState(99));
}

// ═══════════════════════════════════════════════════════════════════════════════
// State base
// ═══════════════════════════════════════════════════════════════════════════════

TEST_F(ArchitectureTest, State_ConfiguresTimeoutAndNextState) {
  StateContext ctx;
  ctx.robot = robot.get();

  TestCustomState state(0, "test", ctx);

  StateLifecycle lifecycle;
  lifecycle.duration = 1.0;
  lifecycle.wait_time = 0.1;
  lifecycle.next_state_id = 2;
  lifecycle.stay_here = false;
  state.ConfigureLifecycle(lifecycle);

  state.Enter(0.0);
  state.UpdateTime(0.5);
  EXPECT_FALSE(state.IsFinished());

  state.UpdateTime(1.2);
  EXPECT_TRUE(state.IsFinished());
  EXPECT_EQ(state.NextState(), 2);
}

// ═══════════════════════════════════════════════════════════════════════════════
// ConfigCompiler + FSM Integration
// ═══════════════════════════════════════════════════════════════════════════════

TEST_F(ArchitectureTest, StateMachineAssembler_AssemblesFsm) {
  auto root = makeMinimalConfig();
  auto config = compileRuntimeConfig(root, *robot);

  wbc::IDHQP solver(*robot);
  auto q0 = pinocchio::neutral(robot->model());
  auto v0 = math::Vector::Zero(robot->nv());
  robot->computeAllTerms(solver.data(), q0, v0);

  FSMHandler fsm;

  StateMachineAssembler::Assemble(config, fsm, *robot, solver.data(),
                                  MakeTestStateFactory());

  // Should have 2 states registered
  EXPECT_EQ(fsm.states().size(), 2u);

  // Start state should be 0
  EXPECT_EQ(fsm.GetCurrentStateId(), 0);

  // Should be able to update
  fsm.Update(0.0);
  EXPECT_EQ(fsm.GetCurrentStateId(), 0);
}

TEST_F(ArchitectureTest, StateMachineAssembler_AssignsConfiguredTaskHandles) {
  auto root = makeCartesianConfig();
  auto config = compileRuntimeConfig(root, *robot);

  wbc::IDHQP solver(*robot);
  auto q0 = pinocchio::neutral(robot->model());
  auto v0 = math::Vector::Zero(robot->nv());
  robot->computeAllTerms(solver.data(), q0, v0);

  FSMHandler fsm;
  StateMachineAssembler::Assemble(config, fsm, *robot, solver.data(),
                                  MakeTestStateFactory());

  const auto& states = fsm.states();
  ASSERT_TRUE(states.count(2));
  const auto* cartesian_state = states.at(2).get();
  ASSERT_NE(cartesian_state, nullptr);

  EXPECT_NE(cartesian_state->Task("jpos_task"), nullptr);
  EXPECT_NE(cartesian_state->Task("ee_task"), nullptr);
  EXPECT_EQ(cartesian_state->Task("missing_task"), nullptr);
}

TEST_F(ArchitectureTest, StateMachineAssembler_FsmRunsMultipleTicks) {
  auto root = makeMinimalConfig();
  auto config = compileRuntimeConfig(root, *robot);

  wbc::IDHQP solver(*robot);
  auto q0 = pinocchio::neutral(robot->model());
  auto v0 = math::Vector::Zero(robot->nv());
  robot->computeAllTerms(solver.data(), q0, v0);

  FSMHandler fsm;

  StateMachineAssembler::Assemble(config, fsm, *robot, solver.data(),
                                  MakeTestStateFactory());

  // Run 100 ticks
  for (int i = 0; i < 100; ++i) {
    fsm.Update(i * 0.001);
  }

  // Should have auto-transitioned to state 1 after duration (1.0s)
  // But at 100*0.001 = 0.1s, still in state 0
  EXPECT_EQ(fsm.GetCurrentStateId(), 0);

  // Run past duration
  for (int i = 100; i < 1500; ++i) {
    fsm.Update(i * 0.001);
  }
  // At t=1.5s, should have transitioned to state 1
  EXPECT_EQ(fsm.GetCurrentStateId(), 1);
}

// ═══════════════════════════════════════════════════════════════════════════════
// ControlArchitecture: end-to-end
// ═══════════════════════════════════════════════════════════════════════════════

TEST_F(ArchitectureTest, ControlArchitecture_EndToEnd) {
  auto root = makeMinimalConfig();
  auto arch = MakeTestArchitecture(root, robot);

  arch->Initialize();

  // Check internals
  EXPECT_NE(arch->solver(), nullptr);
  EXPECT_NE(arch->registry(), nullptr);
  EXPECT_NE(arch->fsmHandler(), nullptr);

  // Run a few ticks
  auto state = MakeZeroRobotState(*robot);

  for (int i = 0; i < 10; ++i) {
    state.time = i * 0.001;
    arch->Update(state, 0.001);
    EXPECT_DOUBLE_EQ(robot->time(), i * 0.001);
    const auto& cmd = arch->command();
    EXPECT_TRUE(cmd.tau.allFinite()) << "NaN tau at tick " << i;
    EXPECT_EQ(cmd.tau.size(), robot->na());
  }
}

TEST_F(ArchitectureTest, YamlFileToIDHQPDataFlow_SolvesAndProducesCommand) {
  namespace fs = std::filesystem;
  const auto stamp = std::to_string(
      std::chrono::steady_clock::now().time_since_epoch().count());
  const fs::path temp_dir =
      fs::temp_directory_path() / ("wbc_core_yaml_to_idhqp_test_" + stamp);
  fs::create_directories(temp_dir);

  const fs::path root_path = temp_dir / "wbc.yaml";
  const fs::path task_pool_path = temp_dir / "task_pool.yaml";
  const fs::path state_machine_path = temp_dir / "state_machine.yaml";

  {
    std::ofstream out(task_pool_path);
    out << "task_pool:\n"
           "  - name: jpos_task\n"
           "    type: JointTask\n"
           "    role: bias_task\n"
           "    kp: 25.0\n"
           "    kd: 5.0\n"
           "    weight: 1.0\n";
  }
  {
    std::ofstream out(state_machine_path);
    out << "state_machine:\n"
           "  - id: 0\n"
           "    name: initialize\n"
           "    stay_here: true\n"
           "    tasks:\n"
           "      - name: jpos_task\n"
           "        weight: 2.0\n"
           "        level: 2\n";
  }
  {
    std::ofstream out(root_path);
    out << "task_pool_yaml: task_pool.yaml\n"
           "state_machine_yaml: state_machine.yaml\n"
           "controller:\n"
           "  dt: 0.001\n"
           "  qddot_ref: false\n"
           "solver:\n"
           "  type: SOLVER_HQP_PROXQP\n"
           "  qp_params:\n"
           "    verbose: false\n"
           "    max_iter: 100\n"
           "regularization:\n"
           "  w_delta_qddot: 1.0e-3\n"
           "  w_lambda: 1.0e-5\n"
           "constraints:\n"
           "  JointTorque:\n"
           "    enabled: true\n"
           "    scale: 1.0\n";
  }

  const YAML::Node loaded =
      ConfigLoader::LoadFileAndResolve(root_path.string());
  auto config = compileRuntimeConfig(loaded, *robot);
  auto arch = std::make_unique<ControlArchitecture>(std::move(config), robot);
  RegisterTestStates(*arch);

  auto state = MakeZeroRobotState(*robot);
  state.time = 0.0;
  arch->Update(state, 0.001);

  EXPECT_EQ(arch->fsmHandler()->GetCurrentStateId(), 0);
  EXPECT_EQ(arch->solver()->solverType(), wbc::solvers::SOLVER_HQP_PROXQP);
  EXPECT_FALSE(arch->registry()->referenceAccelerationEnabled());

  const auto& sol = arch->solver()->solution();
  ASSERT_TRUE(sol.success);
  ASSERT_EQ(sol.qddot_ref.size(), robot->nv());
  ASSERT_EQ(sol.delta_qddot.size(), robot->nv());
  ASSERT_EQ(sol.qddot_sol.size(), robot->nv());
  ASSERT_EQ(sol.tau_cmd.size(), robot->na());
  EXPECT_TRUE(sol.qddot_ref.isZero(1e-12));
  EXPECT_NEAR((sol.delta_qddot - sol.qddot_sol).norm(), 0.0, 1e-9);
  EXPECT_TRUE(sol.tau_cmd.allFinite());

  const auto& cmd = arch->command();
  EXPECT_EQ(cmd.tau.size(), robot->na());
  EXPECT_EQ(cmd.q.size(), robot->na());
  EXPECT_EQ(cmd.qdot.size(), robot->na());
  EXPECT_TRUE(cmd.tau.allFinite());
  EXPECT_TRUE(cmd.q.allFinite());
  EXPECT_TRUE(cmd.qdot.allFinite());

  fs::remove_all(temp_dir);
}

TEST_F(ArchitectureTest, YamlJointCommand_TracksConfiguredJointReference) {
  namespace fs = std::filesystem;
  const auto stamp = std::to_string(
      std::chrono::steady_clock::now().time_since_epoch().count());
  const fs::path temp_dir =
      fs::temp_directory_path() / ("wbc_core_yaml_tracking_test_" + stamp);
  fs::create_directories(temp_dir);

  const fs::path root_path = temp_dir / "wbc_tracking.yaml";

  Eigen::VectorXd target = Eigen::VectorXd::Zero(robot->nq_actuated());
  target(0) = 0.05;
  if (target.size() > 3) {
    target(3) = -0.025;
  }

  {
    std::ofstream out(root_path);
    out << "controller:\n"
           "  dt: 0.001\n"
           "  qddot_ref: false\n"
           "solver:\n"
           "  type: SOLVER_HQP_PROXQP\n"
           "  qp_params:\n"
           "    verbose: false\n"
           "    max_iter: 100\n"
           "    eps_abs: 1.0e-8\n"
           "    eps_rel: 0.0\n"
           "regularization:\n"
           "  w_delta_qddot: 1.0e-5\n"
           "task_pool:\n"
           "  - name: jpos_task\n"
           "    type: JointTask\n"
           "    role: bias_task\n"
           "    kp: 40.0\n"
           "    kd: 8.0\n"
           "    weight: 1.0\n"
           "state_machine:\n"
           "  - id: 0\n"
           "    name: yaml_joint_command\n"
           "    stay_here: true\n"
           "    params:\n";
    WriteYamlVector(out, "target_jpos", target, "      ");
    out << "    tasks:\n"
           "      - name: jpos_task\n"
           "        weight: 1.0\n"
           "        level: 1\n";
  }

  const YAML::Node loaded =
      ConfigLoader::LoadFileAndResolve(root_path.string());
  auto config = compileRuntimeConfig(loaded, *robot);
  auto arch = std::make_unique<ControlArchitecture>(std::move(config), robot);
  RegisterYamlJointCommandState(*arch);

  auto state = MakeZeroRobotState(*robot);
  state.time = 0.0;
  arch->Update(state, 0.001);

  const auto& sol = arch->solver()->solution();
  ASSERT_TRUE(sol.success);

  const auto* active_state = arch->fsmHandler()->GetCurrentState();
  ASSERT_NE(active_state, nullptr);
  auto jpos_task = std::dynamic_pointer_cast<wbc::tasks::TaskJointPosture>(
      active_state->Task("jpos_task"));
  ASSERT_NE(jpos_task, nullptr);
  EXPECT_LT(InfNorm(jpos_task->position_ref() - target), 1e-12);

  const auto& state_cfg = arch->config()->states.at(0);
  const auto problem = arch->registry()->buildProblem(
      robot->time(), robot->q(), robot->qdot(), arch->solver()->data(),
      state_cfg.task_names, state_cfg.task_weights, state_cfg.task_levels,
      state_cfg.contact_names);
  ASSERT_EQ(problem.objectives.size(), 1u);
  ASSERT_TRUE(problem.objectives[0].isMotionConstraint());

  const auto& motion = problem.objectives[0].motionConstraint();
  const Eigen::VectorXd expected_acc = 40.0 * target;
  EXPECT_LT(InfNorm(motion.vector() - expected_acc), 1e-9);
  EXPECT_LT(InfNorm(motion.matrix() * sol.qddot_sol - motion.vector()), 1e-6);
  EXPECT_LT(InfNorm(sol.qddot_sol.tail(robot->na()) - expected_acc), 1e-6);

  const auto& cmd = arch->command();
  const Eigen::VectorXd expected_qdot = 0.001 * expected_acc;
  const Eigen::VectorXd expected_q =
      state.q.tail(robot->na()) + 0.001 * expected_qdot;
  EXPECT_LT(InfNorm(cmd.qdot - expected_qdot), 1e-8);
  EXPECT_LT(InfNorm(cmd.q - expected_q), 1e-8);
  EXPECT_GT(cmd.qdot(0), 0.0);
  if (target.size() > 3) {
    EXPECT_LT(cmd.qdot(3), 0.0);
  }

  fs::remove_all(temp_dir);
}

TEST_F(ArchitectureTest, YamlJointCommand_GravityCompensatesAtStaticReference) {
  namespace fs = std::filesystem;
  const auto stamp = std::to_string(
      std::chrono::steady_clock::now().time_since_epoch().count());
  const fs::path temp_dir =
      fs::temp_directory_path() / ("wbc_core_yaml_gravity_test_" + stamp);
  fs::create_directories(temp_dir);

  const fs::path root_path = temp_dir / "wbc_gravity.yaml";
  auto state = MakeZeroRobotState(*robot);
  const Eigen::VectorXd target = state.q.tail(robot->nq_actuated());

  {
    std::ofstream out(root_path);
    out << "controller:\n"
           "  dt: 0.001\n"
           "  qddot_ref: false\n"
           "solver:\n"
           "  type: SOLVER_HQP_PROXQP\n"
           "  qp_params:\n"
           "    verbose: false\n"
           "    max_iter: 100\n"
           "    eps_abs: 1.0e-8\n"
           "    eps_rel: 0.0\n"
           "regularization:\n"
           "  w_delta_qddot: 1.0e-5\n"
           "task_pool:\n"
           "  - name: jpos_task\n"
           "    type: JointTask\n"
           "    role: bias_task\n"
           "    kp: 60.0\n"
           "    kd: 12.0\n"
           "    weight: 1.0\n"
           "state_machine:\n"
           "  - id: 0\n"
           "    name: yaml_joint_command\n"
           "    stay_here: true\n"
           "    params:\n";
    WriteYamlVector(out, "target_jpos", target, "      ");
    out << "    tasks:\n"
           "      - name: jpos_task\n"
           "        weight: 1.0\n"
           "        level: 1\n";
  }

  const YAML::Node loaded =
      ConfigLoader::LoadFileAndResolve(root_path.string());
  auto config = compileRuntimeConfig(loaded, *robot);
  auto arch = std::make_unique<ControlArchitecture>(std::move(config), robot);
  RegisterYamlJointCommandState(*arch);

  state.time = 0.0;
  arch->Update(state, 0.001);

  const auto& sol = arch->solver()->solution();
  ASSERT_TRUE(sol.success);
  EXPECT_LT(InfNorm(sol.qddot_sol), 1e-7);

  const auto& data = arch->solver()->data();
  const auto& h = robot->nonLinearEffects(data);
  EXPECT_LT(InfNorm(sol.tau_cmd - h), 1e-7);
  EXPECT_LT(InfNorm(arch->command().tau - h), 1e-7);
  EXPECT_LT(InfNorm(arch->command().q - target), 1e-10);
  EXPECT_LT(InfNorm(arch->command().qdot), 1e-10);

  fs::remove_all(temp_dir);
}

TEST_F(ArchitectureTest, PinocchioDynamics_FixedBaseTorqueMatchesSolvedState) {
  auto root = makeMinimalConfig();
  root["controller"]["qddot_ref"] = false;
  root["solver"]["type"] = "SOLVER_HQP_PROXQP";
  root["solver"]["qp_params"]["verbose"] = false;
  root["solver"]["qp_params"]["max_iter"] = 100;

  auto arch = MakeTestArchitecture(root, robot);

  auto state = MakeZeroRobotState(*robot);
  state.time = 0.0;
  arch->Update(state, 0.001);

  const auto& sol = arch->solver()->solution();
  ASSERT_TRUE(sol.success);
  ASSERT_TRUE(robot->is_fixed_base());
  ASSERT_EQ(sol.qddot_sol.size(), robot->nv());
  ASSERT_EQ(sol.tau_cmd.size(), robot->na());

  const auto& data = arch->solver()->data();
  const auto& M = robot->mass(data);
  const auto& h = robot->nonLinearEffects(data);
  const Eigen::VectorXd tau_from_pinocchio = M * sol.qddot_sol + h;

  ASSERT_EQ(tau_from_pinocchio.size(), sol.tau_cmd.size());
  EXPECT_LT(InfNorm(tau_from_pinocchio - sol.tau_cmd), 1e-7);
  EXPECT_TRUE(arch->command().tau.isApprox(sol.tau_cmd, 1e-12));
}

TEST_F(ArchitectureTest,
       YamlFileToIDHQPDataFlow_FloatingBaseContactPassesPinocchioChecks) {
  namespace fs = std::filesystem;
  const auto stamp = std::to_string(
      std::chrono::steady_clock::now().time_since_epoch().count());
  const fs::path temp_dir =
      fs::temp_directory_path() / ("wbc_core_yaml_contact_test_" + stamp);
  fs::create_directories(temp_dir);

  const fs::path root_path = temp_dir / "wbc_contact.yaml";

  {
    std::ofstream out(root_path);
    out << "controller:\n"
           "  dt: 0.001\n"
           "  qddot_ref: false\n"
           "solver:\n"
           "  type: SOLVER_HQP_PROXQP\n"
           "  qp_params:\n"
           "    verbose: false\n"
           "    max_iter: 200\n"
           "    eps_abs: 1.0e-7\n"
           "    eps_rel: 0.0\n"
           "regularization:\n"
           "  w_delta_qddot: 1.0e-4\n"
           "  w_lambda: 1.0e-6\n"
           "task_pool:\n"
           "  - name: jpos_task\n"
           "    type: JointTask\n"
           "    role: bias_task\n"
           "    kp: 25.0\n"
           "    kd: 5.0\n"
           "    weight: 1.0\n"
           "contact_pool:\n"
           "  - name: rfoot\n"
           "    type: SurfaceContact\n"
           "    target_frame: RAnkleRoll\n"
           "    foot_half_length: 0.10\n"
           "    foot_half_width: 0.05\n"
           "    mu: 0.8\n"
           "    fMin: 1.0\n"
           "    fMax: 2000.0\n"
           "state_machine:\n"
           "  - id: 0\n"
           "    name: initialize\n"
           "    stay_here: true\n"
           "    contacts:\n"
           "      - rfoot\n"
           "    tasks:\n"
           "      - name: jpos_task\n"
           "        weight: 1.0\n"
           "        level: 2\n";
  }

  vector<string> pkgs = {model_dir};
  auto fb_robot =
      make_shared<RobotSystem>(model_dir + "/romeo/urdf/romeo.urdf", pkgs,
                               pinocchio::JointModelFreeFlyer());

  const YAML::Node loaded =
      ConfigLoader::LoadFileAndResolve(root_path.string());
  auto config = compileRuntimeConfig(loaded, *fb_robot);
  auto arch =
      std::make_unique<ControlArchitecture>(std::move(config), fb_robot);
  RegisterTestStates(*arch);

  auto state = MakeZeroRobotState(*fb_robot);
  state.time = 0.0;
  arch->Update(state, 0.001);

  const auto& sol = arch->solver()->solution();
  ASSERT_TRUE(sol.success);
  ASSERT_FALSE(fb_robot->is_fixed_base());
  ASSERT_EQ(sol.qddot_sol.size(), fb_robot->nv());
  ASSERT_EQ(sol.tau_cmd.size(), fb_robot->na());
  ASSERT_TRUE(sol.lambda_sol.allFinite());

  const auto& state_cfg = arch->config()->states.at(0);
  const auto problem = arch->registry()->buildProblem(
      fb_robot->time(), fb_robot->q(), fb_robot->qdot(), arch->solver()->data(),
      state_cfg.task_names, state_cfg.task_weights, state_cfg.task_levels,
      state_cfg.contact_names);

  ASSERT_EQ(problem.contacts.size(), 1u);
  ASSERT_EQ(TotalLambdaDim(problem.contacts), sol.lambda_sol.size());

  Eigen::VectorXd generalized_contact_force =
      Eigen::VectorXd::Zero(fb_robot->nv());
  int lambda_offset = 0;
  for (const auto& contact : problem.contacts) {
    const int lambda_dim = contact.lambdaDim();
    const auto lambda = sol.lambda_sol.segment(lambda_offset, lambda_dim);

    const Eigen::VectorXd contact_accel =
        contact.Jc * sol.qddot_sol + contact.Jcdot_qdot;
    EXPECT_LT(InfNorm(contact_accel), 1e-5);

    const Eigen::VectorXd force_margin_lb = contact.Uf * lambda - contact.uf_lb;
    const Eigen::VectorXd force_margin_ub = contact.uf_ub - contact.Uf * lambda;
    EXPECT_GE(force_margin_lb.minCoeff(), -1e-5);
    EXPECT_GE(force_margin_ub.minCoeff(), -1e-5);

    generalized_contact_force.noalias() +=
        contact.Jc.transpose() * contact.T * lambda;
    lambda_offset += lambda_dim;
  }
  ASSERT_EQ(lambda_offset, sol.lambda_sol.size());

  const auto& data = arch->solver()->data();
  const auto& M = fb_robot->mass(data);
  const auto& h = fb_robot->nonLinearEffects(data);
  const Eigen::VectorXd pinocchio_residual =
      M * sol.qddot_sol + h - generalized_contact_force;

  EXPECT_LT(InfNorm(pinocchio_residual.head(6)), 1e-5);
  EXPECT_LT(InfNorm(pinocchio_residual.tail(fb_robot->na()) - sol.tau_cmd),
            1e-5);

  fs::remove_all(temp_dir);
}

TEST_F(ArchitectureTest, ControlArchitecture_StateTransition) {
  auto root = makeMinimalConfig();
  auto arch = MakeTestArchitecture(root, robot);
  arch->Initialize();

  auto state = MakeZeroRobotState(*robot);

  // Start in state 0
  state.time = 0.0;
  arch->Update(state, 0.001);
  EXPECT_EQ(arch->fsmHandler()->GetCurrentStateId(), 0);

  // Request state 1
  arch->RequestState(1);
  state.time = 0.001;
  arch->Update(state, 0.001);
  EXPECT_EQ(arch->fsmHandler()->GetCurrentStateId(), 1);

  // Command should still be valid
  const auto& cmd = arch->command();
  EXPECT_TRUE(cmd.tau.allFinite());
}

TEST_F(ArchitectureTest, ControlArchitecture_InitialCommandHoldsMeasuredState) {
  auto root = makeMinimalConfig();
  auto arch = MakeTestArchitecture(root, robot);
  arch->Initialize();

  auto state = MakeZeroRobotState(*robot);
  state.q.tail(robot->nq_actuated()).setConstant(0.05);
  state.qdot.tail(robot->na()).setConstant(0.02);
  state.time = 0.0;

  arch->Update(state, 0.0);

  const auto& cmd = arch->command();
  EXPECT_TRUE(cmd.q.isApprox(state.q.tail(robot->nq_actuated())));
  EXPECT_TRUE(cmd.qdot.isApprox(state.qdot.tail(robot->na())));
  EXPECT_TRUE(cmd.tau.allFinite());
}

TEST_F(ArchitectureTest, ControlArchitecture_StateMachineReferenceOnly) {
  auto root = makeCartesianConfig();

  // Add a state whose joint reference is fully defined by YAML params.
  YAML::Node jtState;
  jtState["id"] = 3;
  jtState["name"] = "yaml_joint_command";
  jtState["stay_here"] = true;
  YAML::Node target;
  for (int i = 0; i < robot->na(); ++i) {
    target.push_back(0.1);
  }
  jtState["params"]["target_jpos"] = target;
  YAML::Node jtTask;
  jtTask["name"] = "jpos_task";
  jtTask["weight"] = 1.0;
  jtState["tasks"].push_back(jtTask);
  root["state_machine"].push_back(jtState);

  auto arch = MakeTestArchitecture(root, robot);
  RegisterYamlJointCommandState(*arch);
  arch->Initialize();

  auto state = MakeZeroRobotState(*robot);

  // Go to YAML-driven joint reference state.
  state.time = 0.0;
  arch->Update(state, 0.001);
  arch->RequestState(3);
  state.time = 0.001;
  arch->Update(state, 0.001);
  EXPECT_EQ(arch->fsmHandler()->GetCurrentStateId(), 3);

  state.time = 0.002;
  arch->Update(state, 0.001);
  const auto& cmd = arch->command();
  EXPECT_TRUE(cmd.tau.allFinite());
}

TEST_F(ArchitectureTest, ControlArchitecture_LongRunStability) {
  auto root = makeMinimalConfig();
  auto arch = MakeTestArchitecture(root, robot);
  arch->Initialize();

  auto state = MakeZeroRobotState(*robot);

  for (int i = 0; i < 500; ++i) {
    state.time = i * 0.001;
    arch->Update(state, 0.001);
    const auto& cmd = arch->command();
    ASSERT_TRUE(cmd.tau.allFinite()) << "NaN at tick " << i;
  }
}

TEST_F(ArchitectureTest, CommandAdapter_MapsFullCommandPayload) {
  CommandAdapter adapter;

  LowLevelCommand cmd;
  cmd.Initialize(robot->na());

  wbc::IDSolution sol;
  sol.tau_cmd = Eigen::VectorXd::Constant(robot->na(), 1.23);
  sol.qddot_sol = Eigen::VectorXd::Zero(robot->nv());
  sol.q_cmd = pinocchio::neutral(robot->model());
  sol.qdot_cmd = Eigen::VectorXd::Zero(robot->nv());

  const bool ok = adapter.fromSolution(sol, *robot, cmd);

  EXPECT_TRUE(ok);
  EXPECT_EQ(cmd.tau.size(), robot->na());
  EXPECT_TRUE(cmd.tau.isApprox(sol.tau_cmd));
  EXPECT_TRUE(cmd.q.isApprox(sol.q_cmd.head(robot->na())));
  EXPECT_TRUE(cmd.qdot.isApprox(sol.qdot_cmd.head(robot->na())));
}

TEST_F(ArchitectureTest,
       CommandAdapter_RejectsMissingIntegratedCommandPayload) {
  CommandAdapter adapter;

  LowLevelCommand cmd;
  cmd.Initialize(robot->na());

  wbc::IDSolution sol;
  sol.tau_cmd = Eigen::VectorXd::Zero(robot->na());
  sol.qddot_sol = Eigen::VectorXd::Zero(robot->nv());

  const bool ok = adapter.fromSolution(sol, *robot, cmd);

  EXPECT_FALSE(ok);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
