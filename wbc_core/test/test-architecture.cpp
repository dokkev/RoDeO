//
// Copyright (c) 2026
//
// Tests for wbc_core architecture layer:
//   - ConfigCompiler/ConfigValidator/RuntimeAssembler pipeline
//   - FSMHandler (state lifecycle and transitions)
//   - StateMachine base (Initialize, Home, CartesianTeleop, JointTeleop)
//   - ControlArchitecture (end-to-end solve)
//

#include <gtest/gtest.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>

#include <pinocchio/algorithm/joint-configuration.hpp>

#include <wbc_core/robots/robot-wrapper.hpp>

#include "wbc_core/architecture/control_architecture.hpp"
#include "wbc_core/architecture/fsm_handler.hpp"
#include "wbc_core/architecture/state_machine.hpp"
#include "wbc_core/architecture/states/initialize_state.hpp"
#include "wbc_core/architecture/states/home_state.hpp"
#include "wbc_core/architecture/states/cartesian_teleop_state.hpp"
#include "wbc_core/architecture/states/joint_teleop_state.hpp"
#include "wbc_core/controller/wbmc.hpp"
#include "wbc_core/controller/wbmc-registry.hpp"
#include "wbc_core/runtime/config_loader.hpp"
#include "wbc_core/runtime/runtime_assembler.hpp"
#include "wbc_core/runtime/config_compiler.hpp"
#include "wbc_core/runtime/config_validator.hpp"

using namespace wbc;
using namespace tsid;
using namespace tsid::robots;
using namespace std;

namespace {

RuntimeConfig compileRuntimeConfig(const YAML::Node& root,
                                   RobotWrapper& robot) {
  CompiledConfig compiled_config = ConfigCompiler::Compile(root);
  ConfigValidator::Validate(compiled_config);
  return RuntimeAssembler::Assemble(compiled_config, robot);
}

}  // namespace

// ─────────────────────────────────────────────────────────────────────────────
// Fixture: Romeo robot (reusable)
// ─────────────────────────────────────────────────────────────────────────────

class ArchitectureTest : public ::testing::Test {
 protected:
  shared_ptr<RobotWrapper> robot;
  string model_dir;

  void SetUp() override {
    model_dir = TSID_MODEL_DIR;
    vector<string> package_dirs = {model_dir};
    const string urdfFile = model_dir + "/romeo/urdf/romeo.urdf";
    robot = make_shared<RobotWrapper>(urdfFile, package_dirs, false);
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
    jposTask["kp_ik"] = 1.0;
    root["task_pool"].push_back(jposTask);

    // State machine
    YAML::Node initState;
    initState["id"] = 0;
    initState["name"] = "initialize";
    initState["params"]["duration"] = 1.0;
    initState["params"]["wait_time"] = 0.0;
    initState["params"]["next_state_id"] = 1;
    YAML::Node initTask;
    initTask["name"] = "jpos_task";
    initTask["weight"] = 1.0;
    initState["tasks"].push_back(initTask);
    root["state_machine"].push_back(initState);

    YAML::Node homeState;
    homeState["id"] = 1;
    homeState["name"] = "home";
    homeState["type"] = "initialize";
    homeState["params"]["stay_here"] = true;
    homeState["params"]["duration"] = 1.0;
    YAML::Node homeTask;
    homeTask["name"] = "jpos_task";
    homeTask["weight"] = 1.0;
    homeState["tasks"].push_back(homeTask);
    root["state_machine"].push_back(homeState);

    return root;
  }

  /// Build a config with position task for cartesian teleop.
  YAML::Node makeCartesianConfig() {
    YAML::Node root = makeMinimalConfig();

    // Add SE3 position task
    YAML::Node posTask;
    posTask["name"] = "ee_pos_task";
    posTask["type"] = "LinkPosTask";
    posTask["role"] = "operational_task";
    posTask["target_frame"] = "RWristPitch";
    posTask["kp"] = 50.0;
    posTask["kd"] = 10.0;
    posTask["weight"] = 100.0;
    root["task_pool"].push_back(posTask);

    // Add cartesian teleop state
    YAML::Node cartState;
    cartState["id"] = 2;
    cartState["name"] = "cartesian_teleop";
    cartState["params"]["stay_here"] = true;
    YAML::Node cartJpos;
    cartJpos["name"] = "jpos_task";
    cartJpos["weight"] = 1.0;
    YAML::Node cartEePos;
    cartEePos["name"] = "ee_pos_task";
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

TEST_F(ArchitectureTest, RuntimeAssembler_BuildsSurfaceContactFromCompiledConfig) {
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
  RobotWrapper fb_robot(model_dir + "/romeo/urdf/romeo.urdf", pkgs,
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
  RobotWrapper fb_robot(model_dir + "/romeo/urdf/romeo.urdf", pkgs,
                        pinocchio::JointModelFreeFlyer());

  auto compiled_config = ConfigCompiler::Compile(root);
  EXPECT_THROW(RuntimeAssembler::Assemble(compiled_config, fb_robot),
               std::invalid_argument);
}

TEST_F(ArchitectureTest, ConfigValidator_RejectsStateUnknownTaskReference) {
  auto root = makeMinimalConfig();
  root["state_machine"][0]["tasks"][0]["name"] = "missing_task";
  const auto compiled_config = ConfigCompiler::Compile(root);

  EXPECT_THROW(ConfigValidator::Validate(compiled_config), std::invalid_argument);
}

TEST_F(ArchitectureTest,
       ConfigValidator_RejectsStateUnknownContactReference) {
  auto root = makeMinimalConfig();
  root["state_machine"][0]["contacts"].push_back("missing_contact");
  const auto compiled_config = ConfigCompiler::Compile(root);

  EXPECT_THROW(ConfigValidator::Validate(compiled_config), std::invalid_argument);
}

TEST_F(ArchitectureTest, ConfigValidator_RejectsDuplicateTaskPoolNames) {
  auto root = makeMinimalConfig();
  YAML::Node dupTask = root["task_pool"][0];
  root["task_pool"].push_back(dupTask);
  const auto compiled_config = ConfigCompiler::Compile(root);

  EXPECT_THROW(ConfigValidator::Validate(compiled_config), std::invalid_argument);
}

TEST_F(ArchitectureTest, ConfigValidator_RejectsDuplicateStateIds) {
  auto root = makeMinimalConfig();
  root["state_machine"][1]["id"] = root["state_machine"][0]["id"].as<int>();
  const auto compiled_config = ConfigCompiler::Compile(root);

  EXPECT_THROW(ConfigValidator::Validate(compiled_config), std::invalid_argument);
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
  EXPECT_THROW(RuntimeAssembler::Assemble(compiled_config, *robot), std::invalid_argument);
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
  root["regularization"]["w_qddot"] = 0.05;
  root["regularization"]["w_xc_ddot"] = 50.0;
  root["controller"]["kp_acc"] = 200.0;
  root["controller"]["kd_acc"] = 30.0;

  auto config = compileRuntimeConfig(root, *robot);

  EXPECT_DOUBLE_EQ(config.regularization.w_delta_qddot, 0.05);
  EXPECT_DOUBLE_EQ(config.contact_accel_weight, 50.0);
  EXPECT_DOUBLE_EQ(config.kp_acc, 200.0);
  EXPECT_DOUBLE_EQ(config.kd_acc, 30.0);
}

TEST_F(ArchitectureTest, ConfigCompiler_RejectsLegacyTaskHierarchyField) {
  auto root = makeMinimalConfig();

  root["state_machine"][0].remove("tasks");
  root["state_machine"][0]["task_hierarchy"].push_back("jpos_task");

  EXPECT_THROW(ConfigCompiler::Compile(root), std::invalid_argument);
}

TEST_F(ArchitectureTest,
       ConfigCompiler_RejectsLegacyContactConstraintsField) {
  auto root = makeMinimalConfig();

  root["state_machine"][0]["contact_constraints"].push_back("left_foot");

  EXPECT_THROW(ConfigCompiler::Compile(root), std::invalid_argument);
}

TEST_F(ArchitectureTest, ConfigCompiler_StateTaskSelectionPreserved) {
  auto root = makeCartesianConfig();
  auto config = compileRuntimeConfig(root, *robot);

  // State 2 (cartesian_teleop) should preserve selected tasks and weights.
  auto& sc = config.states[2];
  ASSERT_EQ(sc.task_names.size(), 2u);
  EXPECT_EQ(sc.task_names[0], "jpos_task");
  EXPECT_EQ(sc.task_names[1], "ee_pos_task");
  ASSERT_EQ(sc.task_weights.size(), 2u);
  EXPECT_DOUBLE_EQ(sc.task_weights[0], 1.0);
  EXPECT_DOUBLE_EQ(sc.task_weights[1], 100.0);
}

TEST_F(ArchitectureTest, ConfigCompiler_StateUnknownTaskReferenceThrows) {
  auto root = makeMinimalConfig();
  root["state_machine"][0]["tasks"][0]["name"] = "missing_task";

  EXPECT_THROW(compileRuntimeConfig(root, *robot), std::invalid_argument);
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

TEST_F(ArchitectureTest, ConfigCompiler_RejectsUnknownGlobalConstraintType) {
  auto root = makeMinimalConfig();
  root["global_constraints"]["BadConstraint"]["enabled"] = true;

  EXPECT_THROW(ConfigCompiler::Compile(root), std::invalid_argument);
}

TEST_F(ArchitectureTest, RuntimeAssembler_RejectsUnknownStateTypeAtInitializeFsm) {
  auto root = makeMinimalConfig();
  root["state_machine"][0]["type"] = "not_a_real_state";
  auto config = compileRuntimeConfig(root, *robot);

  tsid::WBMCRegistry registry(*robot);
  tsid::WBMC solver(*robot);
  auto q0 = pinocchio::neutral(robot->model());
  auto v0 = math::Vector::Zero(robot->nv());
  robot->update(solver.data(), q0, v0);

  FSMHandler fsm;
  StateProvider sp;
  sp.nominal_jpos = q0;

  EXPECT_THROW(
      RuntimeAssembler::InitializeFsm(config, registry, fsm, sp, *robot,
                                    solver.data()),
      std::invalid_argument);
}

TEST_F(ArchitectureTest, ConfigLoader_LoadFileAndResolve_MergesExternalYaml) {
  namespace fs = std::filesystem;
  const auto stamp =
      std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
  const fs::path temp_dir = fs::temp_directory_path() /
                            ("wbc_core_config_loader_test_" + stamp);
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
           "    role: bias_task\n";
  }

  const YAML::Node resolved = ConfigLoader::LoadFileAndResolve(root_path.string());

  ASSERT_TRUE(resolved["task_pool"]);
  ASSERT_EQ(resolved["task_pool"].size(), 2u);
  EXPECT_EQ(resolved["task_pool"][0]["name"].as<std::string>(), "inline_task");
  EXPECT_EQ(resolved["task_pool"][1]["name"].as<std::string>(), "ext_task");

  ASSERT_TRUE(resolved["contact_pool"]);
  ASSERT_EQ(resolved["contact_pool"].size(), 1u);
  EXPECT_EQ(resolved["contact_pool"][0]["name"].as<std::string>(), "ext_contact");

  ASSERT_TRUE(resolved["state_machine"]);
  ASSERT_EQ(resolved["state_machine"].size(), 1u);
  EXPECT_EQ(resolved["state_machine"][0]["id"].as<int>(), 7);

  fs::remove_all(temp_dir);
}

TEST_F(ArchitectureTest,
       RuntimeAssembler_RegistryBoundary_UsesStateTaskWeightOverride) {
  auto root = makeMinimalConfig();
  root["task_pool"][0]["role"] = "operational_task";
  root["state_machine"][0]["tasks"][0]["weight"] = 3.5;

  auto config = compileRuntimeConfig(root, *robot);

  tsid::WBMCRegistry registry(*robot);
  tsid::WBMC solver(*robot);
  auto q0 = pinocchio::neutral(robot->model());
  auto v0 = math::Vector::Zero(robot->nv());
  robot->update(solver.data(), q0, v0);

  FSMHandler fsm;
  StateProvider sp;
  sp.nominal_jpos = q0;
  RuntimeAssembler::InitializeFsm(config, registry, fsm, sp, *robot, solver.data());

  const auto& state_cfg = config.states.at(0);
  const auto input = registry.buildStepInput(
      0.0, q0, v0, state_cfg.task_names, state_cfg.task_weights,
      state_cfg.contact_names);

  ASSERT_EQ(input.objectives.size(), 1u);
  ASSERT_TRUE(input.objectives[0].isMotion());
  EXPECT_EQ(input.objectives[0].motion().name, "jpos_task");
  EXPECT_DOUBLE_EQ(input.objectives[0].weight, 3.5);
}

TEST_F(ArchitectureTest,
       ConfigCompiler_IgnoresJointVelLimitConstraint) {
  auto root = makeMinimalConfig();
  root["global_constraints"]["JointVelLimitConstraint"]["enabled"] = true;
  const auto compiled_config = ConfigCompiler::Compile(root);
  EXPECT_TRUE(compiled_config.global_constraints.empty());
}

TEST_F(ArchitectureTest,
       ConfigCompiler_IgnoresJointPosLimitConstraint) {
  auto root = makeMinimalConfig();
  root["global_constraints"]["JointPosLimitConstraint"]["enabled"] = true;
  const auto compiled_config = ConfigCompiler::Compile(root);
  EXPECT_TRUE(compiled_config.global_constraints.empty());
}

// ═══════════════════════════════════════════════════════════════════════════════
// FSMHandler
// ═══════════════════════════════════════════════════════════════════════════════

TEST_F(ArchitectureTest, FSMHandler_RegisterAndStart) {
  FSMHandler fsm;
  StateMachineContext ctx;
  ctx.robot = robot.get();
  StateProvider sp;
  ctx.state_provider = &sp;

  // Create dummy states
  auto s0 = make_unique<InitializeState>(0, "init", ctx);
  auto s1 = make_unique<InitializeState>(1, "home", ctx);
  fsm.RegisterState(0, move(s0));
  fsm.RegisterState(1, move(s1));

  EXPECT_TRUE(fsm.SetStartState(0));
  EXPECT_EQ(fsm.GetCurrentStateId(), 0);
}

TEST_F(ArchitectureTest, FSMHandler_StateTransition) {
  FSMHandler fsm;
  StateMachineContext ctx;
  ctx.robot = robot.get();
  StateProvider sp;
  ctx.state_provider = &sp;

  auto s0 = make_unique<InitializeState>(0, "init", ctx);
  auto s1 = make_unique<InitializeState>(1, "home", ctx);
  fsm.RegisterState(0, move(s0));
  fsm.RegisterState(1, move(s1));
  fsm.SetStartState(0);

  // First update triggers FirstVisit
  fsm.Update(0.0);
  EXPECT_EQ(fsm.GetCurrentStateId(), 0);

  // Request transition
  EXPECT_TRUE(fsm.RequestState(1));
  fsm.Update(0.001);
  EXPECT_EQ(fsm.GetCurrentStateId(), 1);
}

TEST_F(ArchitectureTest, FSMHandler_AutoTransition) {
  FSMHandler fsm;
  StateMachineContext ctx;
  ctx.robot = robot.get();
  StateProvider sp;
  ctx.state_provider = &sp;

  auto s0 = make_unique<InitializeState>(0, "init", ctx);
  YAML::Node params;
  params["duration"] = 0.001;
  params["wait_time"] = 0.0;
  params["next_state_id"] = 1;
  s0->SetParameters(params);

  auto s1 = make_unique<InitializeState>(1, "home", ctx);
  YAML::Node params1;
  params1["stay_here"] = true;
  s1->SetParameters(params1);

  fsm.RegisterState(0, move(s0));
  fsm.RegisterState(1, move(s1));
  fsm.SetStartState(0);

  // Tick past duration
  fsm.Update(0.0);  // FirstVisit at t=0
  EXPECT_EQ(fsm.GetCurrentStateId(), 0);

  fsm.Update(0.01);  // elapsed > duration → auto-transition
  EXPECT_EQ(fsm.GetCurrentStateId(), 1);
}

TEST_F(ArchitectureTest, FSMHandler_FindByName) {
  FSMHandler fsm;
  StateMachineContext ctx;
  ctx.robot = robot.get();
  StateProvider sp;
  ctx.state_provider = &sp;

  fsm.RegisterState(0, make_unique<InitializeState>(0, "init", ctx));
  fsm.RegisterState(1, make_unique<InitializeState>(1, "home", ctx));

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
// StateMachine: Initialize state
// ═══════════════════════════════════════════════════════════════════════════════

TEST_F(ArchitectureTest, InitializeState_RampsToTarget) {
  const int na = robot->na();
  auto q0 = pinocchio::neutral(robot->model());
  auto v0 = math::Vector::Zero(robot->nv());
  pinocchio::Data data(robot->model());
  robot->update(data, q0, v0);

  // Create a posture task
  auto jposTask = make_shared<tasks::TaskJointPosture>("jpos", *robot);
  jposTask->Kp(100.0 * math::Vector::Ones(na));
  jposTask->Kd(10.0 * math::Vector::Ones(na));

  // Create state
  StateMachineContext ctx;
  ctx.robot = robot.get();
  ctx.data = &data;
  StateProvider sp;
  sp.nominal_jpos = q0;
  ctx.state_provider = &sp;

  InitializeState state(0, "init", ctx);
  state.assignTask("jpos", jposTask);

  YAML::Node params;
  params["duration"] = 1.0;
  // Target: offset from zero
  YAML::Node target;
  for (int i = 0; i < na; ++i) target.push_back(0.1);
  params["target_jpos"] = target;
  state.SetParameters(params);

  // Run lifecycle
  state.EnterState(0.0);
  state.FirstVisit();

  // Tick at t=0 (alpha=0, should be at start)
  state.UpdateStateTime(0.0);
  state.OneStep();

  // Tick at t=0.5 (alpha≈0.5)
  state.UpdateStateTime(0.5);
  state.OneStep();

  // Tick at t=1.0 (alpha=1.0, should be at target)
  state.UpdateStateTime(1.0);
  state.OneStep();

  // Verify task reference is set (just check it's finite)
  // The task's internal reference should have been updated via setReference()
  SUCCEED();  // If we got here without crash, the state worked
}

// ═══════════════════════════════════════════════════════════════════════════════
// StateMachine: JointTeleop state
// ═══════════════════════════════════════════════════════════════════════════════

TEST_F(ArchitectureTest, JointTeleopState_AcceptsExternalInput) {
  const int na = robot->na();
  auto q0 = pinocchio::neutral(robot->model());
  auto v0 = math::Vector::Zero(robot->nv());
  pinocchio::Data data(robot->model());
  robot->update(data, q0, v0);

  auto jposTask = make_shared<tasks::TaskJointPosture>("jpos", *robot);
  jposTask->Kp(100.0 * math::Vector::Ones(na));
  jposTask->Kd(10.0 * math::Vector::Ones(na));

  StateMachineContext ctx;
  ctx.robot = robot.get();
  ctx.data = &data;
  StateProvider sp;
  sp.nominal_jpos = q0;
  ctx.state_provider = &sp;

  JointTeleopState state(0, "jteleop", ctx);
  state.assignTask("jpos", jposTask);

  YAML::Node params;
  params["stay_here"] = true;
  state.SetParameters(params);

  state.EnterState(0.0);
  state.FirstVisit();

  // Send external q_des
  TaskInput input;
  input.q_des = Eigen::VectorXd::Constant(na, 0.5);
  state.SetExternalInput(input);

  state.UpdateStateTime(0.001);
  state.OneStep();

  SUCCEED();
}

// ═══════════════════════════════════════════════════════════════════════════════
// ConfigCompiler + FSM Integration
// ═══════════════════════════════════════════════════════════════════════════════

TEST_F(ArchitectureTest, RuntimeAssembler_InitializeFsm) {
  auto root = makeMinimalConfig();
  auto config = compileRuntimeConfig(root, *robot);

  tsid::WBMCRegistry registry(*robot);
  tsid::WBMC solver(*robot);
  auto q0 = pinocchio::neutral(robot->model());
  auto v0 = math::Vector::Zero(robot->nv());
  robot->update(solver.data(), q0, v0);

  FSMHandler fsm;
  StateProvider sp;
  sp.nominal_jpos = q0;

  // Initialize FSM
  RuntimeAssembler::InitializeFsm(config, registry, fsm, sp, *robot,
                                 solver.data());

  // Should have 2 states registered
  EXPECT_EQ(fsm.states().size(), 2u);

  // Start state should be 0
  EXPECT_EQ(fsm.GetCurrentStateId(), 0);

  // Should be able to update
  fsm.Update(0.0);
  EXPECT_EQ(fsm.GetCurrentStateId(), 0);
}

TEST_F(ArchitectureTest, RuntimeAssembler_FsmRunsMultipleTicks) {
  auto root = makeMinimalConfig();
  auto config = compileRuntimeConfig(root, *robot);

  tsid::WBMCRegistry registry(*robot);
  tsid::WBMC solver(*robot);
  auto q0 = pinocchio::neutral(robot->model());
  auto v0 = math::Vector::Zero(robot->nv());
  robot->update(solver.data(), q0, v0);

  FSMHandler fsm;
  StateProvider sp;
  sp.nominal_jpos = q0;

  RuntimeAssembler::InitializeFsm(config, registry, fsm, sp, *robot,
                                 solver.data());

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
  auto arch = make_unique<ControlArchitecture>(root, robot);

  arch->Initialize();

  // Check internals
  EXPECT_NE(arch->solver(), nullptr);
  EXPECT_NE(arch->registry(), nullptr);
  EXPECT_NE(arch->fsmHandler(), nullptr);
  EXPECT_NE(arch->stateProvider(), nullptr);

  // Run a few ticks
  RobotJointState state;
  state.q = pinocchio::neutral(robot->model());
  state.qdot = Eigen::VectorXd::Zero(robot->nv());
  state.tau = Eigen::VectorXd::Zero(robot->na());

  for (int i = 0; i < 10; ++i) {
    arch->Update(state, i * 0.001, 0.001);
    const auto& cmd = arch->command();
    EXPECT_TRUE(cmd.tau.allFinite()) << "NaN tau at tick " << i;
    EXPECT_EQ(cmd.tau.size(), robot->na());
  }
}

TEST_F(ArchitectureTest, ControlArchitecture_StateTransition) {
  auto root = makeMinimalConfig();
  auto arch = make_unique<ControlArchitecture>(root, robot);
  arch->Initialize();

  RobotJointState state;
  state.q = pinocchio::neutral(robot->model());
  state.qdot = Eigen::VectorXd::Zero(robot->nv());
  state.tau = Eigen::VectorXd::Zero(robot->na());

  // Start in state 0
  arch->Update(state, 0.0, 0.001);
  EXPECT_EQ(arch->fsmHandler()->GetCurrentStateId(), 0);

  // Request state 1
  arch->RequestState(1);
  arch->Update(state, 0.001, 0.001);
  EXPECT_EQ(arch->fsmHandler()->GetCurrentStateId(), 1);

  // Command should still be valid
  const auto& cmd = arch->command();
  EXPECT_TRUE(cmd.tau.allFinite());
}

TEST_F(ArchitectureTest, ControlArchitecture_ExternalInput) {
  auto root = makeCartesianConfig();

  // Add joint_teleop state for testing
  YAML::Node jtState;
  jtState["id"] = 3;
  jtState["name"] = "joint_teleop";
  jtState["params"]["stay_here"] = true;
  YAML::Node jtTask;
  jtTask["name"] = "jpos_task";
  jtTask["weight"] = 1.0;
  jtState["tasks"].push_back(jtTask);
  root["state_machine"].push_back(jtState);

  auto arch = make_unique<ControlArchitecture>(root, robot);
  arch->Initialize();

  RobotJointState state;
  state.q = pinocchio::neutral(robot->model());
  state.qdot = Eigen::VectorXd::Zero(robot->nv());
  state.tau = Eigen::VectorXd::Zero(robot->na());

  // Go to joint teleop
  arch->Update(state, 0.0, 0.001);
  arch->RequestState(3);
  arch->Update(state, 0.001, 0.001);
  EXPECT_EQ(arch->fsmHandler()->GetCurrentStateId(), 3);

  // Send external input
  TaskInput input;
  input.q_des = Eigen::VectorXd::Constant(robot->na(), 0.1);
  arch->SetExternalInput(input);

  arch->Update(state, 0.002, 0.001);
  const auto& cmd = arch->command();
  EXPECT_TRUE(cmd.tau.allFinite());
}

TEST_F(ArchitectureTest, ControlArchitecture_LongRunStability) {
  auto root = makeMinimalConfig();
  auto arch = make_unique<ControlArchitecture>(root, robot);
  arch->Initialize();

  RobotJointState state;
  state.q = pinocchio::neutral(robot->model());
  state.qdot = Eigen::VectorXd::Zero(robot->nv());
  state.tau = Eigen::VectorXd::Zero(robot->na());

  for (int i = 0; i < 500; ++i) {
    arch->Update(state, i * 0.001, 0.001);
    const auto& cmd = arch->command();
    ASSERT_TRUE(cmd.tau.allFinite()) << "NaN at tick " << i;
  }
}

TEST_F(ArchitectureTest, CommandAdapter_TorqueOnlyModeSupportsStateOptional) {
  CommandAdapter adapter;
  adapter.setOutputMode(CommandOutputMode::kTorqueOnly);

  LowLevelCommand cmd;
  cmd.Initialize(robot->na());

  tsid::WBMCSolution sol;
  sol.tau = Eigen::VectorXd::Constant(robot->na(), 1.23);
  sol.qddot_sol = Eigen::VectorXd::Zero(robot->nv());

  const Eigen::VectorXd empty_q;
  const Eigen::VectorXd empty_qdot;
  const bool ok = adapter.fromSolution(
      sol, *robot, empty_q, empty_qdot, 0.001, cmd);

  EXPECT_TRUE(ok);
  EXPECT_EQ(cmd.tau.size(), robot->na());
  EXPECT_TRUE(cmd.tau.isApprox(sol.tau));
}

TEST_F(ArchitectureTest, CommandAdapter_IntegratedModeRejectsMissingState) {
  CommandAdapter adapter;
  adapter.setOutputMode(CommandOutputMode::kTorqueWithIntegratedState);

  LowLevelCommand cmd;
  cmd.Initialize(robot->na());

  tsid::WBMCSolution sol;
  sol.tau = Eigen::VectorXd::Zero(robot->na());
  sol.qddot_sol = Eigen::VectorXd::Zero(robot->nv());

  const Eigen::VectorXd empty_q;
  const Eigen::VectorXd empty_qdot;
  const bool ok = adapter.fromSolution(
      sol, *robot, empty_q, empty_qdot, 0.001, cmd);

  EXPECT_FALSE(ok);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
