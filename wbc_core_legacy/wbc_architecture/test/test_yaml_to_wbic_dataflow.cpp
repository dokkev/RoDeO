/**
 * @file wbc_core/wbc_architecture/test/test_yaml_to_wbic_dataflow.cpp
 * @brief Validates the data flow from YAML configuration to WBIC QP inputs.
 *
 * Each test picks a specific YAML parameter, sets it to a known value, and
 * asserts that the exact value reaches its expected destination inside the
 * WBIC solver (QPParams, task gains, contact friction cone, etc.).
 *
 * Pipeline under test:
 *   YAML → ConfigCompiler → RuntimeConfig → ControlArchitecture::Initialize()
 *       → QPParams / Task / Contact objects → WBIC solver
 */
#include <sys/types.h>
#include <unistd.h>

#include <filesystem>
#include <fstream>
#include <memory>
#include <string>

#include <gtest/gtest.h>

#include "wbc_architecture/control_architecture.hpp"
#include "wbc_formulation/contact.hpp"
#include "wbc_formulation/interface/task.hpp"
#include "wbc_fsm/state_factory.hpp"
#include "wbc_robot_system/pinocchio_robot_system.hpp"
#include "wbc_solver/wbic.hpp"
#include "wbc_util/constraint_registry.hpp"
#include "wbc_util/task_registry.hpp"

namespace wbc {
namespace {

// ---------------------------------------------------------------------------
// Minimal no-op state (stays indefinitely, no logic).
// ---------------------------------------------------------------------------
class StayState final : public StateMachine {
public:
  StayState(StateId id, const std::string& name, PinocchioRobotSystem* robot,
            TaskRegistry* task_reg, ConstraintRegistry* const_reg,
            StateProvider* sp)
      : StateMachine(id, name, robot, task_reg, const_reg, sp) {}
  void FirstVisit() override {}
  void OneStep() override {}
  void LastVisit() override {}
  bool EndOfState() override { return false; }
  StateId GetNextState() override { return id(); }
};

WBC_REGISTER_STATE(
    "df_stay_state",
    [](StateId id, const std::string& name, const StateMachineConfig& ctx)
        -> std::unique_ptr<StateMachine> {
      return std::make_unique<StayState>(
          id, name, ctx.robot, ctx.task_registry, ctx.constraint_registry,
          ctx.state_provider);
    });


// ---------------------------------------------------------------------------
// Minimal 2-DOF fixed-base URDF (no real hardware needed).
// ---------------------------------------------------------------------------
std::string TwoDofUrdf() {
  return R"(
<robot name="test_two_dof">
  <link name="base_link">
    <inertial><origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="1.0"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
  </link>
  <joint name="joint1" type="revolute">
    <parent link="base_link"/><child link="link1"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="100" velocity="10"/>
  </joint>
  <link name="link1">
    <inertial><origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="1.0"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
  </link>
  <joint name="joint2" type="revolute">
    <parent link="link1"/><child link="link2"/>
    <origin xyz="0 0 0.1" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-3.14" upper="3.14" effort="100" velocity="10"/>
  </joint>
  <link name="link2">
    <inertial><origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="1.0"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
  </link>
</robot>
)";
}

// ---------------------------------------------------------------------------
// Test fixture: writes URDF + YAML to a temp dir, builds ControlArchitecture.
// ---------------------------------------------------------------------------
class YamlToWbicDataFlowTest : public ::testing::Test {
protected:
  void SetUp() override {
    const std::string suffix =
        std::to_string(static_cast<long long>(::getpid())) + "_" +
        std::to_string(counter_++);
    temp_dir_ = std::filesystem::temp_directory_path() /
                ("yaml_wbic_dataflow_" + suffix);
    std::filesystem::create_directories(temp_dir_);
    urdf_path_ = temp_dir_ / "robot.urdf";
    WriteFile(urdf_path_, TwoDofUrdf());
  }

  void TearDown() override {
    std::error_code ec;
    std::filesystem::remove_all(temp_dir_, ec);
  }

  void WriteFile(const std::filesystem::path& p, const std::string& content) {
    std::ofstream f(p);
    f << content;
  }

  // Build an architecture from an inline YAML string.
  std::unique_ptr<ControlArchitecture> MakeArch(const std::string& yaml) {
    const auto yaml_path = temp_dir_ / "config.yaml";
    WriteFile(yaml_path, yaml);
    auto cfg = ControlArchitectureConfig::FromYaml(yaml_path.string(), 0.001);
    cfg.state_provider = std::make_unique<StateProvider>(0.001);
    auto arch = std::make_unique<ControlArchitecture>(std::move(cfg));
    arch->Initialize();
    return arch;
  }

  // Minimal valid YAML with only a JointTask and no contacts (fixed-base).
  std::string BaseYaml(const std::string& extra_task_params = "",
                       const std::string& extra_global    = "") const {
    return
      "robot_model:\n"
      "  urdf_path: \"" + urdf_path_.string() + "\"\n"
      "  is_floating_base: false\n"
      "  base_frame: base_link\n"
      "controller:\n"
      "  enable_gravity_compensation: false\n"
      "  enable_coriolis_compensation: false\n"
      "  enable_inertia_compensation: false\n"
      "  kp_acc: 1.0\n"
      "  kd_acc: 0.1\n"
      "regularization:\n"
      "  w_qddot: 1.0e-6\n"
      "  w_rf: 1.0e-4\n"
      "  w_tau: 1.0e-3\n"
      "  w_tau_dot: 0.0\n"
      "  w_xc_ddot: 1.0e-3\n"
      "  w_f_dot: 1.0e-3\n"
      + extra_global +
      "task_pool:\n"
      "  - name: jpos_task\n"
      "    type: JointTask\n"
      "    kp: 30.0\n"
      "    kd: 3.0\n"
      "    kp_ik: 1.0\n"
      "    weight: 1.0\n"
      + extra_task_params +
      "state_machine:\n"
      "  - id: 1\n"
      "    name: df_stay_state\n"
      "    task_hierarchy:\n"
      "      - {name: jpos_task, priority: 0}\n";
  }

  std::filesystem::path temp_dir_;
  std::filesystem::path urdf_path_;
  inline static int counter_ = 0;
};

// ---------------------------------------------------------------------------
// 1. Regularization weights flow: YAML regularization: → QPParams
// ---------------------------------------------------------------------------
TEST_F(YamlToWbicDataFlowTest, RegularizationWeightsFlowToQpParams) {
  const std::string yaml =
    "robot_model:\n"
    "  urdf_path: \"" + urdf_path_.string() + "\"\n"
    "  is_floating_base: false\n"
    "  base_frame: base_link\n"
    "controller:\n"
    "  enable_gravity_compensation: false\n"
    "  enable_coriolis_compensation: false\n"
    "  enable_inertia_compensation: false\n"
    "  kp_acc: 1.0\n"
    "  kd_acc: 0.1\n"
    "regularization:\n"
    "  w_qddot:  1.23e-5\n"   // sentinel values
    "  w_rf:     2.34e-4\n"
    "  w_tau:    3.45e-3\n"
    "  w_tau_dot: 4.56e-4\n"
    "  w_xc_ddot: 5.67e-3\n"
    "  w_f_dot:   6.78e-3\n"
    "task_pool:\n"
    "  - name: jpos_task\n"
    "    type: JointTask\n"
    "    kp: 1.0\n"
    "    kd: 0.1\n"
    "    kp_ik: 1.0\n"
    "    weight: 1.0\n"
    "state_machine:\n"
    "  - id: 1\n"
    "    name: df_stay_state\n"
    "    task_hierarchy:\n"
    "      - {name: jpos_task, priority: 0}\n";

  auto arch = MakeArch(yaml);
  ASSERT_NE(arch, nullptr);

  const QPParams* qp = arch->GetSolver()->GetWbicData()->qp_params_;
  ASSERT_NE(qp, nullptr);

  EXPECT_NEAR(qp->W_delta_qddot_[0], 1.23e-5, 1e-12)
      << "w_qddot did not reach QPParams::W_delta_qddot_";
  EXPECT_NEAR(qp->W_tau_[0], 3.45e-3, 1e-12)
      << "w_tau did not reach QPParams::W_tau_";
  EXPECT_NEAR(qp->W_tau_dot_[0], 4.56e-4, 1e-12)
      << "w_tau_dot did not reach QPParams::W_tau_dot_";

  // No contacts: W_delta_rf_, W_xc_ddot_, W_f_dot_ are zero-sized (dim=0).
  // Just verify all W_delta_qddot_ entries are uniform.
  EXPECT_TRUE((qp->W_delta_qddot_.array() == qp->W_delta_qddot_[0]).all())
      << "W_delta_qddot_ should be uniform (set via setConstant)";
  EXPECT_TRUE((qp->W_tau_.array() == qp->W_tau_[0]).all())
      << "W_tau_ should be uniform";

  std::cout << "[DataFlow] QPParams: w_qddot=" << qp->W_delta_qddot_[0]
            << " w_tau=" << qp->W_tau_[0]
            << " w_tau_dot=" << qp->W_tau_dot_[0] << "\n";
}

// ---------------------------------------------------------------------------
// 2. Task kp/kd/kp_ik flow: task_pool → task object members
// ---------------------------------------------------------------------------
TEST_F(YamlToWbicDataFlowTest, TaskGainsFlowToTaskObject) {
  const double kp_val   = 77.5;
  const double kd_val   = 14.3;
  const double kp_ik_val = 2.8;
  const double w_val    = 9.1;

  const std::string yaml = BaseYaml(
    "  - name: extra_jpos\n"
    "    type: JointTask\n"
    "    kp: " + std::to_string(kp_val) + "\n"
    "    kd: " + std::to_string(kd_val) + "\n"
    "    kp_ik: " + std::to_string(kp_ik_val) + "\n"
    "    weight: " + std::to_string(w_val) + "\n",
    // Add extra_jpos to state so compiler includes it.
    "");

  // Use a YAML that lists extra_jpos in the task hierarchy so it is compiled.
  const std::string yaml2 =
    "robot_model:\n"
    "  urdf_path: \"" + urdf_path_.string() + "\"\n"
    "  is_floating_base: false\n"
    "  base_frame: base_link\n"
    "controller:\n"
    "  enable_gravity_compensation: false\n"
    "  enable_coriolis_compensation: false\n"
    "  enable_inertia_compensation: false\n"
    "  kp_acc: 1.0\n"
    "  kd_acc: 0.1\n"
    "regularization:\n"
    "  w_qddot: 1.0e-6\n"
    "  w_rf: 1.0e-4\n"
    "  w_tau: 1.0e-3\n"
    "  w_tau_dot: 0.0\n"
    "  w_xc_ddot: 1.0e-3\n"
    "  w_f_dot: 1.0e-3\n"
    "task_pool:\n"
    "  - name: probe_task\n"
    "    type: JointTask\n"
    "    kp: " + std::to_string(kp_val) + "\n"
    "    kd: " + std::to_string(kd_val) + "\n"
    "    kp_ik: " + std::to_string(kp_ik_val) + "\n"
    "    weight: " + std::to_string(w_val) + "\n"
    "state_machine:\n"
    "  - id: 1\n"
    "    name: df_stay_state\n"
    "    task_hierarchy:\n"
    "      - {name: probe_task, priority: 0}\n";

  auto arch = MakeArch(yaml2);
  ASSERT_NE(arch, nullptr);

  const Task* task = arch->GetConfig()->taskRegistry()->GetMotionTask("probe_task");
  ASSERT_NE(task, nullptr);

  // kp and kd are set unconditionally in SetParameters().
  EXPECT_NEAR(task->Kp()[0], kp_val, 1e-9)  << "kp did not reach task->Kp_";
  EXPECT_NEAR(task->Kd()[0], kd_val, 1e-9)  << "kd did not reach task->Kd_";

  // kp_ik is set by the WBIC path in SetParameters().
  EXPECT_NEAR(task->KpIK()[0], kp_ik_val, 1e-9) << "kp_ik did not reach task->kp_ik_";

  // weight is parsed into the default task config (stored in RuntimeConfig).
  // Note: Initialize() resets task->weight_ to kMinWeight; the weight from
  // YAML is stored in DefaultMotionTaskConfigs() and applied at runtime via
  // TaskWeightScheduler. Verify the stored default config here.
  const auto& default_cfgs = arch->GetConfig()->DefaultMotionTaskConfigs();
  const auto it = default_cfgs.find(const_cast<Task*>(task));
  ASSERT_NE(it, default_cfgs.end()) << "probe_task not in DefaultMotionTaskConfigs";
  EXPECT_NEAR(it->second.weight[0], w_val, 1e-9)
      << "weight did not reach DefaultMotionTaskConfigs";

  std::cout << "[DataFlow] Task gains: kp=" << task->Kp()[0]
            << " kd=" << task->Kd()[0]
            << " kp_ik=" << task->KpIK()[0]
            << " stored_weight=" << it->second.weight[0] << "\n";
}

// ---------------------------------------------------------------------------
// 3. Contact friction coefficient flows to PointContact::UfMatrix
//    Row 1: f_x + mu*f_z >= 0  =>  constraint_matrix_(1,2) == mu
// ---------------------------------------------------------------------------
TEST_F(YamlToWbicDataFlowTest, ContactMuFlowsToPointContactFrictionCone) {
  const double mu_val = 0.7;

  const std::string yaml =
    "robot_model:\n"
    "  urdf_path: \"" + urdf_path_.string() + "\"\n"
    "  is_floating_base: false\n"
    "  base_frame: base_link\n"
    "controller:\n"
    "  enable_gravity_compensation: false\n"
    "  enable_coriolis_compensation: false\n"
    "  enable_inertia_compensation: false\n"
    "  kp_acc: 1.0\n"
    "  kd_acc: 0.1\n"
    "regularization:\n"
    "  w_qddot: 1.0e-6\n"
    "  w_rf: 1.0e-4\n"
    "  w_tau: 1.0e-3\n"
    "  w_tau_dot: 0.0\n"
    "  w_xc_ddot: 1.0e-3\n"
    "  w_f_dot: 1.0e-3\n"
    "task_pool:\n"
    "  - name: jpos_task\n"
    "    type: JointTask\n"
    "    kp: 1.0\n"
    "    kd: 0.1\n"
    "    kp_ik: 1.0\n"
    "    weight: 1.0\n"
    "  - name: ee_force\n"
    "    type: ForceTask\n"
    "    contact_name: ee_contact\n"
    "    weight: [0, 0, 0]\n"
    "contact_pool:\n"
    "  - name: ee_contact\n"
    "    type: PointContact\n"
    "    target_frame: link2\n"
    "    mu: " + std::to_string(mu_val) + "\n"
    "state_machine:\n"
    "  - id: 1\n"
    "    name: df_stay_state\n"
    "    task_hierarchy:\n"
    "      - {name: jpos_task, priority: 0}\n"
    "    contact_constraints:\n"
    "      - {name: ee_contact}\n"
    "    force_tasks:\n"
    "      - {name: ee_force}\n";

  auto arch = MakeArch(yaml);
  ASSERT_NE(arch, nullptr);

  Contact* contact = arch->GetConfig()->constraintRegistry()->GetContact("ee_contact");
  ASSERT_NE(contact, nullptr);

  EXPECT_NEAR(contact->Mu(), mu_val, 1e-9)
      << "mu_ not stored correctly in Contact";

  // Force the cone matrix to be built (lazy via cone_dirty_ flag).
  contact->UpdateConeConstraint();
  const Eigen::MatrixXd& Uf = contact->UfMatrix();

  // PointContact friction cone encoding (see contact_constraint.cpp:47-66):
  //   Row 1: f_x + mu*f_z >= 0  → col 0 = 1.0, col 2 = mu
  //   Row 2: -f_x + mu*f_z >= 0 → col 0 = -1.0, col 2 = mu
  ASSERT_GE(Uf.rows(), 5) << "PointContact Uf should have ≥5 rows";
  EXPECT_NEAR(Uf(1, 2), mu_val, 1e-9)
      << "mu_ not encoded in friction cone row 1 (f_x + mu*f_z >= 0)";
  EXPECT_NEAR(Uf(2, 2), mu_val, 1e-9)
      << "mu_ not encoded in friction cone row 2 (-f_x + mu*f_z >= 0)";
  EXPECT_NEAR(Uf(3, 2), mu_val, 1e-9)
      << "mu_ not encoded in friction cone row 3 (f_y + mu*f_z >= 0)";

  std::cout << "[DataFlow] PointContact mu=" << contact->Mu()
            << " Uf(1,2)=" << Uf(1, 2) << "\n";
}

// ---------------------------------------------------------------------------
// 4. SurfaceContact foot dimensions flow to friction/ZMP cone rows
//    Row 5: tau_x + foot_half_width * f_z >= 0 → Uf(5,5) == foot_half_width
//    Row 7: tau_y + foot_half_length * f_z >= 0 → Uf(7,5) == foot_half_length
// ---------------------------------------------------------------------------
TEST_F(YamlToWbicDataFlowTest, SurfaceContactDimensionsFlowToFrictionCone) {
  const double mu_val  = 0.5;
  const double half_l  = 0.15;   // foot_half_length
  const double half_w  = 0.06;   // foot_half_width

  const std::string yaml =
    "robot_model:\n"
    "  urdf_path: \"" + urdf_path_.string() + "\"\n"
    "  is_floating_base: false\n"
    "  base_frame: base_link\n"
    "controller:\n"
    "  enable_gravity_compensation: false\n"
    "  enable_coriolis_compensation: false\n"
    "  enable_inertia_compensation: false\n"
    "  kp_acc: 1.0\n"
    "  kd_acc: 0.1\n"
    "regularization:\n"
    "  w_qddot: 1.0e-6\n"
    "  w_rf: 1.0e-4\n"
    "  w_tau: 1.0e-3\n"
    "  w_tau_dot: 0.0\n"
    "  w_xc_ddot: 1.0e-3\n"
    "  w_f_dot: 1.0e-3\n"
    "task_pool:\n"
    "  - name: jpos_task\n"
    "    type: JointTask\n"
    "    kp: 1.0\n"
    "    kd: 0.1\n"
    "    kp_ik: 1.0\n"
    "    weight: 1.0\n"
    "  - name: surf_force\n"
    "    type: ForceTask\n"
    "    contact_name: surf_contact\n"
    "    weight: [0, 0, 0, 0, 0, 0]\n"
    "contact_pool:\n"
    "  - name: surf_contact\n"
    "    type: SurfaceContact\n"
    "    target_frame: link2\n"
    "    mu: " + std::to_string(mu_val) + "\n"
    "    foot_half_length: " + std::to_string(half_l) + "\n"
    "    foot_half_width: " + std::to_string(half_w) + "\n"
    "state_machine:\n"
    "  - id: 1\n"
    "    name: df_stay_state\n"
    "    task_hierarchy:\n"
    "      - {name: jpos_task, priority: 0}\n"
    "    contact_constraints:\n"
    "      - {name: surf_contact}\n"
    "    force_tasks:\n"
    "      - {name: surf_force}\n";

  auto arch = MakeArch(yaml);
  ASSERT_NE(arch, nullptr);

  Contact* contact = arch->GetConfig()->constraintRegistry()->GetContact("surf_contact");
  ASSERT_NE(contact, nullptr);

  EXPECT_NEAR(contact->Mu(), mu_val, 1e-9);

  // Force cone build.
  contact->UpdateConeConstraint();
  const Eigen::MatrixXd& Uf = contact->UfMatrix();

  // SurfaceContact Uf is 18×6  (see contact_constraint.cpp:121-186).
  ASSERT_EQ(Uf.rows(), 18) << "SurfaceContact Uf should have 18 rows";
  ASSERT_EQ(Uf.cols(), 6)  << "SurfaceContact Uf should have 6 cols";

  // Row 1: friction cone — col 5 (f_z) == mu
  EXPECT_NEAR(Uf(1, 5), mu_val, 1e-9) << "mu not encoded in friction row";

  // Row 5: tau_x + foot_half_width * f_z >= 0 → Uf(5,5) == foot_half_width
  EXPECT_NEAR(Uf(5, 5), half_w, 1e-9)
      << "foot_half_width not in ZMP row (row 5, col 5)";

  // Row 7: tau_y + foot_half_length * f_z >= 0 → Uf(7,5) == foot_half_length
  EXPECT_NEAR(Uf(7, 5), half_l, 1e-9)
      << "foot_half_length not in ZMP row (row 7, col 5)";

  // Row 9: tipping pyramid — col 5 should be (half_l + half_w) * mu
  const double lw_mu = (half_l + half_w) * mu_val;
  EXPECT_NEAR(Uf(9, 5), lw_mu, 1e-9) << "lw_mu not in tipping pyramid row";

  std::cout << "[DataFlow] SurfaceContact mu=" << contact->Mu()
            << " Uf(5,5)=" << Uf(5, 5) << " (foot_half_width)"
            << " Uf(7,5)=" << Uf(7, 5) << " (foot_half_length)\n";
}

// ---------------------------------------------------------------------------
// 5. Per-state weight override flows through RuntimeConfig → task at runtime
//    The state YAML can override weight per-state. Verify the override reaches
//    the task object after SyncActiveState.
// ---------------------------------------------------------------------------
TEST_F(YamlToWbicDataFlowTest, PerStateWeightOverrideReachesTask) {
  const double pool_weight   = 1.0;   // default in task_pool
  const double state_weight  = 75.0;  // override in state

  const std::string yaml =
    "robot_model:\n"
    "  urdf_path: \"" + urdf_path_.string() + "\"\n"
    "  is_floating_base: false\n"
    "  base_frame: base_link\n"
    "controller:\n"
    "  enable_gravity_compensation: false\n"
    "  enable_coriolis_compensation: false\n"
    "  enable_inertia_compensation: false\n"
    "  kp_acc: 1.0\n"
    "  kd_acc: 0.1\n"
    "regularization:\n"
    "  w_qddot: 1.0e-6\n"
    "  w_rf: 1.0e-4\n"
    "  w_tau: 1.0e-3\n"
    "  w_tau_dot: 0.0\n"
    "  w_xc_ddot: 1.0e-3\n"
    "  w_f_dot: 1.0e-3\n"
    "task_pool:\n"
    "  - name: jpos_task\n"
    "    type: JointTask\n"
    "    kp: 1.0\n"
    "    kd: 0.1\n"
    "    kp_ik: 1.0\n"
    "    weight: " + std::to_string(pool_weight) + "\n"
    "start_state_id: 1\n"
    "state_machine:\n"
    "  - id: 1\n"
    "    name: df_stay_state\n"
    "    task_hierarchy:\n"
    "      - name: jpos_task\n"
    "        priority: 0\n"
    "        weight: " + std::to_string(state_weight) + "\n";

  auto arch = MakeArch(yaml);
  ASSERT_NE(arch, nullptr);

  const Task* task = arch->GetConfig()->taskRegistry()->GetMotionTask("jpos_task");
  ASSERT_NE(task, nullptr);

  // Warm up one tick to trigger SyncActiveState → weight scheduler sets target.
  const Eigen::VectorXd q    = Eigen::VectorXd::Zero(2);
  const Eigen::VectorXd qdot = Eigen::VectorXd::Zero(2);
  RobotJointState js;
  js.q = q; js.qdot = qdot; js.tau = Eigen::VectorXd::Zero(2);
  arch->Update(js, 0.0, 0.001);

  // The state's weight override (75.0) should have been scheduled.
  // Default ramp duration = 0.3s = 300 ticks at 1ms. Run 400 ticks to ensure
  // ramp completes with margin.
  for (int i = 1; i < 400; ++i) {
    js.q = q; js.qdot = qdot; js.tau = Eigen::VectorXd::Zero(2);
    arch->Update(js, i * 0.001, 0.001);
  }

  // After ramp completes, weight should equal the state-level override.
  const double w = task->Weight()[0];
  EXPECT_NEAR(w, state_weight, 0.5)   // 0.5 tolerance for ramp convergence
      << "Per-state weight override did not reach task->Weight()";

  std::cout << "[DataFlow] Per-state weight override: target=" << state_weight
            << " actual=" << w << "\n";
}

}  // namespace
}  // namespace wbc
