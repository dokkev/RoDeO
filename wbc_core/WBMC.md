# WBMC Final Architecture Plan

## Goal

Define the **final-form WBMC** as a **nominal-aware strict-priority inverse-dynamics HQP controller**.

This document ignores legacy migration concerns and focuses only on the architecture that the controller should ultimately converge to.

The guiding principle is:

> Solve the primary operational behavior directly in inverse dynamics under full physical constraints, and use lower-priority bias tasks only to select among redundant feasible solutions.

WBMC v1 is intentionally narrow:

* robot-only inverse dynamics
* nominal-centered minimal correction
* strict feasibility first
* contact force treated primarily as an outcome of feasible support-contact solving

Non-goals for v1:

* explicit object dynamics in the optimization
* dexterous hand-object contact-force optimization
* body-part-specific reaction-force task specializations
* strategy / prediction / policy generation inside WBMC

---

## 1. Core Design Philosophy

### 1.1 Physics first

The highest-priority level must represent hard physical feasibility.

This includes:

* floating-base rigid-body dynamics
* rigid contact acceleration consistency
* friction cone constraints
* actuator torque limits
* optional acceleration or actuation bounds

These constraints define the feasible motion set.

### 1.2 Operational tasks are the controller's true objective

Operational tasks are what the robot **must do now**.

Examples:

* end-effector SE(3) motion tracking
* CoM regulation
* base orientation stabilization
* swing-foot tracking
* manipulation contact motion

These tasks are solved directly in inverse dynamics, not through a preceding inverse-kinematics redundancy stage.

### 1.3 Bias tasks are solution selectors

Bias tasks are **not** primary motion objectives.

They are lower-priority preferences that act only after physical feasibility and operational objectives are satisfied.

Examples:

* preferred joint posture
* selected arm shape
* torso orientation preference
* CoM lateral preference
* planner-provided nominal motion
* predictive redundancy prior

Bias tasks exist only to choose among redundant feasible solutions.

### 1.4 Strict hierarchy is mandatory

Operational and bias objectives must not be mixed in the same weighted layer.

The solver hierarchy must remain semantically clean:

* **Level 0:** Physics
* **Level 1:** Operational
* **Level 2:** Bias
* **Level 3:** Regularization

This ensures:

* physics is never violated for tracking convenience
* operational execution is never degraded by bias
* bias only acts inside the remaining feasible subspace

### 1.5 Redundancy is handled implicitly by lower-priority solution selection

WBMC must not rely on a mandatory standalone IK-based redundancy-resolution stage.

Instead:

* higher-priority levels define the feasible task-consistent motion set
* lower-priority bias levels choose a preferred point inside that set

So redundancy is handled by **hierarchical solution selection**, not by an explicit internal kinematic redundancy solver.

### 1.6 Delta form is the default formulation

The controller should use a nominal-centered delta formulation.

Decision variable:

```math
x = [\Delta \ddot q, \lambda]
```

Actual acceleration:

```math
\ddot q = \ddot q_{nom} + \Delta \ddot q
```

This interpretation is central because it cleanly separates:

* **nominal/bias/predictive preference generation**
* **current-time inverse-dynamics feasibility correction**

---

## 2. Final Solver Hierarchy

## Level 0: Physics

Hard feasibility constraints.

### 2.1 Floating-base dynamics

```math
S_f (M(q)\ddot q + h(q,\dot q) - J_c(q)^T T\lambda) = 0
```

Delta form:

```math
S_f (M(q)\Delta \ddot q - J_c(q)^T T\lambda)
= -S_f(M(q)\ddot q_{nom} + h(q,\dot q))
```

### 2.2 Contact acceleration consistency

```math
J_c(q)\ddot q + \dot J_c(q,\dot q)\dot q = 0
```

Delta form:

```math
J_c(q)\Delta \ddot q = -\dot J_c\dot q - J_c\ddot q_{nom}
```

### 2.3 Friction cone

```math
U_f \lambda \le b
```

### 2.4 Torque limits

```math
\tau_{min} \le \tau \le \tau_{max}
```

with torque recovered from inverse dynamics after solve.

### 2.5 Optional hard constraints

As needed:

* joint acceleration limits
* actuation-specific limits
* contact-mode restrictions

---

## Level 1: Operational

Primary motion objectives.

For a task with Jacobian `J` and desired task acceleration `a_des`:

```math
J\ddot q + \dot J \dot q \approx a_{des}
```

Delta form:

```math
J\Delta \ddot q \approx a_{des} - J\ddot q_{nom}
```

Typical operational tasks:

* hand/end-effector SE(3)
* CoM
* base orientation
* swing foot
* wrist / TCP motion tracking under robot-only feasibility constraints

Operational tasks may later have internal strict ordering, but the default design should treat them as the dedicated operational layer above all bias objectives.

---

## Level 2: Bias

Lower-priority solution selection.

### 2.6 Full-joint acceleration bias

```math
\ddot q \approx \ddot q_{bias}
```

Delta form:

```math
\Delta \ddot q \approx \ddot q_{bias} - \ddot q_{nom}
```

If nominal already encodes the preferred motion, then a very natural bias is:

```math
\Delta \ddot q \approx 0
```

### 2.7 Selected-joint bias

```math
S_b \ddot q \approx \ddot q_{bias}^{(b)}
```

Useful for:

* arm shape preference
* torso bias
* neck/head posture preference
* selected posture-shaping DOFs

### 2.8 Task-space bias

```math
J_b \ddot q + \dot J_b \dot q \approx \ddot x_{bias}
```

Useful for:

* torso orientation tendency
* CoM lateral offset bias
* reduced-space geometric bias supplied from an upstream planner

## Level 3: Regularization

Numerical shaping only.

Typical regularizers:

```math
\|\Delta \ddot q\|^2
```

```math
\|\lambda\|^2
```

Optional:

```math
\|\tau\|^2
```

Purpose:

* numerical stability
* minimum-norm correction
* minimum-norm support-contact force selection
* smoother optimization behavior

Regularization must remain the lowest-priority layer.

---

## 3. Final Architectural Abstractions

## 3.1 WBMC

Top-level orchestrator.

Responsibilities:

* maintain robot, task, and contact registrations
* maintain operational task set
* maintain bias task/reference set
* maintain nominal provider
* build per-cycle context
* build HQP hierarchy through reusable blocks
* solve HQP
* decode solution

WBMC must **not** directly assemble low-level formulation matrices as its main design pattern.
It should orchestrate blocks and context.

---

## 3.2 HQPBuildContext

Shared per-cycle data container read by all blocks.

Responsibilities:

* provide all dynamics quantities
* provide all contact-stacked quantities
* provide nominal acceleration
* provide task-derived terms
* provide bounds and references

Recommended content:

* `M`
* `h`
* `nv`, `na`, `nvFloat`
* `dimContact`
* active contact info list
* stacked `Jc`
* stacked contact RHS / drift
* friction matrices and bounds
* torque lower/upper bounds
* measured external generalized wrench if available
* `qddotNominal`

This object is the controller-block interface.

---

## 3.3 HQPBlock

Base abstraction for all reusable constraint/objective blocks.

Each block should:

* have a fixed semantic role
* know its hierarchy level
* know its weight if soft
* build itself from `HQPBuildContext`

Core methods:

* `build(ctx)`
* `constraint()`
* `level()`
* `weight()`

Blocks must remain controller-agnostic.

---

## 3.4 NominalAccelerationProvider

Abstraction that produces `qddot_nominal`.

This is essential for separating controller core from bias generation logic.

Examples of future providers:

* zero nominal
* simple posture-PD nominal
* planner nominal
* retrieval nominal
* predictive redundancy nominal

Suggested interface:

```cpp
class NominalAccelerationProvider {
 public:
  virtual ~NominalAccelerationProvider() = default;

  virtual bool compute(double time,
                       const RobotState& state,
                       const ContactSnapshot& contacts,
                       const OperationalTaskSnapshot& tasks,
                       Eigen::VectorXd& qddot_nominal) = 0;
};
```

This keeps WBMC representation-agnostic.

---

## 3.5 BiasReference / BiasTask abstraction

Bias must not be forced into the exact same abstraction as operational motion tasks.

Bias can be:

* full joint acceleration preference
* selected joint preference
* reduced task-space preference

So the architecture should include a dedicated bias abstraction.

Suggested categories:

* `JointAccelBias`
* `SelectedJointAccelBias`
* `TaskSpaceBias`

---

## 3.6 HierarchyPolicy

Hierarchy must be explicit and configurable.

Suggested default policy:

```cpp
struct WBMCHierarchyPolicy {
  unsigned int physics_level = 0;
  unsigned int operational_level = 1;
  unsigned int bias_level = 2;
  unsigned int regularization_level = 3;
};
```

Later extensions may support:

* multi-level operational ordering
* multiple bias sublevels
* different force-shaping levels

---

## 4. Final Block Taxonomy

## 4.1 Physics blocks

### Required

* `FloatingBaseDynamicsConstraint`
* `ContactConsistencyConstraint`
* `FrictionConeConstraint`
* `TorqueLimitConstraint`

### Optional later

* `JointAccelerationLimitBlock`
* `ActuationLimitBlock`
* `VelocitySafetyBlock`

---

## 4.2 Operational blocks

### Required

* `MotionTask`

This should support any operational `TaskMotion` that produces Jacobian, drift, and desired acceleration.

---

## 4.3 Bias blocks

### Required

* `JointAccelerationBias`
* `SelectedJointAccelerationBias`
* `TaskSpaceBiasTask`

These are essential for making bias a first-class layer instead of an overloaded posture-task concept.

---

## 4.4 Regularization blocks

### Required

* `AccelerationRegularization`
* `ContactForceRegularization`

### Optional later

* `TorqueRegularization`

---

## 5. Final Per-Cycle Algorithm

### Step 1. Update robot dynamics

Compute:

* mass matrix
* nonlinear effects
* task model state

### Step 2. Build contact snapshot

Collect:

* active contacts
* stacked contact Jacobian
* contact drift / RHS
* friction cone matrices
* desired reaction force references

### Step 3. Update operational tasks

For each operational task:

* update task state
* compute Jacobian
* compute drift term
* compute desired task acceleration

### Step 4. Compute nominal acceleration

If a nominal provider exists:

* compute `qddot_nominal`

Otherwise:

* use zero nominal

### Step 5. Build HQPBuildContext

Populate all shared dynamics, contact, reference, and nominal quantities.

### Step 6. Build physics blocks

Insert all Level 0 blocks.

### Step 7. Build operational blocks

Insert all Level 1 operational motion blocks.

### Step 8. Build bias blocks

Insert all Level 2 bias blocks.

### Step 9. Build regularization blocks

Insert all Level 3 regularization blocks.

### Step 10. Assemble HQPData

Group constraints/objectives by hierarchy level.

### Step 11. Solve strict-priority HQP

Use cascade or equivalent strict-priority HQP solver.

### Step 12. Decode solution

Recover:

* `delta_qddot`
* `qddot`
* `lambda`
* `tau`

Return actuator torque command to the hardware layer.

---

## 6. Public API Direction

The final API should reflect the semantic split between operational tasks and bias references.

### Operational registration

```cpp
bool addOperationalTask(TaskMotion& task,
                        double weight,
                        unsigned int level = 1);
```

### Bias registration

```cpp
bool addBiasReference(const BiasReference& bias);
```

or subclass-based:

```cpp
bool addBiasTask(std::shared_ptr<BiasTaskBase> bias);
```

### Nominal provider registration

```cpp
void setNominalProvider(std::shared_ptr<NominalAccelerationProvider> provider);
```

### Contact registration

```cpp
bool addContact(ContactBase& contact);
bool removeContact(const std::string& name);
```

The API should not blur operational tasks, bias tasks, and nominal generation into one generic method.

---

## 7. Final File Organization

```text
controller/
  wbmc-step-input.hpp
  wbmc.hpp
  wbmc-hierarchy-policy.hpp
  wbmc-registry.hpp
  wbmc-solution.hpp

formulations/hqp/
  hqp-build-context.hpp
  hqp-block-base.hpp

formulations/hqp/blocks/
  floating-base-dynamics-block.hpp
  contact-acceleration-block.hpp
  friction-cone-block.hpp
  torque-limit-block.hpp
  motion-task-block.hpp
  joint-accel-bias-block.hpp
  lambda-reference-block.hpp
  selected-joint-accel-bias-block.hpp
  task-space-bias-block.hpp
  qddot-regularization-block.hpp
  rf-regularization-block.hpp
  torque-regularization-block.hpp

bias/
  bias-reference.hpp
  bias-task-base.hpp
  joint-accel-bias.hpp
  selected-joint-bias.hpp
  task-space-bias.hpp

nominal/
  nominal-acceleration-provider.hpp
  zero-nominal-provider.hpp
  posture-pd-nominal-provider.hpp
  planner-nominal-provider.hpp
  retrieval-nominal-provider.hpp

contacts/
  contact-snapshot.hpp
  contact-stacking.hpp

adapters/
  command-adapter.hpp
```

---

## 8. Design Rules

These rules should remain invariant.

1. **Physics is always the top level.**
2. **Operational tasks are always above bias tasks.**
3. **Bias is never mixed into the operational layer.**
4. **Regularization is always last.**
5. **Nominal generation is external to the core ID-HQP formulation.**
6. **Bias is a first-class abstraction, not a disguised posture task.**
7. **The controller orchestrates; blocks formulate.**
8. **Delta formulation is the default controller interpretation.**

---

## 9. Development Priorities

## Phase 1. Core semantics

* [ ] Fix the final mathematical formulation around `x = [delta_qddot, lambda]`
* [ ] Fix the default hierarchy: Physics / Operational / Bias / Regularization
* [ ] Fix the public semantic split between operational, bias, and nominal
* [ ] Define the final `WBMCHierarchyPolicy`
* [ ] Define the final `WBMCSolution` contents

## Phase 2. Core abstractions

* [ ] Finalize `HQPBuildContext`
* [ ] Finalize `HQPBlock` base interface
* [ ] Finalize `NominalAccelerationProvider` interface
* [ ] Finalize `BiasReference` or `BiasTaskBase` abstraction
* [ ] Finalize controller-owned collections for contacts, op tasks, bias tasks, and blocks

## Phase 3. Physics layer

* [ ] Implement `FloatingBaseDynamicsConstraint`
* [ ] Implement hard `ContactConsistencyConstraint`
* [ ] Implement `FrictionConeConstraint`
* [ ] Implement `TorqueLimitConstraint`
* [ ] Validate Level 0 assembly and dimensions under multiple contacts

## Phase 4. Operational layer

* [ ] Implement generic `MotionTask`
* [ ] Support SE(3) operational tasks
* [ ] Support CoM operational tasks
* [ ] Support optional multi-operational-task stacking at the same level
* [ ] Define clear desired-acceleration conventions for all motion tasks

## Phase 5. Bias layer

* [ ] Implement `JointAccelerationBias`
* [ ] Implement `SelectedJointAccelerationBias`
* [ ] Implement `TaskSpaceBiasTask`
* [ ] Define exact semantics for bias weights and masks/selectors
* [ ] Ensure bias blocks never interfere with higher-priority operational execution

## Phase 6. Regularization layer

* [ ] Implement `AccelerationRegularization`
* [ ] Implement `ContactForceRegularization`
* [ ] Add optional `TorqueRegularization`
* [ ] Define default regularization weights

## Phase 7. Nominal pipeline

* [ ] Implement zero nominal provider
* [ ] Implement posture-PD nominal provider
* [ ] Define the task/contact snapshots visible to nominal providers
* [ ] Ensure nominal provider failure falls back cleanly to zero nominal
* [ ] Validate delta-form consistency across all blocks

## Phase 8. Controller orchestration

* [ ] Build per-cycle state update flow
* [ ] Build per-cycle contact stacking flow
* [ ] Build per-cycle task update flow
* [ ] Build per-cycle context assembly flow
* [ ] Build per-cycle block assembly flow
* [ ] Build solver invocation and solution decode flow

## Phase 9. Validation

* [ ] Test no-contact free-space operational control
* [ ] Test single-contact rigid support
* [ ] Test multi-contact support
* [ ] Test bias influence in redundant directions only
* [ ] Test conflict cases where bias should be ignored due to higher priorities
* [ ] Test torque-limit activation
* [ ] Test friction-limit activation
* [ ] Test numerical behavior under changing contact dimension

## Phase 10. Extensions

* [ ] Add planner nominal provider
* [ ] Add retrieval nominal provider
* [ ] Add predictive redundancy nominal provider
* [ ] Add operational sub-priority levels
* [ ] Add additional safety blocks as needed

---

## 10. Immediate Action Checklist

This is the shortest path toward the final-form architecture.

### Formulation

* [ ] Freeze the delta formulation as the default WBMC interpretation
* [ ] Freeze the four-level strict-priority hierarchy
* [ ] Freeze the rule that bias never shares the operational layer

### Core interfaces

* [ ] Write the final `wbmc.hpp` public API
* [ ] Write the final `NominalAccelerationProvider` interface
* [ ] Write the final `BiasReference` or `BiasTaskBase` interface
* [ ] Write the final `HQPBuildContext` layout

### Required blocks

* [ ] `FloatingBaseDynamicsConstraint`
* [ ] `ContactConsistencyConstraint`
* [ ] `FrictionConeConstraint`
* [ ] `TorqueLimitConstraint`
* [ ] `MotionTask`
* [ ] `JointAccelerationBias`
* [ ] `SelectedJointAccelerationBias`
* [ ] `TaskSpaceBiasTask`
* [ ] `AccelerationRegularization`
* [ ] `ContactForceRegularization`

### Required controller behavior

* [ ] Per-cycle dynamics update
* [ ] Per-cycle contact stacking
* [ ] Per-cycle nominal generation
* [ ] Per-cycle block build
* [ ] Strict-priority HQP solve
* [ ] Decode `delta_qddot`, `qddot`, `lambda`, `tau`

### First validation target

* [ ] One operational hand task + one full-joint bias + rigid contact + torque bounds
* [ ] Verify operational task is preserved when bias conflicts
* [ ] Verify bias acts when redundancy exists
* [ ] Verify regularization only shapes the final remaining nullspace

---

## 11. Final Summary

The final WBMC should be built as a **block-based strict-priority inverse-dynamics HQP controller** centered on the following separation:

* **Physics:** what is feasible
* **Operational:** what must be done now
* **Bias:** how to choose among redundant feasible solutions
* **Regularization:** how to stabilize the remaining numerical freedom

Nominal generation is external, bias is first-class, and delta-form inverse dynamics is the default interpretation.

This is the final architectural target.
