# WBC Configuration Guideline

This document traces how YAML parameters flow through `wbc_core` from a YAML file
on disk to the real-time control loop inside `ControlArchitecture::Step()`.

---

## 1. System Overview

```
YAML File
   │
   ▼
ConfigCompiler::Compile()        ← Phase 1: parse, allocate, validate
   ├─ RuntimeConfig (ownership transferred to ControlArchitecture)
   │    └─ states_: StateId → StateConfig  (WBC formulation config)
   └─ state_recipes_: StateId → StateRecipe  (held internally until Phase 2)

ConfigCompiler::InitializeFsm()  ← Phase 2: instantiate states, register
   ├─ StateFactory::Create()          → StateMachine instance
   ├─ StateMachine::AssignFromRecipe() → task/contact handles copied in
   ├─ StateMachine::SetParameters()   → timing/behavior params parsed in
   ├─ FSMHandler::RegisterState()     → StateMachine owned by FSMHandler
   ├─ FSMHandler::FinalizeStates()
   └─ FSMHandler::SetStartState()
   state_recipes_.clear()  ← StateRecipe objects destroyed here
   compiler_.reset()       ← ConfigCompiler destroyed here

────────────────────────────────────────────────── startup complete

ControlArchitecture::Step()       ← Phase 3: real-time loop
   ├─ FSMHandler::Update()              (drives StateMachine lifecycle)
   ├─ RuntimeConfig::ApplyStateOverrides()  (write gains to task objects)
   ├─ RuntimeConfig::BuildFormulation()     (fill WbcFormulation pointers)
   ├─ ApplyDesired()                        (apply external task references)
   └─ WBIC::Solve()
```

### Two parallel runtime stores

At runtime, per-state information lives in **two separate places** with
different responsibilities:

| Store | Owner | What it holds |
|-------|-------|---------------|
| `RuntimeConfig::states_` | `ControlArchitecture` | `StateConfig`: which tasks/contacts/gains to feed the WBC solver |
| `FSMHandler::state_map_` | `ControlArchitecture` | `StateMachine`: lifecycle callbacks (`FirstVisit`/`OneStep`/`LastVisit`), timing params, assigned handles |

`StateRecipe` is a startup-only bridge that populates both. It does not
survive beyond `InitializeFsm`. No YAML nodes are held after startup.

---

## 2. YAML File Structure

```yaml
# ── Global settings ────────────────────────────────────────────
start_state_id: 1          # optional, overrides solver_params.start_state_id

solver_params:
  start_state_id: 1        # fallback (used when top-level key absent)
  w_qddot:  1e-8
  w_rf:     1e-7
  ...

robot_model:
  base_frame: base_link    # optional hints for ControlArchitecture
  ee_frame:   ee_link

# ── Resource pools ─────────────────────────────────────────────
contact_pool:
  - name: lf_contact
    type: SurfaceContact   # or PointContact
    link_name: l_foot
    mu: 0.5
    foot_half_length: 0.11
    foot_half_width:  0.06

global_constraints:        # active in every state (e.g. joint limits)
  - name: jpos_limit
    type: JointPosLimitConstraint
    dt: 0.001

task_pool:
  - name: jpos_task
    type: JointTask
    kp: 100.0              # scalar → broadcast to all joints
    kd: [10.0, 10.0, ...]  # vector → element-wise
    weight: 1.0

  - name: ee_pos
    type: LinkPosTask
    link_name: ee_link
    kp: 200.0
    kd: 20.0

  - name: ee_force
    type: ForceTask
    contact_name: ee_contact  # must match a contact_pool entry

# ── FSM states ─────────────────────────────────────────────────
state_machine:
  - id: 0
    name: initialize
    # type defaults to name when omitted
    params:
      duration: 2.0
      wait_time: 0.5
      next_state_id: 1
      target_jpos: [0.0, 0.0, ...]   # consumed by derived class
    task_hierarchy:
      - name: jpos_task

  - id: 1
    name: ee_task
    type: ee_control       # WBC_REGISTER_STATE key
    params:
      stay_here: true
    task_hierarchy:
      - name: jpos_task
        priority: 1        # lower = higher priority in solver
        kp: 50.0           # per-state gain override
      - name: ee_pos
        priority: 0
      - name: ee_ori
        priority: 0
    force_tasks:
      - name: ee_force
    contact_constraints:
      - name: ee_contact
```

---

## 3. Startup Parameter Flow (Phase 1 + 2)

### 3.1 contact_pool → ConstraintRegistry

`ConfigCompiler::ParseConstraintPool` creates `PointContact` or `SurfaceContact`
objects and stores them in `ConstraintRegistry` (owned by `RuntimeConfig`).

```
YAML contact_pool entry
  │  name, type, link_name, mu, ...
  ▼
SurfaceContact / PointContact   (constructor: robot, link_idx, mu, ...)
  │  ownership
  ▼
ConstraintRegistry::AddConstraint(name, unique_ptr<Constraint>)
  │  non-owning pointer also stored in
  ▼
StateRecipe::contact_by_name[name]  (lookup by state during Phase 2)
StateConfig::contacts               (ordered list for formulation build)
```

### 3.2 global_constraints → ConstraintRegistry + global list

`ParseGlobalConstraints` creates limit-constraint objects (e.g.
`JointPosLimitConstraint`) and appends raw pointers to
`RuntimeConfig::global_constraints_`. These are injected into every
`WbcFormulation` by `BuildFormulation` regardless of active state.

### 3.3 task_pool → TaskRegistry

`ParseTaskPool` creates task objects, applies YAML gains, stores them in
`TaskRegistry`, and captures **default** gains in
`RuntimeConfig::default_motion_task_cfg_` / `default_force_task_cfg_`.

```
YAML task_pool entry
  │  name, type, kp, kd, ki, weight, kp_ik
  ▼
JointTask / LinkPosTask / LinkOriTask / ComTask / ForceTask
  │  SetKp / SetKd / SetWeight applied immediately
  │  ownership
  ▼
TaskRegistry::AddMotionTask(name, unique_ptr<Task>)
  │
  ├─ default TaskConfig snapshot → default_motion_task_cfg_[task*]
  └─ non-owning pointer available to states via GetMotionTask(name)
```

**Default gains** are the fallback restored when transitioning away from a
state that had per-state overrides (see §4.3).

### 3.4 state_machine → StateConfig + StateRecipe

`ParseStateMachine` → `ParseState` produces one `StateConfig` and one
`StateRecipe` per state entry.

| Destination | Content | Lifetime |
|-------------|---------|----------|
| `StateConfig` | runtime-ready task/contact pointer lists, override configs | permanent (in `RuntimeConfig::states_`) |
| `StateRecipe` | same pointers keyed by name + raw `YAML::Node params` | startup only; freed after `InitializeFsm` |

**Motion task ordering** inside `StateConfig::motion` follows two sort keys:
1. `priority:` field (lower = higher priority); default `99`
2. YAML declaration order for ties

**Per-state gain overrides** are parsed once into `StateConfig::owned_task_cfg`
(heap-allocated at startup). At runtime, `ApplyStateOverrides` writes them to
the task objects with no YAML parsing.

### 3.5 FSM state instantiation (Phase 2)

`ConfigCompiler::InitializeFsm` loops over all stored `StateRecipe` objects:

```
StateRecipe
  │  recipe.type  (WBC_REGISTER_STATE key)
  ▼
StateFactory::Create(type, id, name, StateMachineConfig)
  │  factory lambda calls concrete constructor
  ▼
StateMachine instance  (e.g. EeTaskState)

state->AssignFromRecipe(recipe)
  │  populates assigned_tasks_, assigned_force_tasks_,
  │  assigned_contacts_, assigned_constraints_
  ▼
state->SetParameters(recipe.params)
  │  base class:  duration, wait_time, next_state_id, stay_here
  │  derived class: target_jpos, hold_threshold, ...
  ▼
fsm_handler.RegisterState(id, std::move(state))

── after all states registered ──

fsm_handler.FinalizeStates()    // builds sorted state catalog once
fsm_handler.SetStartState(start_id)
state_recipes_.clear()          // all recipe memory freed here
```

**Important:** `SetParameters` is called by `InitializeFsm` after construction.
Factory lambdas must **not** call `SetParameters` themselves
(see `StateMachineConfig::params` doc).

---

## 4. Runtime: How FSMHandler and RuntimeConfig Work Together

### 4.1 Decoupling principle

`FSMHandler` and `RuntimeConfig` have **no direct coupling** to each other.
They share a common key type (`StateId`) and are both indexed with the same IDs
set during `ConfigCompiler::InitializeFsm`. `ControlArchitecture` is the sole
mediator.

```
                    StateId (shared key)
                         │
          ┌──────────────┼──────────────┐
          ▼                             ▼
FSMHandler::state_map_        RuntimeConfig::states_
  [0] → StateMachine             [0] → StateConfig
  [1] → StateMachine   ←key─    [1] → StateConfig
  [2] → StateMachine             [2] → StateConfig

  Knows: when to transition,      Knows: which tasks/contacts/gains
         what behavior to run             to feed the WBC solver

  Does NOT touch: RuntimeConfig   Does NOT touch: FSMHandler
```

`ControlArchitecture` queries the active `StateId` from `FSMHandler` each tick,
then uses that ID to look up the matching `StateConfig` in `RuntimeConfig`.

### 4.2 Full Step() tick sequence

```
ControlArchitecture::Step()
  │
  ├─ [every tick] update StateProvider meta (servo_dt, current_time, nominal_jpos)
  │
  ├─ [every tick] consume external transition request
  │    fsm_handler_->ConsumeRequestedState()  →  ForceTransition() if pending
  │
  ├─ [every tick] FSM update
  │    fsm_handler_->Update(t)
  │      ├─ (first visit of current state) EnterState(t) + FirstVisit()
  │      ├─ UpdateStateTime(t) + OneStep()       ← state behavior runs here
  │      └─ if EndOfState():
  │           LastVisit()  (current state cleanup)
  │           GetNextState() → next_id
  │           new state: EnterState(t) + FirstVisit()   ← same tick!
  │
  ├─ [every tick] read active state ID from FSMHandler
  │    state_id = fsm_handler_->GetCurrentStateId()
  │
  ├─ [on state change only]  ← ControlArchitecture detects via applied_state_id_
  │    cached_state_ = runtime_config_->FindState(state_id)
  │    runtime_config_->ApplyStateOverrides(*cached_state_)   ← write gains to tasks
  │    runtime_config_->BuildFormulation(*cached_state_, formulation_)  ← fill WbcFormulation
  │
  ├─ [every tick] apply external task references
  │    ApplyDesired(*cached_state_)
  │      seqlock copy of task_reference_snapshot_
  │      → task->UpdateDesired(pos, vel, acc)   for ee_pos, ee_ori, com, joint
  │      → force_task->UpdateDesired(wrench)    for ee_force
  │
  ├─ [every tick] update kinematics
  │    for each task in formulation: UpdateJacobian, UpdateJacobianDotQdot, UpdateOpCommand
  │    for each contact:             UpdateJacobian, UpdateConstraint, UpdateOpCommand
  │    for each constraint:          UpdateJacobian, UpdateConstraint
  │
  ├─ [every tick] solver
  │    solver_->FindConfiguration(formulation_, ...)  → cmd_.q, cmd_.qdot, qddot_cmd
  │    solver_->MakeTorque(formulation_, qddot_cmd)   → cmd_.tau
  │
  └─ [every tick] physics compensations + joint PID feedback → final cmd_.tau
```

**Key optimizations:**
- `BuildFormulation` and `ApplyStateOverrides` run **only on state change**, not every tick.
  `cached_state_` (a `const StateConfig*`) and `applied_state_id_` implement this guard.
- `WbcFormulation` vectors are pre-reserved at startup (`ReserveFormulationCapacity`):
  `BuildFormulation` refills them with no heap allocation on steady-state ticks.
- Task Jacobians are recomputed every tick from the current robot model state.

### 4.3 State change detection

```cpp
// In Step() — how ControlArchitecture bridges FSMHandler and RuntimeConfig:
fsm_handler_->Update(current_time_);
const StateId state_id = fsm_handler_->GetCurrentStateId();  // query FSMHandler

const bool state_changed = (state_id != applied_state_id_);  // CA-local guard
if (state_changed) {
  cached_state_ = runtime_config_->FindState(state_id);      // query RuntimeConfig
  runtime_config_->ApplyStateOverrides(*cached_state_, ...);
  runtime_config_->BuildFormulation(*cached_state_, formulation_);
  applied_state_id_ = state_id;
}
// Every tick after state entry: use cached_state_ directly (pointer into RuntimeConfig::states_)
ApplyDesired(*cached_state_);
```

`FindState` returns a pointer stable for the lifetime of `RuntimeConfig`
(the `unordered_map` is never modified after startup).

### 4.4 FSM state lifecycle details

`FSMHandler::Update` drives the `StateMachine` lifecycle:

```
FirstVisit tick:   EnterState(t)      — records start_time_, updates StateProvider
                   FirstVisit()        — derived class: cache task pointers, set goals

Normal ticks:      UpdateStateTime(t) — updates current_time_ = t - start_time_
                   OneStep()          — derived class: read sensor/estimate, update tasks

Last tick:         EndOfState() → true   when current_time_ >= duration_ + wait_time_
                   LastVisit()             — derived class: cleanup
                   GetNextState() → next_state_id_ (set by SetParameters)
                   → immediately runs new state's EnterState + FirstVisit in same tick
```

The same-tick transition means `ControlArchitecture::Step` sees the new `state_id`
immediately after `Update(t)` returns, so `ApplyStateOverrides` and
`BuildFormulation` run for the new state within the same control cycle.

### 4.5 Per-state gain overrides

`ApplyStateOverrides` (called once per state entry, not every tick):

```
for each task in state.motion:
  if state.motion_cfg[i] != nullptr:   ← per-state YAML override exists
    task->SetKp(override.kp)
    task->SetKd(override.kd)  ...
  else:                                ← no override: restore pool defaults
    task->SetKp(default_motion_task_cfg_[task].kp)  ...
```

This mutates task objects shared across states; defaults are always restored
on entry to any state that does not override a given task.

### 4.6 Task references (external inputs)

```
External thread                      RT thread (Step)
──────────────                       ─────────────────
SetTaskReference(ref)
  mutex lock
  task_reference_sequence_ += 1  (odd = writing)
  task_reference_ = ref
  task_reference_snapshot_ = ref
  task_reference_sequence_ += 1  (even = done)
                                     ApplyDesired(*cached_state_)
                                       seqlock read: retry if sequence odd or changed
                                       task_reference_snapshot_ = task_reference_
                                       ApplyEndEffectorReference → ee_pos/ee_ori->UpdateDesired
                                       ApplyComReference         → com->UpdateDesired
                                       ApplyJointReference       → joint->UpdateDesired
                                       ApplyForceReference       → ee_force->UpdateDesired
```

Task references are **not** YAML-sourced. They come from an external controller
or demo state machine and are applied to the task objects each tick via
`UpdateDesired`. The `cached_state_->ee_pos`, `ee_ori`, `com`, `joint`,
`ee_force` cached handles (set during `ParseState`) make this O(1) lookup.

---

## 5. Memory Ownership Summary

| Object | Owner | Lifetime |
|--------|-------|----------|
| `PinocchioRobotSystem` | `ControlArchitecture::robot_` | permanent |
| `RuntimeConfig` | `ControlArchitecture::runtime_config_` | permanent |
| `ConfigCompiler` | `ControlArchitecture::compiler_` | startup only (freed after `Initialize()`) |
| `StateRecipe` | `ConfigCompiler::state_recipes_` | startup only (freed inside `InitializeFsm`) |
| `Task` / `ForceTask` | `TaskRegistry` (inside `RuntimeConfig`) | permanent |
| `Constraint` / `Contact` | `ConstraintRegistry` (inside `RuntimeConfig`) | permanent |
| `StateConfig` | `RuntimeConfig::states_` | permanent |
| `TaskConfig` override | `StateConfig::owned_task_cfg` | permanent |
| `StateMachine` instances | `FSMHandler::state_map_` | permanent |
| `TaskReference` snapshot | `ControlArchitecture` | per-tick (seqlock copy) |

---

## 6. Parameter Type Reference

| Parameter | YAML location | Parsed by | Stored in | Used at |
|-----------|---------------|-----------|-----------|---------|
| `kp`, `kd`, `ki`, `weight`, `kp_ik` | `task_pool[*]` | `ParseTaskPool` | `default_motion_task_cfg_` | on state entry (restore defaults) |
| per-state gain override | `state_machine[*].task_hierarchy[*]` | `ParseState` | `StateConfig::owned_task_cfg` | on state entry via `ApplyStateOverrides` |
| `contact_name`, `link_name`, `mu` | `contact_pool[*]` | `ParseConstraintPool` | `ConstraintRegistry` | BuildFormulation every tick |
| `duration`, `wait_time`, `next_state_id`, `stay_here` | `state_machine[*].params` | `StateMachine::SetParameters` | `StateMachine` base class fields | `EndOfState()` / `GetNextState()` every tick |
| state-specific params (`target_jpos`, etc.) | `state_machine[*].params` | derived `SetParameters` | derived class fields | `FirstVisit()` / `OneStep()` |
| `start_state_id` | top-level or `solver_params.*` | `ParseTopLevelStartStateId` / `ParseSolverParams` | `RuntimeConfig::configured_start_state_id_` | `InitializeFsm` once |
| solver weights (`w_qddot`, ...) | `solver_params` | `ParseSolverParams` | `RuntimeConfig::solver_params_` | `InitializeSolver` once |
| `base_frame`, `ee_frame` | `robot_model` | `ParseRobotModelHints` | `RuntimeConfig::robot_model_hints_` | `Initialize` once |

---

## 7. `start_state_id` Resolution Priority

1. Top-level `start_state_id:` key in the YAML file
2. `solver_params.start_state_id:` (used only when top-level key is absent)
3. First state listed in `state_machine:` sequence

---

## 8. `WBC_REGISTER_STATE` Macro

State types are registered at static initialization time (before `main`):

```cpp
// In my_state.cpp
WBC_REGISTER_STATE(
  "my_state",
  [](StateId id, const std::string& name, const StateMachineConfig& ctx) {
    return std::make_unique<MyState>(id, name, ctx);
  }
);
```

The macro uses an anonymous namespace (internal linkage). This only works with
**shared libraries** (`.so`/`.dylib`). Static archive linkage silently
suppresses the symbol; the state will not be registered.

Registration failures (empty key, null creator, or duplicate key) call
`std::abort()` — all such cases are programming errors detectable at startup.

The `ctx.params` field carries the raw YAML node. Factory lambdas may inspect
it for constructor-time decisions but must **not** call `SetParameters` —
`ConfigCompiler::InitializeFsm` calls it once after construction.

---

## 9. Adding a New State Type — Checklist

1. Create `my_state.hpp` / `my_state.cpp`, derive from `StateMachine`.
2. Override `FirstVisit()`, `OneStep()`, `LastVisit()`.
3. Override `SetParameters(const YAML::Node&)` to parse state-specific params
   (call `StateMachine::SetParameters(node)` first for common fields).
4. Add `WBC_REGISTER_STATE("my_state_key", lambda)` in the `.cpp` file.
5. Ensure the library is built as a shared library and linked.
6. In your YAML:
   ```yaml
   state_machine:
     - id: 5
       name: my_instance
       type: my_state_key       # must match WBC_REGISTER_STATE first arg
       params:
         duration: 3.0
         my_param: 42.0
       task_hierarchy:
         - name: jpos_task
   ```
