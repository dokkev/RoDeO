# Control Architecture Code Structure

This document defines the intended ownership boundaries between config loading,
config compilation, runtime assembly, FSM wiring, and IDProblem registry usage.

The main rule is:

```text
YAML files
  -> ConfigLoader
  -> ConfigCompiler
  -> ConfigValidator
  -> RuntimeAssembler::Assemble
  -> ControlArchitecture::Initialize
  -> BindRegistry
  -> StateMachineAssembler
  -> IDProblemRegistry + FSMHandler
  -> IDHQP
```

`control_architecture` owns controller-level composition. `wbc_core` owns the
math/runtime primitives used to formulate and solve inverse dynamics problems.

## Layer Map

### ConfigLoader

Files:

- `include/control_architecture/runtime/config_loader.hpp`
- `src/runtime/config_loader.cpp`

Role:

- Loads the root YAML file.
- Resolves external YAML includes such as `task_pool_yaml` and
  `state_machine_yaml`.
- Returns one merged `YAML::Node`.

Rules:

- Does not create tasks, contacts, solvers, states, or registries.
- Does not need `RobotSystem`.
- Does not validate robot-specific dimensions or frame names.
- Should only handle file/path composition.

### YamlParser

Files:

- `include/control_architecture/runtime/yaml_parser.hpp`
- `src/runtime/yaml_parser.cpp`

Role:

- Small YAML utility helpers, such as `RequireField` and `RejectField`.

Rules:

- No domain logic.
- No robot, task, solver, or FSM ownership.
- Keep this as a thin error-message helper.

### ConfigCompiler

Files:

- `include/control_architecture/runtime/config_compiler.hpp`
- `src/runtime/config_compiler.cpp`
- `include/control_architecture/runtime/compiled_config.hpp`

Role:

- Converts a merged `YAML::Node` into `CompiledConfig`.
- Parses string tags into typed enums:
  - task types
  - contact types
  - constraint types
  - solver backend enum
- Rejects removed/legacy YAML fields early.
- Preserves state selections:
  - active task names
  - per-state weight overrides
  - per-state level overrides
  - active contact names

Rules:

- Does not create TSID/WBC task objects.
- Does not create contacts.
- Does not touch `RobotSystem`.
- Does not touch `IDProblemRegistry`.
- Does not initialize FSM states.
- State `name` is the `StateFactory` key. A separate YAML `type` or
  `implementation` field is not allowed.

Output:

```text
CompiledConfig
  task/contact/constraint/state/solver specs only
  no runtime object ownership
```

### ConfigValidator

Files:

- `include/control_architecture/runtime/config_validator.hpp`
- `src/runtime/config_validator.cpp`

Role:

- Validates cross references in `CompiledConfig`.
- Catches duplicate names and duplicate state IDs.
- Catches state references to unknown task/contact names.

Rules:

- Pure config validation only.
- Does not create runtime objects.
- Does not touch `RobotSystem`.
- Does not touch `IDProblemRegistry`.

### RuntimeAssembler

Files:

- `include/control_architecture/runtime/runtime_assembler.hpp`
- `src/runtime/runtime_assembler.cpp`
- `include/control_architecture/runtime/runtime_config.hpp`

Role:

- Converts validated `CompiledConfig` into `RuntimeConfig`.
- Creates concrete runtime task/contact objects from typed specs.
- Stores task/contact ownership in `RuntimeConfig`.

Current split:

```text
RuntimeAssembler::Assemble(compiled, robot)
  -> creates RuntimeConfig
  -> owns task/contact objects
  -> no registry or FSM side effects

BindRegistry(config, registry, robot, data)
  -> registers RuntimeConfig tasks, contacts, constraints, regularization,
     and qddot_ref policy into IDProblemRegistry
```

Rules:

- May depend on `RobotSystem`, because task/contact constructors need robot
  model information.
- May create task/contact runtime objects.
- `RuntimeAssembler::Assemble` does not wire references into
  `IDProblemRegistry`.
- `BindRegistry` may wire existing runtime objects into `IDProblemRegistry`,
  but does not create new task/contact objects.
- Does not instantiate state objects through `StateFactory`.
- Should not run control-loop logic.
- Should not solve QPs.
- Should not parse YAML directly.

Important ownership rule:

```text
RuntimeConfig owns task/contact objects.
IDProblemRegistry stores non-owning references to those objects.
State stores task handles assigned from RuntimeConfig.
```

### StateMachineAssembler

Files:

- `include/control_architecture/runtime/state_machine_assembler.hpp`
- `src/runtime/state_machine_assembler.cpp`

Role:

- Wires `RuntimeConfig` into `FSMHandler`.
- Creates concrete `State` instances through `StateFactory`.
- Assigns configured task handles to each state.
- Sets state parameters and start state.

Rules:

- Does not parse YAML.
- Does not create task/contact runtime objects.
- Does not touch `IDProblemRegistry`.
- Does not solve QPs.
- Uses YAML state `name` as the factory key.

The runtime wiring is intentionally split:

```text
BindRegistry
  -> RuntimeConfig + IDProblemRegistry
  -> registers tasks, contacts, constraints, regularization, qddot_ref policy

StateMachineAssembler
  -> RuntimeConfig + FSMHandler + StateFactory
  -> creates states and assigns configured task handles
```

### RuntimeConfig

File:

- `include/control_architecture/runtime/runtime_config.hpp`

Role:

- Runtime object graph owned by `control_architecture`.
- Holds:
  - `task_pool`
  - `contact_pool`
  - constraints
  - per-state activation maps
  - solver backend config
  - ID regularization config
  - qddot reference enable flag

Rules:

- Owns tasks and contacts.
- Does not solve.
- Does not compute per-tick snapshots.
- Does not parse YAML.

### StateFactory and State

Files:

- `include/control_architecture/state_machine/state_factory.hpp`
- `src/state_machine/state_factory.cpp`
- `include/control_architecture/state_machine/state_machine.hpp`
- `include/control_architecture/state_machine/robot_control_profile.hpp`

Role:

- `StateFactory` maps state `name` strings to constructors.
- Concrete state classes are expected to live in robot-specific controller
  packages or behavior/interface packages.
- `RobotControlProfile` is a ROS-free virtual profile for robot-specific
  control hooks. ROS wrappers can derive from the generic controller and return
  a profile directly, without pluginlib for the FSM itself.
- `State` receives a lightweight context:
  - `RobotSystem*`
  - `pinocchio::Data*`

Rules:

- `control_architecture` should not include concrete robot-specific states.
- `control_architecture` does not register default concrete states.
- YAML state `name` is the factory key.
- Custom states are added before `Initialize()` through
  `StateFactory::Register`, `ControlArchitecture::RegisterState`, or a
  `RobotControlProfile`.
- States can update task references or desired trajectories, but should not
  solve IDHQP directly.

### ControlArchitecture

Files:

- `include/control_architecture/control_architecture.hpp`
- `src/control_architecture.cpp`

Role:

- Runtime orchestrator.
- Owns:
  - `RuntimeConfig`
  - `RobotSystem`
  - `IDProblemRegistry`
  - `IDHQP`
  - `FSMHandler`
  - `StateFactory`
  - `RobotCommand`
  - `RobotLogger`

Initialize flow:

```text
ControlArchitecture::Initialize()
  -> creates IDProblemRegistry and IDHQP
  -> bootstraps robot model terms once
  -> BindRegistry(...)
  -> StateMachineAssembler::Assemble(...)
  -> initializes output command buffer
```

State constructors must be registered before `Initialize()`. The generic ROS
controller does this by calling a robot-specific `RobotControlProfile` supplied
by a thin derived controller class. Non-ROS applications can use the same
profile or call the registration function directly before `Initialize()`.

Per-tick flow:

```text
ControlArchitecture::Update(state, dt)
  -> RobotSystem::updateState(state)
  -> Step(dt)

Step(dt)
  -> UpdateModelTerms()
  -> UpdateStateMachine(current_time)
  -> BuildProblem(current_time)
  -> IDHQP::solve(problem, dt)
  -> fill RobotCommand and RobotLogger from IDSolution
```

Rules:

- Does not parse YAML.
- Does not create task/contact objects directly.
- Does not contain robot-specific state classes.
- Should remain an orchestration class.
- Uses `BindRegistry` and `StateMachineAssembler` instead of a single
  catch-all runtime wiring call.

### IDProblemRegistry

File:

- `wbc_core/include/wbc_core/controller/id-problem-registry.hpp`

Role:

- Lower-level runtime registry in `wbc_core`.
- Stores references to already-created task/contact primitives.
- Converts active task/contact selections into an `IDProblem` snapshot.
- Resolves qddot reference policy.
- Applies torque bounds and external generalized wrench pointers.

Rules:

- Does not parse YAML.
- Does not know `RuntimeConfig`.
- Does not know `State`.
- Does not own task/contact objects.
- Does not own controller architecture flow.
- Does not solve QP.

Per-tick role:

```text
IDProblemRegistry::buildProblem(...)
  -> snapshots active motion tasks
  -> snapshots active contact constraints
  -> inserts regularization, torque bounds, qddot_ref
  -> returns IDProblem
```

`ControlArchitecture` passes the already-updated `solver_->data()` into the
registry to avoid recomputing Pinocchio model terms in the 1 kHz loop.

### IDHQP

Files:

- `wbc_core/include/wbc_core/controller/id-hqp.hpp`
- `wbc_core/src/controller/id-hqp.cpp`

Role:

- Solves a ready `IDProblem`.
- Owns HQP cascade solver backend.
- Decodes qddot, lambda, integrated command state, and separated torque
  components into `IDSolution`.

Rules:

- Does not parse YAML.
- Does not know FSM states.
- Does not own task/contact pools.
- Does not decide which tasks are active.

## Dependency Direction

Allowed direction:

```text
controller package
  -> control_architecture
  -> wbc_core
```

Forbidden direction:

```text
wbc_core -> control_architecture
wbc_core -> controller package
control_architecture -> concrete robot controller states
IDProblemRegistry -> RuntimeConfig
ConfigCompiler -> RobotSystem
ConfigLoader -> RobotSystem
```

## Intended Runtime Ownership

```text
ControlArchitecture
  owns RuntimeConfig
    owns TaskMotion objects
    owns ContactBase objects
    owns per-state selection config

ControlArchitecture
  owns IDProblemRegistry
    stores non-owning task/contact references

ControlArchitecture
  owns FSMHandler
    owns State objects
      stores assigned task handles

ControlArchitecture
  owns IDHQP
    owns solver backend and Pinocchio Data used for solve
```

## Why This Split Matters

The control loop should stay small and deterministic:

```text
RobotState in
  -> RobotSystem update
  -> FSM updates task references
  -> IDProblem snapshot
  -> IDHQP solve
  -> command out
```

The YAML and assembly code should run outside the hot path. Runtime object
creation, YAML parsing, field validation, and factory lookup should not happen
inside the 1 kHz loop.

## Open Cleanup Items

- Audit `RuntimeConfig` for fields that are parsed but unused after the move to
  IDHQP.
- Keep controller-specific states in controller packages, not in
  `control_architecture`.
